"""Vendored SparkVSR inference core and window execution pipeline.

Attribution:
Adapted from taco-group/SparkVSR (Apache-2.0 License)
Upstream: https://github.com/taco-group/SparkVSR
Commit: 8e788bc754e60ca34707248ef72b53faae558a25

Modifications:
- Precomputed empty-prompt text embeddings (T5 encoder removed from inference)
- Scene-cut-aware window scheduling and intra-shot overlap blending
- PyTorch bfloat16 / TF32 / cuDNN optimizations
- Memory cleanup and VAE slicing/tiling
- I/O contract changed to [0, 1] video tensors (normalization to [-1, 1] happens inside
  process_video_ref_i2v rather than in the caller)

get_resize_crop_region_for_grid, prepare_rotary_positional_embeddings and the transformer
invocation inside process_video_ref_i2v are ported verbatim from upstream.
"""

import contextlib
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import CogVideoXDPMScheduler, CogVideoXImageToVideoPipeline
from diffusers.models.embeddings import get_3d_rotary_pos_embed

from ..config import (
    DEFAULT_SR_NOISE_STEP,
    EMPTY_PROMPT_EMBED_PATH,
    FP8_QUANTIZE,
    SPARKVSR_MODEL_DIR,
    TORCH_COMPILE,
    TORCH_COMPILE_MODE,
    TORCH_COMPILE_VAE,
)


def preprocess_video_match(
    video: torch.Tensor,
) -> Tuple[torch.Tensor, int, int, int, int, int, int]:
    """Pad video temporally to 8n+1 frames and spatially to multiples of 4.

    Args:
        video: Tensor of shape [B, C, T, H, W]

    Returns:
        padded_video: Tensor of shape [B, C, T_pad, H_pad, W_pad]
        orig_t, orig_h, orig_w: Original dimensions
        pad_t, pad_h, pad_w: Added padding amounts
    """
    b, c, orig_t, orig_h, orig_w = video.shape

    # Temporal length must be at least 9 and satisfy (T - 1) % 8 == 0 (8n+1)
    if orig_t < 9:
        target_t = 9
    else:
        target_t = ((orig_t - 2) // 8 + 1) * 8 + 1

    pad_t = target_t - orig_t

    # Spatial dimensions are padded to a multiple of 16, not upstream's 4.
    # The VAE downsamples by 8 and the transformer patchifies by 2, so RoPE builds
    # its grid at H//16 while patchify sees H//8. At a multiple of 4 those two
    # disagree whenever H//8 is odd (1080 is), and the mismatch is silent.
    # Over-padding is free here: remove_padding_and_extra_frames crops back.
    pad_h = (16 - (orig_h % 16)) % 16
    pad_w = (16 - (orig_w % 16)) % 16

    padded_video = video
    if pad_t > 0:
        # Replicate last frame temporally
        last_frame = video[:, :, -1:, :, :].repeat(1, 1, pad_t, 1, 1)
        padded_video = torch.cat([padded_video, last_frame], dim=2)

    if pad_h > 0 or pad_w > 0:
        # Replicate padding on a 5D input goes through replication_pad3d, which requires
        # all three trailing dims: (left, right, top, bottom, front, back). The temporal
        # pair stays zero — frames were already extended above.
        padded_video = F.pad(padded_video, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")

    return padded_video, orig_t, orig_h, orig_w, pad_t, pad_h, pad_w


def remove_padding_and_extra_frames(
    video: torch.Tensor,
    orig_t: int,
    orig_h: int,
    orig_w: int,
) -> torch.Tensor:
    """Crop video tensor back to original dimensions [B, C, orig_t, orig_h, orig_w]."""
    return video[:, :, :orig_t, :orig_h, :orig_w]


def make_spatial_tiles(
    height: int,
    width: int,
    tile_size: int = 512,
    overlap: int = 64,
) -> List[Tuple[int, int, int, int]]:
    """Generate bounding boxes for spatial tiles: (y_start, y_end, x_start, x_end)."""
    step = tile_size - overlap
    tiles = []
    y = 0
    while y < height:
        y_end = min(y + tile_size, height)
        y_start = max(0, y_end - tile_size)
        x = 0
        while x < width:
            x_end = min(x + tile_size, width)
            x_start = max(0, x_end - tile_size)
            tiles.append((y_start, y_end, x_start, x_end))
            if x_end == width:
                break
            x += step
        if y_end == height:
            break
        y += step
    return tiles


def get_valid_tile_region(
    tile_idx: int,
    total_tiles: int,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
    height: int,
    width: int,
    overlap: int,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Compute crop slices for pasting tile into destination canvas without seams."""
    crop_top = overlap // 2 if y_start > 0 else 0
    crop_bottom = overlap // 2 if y_end < height else 0
    crop_left = overlap // 2 if x_start > 0 else 0
    crop_right = overlap // 2 if x_end < width else 0

    tile_h = y_end - y_start
    tile_w = x_end - x_start

    src_slice = (
        crop_top,
        tile_h - crop_bottom,
        crop_left,
        tile_w - crop_right,
    )
    dst_slice = (
        y_start + crop_top,
        y_end - crop_bottom,
        x_start + crop_left,
        x_end - crop_right,
    )
    return src_slice, dst_slice


# ==================== ROTARY POSITIONAL EMBEDDINGS ====================
# Ported verbatim from upstream sparkvsr_inference_script.py.


def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    transformer_config: Any,
    vae_scale_factor_spatial: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
    grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

    p = transformer_config.patch_size
    p_t = transformer_config.patch_size_t

    base_size_width = transformer_config.sample_width // p
    base_size_height = transformer_config.sample_height // p

    if p_t is None:
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            device=device,
        )
    else:
        base_num_frames = (num_frames + p_t - 1) // p_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(max(base_size_height, grid_height), max(base_size_width, grid_width)),
            device=device,
        )

    return freqs_cos, freqs_sin


class SparkVSRPipeline:
    """Manages the CogVideoX SparkVSR model, weights, and precomputed embeddings."""

    def __init__(self, pipeline: CogVideoXImageToVideoPipeline, empty_prompt_embeds: torch.Tensor):
        self.pipeline = pipeline
        self.empty_prompt_embeds = empty_prompt_embeds
        self.device = pipeline.device
        self.dtype = pipeline.dtype

    @classmethod
    def load(cls, model_dir: str = SPARKVSR_MODEL_DIR, embed_path: str = EMPTY_PROMPT_EMBED_PATH) -> "SparkVSRPipeline":
        """Load CogVideoX pipeline without T5 text encoder using precomputed embedding."""
        if not os.path.exists(embed_path):
            raise FileNotFoundError(
                f"Empty-prompt embedding missing at {embed_path}. "
                "Run `modal run -m modal_app.download_weights` first."
            )

        print(f"Loading SparkVSR CogVideoX pipeline from {model_dir} (bfloat16)...")
        # Enable CUDA / cuDNN performance flags
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Attention is the dominant cost in the transformer forward and grows quadratically
        # with resolution. Make the fast kernels explicitly available rather than trusting
        # the default backend order, and keep the math fallback so an unsupported shape
        # degrades instead of raising.
        for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
            with contextlib.suppress(AttributeError, RuntimeError):
                getattr(torch.backends.cuda, name)(True)
        with contextlib.suppress(AttributeError, RuntimeError):
            torch.backends.cuda.enable_cudnn_sdp(True)

        pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
            model_dir,
            text_encoder=None,
            tokenizer=None,
            torch_dtype=torch.bfloat16,
        )

        # Configure trailing timestep scheduler. from_config wants the config dict the
        # pipeline already loaded, not a directory path — passing a path is deprecated
        # and slated for removal in diffusers v1.0.0.
        pipeline.scheduler = CogVideoXDPMScheduler.from_config(
            pipeline.scheduler.config,
            timestep_spacing="trailing",
        )

        pipeline.to("cuda")

        # Enable VAE slicing and tiling for memory efficiency
        if hasattr(pipeline.vae, "enable_slicing"):
            pipeline.vae.enable_slicing()
        if hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()

        print(f"Loading empty-prompt embedding from {embed_path}...")
        empty_prompt_embeds = torch.load(embed_path, map_location="cuda", weights_only=True)
        if empty_prompt_embeds.dtype != torch.bfloat16:
            empty_prompt_embeds = empty_prompt_embeds.to(torch.bfloat16)

        _apply_inference_optimizations(pipeline)

        return cls(pipeline, empty_prompt_embeds)


def _apply_inference_optimizations(pipeline) -> None:
    """Apply FP8 quantisation and torch.compile to the loaded pipeline, in that order.

    Order matters: torchao swaps the linear layers, so quantising after compiling would
    invalidate the compiled graph.

    Every step degrades gracefully. An optimisation that fails to apply prints why and
    leaves the pipeline in its previous working state, because a slower correct render
    beats a container that will not start.
    """
    if FP8_QUANTIZE:
        try:
            from torchao.quantization import (
                float8_dynamic_activation_float8_weight,
                quantize_,
            )

            print("Applying FP8 dynamic quantisation to the transformer...")
            quantize_(pipeline.transformer, float8_dynamic_activation_float8_weight())
            print("FP8 quantisation applied.")
        except Exception as e:  # noqa: BLE001 - never block startup on an optimisation
            print(f"FP8 quantisation unavailable, continuing in bfloat16: {e}")

    if TORCH_COMPILE:
        try:
            print(f"Compiling transformer (mode={TORCH_COMPILE_MODE})...")
            pipeline.transformer = torch.compile(
                pipeline.transformer,
                mode=TORCH_COMPILE_MODE,
                # Window length varies at shot ends, so a static graph would recompile on
                # every odd-sized window. Dynamic shapes trade a little peak speed for one
                # compile instead of many.
                dynamic=True,
            )
            if TORCH_COMPILE_VAE:
                pipeline.vae.decode = torch.compile(
                    pipeline.vae.decode, mode=TORCH_COMPILE_MODE, dynamic=True
                )
            print("Compile registered (graph builds on the first window).")
        except Exception as e:  # noqa: BLE001
            print(f"torch.compile unavailable, continuing eager: {e}")


def process_video_ref_i2v(
    spark_pipeline: SparkVSRPipeline,
    video: torch.Tensor,
    ref_dict: Dict[int, torch.Tensor],
    sr_noise_step: int = DEFAULT_SR_NOISE_STEP,
    ref_guidance_scale: float = 1.0,
    noise_step: int = 0,
) -> torch.Tensor:
    """Run SparkVSR single-step super-resolution on a single chunk.

    Args:
        spark_pipeline: Loaded SparkVSRPipeline wrapper
        video: Input low-resolution/pre-scaled video tensor [1, C, T, H, W] in [0, 1]
        ref_dict: Dictionary mapping local frame index (0..T-1) -> HR reference tensor [C, H, W] in [0, 1]
        sr_noise_step: Timestep fed to the transformer (default 399)
        ref_guidance_scale: CFG guidance scale for reference adherence (1.0 = single pass)
        noise_step: Optional forward-diffusion noise level applied to the LQ latent.
            Upstream defaults this to 0, i.e. no noise is added at all.

    Returns:
        Restored output video tensor [1, C, T, H, W] in [0, 1]
    """
    pipe = spark_pipeline.pipeline
    device = pipe.device
    dtype = pipe.dtype

    # Pad video to 8n+1 frames and spatial multiples of 16
    video_padded, orig_t, orig_h, orig_w, pad_t, pad_h, pad_w = preprocess_video_match(video)
    video_padded = video_padded.to(device=device, dtype=dtype)

    # Normalize to [-1, 1] for the VAE (upstream callers hand the VAE [-1, 1] directly)
    video_norm = (video_padded * 2.0) - 1.0

    with torch.no_grad():
        # 1. VAE encode low-quality video latents
        scaling_factor = pipe.vae.config.scaling_factor
        lq_latent = pipe.vae.encode(video_norm).latent_dist.sample() * scaling_factor
        # lq_latent: [B, 16, F_lat, H_lat, W_lat]

        batch_size, _num_channels, num_frames, height, width = lq_latent.shape

        # 2. Construct reference condition latents
        full_ref_latent = torch.zeros_like(lq_latent)

        for frame_idx, ref_frame in ref_dict.items():
            if frame_idx < 0 or frame_idx >= orig_t:
                continue
            target_lat_idx = frame_idx // 4
            if not (0 <= target_lat_idx < num_frames):
                continue

            # Reference frame [C, H, W] -> [1, C, 4, H, W] normalized to [-1, 1]
            ref_tensor = ref_frame.unsqueeze(0).to(device=device, dtype=dtype)
            if pad_h > 0 or pad_w > 0:
                ref_tensor = F.pad(ref_tensor, (0, pad_w, 0, pad_h), mode="replicate")
            ref_norm = (ref_tensor * 2.0) - 1.0
            # Repeat 4x temporally to form a valid VAE chunk
            chunk = ref_norm.unsqueeze(2).repeat(1, 1, 4, 1, 1)

            lat = pipe.vae.encode(chunk).latent_dist.sample() * scaling_factor
            full_ref_latent[:, :, target_lat_idx, :, :] = lat[0, :, 0, :, :]

        # 3. Dual-pass / CFG assembly (uncond first, then cond -- matches upstream)
        do_classifier_free_guidance = abs(ref_guidance_scale - 1.0) > 1e-3

        if do_classifier_free_guidance:
            input_latent_cond = torch.cat([lq_latent, full_ref_latent], dim=1)
            uncond_ref_latent = torch.zeros_like(full_ref_latent)
            input_latent_uncond = torch.cat([lq_latent, uncond_ref_latent], dim=1)
            # [2*B, 32, F, H, W]
            input_latent = torch.cat([input_latent_uncond, input_latent_cond], dim=0)
        else:
            input_latent = torch.cat([lq_latent, full_ref_latent], dim=1)  # [B, 32, F, H, W]

        # 4. patch_size_t frame replication
        patch_size_t = pipe.transformer.config.patch_size_t
        ncopy = 0
        if patch_size_t is not None:
            ncopy = input_latent.shape[2] % patch_size_t
            first_frame = input_latent[:, :, :1, :, :]
            input_latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), input_latent], dim=2)

        # 5. Prompt embedding (text encoder removed: always the precomputed empty prompt)
        prompt_embedding = spark_pipeline.empty_prompt_embeds.to(device=device, dtype=dtype)
        if prompt_embedding.shape[0] != batch_size:
            prompt_embedding = prompt_embedding.repeat(batch_size, 1, 1)

        # Transformer expects [B, F, C, H, W]
        latents = input_latent.permute(0, 2, 1, 3, 4)

        if do_classifier_free_guidance:
            prompt_embedding = torch.cat([prompt_embedding, prompt_embedding], dim=0)

        # 6. Optional forward-diffusion noise on the LQ half only (default: skipped)
        if noise_step != 0:
            lq_part = latents[:, :, :16, :, :]
            ref_part = latents[:, :, 16:, :, :]

            noise = torch.randn_like(lq_part)
            add_timesteps = torch.full(
                (latents.shape[0],),
                fill_value=noise_step,
                dtype=torch.long,
                device=device,
            )
            lq_part = pipe.scheduler.add_noise(
                lq_part.transpose(1, 2), noise.transpose(1, 2), add_timesteps
            ).transpose(1, 2)
            latents = torch.cat([lq_part, ref_part], dim=2)

        timesteps = torch.full(
            (latents.shape[0],),
            fill_value=sr_noise_step,
            dtype=torch.long,
            device=device,
        )

        # 7. Rotary positional embeddings
        vae_scale_factor_spatial = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        transformer_config = pipe.transformer.config
        rotary_emb = (
            prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # 8. ofs embedding (CogVideoX1.5 I2V)
        ofs = None
        if transformer_config.ofs_embed_dim is not None:
            ofs = torch.full((latents.shape[0],), fill_value=2.0, device=device, dtype=dtype)

        # 9. Single transformer forward
        predicted_noise = pipe.transformer(
            hidden_states=latents,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            ofs=ofs,
            return_dict=False,
        )[0]

        # 10. Recover the clean latent via the velocity parameterization
        predicted_noise_slice = predicted_noise[:, :, :16, :, :].transpose(1, 2)
        lq_sample = latents[:, :, :16, :, :].transpose(1, 2)

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = predicted_noise_slice.chunk(2)
            predicted_noise_slice = noise_pred_uncond + ref_guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            lq_sample = lq_sample.chunk(2)[1]
            timesteps = timesteps.chunk(2)[0]

        latent_generate = pipe.scheduler.get_velocity(
            predicted_noise_slice, lq_sample, timesteps
        )

        if patch_size_t is not None and ncopy > 0:
            latent_generate = latent_generate[:, :, ncopy:, :, :]

        # 11. VAE decode and denormalize back to [0, 1]
        decoded = pipe.vae.decode(latent_generate / scaling_factor).sample
        out_video = (decoded * 0.5 + 0.5).clamp(0.0, 1.0)

        # 12. Remove temporal and spatial padding
        out_video = remove_padding_and_extra_frames(out_video, orig_t, orig_h, orig_w)

    return out_video


def execute_planned_windows(
    spark_pipeline: SparkVSRPipeline,
    video_tensor: torch.Tensor,
    windows: List[Dict[str, Any]],
    ref_frames: Dict[int, torch.Tensor],
    ref_guidance_scale: float = 1.0,
    tile_size: Optional[int] = None,
    tile_overlap: int = 64,
) -> torch.Tensor:
    """Execute super-resolution over all planned windows, blending intra-shot overlaps.

    Args:
        spark_pipeline: Loaded SparkVSR pipeline
        video_tensor: Pre-upscaled video tensor [1, C, Total_Frames, H, W]
        windows: List of window dicts produced by cutplan (carrying shot_id, start, end, ref_indices)
        ref_frames: Dict mapping global frame index -> reference frame tensor [C, H, W]
        ref_guidance_scale: Reference adherence strength
        tile_size: Optional spatial tile size for memory relief
        tile_overlap: Overlap between spatial tiles

    Returns:
        Complete restored video tensor [1, C, Total_Frames, H, W] on CPU
    """
    _, num_channels, total_frames, height, width = video_tensor.shape
    accum_video = torch.zeros((1, num_channels, total_frames, height, width), dtype=torch.float32)
    weight_map = torch.zeros((1, 1, total_frames, 1, 1), dtype=torch.float32)

    for w_idx, win in enumerate(windows):
        start = win["start_frame"]
        end = win["end_frame"]
        shot_id = win["shot_id"]
        win_ref_indices = win.get("ref_indices", [])

        # Extract window slice [1, C, T_win, H, W]
        win_slice = video_tensor[:, :, start:end, :, :]
        t_win = end - start

        # Map global reference indices to window-local indices
        local_ref_dict = {}
        for global_idx in win_ref_indices:
            if start <= global_idx < end and global_idx in ref_frames:
                local_ref_dict[global_idx - start] = ref_frames[global_idx]

        if tile_size is not None and (height > tile_size or width > tile_size):
            # Spatial tiling path
            tiles = make_spatial_tiles(height, width, tile_size=tile_size, overlap=tile_overlap)
            out_win = torch.zeros((1, num_channels, t_win, height, width), dtype=torch.float32, device="cuda")
            for t_i, (y0, y1, x0, x1) in enumerate(tiles):
                tile_in = win_slice[:, :, :, y0:y1, x0:x1]
                tile_ref_dict = {
                    k: v[:, y0:y1, x0:x1] for k, v in local_ref_dict.items()
                }
                tile_out = process_video_ref_i2v(
                    spark_pipeline,
                    tile_in,
                    tile_ref_dict,
                    ref_guidance_scale=ref_guidance_scale,
                )
                src_slice, dst_slice = get_valid_tile_region(
                    t_i, len(tiles), y0, y1, x0, x1, height, width, tile_overlap
                )
                out_win[:, :, :, dst_slice[0]:dst_slice[1], dst_slice[2]:dst_slice[3]] = (
                    tile_out[:, :, :, src_slice[0]:src_slice[1], src_slice[2]:src_slice[3]]
                )
            out_win_cpu = out_win.cpu()
        else:
            # Direct full-frame execution
            out_win = process_video_ref_i2v(
                spark_pipeline,
                win_slice,
                local_ref_dict,
                ref_guidance_scale=ref_guidance_scale,
            )
            out_win_cpu = out_win.cpu()

        # Compute blending weights for intra-shot overlaps
        # Find if previous window was in the same shot
        prev_in_shot = (w_idx > 0 and windows[w_idx - 1]["shot_id"] == shot_id)
        next_in_shot = (w_idx + 1 < len(windows) and windows[w_idx + 1]["shot_id"] == shot_id)

        w_weights = torch.ones((1, 1, t_win, 1, 1), dtype=torch.float32)

        # If overlapping with previous window in the same shot, ramp up linearly
        if prev_in_shot:
            prev_end = windows[w_idx - 1]["end_frame"]
            overlap_len = max(0, prev_end - start)
            if overlap_len > 0:
                ramp = torch.linspace(0.0, 1.0, overlap_len + 2)[1:-1].view(1, 1, overlap_len, 1, 1)
                w_weights[:, :, :overlap_len, :, :] = ramp

        # If overlapping with next window in the same shot, ramp down linearly
        if next_in_shot:
            next_start = windows[w_idx + 1]["start_frame"]
            overlap_len = max(0, end - next_start)
            if overlap_len > 0:
                ramp = torch.linspace(1.0, 0.0, overlap_len + 2)[1:-1].view(1, 1, overlap_len, 1, 1)
                w_weights[:, :, -overlap_len:, :, :] = torch.min(
                    w_weights[:, :, -overlap_len:, :, :], ramp
                )

        accum_video[:, :, start:end, :, :] += out_win_cpu * w_weights
        weight_map[:, :, start:end, :, :] += w_weights

    # Normalize by accumulated weights
    weight_map = torch.clamp(weight_map, min=1e-5)
    accum_video = accum_video / weight_map
    return torch.clamp(accum_video, 0.0, 1.0)
