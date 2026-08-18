"""FlashVSR-Pro inference service classes for Modal.

Three classes with different GPU requirements:
- FlashVSRFull:     A100-80GB, full mode (Wan2.1 VAE)
- FlashVSRTiny:     A10G, tiny mode (TCDecoder)
- FlashVSRTinyLong: A10G, tiny-long mode (TCDecoder)
"""

from __future__ import annotations

import io
import os
import sys
import time
from contextlib import redirect_stdout

import modal

from .app import app, io_volume, model_volume
from .config import (
    CONTAINER_IDLE_TIMEOUT,
    CONTAINER_TIMEOUT,
    DIT_PATH,
    GPU_FULL,
    GPU_TINY,
    GPU_TINY_LONG,
    IO_MOUNT_PATH,
    JOBS_PREFIX,
    LQ_PROJ_PATH,
    MODEL_DIR,
    MODEL_MOUNT_PATH,
    PROMPT_TENSOR_PATH,
    TCD_VAE_PATH,
    WAN_VAE_PATH,
)
from .image import flashvsr_image
from .types import InferenceRequest

# Re-exported for callers that still import it from here.
__all__ = ["FlashVSRFull", "FlashVSRTiny", "FlashVSRTinyLong", "InferenceRequest"]


def _load_models(mode: str, device: str = "cuda"):
    """Load FlashVSR-Pro models from Volume paths.

    Mirrors infer.py:init_pipeline() but uses Volume mount paths.
    Returns (pipe, vae_instance).
    """
    import torch

    sys.path.insert(0, "/workspace/FlashVSR-Pro")

    from diffsynth import (
        FlashVSRFullPipeline,
        FlashVSRTinyLongPipeline,
        FlashVSRTinyPipeline,
        ModelManager,
    )
    from utils import vae_manager
    from utils.utils import Causal_LQ4x_Proj

    dtype = torch.bfloat16

    # Determine VAE type
    vae_type = "wan2.1" if mode == "full" else "tcd"
    vae_weight = WAN_VAE_PATH if mode == "full" else TCD_VAE_PATH

    # Load VAE
    with redirect_stdout(io.StringIO()):
        vae_system = vae_manager.VAESystem(device=device, dtype=dtype)
        vae_model = vae_system.load_vae(
            vae_type=vae_type,
            weight_path=vae_weight,
            mode=mode,
            tile_vae=False,
            model_dir=MODEL_DIR,
        )

    print(f"VAE loaded: {vae_type}")

    # Load DiT
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    with redirect_stdout(io.StringIO()):
        mm.load_models([DIT_PATH])

    # Create pipeline
    with redirect_stdout(io.StringIO()):
        if mode == "full":
            pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
            pipe.vae = vae_model
        elif mode == "tiny":
            pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)
            pipe.TCDecoder = vae_model
        else:  # tiny-long
            pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
            pipe.TCDecoder = vae_model

    # Load LQ projector
    lq_proj = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)
    if os.path.exists(LQ_PROJ_PATH):
        # weights_only is explicit: torch >= 2.6 flipped the default, and leaving it
        # implicit makes the load behavior depend on the pinned torch version.
        lq_proj.load_state_dict(
            torch.load(LQ_PROJ_PATH, map_location="cpu", weights_only=True), strict=True
        )
    else:
        print(f"[Warning] LQ Projector not found at {LQ_PROJ_PATH}")

    # Load prompt tensor and pass explicitly to init_cross_kv
    ctx_tensor = torch.load(PROMPT_TENSOR_PATH, map_location=device, weights_only=True)

    with redirect_stdout(io.StringIO()):
        pipe.denoising_model().LQ_proj_in = lq_proj.to(device, dtype=dtype)
        pipe.to(device)
        pipe.enable_vram_management(num_persistent_param_in_dit=None)
        pipe.init_cross_kv(context_tensor=ctx_tensor)
        pipe.load_models_to_device(["dit", "vae"])

    print("Pipeline initialized.")
    return pipe, vae_system


def _run_inference(pipe, vae_instance, req: InferenceRequest):
    """Run inference on a single input. Mirrors infer.py:main() inference section."""
    import shutil
    import warnings

    import torch

    sys.path.insert(0, "/workspace/FlashVSR-Pro")

    try:
        from utils.audio_utils import copy_video_with_audio, has_audio_stream
        audio_available = True
    except ImportError as e:
        warnings.warn(f"Audio utilities not available: {e}")
        audio_available = False

        def has_audio_stream(path):
            return False

        def copy_video_with_audio(original_video_path, processed_video_path, output_path):
            shutil.copy2(processed_video_path, output_path)
            return True

    from .video_io import (
        prepare_input_tensor,
        save_video,
        save_video_with_audio_piped,
        tensor2video,
    )

    try:
        from utils.tile_utils import apply_tiled_inference_simple
    except ImportError:
        apply_tiled_inference_simple = None

    device = "cuda"
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(req.dtype, torch.bfloat16)

    input_abs = os.path.join(IO_MOUNT_PATH, req.input_path)
    output_abs = os.path.join(IO_MOUNT_PATH, req.output_path)
    os.makedirs(os.path.dirname(output_abs), exist_ok=True)

    # Reload volume to see latest uploads
    io_volume.reload()

    if not os.path.exists(input_abs):
        raise FileNotFoundError(f"Input not found on the io volume: {req.input_path}")

    # Prepare input
    LQ, th, tw, F, fps, input_video_path, total_frames_orig, exact_h, exact_w = prepare_input_tensor(
        input_abs, scale=req.scale, dtype=dtype, device=device
    )

    if req.fps is not None:
        fps = req.fps

    # Pipeline kwargs
    pipeline_kwargs = {
        "prompt": "",
        "negative_prompt": "",
        "cfg_scale": 1.0,
        "num_inference_steps": 1,
        "seed": req.seed,
        "LQ_video": LQ,
        "num_frames": F,
        "height": th,
        "width": tw,
        "is_full_block": False,
        "if_buffer": True,
        "topk_ratio": req.sparse_ratio * 768 * 1280 / (th * tw),
        "kv_ratio": req.kv_ratio,
        "local_range": req.local_range,
        "color_fix": req.color_fix,
    }

    # VAE tiling
    if req.tile_vae:
        pipeline_kwargs["tiled"] = True
        vae_tile_size_latent = max(32, req.tile_size // 8)
        vae_overlap_latent = max(4, req.overlap // 8)
        pipeline_kwargs["tile_size"] = (vae_tile_size_latent, vae_tile_size_latent)
        pipeline_kwargs["tile_stride"] = (
            vae_tile_size_latent - vae_overlap_latent,
            vae_tile_size_latent - vae_overlap_latent,
        )

    # Run inference
    inference_start = time.time()

    if req.tile_dit and apply_tiled_inference_simple is not None:
        tile_kwargs = pipeline_kwargs.copy()
        tile_kwargs.pop("LQ_video", None)
        vae_tile_size_tuple = tile_kwargs.pop("tile_size", None)
        video = apply_tiled_inference_simple(
            pipe,
            LQ,
            tile_size=req.tile_size,
            overlap=req.overlap,
            tile_size_vae=vae_tile_size_tuple,
            **tile_kwargs,
        )
    else:
        video = pipe(**pipeline_kwargs)

    print(f"Inference completed in {time.time() - inference_start:.2f}s")

    # Crop to exact resolution
    if video.shape[-2] != exact_h or video.shape[-1] != exact_w:
        curr_h, curr_w = video.shape[-2], video.shape[-1]
        pad_h = curr_h - exact_h
        pad_w = curr_w - exact_w
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        video = video[..., pad_top : pad_top + exact_h, pad_left : pad_left + exact_w]

    frames = tensor2video(video)

    # Match frame count
    if len(frames) > total_frames_orig:
        frames = frames[:total_frames_orig]
    elif len(frames) < total_frames_orig:
        frames.extend([frames[-1]] * (total_frames_orig - len(frames)))

    # Save with or without audio
    if req.keep_audio and audio_available and input_video_path and has_audio_stream(input_video_path):
        print("Preserving audio...")
        success = save_video_with_audio_piped(frames, output_abs, input_video_path, fps=fps, quality=req.quality)
        if not success:
            temp_output = output_abs.replace(".mp4", "_temp.mp4")
            save_video(frames, temp_output, fps=fps, quality=req.quality)
            copy_video_with_audio(input_video_path, temp_output, output_abs)
            if os.path.exists(temp_output):
                os.remove(temp_output)
    else:
        save_video(frames, output_abs, fps=fps, quality=req.quality)

    # HTTP jobs own their whole directory, so the upload is dead weight once the
    # output exists. User-managed `inputs/` files are never touched.
    if req.input_path.startswith(f"{JOBS_PREFIX}/") and os.path.exists(input_abs):
        os.remove(input_abs)

    # Commit output to volume
    io_volume.commit()

    print(f"Output saved: {output_abs}")
    return {"output_path": req.output_path, "status": "success", "job_id": req.job_id}


# ---------------------------------------------------------------------------
# Shared volumes config for all classes
# ---------------------------------------------------------------------------
_volumes = {MODEL_MOUNT_PATH: model_volume, IO_MOUNT_PATH: io_volume}


@app.cls(
    image=flashvsr_image,
    gpu=GPU_FULL,
    volumes=_volumes,
    scaledown_window=CONTAINER_IDLE_TIMEOUT,
    timeout=CONTAINER_TIMEOUT,
)
class FlashVSRFull:
    """Full-mode inference (Wan2.1 VAE) on A100-80GB."""

    @modal.enter()
    def setup(self):
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        self.pipe, self.vae_instance = _load_models("full")

    @modal.method()
    def infer(self, req: InferenceRequest) -> dict:
        return _run_inference(self.pipe, self.vae_instance, req)


@app.cls(
    image=flashvsr_image,
    gpu=GPU_TINY,
    volumes=_volumes,
    scaledown_window=CONTAINER_IDLE_TIMEOUT,
    timeout=CONTAINER_TIMEOUT,
)
class FlashVSRTiny:
    """Tiny-mode inference (TCDecoder) on A10G."""

    @modal.enter()
    def setup(self):
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        self.pipe, self.vae_instance = _load_models("tiny")

    @modal.method()
    def infer(self, req: InferenceRequest) -> dict:
        return _run_inference(self.pipe, self.vae_instance, req)


@app.cls(
    image=flashvsr_image,
    gpu=GPU_TINY_LONG,
    volumes=_volumes,
    scaledown_window=CONTAINER_IDLE_TIMEOUT,
    timeout=CONTAINER_TIMEOUT,
)
class FlashVSRTinyLong:
    """Tiny-long mode inference (TCDecoder, streaming) on A10G."""

    @modal.enter()
    def setup(self):
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        self.pipe, self.vae_instance = _load_models("tiny-long")

    @modal.method()
    def infer(self, req: InferenceRequest) -> dict:
        return _run_inference(self.pipe, self.vae_instance, req)
