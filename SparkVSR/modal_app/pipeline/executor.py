"""Streaming window executor for SparkVSR.

Replaces the whole-video accumulator in `core.execute_planned_windows` with a streaming
scheduler that never holds more than one window plus a small rolling blend buffer in RAM:

1. The source is held once as a uint8 CPU tensor at SOURCE resolution.
2. Each window is sliced, bilinearly upscaled to the target resolution ON THE GPU, and
   handed straight to inference. No whole-video pre-upscale ever exists.
3. Output frames are pushed to the encoder as soon as they finalize. Windows are ordered
   and only overlap their immediate neighbour, so once window `i` has been accumulated,
   every frame below `windows[i + 1]["start_frame"]` is final.
4. Overlaps are cross-faded ONLY between windows carrying the same `shot_id`. A window
   boundary that coincides with a scene cut is a hard cut in the output too — blending
   across it would ghost one shot into the next.

The window inference contract is `core.process_video_ref_i2v(spark_pipeline, video_BCTHW,
ref_dict_localidx_to_CHW, ref_guidance_scale=...) -> [1, C, T, H, W]` in [0, 1]. Nothing
here depends on its internals.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..config import PROFILE_FIRST_WINDOW
from .core import get_valid_tile_region, make_spatial_tiles, process_video_ref_i2v


def _run_window_profiled(*args, **kwargs) -> torch.Tensor:
    """Run one window under torch.profiler and print the CUDA-time breakdown.

    Used once per investigation to attribute the inference cost across VAE encode,
    the transformer forward and VAE decode, so optimisation effort goes where the
    time actually is. The window still renders normally.
    """
    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        out = _run_window(*args, **kwargs)
        torch.cuda.synchronize()

    print("=== Profiler: top CUDA operators for window 0 ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    return out


def compute_window_weights(
    windows: Sequence[Dict[str, Any]],
    index: int,
) -> List[float]:
    """Per-frame blend weights for `windows[index]`, ramping only inside the same shot.

    Ported from `core.execute_planned_windows`: linear ramp-up over the overlap shared with
    the previous window and linear ramp-down over the overlap shared with the next window,
    each gated on `shot_id` equality. Two adjacent in-shot ramps sum to exactly 1.0, and a
    frame that receives no ramp keeps weight 1.0 (so it is final the moment it is written).
    """
    win = windows[index]
    start = win["start_frame"]
    end = win["end_frame"]
    shot_id = win["shot_id"]
    t_win = end - start
    if t_win <= 0:
        return []

    weights = [1.0] * t_win

    prev_in_shot = index > 0 and windows[index - 1]["shot_id"] == shot_id
    if prev_in_shot:
        overlap_len = min(t_win, max(0, windows[index - 1]["end_frame"] - start))
        if overlap_len > 0:
            ramp = torch.linspace(0.0, 1.0, overlap_len + 2)[1:-1].tolist()
            for j in range(overlap_len):
                weights[j] = ramp[j]

    next_in_shot = index + 1 < len(windows) and windows[index + 1]["shot_id"] == shot_id
    if next_in_shot:
        overlap_len = min(t_win, max(0, end - windows[index + 1]["start_frame"]))
        if overlap_len > 0:
            ramp = torch.linspace(1.0, 0.0, overlap_len + 2)[1:-1].tolist()
            base = t_win - overlap_len
            for j in range(overlap_len):
                weights[base + j] = min(weights[base + j], ramp[j])

    return weights


def _upscale_to_target(
    frames_thwc_uint8: torch.Tensor,
    target_size: Tuple[int, int],
    device: str,
) -> torch.Tensor:
    """[T, H, W, C] uint8 CPU -> [1, C, T, out_h, out_w] float32 in [0, 1] on `device`."""
    out_h, out_w = target_size
    x = frames_thwc_uint8.to(device=device, non_blocking=True)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)  # [T, C, H, W]
    if x.shape[-2] != out_h or x.shape[-1] != out_w:
        x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # [1, C, T, out_h, out_w]


def _run_window(
    spark_pipeline: Any,
    window_input: torch.Tensor,
    local_ref_dict: Dict[int, torch.Tensor],
    ref_guidance_scale: float,
    tile_size: Optional[int],
    tile_overlap: int,
    device: str,
) -> torch.Tensor:
    """Run one window through SparkVSR, optionally spatially tiled. Returns [1, C, T, H, W]."""
    _, num_channels, t_win, height, width = window_input.shape

    if tile_size is None or (height <= tile_size and width <= tile_size):
        return process_video_ref_i2v(
            spark_pipeline,
            window_input,
            local_ref_dict,
            ref_guidance_scale=ref_guidance_scale,
        )

    tiles = make_spatial_tiles(height, width, tile_size=tile_size, overlap=tile_overlap)
    out_win = torch.zeros(
        (1, num_channels, t_win, height, width), dtype=torch.float32, device=device
    )
    for t_i, (y0, y1, x0, x1) in enumerate(tiles):
        tile_in = window_input[:, :, :, y0:y1, x0:x1]
        tile_ref_dict = {k: v[:, y0:y1, x0:x1] for k, v in local_ref_dict.items()}
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
            tile_out[:, :, :, src_slice[0]:src_slice[1], src_slice[2]:src_slice[3]].to(out_win.device)
        )
        del tile_out
    return out_win


def execute_windows_streaming(
    spark_pipeline: Any,
    source_frames_uint8: torch.Tensor,
    windows: Sequence[Dict[str, Any]],
    ref_frames: Dict[int, torch.Tensor],
    write_frame: Callable[[Any], None],
    target_size: Tuple[int, int],
    frame_offset: int = 0,
    ref_guidance_scale: float = 1.0,
    tile_size: Optional[int] = None,
    tile_overlap: int = 64,
    device: str = "cuda",
    accum_dtype: torch.dtype = torch.float16,
) -> Dict[str, Any]:
    """Execute the window plan and stream finalized frames to `write_frame`, in order.

    Args:
        spark_pipeline: Loaded SparkVSRPipeline
        source_frames_uint8: [T_seg, H, W, C] uint8 CPU tensor at SOURCE resolution,
            covering global frames [frame_offset, frame_offset + T_seg)
        windows: Window plan dicts with GLOBAL frame indices (start_frame, end_frame,
            shot_id, ref_indices). Windows outside the segment are ignored.
        ref_frames: Global frame index -> HR reference [C, out_h, out_w] in [0, 1]
        write_frame: Callable receiving one HWC uint8 RGB frame per finalized frame
        target_size: (out_h, out_w)
        frame_offset: Global index of `source_frames_uint8[0]`
        ref_guidance_scale: Reference adherence strength
        tile_size / tile_overlap: Optional spatial tiling for VRAM relief
        device: Inference device
        accum_dtype: Storage dtype for the rolling blend buffer (fp16 halves its footprint
            and is far finer than the 8-bit output quantization)

    Returns:
        Stats dict: frames_written, windows_executed, upscale_sec, inference_sec, blend_sec

    Raises:
        ValueError: if a window straddles the segment boundary (segments must be cut on
            shot boundaries, and no window ever crosses a shot boundary).
    """
    out_h, out_w = target_size
    n_src = int(source_frames_uint8.shape[0])
    seg_start = int(frame_offset)
    seg_end = seg_start + n_src
    src_is_target = (
        source_frames_uint8.shape[1] == out_h and source_frames_uint8.shape[2] == out_w
    )

    # Only windows belonging to this segment, in emission order.
    wins: List[Dict[str, Any]] = sorted(
        (
            w
            for w in windows
            if w["end_frame"] > seg_start and w["start_frame"] < seg_end
        ),
        key=lambda w: (w["start_frame"], w["end_frame"]),
    )
    for w in wins:
        if w["start_frame"] < seg_start or w["end_frame"] > seg_end:
            raise ValueError(
                f"Window {w.get('window_id')} [{w['start_frame']}, {w['end_frame']}) straddles "
                f"segment [{seg_start}, {seg_end}); segments must be split on shot boundaries."
            )

    stats = {
        "frames_written": 0,
        "windows_executed": 0,
        "upscale_sec": 0.0,
        "inference_sec": 0.0,
        "blend_sec": 0.0,
        # Per-window inference seconds. torch.compile pays a one-off graph build on the
        # first window, so only the steady-state windows say whether it actually helped;
        # a total would hide the answer behind the warmup on any short clip.
        "window_sec": [],
    }

    # Rolling blend buffer: global frame index -> [weighted_sum CHW, accumulated_weight].
    # Only frames inside an unresolved overlap ever live here.
    pending: Dict[int, List[Any]] = {}
    next_write = seg_start

    def _emit_passthrough(global_idx: int) -> None:
        """Frames covered by no window (plan gaps) are still emitted, bilinear-upscaled."""
        src = source_frames_uint8[global_idx - seg_start]
        if src_is_target:
            write_frame(src.contiguous().numpy())
            return
        up = _upscale_to_target(src.unsqueeze(0), (out_h, out_w), device)
        frame = up[0, :, 0].clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
        write_frame(frame.permute(1, 2, 0).cpu().contiguous().numpy())
        del up, frame

    def _flush(upto: int) -> None:
        nonlocal next_write
        t0 = time.perf_counter()
        while next_write < upto:
            g = next_write
            entry = pending.pop(g, None)
            if entry is None:
                _emit_passthrough(g)
            else:
                acc, weight = entry
                frame = acc.float()
                if weight > 0:
                    frame = frame / weight
                frame = frame.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
                write_frame(frame.permute(1, 2, 0).contiguous().numpy())
                del acc, frame
            stats["frames_written"] += 1
            next_write += 1
        stats["blend_sec"] += time.perf_counter() - t0

    for i, win in enumerate(wins):
        start = win["start_frame"]
        end = win["end_frame"]
        t_win = end - start
        if t_win <= 0:
            continue

        t_up = time.perf_counter()
        window_input = _upscale_to_target(
            source_frames_uint8[start - seg_start : end - seg_start],
            (out_h, out_w),
            device,
        )
        stats["upscale_sec"] += time.perf_counter() - t_up

        local_ref_dict = {
            g - start: ref_frames[g]
            for g in win.get("ref_indices", [])
            if start <= g < end and g in ref_frames
        }

        t_inf = time.perf_counter()
        if PROFILE_FIRST_WINDOW and i == 0:
            out_win = _run_window_profiled(
                spark_pipeline,
                window_input,
                local_ref_dict,
                ref_guidance_scale,
                tile_size,
                tile_overlap,
                device,
            )
        else:
            out_win = _run_window(
                spark_pipeline,
                window_input,
                local_ref_dict,
                ref_guidance_scale,
                tile_size,
                tile_overlap,
                device,
            )
        win_sec = time.perf_counter() - t_inf
        stats["inference_sec"] += win_sec
        stats["window_sec"].append(round(win_sec, 3))
        stats["windows_executed"] += 1
        del window_input

        t_blend = time.perf_counter()
        weights = compute_window_weights(wins, i)
        for j in range(t_win):
            g = start + j
            w_j = weights[j]
            frame = out_win[0, :, j].clamp(0.0, 1.0).mul(w_j).to("cpu", accum_dtype)
            entry = pending.get(g)
            if entry is None:
                pending[g] = [frame, w_j]
            else:
                entry[0].add_(frame)
                entry[1] += w_j
            del frame
        del out_win
        stats["blend_sec"] += time.perf_counter() - t_blend

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Every frame below the next window's start can no longer receive a contribution.
        boundary = wins[i + 1]["start_frame"] if i + 1 < len(wins) else seg_end
        _flush(min(max(boundary, next_write), seg_end))

    # Trailing frames plus any segment tail covered by no window at all.
    _flush(seg_end)

    if pending:
        raise RuntimeError(
            f"Streaming executor finished with {len(pending)} unflushed frames "
            f"(first={min(pending)}); frame accounting is broken."
        )
    if stats["frames_written"] != n_src:
        raise RuntimeError(
            f"Streaming executor wrote {stats['frames_written']} frames for a "
            f"{n_src}-frame segment; output length must equal input length."
        )

    return stats
