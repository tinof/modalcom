"""SparkVSR GPU Service running on RTX PRO 6000 (96 GB).

Implements the end-to-end super-resolution pipeline:
Probe -> CutPlan -> Segment -> [per segment: Decode -> ReferenceGen -> Streaming Windows
-> Streaming NVENC Encode] -> Concat (stream copy)

Memory model
------------
Nothing whole-video and float32 is ever allocated. The source is held once per segment as
a uint8 CPU tensor at SOURCE resolution (~6.2 MB/frame at 1080p, versus ~99.5 MB/frame for
float32 at 4K); the bilinear upscale to the target resolution happens per window on the
GPU; output frames are streamed to a long-lived ffmpeg process as soon as they finalize,
so no output tensor accumulates either.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

from .app import app, io_volume, models_volume
from .config import (
    CONTAINER_CPU_COUNT,
    CONTAINER_IDLE_TIMEOUT,
    CONTAINER_MEMORY_MB,
    CONTAINER_TIMEOUT,
    DEFAULT_CHUNK_SIZE,
    DRIVER_TIMEOUT,
    DEFAULT_OVERLAP,
    DEFAULT_REF_GUIDANCE_SCALE,
    DEFAULT_REF_MODE,
    DEFAULT_SR_NOISE_STEP,
    DEFAULT_TARGET_HEIGHT,
    DEFAULT_TARGET_WIDTH,
    EMPTY_PROMPT_EMBED_PATH,
    GPU_TYPE,
    IO_MOUNT_PATH,
    JOBS_DIR,
    MAX_PARALLEL_CONTAINERS,
    MODEL_MOUNT_PATH,
    PARALLEL_SEGMENT_TARGET_FRAMES,
    REF_MODE_PISASR,
    SPARKVSR_MODEL_DIR,
    WORKER_RETRIES,
)
from .image import gpu_image
from .weights import verify_weights
from .pipeline.cutplan import build_plan

# torch and every pipeline module that imports it exist only inside the GPU image.
# The local entrypoint imports this module to build the app graph, so these must be
# deferred to the container or `modal run` fails on the client with ModuleNotFoundError.
with gpu_image.imports():
    import torch

    from .pipeline.core import SparkVSRPipeline
    from .pipeline.executor import execute_windows_streaming
    from .pipeline.refgen import generate_reference_keyframes
    from .pipeline.video_io import (
        StreamingVideoEncoder,
        concat_segments_no_reencode,
        decode_video_to_uint8,
        extract_raw_reference_frames,
        probe_frame_count,
        probe_video_metadata,
        select_best_encoder,
    )

# --- Long-input segmentation thresholds -------------------------------------------------
#
# With streaming output a single pass costs roughly (source uint8 at source resolution) +
# (one window on the GPU) + (a rolling blend buffer of at most `overlap` target-resolution
# frames). At 1080p that is ~6.2 MB per source frame, so 1500 frames is ~9.3 GB of the
# 32 GB container budget, leaving ample headroom for reference keyframes and the blend
# buffer. 1500 frames is also ~60 s at 25 fps, which keeps one decode+inference+encode pass
# comfortably inside CONTAINER_TIMEOUT (3600 s) at realistic per-window inference costs.
# Below the threshold the single-pass path is kept, because segmentation costs an extra
# remux and forfeits cross-segment reference caching for no benefit.
SEGMENT_THRESHOLD_FRAMES = 1500
# Once segmenting, aim for segments of this length. Segment boundaries are ALWAYS shot
# boundaries taken from the cut plan, so a segment may overshoot this when a single shot is
# longer than the target: splitting mid-shot would place a join where the two sides were
# denoised independently, which is exactly the visible seam the spec forbids.
SEGMENT_TARGET_FRAMES = 1200


def compute_target_dimensions(
    orig_w: int,
    orig_h: int,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
) -> Tuple[int, int]:
    """Compute target dimensions preserving source aspect ratio."""
    if target_height is None and target_width is None:
        target_height = DEFAULT_TARGET_HEIGHT
        target_width = DEFAULT_TARGET_WIDTH

    if target_height is not None and target_width is None:
        scale = target_height / orig_h
        target_width = int(round(orig_w * scale))
    elif target_width is not None and target_height is None:
        scale = target_width / orig_w
        target_height = int(round(orig_h * scale))

    # Ensure even dimensions for video codecs
    target_width = (target_width // 2) * 2
    target_height = (target_height // 2) * 2
    return target_width, target_height


# Host RAM per source frame at 1080p is ~6.2 MB (uint8, HWC). A worker holds its whole
# segment decoded at once, so a segment's frame count is bounded by container memory.
BYTES_PER_SOURCE_FRAME_1080P = 1920 * 1080 * 3


def _preflight_plan(
    plan: Dict[str, Any], segments: List[Tuple[int, int]], total_frames: int
) -> None:
    """Validate a fan-out plan on CPU before any GPU container is started.

    Each failure here is one that would otherwise appear inside a worker, mid-fan-out,
    after the rest of the fleet had already been billed.
    """
    if not segments:
        raise RuntimeError("Planner produced no segments.")

    # Segments must tile [0, total_frames) exactly.
    if segments[0][0] != 0 or segments[-1][1] != total_frames:
        raise RuntimeError(
            f"Segments {segments[0][0]}..{segments[-1][1]} do not span [0, {total_frames})."
        )
    for (_, end), (nxt, _) in zip(segments, segments[1:]):
        if end != nxt:
            raise RuntimeError(f"Segments are not contiguous at frame {end} -> {nxt}.")

    # Every window must lie inside exactly one segment: the executor rejects a window that
    # straddles a segment boundary, and one that runs past the video end never renders.
    for w in plan.get("windows", []):
        ws, we = w["start_frame"], w["end_frame"]
        if we > total_frames:
            raise RuntimeError(
                f"Window {w.get('window_id')} ends at {we}, past the video's {total_frames} frames."
            )
        if not any(s <= ws and we <= e for s, e in segments):
            raise RuntimeError(
                f"Window {w.get('window_id')} [{ws}, {we}) straddles a segment boundary."
            )

    # A shot longer than the segment target cannot be split without placing a join where
    # the two sides denoised independently, so it stays whole on one worker. Fail here
    # rather than OOM on a billed GPU after a container start and a model load.
    longest = max(e - s for s, e in segments)
    budget_frames = int(CONTAINER_MEMORY_MB * 1024 * 1024 * 0.5 // BYTES_PER_SOURCE_FRAME_1080P)
    if longest > budget_frames:
        raise RuntimeError(
            f"The longest segment is {longest} frames, above the ~{budget_frames} a "
            f"{CONTAINER_MEMORY_MB} MB worker can hold decoded at 1080p. This happens when a "
            "single unbroken shot is longer than the segment target, since splitting mid-shot "
            "would create a visible seam. Raise container memory or split the source."
        )


def plan_segments(
    plan: Dict[str, Any],
    total_frames: int,
    threshold_frames: int = SEGMENT_THRESHOLD_FRAMES,
    target_frames: int = SEGMENT_TARGET_FRAMES,
) -> List[Tuple[int, int]]:
    """Split [0, total_frames) into half-open segments that begin on shot boundaries.

    Inputs at or below `threshold_frames` are returned as a single segment (the existing
    single-pass path). Above it, the only admissible split points are the shot starts from
    the cut plan, so every join in the concatenated output lands on a scene cut and
    introduces no visible discontinuity. No window can straddle a segment boundary, because
    the planner never lets a window cross a shot boundary.
    """
    if total_frames <= 0:
        return []
    if total_frames <= threshold_frames:
        return [(0, total_frames)]

    bounds = {0, total_frames}
    for shot in plan.get("shots", []):
        s = int(shot["start_frame"])
        if 0 < s < total_frames:
            bounds.add(s)
    sorted_bounds = sorted(bounds)

    segments: List[Tuple[int, int]] = []
    start_i = 0
    n = len(sorted_bounds)
    while start_i < n - 1:
        end_i = start_i + 1
        # Always take at least one shot, then extend while still within the target length.
        while end_i + 1 < n and (sorted_bounds[end_i + 1] - sorted_bounds[start_i]) <= target_frames:
            end_i += 1
        segments.append((sorted_bounds[start_i], sorted_bounds[end_i]))
        start_i = end_i

    return segments


def _optional_secret(name: str) -> list:
    """Mount a Secret only if it exists.

    Modal has no optional-secret flag: naming a missing Secret in the decorator
    fails the whole deploy. Only `--ref-mode api` needs `fal`, so probe for it
    and mount nothing when it is absent. The web tier rejects api-mode requests
    with instructions in that case, so the failure stays legible.
    """
    try:
        secret = modal.Secret.from_name(name)
        secret.hydrate()
        return [secret]
    except Exception:
        return []


@app.cls(
    image=gpu_image,
    gpu=GPU_TYPE,
    cpu=CONTAINER_CPU_COUNT,
    memory=CONTAINER_MEMORY_MB,
    timeout=CONTAINER_TIMEOUT,
    scaledown_window=CONTAINER_IDLE_TIMEOUT,
    # Bounds the fan-out. Every worker bills a full GPU, so this caps burn rate as well
    # as concurrency; Modal's own plan limit applies on top.
    max_containers=MAX_PARALLEL_CONTAINERS,
    retries=WORKER_RETRIES,
    volumes={
        MODEL_MOUNT_PATH: models_volume,
        IO_MOUNT_PATH: io_volume,
    },
    secrets=_optional_secret("fal"),
)
class SparkVSRService:
    @modal.enter()
    def setup(self):
        """Verify weights and load SparkVSR CogVideoX pipeline on container startup."""
        print(f"Starting SparkVSR container on {GPU_TYPE}...")

        # 1. Integrity check
        missing = verify_weights()
        if missing:
            raise RuntimeError(
                f"Container startup failed: missing model components: {missing}. "
                "Run `modal run -m modal_app.download_weights` to provision the volume."
            )

        # 2. Load model
        t0 = time.perf_counter()
        self.pipeline = SparkVSRPipeline.load(
            model_dir=SPARKVSR_MODEL_DIR,
            embed_path=EMPTY_PROMPT_EMBED_PATH,
        )
        load_sec = time.perf_counter() - t0
        print(f"SparkVSR pipeline loaded in {load_sec:.2f}s.")

        # 3. Probe video encoder
        self.encoder, self.tune = select_best_encoder()

    def _process_segment(
        self,
        *,
        job_id: str,
        seg_index: int,
        seg_total: int,
        seg_start: int,
        seg_end: int,
        input_path: str,
        output_path: str,
        job_dir: Path,
        plan: Dict[str, Any],
        meta,
        out_w: int,
        out_h: int,
        ref_mode: str,
        ref_guidance_scale: float,
        keep_audio: bool,
        tile_size: Optional[int],
        tile_overlap: int,
        timing: Dict[str, float],
    ) -> Dict[str, Any]:
        """Decode, reference, super-resolve and encode a single [seg_start, seg_end) segment.

        Only this segment's frames are ever resident, and its output frames are streamed to
        ffmpeg as they finalize.
        """
        label = f"[Job {job_id}] segment {seg_index + 1}/{seg_total} frames [{seg_start}, {seg_end})"

        # --- Decode (uint8, source resolution, this segment only) ---------------------
        t0 = time.perf_counter()
        frames_u8, _ = decode_video_to_uint8(input_path, start_frame=seg_start, end_frame=seg_end)
        timing["decode_sec"] += time.perf_counter() - t0
        src_gb = frames_u8.numel() / (1024 ** 3)
        print(f"{label}: decoded {frames_u8.shape[0]} frames as uint8 ({src_gb:.2f} GB host RAM)")

        seg_windows = [
            w for w in plan["windows"] if w["end_frame"] > seg_start and w["start_frame"] < seg_end
        ]
        seg_ref_indices = sorted(
            {idx for idx in plan["all_ref_indices"] if seg_start <= idx < seg_end}
        )

        # --- Reference keyframes for this segment only --------------------------------
        t0 = time.perf_counter()
        raw_refs = extract_raw_reference_frames(input_path, seg_ref_indices)
        ref_dict = generate_reference_keyframes(
            ref_mode=ref_mode,
            raw_frame_dict=raw_refs,
            target_width=out_w,
            target_height=out_h,
            job_dir=job_dir,
        )
        del raw_refs
        timing["ref_gen_sec"] += time.perf_counter() - t0
        print(f"{label}: {len(ref_dict)} reference frames, {len(seg_windows)} windows")

        # --- Streaming windows -> streaming encoder -----------------------------------
        encoder_ctx = StreamingVideoEncoder(
            output_path=output_path,
            width=out_w,
            height=out_h,
            fps=meta.fps,
            # Audio is stream-copied on the final (single-pass) encode, or in the concat
            # remux when the input was segmented.
            audio_source_path=input_path if seg_total == 1 else None,
            keep_audio=keep_audio if seg_total == 1 else False,
            encoder=self.encoder,
            tune=self.tune,
        )

        try:
            with encoder_ctx as enc:
                stats = execute_windows_streaming(
                    spark_pipeline=self.pipeline,
                    source_frames_uint8=frames_u8,
                    windows=seg_windows,
                    ref_frames=ref_dict,
                    write_frame=enc.write_frame,
                    target_size=(out_h, out_w),
                    frame_offset=seg_start,
                    ref_guidance_scale=ref_guidance_scale,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                raise RuntimeError(
                    f"CUDA Out of Memory during SparkVSR inference at {out_w}x{out_h}. "
                    "Run with spatial tiling enabled (--tile or tile_size=512) to fit VRAM."
                ) from e
            raise
        finally:
            del frames_u8, ref_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        timing["upscale_sec"] += stats["upscale_sec"]
        timing["inference_sec"] += stats["inference_sec"]
        timing["blend_sec"] += stats["blend_sec"]
        timing["encode_sec"] += encoder_ctx.write_seconds

        wins = stats.get("window_sec") or []
        steady = wins[1:] or wins  # drop the compile warmup window
        steady_median = sorted(steady)[len(steady) // 2] if steady else 0.0
        # Only the scalar goes into `timing`; callers format that dict as floats.
        timing["steady_window_sec"] = steady_median

        print(
            f"{label}: {stats['windows_executed']} windows, "
            f"{stats['frames_written']} frames written "
            f"(infer {stats['inference_sec']:.1f}s, upscale {stats['upscale_sec']:.1f}s, "
            f"encode {encoder_ctx.write_seconds:.1f}s)"
        )
        if wins:
            print(
                f"{label}: per-window inference {wins} | "
                f"first {wins[0]:.1f}s, steady-state median {steady_median:.1f}s"
            )
        return stats

    @modal.method()
    def process_segment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Render one segment of a job. The unit of work for the parallel path.

        Segments are cut only on shot boundaries and windows never straddle a shot, so a
        segment has no dependency on any other segment: no halo frames, no shared blend
        state, nothing to exchange between workers. Each one decodes its own frame range,
        builds its own references, and writes a self-contained HEVC segment that the driver
        later concatenates with `-c copy`.

        The payload carries only this segment's windows and reference indices rather than
        the whole plan, and audio is left to the concat pass.
        """
        job_id = payload["job_id"]
        job_dir = Path(JOBS_DIR) / job_id
        input_path = str(job_dir / payload["input_filename"])

        # The driver wrote the segment directory after this container may have started.
        io_volume.reload()

        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Job input file not found: {input_path}")

        meta = probe_video_metadata(input_path)
        meta.total_frames = probe_frame_count(input_path)
        out_w, out_h = compute_target_dimensions(
            meta.width, meta.height, payload["target_height"], payload["target_width"]
        )

        timing = {k: 0.0 for k in ("decode_sec", "ref_gen_sec", "upscale_sec", "inference_sec", "blend_sec", "encode_sec")}

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        stats = self._process_segment(
            job_id=job_id,
            seg_index=payload["seg_index"],
            seg_total=payload["seg_total"],
            seg_start=payload["seg_start"],
            seg_end=payload["seg_end"],
            input_path=input_path,
            output_path=str(job_dir / payload["segment_rel_path"]),
            job_dir=job_dir,
            # _process_segment only ever reads these two keys off the plan.
            plan={
                "windows": payload["windows"],
                "all_ref_indices": payload["all_ref_indices"],
            },
            meta=meta,
            out_w=out_w,
            out_h=out_h,
            ref_mode=payload["ref_mode"],
            ref_guidance_scale=payload["ref_guidance_scale"],
            # Audio is attached once, by the driver's concat pass.
            keep_audio=False,
            tile_size=payload["tile_size"],
            tile_overlap=payload["tile_overlap"],
            timing=timing,
        )

        io_volume.commit()

        peak_vram_gb = (
            torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        )
        return {
            "seg_index": payload["seg_index"],
            "seg_start": payload["seg_start"],
            "seg_end": payload["seg_end"],
            "segment_rel_path": payload["segment_rel_path"],
            "windows_executed": stats["windows_executed"],
            "frames_written": stats["frames_written"],
            "peak_vram_gb": peak_vram_gb,
            "encoder": self.encoder,
            "tune": self.tune,
            "timing": timing,
        }

    @modal.method()
    def process_video(
        self,
        job_id: str,
        input_filename: str,
        output_filename: str = "output.mp4",
        target_height: Optional[int] = DEFAULT_TARGET_HEIGHT,
        target_width: Optional[int] = None,
        ref_mode: str = DEFAULT_REF_MODE,
        ref_guidance_scale: float = DEFAULT_REF_GUIDANCE_SCALE,
        cut_aware: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        keep_audio: bool = True,
        tile_size: Optional[int] = None,
        tile_overlap: int = 64,
    ) -> Dict[str, Any]:
        """Execute super-resolution on an uploaded video job.

        Args:
            job_id: Unique job identifier
            input_filename: Input video filename inside jobs/<job_id>/
            output_filename: Output video filename inside jobs/<job_id>/
            target_height: Requested target height (preserves aspect ratio)
            target_width: Requested target width
            ref_mode: Reference generation mode ("pisasr", "api", "no_ref")
            ref_guidance_scale: Reference adherence strength (1.0 = single pass)
            cut_aware: Whether to use scene-cut aware chunking
            chunk_size: Frames per temporal chunk
            overlap: Overlap between consecutive chunks in the same shot
            keep_audio: Stream-copy audio track from source
            tile_size: Optional spatial tile size for memory relief
            tile_overlap: Spatial tile overlap

        Returns:
            Dict containing job status, output path, metrics, and plan
        """
        job_dir = Path(JOBS_DIR) / job_id
        input_path = str(job_dir / input_filename)
        output_path = str(job_dir / output_filename)
        plan_path = job_dir / "plan.json"

        # Mounted Volume state is fixed at container creation; a warm container reused for
        # a later job would not see that job's uploaded input without this.
        io_volume.reload()

        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Job input file not found: {input_path}")

        timing = {
            "probe_sec": 0.0,
            "planning_sec": 0.0,
            "decode_sec": 0.0,
            "ref_gen_sec": 0.0,
            "upscale_sec": 0.0,
            "inference_sec": 0.0,
            "blend_sec": 0.0,
            "encode_sec": 0.0,
            "concat_sec": 0.0,
        }
        total_t0 = time.perf_counter()

        # Reset GPU peak memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        print(f"=== [Job {job_id}] Step 1/5: Probing Input ===")
        t0 = time.perf_counter()
        meta = probe_video_metadata(input_path)
        # decord's frame count is authoritative; ffprobe's nb_frames may only be an estimate.
        meta.total_frames = probe_frame_count(input_path)
        timing["probe_sec"] = time.perf_counter() - t0
        print(
            f"Probed {meta.total_frames} frames ({meta.width}x{meta.height} @ {meta.fps:.2f} fps) "
            f"in {timing['probe_sec']:.2f}s"
        )
        if meta.total_frames <= 0:
            raise ValueError(f"Input video has no decodable frames: {input_path}")

        out_w, out_h = compute_target_dimensions(meta.width, meta.height, target_height, target_width)
        print(f"Target dimensions: {out_w}x{out_h} (source: {meta.width}x{meta.height})")

        print(f"=== [Job {job_id}] Step 2/5: Scene-Cut & Window Planning ===")
        t0 = time.perf_counter()
        plan = build_plan(
            video_path=input_path,
            total_frames=meta.total_frames,
            fps=meta.fps,
            cut_aware=cut_aware,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        timing["planning_sec"] = time.perf_counter() - t0
        print(
            f"Plan generated: {len(plan['shots'])} shots, {len(plan['windows'])} windows, "
            f"{len(plan['all_ref_indices'])} reference frames in {timing['planning_sec']:.2f}s"
        )

        print(f"=== [Job {job_id}] Step 3/5: Long-Input Segmentation ===")
        segments = plan_segments(plan, meta.total_frames)
        plan["segment_threshold_frames"] = SEGMENT_THRESHOLD_FRAMES
        plan["segment_target_frames"] = SEGMENT_TARGET_FRAMES
        plan["segments"] = [{"start_frame": s, "end_frame": e} for s, e in segments]
        with open(plan_path, "w") as f:
            json.dump(plan, f, indent=2)

        if len(segments) == 1:
            print(
                f"{meta.total_frames} frames <= {SEGMENT_THRESHOLD_FRAMES} frame threshold: "
                "single-pass path."
            )
        else:
            print(
                f"{meta.total_frames} frames > {SEGMENT_THRESHOLD_FRAMES} frame threshold: "
                f"{len(segments)} segments split on shot boundaries "
                f"{[f'[{s},{e})' for s, e in segments]}"
            )

        print(f"=== [Job {job_id}] Step 4/5: Streaming Super-Resolution ===")
        seg_dir = job_dir / "segments"
        segment_paths: List[str] = []
        total_windows_run = 0
        total_frames_written = 0

        if len(segments) > 1:
            shutil.rmtree(seg_dir, ignore_errors=True)
            seg_dir.mkdir(parents=True, exist_ok=True)

        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            if len(segments) == 1:
                seg_out = output_path
            else:
                seg_out = str(seg_dir / f"seg_{seg_idx:04d}.mp4")
            segment_paths.append(seg_out)

            stats = self._process_segment(
                job_id=job_id,
                seg_index=seg_idx,
                seg_total=len(segments),
                seg_start=seg_start,
                seg_end=seg_end,
                input_path=input_path,
                output_path=seg_out,
                job_dir=job_dir,
                plan=plan,
                meta=meta,
                out_w=out_w,
                out_h=out_h,
                ref_mode=ref_mode,
                ref_guidance_scale=ref_guidance_scale,
                keep_audio=keep_audio,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                timing=timing,
            )
            total_windows_run += stats["windows_executed"]
            total_frames_written += stats["frames_written"]

        if total_frames_written != meta.total_frames:
            raise RuntimeError(
                f"Output frame count {total_frames_written} does not match input frame count "
                f"{meta.total_frames}."
            )

        print(f"=== [Job {job_id}] Step 5/5: Assembling Output ===")
        if len(segments) > 1:
            t0 = time.perf_counter()
            concat_segments_no_reencode(
                segment_paths=segment_paths,
                output_path=output_path,
                audio_source_path=input_path,
                keep_audio=keep_audio,
                work_dir=str(seg_dir),
            )
            timing["concat_sec"] = time.perf_counter() - t0
            print(
                f"Concatenated {len(segment_paths)} segments with -c copy (no re-encode) "
                f"in {timing['concat_sec']:.2f}s"
            )
            shutil.rmtree(seg_dir, ignore_errors=True)
        else:
            print("Single segment: output written directly by the streaming encoder.")

        # See the parallel driver: frames_written is counted upstream of the muxer, so it
        # cannot see a truncated or short-concatenated file. Probe what actually shipped.
        actual_frames = probe_frame_count(output_path)
        if actual_frames != meta.total_frames:
            raise RuntimeError(
                f"Encoded output has {actual_frames} frames but the source has "
                f"{meta.total_frames}. Frames were lost after inference (muxing or concat)."
            )

        total_sec = time.perf_counter() - total_t0
        timing["total_sec"] = total_sec

        peak_vram_gb = 0.0
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        print(f"=== [Job {job_id}] Complete in {total_sec:.2f}s (Peak VRAM: {peak_vram_gb:.2f} GB) ===")
        io_volume.commit()

        return {
            "status": "completed",
            "job_id": job_id,
            "output_path": output_path,
            "output_filename": output_filename,
            "target_width": out_w,
            "target_height": out_h,
            "total_frames": meta.total_frames,
            "frames_written": total_frames_written,
            "num_windows": len(plan["windows"]),
            "num_windows_executed": total_windows_run,
            "num_shots": len(plan["shots"]),
            "num_segments": len(segments),
            "segments": [{"start_frame": s, "end_frame": e} for s, e in segments],
            "num_references": len(plan["all_ref_indices"]),
            "peak_vram_gb": peak_vram_gb,
            "container_memory_mb": CONTAINER_MEMORY_MB,
            "encoder": self.encoder,
            "timing": timing,
        }


@app.function(
    image=gpu_image,
    # The driver probes, plans, fans out and concatenates. None of that touches the GPU,
    # and it spends most of its life blocked on workers, so giving it a GPU would bill an
    # idle card for the whole job.
    cpu=CONTAINER_CPU_COUNT,
    memory=CONTAINER_MEMORY_MB,
    timeout=DRIVER_TIMEOUT,
    volumes={IO_MOUNT_PATH: io_volume},
)
def process_video_parallel(
    job_id: str,
    input_filename: str,
    output_filename: str = "output.mp4",
    target_height: Optional[int] = DEFAULT_TARGET_HEIGHT,
    target_width: Optional[int] = None,
    ref_mode: str = DEFAULT_REF_MODE,
    ref_guidance_scale: float = DEFAULT_REF_GUIDANCE_SCALE,
    cut_aware: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    keep_audio: bool = True,
    tile_size: Optional[int] = None,
    tile_overlap: int = 64,
    segment_frames: int = PARALLEL_SEGMENT_TARGET_FRAMES,
) -> Dict[str, Any]:
    """Render a job by fanning its segments across GPU workers, then concatenating.

    Same contract as `SparkVSRService.process_video`, but wall clock divides by the worker
    count at identical total GPU cost, because segments are independent.
    """
    job_dir = Path(JOBS_DIR) / job_id
    input_path = str(job_dir / input_filename)
    output_path = str(job_dir / output_filename)

    # A Volume is mounted at its state as of container creation. This driver keeps a warm
    # container after a job, so a second job dispatched into it would not see its own
    # freshly uploaded input without an explicit reload.
    io_volume.reload()

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Job input file not found: {input_path}")

    timing = {"probe_sec": 0.0, "planning_sec": 0.0, "dispatch_sec": 0.0, "concat_sec": 0.0}
    total_t0 = time.perf_counter()

    print(f"=== [Job {job_id}] Step 1/4: Probing Input ===")
    t0 = time.perf_counter()
    meta = probe_video_metadata(input_path)
    meta.total_frames = probe_frame_count(input_path)
    timing["probe_sec"] = time.perf_counter() - t0
    if meta.total_frames <= 0:
        raise ValueError(f"Input video has no decodable frames: {input_path}")

    out_w, out_h = compute_target_dimensions(meta.width, meta.height, target_height, target_width)
    print(
        f"Probed {meta.total_frames} frames ({meta.width}x{meta.height} @ {meta.fps:.2f} fps); "
        f"target {out_w}x{out_h}"
    )

    print(f"=== [Job {job_id}] Step 2/4: Planning & Segmentation ===")
    t0 = time.perf_counter()
    plan = build_plan(
        video_path=input_path,
        total_frames=meta.total_frames,
        fps=meta.fps,
        cut_aware=cut_aware,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    # threshold_frames=0 forces a split at every admissible shot boundary rather than
    # only for long inputs: here segmentation exists to create parallelism, not to bound
    # memory. Boundaries are still shot boundaries, so joins stay invisible.
    segments = plan_segments(plan, meta.total_frames, threshold_frames=0, target_frames=segment_frames)
    timing["planning_sec"] = time.perf_counter() - t0
    print(
        f"Plan: {len(plan['shots'])} shots, {len(plan['windows'])} windows -> "
        f"{len(segments)} parallel segments (target {segment_frames} frames)"
    )

    # Pre-flight on CPU, before a single GPU starts. Every check below would otherwise
    # surface inside a worker, after the rest of the fleet had already been paid for.
    _preflight_plan(plan, segments, meta.total_frames)

    seg_dir = job_dir / "segments"
    shutil.rmtree(seg_dir, ignore_errors=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    plan["segments"] = [{"start_frame": s, "end_frame": e} for s, e in segments]
    plan["parallel"] = True
    with open(job_dir / "plan.json", "w") as f:
        json.dump(plan, f, indent=2)
    io_volume.commit()

    payloads = []
    for idx, (seg_start, seg_end) in enumerate(segments):
        seg_windows = [
            w for w in plan["windows"] if w["end_frame"] > seg_start and w["start_frame"] < seg_end
        ]
        seg_refs = sorted({i for i in plan["all_ref_indices"] if seg_start <= i < seg_end})
        payloads.append({
            "job_id": job_id,
            "input_filename": input_filename,
            "segment_rel_path": f"segments/seg_{idx:04d}.mp4",
            "seg_index": idx,
            "seg_total": len(segments),
            "seg_start": seg_start,
            "seg_end": seg_end,
            "windows": seg_windows,
            "all_ref_indices": seg_refs,
            "target_height": target_height,
            "target_width": target_width,
            "ref_mode": ref_mode,
            "ref_guidance_scale": ref_guidance_scale,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
        })

    print(f"=== [Job {job_id}] Step 3/4: Fan-out across up to {MAX_PARALLEL_CONTAINERS} GPUs ===")
    t0 = time.perf_counter()
    service = SparkVSRService()
    results = list(service.process_segment.map(payloads, order_outputs=True))
    timing["dispatch_sec"] = time.perf_counter() - t0

    # Ordering is guaranteed by order_outputs=True, but a permutation is the one corruption
    # the frame-count check structurally cannot see: the sum is identical for any order, so
    # a scrambled episode would pass every other gate. Assert the tiling explicitly.
    results.sort(key=lambda r: r["seg_index"])
    if [r["seg_start"] for r in results] != [s for s, _ in segments]:
        raise RuntimeError("Segment results do not match the dispatched plan order.")
    if results[0]["seg_start"] != 0 or results[-1]["seg_end"] != meta.total_frames:
        raise RuntimeError("Segment results do not span the full video.")
    for prev, nxt in zip(results, results[1:]):
        if prev["seg_end"] != nxt["seg_start"]:
            raise RuntimeError(
                f"Gap or overlap between segments {prev['seg_index']} and {nxt['seg_index']}."
            )

    # Every worker probes NVENC independently, and select_best_encoder falls back to 8-bit
    # libx264 on a transient probe failure. Concatenating mixed codecs with -c copy yields
    # a file most decoders cut short at the switch, so refuse before writing it.
    encoders = {(r.get("encoder"), r.get("tune")) for r in results}
    if len(encoders) > 1:
        raise RuntimeError(
            f"Workers disagreed on the video encoder ({sorted(str(e) for e in encoders)}); "
            "the segments cannot be concatenated with -c copy. Re-run the job."
        )

    frames_written = sum(r["frames_written"] for r in results)
    windows_run = sum(r["windows_executed"] for r in results)
    peak_vram_gb = max((r["peak_vram_gb"] for r in results), default=0.0)
    for key in ("decode_sec", "ref_gen_sec", "upscale_sec", "inference_sec", "blend_sec", "encode_sec"):
        # Summed across workers, so this is GPU-seconds spent, not wall clock.
        timing[key] = sum(r["timing"].get(key, 0.0) for r in results)
    steady = [r["timing"].get("steady_window_sec", 0.0) for r in results if r["timing"].get("steady_window_sec")]
    timing["steady_window_sec"] = sorted(steady)[len(steady) // 2] if steady else 0.0

    if frames_written != meta.total_frames:
        raise RuntimeError(
            f"Output frame count {frames_written} does not match input frame count "
            f"{meta.total_frames}."
        )

    print(f"=== [Job {job_id}] Step 4/4: Assembling Output ===")
    io_volume.reload()
    t0 = time.perf_counter()
    concat_segments_no_reencode(
        segment_paths=[str(job_dir / r["segment_rel_path"]) for r in results],
        output_path=output_path,
        audio_source_path=input_path,
        keep_audio=keep_audio,
        work_dir=str(seg_dir),
    )
    timing["concat_sec"] = time.perf_counter() - t0
    shutil.rmtree(seg_dir, ignore_errors=True)

    # frames_written counts frames pushed into each segment encoder's stdin, which is
    # upstream of every way the muxer can still lose them: the -shortest truncation that
    # cost 5 frames twice already, and B-frame/timebase mismatches at the ~180 join points
    # of an episode-length concat. Probe the file that actually ships.
    actual_frames = probe_frame_count(output_path)
    if actual_frames != meta.total_frames:
        raise RuntimeError(
            f"Encoded output has {actual_frames} frames but the source has "
            f"{meta.total_frames}. Frames were lost after inference (muxing or concat)."
        )

    total_sec = time.perf_counter() - total_t0
    timing["total_sec"] = total_sec
    gpu_sec = sum(r["timing"].get("inference_sec", 0.0) for r in results)
    print(
        f"=== [Job {job_id}] Complete in {total_sec:.2f}s wall "
        f"({gpu_sec:.0f}s GPU inference across {len(results)} workers, "
        f"peak VRAM {peak_vram_gb:.2f} GB) ==="
    )
    io_volume.commit()

    return {
        "status": "completed",
        "job_id": job_id,
        "output_path": output_path,
        "output_filename": output_filename,
        "target_width": out_w,
        "target_height": out_h,
        "total_frames": meta.total_frames,
        "frames_written": frames_written,
        "num_windows": len(plan["windows"]),
        "num_windows_executed": windows_run,
        "num_shots": len(plan["shots"]),
        "num_segments": len(segments),
        "segments": plan["segments"],
        "num_references": len(plan["all_ref_indices"]),
        "peak_vram_gb": peak_vram_gb,
        "container_memory_mb": CONTAINER_MEMORY_MB,
        "encoder": results[0].get("encoder", "unknown") if results else "unknown",
        "timing": timing,
    }
