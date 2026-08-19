"""Smoke test: upscale sample/jopet_10s.mkv using SparkVSR on Modal.

Usage:
    modal run run_sample.py
"""

import uuid
from pathlib import Path

from modal_app.app import app, io_volume
from modal_app.config import (
    DEFAULT_REF_MODE,
    DEFAULT_TARGET_HEIGHT,
    PARALLEL,
    PARALLEL_SEGMENT_TARGET_FRAMES,
)
from modal_app.service import SparkVSRService, process_video_parallel


@app.local_entrypoint()
def main(
    input_file: str = "sample/jopet_10s.mkv",
    target_height: int = DEFAULT_TARGET_HEIGHT,
    ref_mode: str = DEFAULT_REF_MODE,
    cut_aware: bool = True,
    tile: bool = False,
    output_file: str = "sample/jopet_10s_restored.mp4",
    parallel: bool = PARALLEL,
    segment_frames: int = PARALLEL_SEGMENT_TARGET_FRAMES,
):
    input_path = Path(input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Sample input not found at {input_path}")

    job_id = f"smoke_{uuid.uuid4().hex[:8]}"
    print(f"Starting SparkVSR smoke test ({job_id}) on {input_file}...")

    # Upload video directly to the IO volume
    print(f"Pushing {input_path.name} ({input_path.stat().st_size / (1024*1024):.1f} MB) to volume...")
    with io_volume.batch_upload() as batch:
        batch.put_file(input_path.as_posix(), f"jobs/{job_id}/input.mkv")

    if parallel:
        print(f"Submitting job to the parallel driver (segments of ~{segment_frames} frames)...")
        result = process_video_parallel.remote(
            job_id=job_id,
            input_filename="input.mkv",
            output_filename="output.mp4",
            target_height=target_height,
            ref_mode=ref_mode,
            cut_aware=cut_aware,
            tile_size=512 if tile else None,
            segment_frames=segment_frames,
        )
    else:
        print("Submitting job to SparkVSRService (single container)...")
        result = SparkVSRService().process_video.remote(
            job_id=job_id,
            input_filename="input.mkv",
            output_filename="output.mp4",
            target_height=target_height,
            ref_mode=ref_mode,
            cut_aware=cut_aware,
            tile_size=512 if tile else None,
        )

    print(f"\nProcessing finished with status: {result['status']}")
    print(f"  Target resolution: {result['target_width']}x{result['target_height']}")
    print(f"  Processed {result['total_frames']} frames in {result['num_windows']} windows across {result['num_shots']} shots")
    print(f"  Peak VRAM allocated: {result['peak_vram_gb']:.2f} GB")
    print(f"  Encoder used: {result['encoder']}")
    print("  Stage timings:")
    for stage, sec in result["timing"].items():
        print(f"    - {stage}: {sec:.2f}s")

    # Download output file from volume
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nRetrieving result to {out_path}...")
    with out_path.open("wb") as f_out:
        for chunk in io_volume.read_file(f"jobs/{job_id}/output.mp4"):
            f_out.write(chunk)

    print(f"Smoke test complete! Restored video saved to: {out_path} ({out_path.stat().st_size / (1024*1024):.1f} MB)")
