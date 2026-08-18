"""Batch and long-video processing orchestrator for FlashVSR-Pro.

Provides:
- process_long_video(): Split → process chunks → merge
- batch_inference(): Fan out multiple videos via .map()
"""

from __future__ import annotations

import os
import subprocess
import uuid

import modal

from .app import app, io_volume
from .config import IO_MOUNT_PATH
from .service import FlashVSRFull, FlashVSRTiny, FlashVSRTinyLong, InferenceRequest

# The orchestrator only splits and concatenates with ffmpeg and then fans work
# out to the GPU classes, so it has no reason to pay the GPU image's cold start.
orchestrator_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .add_local_python_source("modal_app")
)


def _get_service_class(mode: str):
    """Get the appropriate service class for a mode."""
    if mode == "full":
        return FlashVSRFull
    elif mode == "tiny-long":
        return FlashVSRTinyLong
    else:
        return FlashVSRTiny


@app.function(
    image=orchestrator_image,
    volumes={IO_MOUNT_PATH: io_volume},
    timeout=3600,
)
def process_long_video(
    input_path: str,
    output_path: str,
    mode: str = "tiny",
    chunk_seconds: float = 10.0,
    **kwargs,
) -> dict:
    """Process a long video by splitting into chunks, processing each, then merging.

    Args:
        input_path: Input video path on io_volume (relative to IO_MOUNT_PATH).
        output_path: Output video path on io_volume (relative to IO_MOUNT_PATH).
        mode: Inference mode ("full", "tiny", "tiny-long").
        chunk_seconds: Duration of each chunk in seconds.
        **kwargs: Additional InferenceRequest parameters.

    Returns:
        dict with output_path and status.
    """
    io_volume.reload()

    input_abs = os.path.join(IO_MOUNT_PATH, input_path)
    output_abs = os.path.join(IO_MOUNT_PATH, output_path)
    job_id = uuid.uuid4().hex[:8]
    chunks_dir = os.path.join(IO_MOUNT_PATH, f"_chunks/{job_id}")
    processed_dir = os.path.join(IO_MOUNT_PATH, f"_processed/{job_id}")
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Split input video into chunks using ffmpeg
    chunk_pattern = os.path.join(chunks_dir, "chunk_%04d.mp4")
    split_cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", input_abs,
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-reset_timestamps", "1",
        "-c", "copy",
        chunk_pattern,
    ]
    subprocess.run(split_cmd, check=True)
    io_volume.commit()

    # List chunks
    chunk_files = sorted(
        f for f in os.listdir(chunks_dir) if f.endswith(".mp4")
    )
    print(f"Split into {len(chunk_files)} chunks")

    if not chunk_files:
        return {"output_path": output_path, "status": "error", "message": "No chunks created"}

    # Process each chunk via the appropriate service class
    ServiceClass = _get_service_class(mode)
    service = ServiceClass()

    # Build requests for each chunk
    requests = []
    for chunk_file in chunk_files:
        chunk_input = f"_chunks/{job_id}/{chunk_file}"
        chunk_output = f"_processed/{job_id}/{chunk_file}"
        req = InferenceRequest(
            input_path=chunk_input,
            output_path=chunk_output,
            mode=mode,
            keep_audio=kwargs.get("keep_audio", False),
            **{k: v for k, v in kwargs.items() if k != "keep_audio"},
        )
        requests.append(req)

    # Process sequentially (chunks must maintain order)
    for req in requests:
        result = service.infer.remote(req)
        print(f"Processed: {req.input_path} -> {result}")

    # Merge processed chunks using ffmpeg concat
    io_volume.reload()
    concat_list_path = os.path.join(processed_dir, "concat_list.txt")
    with open(concat_list_path, "w") as f:
        for chunk_file in chunk_files:
            chunk_abs = os.path.join(processed_dir, chunk_file)
            f.write(f"file '{chunk_abs}'\n")

    os.makedirs(os.path.dirname(output_abs), exist_ok=True)
    merge_cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        output_abs,
    ]
    subprocess.run(merge_cmd, check=True)

    # Cleanup temp directories
    import shutil
    shutil.rmtree(chunks_dir, ignore_errors=True)
    shutil.rmtree(processed_dir, ignore_errors=True)

    io_volume.commit()
    print(f"Merged output: {output_abs}")
    return {"output_path": output_path, "status": "success"}


@app.function(
    image=orchestrator_image,
    volumes={IO_MOUNT_PATH: io_volume},
    timeout=3600,
)
def batch_inference(
    requests: list[dict],
    mode: str = "tiny",
) -> list[dict]:
    """Process multiple videos in parallel using .map().

    Args:
        requests: List of dicts, each with at least "input_path" and "output_path".
        mode: Inference mode applied to all requests.

    Returns:
        List of result dicts.
    """
    ServiceClass = _get_service_class(mode)
    service = ServiceClass()

    inference_requests = [
        InferenceRequest(mode=mode, **req_dict)
        for req_dict in requests
    ]

    results = list(service.infer.map(inference_requests))
    return results
