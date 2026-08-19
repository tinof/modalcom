"""Submit a video to the deployed SparkVSR endpoint and save the restored result locally.

Usage:
    export MODAL_KEY=... MODAL_SECRET=...      # modal workspace proxy-tokens create
    python sparkvsr_client.py \
        --endpoint https://<workspace>--sparkvsr-api.modal.run \
        --input sample/jopet_10s.mkv \
        --target-height 2160 \
        --ref-mode pisasr
"""

import argparse
import mimetypes
import os
import sys
import time
from pathlib import Path

import requests


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_restored.mp4")


def _auth_headers() -> dict[str, str]:
    """Proxy-auth token pair, created with `modal workspace proxy-tokens create`."""
    key = os.getenv("MODAL_KEY")
    secret = os.getenv("MODAL_SECRET")
    if not key or not secret:
        sys.exit("Set MODAL_KEY and MODAL_SECRET environment variables to a Modal proxy auth token pair.")
    return {"Modal-Key": key, "Modal-Secret": secret}


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a video to the Modal SparkVSR endpoint and save the super-resolved output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--endpoint", required=True, help="e.g. https://<workspace>--sparkvsr-api.modal.run")
    parser.add_argument("--input", required=True, help="Path to a local video file")
    parser.add_argument("--output", help="Output path. Defaults to <name>_restored.mp4")
    parser.add_argument("--target-height", type=int, default=2160, help="Target vertical resolution (e.g. 2160 for 4K)")
    parser.add_argument("--target-width", type=int, default=None, help="Target horizontal resolution (auto if omitted)")
    parser.add_argument("--ref-mode", default="pisasr", choices=["pisasr", "api", "no_ref"], help="Reference generation mode")
    parser.add_argument("--ref-guidance", type=float, default=1.0, help="Reference guidance scale (1.0 = standard pass)")
    parser.add_argument("--no-cut-aware", action="store_true", help="Disable scene-cut detection and use fixed windows")
    parser.add_argument("--no-audio", action="store_true", help="Do not copy audio from the input file")
    parser.add_argument("--tile", action="store_true", help="Enable spatial tiling for extreme resolutions")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size when spatial tiling is enabled")
    parser.add_argument("--chunk-size", type=int, default=49, help="Frames per window (must be 8n+1)")
    parser.add_argument("--overlap", type=int, default=8, help="Overlap between windows in same shot")
    parser.add_argument("--timeout", type=int, default=3600, help="Max seconds to wait for the job")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between status polls")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    endpoint = args.endpoint.rstrip("/")
    headers = _auth_headers()
    mime_type = mimetypes.guess_type(str(input_path))[0] or "video/mp4"

    data = {
        "target_height": str(args.target_height),
        "ref_mode": args.ref_mode,
        "ref_guidance_scale": str(args.ref_guidance),
        "cut_aware": str(not args.no_cut_aware).lower(),
        "keep_audio": str(not args.no_audio).lower(),
        "tile": str(args.tile).lower(),
        "tile_size": str(args.tile_size),
        "chunk_size": str(args.chunk_size),
        "overlap": str(args.overlap),
    }
    if args.target_width is not None:
        data["target_width"] = str(args.target_width)

    print(f"Uploading {input_path.name} to {endpoint}...")
    with input_path.open("rb") as f:
        files = {"file": (input_path.name, f, mime_type)}
        submit = requests.post(
            f"{endpoint}/upscale",
            data=data,
            files=files,
            headers=headers,
            timeout=args.timeout,
        )

    if submit.status_code >= 400:
        sys.exit(f"Submission rejected ({submit.status_code}): {submit.text}")

    payload = submit.json()
    call_id = payload["call_id"]
    job_id = payload["job_id"]
    print(f"Job submitted successfully (job_id={job_id}, call_id={call_id}). Waiting for GPU processing...")

    deadline = time.monotonic() + args.timeout
    while True:
        response = requests.get(
            f"{endpoint}/result/{call_id}",
            headers=headers,
            timeout=120,
            stream=True,
        )
        if response.status_code != 202:
            break
        response.close()
        if time.monotonic() >= deadline:
            sys.exit(f"Timed out after {args.timeout}s waiting for job {call_id}.")
        print(".", end="", flush=True)
        time.sleep(args.poll_interval)

    print()
    if response.status_code >= 400:
        sys.exit(f"Job failed ({response.status_code}): {response.text}")

    output_path = Path(args.output) if args.output else _default_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"Successfully downloaded restored video to: {output_path}")


if __name__ == "__main__":
    run()
