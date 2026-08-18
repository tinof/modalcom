import argparse
import mimetypes
import os
import re
import sys
import time
from pathlib import Path

import requests


def _extract_filename(content_disposition: str | None) -> str | None:
    if not content_disposition:
        return None

    # Supports: attachment; filename="example.png"
    match = re.search(r'filename="?([^";]+)"?', content_disposition)
    if match:
        return match.group(1)
    return None


def _default_output_path(input_path: Path, response: requests.Response) -> Path:
    header_name = _extract_filename(response.headers.get("content-disposition"))
    if header_name:
        return input_path.with_name(header_name)

    content_type = response.headers.get("content-type", "").lower()
    extension = ".mp4" if "video" in content_type else ".png"
    return input_path.with_name(f"{input_path.stem}_upscaled{extension}")


def _auth_headers() -> dict[str, str]:
    """Proxy-auth token pair, created with `modal workspace proxy-tokens create`."""
    key = os.getenv("MODAL_KEY")
    secret = os.getenv("MODAL_SECRET")
    if not key or not secret:
        sys.exit("Set MODAL_KEY and MODAL_SECRET to a Modal proxy auth token pair.")
    return {"Modal-Key": key, "Modal-Secret": secret}


def run() -> None:
    parser = argparse.ArgumentParser(description="Upload media to Modal RTX upscaler and save output locally.")
    parser.add_argument("--endpoint", required=True, help="Modal endpoint URL, e.g. https://<user>--rtx-media-upscaler-api.modal.run")
    parser.add_argument("--input", required=True, help="Path to local image or video file")
    parser.add_argument("--output", help="Optional output path. Defaults to *_upscaled.<ext>")
    parser.add_argument("--resize-type", default="scale by multiplier", choices=["scale by multiplier", "target dimensions"])
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--quality",
        default="HIGHBITRATE_ULTRA",
        choices=[
            "BICUBIC", "LOW", "MEDIUM", "HIGH", "ULTRA",
            "HIGHBITRATE_LOW", "HIGHBITRATE_MEDIUM", "HIGHBITRATE_HIGH", "HIGHBITRATE_ULTRA",
        ],
        help="Upscaling model. HIGHBITRATE_* skips artifact suppression for clean sources.",
    )
    parser.add_argument(
        "--preprocess",
        default="",
        choices=[
            "",
            "DENOISE_LOW", "DENOISE_MEDIUM", "DENOISE_HIGH", "DENOISE_ULTRA",
            "DEBLUR_LOW", "DEBLUR_MEDIUM", "DEBLUR_HIGH", "DEBLUR_ULTRA",
        ],
        help="Optional same-resolution restoration pass applied before upscaling.",
    )
    parser.add_argument(
        "--no-keep-aspect-ratio",
        action="store_false",
        dest="keep_aspect_ratio",
        default=True,
        help="Stretch to exact target dimensions instead of preserving aspect ratio inside the box",
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Max seconds to wait for the job to finish")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between result polls")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    endpoint = args.endpoint.rstrip("/")
    headers = _auth_headers()

    mime_type = mimetypes.guess_type(str(input_path))[0] or "application/octet-stream"
    data = {
        "resize_type": args.resize_type,
        "scale": str(args.scale),
        "width": str(args.width),
        "height": str(args.height),
        "keep_aspect_ratio": str(args.keep_aspect_ratio).lower(),
        "quality": args.quality,
        "preprocess": args.preprocess,
    }

    with input_path.open("rb") as f:
        files = {"file": (input_path.name, f, mime_type)}
        submit = requests.post(
            f"{endpoint}/upscale",
            data=data,
            files=files,
            headers=headers,
            timeout=args.timeout,
        )
    submit.raise_for_status()

    call_id = submit.json()["call_id"]
    print(f"Submitted job {call_id}; waiting for the GPU worker to finish...")

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
        time.sleep(args.poll_interval)

    response.raise_for_status()

    output_path = Path(args.output) if args.output else _default_output_path(input_path, response)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"Saved upscaled output to: {output_path}")


if __name__ == "__main__":
    run()
