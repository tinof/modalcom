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


def _fail(response: requests.Response, context: str) -> None:
    """Exit with the server's own error detail instead of a bare HTTP status line."""
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    hints = {404: " (the result has expired)", 410: " (the output was already downloaded)"}
    hint = hints.get(response.status_code, "") if context == "Job" else ""
    sys.exit(f"{context} failed with HTTP {response.status_code}{hint}: {detail}")


# Answers from the proxy while the web container is cold or restarting, not job outcomes.
TRANSIENT_STATUSES = {502, 503, 504}


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
    parser.add_argument("--input", help="Path to local image or video file")
    parser.add_argument(
        "--call-id",
        help="Resume waiting for an already submitted job instead of uploading again",
    )
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
    parser.add_argument("--timeout", type=int, default=3 * 3600, help="Max seconds to wait for the job to finish")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between result polls")
    args = parser.parse_args()

    if not args.input and not args.call_id:
        parser.error("--input is required unless --call-id resumes an existing job")
    input_path = Path(args.input) if args.input else Path.cwd() / "output"
    if args.input and not args.call_id and not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    endpoint = args.endpoint.rstrip("/")
    headers = _auth_headers()

    if args.call_id:
        call_id = args.call_id
    else:
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
        if submit.status_code >= 400:
            _fail(submit, "Upload")

        call_id = submit.json()["call_id"]
        print(f"Submitted job {call_id}; waiting for the GPU worker to finish...")
        print(f"(If this client stops, resume with --call-id {call_id})")

    deadline = time.monotonic() + args.timeout
    delay = args.poll_interval
    while True:
        try:
            response = requests.get(
                f"{endpoint}/result/{call_id}",
                headers=headers,
                timeout=120,
                stream=True,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            print(f"Poll failed ({exc.__class__.__name__}); retrying.")
            response = None
        if response is not None and response.status_code not in {202, *TRANSIENT_STATUSES}:
            break
        if response is not None:
            transient = response.status_code in TRANSIENT_STATUSES
            response.close()
        else:
            transient = True
        if time.monotonic() >= deadline:
            sys.exit(
                f"Timed out after {args.timeout}s waiting for job {call_id}. "
                f"Resume with --call-id {call_id}."
            )
        # Back off on errors only; a plain 202 keeps the configured cadence.
        delay = min(delay * 2, 60.0) if transient else args.poll_interval
        time.sleep(delay)

    if response.status_code >= 400:
        _fail(response, "Job")

    output_path = Path(args.output) if args.output else _default_output_path(input_path, response)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The server deletes the output once it has been served, so write to a side file and
    # only rename after the byte count matches. A truncated file never looks complete.
    part_path = output_path.with_name(output_path.name + ".part")
    written = 0
    with part_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                written += len(chunk)

    # Content-Length counts encoded bytes, so it only matches an unencoded body.
    expected = response.headers.get("content-length")
    encoded = response.headers.get("content-encoding", "identity") != "identity"
    if expected is not None and not encoded and int(expected) != written:
        sys.exit(
            f"Download truncated: got {written} of {expected} bytes, kept in {part_path}. "
            "The server has already cleaned up this job, so it must be submitted again."
        )
    part_path.replace(output_path)

    print(f"Saved upscaled output to: {output_path}")

if __name__ == "__main__":
    run()
