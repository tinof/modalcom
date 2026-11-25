"""Run the deployed worker on a job already staged on the Volume, warm, several times.

Bypasses the web tier (no proxy tokens needed) while exercising the real worker. Repeats
the run because CLAUDE.md section 6 only trusts warm numbers -- the effect load is paid
once per container, and a cold run reads several fps slower.

    python scripts/verify_roundtrip.py [job-id] [runs] [options]

The worker deletes its input after each job, so with runs > 1 stage the same file as
<job-id>, <job-id>-2, <job-id>-3, ... (run N uses the N-th name). The call id is printed
before waiting, so a dropped client can find the job again.
"""

import argparse
import os

import modal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job_id", nargs="?", default="verify-gpu")
    parser.add_argument("runs", nargs="?", type=int, default=3)
    parser.add_argument("--input-name", default="input.mkv",
                        help="Original file name: its suffix selects /<job-id>/input<suffix>, "
                             "its stem names the output")
    parser.add_argument("--mime-type", default="",
                        help="Default: video/x-matroska, or image/png for an image suffix")
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--quality", default="HIGHBITRATE_ULTRA")
    parser.add_argument("--preprocess", default="")
    parser.add_argument("--truehdr", nargs="?", const="on", default="")
    parser.add_argument("--max-cll", default="")
    parser.add_argument("--container", default="mp4")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--preprocess-strength", type=float, default=1.0)
    args = parser.parse_args()

    suffix = os.path.splitext(args.input_name)[1].lower()
    is_image = suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    mime_type = args.mime_type or ("image/png" if is_image else "video/x-matroska")

    # Same knob as modal_app.py, so a bench deploy (rtx-bench-<x>) can be verified too.
    app_name = os.getenv("MODAL_APP_NAME", "rtx-media-upscaler")
    worker = modal.Cls.from_name(app_name, "UpscaleWorker")()
    for attempt in range(1, args.runs + 1):
        job_id = args.job_id if attempt == 1 else f"{args.job_id}-{attempt}"
        call = worker.run.spawn(
            job_id=job_id,
            input_name=args.input_name,
            mime_type=mime_type,
            resize_type="scale by multiplier",
            scale=args.scale,
            width=0,
            height=0,
            quality=args.quality,
            keep_aspect_ratio=True,
            preprocess=args.preprocess or None,
            truehdr=args.truehdr,
            max_cll=args.max_cll,
            container=args.container,
            strength=args.strength,
            preprocess_strength=args.preprocess_strength,
        )
        print(f"run {attempt}: call {call.object_id}", flush=True)
        try:
            print(f"run {attempt}: {call.get()}", flush=True)
        except Exception as exc:  # noqa: BLE001 - report the worker's error and go on
            print(f"run {attempt}: FAILED {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
