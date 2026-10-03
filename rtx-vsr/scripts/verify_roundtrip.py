"""Run the deployed worker on a job already staged on the Volume, warm, several times.

Bypasses the web tier (no proxy tokens needed) while exercising the real worker. Repeats
the run because CLAUDE.md section 6 only trusts warm numbers -- the effect load is paid
once per container, and a cold run reads several fps slower.

    python scripts/verify_roundtrip.py [job-id] [runs]
"""

import os
import sys

import modal


def main() -> None:
    job_id = sys.argv[1] if len(sys.argv) > 1 else "verify-gpu"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    # Same knob as modal_app.py, so a bench deploy (rtx-bench-<x>) can be verified too.
    app_name = os.getenv("MODAL_APP_NAME", "rtx-media-upscaler")
    worker = modal.Cls.from_name(app_name, "UpscaleWorker")()
    for attempt in range(1, runs + 1):
        result = worker.run.remote(
            job_id=job_id,
            input_name="input.mkv",
            mime_type="video/x-matroska",
            resize_type="scale by multiplier",
            scale=2.0,
            width=0,
            height=0,
            quality="HIGHBITRATE_ULTRA",
            keep_aspect_ratio=True,
        )
        print(f"run {attempt}: {result}")


if __name__ == "__main__":
    main()
