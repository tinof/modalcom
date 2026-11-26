"""CPU orchestrator of a FlashVSR+ job: plan, fan out to the GPU workers, join.

Started with `.spawn()` on the deployed app (see client.py), so a job survives a
dropped session. It deliberately does not import `plus_service`: that module
defines the GPU image, whose local-directory layer does not exist in this
container. The worker class is looked up by name instead.

The steps themselves are in plus_job.py, which has no Modal in it.
"""

from __future__ import annotations

import time

import modal

from .config import (
    PLUS_APP_NAME,
    PLUS_ENV,
    PLUS_FFMPEG_INSTALL,
    PLUS_IO_MOUNT_PATH,
    PLUS_NUMPY,
)
from .plus_app import plus_app, plus_io_volume

# The orchestrator needs the same ffmpeg as the workers (same demuxer and
# bitstream filters), numpy for plus_video, and nothing else.
plus_cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "xz-utils")
    .run_commands(*PLUS_FFMPEG_INSTALL)
    .uv_pip_install(PLUS_NUMPY)
    .env(PLUS_ENV)
    .add_local_python_source("modal_app")
)


@plus_app.function(
    image=plus_cpu_image,
    volumes={PLUS_IO_MOUNT_PATH: plus_io_volume},
    timeout=24 * 60 * 60,
    cpu=8,
    memory=8192,
)
def run_job(
    job_id: str,
    input_path: str,
    params: dict | None = None,
    planning: dict | None = None,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> dict:
    """Upscale `input_path` (relative to the io Volume) into jobs/<job_id>/output.mkv.

    An excerpt (`start_seconds`, `duration_seconds`) is joined video-only: only
    a full-length result lines up with the source's audio and subtitles.
    Running the same job id again resumes it.
    """
    from .plus_job import finish_job, prepare_job, raise_failures
    from .plus_plan import Plan

    started = time.monotonic()
    plus_io_volume.reload()
    job = prepare_job(
        PLUS_IO_MOUNT_PATH, job_id, input_path, params, planning, start_seconds, duration_seconds
    )
    plus_io_volume.commit()

    plan = Plan.from_json(job["plan"])
    digest = job["digest"]
    worker = modal.Cls.from_name(PLUS_APP_NAME, "PlusWorker")()
    count = len(plan.segments)
    # return_exceptions: let every segment finish before failing the job, so a
    # second run has as little left to do as possible.
    segment_reports = list(worker.upscale_segment.map(
        [job_id] * count,
        job["plan"]["segments"],
        [job["params"]] * count,
        [digest] * count,
        return_exceptions=True,
        wrap_returned_exceptions=False,
    ))
    raise_failures("segments", [segment.name() for segment in plan.segments], segment_reports)
    seam_reports = list(worker.encode_seam.map(
        [job_id] * len(plan.seams),
        job["plan"]["seams"],
        [digest] * len(plan.seams),
        return_exceptions=True,
        wrap_returned_exceptions=False,
    ))
    raise_failures("seams", [seam.name() for seam in plan.seams], seam_reports)

    plus_io_volume.reload()
    report = finish_job(PLUS_IO_MOUNT_PATH, job_id, segment_reports, seam_reports, started)
    plus_io_volume.commit()
    return report
