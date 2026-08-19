"""Authenticated HTTP Web API and job queue for SparkVSR."""

import contextlib
import shutil
import uuid
from pathlib import Path
from typing import Optional

import modal

from .app import app, io_volume
from .config import (
    APP_NAME,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_REF_GUIDANCE_SCALE,
    DEFAULT_REF_MODE,
    DEFAULT_TARGET_HEIGHT,
    IO_MOUNT_PATH,
    JOBS_DIR,
    PARALLEL,
    UPLOAD_CHUNK,
    VIDEO_EXTENSIONS,
    validate_request,
)
from .image import web_image

_JOBS_PATH = Path(JOBS_DIR)


async def _cleanup_job_dir(job_dir: Path) -> None:
    """Reclaim the bulky job files once the result has been served.

    The cut plan and the generated reference frames are deliberately kept: the
    plan is what makes cut detection auditable, and the references are what you
    inspect when you want to know whether reference quality capped the result.
    Both are small; the input and output videos are not.
    """
    for name in ("output.mp4",):
        with contextlib.suppress(OSError):
            (job_dir / name).unlink()
    for stale_input in job_dir.glob("input.*"):
        with contextlib.suppress(OSError):
            stale_input.unlink()
    await io_volume.commit.aio()


@app.function(
    image=web_image,
    timeout=900,
    volumes={IO_MOUNT_PATH: io_volume},
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app(requires_proxy_auth=True)
def api():
    from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse

    web_app = FastAPI(
        title="SparkVSR Super-Resolution API",
        description="Scene-cut aware video super-resolution powered by SparkVSR on Modal RTX PRO 6000 GPUs.",
        version="1.0.0",
    )

    @web_app.get("/health")
    def health() -> dict:
        return {"status": "ok", "app": APP_NAME}

    @web_app.post("/upscale")
    async def upscale(
        file: UploadFile = File(...),
        target_height: Optional[int] = Form(DEFAULT_TARGET_HEIGHT),
        target_width: Optional[int] = Form(None),
        ref_mode: str = Form(DEFAULT_REF_MODE),
        ref_guidance_scale: float = Form(DEFAULT_REF_GUIDANCE_SCALE),
        cut_aware: bool = Form(True),
        chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
        overlap: int = Form(DEFAULT_OVERLAP),
        keep_audio: bool = Form(True),
        tile: bool = Form(False),
        tile_size: Optional[int] = Form(None),
    ):
        try:
            params = validate_request(
                target_height=target_height,
                target_width=target_width,
                ref_mode=ref_mode,
                ref_guidance_scale=ref_guidance_scale,
                cut_aware=cut_aware,
                chunk_size=chunk_size,
                overlap=overlap,
                keep_audio=keep_audio,
                tile=tile,
                tile_size=tile_size,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Reject api mode without credentials here, before any GPU container is
        # started. The web tier does not mount the secret, so probe for its
        # existence rather than reading FAL_KEY.
        if params["ref_mode"] == "api":
            try:
                await modal.Secret.from_name("fal").hydrate.aio()
            except modal.exception.NotFoundError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "ref_mode='api' requires a Modal Secret named 'fal' holding FAL_KEY. "
                        "Create it with: modal secret create fal FAL_KEY=<your-key>"
                    ),
                ) from exc

        input_name = file.filename or "upload.mp4"
        suffix = Path(input_name).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported video extension {suffix or '(none)'}; expected one of {sorted(VIDEO_EXTENSIONS)}.",
            )

        job_id = uuid.uuid4().hex
        job_dir = _JOBS_PATH / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        input_filename = f"input{suffix}"
        input_path = job_dir / input_filename

        # Stream upload directly to the Volume
        bytes_written = 0
        with input_path.open("wb") as dest:
            while chunk := await file.read(UPLOAD_CHUNK):
                dest.write(chunk)
                bytes_written += len(chunk)

        if bytes_written == 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            await io_volume.commit.aio()
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        await io_volume.commit.aio()

        # Spawn the renderer. The parallel driver fans segments across GPU workers and
        # returns the same result shape as the single-container path, so the polling
        # endpoint below is unchanged either way.
        if PARALLEL:
            driver = modal.Function.from_name(APP_NAME, "process_video_parallel")
        else:
            driver = modal.Cls.from_name(APP_NAME, "SparkVSRService")().process_video

        call = await driver.spawn.aio(
            job_id=job_id,
            input_filename=input_filename,
            output_filename="output.mp4",
            **params,
        )

        return {
            "call_id": call.object_id,
            "job_id": job_id,
            "ref_mode": params["ref_mode"],
            "target_height": params["target_height"],
        }

    @web_app.get("/result/{call_id}")
    async def result(call_id: str, background_tasks: BackgroundTasks):
        function_call = modal.FunctionCall.from_id(call_id)
        try:
            job_res = await function_call.get.aio(timeout=0)
        except modal.exception.OutputExpiredError:
            raise HTTPException(status_code=404, detail="Job result has expired.") from None
        except TimeoutError:
            return JSONResponse(content={"status": "pending"}, status_code=202)
        except Exception as exc:
            msg = str(exc)
            raise HTTPException(status_code=500, detail=f"Super-resolution failed: {msg}") from exc

        job_id = job_res.get("job_id")
        if not job_id:
            return job_res

        await io_volume.reload.aio()
        job_dir = _JOBS_PATH / job_id
        output_path = job_dir / "output.mp4"

        if not output_path.exists():
            raise HTTPException(status_code=410, detail="Output file missing or already cleaned up.")

        background_tasks.add_task(_cleanup_job_dir, job_dir)
        return FileResponse(
            path=output_path.as_posix(),
            filename=f"{job_id}_restored.mp4",
            media_type="video/mp4",
        )

    return web_app


__all__ = ["api", "web_image"]
