"""Authenticated HTTP job queue for FlashVSR-Pro.

Deploy with:
    modal deploy -m modal_app.deploy

Two ways in:

1. Upload/download over HTTP (`POST /upscale` -> `GET /result/{call_id}`). The
   upload is streamed to the io Volume, the GPU worker is `spawn`ed, and the
   caller polls for the finished file. Use `flashvsr_client.py`.
2. Volume-resident files (`POST /upscale_path`). For inputs you already pushed
   with `modal volume put flashvsr-io`; the output stays on the Volume and you
   fetch it with `modal volume get`. Preferred for very large media.

Why a queue and not a plain request/response: Modal enforces a 150-second
timeout on web requests. A `timeout=` on the GPU function does not extend that
window, so anything slower than ~2 minutes has to spawn and poll.

Every endpoint requires Modal proxy auth. Create a token pair with
`modal workspace proxy-tokens create` and send it as Modal-Key / Modal-Secret.
"""

import shutil
import uuid
from pathlib import Path
from typing import Optional

import modal

from .app import app, io_volume
from .config import (
    APP_NAME,
    IO_MOUNT_PATH,
    JOBS_DIR,
    JOBS_PREFIX,
    MODE_CLASSES,
    MODE_GPUS,
    UPLOAD_CHUNK,
    VIDEO_EXTENSIONS,
    validate_request,
)
from .types import InferenceRequest

# The web tier only validates parameters and moves bytes, so it stays off the
# GPU image entirely -- no torch, no CUDA kernels, no multi-minute cold start.
# It deliberately does not import `service.py`: that module pulls in `image.py`,
# whose `add_local_dir` build layers resolve paths that do not exist in the web
# container. Workers are looked up by name at request time instead.
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "fastapi[standard]==0.141.1",
        "python-multipart==0.0.32",
    )
    # Modal 1.x does not automount local packages; without this the container
    # cannot import `modal_app.config` / `modal_app.types` at request time.
    .add_local_python_source("modal_app")
)

_JOBS_PATH = Path(JOBS_DIR)


def _worker_for(mode: str):
    """Resolve the deployed GPU class that serves `mode`."""
    return modal.Cls.from_name(APP_NAME, MODE_CLASSES[mode])


async def _cleanup_job_dir(job_dir: Path) -> None:
    shutil.rmtree(job_dir, ignore_errors=True)
    await io_volume.commit.aio()


@app.function(
    image=web_image,
    timeout=900,
    volumes={IO_MOUNT_PATH: io_volume},
)
@modal.asgi_app(requires_proxy_auth=True)
@modal.concurrent(max_inputs=20)
def api():
    from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel

    web_app = FastAPI(
        title="FlashVSR-Pro",
        description="Submit a video, upscale it with FlashVSR on a Modal GPU, then download the output.",
        version="2.0.0",
    )

    class PathRequest(BaseModel):
        input_path: str
        output_path: str
        mode: str = "tiny"
        scale: float = 2.0
        seed: int = 0
        sparse_ratio: float = 2.0
        kv_ratio: float = 3.0
        local_range: int = 11
        color_fix: bool = False
        fps: Optional[float] = None
        quality: int = 10
        keep_audio: bool = False
        tile_vae: Optional[bool] = None
        tile_size: int = 256
        overlap: int = 24
        dtype: str = "bf16"

    @web_app.get("/health")
    def health() -> dict:
        return {"status": "ok", "modes": dict(MODE_GPUS)}

    @web_app.post("/upscale")
    async def upscale(
        file: UploadFile = File(...),
        mode: str = Form("tiny"),
        scale: float = Form(2.0),
        seed: int = Form(0),
        sparse_ratio: float = Form(2.0),
        kv_ratio: float = Form(3.0),
        local_range: int = Form(11),
        color_fix: bool = Form(False),
        fps: Optional[float] = Form(None),
        quality: int = Form(10),
        keep_audio: bool = Form(False),
        tile_vae: Optional[bool] = Form(None),
        tile_size: int = Form(256),
        overlap: int = Form(24),
        dtype: str = Form("bf16"),
    ):
        try:
            params = validate_request(
                mode=mode,
                scale=scale,
                seed=seed,
                sparse_ratio=sparse_ratio,
                kv_ratio=kv_ratio,
                local_range=local_range,
                color_fix=color_fix,
                fps=fps,
                quality=quality,
                keep_audio=keep_audio,
                tile_vae=tile_vae,
                tile_size=tile_size,
                overlap=overlap,
                dtype=dtype,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        input_name = file.filename or "upload.mp4"
        suffix = Path(input_name).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported input extension {suffix or '(none)'}; expected one of {sorted(VIDEO_EXTENSIONS)}.",
            )

        job_id = uuid.uuid4().hex
        job_dir = _JOBS_PATH / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        input_path = job_dir / f"input{suffix}"

        # Stream to the Volume so a multi-GB upload never lands in memory.
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

        req = InferenceRequest(
            input_path=f"{JOBS_PREFIX}/{job_id}/{input_path.name}",
            output_path=f"{JOBS_PREFIX}/{job_id}/output.mp4",
            job_id=job_id,
            **params,
        )
        call = await _worker_for(params["mode"])().infer.spawn.aio(req)
        return {"call_id": call.object_id, "job_id": job_id, "mode": params["mode"]}

    @web_app.post("/upscale_path")
    async def upscale_path(body: PathRequest):
        """Run against files already on the io Volume. Output stays on the Volume."""
        try:
            params = validate_request(**body.model_dump(exclude={"input_path", "output_path"}))
            for field in ("input_path", "output_path"):
                value = getattr(body, field)
                if value.startswith("/") or ".." in Path(value).parts:
                    raise ValueError(f"{field} must be a relative path inside the volume.")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        req = InferenceRequest(
            input_path=body.input_path,
            output_path=body.output_path,
            **params,
        )
        call = await _worker_for(params["mode"])().infer.spawn.aio(req)
        return {"call_id": call.object_id, "output_path": body.output_path, "mode": params["mode"]}

    @web_app.get("/result/{call_id}")
    async def result(call_id: str, background_tasks: BackgroundTasks):
        function_call = modal.FunctionCall.from_id(call_id)
        try:
            job = await function_call.get.aio(timeout=0)
        except modal.exception.OutputExpiredError:
            raise HTTPException(status_code=404, detail="Job result has expired.") from None
        except TimeoutError:
            return JSONResponse(content={"status": "pending"}, status_code=202)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if any(
                marker in msg
                for marker in (
                    "mode must be one of",
                    "dtype must be one of",
                    "tile_dit is not supported",
                    "requires tile_vae=True",
                    "must be between",
                    "must be smaller than",
                    "Input not found",
                )
            ):
                raise HTTPException(status_code=400, detail=msg) from exc
            raise HTTPException(status_code=500, detail=f"Upscaling failed: {exc}") from exc

        # Volume-path jobs leave their output on the Volume for `modal volume get`.
        if not job.get("job_id"):
            return job

        await io_volume.reload.aio()
        job_dir = _JOBS_PATH / job["job_id"]
        output_path = Path(IO_MOUNT_PATH) / job["output_path"]
        if not output_path.exists():
            raise HTTPException(status_code=410, detail="Output already downloaded or cleaned up.")

        background_tasks.add_task(_cleanup_job_dir, job_dir)
        return FileResponse(
            path=output_path.as_posix(),
            filename=f"{job['job_id']}_upscaled.mp4",
            media_type="video/mp4",
        )

    return web_app


__all__ = ["api", "web_image"]
