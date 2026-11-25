"""FlashVSR+ GPU worker: one planned segment per call, written to the io Volume.

One container takes one input at a time (no `@modal.concurrent`): the model
uses the whole GPU, and Modal scales out to `PLUS_MAX_CONTAINERS` containers
for a job's segments. More containers shorten the wall time. They do not
change the cost: the same segments need the same GPU-seconds either way.

The work itself is in plus_runtime.py (`render_segment`) and plus_job.py
(`render_seam`); this module adds the container, the model cache and the
Volume's `reload()` / `commit()`.

NOT YET RUN ON A GPU (owner decision, 2026-11-25: code only). The same
functions are tested on the CPU with a stub model in tests/test_plus_runtime.py.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from fractions import Fraction

import modal

from .config import (
    PLUS_GPU,
    PLUS_IO_MOUNT_PATH,
    PLUS_MAX_CONTAINERS,
    PLUS_MODEL_MOUNT_PATH,
    PLUS_REPO_SHA,
    PLUS_VIDEO_ENCODER,
    PLUS_WORKER_CPU,
    PLUS_WORKER_MEMORY_MIB,
)
from .plus_app import plus_app, plus_io_volume, plus_model_volume
from .plus_image import plus_image

# Parameters that decide which weights are on the GPU and how they were built.
# A call with different ones rebuilds the pipeline; the rest is applied per run.
_PIPELINE_KEYS = (
    "mode", "seed", "decode_every", "tiny_decoder_mode", "resident", "attn_kernel",
    "color_correction", "compile_dit",
)


@plus_app.cls(
    image=plus_image,
    gpu=PLUS_GPU,
    cpu=PLUS_WORKER_CPU,
    memory=PLUS_WORKER_MEMORY_MIB,
    volumes={PLUS_MODEL_MOUNT_PATH: plus_model_volume, PLUS_IO_MOUNT_PATH: plus_io_volume},
    # A 90 s segment at 0.4 frames per second takes about 95 minutes.
    timeout=4 * 60 * 60,
    # One retry covers a preempted or broken host. A bug fails twice and stops;
    # running the job again resumes it, so nothing more is gained by retrying here.
    retries=1,
    max_containers=PLUS_MAX_CONTAINERS,
    scaledown_window=120,
)
class PlusWorker:
    @modal.enter()
    def setup(self):
        import torch

        from .plus_video import ENCODER_START_ATTEMPTS, encode_command

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        self._pipeline = None
        self._pipeline_key = None

        self.gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        ).stdout.strip() or "unknown"
        print(f"GPU: {self.gpu}; video encoder: {PLUS_VIDEO_ENCODER}; upstream {PLUS_REPO_SHA[:12]}")

        # A real encode with the production arguments. An encoder listed by
        # `ffmpeg -encoders` can still fail to open a session (wrong driver for
        # the SDK the build links). Fail the container now, loudly, instead of
        # discovering it after an hour of inference. There is no CPU fallback:
        # a silent one hid a real defect in rtx-vsr until someone read the logs.
        # NVENC also refuses a session at random with an explicit level, so one
        # failed attempt proves nothing; all of them failing does.
        failure = ""
        for attempt in range(1, ENCODER_START_ATTEMPTS + 1):
            with tempfile.TemporaryDirectory() as scratch:
                probe = subprocess.run(
                    encode_command(256, 256, Fraction(25), os.path.join(scratch, "probe.mp4"),
                                   encoder=PLUS_VIDEO_ENCODER),
                    input=bytes(256 * 256 * 3 * 8), capture_output=True, check=False,
                )
            if probe.returncode == 0:
                break
            failure = probe.stderr.decode(errors="ignore").strip()[-600:]
            print(f"WARNING: encoder probe attempt {attempt} failed: {failure[-200:]}")
        else:
            raise RuntimeError(
                f"Video encoder {PLUS_VIDEO_ENCODER} cannot encode on this host ({self.gpu}): {failure}"
            )

    def _pipeline_for(self, params: dict):
        from .plus_runtime import build_pipeline

        key = tuple(params[name] for name in _PIPELINE_KEYS)
        if self._pipeline is None or self._pipeline_key != key:
            import torch

            # Drop the old preset first: two DiTs do not need to share the card.
            self._pipeline = None
            torch.cuda.empty_cache()
            started = time.monotonic()
            self._pipeline = build_pipeline(params)
            self._pipeline_key = key
            print(f"Loaded preset {params['mode']} in {time.monotonic() - started:.1f}s "
                  f"(attention kernel: {self._pipeline.config.attn_kernel})")
        return self._pipeline

    @modal.method()
    def upscale_segment(self, job_id: str, segment: dict, params: dict, digest: str) -> dict:
        """Render one planned segment; skips work that is already on the Volume."""
        from .plus_runtime import render_segment

        plus_io_volume.reload()
        report = render_segment(
            PLUS_IO_MOUNT_PATH, job_id, segment, params, digest,
            pipeline_for=self._pipeline_for, encoder=PLUS_VIDEO_ENCODER, host=self.gpu,
        )
        plus_io_volume.commit()
        return report

    @modal.method()
    def encode_seam(self, job_id: str, seam: dict, digest: str) -> dict:
        """Cross-fade and encode one soft seam.

        On the GPU class only because the seam must come from the same encoder
        as the segments around it.
        """
        from .plus_job import render_seam

        plus_io_volume.reload()
        report = render_seam(
            PLUS_IO_MOUNT_PATH, job_id, seam, digest, encoder=PLUS_VIDEO_ENCODER, host=self.gpu
        )
        plus_io_volume.commit()
        return report
