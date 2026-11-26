"""plus_runtime.py against upstream's real stream runner, on the CPU, without weights.

The DiT and the TinyDecoder are replaced by stubs that pass the conditioning
frames straight through, so the model becomes the identity. Everything else is
upstream's own code at the pinned commit: `StreamRunner.run`, `TemporalVideoStream`,
the `TinyDecoder` adapter and its frame slicing. If our frame source, our writer
or the resident transport miscounted or reordered a single frame, the frames
coming out would not be the frames going in.

Needs torch and a checkout of upstream at `PLUS_REPO_SHA`:

    git clone https://github.com/OpenImagingLab/FlashVSR_Plus.git /some/dir
    git -C /some/dir checkout <PLUS_REPO_SHA>
    FLASHVSR_PLUS_SRC=/some/dir uv run --python 3.13 --with torch --with numpy \
        --with pytest --with opencv-python-headless --with imageio --with imageio-ffmpeg \
        --with einops --with safetensors --with pillow --with tqdm pytest tests/test_plus_runtime.py

Skipped otherwise. It does not prove anything about the real model, the GPU
kernels, VRAM or speed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from modal_app.config import PLUS_REPO_SHA
from modal_app.plus_plan import plan_canvas

torch = pytest.importorskip("torch")

_SRC = os.environ.get("FLASHVSR_PLUS_SRC", "")
if not _SRC or not os.path.isdir(os.path.join(_SRC, "inference")):
    pytest.skip("set FLASHVSR_PLUS_SRC to an upstream checkout", allow_module_level=True)
sys.path.insert(0, _SRC)

from inference.config import DecoderConfig, InferenceConfig  # noqa: E402
from inference.decoders import TinyDecoder  # noqa: E402
from inference.loading import ModelBundle  # noqa: E402
from inference.loading.model_bundle import FlowMatchInferenceScheduler  # noqa: E402
from inference.runners import StreamRunner  # noqa: E402

from modal_app import plus_runtime  # noqa: E402

SRC_W, SRC_H = 60, 40  # 2x -> 120x80, padded to a 128x128 canvas (pads 4 and 24)


class _Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._stream_cache1 = None
        self._stream_cache2 = None
        self.frames_seen = 0

    def clear_cache(self):
        self._stream_cache1 = self._stream_cache2 = None

    def stream_forward(self, clip):
        self.frames_seen += clip.shape[2]
        return [torch.zeros(1, 1, 4)]


class _Denoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.LQ_proj_in = _Projection()
        self.moves: list[str] = []

    def to(self, *args, **kwargs):
        self.moves.append(str(args[0]) if args else str(kwargs.get("device")))
        return super().to(*args, **kwargs)

    def forward_stream(self, x, timestep, context, *, layer_conditions, state, context_mask=None):
        return torch.zeros_like(x)


class _DecoderModule(torch.nn.Module):
    """Returns its conditioning frames: decode_video's [N,T,3,H,W] in [0, 1]."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.mem = [None] * 8
        self.calls: list[tuple[int, int]] = []

    def clean_mem(self):
        self.mem = [None] * 8

    def decode_video(self, x, parallel=True, show_progress_bar=False, cond=None):
        latents, frames = x.shape[1], cond.shape[2]
        # The real decoder trims three frames on its very first call only.
        expected = 4 * latents - (3 if self.mem[-8] is None else 0)
        assert frames == expected, f"decoder got {frames} frames for {latents} latents"
        self.mem[-8] = torch.zeros(1)
        self.calls.append((latents, frames))
        return ((cond + 1.0) / 2.0).transpose(1, 2)


def _pipeline(decode_every: int):
    config = InferenceConfig(
        is_causal=True, is_sparse=True, attn_kernel="sdpa", block_size=2,
        decoder=DecoderConfig(kind="tiny", decode_every=decode_every,
                              tiny_decoder_parallel_mode="serial"),
        device="cpu", dtype="fp32", stream_cache_offload_mode="never",
    )
    config.validate()
    bundle = ModelBundle(
        _Denoiser(),
        TinyDecoder(_DecoderModule(), parallel_mode="serial"),
        FlowMatchInferenceScheduler(shift=config.sigma_shift),
        context=torch.zeros(4, 8),
    ).eval()
    return SimpleNamespace(runner=StreamRunner(config, bundle), config=config, bundle=bundle)


def _frames(count: int) -> np.ndarray:
    """Flat frames whose grey level is their index: any reorder or repeat shows."""
    levels = (np.arange(count) % 250 + 3).astype(np.uint8)
    return np.broadcast_to(levels[:, None, None, None], (count, SRC_H, SRC_W, 3)).copy()


def _chunks(frames: np.ndarray, size: int = 16):
    for start in range(0, len(frames), size):
        # Read-only, like the arrays SegmentReader builds from a bytes object.
        chunk = frames[start:start + size].copy()
        chunk.setflags(write=False)
        yield chunk


def _run(count: int, *, resident: bool, decode_every: int = 28, limit: int | None = None):
    pipeline = _pipeline(decode_every)
    canvas = plan_canvas(SRC_W, SRC_H, 2.0)
    source = _frames(count)
    received: list[np.ndarray] = []
    writer = plus_runtime.FrameWriter(received.append, canvas, limit=count if limit is None else limit)
    metadata = plus_runtime.run_stream(pipeline, _chunks(source), canvas, writer, resident=resident)
    output = np.concatenate(received) if received else np.zeros((0, canvas.out_h, canvas.out_w, 3), np.uint8)
    return pipeline, source, output, metadata, writer


def test_the_checkout_is_the_pinned_commit():
    head = subprocess.run(["git", "-C", _SRC, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head == PLUS_REPO_SHA


@pytest.mark.parametrize("resident", [False, True], ids=["stock", "resident"])
@pytest.mark.parametrize("count", [1, 2, 5, 16, 17, 18, 33, 112, 113, 114, 129, 250])
def test_every_frame_comes_back_once_and_in_order(count, resident):
    _, source, output, metadata, writer = _run(count, resident=resident)
    assert writer.frames == count
    assert output.shape == (count, 80, 120, 3)
    # Flat frames survive bicubic, reflect padding and the crop exactly.
    assert [int(frame[0, 0, 0]) for frame in output] == [int(frame[0, 0, 0]) for frame in source]
    assert all((frame == frame[0, 0, 0]).all() for frame in output)
    assert metadata["input_streaming"] is True


@pytest.mark.parametrize("decode_every", [4, 8, 28])
def test_decode_interval_does_not_change_the_result(decode_every):
    _, source, output, _, _ = _run(250, resident=True, decode_every=decode_every)
    assert [int(frame[0, 0, 0]) for frame in output] == [int(frame[0, 0, 0]) for frame in source]


def test_stock_and_resident_transport_agree():
    _, _, stock, _, _ = _run(250, resident=False)
    _, _, resident, _, _ = _run(250, resident=True)
    assert np.array_equal(stock, resident)


def test_resident_transport_keeps_the_dit_in_place_and_stock_moves_it():
    stock, *_ = _run(250, resident=False)
    assert "cpu" in stock.bundle.denoiser.moves, "upstream offloads the DiT around each decode"
    resident, *_ = _run(250, resident=True)
    assert resident.bundle.denoiser.moves == []


def test_resident_transport_is_undone_after_the_run():
    import inference.runners.stream as stream_module

    offload, restore = stream_module._offload_dit, stream_module._restore_dit
    pipeline = _pipeline(28)
    decoder = pipeline.bundle.decoder
    with plus_runtime.resident_transport(pipeline):
        assert stream_module._offload_dit is not offload
        assert pipeline.bundle.decoder is not decoder
        assert isinstance(pipeline.bundle.decoder, TinyDecoder)
    assert (stream_module._offload_dit, stream_module._restore_dit) == (offload, restore)
    assert pipeline.bundle.decoder is decoder


def test_resident_transport_is_undone_when_the_run_fails():
    import inference.runners.stream as stream_module

    offload = stream_module._offload_dit
    pipeline = _pipeline(28)
    canvas = plan_canvas(SRC_W, SRC_H, 2.0)

    def broken():
        yield _frames(16)
        raise RuntimeError("decoder died")

    writer = plus_runtime.FrameWriter(lambda frames: None, canvas, limit=100)
    with pytest.raises(RuntimeError, match="decoder died"):
        plus_runtime.run_stream(pipeline, broken(), canvas, writer, resident=True)
    assert stream_module._offload_dit is offload


def test_writer_limit_drops_the_models_end_padding():
    _, _, output, _, writer = _run(40, resident=True, limit=30)
    assert writer.frames == 30 and len(output) == 30


def test_prepare_and_convert_round_trip_a_real_picture():
    canvas = plan_canvas(120, 80, 1.0, multiple=128)
    generator = np.random.default_rng(0)
    frames = generator.integers(0, 256, size=(3, 80, 120, 3), dtype=np.uint8)
    tensor = plus_runtime.prepare_frames(frames, canvas, "cpu", torch.float32)
    assert tuple(tensor.shape) == (1, 3, 3, 128, 128)
    assert float(tensor.min()) >= -1.0 and float(tensor.max()) <= 1.0
    back = np.concatenate(list(plus_runtime.tensor_to_frames(tensor, canvas)))
    # Scale 1 is no resampling: pad, normalise, crop and round must be lossless.
    assert np.array_equal(back, frames)


def test_prepare_frames_pads_by_reflection():
    canvas = plan_canvas(SRC_W, SRC_H, 2.0)
    assert (canvas.pad_left, canvas.pad_top) == (4, 24)
    frames = _frames(2)
    tensor = plus_runtime.prepare_frames(frames, canvas, "cpu", torch.float32)
    assert tuple(tensor.shape) == (1, 3, 2, 128, 128)
    # A flat picture stays flat across the reflected border.
    expected = float(frames[1, 0, 0, 0]) / 127.5 - 1.0
    assert torch.allclose(tensor[0, :, 1], torch.full((3, 128, 128), expected), atol=1e-5)


def test_bypass_writes_the_bicubic_upscale():
    canvas = plan_canvas(SRC_W, SRC_H, 2.0)
    received: list[np.ndarray] = []
    writer = plus_runtime.FrameWriter(received.append, canvas, limit=20)
    metadata = plus_runtime.run_bypass(_chunks(_frames(20)), canvas, writer)
    output = np.concatenate(received)
    assert metadata == {"runner": "bypass"}
    assert output.shape == (20, 80, 120, 3)
    assert [int(frame[0, 0, 0]) for frame in output] == [int(frame[0, 0, 0]) for frame in _frames(20)]


def test_float32_rope_matches_upstreams_float64_version():
    import models.attention as attention

    stock = attention.apply_rope_3d
    shape = (2, 4, 6)
    x = torch.randn(1, 3, shape[0] * shape[1] * shape[2], 24, dtype=torch.bfloat16)
    reference = stock(x, shape, temporal_offset=5).float()
    with plus_runtime.float32_rope():
        assert attention.apply_rope_3d is not stock
        ours = attention.apply_rope_3d(x, shape, temporal_offset=5).float()
    assert attention.apply_rope_3d is stock
    assert ours.shape == reference.shape
    # Both round to bf16 at the end; they may differ by one bf16 step.
    assert torch.allclose(ours, reference, rtol=2e-2, atol=2e-2)
    assert (ours - reference).abs().mean() < 1e-3


# --- a whole job: plan, render, seam, join, resume -----------------------------------

from synthetic import FRAMES, decode_all, have_ffmpeg, make_source, read_index  # noqa: E402

from modal_app import plus_job  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(not have_ffmpeg(), reason="needs ffmpeg with libx264 and libx265")
PLANNING = {"segment_seconds": 3, "search_seconds": 1, "warmup_frames": 16}


@pytest.fixture(scope="module")
def job_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("volume")
    (root / "inputs").mkdir()
    os.replace(make_source(tmp_path_factory.mktemp("source")), root / "inputs" / "source.mkv")
    return str(root)


def _render_all(root: str, job: dict, pipeline_for) -> tuple[list[dict], list[dict]]:
    segments = [
        plus_runtime.render_segment(
            root, job["job_id"], segment, job["params"], job["digest"],
            pipeline_for=pipeline_for, encoder="libx265", host="cpu-test",
        )
        for segment in job["plan"]["segments"]
    ]
    seams = [
        plus_job.render_seam(root, job["job_id"], seam, job["digest"], encoder="libx265", host="cpu-test")
        for seam in job["plan"]["seams"]
    ]
    return segments, seams


@needs_ffmpeg
def test_whole_job_with_the_identity_model(job_root):
    import time

    job = plus_job.prepare_job(job_root, "full", "inputs/source.mkv", {"mode": "turbo"}, PLANNING)
    assert len(job["plan"]["segments"]) == 3 and len(job["plan"]["seams"]) == 1
    assert (job["canvas"]["canvas_w"], job["canvas"]["canvas_h"]) == (640, 384)
    assert os.path.exists(os.path.join(job_root, "inputs", "source.mkv.scenes.json"))

    pipeline = _pipeline(28)
    segments, seams = _render_all(job_root, job, lambda params: pipeline)
    assert [report["skipped"] for report in segments] == [False, False, False]
    assert [report["model_frames"] for report in segments] == [100, 83, 91]
    assert all(report["runner"]["runner"] == "stream" for report in segments)

    report = plus_job.finish_job(job_root, "full", segments, seams, time.monotonic())
    assert report["frames"] == FRAMES and report["soft_seams"] == 1
    assert report["output"]["carried_source_streams"] is True
    assert [stream["codec_type"] for stream in report["streams"]] == ["video", "audio", "subtitle"]
    output = os.path.join(job_root, report["output"]["output"])
    assert [read_index(frame, scale=2) for frame in decode_all(output, 640, 360)] == list(range(FRAMES))
    assert os.path.exists(os.path.join(plus_job.job_dir(job_root, "full"), "report.json"))


@needs_ffmpeg
def test_second_run_renders_only_what_is_missing(job_root):
    job = plus_job.prepare_job(job_root, "resume", "inputs/source.mkv", {"mode": "turbo"}, PLANNING)
    pipeline = _pipeline(28)
    _render_all(job_root, job, lambda params: pipeline)
    os.remove(os.path.join(plus_job.parts_dir(job_root, "resume"), "seg_00001.mp4"))

    # Planning again gives the same job: same plan, same digest.
    again = plus_job.prepare_job(job_root, "resume", "inputs/source.mkv", {"mode": "turbo"}, PLANNING)
    assert again["digest"] == job["digest"]
    segments, seams = _render_all(job_root, again, lambda params: pipeline)
    assert [report["skipped"] for report in segments] == [True, False, True]
    assert [report["skipped"] for report in seams] == [True]


@needs_ffmpeg
def test_other_parameters_do_not_reuse_finished_segments(job_root):
    job = plus_job.prepare_job(job_root, "params", "inputs/source.mkv", {"mode": "turbo"}, PLANNING)
    pipeline = _pipeline(28)
    _render_all(job_root, job, lambda params: pipeline)
    changed = plus_job.prepare_job(job_root, "params", "inputs/source.mkv", {"mode": "base"}, PLANNING)
    assert changed["digest"] != job["digest"]
    segments, _ = _render_all(job_root, changed, lambda params: pipeline)
    assert [report["skipped"] for report in segments] == [False, False, False]


@needs_ffmpeg
def test_bypass_excerpt_never_loads_the_model(job_root):
    import time

    def no_model(params):
        raise AssertionError("a bypass job must not load the model")

    job = plus_job.prepare_job(
        job_root, "bypass", "inputs/source.mkv", {"bypass_model": True}, PLANNING,
        start_seconds=2.0, duration_seconds=4.0,
    )
    segments, seams = _render_all(job_root, job, no_model)
    report = plus_job.finish_job(job_root, "bypass", segments, seams, time.monotonic())
    assert report["frames"] == 100
    assert report["output"]["carried_source_streams"] is False
    output = os.path.join(job_root, report["output"]["output"])
    assert [read_index(frame, scale=2) for frame in decode_all(output, 640, 360)] == list(range(50, 150))


@needs_ffmpeg
def test_a_model_that_returns_too_few_frames_fails_the_segment(job_root):
    job = plus_job.prepare_job(job_root, "short", "inputs/source.mkv", {"mode": "turbo"}, PLANNING)
    pipeline = _pipeline(28)
    original = plus_runtime.FrameWriter.append

    def lossy(self, video, **kwargs):
        return original(self, video[:, :, :-1], **kwargs)

    plus_runtime.FrameWriter.append = lossy
    try:
        with pytest.raises(Exception, match="planned"):
            plus_runtime.render_segment(
                job_root, "short", job["plan"]["segments"][0], job["params"], job["digest"],
                pipeline_for=lambda params: pipeline, encoder="libx265", host="cpu-test",
            )
    finally:
        plus_runtime.FrameWriter.append = original
    assert not os.path.exists(os.path.join(plus_job.parts_dir(job_root, "short"), "seg_00000.ok"))


def test_job_ids_and_paths_are_validated(tmp_path):
    for job_id in ("", "a/b", ".hidden"):
        with pytest.raises(ValueError):
            plus_job.prepare_job(str(tmp_path), job_id, "inputs/x.mkv")
    for path in ("/abs/x.mkv", "../x.mkv", "a/../../x.mkv"):
        with pytest.raises(ValueError):
            plus_job.prepare_job(str(tmp_path), "job", path)
    with pytest.raises(FileNotFoundError):
        plus_job.prepare_job(str(tmp_path), "job", "inputs/missing.mkv")
    with pytest.raises(ValueError, match="Unknown planning options"):
        plus_job.prepare_job(str(tmp_path), "job", "inputs/x.mkv", planning={"segment_secs": 3})
