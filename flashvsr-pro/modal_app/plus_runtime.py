"""Drive upstream FlashVSR+ from our own frame source and frame sink.

Imported only inside the GPU container: it needs torch and the upstream package
installed in `plus_image.py`. Written against upstream commit `PLUS_REPO_SHA`
(config.py). Every upstream name used here was read at that commit:

- `inference.presets.stream_config_for_mode`, `inference.build_inference_pipeline`
- `inference.runners.stream.StreamRunner.run(stream, context, segment_writer=...)`
  and its module-level `_offload_dit` / `_restore_dit`
- `inference.preprocessing.TemporalVideoStream`
- `inference.decoders.tiny.TinyDecoder` (`_decode_chunks`, `decode_stream_segment`)
- `models.attention.apply_rope_3d`

NOT YET RUN ON A GPU. This module was written from the source without a Modal
build (owner decision, 2026-11-25). Treat the first run as its test.

Where stock upstream leaves the GPU on the incremental (`decode_every`) path,
at a 3840x2176 canvas:

1. `_offload_dit` moves the whole DiT (5.7 GB) and the attention cache to the
   CPU before every TinyDecoder call, and `_restore_dit` moves the DiT back.
2. `TinyDecoder._decode_chunks` copies every decoded 16-frame chunk to the CPU,
   concatenates there, and `decode_stream_segment` rescales it in bf16 on the CPU.
3. The decoder's temporal memory is copied to the CPU and back around each call.
4. The input stream holds float32 CPU frames (100 MB each).

All four exist to fit a 24 GB card. On 96 GB they only cost time.
`resident_transport` removes 1 to 3 for the duration of one run and puts the
stock behaviour back afterwards, so the same container can measure both. Item 4
is ours to choose: resident runs feed GPU tensors in the model's dtype.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F

from .config import PLUS_CKPT_DIR, PLUS_CONTEXT_PATH, PLUS_TINY_DECODER_PATH
from .plus_job import load_job, marker_matches, parts_dir, publish, publish_json
from .plus_plan import Canvas, Segment
from .plus_video import (
    SegmentEncoder,
    SegmentReader,
    SegmentSink,
    SourceInfo,
    canvas_deliver,
    fingerprint,
    segment_files,
)


def build_pipeline(params: dict):
    """Load one preset onto the GPU. `params` comes from `validate_plus_params`."""
    from inference import ModelPaths, build_inference_pipeline
    from inference.presets import stream_config_for_mode

    mode = params["mode"]
    config = stream_config_for_mode(
        mode,
        device="cuda:0",
        seed=params["seed"],
        decode_every=params["decode_every"],
        tiny_decoder_mode=params["tiny_decoder_mode"],
        # 96 GB holds every layer's KV history; never trade it for host copies.
        cache_offload_mode="never" if params["resident"] else "auto",
        attn_kernel=params["attn_kernel"],
        color_correction=params["color_correction"],
    )
    paths = ModelPaths(
        dit=f"{PLUS_CKPT_DIR}/{mode}/dit.safetensors",
        lq_projection=f"{PLUS_CKPT_DIR}/{mode}/lq_proj_in.safetensors",
        context=PLUS_CONTEXT_PATH,
        tiny_decoder=PLUS_TINY_DECODER_PATH,
    )
    pipeline = build_inference_pipeline(config, paths)
    if params.get("compile_dit"):
        # Experimental, for the benchmark only: a warm container runs hundreds of
        # identical-shape steps. Unverified; compare its output with an
        # uncompiled run before trusting it.
        denoiser = pipeline.bundle.denoiser
        denoiser.forward_stream = torch.compile(denoiser.forward_stream)
    torch.cuda.synchronize()
    return pipeline


def prepare_frames(frames: np.ndarray, canvas: Canvas, device, dtype) -> torch.Tensor:
    """uint8 [n,h,w,3] source frames -> [1,3,n,canvas_h,canvas_w] in [-1, 1].

    Bicubic upscale to the output size on the GPU, then reflect-pad to the
    model's canvas. The padding is cropped off again in `tensor_to_frames`.
    """
    # The reader's arrays view a read-only bytes object; torch wants writable memory.
    x = torch.from_numpy(np.array(frames, copy=True)).to(device)
    x = x.permute(0, 3, 1, 2).float()
    if (x.shape[2], x.shape[3]) != (canvas.out_h, canvas.out_w):
        x = F.interpolate(x, size=(canvas.out_h, canvas.out_w), mode="bicubic", align_corners=False)
        # Bicubic overshoots on hard edges.
        x = x.clamp_(0.0, 255.0)
    left, top = canvas.pad_left, canvas.pad_top
    right = canvas.canvas_w - canvas.out_w - left
    bottom = canvas.canvas_h - canvas.out_h - top
    if left or right or top or bottom:
        x = F.pad(x, (left, right, top, bottom), mode="reflect")
    x = x.div_(127.5).sub_(1.0).to(dtype)
    return x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def tensor_to_frames(video: torch.Tensor, canvas: Canvas, chunk: int = 8) -> Iterator[np.ndarray]:
    """[1,3,F,canvas_h,canvas_w] in [-1, 1] -> uint8 [n,out_h,out_w,3] chunks.

    The crop and the conversion run on the GPU even when the tensor arrives on
    the CPU (the stock path), so the host only ever sees compact uint8 frames.
    """
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
        raise ValueError(f"Expected one video [1,3,F,H,W], got {tuple(video.shape)}.")
    top, left = canvas.pad_top, canvas.pad_left
    for start in range(0, video.shape[2], chunk):
        x = video[0, :, start:start + chunk, top:top + canvas.out_h, left:left + canvas.out_w]
        if x.device.type != "cuda" and torch.cuda.is_available():
            x = x.to("cuda")
        # Out of place first: .float() returns the caller's tensor for fp32 input.
        x = (x.float() + 1.0).mul_(127.5).round_().clamp_(0.0, 255.0).to(torch.uint8)
        yield x.permute(1, 2, 3, 0).contiguous().cpu().numpy()


class FrameWriter:
    """Upstream's `segment_writer` contract, delivering uint8 frames to a callback.

    `StreamRunner.run` calls `append(video)` with [1,3,F,H,W] in [-1, 1] once per
    decoded segment. The model pads its input at the end, so more frames than
    were fed can come back; `limit` drops them.
    """

    def __init__(self, write, canvas: Canvas, limit: int):
        self._write = write
        self._canvas = canvas
        self._limit = limit
        self.frames = 0

    def append(self, video: torch.Tensor, *, value_range: str = "neg_one_to_one") -> None:
        if value_range != "neg_one_to_one":
            raise ValueError(f"Unsupported value_range {value_range!r}.")
        for frames in tensor_to_frames(video, self._canvas):
            remaining = self._limit - self.frames
            if remaining <= 0:
                return
            frames = frames[:remaining]
            self._write(frames)
            self.frames += len(frames)

    def close(self) -> None:
        return None

    def verify(self) -> dict:
        return {"frames": self.frames}


def _clone_memory(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [_clone_memory(item) for item in value]
    return value.detach().clone()


def _resident_decoder(stock):
    """A TinyDecoder that keeps its outputs and its temporal memory on the GPU."""
    from inference.decoders.base import DecodeResult
    from inference.decoders.tiny import TinyDecoder

    if not isinstance(stock, TinyDecoder):
        raise TypeError(f"Resident transport needs the TinyDecoder, got {type(stock).__name__}.")

    class ResidentTinyDecoder(TinyDecoder):
        def _decode_chunks(self, latents, lq_video, *, device, dtype, use_parallel, first=True):
            # Same slicing as upstream's _decode_chunks; only the `.to("cpu")`
            # per chunk is gone.
            latent_chunk_size = 4
            outputs = []
            for latent_start in range(0, latents.shape[2], latent_chunk_size):
                latent_stop = min(latent_start + latent_chunk_size, latents.shape[2])
                current_latents = latents[:, :, latent_start:latent_stop].to(device=device, dtype=dtype)
                if first and latent_start == 0:
                    frame_start = 0
                    frame_count = 4 * (latent_stop - latent_start) - 3
                else:
                    frame_start = 4 * latent_start - (3 if first else 0)
                    frame_count = 4 * (latent_stop - latent_start)
                current_lq = lq_video[:, :, frame_start:frame_start + frame_count].to(
                    device=device, dtype=dtype
                )
                current = self.module.decode_video(
                    current_latents.transpose(1, 2),
                    parallel=use_parallel,
                    show_progress_bar=False,
                    cond=current_lq,
                ).transpose(1, 2)
                outputs.append(current.detach())
                del current_latents, current_lq, current
            return torch.cat(outputs, dim=2)

        @torch.inference_mode()
        def decode_stream_segment(self, latents, lq_video, *, first: bool = False):
            if first:
                self.reset()
                self.last_auto_fallbacks = 0
            parameter = next(self.module.parameters())
            device, dtype = parameter.device, parameter.dtype
            if self.parallel_mode == "auto":
                candidates = self._auto_candidates(lq_video, device)
            else:
                candidates = [(self.parallel_mode, self.parallel_mode == "parallel")]
            # A snapshot is needed only when a failed attempt can be retried.
            saved = _clone_memory(self.module.mem) if len(candidates) > 1 else None
            for name, use_parallel in candidates:
                try:
                    value = self._decode_chunks(
                        latents, lq_video, device=device, dtype=dtype,
                        use_parallel=use_parallel, first=first,
                    )
                except torch.cuda.OutOfMemoryError:
                    if saved is None or name == candidates[-1][0]:
                        raise
                    self.module.mem = _clone_memory(saved)
                    self.last_auto_fallbacks += 1
                    torch.cuda.empty_cache()
                    continue
                self.last_execution_mode = name
                return DecodeResult(value.mul_(2).sub_(1), None)
            raise RuntimeError("Tiny Decoder produced no output")

    return ResidentTinyDecoder(
        stock.module,
        parallel=stock.parallel,
        parallel_mode=stock.parallel_mode,
        channels_last=stock.channels_last,
    )


@contextlib.contextmanager
def resident_transport(pipeline):
    """Keep the DiT, the KV cache and the decoder output on the GPU for one run."""
    import inference.runners.stream as stream_module

    bundle = pipeline.bundle
    stock = (stream_module._offload_dit, stream_module._restore_dit, bundle.decoder)

    def keep_on_gpu(bundle, state, device):
        return None

    stream_module._offload_dit = keep_on_gpu
    stream_module._restore_dit = keep_on_gpu
    bundle.decoder = _resident_decoder(bundle.decoder)
    try:
        yield
    finally:
        stream_module._offload_dit, stream_module._restore_dit, bundle.decoder = stock


@contextlib.contextmanager
def float32_rope():
    """Rotary embeddings in float32 instead of upstream's float64, for one run.

    Benchmark option. Upstream computes `apply_rope_3d` in float64 over every
    token of every attention layer, which is slow on this GPU. The float32
    version changes the result slightly, so compare outputs before adopting it.
    """
    import models.attention as attention
    from models.embeddings import rope_angles_3d

    stock = attention.apply_rope_3d

    def apply_rope_3d(x, shape, *, temporal_offset: int = 0):
        angles = rope_angles_3d(shape, x.shape[-1], x.device, temporal_offset=temporal_offset)
        x_float = x.float().reshape(*x.shape[:-1], -1, 2)
        cos = angles.cos().view(1, 1, angles.shape[0], -1)
        sin = angles.sin().view(1, 1, angles.shape[0], -1)
        first, second = x_float[..., 0], x_float[..., 1]
        rotated = torch.stack((first * cos - second * sin, second * cos + first * sin), dim=-1)
        return rotated.flatten(-2).to(dtype=x.dtype)

    attention.apply_rope_3d = apply_rope_3d
    try:
        yield
    finally:
        attention.apply_rope_3d = stock


def run_stream(
    pipeline,
    chunks: Iterable[np.ndarray],
    canvas: Canvas,
    writer: FrameWriter,
    *,
    resident: bool,
    rope_float32: bool = False,
    progress=None,
) -> dict:
    """Run one stream of the model over `chunks` and hand its frames to `writer`.

    Returns upstream's runner metadata (chunk counts, cache policy, decoder mode).
    """
    from inference.config import TEMPORAL_DOWNSAMPLE
    from inference.preprocessing import TemporalVideoStream
    from inference.randomness import set_inference_seed

    runner, config = pipeline.runner, pipeline.config
    device = torch.device(config.device)
    dtype = runner._execution_dtype()

    def tensors():
        for frames in chunks:
            if resident:
                yield prepare_frames(frames, canvas, device, dtype)
            else:
                # What upstream's own loader yields: float32 on the CPU.
                yield prepare_frames(frames, canvas, device, torch.float32).cpu()

    set_inference_seed(config.seed)
    stream = TemporalVideoStream(tensors(), config.temporal_compression, TEMPORAL_DOWNSAMPLE)
    with contextlib.ExitStack() as stack:
        if resident:
            stack.enter_context(resident_transport(pipeline))
        if rope_float32:
            stack.enter_context(float32_rope())
        try:
            output = runner.run(stream, None, progress=progress, segment_writer=writer)
        finally:
            stream.close()
    return dict(output.metadata)


def run_bypass(chunks: Iterable[np.ndarray], canvas: Canvas, writer: FrameWriter) -> dict:
    """The I/O path without the model: decode, bicubic upscale, pad, crop, encode.

    For colour and frame-count checks. What comes out is exactly the picture the
    model would have been conditioned on.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for frames in chunks:
        writer.append(prepare_frames(frames, canvas, device, torch.float32))
    return {"runner": "bypass"}


class GpuSampler:
    """Sample GPU utilisation and memory with nvidia-smi while a job runs."""

    def __init__(self, interval_ms: int = 2000):
        self.utilization: list[int] = []
        self.memory_mib: list[int] = []
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._interval_ms = interval_ms

    def __enter__(self) -> "GpuSampler":
        try:
            self._process = subprocess.Popen(
                [
                    "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits", "-lms", str(self._interval_ms),
                ],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except OSError as exc:
            print(f"WARNING: GPU sampling unavailable ({exc}); utilisation will not be reported.")
            return self
        self._thread = threading.Thread(target=self._pump, name="gpu-sampler", daemon=True)
        self._thread.start()
        return self

    def _pump(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                self.utilization.append(int(parts[0]))
                self.memory_mib.append(int(parts[1]))

    def __exit__(self, *exc_info) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process.wait()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        if not self.utilization:
            return {"samples": 0}
        ordered = sorted(self.utilization)
        return {
            "samples": len(ordered),
            "utilization_mean": round(sum(ordered) / len(ordered), 1),
            "utilization_p10": ordered[len(ordered) // 10],
            "memory_peak_mib": max(self.memory_mib),
        }


_RUNNER_REPORT_KEYS = (
    "runner", "chunks", "attention_topk", "cache_offload_mode", "cache_offload_decision_reason",
    "tiny_decoder_execution_mode", "tiny_decoder_auto_fallbacks", "input_buffer_peak_frames",
)


def render_segment(
    root: str,
    job_id: str,
    segment: dict,
    params: dict,
    digest: str,
    *,
    pipeline_for: Callable[[dict], object],
    encoder: str,
    host: str,
) -> dict:
    """Render one planned segment of a job under `root`. Returns its report.

    `digest` identifies the job's source, plan and parameters. A segment whose
    marker carries the same digest is returned as is. `pipeline_for(params)`
    supplies the loaded model and is not called for a bypass run.
    """
    seg = Segment(**segment)
    names = segment_files(seg)
    folder = parts_dir(root, job_id)
    targets = {kind: os.path.join(folder, name) for kind, name in names.items()}
    required = [targets[kind] for kind, frames in
                (("video", seg.video), ("head", seg.head), ("tail", seg.tail)) if frames]
    cached = marker_matches(targets["marker"], digest, required)
    if cached is not None:
        print(f"{seg.name()}: already rendered, skipping.")
        return {**cached, "skipped": True}

    job = load_job(root, job_id)
    info = SourceInfo.from_json(job["source"])
    canvas = Canvas(**job["canvas"])
    source_path = os.path.join(root, job["input_path"])
    on_gpu = torch.cuda.is_available()

    started = time.monotonic()
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory() as scratch:
        # Scratch stays on local disk; only finished files are published. Work
        # in progress on a Volume is lost by a reload.
        local_video = os.path.join(scratch, names["video"])
        reader = SegmentReader(source_path, info, seg.model_start, seg.model_end, scratch_dir=scratch)
        sink_encoder = None
        try:
            if seg.video:
                sink_encoder = SegmentEncoder(
                    canvas.out_w, canvas.out_h, info.fps, local_video,
                    encoder=encoder, deliver=canvas_deliver(canvas),
                )
            sink = SegmentSink(seg, sink_encoder)
            writer = FrameWriter(sink.write, canvas, limit=seg.model_frames)
            with GpuSampler() as sampler:
                if params["bypass_model"]:
                    metadata = run_bypass(reader, canvas, writer)
                else:
                    metadata = run_stream(
                        pipeline_for(params), reader, canvas, writer,
                        resident=params["resident"], rope_float32=params["rope_float32"],
                        progress=lambda message: print(f"{seg.name()}: {message}"),
                    )
            reader.finish()
            head, tail = sink.finish()
        except BaseException:
            reader.close()
            if sink_encoder is not None:
                sink_encoder.abort()
            raise
        seconds = time.monotonic() - started

        report = {
            "segment": seg.name(),
            "digest": digest,
            "model_frames": seg.model_frames,
            "video_frames": seg.video,
            "seconds": round(seconds, 2),
            "fps": round(seg.model_frames / seconds, 3),
            "decode_wait_seconds": round(reader.wait_seconds, 2),
            "encode_wait_seconds": round(sink_encoder.wait_seconds, 2) if sink_encoder else 0.0,
            "torch_peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if on_gpu else None,
            "gpu": host,
            "gpu_samples": sampler.summary(),
            "encoder": encoder,
            "runner": {key: metadata[key] for key in _RUNNER_REPORT_KEYS if key in metadata},
            "skipped": False,
        }
        if seg.video:
            found = fingerprint(local_video)
            if found.frames != seg.video:
                raise RuntimeError(
                    f"{seg.name()}: the encoded file has {found.frames} frames, planned {seg.video}."
                )
            report["extradata_md5"] = found.extradata_md5
            report["level"] = found.level
            report["megabits_per_second"] = round(
                os.path.getsize(local_video) * 8 / 1e6 / float(Fraction(seg.video) / info.fps), 2
            )
            publish(local_video, targets["video"])
        for kind, frames in (("head", head), ("tail", tail)):
            if frames is not None:
                local = os.path.join(scratch, names[kind])
                np.save(local, frames)
                publish(local, targets[kind])
    # The marker goes last: it is the claim that everything above is complete.
    publish_json(report, targets["marker"])
    print(f"{seg.name()}: {seg.model_frames} frames in {seconds:.0f}s ({report['fps']} fps)")
    return report
