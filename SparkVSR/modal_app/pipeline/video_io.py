"""Video decoding, encoding, and metadata probe utilities for SparkVSR.

Ported from ngx-vsr and FlashVSR-Pro with probed hardware encoder selection,
10-bit HEVC encoding with explicit bt709 color tagging, rational frame rate
handling, and audio stream-copy.
"""

import contextlib
import os
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

NVENC_QUALITY_ARGS = [
    "-preset", "p6",
    "-rc", "vbr",
    "-cq", "19",
    "-b:v", "0",
    "-multipass", "fullres",
    "-spatial-aq", "1",
    "-temporal-aq", "1",
    "-aq-strength", "8",
    "-rc-lookahead", "32",
    "-bf", "4",
    "-b_ref_mode", "middle",
]


class VideoMetadata:
    """Metadata describing a probed video file."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        total_frames: int,
        has_audio: bool,
        duration: float = 0.0,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.total_frames = total_frames
        self.has_audio = has_audio
        self.duration = duration

    def __repr__(self) -> str:
        return (
            f"VideoMetadata(width={self.width}, height={self.height}, fps={self.fps:.3f}, "
            f"total_frames={self.total_frames}, has_audio={self.has_audio}, duration={self.duration:.2f}s)"
        )


def probe_video_metadata(video_path: str) -> VideoMetadata:
    """Probe video dimensions, frame rate, frame count, and audio presence via ffprobe."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Probe video stream
    res_v = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=0", video_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if res_v.returncode != 0:
        raise ValueError(f"Failed to probe video stream: {res_v.stderr.strip()[:400]}")

    fields = {}
    for line in res_v.stdout.splitlines():
        k, _, v = line.partition("=")
        fields[k.strip()] = v.strip()

    try:
        width = int(fields["width"])
        height = int(fields["height"])
    except (KeyError, ValueError) as e:
        raise ValueError(f"Video has no decodable video stream ({video_path})") from e

    fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = fields.get(key, "")
        if "/" in raw:
            num, _, den = raw.partition("/")
            with contextlib.suppress(ValueError, ZeroDivisionError):
                fps = int(num) / int(den)
        if fps > 0:
            break
    if fps <= 0:
        fps = 25.0

    duration = 0.0
    with contextlib.suppress(ValueError):
        duration = float(fields.get("duration", 0.0))

    total_frames = 0
    with contextlib.suppress(ValueError):
        total_frames = int(fields.get("nb_frames", 0))

    if total_frames <= 0 and duration > 0:
        total_frames = int(round(duration * fps))

    # Probe audio stream presence
    res_a = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=index",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    has_audio = (res_a.returncode == 0 and bool(res_a.stdout.strip()))

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        has_audio=has_audio,
        duration=duration,
    )


def probe_frame_count(video_path: str) -> int:
    """Return the exact decodable frame count (decord ground truth, not ffprobe's estimate)."""
    import decord

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    return len(vr)


def decode_video_to_uint8(
    video_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    batch_size: int = 32,
) -> Tuple[torch.Tensor, VideoMetadata]:
    """Decode a frame range into a CPU uint8 tensor [T, H, W, C] at SOURCE resolution.

    This is the memory-safe replacement for the old whole-video float32 decode: uint8 at
    source resolution costs ~6.2 MB/frame at 1080p versus ~99.5 MB/frame for float32 at 4K.

    Args:
        video_path: Path to the input video
        start_frame: First frame to decode (inclusive)
        end_frame: One past the last frame to decode (exclusive); None means end of video
        batch_size: Frames decoded per decord call (bounds decord's own scratch buffers)

    Returns:
        (frames_uint8 [T, H, W, C] on CPU, metadata whose total_frames is the FULL video length)
    """
    meta = probe_video_metadata(video_path)

    import decord
    decord.bridge.set_bridge("torch")

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = len(vr)
    # decord's frame count is authoritative; ffprobe's nb_frames can be an estimate.
    meta.total_frames = total_frames

    start_frame = max(0, int(start_frame))
    end_frame = total_frames if end_frame is None else min(int(end_frame), total_frames)
    num = max(0, end_frame - start_frame)
    if num == 0:
        return torch.empty((0, meta.height, meta.width, 3), dtype=torch.uint8), meta

    out: Optional[torch.Tensor] = None
    for base in range(start_frame, end_frame, batch_size):
        stop = min(base + batch_size, end_frame)
        batch = vr.get_batch(list(range(base, stop)))  # [b, H, W, C] uint8
        if batch.dtype != torch.uint8:
            batch = batch.to(torch.uint8)
        if out is None:
            # Take geometry from decord, not ffprobe: rotation metadata can make the two
            # disagree, and decord's frames are what actually feeds the model.
            _, dec_h, dec_w, dec_c = batch.shape
            meta.height, meta.width = int(dec_h), int(dec_w)
            out = torch.empty((num, dec_h, dec_w, dec_c), dtype=torch.uint8)
        out[base - start_frame : stop - start_frame] = batch
        del batch

    assert out is not None
    return out, meta


def decode_video_to_tensor(video_path: str) -> Tuple[torch.Tensor, VideoMetadata]:
    """Deprecated: whole-video float32 [1, C, T, H, W] decode in [0, 1].

    Retained only for backwards compatibility. Allocates ~4x the uint8 footprint and is
    the direct cause of container OOM on long inputs. Prefer `decode_video_to_uint8`.
    """
    frames_u8, meta = decode_video_to_uint8(video_path)
    tensor = frames_u8.permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
    return tensor, meta


def extract_raw_reference_frames(video_path: str, ref_indices: List[int]) -> Dict[int, Image.Image]:
    """Extract raw reference frames as PIL Images by global frame indices."""
    if not ref_indices:
        return {}

    import decord
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_len = len(vr)

    valid_indices = [idx for idx in ref_indices if 0 <= idx < total_len]
    if not valid_indices:
        return {}

    # decord's bridge is process-global: once decode_video_to_uint8 has set it to
    # "torch", get_batch returns a Tensor rather than an NDArray. Accept either.
    batch = vr.get_batch(valid_indices)
    frames = batch.asnumpy() if hasattr(batch, "asnumpy") else batch.numpy()
    results = {}
    for idx, arr in zip(valid_indices, frames):
        results[idx] = Image.fromarray(arr)

    return results


def encoder_works(encoder: str, extra_args: Tuple[str, ...] = ()) -> bool:
    """Probe if an encoder works by encoding a single test frame."""
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
            "-frames:v", "1", "-c:v", encoder, *extra_args, "-f", "null", "-",
        ],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def select_best_encoder() -> Tuple[str, Optional[str]]:
    """Select the highest-quality available video encoder (hevc_nvenc -> h264_nvenc -> libx264)."""
    # Test hevc_nvenc
    if encoder_works("hevc_nvenc", ("-profile:v", "main10", "-pix_fmt", "p010le")):
        tune = "uhq" if encoder_works("hevc_nvenc", ("-tune", "uhq")) else None
        print(f"Selected video encoder: hevc_nvenc (Main 10, tune={tune or 'none'})")
        return "hevc_nvenc", tune

    if encoder_works("h264_nvenc"):
        tune = "uhq" if encoder_works("h264_nvenc", ("-tune", "uhq")) else None
        print(f"Selected video encoder: h264_nvenc (tune={tune or 'none'})")
        return "h264_nvenc", tune

    print("WARNING: NVENC encoders unavailable, falling back to CPU libx264.")
    return "libx264", None


def build_encode_command(
    output_path: str,
    width: int,
    height: int,
    fps: float,
    audio_source_path: Optional[str] = None,
    keep_audio: bool = True,
    encoder: Optional[str] = None,
    tune: Optional[str] = None,
) -> Tuple[List[str], str, Optional[str]]:
    """Build the ffmpeg command for a raw rgb24 stdin pipe -> encoded video file.

    Single source of truth for the encoder ladder, NVENC quality args, BT.709 tagging,
    rational frame rate and audio stream-copy. Both the streaming encoder and the
    whole-tensor encoder go through here.

    Returns:
        (cmd, encoder, tune)
    """
    if encoder is None:
        encoder, tune = select_best_encoder()

    fps_str = str(Fraction(fps).limit_denominator(1001))

    tune_args = ["-tune", tune] if tune else []
    if encoder == "hevc_nvenc":
        codec_args = [
            "-c:v", encoder, *NVENC_QUALITY_ARGS, *tune_args,
            "-profile:v", "main10", "-tier", "high", "-pix_fmt", "p010le",
        ]
    elif encoder.endswith("_nvenc"):
        codec_args = ["-c:v", encoder, *NVENC_QUALITY_ARGS, *tune_args, "-pix_fmt", "yuv420p"]
    else:
        codec_args = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", fps_str,
        "-i", "pipe:0",
    ]

    has_audio_input = bool(audio_source_path and keep_audio and os.path.isfile(audio_source_path))
    if has_audio_input:
        cmd.extend([
            "-i", audio_source_path,
            "-map", "0:v:0", "-map", "1:a:0?",
        ])
    else:
        cmd.extend(["-map", "0:v:0"])

    # BT.709 color primaries, transfer characteristics, and matrix coefficients tagging
    cmd.extend([
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        *codec_args,
    ])

    if has_audio_input:
        # No -shortest: it truncates the video stream against the copied audio and
        # silently drops trailing frames (301 in -> 296 out on the 10 s sample). A
        # restoration pass must preserve every source frame; a fractional audio
        # overhang from the same source is the lesser evil.
        cmd.extend(["-c:a", "copy"])

    cmd.extend(["-movflags", "+faststart", output_path])
    return cmd, encoder, tune


class StreamingVideoEncoder:
    """Context manager wrapping a single long-lived ffmpeg process fed frame by frame.

    Lets the pipeline emit output frames as soon as they finalize instead of accumulating
    a whole-video output tensor in host RAM.

    Usage:
        with StreamingVideoEncoder(path, w, h, fps, ...) as enc:
            for frame in frames:            # frame: HWC uint8 (numpy or torch)
                enc.write_frame(frame)
        print(enc.frames_written, enc.write_seconds)
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        audio_source_path: Optional[str] = None,
        keep_audio: bool = True,
        encoder: Optional[str] = None,
        tune: Optional[str] = None,
    ):
        self.output_path = output_path
        self.width = int(width)
        self.height = int(height)
        self.fps = fps
        self.frames_written = 0
        self.write_seconds = 0.0
        self._proc: Optional[subprocess.Popen] = None
        self._cmd, self.encoder, self.tune = build_encode_command(
            output_path=output_path,
            width=self.width,
            height=self.height,
            fps=fps,
            audio_source_path=audio_source_path,
            keep_audio=keep_audio,
            encoder=encoder,
            tune=tune,
        )

    def open(self) -> "StreamingVideoEncoder":
        if self._proc is not None:
            return self
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        self._proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        return self

    def __enter__(self) -> "StreamingVideoEncoder":
        return self.open()

    def write_frame(self, frame) -> None:
        """Write one HWC uint8 RGB frame (numpy array or torch tensor) to the encoder."""
        if self._proc is None:
            raise RuntimeError("StreamingVideoEncoder is not open.")

        t0 = time.perf_counter()
        if isinstance(frame, torch.Tensor):
            arr = frame.detach().to("cpu").contiguous().numpy()
        else:
            arr = frame
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.shape[0] != self.height or arr.shape[1] != self.width or arr.shape[2] != 3:
            raise ValueError(
                f"Frame shape {arr.shape} does not match encoder geometry "
                f"({self.height}, {self.width}, 3)."
            )
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        try:
            self._proc.stdin.write(arr.tobytes())
        except BrokenPipeError as e:
            stderr = self._proc.stderr.read().decode("utf-8", errors="replace")
            self._proc.wait()
            raise RuntimeError(f"FFmpeg encoder died mid-stream: {stderr}") from e

        self.frames_written += 1
        self.write_seconds += time.perf_counter() - t0

    def close(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        t0 = time.perf_counter()
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        stderr = proc.stderr.read()
        proc.wait()
        self.write_seconds += time.perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError(
                f"FFmpeg encoding failed (code {proc.returncode}): "
                f"{stderr.decode('utf-8', errors='replace')}"
            )

    def abort(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.abort()
            return False
        self.close()
        return False


def encode_video_tensor(
    tensor: torch.Tensor,
    output_path: str,
    fps: float,
    audio_source_path: Optional[str] = None,
    keep_audio: bool = True,
    encoder: Optional[str] = None,
    tune: Optional[str] = None,
):
    """Encode a restored video tensor [1, C, T, H, W] in [0, 1] to an MP4/MKV file.

    Thin wrapper over StreamingVideoEncoder; retains every encoding property of the
    original implementation (probed 10-bit HEVC NVENC, BT.709 tagging, rational frame
    rate, lossless audio stream-copy).
    """
    _, _num_channels, total_frames, height, width = tensor.shape

    with StreamingVideoEncoder(
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
        audio_source_path=audio_source_path,
        keep_audio=keep_audio,
        encoder=encoder,
        tune=tune,
    ) as enc:
        for t in range(total_frames):
            frame = tensor[0, :, t, :, :].clamp(0.0, 1.0)
            frame_uint8 = (frame.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
            enc.write_frame(frame_uint8)


def concat_segments_no_reencode(
    segment_paths: List[str],
    output_path: str,
    audio_source_path: Optional[str] = None,
    keep_audio: bool = True,
    work_dir: Optional[str] = None,
) -> None:
    """Join encoded segments with the ffmpeg concat demuxer using `-c copy` (no re-encode).

    All segments must share codec, resolution and frame rate, which holds because they are
    produced by one `build_encode_command` configuration. Audio is stream-copied from the
    original source in the same (single) remux pass, so nothing is ever re-encoded.
    """
    if not segment_paths:
        raise ValueError("concat_segments_no_reencode called with no segments.")

    has_audio_input = bool(audio_source_path and keep_audio and os.path.isfile(audio_source_path))

    if len(segment_paths) == 1 and not has_audio_input:
        shutil.move(segment_paths[0], output_path)
        return

    work_dir = work_dir or os.path.dirname(os.path.abspath(output_path))
    os.makedirs(work_dir, exist_ok=True)
    list_path = os.path.join(work_dir, "segments.txt")

    with open(list_path, "w") as fh:
        for seg in segment_paths:
            abs_seg = os.path.abspath(seg)
            # concat demuxer escaping: close the quote, escape the quote, reopen
            escaped = abs_seg.replace("'", "'\\''")
            fh.write(f"file '{escaped}'\n")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", list_path,
    ]

    if has_audio_input:
        # As in build_encode_command: -shortest truncates the concatenated video
        # against the copied audio track (1551 frames in, 1546 out on the 60 s sample).
        cmd.extend(["-i", audio_source_path, "-map", "0:v:0", "-map", "1:a:0?"])
    else:
        cmd.extend(["-map", "0:v:0"])

    cmd.extend(["-c", "copy", "-movflags", "+faststart", output_path])

    res = subprocess.run(cmd, capture_output=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(
            f"FFmpeg segment concat failed (code {res.returncode}): "
            f"{res.stderr.decode('utf-8', errors='replace')[:800]}"
        )
