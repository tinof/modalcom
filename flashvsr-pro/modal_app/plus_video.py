"""Video I/O for the FlashVSR+ worker: probe, scene scores, decode, encode, join.

No torch and no Modal in this module. The GPU worker hands it uint8 frames and
the CPU orchestrator joins what the workers wrote, so the whole path except the
model can run, and is tested, on any machine with ffmpeg.

The decode and encode commands are ported from rtx-vsr (`_ffmpeg_decode_command`
and `_ffmpeg_encode_command` in rtx-vsr/modal_app.py), where each flag was
measured. Three of them are load-bearing and easy to lose:

- swscale's default RGB conversion uses lookup tables that sit about 1.6 luma
  levels dark; `accurate_rnd+full_chroma_int` leaves that path.
- Raw RGB carries no colour metadata. Without an explicit matrix swscale encodes
  BT.601 while players read HD and UHD as BT.709, a visible hue shift.
- rawvideo output defaults to constant frame rate and duplicates frames to fill
  timestamp gaps; `-fps_mode passthrough` hands frames over one to one.

Unlike rtx-vsr, a segment here may start on any frame, not only on a keyframe,
because the cuts follow scene changes and the source's GOP does not.
"""

from __future__ import annotations

import bisect
import contextlib
import hashlib
import json
import queue
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from .config import (
    PLUS_HEVC_LEVEL5_MAX_PIXELS,
    PLUS_NVENC_CQ,
    PLUS_NVENC_LEVEL,
    PLUS_NVENC_MAX_MBPS,
    PLUS_NVENC_PRESET,
)
from .plus_plan import Canvas, Plan, Seam, Segment, seam_weights

# ffprobe colour-space names that swscale's in_color_matrix accepts under another name.
_SWS_MATRIX = {
    "bt709": "bt709",
    "smpte170m": "smpte170m",
    "bt470bg": "bt470bg",
    "smpte240m": "smpte240m",
    "fcc": "fcc",
    "bt2020nc": "bt2020",
    "bt2020c": "bt2020",
}
_INTERLACED = frozenset({"tt", "bb", "tb", "bt"})
_HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})

# NVENC quality arguments: rtx-vsr's VMAF-chosen point (see config.py).
NVENC_QUALITY_ARGS = [
    "-preset", PLUS_NVENC_PRESET,
    "-rc", "vbr", "-cq", PLUS_NVENC_CQ, "-b:v", "0",
    *(["-maxrate", f"{PLUS_NVENC_MAX_MBPS}M", "-bufsize", f"{PLUS_NVENC_MAX_MBPS}M"]
      if PLUS_NVENC_MAX_MBPS > 0 else []),
    "-multipass", "fullres", "-tune", "uhq",
    "-spatial-aq", "1", "-temporal-aq", "1", "-aq-strength", "10",
    "-rc-lookahead", "32",
    "-bf", "5", "-b_ref_mode", "middle",
]


class VideoError(RuntimeError):
    """A video step produced something other than what was planned."""


def _run(command: list[str], *, what: str) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise VideoError(f"{what} failed ({result.returncode}): {result.stderr.strip()[-800:]}")
    return result


# ---------------------------------------------------------------------------
# Source probe
# ---------------------------------------------------------------------------
@dataclass
class SourceInfo:
    width: int
    height: int
    fps: Fraction
    time_base: Fraction
    # Seconds; ffmpeg's input -ss is relative to the container's start time.
    format_start: float
    video_start: float
    color_space: str
    color_range: str
    color_transfer: str
    pix_fmt: str
    field_order: str
    # pts of every video frame in display order, and of the keyframes.
    frame_pts: list[int] = field(repr=False)
    keyframe_pts: list[int] = field(repr=False)

    @property
    def frames(self) -> int:
        return len(self.frame_pts)

    def to_json(self) -> dict:
        data = asdict(self)
        data["fps"] = f"{self.fps.numerator}/{self.fps.denominator}"
        data["time_base"] = f"{self.time_base.numerator}/{self.time_base.denominator}"
        return data

    @staticmethod
    def from_json(data: dict) -> "SourceInfo":
        data = dict(data)
        data["fps"] = Fraction(data["fps"])
        data["time_base"] = Fraction(data["time_base"])
        return SourceInfo(**data)


def _float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def check_constant_frame_rate(frame_pts: list[int]) -> None:
    """Raise unless consecutive frames are evenly spaced (one tick of slack).

    The encoder writes a constant frame rate, so a variable-rate source would
    drift against its audio. One tick of slack covers time bases that cannot
    represent the frame duration exactly (23.976 fps in 1/1000 s).
    """
    if len(frame_pts) < 3:
        return
    deltas = [b - a for a, b in zip(frame_pts, frame_pts[1:])]
    if min(deltas) <= 0:
        raise VideoError("Source has duplicate or non-increasing video timestamps.")
    if max(deltas) - min(deltas) > 1:
        raise VideoError(
            "Source has a variable frame rate "
            f"(frame durations from {min(deltas)} to {max(deltas)} ticks); not supported."
        )


def probe_source(path: str) -> SourceInfo:
    """Stream facts plus the pts of every frame of the first video stream."""
    probe = _run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,time_base,start_time,"
            "color_space,color_range,color_transfer,pix_fmt,field_order"
            ":format=start_time",
            "-of", "json", path,
        ],
        what="ffprobe (stream)",
    )
    data = json.loads(probe.stdout)
    if not data.get("streams"):
        raise VideoError(f"No video stream in {path}.")
    stream = data["streams"][0]
    fps = Fraction(stream.get("r_frame_rate") or "0/1")
    if fps <= 0:
        fps = Fraction(stream.get("avg_frame_rate") or "0/1")
    if fps <= 0:
        raise VideoError(f"Cannot read a frame rate from {path}.")

    # Demux only: a packet scan of a two-hour film takes seconds.
    packets = _run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "packet=pts,flags", "-of", "csv=p=0", path,
        ],
        what="ffprobe (packets)",
    ).stdout.split()
    frame_pts: list[int] = []
    keyframes: list[int] = []
    for row in packets:
        pts, _, flags = row.partition(",")
        if not pts.lstrip("-").isdigit():
            continue
        # Packets flagged D (discard) are decoded and never shown.
        if "D" in flags:
            continue
        frame_pts.append(int(pts))
        if "K" in flags:
            keyframes.append(int(pts))
    frame_pts = sorted(set(frame_pts))
    if not frame_pts:
        raise VideoError(f"No video packets in {path}.")

    info = SourceInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        time_base=Fraction(stream["time_base"]),
        format_start=_float(data.get("format", {}).get("start_time")),
        video_start=_float(stream.get("start_time")),
        color_space=stream.get("color_space", "") or "",
        color_range=stream.get("color_range", "") or "",
        color_transfer=stream.get("color_transfer", "") or "",
        pix_fmt=stream.get("pix_fmt", "") or "",
        field_order=stream.get("field_order", "") or "",
        frame_pts=frame_pts,
        keyframe_pts=sorted(set(keyframes)),
    )
    if info.field_order in _INTERLACED:
        raise VideoError(
            f"Source is flagged interlaced (field_order={info.field_order}). "
            "Deinterlace it first; the model expects progressive frames."
        )
    if info.color_transfer in _HDR_TRANSFERS:
        raise VideoError(
            f"Source is HDR ({info.color_transfer}). This worker reads and writes SDR BT.709 only."
        )
    check_constant_frame_rate(frame_pts)
    return info


def frame_range(info: SourceInfo, start_seconds: float | None, duration_seconds: float | None) -> tuple[int, int]:
    """[first, last) frame numbers for an optional excerpt of the source."""
    first = 0
    if start_seconds:
        first = min(info.frames - 1, max(0, int(round(start_seconds * info.fps))))
    last = info.frames
    if duration_seconds:
        last = min(info.frames, first + max(1, int(round(duration_seconds * info.fps))))
    return first, last


# ---------------------------------------------------------------------------
# Scene scores
# ---------------------------------------------------------------------------
_SCENE_FRAME = re.compile(r"^frame:(\d+)\s")
_SCENE_SCORE = re.compile(r"^lavfi\.scene_score=([0-9.]+)")


def scene_score_command(path: str) -> list[str]:
    """One pass that prints ffmpeg's scene score for every frame, at low resolution."""
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", "scale=320:-2:flags=fast_bilinear,select='gte(scene,0)',metadata=print:file=-",
        "-f", "null", "-",
    ]


def parse_scene_scores(text: str) -> list[float]:
    """Scores indexed by frame number: the change of frame n against frame n-1."""
    scores: list[float] = []
    current: int | None = None
    for line in text.splitlines():
        match = _SCENE_FRAME.match(line)
        if match:
            current = int(match.group(1))
            continue
        match = _SCENE_SCORE.match(line)
        if match and current is not None:
            if current >= len(scores):
                scores.extend([0.0] * (current + 1 - len(scores)))
            scores[current] = float(match.group(1))
            current = None
    return scores


def scene_scores(path: str, expected_frames: int) -> list[float]:
    scores = parse_scene_scores(_run(scene_score_command(path), what="scene detection").stdout)
    # The filter numbers the frames it sees. If that disagrees with the packet
    # scan, frame numbers would point at the wrong pictures.
    if len(scores) != expected_frames:
        raise VideoError(
            f"Scene pass decoded {len(scores)} frames, the packet scan found {expected_frames}."
        )
    return scores


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
def decode_command(
    path: str,
    info: SourceInfo,
    start_frame: int,
    end_frame: int,
    *,
    pts_log: str | None = None,
) -> list[str]:
    """rawvideo rgb24 of source frames [start_frame, end_frame), frame-exact.

    Seeks to one second before the keyframe at or before the first frame (input
    -ss is relative to the container's start time), keeps original timestamps
    and cuts with `trim` in the stream's own time base. `pts_log` records the
    pts of every delivered frame so the reader can prove it got the right ones.
    """
    if not 0 <= start_frame < end_frame <= info.frames:
        raise ValueError(f"Frame range [{start_frame}, {end_frame}) outside 0..{info.frames}.")
    start_pts = info.frame_pts[start_frame]
    bounds = [f"start_pts={start_pts}"]
    if end_frame < info.frames:
        bounds.append(f"end_pts={info.frame_pts[end_frame]}")
    seek: list[str] = ["-copyts"]
    if start_frame > 0:
        position = bisect.bisect_right(info.keyframe_pts, start_pts) - 1
        keyframe = info.keyframe_pts[position] if position >= 0 else info.frame_pts[0]
        seconds = float(keyframe * info.time_base) - info.format_start - 1.0
        if seconds > 0:
            seek += ["-noaccurate_seek", "-ss", f"{seconds:.6f}"]
    matrix = _SWS_MATRIX.get(info.color_space)
    if matrix is None:
        # Untagged streams: the rule NVDEC and every player use.
        matrix = "bt709" if info.height > 576 else "smpte170m"
    in_range = "pc" if info.color_range == "pc" else "tv"
    stats: list[str] = []
    if pts_log is not None:
        stats = ["-enc_time_base:v", "demux", "-stats_enc_pre", pts_log,
                 "-stats_enc_pre_fmt", "{pts}"]
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", *seek,
        "-i", path, "-map", "0:v:0", "-an", "-sn", "-dn",
        "-fps_mode", "passthrough",
        "-vf", f"trim={':'.join(bounds)},scale=in_color_matrix={matrix}:in_range={in_range}"
               ":flags=bicubic+accurate_rnd+full_chroma_int",
        *stats,
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]


class SegmentReader:
    """Decode one frame range on a thread into a bounded queue of uint8 chunks.

    Iterating yields arrays of shape [n, height, width, 3]. `finish()` raises
    unless exactly the planned frames arrived, starting at the planned pts.
    """

    def __init__(
        self,
        path: str,
        info: SourceInfo,
        start_frame: int,
        end_frame: int,
        *,
        scratch_dir: str,
        chunk_frames: int = 16,
        queue_chunks: int = 4,
    ):
        self.info = info
        self.start_frame, self.end_frame = start_frame, end_frame
        self.expected = end_frame - start_frame
        self.chunk_frames = chunk_frames
        self.frames_read = 0
        self.wait_seconds = 0.0
        self._pts_log = str(Path(scratch_dir) / f"decode_{start_frame}.pts")
        self._stderr_path = Path(scratch_dir) / f"decode_{start_frame}.log"
        self._stderr = self._stderr_path.open("wb")
        self._process = subprocess.Popen(
            decode_command(path, info, start_frame, end_frame, pts_log=self._pts_log),
            stdout=subprocess.PIPE, stderr=self._stderr,
        )
        self._queue: queue.Queue = queue.Queue(maxsize=queue_chunks)
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._pump, name="segment-reader", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        frame_bytes = self.info.width * self.info.height * 3
        stdout = self._process.stdout
        assert stdout is not None
        try:
            while not self._stop.is_set():
                data = stdout.read(frame_bytes * self.chunk_frames)
                if not data:
                    break
                if len(data) % frame_bytes:
                    raise VideoError("Decoder ended in the middle of a frame.")
                frames = np.frombuffer(data, dtype=np.uint8).reshape(
                    -1, self.info.height, self.info.width, 3
                )
                if not _put_until_stopped(self._queue, frames, self._stop):
                    return
        except BaseException as exc:  # noqa: BLE001 - handed to the consumer
            self._error = exc
        finally:
            _put_until_stopped(self._queue, None, self._stop)

    def __iter__(self):
        while True:
            waited = time.monotonic()
            item = self._queue.get()
            self.wait_seconds += time.monotonic() - waited
            if item is None:
                break
            self.frames_read += len(item)
            yield item
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        """Stop the decoder whether or not it was read to the end."""
        self._stop.set()
        with contextlib.suppress(queue.Empty):
            while True:
                self._queue.get_nowait()
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()
        self._thread.join(timeout=10)
        if not self._stderr.closed:
            self._stderr.close()

    def finish(self) -> None:
        returncode = self._process.wait()
        self._thread.join(timeout=10)
        self._stderr.close()
        log = self._stderr_path.read_text(errors="ignore").strip()
        if returncode != 0:
            raise VideoError(f"Decoder failed ({returncode}): {log[-800:]}")
        if self.frames_read != self.expected:
            raise VideoError(
                f"Decoder delivered {self.frames_read} frames for "
                f"[{self.start_frame}, {self.end_frame}), expected {self.expected}. {log[-300:]}"
            )
        delivered = Path(self._pts_log).read_text().split()
        planned = self.info.frame_pts[self.start_frame]
        if not delivered or int(delivered[0]) != planned:
            raise VideoError(
                f"Decoder started at pts {delivered[:1] or 'nothing'}, expected {planned}."
            )


def _put_until_stopped(target: queue.Queue, item, stop: threading.Event) -> bool:
    """Put into a bounded queue, giving up once `stop` is set (a dead consumer never drains it)."""
    while not stop.is_set():
        try:
            target.put(item, timeout=0.5)
            return True
        except queue.Full:
            continue
    return False


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------
def hevc_level(width: int, height: int) -> str | None:
    """The pinned level for outputs that fit level 5.x, else None (driver's choice)."""
    if PLUS_NVENC_LEVEL and width * height <= PLUS_HEVC_LEVEL5_MAX_PIXELS:
        return PLUS_NVENC_LEVEL
    return None


def encoder_codec_args(encoder: str, width: int, height: int) -> list[str]:
    """Codec arguments for one segment. Every segment of a job must use the same ones."""
    if encoder == "hevc_nvenc":
        level = hevc_level(width, height)
        return [
            "-c:v", "hevc_nvenc", *NVENC_QUALITY_ARGS,
            "-profile:v", "main10", "-tier", "high", "-pix_fmt", "p010le",
            *(["-level", level] if level else []),
        ]
    if encoder == "libx265":
        # CPU stand-in for tests and for a machine without NVENC. Closed GOPs so
        # that every file starts on an IDR, as NVENC's do.
        return [
            "-c:v", "libx265", "-preset", "medium", "-crf", "16",
            "-profile:v", "main10", "-pix_fmt", "yuv420p10le",
            "-x265-params", "open-gop=0:log-level=error",
        ]
    raise ValueError(f"Unsupported video encoder {encoder!r} (use hevc_nvenc or libx265).")


def encode_command(
    width: int,
    height: int,
    fps: Fraction,
    output_path: str,
    *,
    encoder: str,
    deliver: tuple[int, int] | None = None,
) -> list[str]:
    """Encode rgb24 frames from stdin into a video-only mp4, tagged BT.709.

    `deliver` resamples to another size with Lanczos before the encode, for a
    job that renders below its delivery size.
    """
    out_w, out_h = deliver or (width, height)
    resize = f"scale={out_w}:{out_h}:flags=lanczos+accurate_rnd+full_chroma_int," if (out_w, out_h) != (width, height) else ""
    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", f"{fps.numerator}/{fps.denominator}", "-i", "pipe:0",
        "-map", "0:v:0", "-an",
        "-vf", f"{resize}scale=out_color_matrix=bt709:out_range=tv"
               ":flags=bicubic+accurate_rnd+full_chroma_int,"
               "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        *encoder_codec_args(encoder, out_w, out_h),
        output_path,
    ]


# NVENC refuses to create a session at random when an explicit level is set
# (rtx-vsr measured error 8 on 3 to 5 of 30 CreateEncoder calls, and the next
# attempt almost always succeeds). ffmpeg opens the encoder when the first frame
# arrives, which here is minutes into a segment, so a failed start is retried
# with the frames written so far instead of failing the segment.
ENCODER_START_ATTEMPTS = 6
# Frames kept for such a retry. Once this many are through, the encoder has
# started and any later failure is a real one.
ENCODER_REPLAY_FRAMES = 48


class SegmentEncoder:
    """Feed uint8 RGB frames to one ffmpeg encode from a writer thread.

    The model never waits on the encoder beyond a bounded queue, and ffmpeg's
    stderr goes to a file, so a chatty encoder cannot fill a pipe and deadlock.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: Fraction,
        output_path: str,
        *,
        encoder: str,
        deliver: tuple[int, int] | None = None,
        queue_chunks: int = 4,
    ):
        self.width, self.height = width, height
        self.output_path = output_path
        self.frames_written = 0
        self.wait_seconds = 0.0
        self.starts = 0
        self._command = encode_command(width, height, fps, output_path, encoder=encoder, deliver=deliver)
        self._stderr_path = Path(f"{output_path}.log")
        self._stderr = self._stderr_path.open("ab")
        self._replay: list[np.ndarray] | None = []
        self._replay_frames = 0
        self._aborted = False
        self._error: BaseException | None = None
        self._process = self._spawn()
        self._queue: queue.Queue = queue.Queue(maxsize=queue_chunks)
        self._thread = threading.Thread(target=self._pump, name="segment-encoder", daemon=True)
        self._thread.start()

    def _spawn(self) -> subprocess.Popen:
        self.starts += 1
        return subprocess.Popen(
            self._command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr,
        )

    def _log_tail(self, limit: int = 800) -> str:
        if not self._stderr.closed:
            self._stderr.flush()
        return self._stderr_path.read_text(errors="ignore").strip()[-limit:]

    def _send(self, frames: np.ndarray) -> None:
        stdin = self._process.stdin
        assert stdin is not None
        stdin.write(frames.tobytes())

    def _restart(self, reason: object) -> None:
        """Start ffmpeg again and feed it every frame written so far."""
        while True:
            if self._aborted or self._replay is None or self.starts >= ENCODER_START_ATTEMPTS:
                raise VideoError(f"Encoder failed after {self.starts} start(s): {reason} {self._log_tail()}")
            print(
                f"WARNING: encoder start {self.starts} failed ({reason}; {self._log_tail(200)!r}); "
                f"starting it again with the {self._replay_frames} frames written so far."
            )
            self._discard_process()
            self._process = self._spawn()
            try:
                for frames in self._replay:
                    self._send(frames)
                return
            except OSError as exc:
                reason = exc

    def _discard_process(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()
        # Closing flushes; on a dead process that raises, here instead of in a
        # destructor later.
        with contextlib.suppress(OSError, ValueError):
            if self._process.stdin is not None:
                self._process.stdin.close()

    def _deliver(self, frames: np.ndarray) -> None:
        while True:
            try:
                self._send(frames)
                break
            except OSError as exc:  # a dead ffmpeg shows up as a broken pipe
                # Raises once the attempts are used up or start-up is over.
                self._restart(exc)
        if self._replay is not None:
            self._replay.append(frames)
            self._replay_frames += len(frames)
            if self._replay_frames >= ENCODER_REPLAY_FRAMES:
                self._replay = None

    def _finish(self) -> None:
        while True:
            with contextlib.suppress(OSError):
                if self._process.stdin is not None:
                    self._process.stdin.close()
            returncode = self._process.wait()
            if returncode == 0:
                return
            # A clip shorter than the replay window can fail to start without
            # ever breaking the pipe; the exit code is then the first sign.
            self._restart(f"exit code {returncode}")

    def _pump(self) -> None:
        finished = False
        try:
            while True:
                frames = self._queue.get()
                if frames is None:
                    finished = True
                    self._finish()
                    return
                self._deliver(frames)
        except BaseException as exc:  # noqa: BLE001 - re-raised by write()/close()
            self._error = exc
            # Keep draining so the producer never blocks on a dead encoder.
            while not finished and self._queue.get() is not None:
                pass

    def _failure(self) -> VideoError:
        if isinstance(self._error, VideoError):
            return self._error
        return VideoError(f"Encoder failed: {self._error or ''} {self._log_tail()}")

    def write(self, frames: np.ndarray) -> None:
        """Queue frames of shape [n, height, width, 3], dtype uint8."""
        if frames.dtype != np.uint8 or frames.shape[1:] != (self.height, self.width, 3):
            raise ValueError(
                f"Encoder expects uint8 [n,{self.height},{self.width},3], got {frames.dtype} {frames.shape}."
            )
        if self._error is not None:
            raise self._failure()
        waited = time.monotonic()
        self._queue.put(np.ascontiguousarray(frames))
        self.wait_seconds += time.monotonic() - waited
        self.frames_written += len(frames)

    def close(self) -> None:
        """Flush, wait for ffmpeg, and raise if it did not finish cleanly."""
        self._queue.put(None)
        self._thread.join()
        if not self._stderr.closed:
            self._stderr.close()
        if self._error is not None:
            raise self._failure()

    def abort(self) -> None:
        """Stop without finishing. Safe to call after a failure or a close."""
        self._aborted = True
        if self._thread.is_alive():
            # Make room for the sentinel: a full queue would drop it and leave
            # the writer thread blocked with the frames it holds.
            with contextlib.suppress(queue.Empty):
                while True:
                    self._queue.get_nowait()
            self._queue.put(None)
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()
        self._thread.join(timeout=10)
        if not self._thread.is_alive():
            self._discard_process()
        if not self._stderr.closed:
            self._stderr.close()


def encode_frames(
    frames: np.ndarray,
    fps: Fraction,
    output_path: str,
    *,
    encoder: str,
    deliver: tuple[int, int] | None = None,
) -> None:
    """Encode a short in-memory clip (a seam) with the same settings as a segment."""
    sink = SegmentEncoder(
        frames.shape[2], frames.shape[1], fps, output_path, encoder=encoder, deliver=deliver
    )
    try:
        sink.write(frames)
        sink.close()
    except BaseException:
        sink.abort()
        raise


class SegmentSink:
    """Route a segment's output frames, in model order, to where the plan sends them.

    Order: `warmup` frames dropped, `head` frames to a sidecar, `video` frames to
    the encoder, `tail` frames to a sidecar. Frames beyond the planned total are
    ignored: the model pads its input at the end and may hand back the padding.
    """

    def __init__(self, segment: Segment, encoder: SegmentEncoder | None):
        self.segment = segment
        self.encoder = encoder
        self.received = 0
        self._head: list[np.ndarray] = []
        self._tail: list[np.ndarray] = []
        if segment.video > 0 and encoder is None:
            raise ValueError("A segment with video frames needs an encoder.")

    def write(self, frames: np.ndarray) -> None:
        segment = self.segment
        marks = (
            segment.warmup,
            segment.warmup + segment.head,
            segment.warmup + segment.head + segment.video,
            segment.model_frames,
        )
        start = self.received
        stop = start + len(frames)
        self.received = stop

        def part(low: int, high: int) -> np.ndarray | None:
            a, b = max(start, low), min(stop, high)
            return frames[a - start:b - start] if b > a else None

        head = part(marks[0], marks[1])
        if head is not None:
            self._head.append(head.copy())
        video = part(marks[1], marks[2])
        if video is not None and self.encoder is not None:
            self.encoder.write(video)
        tail = part(marks[2], marks[3])
        if tail is not None:
            self._tail.append(tail.copy())

    def finish(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Close the encoder and return the (head, tail) sidecar arrays."""
        if self.received < self.segment.model_frames:
            raise VideoError(
                f"{self.segment.name()}: the model returned {self.received} frames, "
                f"planned {self.segment.model_frames}."
            )
        if self.encoder is not None:
            self.encoder.close()
            if self.encoder.frames_written != self.segment.video:
                raise VideoError(
                    f"{self.segment.name()}: encoded {self.encoder.frames_written} frames, "
                    f"planned {self.segment.video}."
                )
        head = np.concatenate(self._head) if self._head else None
        tail = np.concatenate(self._tail) if self._tail else None
        return head, tail


def blend_seam(tail: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Cross-fade the earlier segment's tail into the later segment's head."""
    if tail.shape != head.shape or tail.dtype != np.uint8:
        raise VideoError(f"Seam sidecars differ: tail {tail.shape}, head {head.shape}.")
    weights = np.asarray(seam_weights(len(tail)), dtype=np.float32).reshape(-1, 1, 1, 1)
    mixed = tail.astype(np.float32) * (1.0 - weights) + head.astype(np.float32) * weights
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Verify, join, mux
# ---------------------------------------------------------------------------
@dataclass
class VideoFingerprint:
    frames: int
    codec: str
    profile: str
    level: int
    pix_fmt: str
    width: int
    height: int
    fps: str
    extradata_md5: str
    first_vcl_nal: int | None

    def codec_parameters(self) -> tuple:
        """What must be identical across files for a stream-copy concat to be safe."""
        return (self.codec, self.profile, self.level, self.pix_fmt, self.width, self.height,
                self.fps, self.extradata_md5)


_HEVC_IDR_NALS = frozenset({19, 20})  # IDR_W_RADL, IDR_N_LP


def first_vcl_nal_type(annexb: bytes) -> int | None:
    """NAL unit type of the first picture slice in an HEVC Annex B stream."""
    position, size = 0, len(annexb)
    while position < size - 3:
        if annexb[position:position + 3] == b"\x00\x00\x01":
            nal_type = (annexb[position + 3] >> 1) & 0x3F
            if nal_type <= 31:  # VCL NAL unit types are 0..31
                return nal_type
            position += 3
        else:
            position += 1
    return None


def fingerprint(path: str) -> VideoFingerprint:
    probe = _run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
            "-show_data_hash", "md5",
            "-show_entries",
            "stream=codec_name,profile,level,pix_fmt,width,height,r_frame_rate,"
            "nb_read_packets,extradata_hash",
            "-of", "json", path,
        ],
        what=f"ffprobe ({Path(path).name})",
    )
    streams = json.loads(probe.stdout).get("streams") or []
    if not streams:
        raise VideoError(f"No video stream in {path}.")
    stream = streams[0]
    first_nal = None
    if stream.get("codec_name") == "hevc":
        packet = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
                "-map", "0:v:0", "-c", "copy", "-bsf:v", "hevc_mp4toannexb",
                "-frames:v", "1", "-f", "hevc", "pipe:1",
            ],
            capture_output=True, check=False,
        )
        if packet.returncode == 0:
            first_nal = first_vcl_nal_type(packet.stdout)
    return VideoFingerprint(
        frames=int(stream.get("nb_read_packets", 0)),
        codec=stream.get("codec_name", ""),
        profile=stream.get("profile", ""),
        level=int(stream.get("level", 0)),
        pix_fmt=stream.get("pix_fmt", ""),
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        fps=stream.get("r_frame_rate", ""),
        extradata_md5=stream.get("extradata_hash", ""),
        first_vcl_nal=first_nal,
    )


def check_parts(parts: list[tuple[str, int, VideoFingerprint]]) -> None:
    """Fail unless every part has its planned frames, an IDR first, and common parameters.

    `parts` is (name, planned frames, fingerprint) in playback order. Files with
    different parameter sets cannot be joined by stream copy: the joined file
    carries one of them and a decoder glitches on the others.
    """
    if not parts:
        raise VideoError("Nothing to join.")
    reference_name, _, reference = parts[0]
    problems = []
    for name, planned, found in parts:
        if found.frames != planned:
            problems.append(f"{name}: {found.frames} frames, planned {planned}")
        if found.codec == "hevc" and found.first_vcl_nal not in _HEVC_IDR_NALS:
            problems.append(f"{name}: first picture is NAL type {found.first_vcl_nal}, not an IDR")
        if found.codec_parameters() != reference.codec_parameters():
            problems.append(
                f"{name}: codec parameters differ from {reference_name} "
                f"({found.codec_parameters()} vs {reference.codec_parameters()})"
            )
    if problems:
        raise VideoError("Segments cannot be joined: " + "; ".join(problems))


def write_concat_list(list_path: str, files: list[tuple[str, int]], fps: Fraction) -> None:
    """ffconcat list with an exact duration per file, so joins never drift."""
    lines = ["ffconcat version 1.0"]
    for path, frames in files:
        escaped = path.replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
        lines.append(f"duration {float(Fraction(frames) / fps):.9f}")
    Path(list_path).write_text("\n".join(lines) + "\n")


def join_command(
    list_path: str,
    output_path: str,
    *,
    source_path: str | None = None,
    video_offset: float = 0.0,
) -> list[str]:
    """Join the parts by stream copy and, with a source, carry over its other streams.

    With `source_path`, audio, subtitles, attachments, chapters and metadata are
    stream-copied from it into Matroska. `video_offset` is how much later than
    the container start the source's video began, so the new video keeps the
    same position against the audio.
    """
    offset = ["-itsoffset", f"{video_offset:.6f}"] if video_offset > 0.0005 else []
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *offset, "-f", "concat", "-safe", "0", "-i", list_path,
    ]
    if source_path is None:
        return [*command, "-map", "0:v:0", "-c", "copy", output_path]
    return [
        *command, "-i", source_path,
        "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?", "-map", "1:t?",
        "-map_metadata", "1", "-map_chapters", "1",
        "-c", "copy", output_path,
    ]


def join_parts(
    plan: Plan,
    parts_dir: str,
    output_path: str,
    info: SourceInfo,
    *,
    source_path: str | None,
    extension: str = ".mp4",
) -> dict:
    """Verify the planned parts, join them, and verify the result. Returns a report."""
    order = plan.concat_order()
    files = [(str(Path(parts_dir) / f"{stem}{extension}"), frames) for stem, frames in order]
    missing = [path for path, _ in files if not Path(path).exists()]
    if missing:
        raise VideoError(f"Missing parts: {', '.join(Path(p).name for p in missing[:8])}")
    prints = [(Path(path).name, frames, fingerprint(path)) for path, frames in files]
    check_parts(prints)

    list_path = str(Path(parts_dir) / "concat.ffconcat")
    write_concat_list(list_path, files, info.fps)
    # Only a full-length job lines up with the source's audio and subtitles.
    whole = plan.first_frame == 0 and plan.last_frame == info.frames
    offset = max(0.0, info.video_start - info.format_start) if whole else 0.0
    _run(
        join_command(list_path, output_path,
                     source_path=source_path if whole else None, video_offset=offset),
        what="join",
    )
    joined = fingerprint(output_path)
    if joined.frames != plan.total_frames:
        raise VideoError(f"Joined file has {joined.frames} frames, planned {plan.total_frames}.")
    expected_fps = f"{info.fps.numerator}/{info.fps.denominator}"
    if Fraction(joined.fps) != info.fps:
        raise VideoError(f"Joined file reports {joined.fps} fps, source is {expected_fps}.")
    return {
        "output": output_path,
        "frames": joined.frames,
        "fps": joined.fps,
        "codec": joined.codec,
        "profile": joined.profile,
        "level": joined.level,
        "pix_fmt": joined.pix_fmt,
        "size": f"{joined.width}x{joined.height}",
        "parts": len(files),
        "seams": len(plan.seams),
        "carried_source_streams": bool(whole and source_path),
    }


def stream_summary(path: str) -> list[dict]:
    """Codec, colour tags and timing of every stream, for the final report."""
    probe = _run(
        [
            "ffprobe", "-v", "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,profile,level,pix_fmt,width,height,"
            "r_frame_rate,start_time,duration,color_space,color_primaries,color_transfer,"
            "color_range:stream_tags=language",
            "-of", "json", path,
        ],
        what="ffprobe (summary)",
    )
    return json.loads(probe.stdout).get("streams", [])


def file_digest(path: str, *, limit: int = 8 * 1024 * 1024) -> str:
    """Cheap identity of a large file: its size plus a hash of its first bytes."""
    target = Path(path)
    digest = hashlib.sha256(str(target.stat().st_size).encode())
    with target.open("rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()[:20]


def segment_files(segment: Segment) -> dict[str, str]:
    """File names a segment writes, relative to the job's segments directory."""
    stem = segment.name()
    return {
        "video": f"{stem}.mp4",
        "head": f"{stem}.head.npy",
        "tail": f"{stem}.tail.npy",
        "marker": f"{stem}.ok",
    }


def seam_files(seam: Seam) -> dict[str, str]:
    stem = seam.name()
    return {"video": f"{stem}.mp4", "marker": f"{stem}.ok"}


def canvas_deliver(canvas: Canvas) -> tuple[int, int] | None:
    """The (width, height) the encoder must resample to, or None when it already matches."""
    if (canvas.deliver_w, canvas.deliver_h) == (canvas.out_w, canvas.out_h):
        return None
    return canvas.deliver_w, canvas.deliver_h
