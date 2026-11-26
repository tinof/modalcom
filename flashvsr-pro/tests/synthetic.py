"""A synthetic source whose frames carry their own number, for the video tests.

Three grey blocks in the top-left corner spell the frame index in base 8, so a
lost, repeated or reordered frame is visible in the decoded result. The picture
changes completely at frame CUT, which gives the planner one real scene change.
"""

from __future__ import annotations

import shutil
import subprocess
from fractions import Fraction

import numpy as np
import pytest

WIDTH, HEIGHT, FRAMES, FPS = 320, 180, 250, Fraction(25)
CUT = 100  # first frame of the second "scene"
BLOCK = 32
LEVELS = [20 + 30 * step for step in range(8)]


def have_ffmpeg() -> bool:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return False
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return "libx265" in encoders and "libx264" in encoders


needs_ffmpeg = pytest.mark.skipif(not have_ffmpeg(), reason="needs ffmpeg with libx264 and libx265")


def source_frame(index: int) -> np.ndarray:
    """A frame whose three top-left blocks spell `index` in base 8."""
    frame = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    ramp = np.linspace(40, 200, WIDTH, dtype=np.uint8)
    if index < CUT:
        frame[:] = ramp[None, :, None]
    else:
        frame[..., 0] = 200 - ramp[None, :] // 2
        frame[..., 1] = 60
        frame[..., 2] = ramp[None, :]
    for digit in range(3):
        level = LEVELS[(index // 8**digit) % 8]
        frame[:BLOCK, digit * BLOCK:(digit + 1) * BLOCK] = level
    return frame


def read_index(frame: np.ndarray, scale: int = 1) -> int:
    size = BLOCK * scale
    value = 0
    for digit in range(3):
        block = frame[size // 4:size * 3 // 4, digit * size + size // 4:digit * size + size * 3 // 4]
        value += int(round((float(block.mean()) - 20) / 30)) * 8**digit
    return value


def decode_all(path: str, width: int, height: int) -> np.ndarray:
    raw = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-map", "0:v:0",
            "-fps_mode", "passthrough",
            "-vf", "scale=in_color_matrix=bt709:in_range=tv:flags=bicubic+accurate_rnd+full_chroma_int",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


def make_source(folder) -> str:
    """H.264 with B-frames and a 2 s GOP, AAC audio and a subtitle track, in Matroska."""
    subtitles = folder / "subs.srt"
    subtitles.write_text("1\n00:00:01,000 --> 00:00:03,000\nfirst line\n\n2\n00:00:06,000 --> 00:00:08,000\nsecond line\n")
    path = folder / "source.mkv"
    frames = b"".join(source_frame(index).tobytes() for index in range(FRAMES))
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", "25", "-i", "pipe:0",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
            "-i", str(subtitles),
            "-map", "0:v", "-map", "1:a", "-map", "2:s",
            "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=bicubic+accurate_rnd+full_chroma_int,"
                   "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv",
            "-c:v", "libx264", "-preset", "fast", "-crf", "10", "-g", "50", "-bf", "2",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-c:s", "srt",
            "-metadata:s:a:0", "language=fin", "-metadata:s:s:0", "language=fin",
            str(path),
        ],
        input=frames, check=True,
    )
    return str(path)
