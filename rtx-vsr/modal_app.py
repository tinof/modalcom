import contextlib
import dataclasses
import math
import mmap
import os
import queue
import re
import resource
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from enum import Enum
from fractions import Fraction
from pathlib import Path

import modal

# MODAL_APP_NAME lets a benchmark variant (e.g. a different GPU) deploy alongside the
# production app instead of replacing it. Both share the jobs Volume, which is safe
# because job ids are unique.
app = modal.App(os.getenv("MODAL_APP_NAME", "rtx-media-upscaler"))

# RTX PRO 6000 (Blackwell) is the only Modal GPU with 9th-gen NVENC; the
# data-center Blackwell parts (B200/B300) have no NVENC engine at all.
GPU_TYPE = os.getenv("MODAL_GPU", "RTX-PRO-6000")
# Both knobs exist so a benchmark variant can sweep them without editing this file.
# See CLAUDE.md section 6 for the measurements behind the defaults.
WORKER_CONCURRENCY = int(os.getenv("MODAL_WORKER_CONCURRENCY", "1"))
WORKER_CPU = float(os.getenv("MODAL_WORKER_CPU", "6"))
# Benchmarks only: pinning the pool to one container is the only way to measure input
# concurrency, since the autoscaler otherwise scales out rather than packing a container.
WORKER_MAX_CONTAINERS = int(os.getenv("MODAL_WORKER_MAX_CONTAINERS", "0")) or None
# Encoder quality point, chosen by measurement (2026-10-03, see docs/PERF-HISTORY.md):
# 1080p->4K BBC Blu-ray, same VSR frames, scored with VMAF 4K against a lossless encode.
# The encoder is the pipeline's binding stage (VSR alone is 3.5 ms/frame). One NVENC
# session, 4K encode only: P4+qres+UHQ 65 fps, P5+fullres+UHQ 38.5, P6 27.5, P7 ~24;
# end to end at these defaults ~36 fps.
#   * UHQ beats HQ: +1.7 VMAF at 19% fewer bits (P5, CQ 24). UHQ also turns on NVENC's
#     temporal filter and deeper lookahead, which PyNvVideoCodec cannot set separately.
#   * P5+fullres beats P4+qres by ~1.1 VMAF at equal bitrate.
#   * P6/P7 add 0.06-0.15 VMAF over P5 for ~30% more encode time: not worth it.
# Both encode paths read these, so the piped fallback stays directly comparable.
NVENC_PRESET = os.getenv("MODAL_NVENC_PRESET", "p5")
# fullres = two-pass at full resolution, qres = two-pass at quarter resolution (cheaper).
NVENC_MULTIPASS = os.getenv("MODAL_NVENC_MULTIPASS", "fullres")
# uhq or hq. ffmpeg spells them -tune uhq/hq; PyNvVideoCodec wants uhq/high_quality.
NVENC_TUNING = os.getenv("MODAL_NVENC_TUNING", "uhq")
# CQ is a file-size choice once the bitrate ceiling below is lifted: on the reference
# Blu-ray CQ 18 ~= 80 Mbps (VMAF 94.3) and CQ 24 ~= 35 Mbps (VMAF 91.3). 20 sits between.
NVENC_CQ = os.getenv("MODAL_NVENC_CQ", "20")
# Without an explicit level the driver picks one per session -- 5.0 or 6.0 for the same
# input -- and each brings a different default VBV ceiling (~20 vs ~48 Mbps). That was
# the "same job, 2.4x bigger file" bug. 5.1 High tier allows 160 Mbps and is the level
# 4K TVs and players decode. It only fits outputs up to 8,912,896 luma samples (4096x2176);
# _hevc_level() leaves larger outputs to the driver. "" leaves every output to the driver.
NVENC_LEVEL = os.getenv("MODAL_NVENC_LEVEL", "5.1")
# The VBV ceiling CQ is allowed to reach, in Mbit/s (buffer = 1 s at that rate, which
# also fits Level 5.0 High's CPB limit). PyNvVideoCodec
# zeroes maxBitRate whenever `cq` is set and the driver then substitutes its own ~32 Mbps
# ceiling at 5.1 -- which held every encode below what CQ asked for (CQ 18 and CQ 24 came
# out the same size). It is restored with Reconfigure() before the first frame.
NVENC_MAX_MBPS = int(os.getenv("MODAL_NVENC_MAX_MBPS", "100"))
NVENC_AQ_STRENGTH = os.getenv("MODAL_NVENC_AQ_STRENGTH", "10")
NVENC_BFRAMES = os.getenv("MODAL_NVENC_BFRAMES", "5")
NVENC_REFS = os.getenv("MODAL_NVENC_REFS", "5")
# Split-Frame Encoding stripes each frame across the GPU's several NVENC engines. A
# single session is otherwise capped at one engine's throughput no matter how many the
# card has (NVENC Application Note, SDK 13). HEVC/AV1 only. 0=auto, 2/3/4=forced N-way,
# 15=off. Measured standalone at 4K with production args: off 20.2 fps, 3-way 27.9.
NVENC_SFE = os.getenv("MODAL_NVENC_SFE", "")
# Feed NVENC straight from CUDA memory (PyNvVideoCodec) instead of piping 4K rawvideo to
# a second ffmpeg. Measured on RTX PRO 6000 at 4K: 35 fps vs 19.5 for the identical
# quality point (p6 + fullres), because the 24.9 MB/frame host round-trip disappears.
# This is what NVEncC does and why a consumer 4070 beats our pipeline.
#
# Was briefly disabled on 2026-08-18 because it emitted alternating-column striping on
# every frame. That was not this path being wrong -- it was the missing
# torch.cuda.synchronize() before Encode(), fixed in _flush below. Turning this off is a
# ~6x throughput cut, so do not reach for it as a workaround; run
# scripts/probe_p010_encode.py instead, which reproduces the class of bug in minutes.
GPU_ENCODER = os.getenv("MODAL_GPU_ENCODER", "1") == "1"
# The last host round-trip: decode in-process with NVDEC so frames reach VSR as CUDA
# tensors instead of crossing a rawvideo pipe from a decode subprocess. The pipe carries
# *input*-sized frames and already measures near-free (decode-wait ~1%), so this buys
# architecture completeness rather than throughput. Falls back to the piped decoder for
# any input NVDEC cannot handle.
GPU_DECODER = os.getenv("MODAL_GPU_DECODER", "1") == "1"
# Parallel NVENC sessions for one video. One session at the quality point above is the
# pipeline's ceiling (~38 fps at 4K), and the RTX PRO 6000 has four NVENC engines that
# independent sessions use concurrently. Inputs of at least SEGMENTED_MIN_SECONDS are
# cut at source keyframes into segments of up to SEGMENT_SECONDS; segment i goes to
# session i mod N, each segment is muxed to its own mp4 on local disk, and ffmpeg's
# concat demuxer joins them. Measured 2026-10-04 (docs/PERF-HISTORY.md): 4 pipelines
# 82.8 fps against 40.0 for one, all four streams pixel-clean.
#
# Static assignment, not a work queue, keeps the output byte-identical run to run: a
# session's rate control carries across its segments, so which segments it saw must not
# depend on scheduling. Segmenting also bounds memory: FFmpegMuxer holds every packet in
# RAM until Finalize(), which for one muxer per film was ~78 Mbps x duration.
NVENC_SESSIONS = max(1, int(os.getenv("MODAL_NVENC_SESSIONS", "4")))
SEGMENT_SECONDS = float(os.getenv("MODAL_SEGMENT_SECONDS", "30"))
# Shorter inputs keep the unsegmented path: four CreateEncoder calls cost ~3 s, which a
# short clip does not earn back, and the in-process NVDEC path stays in use for them.
SEGMENTED_MIN_SECONDS = float(os.getenv("MODAL_SEGMENTED_MIN_SECONDS", "30"))
# CreateEncoder with an explicit level fails nvEncInitializeEncoder with error 8 at
# random -- 3/30 on a fresh container, 5/30 with a VSR effect loaded, 3/20 with three
# sessions alive (2026-10-04) -- and the next attempt almost always succeeds (one double
# failure in 80). Six attempts leave ~1e-5 before the no-level fallback.
NVENC_CREATE_ATTEMPTS = 6
MAX_PIXELS = 1024 * 1024 * 16
# Frames per inference batch, capped independently of output size.
#
# 1 is the measured optimum, not a workaround. Batching briefly looked like the cause of
# frame corruption; the real cause was the missing encoder sync in _flush, and with that
# fixed a batch of 2 is equally clean (verified 2026-08-19: Laplacian 1.07, 301/301
# frames). Batching just does not pay -- warm on the reference clip, batch=1 measured
# 56.0 fps against batch=2's 53.2, because a larger batch makes the encoder wait on a
# longer run of inference rather than overlapping with it.
MAX_BATCH = int(os.getenv("MODAL_MAX_BATCH", "1")) or None
MAX_OUTPUT_EDGE = 16384
MAX_VSR_EDGE = 15360
# Largest output TrueHDR was verified at (scripts/probe_truehdr.py `sizes`, 2026-10-04).
MAX_HDR_OUTPUT_PIXELS = 8192 * 4320
JOBS_DIR = Path("/jobs")

# RTX Video Super Resolution is an NGX feature. The nvidia-vfx wheel bundles the
# NGX *snippet* but its runtime dlopens the driver-side libnvidia-ngx.so.1, which
# Modal's container runtime does not inject. Without it NvVFX_Load fails with
# "effect has not been properly initialized" (NVCV_ERR_INITIALIZATION, -12) on
# every GPU type. So we install that one library from the matching driver package.
# Keep this in sync with the host driver reported by nvidia-smi.
DRIVER_VERSION = "580.95.05"
DRIVER_RUN_URL = (
    f"https://us.download.nvidia.com/XFree86/Linux-x86_64/{DRIVER_VERSION}"
    f"/NVIDIA-Linux-x86_64-{DRIVER_VERSION}.run"
)

# Pinned to a month-end autobuild: BtbN prunes the daily tags but keeps every month-end
# release (back to 2024-11 as of 2026-10), whereas the `latest` tag is rebuilt daily and
# would silently swap the NVENC SDK under the next image build. The checksum makes any
# change to the artifact fail the build instead of degrading to libx264 at runtime.
FFMPEG_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-09-30-13-08"
    "/ffmpeg-n8.1.3-9-g29e619e767-linux64-gpl-8.1.tar.xz"
)
FFMPEG_SHA256 = "97ce978979194b5cf7e06a5e68020dbdaa7a4f3294c5452b6a1bc347100dbb79"

# The web tier only moves bytes, so it stays off the GPU image entirely.
web_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "fastapi[standard]==0.141.1",
    "python-multipart==0.0.32",
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "xz-utils", "libsm6", "libxext6", "libxrender1", "libglib2.0-0")
    .run_commands(
        # Driver-side NGX library (see DRIVER_VERSION above).
        f"curl -fsSL -o /tmp/nv.run {DRIVER_RUN_URL}",
        "sh /tmp/nv.run --extract-only --target /tmp/nvx",
        f"cp /tmp/nvx/libnvidia-ngx.so.{DRIVER_VERSION} /usr/lib/x86_64-linux-gnu/",
        f"ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ngx.so.{DRIVER_VERSION}"
        " /usr/lib/x86_64-linux-gnu/libnvidia-ngx.so.1",
        "ldconfig",
        "rm -rf /tmp/nv.run /tmp/nvx",
        # Debian's ffmpeg 5.x works but has no av1_nvenc. Use the pinned n8.1 build.
        # Do NOT move this to "master-latest": those builds link nv-codec-headers
        # for Video Codec SDK 13.1.15, which requires driver 610.0+. On this host
        # (580.95.05) every NVENC encoder then fails to initialise and the worker
        # silently falls back to CPU libx264. n8.1 targets an SDK the driver supports.
        f"curl -fsSL -o /tmp/ff.tar.xz {FFMPEG_URL}",
        f"echo '{FFMPEG_SHA256}  /tmp/ff.tar.xz' | sha256sum -c -",
        "mkdir -p /opt/ffmpeg",
        "tar -xJf /tmp/ff.tar.xz -C /opt/ffmpeg --strip-components=1",
        "cp /opt/ffmpeg/bin/ffmpeg /opt/ffmpeg/bin/ffprobe /usr/local/bin/",
        "rm -rf /tmp/ff.tar.xz /opt/ffmpeg",
        "ffmpeg -version",
    )
    .uv_pip_install(
        "numpy==2.5.2",
        "opencv-python-headless==5.0.0.93",
        "pillow==12.3.0",
        "torch==2.13.0",
        "nvidia-vfx==0.2.0.0",
        "PyNvVideoCodec==2.2.3",
    )
    # NVENC_PRESET is read again at container import, where the deploying shell's
    # environment does not exist. Bake it into the image or the container silently
    # falls back to the default and a preset experiment measures nothing.
    .env({
        "MODAL_NVENC_PRESET": NVENC_PRESET,
        "MODAL_NVENC_MULTIPASS": NVENC_MULTIPASS,
        "MODAL_NVENC_CQ": NVENC_CQ,
        "MODAL_NVENC_TUNING": NVENC_TUNING,
        "MODAL_NVENC_LEVEL": NVENC_LEVEL,
        "MODAL_NVENC_MAX_MBPS": str(NVENC_MAX_MBPS),
        "MODAL_NVENC_AQ_STRENGTH": NVENC_AQ_STRENGTH,
        "MODAL_NVENC_BFRAMES": NVENC_BFRAMES,
        "MODAL_NVENC_REFS": NVENC_REFS,
        "MODAL_NVENC_SFE": NVENC_SFE,
        "MODAL_GPU_ENCODER": "1" if GPU_ENCODER else "0",
        "MODAL_GPU_DECODER": "1" if GPU_DECODER else "0",
        "MODAL_MAX_BATCH": str(MAX_BATCH or 0),
        "MODAL_NVENC_SESSIONS": str(NVENC_SESSIONS),
        "MODAL_SEGMENT_SECONDS": str(SEGMENT_SECONDS),
        "MODAL_SEGMENTED_MIN_SECONDS": str(SEGMENTED_MIN_SECONDS),
    })
)

jobs_volume = modal.Volume.from_name("rtx-upscaler-jobs", create_if_missing=True)


class UpscaleType(str, Enum):
    SCALE_BY = "scale by multiplier"
    TARGET_DIMENSIONS = "target dimensions"


# nvvfx.VideoSuperRes.QualityLevel groups its models into families. The upscale families
# change resolution; DENOISE/DEBLUR are same-resolution restoration passes, which is why
# they are only valid as `preprocess`. All 19 were probed working on RTX-PRO-6000 /
# driver 580.95.05 — see the verification protocol in CLAUDE.md.
UPSCALE_QUALITIES = frozenset(
    {"BICUBIC", "LOW", "MEDIUM", "HIGH", "ULTRA"}
    | {f"HIGHBITRATE_{level}" for level in ("LOW", "MEDIUM", "HIGH", "ULTRA")}
)
SAME_RES_QUALITIES = frozenset(
    {f"DENOISE_{level}" for level in ("LOW", "MEDIUM", "HIGH", "ULTRA")}
    | {f"DEBLUR_{level}" for level in ("LOW", "MEDIUM", "HIGH", "ULTRA")}
)


class UpscaleOnlyError(ValueError):
    """Raised when requested dimensions are smaller than input dimensions."""


class OutputDimensionExceededError(ValueError):
    """Raised when requested dimensions exceed the maximum allowed edge."""


class OutputIntegrityError(RuntimeError):
    """Raised when super-resolution output is corrupted (NaN/Inf or collapsed channel)."""


class PlannedDimensions:
    __slots__ = ("output_width", "output_height", "sr_width", "sr_height")

    def __init__(self, output_width: int, output_height: int, sr_width: int, sr_height: int) -> None:
        self.output_width = output_width
        self.output_height = output_height
        self.sr_width = sr_width
        self.sr_height = sr_height

    @property
    def is_hybrid(self) -> bool:
        return (self.output_width, self.output_height) != (self.sr_width, self.sr_height)

    def __repr__(self) -> str:
        return (
            f"PlannedDimensions(output_width={self.output_width}, output_height={self.output_height}, "
            f"sr_width={self.sr_width}, sr_height={self.sr_height})"
        )


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _normalize_resize_type(raw_resize_type: str) -> UpscaleType:
    normalized = raw_resize_type.strip().lower()
    alias_map = {
        UpscaleType.SCALE_BY.value: UpscaleType.SCALE_BY,
        "scale_by": UpscaleType.SCALE_BY,
        "scale-by": UpscaleType.SCALE_BY,
        "scale": UpscaleType.SCALE_BY,
        UpscaleType.TARGET_DIMENSIONS.value: UpscaleType.TARGET_DIMENSIONS,
        "target_dimensions": UpscaleType.TARGET_DIMENSIONS,
        "target-dimensions": UpscaleType.TARGET_DIMENSIONS,
        "target": UpscaleType.TARGET_DIMENSIONS,
    }
    if normalized not in alias_map:
        raise ValueError(f"Unsupported resize type: {raw_resize_type}")
    return alias_map[normalized]


def _normalize_quality(raw_quality: str) -> str:
    """Validate a QualityLevel name without importing nvvfx (the web tier has no GPU image)."""
    normalized = raw_quality.strip().upper().replace("-", "_")
    if normalized not in UPSCALE_QUALITIES and normalized not in SAME_RES_QUALITIES:
        raise ValueError(
            f"Unsupported quality: {raw_quality}. Choose one of "
            f"{sorted(UPSCALE_QUALITIES | SAME_RES_QUALITIES)}."
        )
    return normalized


# SDR->HDR10 output (nvvfx.TrueHDR). The defaults reproduce the reference StaxRip/NVEncC
# settings this feature was built to match:
#   --vpp-ngx-truehdr contrast=102,saturation=102,middlegray=46,maxluminance=680
#   --master-display "G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(6800000,1)"
#   --max-cll "680,300"
# NVEncC passes the four tunables 1:1 to the same NGX feature, with the same ranges.
# The mastering primaries are BT.2020 (not P3) with a D65 white point; the mastering peak
# and MaxCLL follow maxluminance unless given explicitly.
TRUEHDR_DEFAULTS = {"contrast": 102, "saturation": 102, "middlegray": 46, "maxluminance": 680}
TRUEHDR_RANGES = {
    "contrast": (0, 200), "saturation": (0, 200), "middlegray": (10, 100), "maxluminance": (400, 2000),
}
TRUEHDR_KEY_ALIASES = {
    "contrast": "contrast", "saturation": "saturation",
    "middlegray": "middlegray", "middle_gray": "middlegray",
    "maxluminance": "maxluminance", "luminance": "maxluminance", "max_luminance": "maxluminance",
}
DEFAULT_MASTER_PRIMARIES = (8500, 39850, 6550, 2300, 35400, 14600, 15635, 16450)  # G, B, R, WP
DEFAULT_MASTER_MIN_LUMINANCE = 1  # 0.0001 cd/m2
DEFAULT_MAX_FALL = 300
_ON_VALUES = frozenset({"on", "true", "yes", "1", "default"})
_OFF_VALUES = frozenset({"", "off", "false", "no", "0", "none"})


class HdrRequestError(ValueError):
    """An HDR10 request that cannot be honoured for this input (reported as HTTP 400)."""


@dataclasses.dataclass(frozen=True)
class HdrSettings:
    """TrueHDR tunables plus the HDR10 static metadata written into the stream."""

    contrast: int
    saturation: int
    middle_gray: int
    luminance: int
    debanding: bool
    # Mastering display: G, B, R, white point x/y in 0.00002 units (HEVC MDCV order).
    primaries: tuple[int, ...]
    max_mastering_luminance: int  # 0.0001 cd/m2
    min_mastering_luminance: int  # 0.0001 cd/m2
    # Written at encode time. With measure_cll these are only the placeholder (the old
    # static default); patch_cll_sei() replaces them with the measured values after the
    # encode, and they stay in the file only if that patch is refused (with a WARNING).
    max_cll: int
    max_fall: int
    measure_cll: bool  # no explicit max_cll: signal the light levels measured per job

    def describe(self) -> str:
        cll = ("measured" if self.measure_cll
               else f"{self.max_cll}/{self.max_fall}")
        return (
            f"truehdr(contrast={self.contrast}, saturation={self.saturation}, "
            f"middlegray={self.middle_gray}, maxluminance={self.luminance}, "
            f"debanding={'on' if self.debanding else 'off'}; "
            f"master L {self.max_mastering_luminance / 10000:g}/"
            f"{self.min_mastering_luminance / 10000:g} nits, "
            f"MaxCLL/MaxFALL {cll})"
        )


def parse_hdr_settings(truehdr: str | None, master_display: str | None = None,
                       max_cll: str | None = None) -> HdrSettings | None:
    """Validate the HDR form fields without importing nvvfx (the web tier has no GPU image).

    `truehdr` is empty/"off" for SDR output, "on" for the defaults, or NVEncC
    --vpp-ngx-truehdr syntax: "contrast=102,saturation=102,middlegray=46,maxluminance=680",
    plus "debanding=on|off". `master_display` takes the NVEncC/x265 --master-display string
    and `max_cll` "MaxCLL,MaxFALL"; both are only valid together with `truehdr`. An empty
    `max_cll` means "measure it": the job signals the MaxCLL/MaxFALL of its own frames.
    0 is HDR10's "unknown", so either value may be 0.
    """
    raw = (truehdr or "").strip()
    master_raw = (master_display or "").strip()
    cll_raw = (max_cll or "").strip()
    if raw.lower() in _OFF_VALUES:
        if master_raw or cll_raw:
            raise ValueError("master_display and max_cll apply only together with truehdr.")
        return None

    values = dict(TRUEHDR_DEFAULTS)
    debanding = True
    if raw.lower() not in _ON_VALUES:
        for item in raw.split(","):
            key, sep, value = item.partition("=")
            key = key.strip().lower().replace("-", "_")
            value = value.strip().lower()
            if not sep or not key or not value:
                raise ValueError(f"truehdr: expected key=value pairs, got {item.strip()!r}.")
            if key in {"debanding", "deband"}:
                if value not in _ON_VALUES | _OFF_VALUES:
                    raise ValueError(f"truehdr: debanding must be on or off, got {value!r}.")
                debanding = value in _ON_VALUES
                continue
            name = TRUEHDR_KEY_ALIASES.get(key)
            if name is None:
                raise ValueError(
                    f"truehdr: unknown key {key!r}; use contrast, saturation, middlegray, "
                    "maxluminance and debanding."
                )
            try:
                values[name] = int(value)
            except ValueError:
                raise ValueError(f"truehdr: {name} must be an integer, got {value!r}.") from None
    for name, (low, high) in TRUEHDR_RANGES.items():
        if not low <= values[name] <= high:
            raise ValueError(f"truehdr: {name} must be in [{low}, {high}], got {values[name]}.")
    luminance = values["maxluminance"]

    if master_raw:
        compact = "".join(master_raw.split()).upper()
        match = re.fullmatch(
            r"G\((\d+),(\d+)\)B\((\d+),(\d+)\)R\((\d+),(\d+)\)WP\((\d+),(\d+)\)L\((\d+),(\d+)\)",
            compact,
        )
        if match is None:
            raise ValueError(
                "master_display must look like "
                "G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min), as in NVEncC/x265 --master-display."
            )
        numbers = [int(group) for group in match.groups()]
        primaries, (max_l, min_l) = tuple(numbers[:8]), numbers[8:]
        if any(value > 50000 for value in primaries):
            raise ValueError("master_display: chromaticities are in 0.00002 units, at most 50000.")
        if not 0 <= min_l < max_l <= 10000 * 10000:
            raise ValueError(
                "master_display: L(max,min) is in 0.0001 cd/m2 with min < max <= 10000 nits."
            )
    else:
        primaries = DEFAULT_MASTER_PRIMARIES
        max_l, min_l = luminance * 10000, DEFAULT_MASTER_MIN_LUMINANCE

    if cll_raw:
        parts = [part.strip() for part in cll_raw.split(",")]
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError('max_cll must be "MaxCLL,MaxFALL" in nits, e.g. "680,300".')
        cll, fall = (int(part) for part in parts)
        if cll > 65535 or fall > 65535:
            raise ValueError("max_cll: MaxCLL and MaxFALL must be at most 65535.")
        if cll and fall and fall > cll:
            raise ValueError("max_cll: MaxFALL cannot exceed MaxCLL (0 means unknown).")
        measure = False
    else:
        # Placeholder written at encode time and replaced after it (patch_cll_sei).
        cll, fall = luminance, min(DEFAULT_MAX_FALL, luminance)
        measure = True

    return HdrSettings(
        contrast=values["contrast"], saturation=values["saturation"],
        middle_gray=values["middlegray"], luminance=luminance, debanding=debanding,
        primaries=primaries, max_mastering_luminance=max_l, min_mastering_luminance=min_l,
        max_cll=cll, max_fall=fall, measure_cll=measure,
    )


_ANNEXB_START = b"\x00\x00\x00\x01"


def _escape_rbsp(rbsp: bytes) -> bytes:
    """HEVC emulation prevention: a 03 after every 00 00 that precedes a byte <= 03."""
    escaped = bytearray()
    zeros = 0
    for byte in rbsp:
        if zeros >= 2 and byte <= 3:
            escaped.append(3)
            zeros = 0
        escaped.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(escaped)


def _sei_nal(payload_type: int, payload: bytes) -> bytes:
    """One Annex-B prefix SEI NAL unit (type 39) carrying a single SEI message."""
    rbsp = bytes([payload_type, len(payload)]) + payload + b"\x80"  # + rbsp_trailing_bits
    # Emulation prevention over everything after the NAL header. The default minimum
    # mastering luminance (1) is 00 00 00 01 in the payload -- a start code otherwise.
    return _ANNEXB_START + b"\x4e\x01" + _escape_rbsp(rbsp)


def _cll_sei_nal(max_cll: int, max_fall: int) -> bytes:
    """The content light level SEI (payload 144) as an Annex-B NAL unit."""
    return _sei_nal(144, struct.pack(">2H", max_cll, max_fall))


def hdr10_sei_nals(hdr: HdrSettings) -> bytes:
    """Mastering display colour volume (137) and content light level (144), one NAL each.

    Separate NAL units, as x265 writes them; NVENC's own picture-timing SEI must not share
    a NAL unit with other payload types.
    """
    mdcv = struct.pack(
        ">8H2I", *hdr.primaries, hdr.max_mastering_luminance, hdr.min_mastering_luminance
    )
    return _sei_nal(137, mdcv) + _cll_sei_nal(hdr.max_cll, hdr.max_fall)


def insert_sei_before_irap_slice(packet: bytes, sei: bytes) -> bytes:
    """Insert `sei` ahead of the first slice of an IRAP access unit; other packets unchanged.

    The SEI goes after the parameter sets and any SEI NVENC wrote (picture timing), just
    before the first VCL NAL unit, which is where a prefix SEI belongs. Only the NAL
    headers ahead of the first slice are scanned; slice data is never walked in Python.
    IRAP is decided from the NAL type (16-21), so the periodic IDRs and every segment's
    FORCEIDR frame are covered alike.
    """
    index = packet.find(b"\x00\x00\x01")
    while index != -1 and index + 3 < len(packet):
        nal_type = (packet[index + 3] >> 1) & 0x3F
        if nal_type < 32:  # first VCL NAL unit of the access unit
            if not 16 <= nal_type <= 21:
                return packet
            cut = index - 1 if index > 0 and packet[index - 1] == 0 else index
            return packet[:cut] + sei + packet[cut:]
        index = packet.find(b"\x00\x00\x01", index + 3)
    return packet


def cll_sei_values(measured_cll: float, measured_fall: float,
                   placeholder: tuple[int, int]) -> tuple[int, int] | None:
    """The MaxCLL/MaxFALL to write over `placeholder`, or None if no value fits in place.

    Rounded up, clamped to 1..65535, MaxFALL <= MaxCLL. The patch overwrites bytes in
    place, so the escaped NAL must be exactly as long as the placeholder's. Where emulation
    prevention changes the length (MaxCLL 512 / MaxFALL 3 is 02 00 00 03, which gains a
    03), MaxCLL is nudged up one nit at a time: far below the measurement's own precision.
    """
    target = len(_cll_sei_nal(*placeholder))
    cll = min(max(math.ceil(measured_cll), 1), 65535)
    fall = min(max(math.ceil(measured_fall), 1), cll)
    for _ in range(8):
        if len(_cll_sei_nal(cll, fall)) == target:
            return cll, fall
        if cll == 65535:
            break
        cll += 1
    return None


def patch_cll_sei(paths, hdr: HdrSettings, measured_cll: float, measured_fall: float,
                  expected: int) -> tuple[int, int] | None:
    """Overwrite the placeholder CLL SEI in local mp4 intermediates with measured values.

    The encode wrote `hdr.max_cll`/`hdr.max_fall` into every IRAP's CLL SEI before the
    light levels were known. In an mp4 the NAL unit is length-prefixed, so the search
    pattern is the 4-byte big-endian length plus the escaped NAL. Every file is scanned
    first; only if the total number of matches equals `expected` (the SEIs actually
    inserted) is anything written, at the same offsets and the same length. Returns the
    written (MaxCLL, MaxFALL), or None after a WARNING, with the files untouched and the
    placeholder values still in them. Never raises for a refusal: the job still succeeds.
    """
    placeholder = (hdr.max_cll, hdr.max_fall)

    def refuse(cause: str) -> None:
        print(
            f"WARNING: HDR10 MaxCLL/MaxFALL not patched ({cause}); the output keeps the "
            f"static placeholder {placeholder[0]}/{placeholder[1]} nits instead of the measured "
            "values. Pass max_cll explicitly to choose them."
        )

    values = cll_sei_values(measured_cll, measured_fall, placeholder)
    if values is None:
        refuse(f"no value near {measured_cll:.1f}/{measured_fall:.1f} keeps the SEI length")
        return None
    old = _cll_sei_nal(*placeholder)[len(_ANNEXB_START):]
    new = _cll_sei_nal(*values)[len(_ANNEXB_START):]
    if len(old) != len(new):  # cll_sei_values guarantees this; never write otherwise
        refuse("escaped SEI lengths differ")
        return None
    pattern = struct.pack(">I", len(old)) + old
    replacement = struct.pack(">I", len(new)) + new
    if expected <= 0:
        refuse("no CLL SEI was inserted during the encode")
        return None

    hits: list[tuple[str, list[int]]] = []
    for path in paths:
        offsets: list[int] = []
        try:
            with open(path, "rb") as handle, mmap.mmap(
                handle.fileno(), 0, access=mmap.ACCESS_READ
            ) as view:
                at = view.find(pattern)
                while at != -1:
                    offsets.append(at)
                    at = view.find(pattern, at + len(pattern))
        except (OSError, ValueError) as exc:  # ValueError: mmap of an empty file
            refuse(f"cannot scan {Path(path).name}: {exc}")
            return None
        hits.append((os.fspath(path), offsets))
    found = sum(len(offsets) for _, offsets in hits)
    if found != expected:
        refuse(f"found {found} placeholder CLL SEIs in the intermediates, expected {expected}")
        return None

    if replacement != pattern:
        for path, offsets in hits:
            if not offsets:
                continue
            fd = os.open(path, os.O_RDWR)
            try:
                for at in offsets:
                    os.pwrite(fd, replacement, at)
                os.fsync(fd)
            finally:
                os.close(fd)
    return values


SDR_VUI_BSF = ("hevc_metadata=colour_primaries=1:transfer_characteristics=1"
               ":matrix_coefficients=1:video_full_range_flag=0")
# chroma_sample_loc_type=2 (top-left) is what _hdr10_to_p010 actually samples, unlike
# NVEncC, which signals 2 from --chromaloc 2 while its 4:2:0 kernel samples type 0.
HDR_VUI_BSF = ("hevc_metadata=colour_primaries=9:transfer_characteristics=16"
               ":matrix_coefficients=9:video_full_range_flag=0:chroma_sample_loc_type=2")


def _remux_color_args(hdr: HdrSettings | None) -> list[str]:
    """Colour signalling for the final `-c:v copy` remux.

    NVENC writes no colour description, so hevc_metadata rewrites the SPS VUI. HDR also
    sets the output stream's colour fields, which is what makes the mp4 muxer write a
    `colr` (nclx) box -- hevc_metadata alone edits only the bitstream, and players that
    trust the container would treat the file as SDR. The SDR arguments are the old ones exactly, so
    this remux leaves SDR output as it was. This copy remux has no way to attach the side data
    ffmpeg's mp4 muxer would need for `mdcv`/`clli` boxes; the in-band SEI carries them, as it
    does in NVEncC's output.
    """
    if hdr is None:
        return ["-bsf:v", SDR_VUI_BSF]
    return [
        "-bsf:v", HDR_VUI_BSF,
        "-color_primaries", "bt2020", "-color_trc", "smpte2084",
        "-colorspace", "bt2020nc", "-color_range", "tv",
    ]


def _aligned_dimension(size: int) -> int:
    return max(8, round(size / 8) * 8)


def plan_dimensions(
    input_width: int,
    input_height: int,
    resize_type: str,
    scale: float,
    width: int,
    height: int,
    keep_aspect_ratio: bool = True,
) -> PlannedDimensions:
    selected_type = _normalize_resize_type(resize_type)

    if selected_type == UpscaleType.SCALE_BY:
        if scale < 1.0:
            raise ValueError("Scale must be >= 1.0 when resize_type is scale by multiplier.")
        raw_w = input_width * scale
        raw_h = input_height * scale
    else:
        if width < 1 or height < 1:
            raise ValueError("Output dimensions must be positive.")
        if keep_aspect_ratio:
            ratio = min(width / input_width, height / input_height)
            raw_w = input_width * ratio
            raw_h = input_height * ratio
        else:
            raw_w = width
            raw_h = height

    output_width = _aligned_dimension(int(round(raw_w)))
    output_height = _aligned_dimension(int(round(raw_h)))

    if output_width < input_width or output_height < input_height:
        raise UpscaleOnlyError(
            f"Upscaling only: requested output {output_width}x{output_height} is smaller than input {input_width}x{input_height}."
        )

    if output_width > MAX_OUTPUT_EDGE or output_height > MAX_OUTPUT_EDGE:
        raise OutputDimensionExceededError(
            f"Output dimension exceeds maximum supported edge of {MAX_OUTPUT_EDGE} (got {output_width}x{output_height})."
        )

    if output_width > MAX_VSR_EDGE or output_height > MAX_VSR_EDGE:
        sr_scale = min(MAX_VSR_EDGE / output_width, MAX_VSR_EDGE / output_height)
        sr_w = _aligned_dimension(int(round(output_width * sr_scale)))
        sr_h = _aligned_dimension(int(round(output_height * sr_scale)))
        if sr_w > MAX_VSR_EDGE:
            sr_w = (MAX_VSR_EDGE // 8) * 8
        if sr_h > MAX_VSR_EDGE:
            sr_h = (MAX_VSR_EDGE // 8) * 8
    else:
        sr_w = output_width
        sr_h = output_height

    return PlannedDimensions(
        output_width=output_width,
        output_height=output_height,
        sr_width=sr_w,
        sr_height=sr_h,
    )



def _integrity_stats(input_frame, output_frame, valid=None):
    """[valid, 3 input channel means, 3 output channel means] as one CUDA tensor.

    Kept on the GPU so a frame's checks cost one device->host read in total
    (_check_frame_integrity): each .item() is a host sync, and the four segment threads
    share the legacy stream. `valid` defaults to "output is finite"; TrueHDR passes its
    alpha check, since its integer codes cannot be NaN.
    """
    import torch

    if valid is None:
        valid = torch.isfinite(output_frame).all()
    return torch.cat([
        valid.reshape(1).to(torch.float32),
        input_frame.float().mean(dim=(-2, -1))[:3],
        output_frame.float().mean(dim=(-2, -1))[:3],
    ])


def _check_frame_integrity(stats, out_width: int, out_height: int,
                           stage: str = "Super-resolution") -> None:
    """Raise on non-finite output or a collapsed channel; `stats` from _integrity_stats.

    `stats` may hold several stages' 7-value blocks back to back (VSR, then TrueHDR),
    read back in one transfer; `stage` then names them in order, separated by "+".
    """
    import torch

    values = stats.tolist() if torch.is_tensor(stats) else list(stats)
    stages = stage.split("+")
    channel_names = ["Red", "Green", "Blue"]
    for block, name in enumerate(stages):
        valid, *means = values[7 * block:7 * block + 7]
        if not valid:
            what = "invalid RGB10A2 words (alpha bits not 3)" if name == "TrueHDR" else \
                "non-finite values (NaN or Inf)"
            raise OutputIntegrityError(f"{name} produced {what} at {out_width}x{out_height}.")
        for c in range(3):
            in_m, out_m = means[c], means[3 + c]
            if in_m >= 0.01 and (out_m < 0.001 or out_m / (in_m + 1e-7) < 0.01):
                raise OutputIntegrityError(
                    f"{name} output corrupt: {channel_names[c]} channel collapsed "
                    f"(input mean {in_m:.4f}, output mean {out_m:.4f}) at {out_width}x{out_height}."
                )


def _batch_size_for_output(output_width: int, output_height: int) -> int:
    out_pixels = output_width * output_height
    size = max(1, MAX_PIXELS // out_pixels)
    return min(size, MAX_BATCH) if MAX_BATCH else size


def _encoder_works(encoder: str, extra_args: tuple[str, ...] = ()) -> bool:
    """One-frame probe of an encoder, optionally with the extra options we want to use."""
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


# Quality options shared by both NVENC encoders. These mirror the settings the reference
# NVEncC pipeline uses: VBR with a quality target, full-resolution two-pass, spatial and
# temporal AQ, a deep lookahead, and B-frames with a middle B-reference. 9th-gen NVENC
# (RTX PRO 6000, Video Codec SDK 13.0 / driver 570+) supports all of them; an encoder that
# does not will fail loudly through the ffmpeg stderr tail rather than degrade silently.
NVENC_QUALITY_ARGS = [
    "-preset", NVENC_PRESET,
    "-rc", "vbr", "-cq", NVENC_CQ, "-b:v", "0",
    # Same ceiling as the GPU path; without -maxrate ffmpeg leaves maxBitRate at 0 too.
    *(["-maxrate", f"{NVENC_MAX_MBPS}M", "-bufsize", f"{NVENC_MAX_MBPS}M"]
      if NVENC_MAX_MBPS > 0 else []),
    "-multipass", NVENC_MULTIPASS,
    "-spatial-aq", "1", "-temporal-aq", "1", "-aq-strength", NVENC_AQ_STRENGTH,
    "-rc-lookahead", "32",
    "-bf", NVENC_BFRAMES, "-b_ref_mode", "middle",
]


def _hevc_level(width: int, height: int) -> str | None:
    """NVENC_LEVEL for outputs that fit level 5.x (up to 4096x2176), else None.

    Above 4K the driver picks the level (it chose 6.0 or 6.2 at 8K; with the ceiling
    restored that no longer changes the file size). With the VBV buffer at 1.6x the
    ceiling, a session at a different resolution than the previous job in the same warm
    container failed nvEncInitializeEncoder with error 8 (4K->8K and 8K->4K); a 1 s buffer
    fixed it for 4K, 1620p and 8K in both orders (2026-10-03).
    """
    if NVENC_LEVEL and width * height <= 8_912_896:  # level 5.x MaxLumaPs
        return NVENC_LEVEL
    return None


def _ffmpeg_encode_command(
    encoder: str,
    output_width: int,
    output_height: int,
    fps: float,
    audio_source_path: str,
    output_path: str,
    tune: str | None = None,
    refs: bool = False,
) -> list[str]:
    tune_args = ["-tune", tune] if tune else []
    refs_args = ["-refs", NVENC_REFS] if refs else []
    # Split-Frame Encoding is HEVC/AV1 only, so it is not added for h264_nvenc.
    sfe_args = ["-split_encode_mode", NVENC_SFE] if NVENC_SFE else []
    if encoder == "hevc_nvenc":
        # 10-bit Main 10 even from 8-bit sources: the extra encode precision costs nothing
        # on NVENC and keeps VSR's smooth gradients from banding.
        level = _hevc_level(output_width, output_height)
        codec_args = [
            "-c:v", encoder, *NVENC_QUALITY_ARGS, *tune_args, *refs_args, *sfe_args,
            "-profile:v", "main10", "-tier", "high", "-pix_fmt", "p010le",
            *(["-level", level] if level else []),
        ]
    elif encoder.endswith("_nvenc"):
        codec_args = ["-c:v", encoder, *NVENC_QUALITY_ARGS, *tune_args, *refs_args,
                      "-pix_fmt", "yuv420p"]
    else:
        codec_args = ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
                      "-pix_fmt", "yuv420p"]
    # Rational fps avoids float drift for NTSC rates like 29.97/23.976.
    fps_str = str(Fraction(fps).limit_denominator(1001))
    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{output_width}x{output_height}", "-r", fps_str,
        "-i", "pipe:0",
        "-i", audio_source_path,
        "-map", "0:v:0",
        # Raw RGB carries no color metadata, so swscale would convert with its bt601
        # default while players read HD output as bt709 — a visible hue shift. Convert
        # with bt709 explicitly, then setparams tags primaries and transfer as well:
        # the -colorspace/-color_primaries output options alone left both "unknown" in
        # the encoded stream, since NVENC writes its VUI from the frame properties.
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,"
               "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        *codec_args,
        *_audio_args(audio_source_path),
        "-shortest",
        "-movflags", "+faststart",
        output_path,
    ]


def _put_until_stopped(target: queue.Queue, item, stop: threading.Event) -> bool:
    """Put into a bounded queue, giving up once `stop` is set.

    A plain put() blocks forever when the consumer has died mid-job: the reader thread
    then stays alive in the warm container, pinning the NVDEC session and its queued
    CUDA frames until the container scales down.
    """
    while not stop.is_set():
        try:
            target.put(item, timeout=0.5)
            return True
        except queue.Full:
            continue
    return False


def _stop_reader(reader, frame_queue: queue.Queue, stop: threading.Event) -> None:
    """Release a reader thread after the consumer stops, whether it finished or raised."""
    stop.set()
    with contextlib.suppress(queue.Empty):
        while True:
            frame_queue.get_nowait()
    if reader is not None:
        reader.join(timeout=10)
        if reader.is_alive():
            print("WARNING: video reader thread did not exit within 10s of the job ending.")


# Audio codecs the mp4 muxer accepts as-is. Anything else (DTS, TrueHD, PCM, Vorbis, ...)
# is re-encoded to AAC, per track, with a warning.
MP4_AUDIO_COPY_CODECS = frozenset({"aac", "ac3", "eac3", "opus", "alac", "mp3", "flac"})


def _audio_args(source_path: str, input_index: int = 1) -> list[str]:
    """Map every audio track of input `input_index` into the output, copying when possible.

    Output contract (changed 2026-10-03): all audio tracks are kept, and each one is
    stream-copied unless mp4 cannot carry its codec, in which case only that track is
    re-encoded to AAC 192k. Before, only the first track was kept.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=codec_name", "-of", "csv=p=0", source_path,
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print(
            "WARNING: could not probe audio codecs "
            f"({result.stderr.strip()[:300]}); re-encoding all audio tracks to AAC."
        )
        return ["-map", f"{input_index}:a?", "-c:a", "aac", "-b:a", "192k"]
    codecs = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not codecs:
        return []
    args = ["-map", f"{input_index}:a"]
    reencoded = []
    for index, codec in enumerate(codecs):
        if codec in MP4_AUDIO_COPY_CODECS:
            args += [f"-c:a:{index}", "copy"]
        else:
            args += [f"-c:a:{index}", "aac", f"-b:a:{index}", "192k"]
            reencoded.append(f"#{index} {codec}")
    if reencoded:
        print(
            "WARNING: mp4 cannot carry these audio tracks as-is, re-encoding them to AAC "
            f"192k: {', '.join(reencoded)}."
        )
    return args


def _probe_video_metadata(input_path: str) -> tuple[int, int, float]:
    """Read width/height/fps with ffprobe; the decoder is ffmpeg, so ask ffmpeg."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=0", input_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"Failed to probe uploaded video: {result.stderr.strip()[:500]}")

    fields = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        fields[key.strip()] = value.strip()

    try:
        width = int(fields["width"])
        height = int(fields["height"])
    except (KeyError, ValueError) as exc:
        raise ValueError("Uploaded video has no decodable video stream.") from exc

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
        fps = 30.0
    return width, height, fps


def _rgb_to_p010(rgb):
    """(H,W,3) float [0,1] CUDA tensor -> P010 (H*3//2, W) uint16 CUDA tensor.

    BT.709, limited (TV) range, 10-bit values left-shifted into the high bits, which is
    what P010 means. This replaces ffmpeg's CPU swscale rgb24->p010 conversion; doing it
    here is what lets the frame go straight from VSR into NVENC without touching the host.
    Verified against a decode round-trip: max channel error 3/255 at cq 18.
    """
    import torch

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) / 1.8556
    cr = (r - y) / 1.5748

    y10 = (64.0 + 876.0 * y).round().clamp(0, 1023)
    # 4:2:0 chroma by 2x2 box average.
    cb10 = (512.0 + 896.0 * cb.unfold(0, 2, 2).unfold(1, 2, 2).mean(dim=(-2, -1))).round().clamp(0, 1023)
    cr10 = (512.0 + 896.0 * cr.unfold(0, 2, 2).unfold(1, 2, 2).mean(dim=(-2, -1))).round().clamp(0, 1023)

    height, width = y10.shape
    # torch has no uint16 left-shift on CUDA, so scale by 64 (<<6) in int32 and cast last.
    packed = torch.empty((height * 3 // 2, width), dtype=torch.int32, device=rgb.device)
    packed[:height] = y10.to(torch.int32)
    uv = packed[height:].view(height // 2, width // 2, 2)
    uv[..., 0] = cb10.to(torch.int32)
    uv[..., 1] = cr10.to(torch.int32)
    return (packed * 64).to(torch.uint16)


# Packed R10G10B10A2 words (nvvfx RGB10A2 and TrueHDR output; also ffmpeg's x2bgr10le):
# R in bits 0-9, G 10-19, B 20-29, alpha 30-31. torch has few uint32 ops on CUDA, so the
# bit work runs on an int32 view: alpha=3 sets the sign bit, which the masks drop again.
RGB10A2_ALPHA = -0x40000000  # 3 << 30 as int32


def _unpack_rgb10a2(packed):
    """(H,W) uint32 RGB10A2 CUDA tensor -> (3,H,W) float32 in [0,1] (code / 1023)."""
    import torch

    words = packed.view(torch.int32)
    channels = [(words >> shift) & 0x3FF for shift in (0, 10, 20)]
    return torch.stack(channels).to(torch.float32).div_(1023.0)


def _rgb10a2_alpha_ok(packed):
    """0-d CUDA bool tensor: every word carries alpha 3 (a wrong layout fails loudly)."""
    import torch

    return (((packed.view(torch.int32) >> 30) & 3) == 3).all()


def _pq_nits_table(device):
    """1024-entry ST 2084 EOTF: full-range 10-bit PQ code -> cd/m2."""
    import torch

    m1, m2 = 2610 / 16384, 2523 / 4096 * 128
    c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    e = torch.arange(1024, dtype=torch.float64) / 1023.0
    p = e.pow(1 / m2)
    nits = 10000.0 * ((p - c1).clamp(min=0) / (c2 - c3 * p)).pow(1 / m1)
    return nits.to(torch.float32).to(device)


class _HdrLightLevels:
    """Running MaxCLL / MaxFALL of the encoded frames, kept on the GPU (no host syncs).

    One instance per worker thread; the totals are combined after the threads finish,
    because four threads updating one tensor could lose updates.
    """

    def __init__(self, device) -> None:
        import torch

        self.table = _pq_nits_table(device)
        self.max_code = torch.zeros((), dtype=torch.int32, device=device)
        self.max_fall = torch.zeros((), dtype=torch.float32, device=device)

    def update(self, red, green, blue) -> None:
        """One frame's (H,W) int32 PQ code planes."""
        import torch

        brightest = torch.maximum(torch.maximum(red, green), blue)
        torch.maximum(self.max_code, brightest.amax(), out=self.max_code)
        torch.maximum(self.max_fall, self.table[brightest].mean(), out=self.max_fall)

    @staticmethod
    def combine(levels: list["_HdrLightLevels"]) -> tuple[float, float]:
        """(MaxCLL, MaxFALL) in nits across all threads."""
        if not levels:
            return 0.0, 0.0
        table = levels[0].table
        cll = max(float(table[level.max_code]) for level in levels)
        fall = max(float(level.max_fall) for level in levels)
        return cll, fall


def _signal_light_levels(hdr: HdrSettings, levels: list[_HdrLightLevels], paths,
                         expected_sei: int) -> None:
    """Combine the measured MaxCLL/MaxFALL, patch them into the intermediates, log both.

    Runs after every frame is encoded and before the final remux. With an explicit max_cll
    the stream already carries those values and nothing is patched. All segments get the
    same whole-job values.
    """
    cll, fall = _HdrLightLevels.combine(levels)
    written: tuple[int, int] | None = (hdr.max_cll, hdr.max_fall)
    source = "explicit max_cll"
    if hdr.measure_cll:
        if not levels:
            print(
                "WARNING: HDR10 light levels were not measured; the output keeps the static "
                f"placeholder MaxCLL/MaxFALL {hdr.max_cll}/{hdr.max_fall} nits."
            )
            written = None
        else:
            written = patch_cll_sei(paths, hdr, cll, fall, expected_sei)
        source = "measured" if written else "static fallback"
        written = written or (hdr.max_cll, hdr.max_fall)
    print(
        f"HDR10 light levels: measured MaxCLL/MaxFALL {cll:.1f}/{fall:.1f} nits; "
        f"written {written[0]}/{written[1]} ({source})."
    )


def _hdr10_to_p010(packed, levels: "_HdrLightLevels | None" = None):
    """TrueHDR output -> P010 for NVENC: BT.2020 non-constant luminance, limited range.

    `packed` is (H,W) uint32 RGB10A2 holding full-range 10-bit PQ codes in BT.2020
    primaries (black = 0, white = the luminance setting). 4:2:0 chroma is sited top-left
    (chroma_sample_loc_type 2, signalled in the VUI): a [1,2,1]/4 filter in both axes,
    centred on the even rows and columns, edges replicated. The filter is built from
    slices and adds rather than conv2d, so the result cannot depend on a cuDNN algorithm
    choice and stays byte-identical run to run.
    """
    import torch
    import torch.nn.functional as F

    # int32 planes throughout: an int64 stack of a 4K frame alone was ~200 MB of traffic.
    words = packed.view(torch.int32)
    codes = [(words >> shift) & 0x3FF for shift in (0, 10, 20)]
    if levels is not None:
        levels.update(*codes)
    r, g, b = (plane.to(torch.float32).div_(1023.0) for plane in codes)
    y = 0.2627 * r + 0.6780 * g + 0.0593 * b
    cb = (b - y) / 1.8814
    cr = (r - y) / 1.4746

    height, width = y.shape
    chroma = F.pad(torch.stack([cb, cr]).unsqueeze(0), (1, 1, 1, 1), mode="replicate")[0]
    across = chroma[:, :, 0:width:2] + 2 * chroma[:, :, 1:width + 1:2] + chroma[:, :, 2:width + 2:2]
    sited = (across[:, 0:height:2] + 2 * across[:, 1:height + 1:2] + across[:, 2:height + 2:2]) / 16

    y10 = (64.0 + 876.0 * y).round().clamp(0, 1023)
    c10 = (512.0 + 896.0 * sited).round().clamp(0, 1023)
    # torch has no uint16 left-shift on CUDA, so scale by 64 (<<6) in int32 and cast last.
    out = torch.empty((height * 3 // 2, width), dtype=torch.int32, device=packed.device)
    out[:height] = y10.to(torch.int32)
    uv = out[height:].view(height // 2, width // 2, 2)
    uv[..., 0] = c10[0].to(torch.int32)
    uv[..., 1] = c10[1].to(torch.int32)
    return (out * 64).to(torch.uint16)


# ffprobe colour_space values -> swscale in_color_matrix names.
_SWS_MATRIX = {
    "bt709": "bt709", "smpte170m": "smpte170m", "bt470bg": "bt470",
    "smpte240m": "smpte240m", "fcc": "fcc", "bt2020nc": "bt2020", "bt2020c": "bt2020",
}
INTERLACED_FIELD_ORDERS = frozenset({"tt", "bb", "tb", "bt"})


def _probe_decode_hints(input_path: str) -> dict[str, str]:
    """field_order, colour tags, pix_fmt and height of the first video stream."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=field_order,color_space,color_range,color_transfer,pix_fmt,height",
            "-of", "default=noprint_wrappers=1", input_path,
        ],
        capture_output=True, text=True, check=False,
    )
    hints = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        hints[key.strip()] = value.strip()
    return hints


HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})


def _is_hdr_source(hints: dict[str, str]) -> bool:
    """PQ or HLG transfer: already HDR, so TrueHDR (an SDR->HDR model) must not run."""
    return hints.get("color_transfer", "") in HDR_TRANSFERS


def _is_ten_bit_sdr(hints: dict[str, str]) -> bool:
    """An SDR source with 10 or more bits per sample (yuv420p10le, p010le, ...).

    Those decode as x2bgr10le and run VSR in RGB10A2, so the extra bits reach VSR instead
    of being rounded to 8 at decode. HDR sources keep the 8-bit path they always had.
    """
    pix_fmt = hints.get("pix_fmt", "")
    # Planar/semi-planar 9-16 bit (yuv420p10le, p010le, ...) and packed 16-bit RGB (rgb48le,
    # bgr48be, rgba64le, ...).
    deep = re.search(r"p0(1[0-6])|(?:9|1[0-6]|48|64)(?:le|be)$", pix_fmt) is not None
    return deep and not _is_hdr_source(hints)


def _ffmpeg_decode_command(
    input_path: str,
    use_nvdec: bool,
    segment: "VideoSegment | None" = None,
    pts_log: str | None = None,
    pix_fmt: str = "rgb24",
) -> list[str]:
    """rawvideo rgb24 decode of the whole input, or of one segment of it.

    `pix_fmt="x2bgr10le"` keeps 10-bit sources at 10 bits: one little-endian word per
    pixel, R in bits 0-9, G 10-19, B 20-29, alpha 3 -- exactly nvvfx's RGB10A2 layout.

    A segment seeks to the keyframe at or before its start (input -ss, which is relative
    to the container's start_time -- with an absolute time MPEG-TS lost 254 of 1526
    frames), keeps original timestamps (-copyts) and cuts [start_pts, end_pts) with trim,
    in the stream's own time base. Seeking 1 s early costs one GOP of decode and makes a
    demuxer that lands late impossible to miss: `pts_log` records the source pts of every
    delivered frame, and _SegmentDecoder.finish() checks the first one is exactly the
    segment's start. Verified frame-exact (framemd5 against a full decode) on mkv, mp4,
    ts and m2ts, interlace-flagged H.264 and progressive H.264.
    """
    hwaccel = ["-hwaccel", "cuda"] if use_nvdec else []
    seek: list[str] = []
    stream_map: list[str] = []
    trim = ""
    if segment is not None:
        # The keyframe scan reads v:0, so the decode must too (ffmpeg's default pick is the
        # largest video stream, which can be an attached cover image).
        stream_map = ["-map", "0:v:0"]
        seek = ["-copyts"]
        if segment.start_pts is not None:
            start_seconds = float(segment.start_pts * segment.time_base) - segment.start_time
            seek_seconds = max(0.0, start_seconds - 1.0)
            seek += ["-noaccurate_seek", "-ss", f"{seek_seconds:.6f}"]
        bounds = [f"start_pts={segment.start_pts}"] if segment.start_pts is not None else []
        bounds += [f"end_pts={segment.end_pts}"] if segment.end_pts is not None else []
        trim = f"trim={':'.join(bounds)}," if bounds else ""
    stats: list[str] = []
    if pts_log is not None:
        # The rawvideo encoder's own time base is 1/fps; demux keeps the source pts exact.
        stats = ["-enc_time_base:v", "demux", "-stats_enc_pre", pts_log,
                 "-stats_enc_pre_fmt", "{pts}"]
    # Untagged streams are the norm on Blu-ray and broadcast HD, and swscale treats an
    # unspecified matrix as BT.601 -- a visible hue shift on HD, since the encode side
    # tags the output bt709. Pick the matrix the way NVDEC's own RGB path does: the
    # stream's tag if present, else BT.709 above 576 lines and BT.601 at or below.
    hints = _probe_decode_hints(input_path)
    matrix = _SWS_MATRIX.get(hints.get("color_space", ""))
    if matrix is None:
        height = int(hints["height"]) if hints.get("height", "").isdigit() else 1080
        matrix = "bt709" if height > 576 else "smpte170m"
    in_range = "pc" if hints.get("color_range") == "pc" else "tv"
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", *hwaccel, *seek,
        "-i", input_path, *stream_map, "-an", "-sn", "-dn",
        # rawvideo output defaults to CFR, which duplicates frames to fill timestamp gaps
        # (measured 306 frames from a 304-frame Blu-ray cut). Pass frames through 1:1.
        "-fps_mode", "passthrough",
        # Both flags are load-bearing. Without full_chroma_int swscale converts through
        # integer lookup tables that sit ~1.6 Y levels low (measured -1.64 on real content
        # and on a neutral grey ramp; exact with the flags: +0.004). accurate_rnd alone
        # does not leave the table path, and NVDEC hands over nv12, where
        # full_chroma_int alone is enough but yuv420p (CPU decode) needs both.
        "-vf", f"{trim}scale=in_color_matrix={matrix}:in_range={in_range}"
               ":flags=bicubic+accurate_rnd+full_chroma_int",
        *stats,
        "-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1",
    ]


def _nvdec_can_decode(input_path: str) -> bool:
    """NVDEC has no engine for some codecs (e.g. 10-bit H.264); probe one frame first."""
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-hwaccel", "cuda",
            "-i", input_path, "-frames:v", "1", "-f", "null", "-",
        ],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


@dataclasses.dataclass(frozen=True)
class VideoSegment:
    """[start_pts, end_pts) of the first video stream, in its time base.

    None means unbounded: the first segment has no lower bound and the last none upper,
    so the segments together yield exactly what a full decode does, including any frames
    shown before the first keyframe.
    """

    index: int
    start_pts: int | None
    end_pts: int | None
    time_base: Fraction
    start_time: float


def _plan_segments(input_path: str, sessions: int) -> list[VideoSegment] | None:
    """Cut points at source keyframes for the parallel encoder, or None to stay unsegmented.

    Aims for a multiple of `sessions` segments of at most SEGMENT_SECONDS each, every cut
    snapped to the keyframe nearest its ideal position, so static assignment (segment i
    to session i mod N) gives every session about the same amount of video.
    """
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=time_base:format=start_time,duration",
            "-of", "default=noprint_wrappers=1", input_path,
        ],
        capture_output=True, text=True, check=False,
    )
    fields: dict[str, str] = {}
    for line in probe.stdout.splitlines():
        key, _, value = line.partition("=")
        fields.setdefault(key.strip(), value.strip())
    try:
        time_base = Fraction(fields["time_base"])
        duration = float(fields["duration"])
    except (KeyError, ValueError, ZeroDivisionError):
        return None
    if duration < SEGMENTED_MIN_SECONDS:
        return None
    try:
        start_time = float(fields.get("start_time", "0"))
    except ValueError:
        start_time = 0.0

    # Demux only: a packet scan of a two-hour film takes seconds.
    packets = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "packet=pts,flags", "-of", "csv=p=0", input_path,
        ],
        capture_output=True, text=True, check=False,
    ).stdout.split()
    keyframes = []
    for row in packets:
        pts, _, flags = row.partition(",")
        if "K" in flags and pts.lstrip("-").isdigit():
            keyframes.append(int(pts))
    keyframes = sorted(set(keyframes))

    def segment(index: int, start: int | None, end: int | None) -> VideoSegment:
        return VideoSegment(index, start, end, time_base, start_time)

    if len(keyframes) < 2:
        # Nowhere to cut, but still take the segmented path: it bounds the muxer's memory.
        return [segment(0, None, None)]
    count = sessions * max(1, -(-int(duration) // int(max(1.0, sessions * SEGMENT_SECONDS))))
    first, span = keyframes[0], keyframes[-1] - keyframes[0]
    cuts: list[int] = []
    for k in range(1, count):
        ideal = first + span * k / count
        nearest = min(keyframes, key=lambda pts: abs(pts - ideal))
        if nearest > first and (not cuts or nearest > cuts[-1]):
            cuts.append(nearest)
    bounds = [None, *cuts, None]
    return [segment(i, bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


class _SegmentDecoder:
    """One segment's piped ffmpeg decode, read on its own thread into a bounded queue.

    Started one segment ahead of use, so ffmpeg's start-up (0.5 s idle, 2-6 s with eight
    decoders launching at once on 6 cores) overlaps the previous segment's encode.
    """

    def __init__(self, input_path: str, use_nvdec: bool, segment: VideoSegment,
                 frame_bytes: int, scratch_dir: Path, pix_fmt: str = "rgb24") -> None:
        self.segment = segment
        self.frame_bytes = frame_bytes
        self.stop = threading.Event()
        self.pts_log = scratch_dir / f"segment-{segment.index:05d}.pts"
        self.log_path = scratch_dir / f"segment-{segment.index:05d}-decode.log"
        self.log_handle = open(self.log_path, "wb")
        try:
            self.proc = subprocess.Popen(
                _ffmpeg_decode_command(
                    input_path, use_nvdec, segment, self.pts_log.as_posix(), pix_fmt
                ),
                stdout=subprocess.PIPE,
                stderr=self.log_handle,
            )
        except BaseException:
            self.log_handle.close()
            raise
        self.queue: queue.Queue = queue.Queue(maxsize=8)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        try:
            while True:
                buffer = self.proc.stdout.read(self.frame_bytes)
                if not buffer or len(buffer) < self.frame_bytes:
                    break
                if not _put_until_stopped(self.queue, buffer, self.stop):
                    return
        except Exception as exc:  # surfaced on the consuming thread
            _put_until_stopped(self.queue, exc, self.stop)
        finally:
            _put_until_stopped(self.queue, None, self.stop)

    def frames(self):
        while True:
            item = self.queue.get()
            if isinstance(item, Exception):
                raise RuntimeError(f"Decoding segment {self.segment.index} failed.") from item
            if item is None:
                return
            yield item

    def close(self) -> None:
        """Release the process and reader, whether the segment finished or not."""
        with contextlib.suppress(Exception):
            self.proc.stdout.close()
        if self.proc.poll() is None:
            self.proc.kill()
        _stop_reader(self.reader, self.queue, self.stop)
        self.proc.wait()
        self.log_handle.close()

    def finish(self, frame_count: int) -> None:
        """Fail loudly unless ffmpeg delivered exactly this segment's frames."""
        return_code = self.proc.wait()
        self.log_handle.close()
        tail = self.log_path.read_bytes()[-1500:].decode("utf-8", errors="replace").strip()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg decode of segment {self.segment.index} failed: {tail}")
        pts = [int(value) for value in self.pts_log.read_text().split()]
        start = self.segment.start_pts
        problem = None
        if len(pts) != frame_count:
            problem = f"logged {len(pts)} frames but delivered {frame_count}"
        elif frame_count == 0:
            problem = "no frames decoded"
        elif start is not None and pts[0] != start:
            # The seek landed after the segment's keyframe: frames would be lost.
            problem = f"first frame pts {pts[0]}, expected the keyframe at {start}"
        elif self.segment.end_pts is not None and max(pts) >= self.segment.end_pts:
            problem = f"frame pts {max(pts)} is past the segment end {self.segment.end_pts}"
        if problem:
            raise RuntimeError(f"Segment {self.segment.index} decode is not frame-exact: {problem}. {tail}")


def _nvenc_encoder_kwargs() -> dict[str, str]:
    # Equivalent to NVEncC `--qvbr <cq> --preset p5 --tune uhq --tier high --level 5.1
    # --max-bitrate 100000 --multipass 2pass-full --aq --aq-strength N --aq-temporal
    # --bframes 5 --lookahead 32`. Main 10 needs no kwarg: the profile autoselects
    # from the P010 input surface. Reference lists are left to the driver (NVEncC's
    # `MultiRef L0:auto L1:auto`); numrefl0/1 forced 5+5 and measured no faster.
    #
    # Two spellings here are load-bearing and were wrong before. PyNvVideoCodec's
    # option parser drops unknown keys silently (they land in a map<string,string>
    # with no validation), so a typo costs quality with no error:
    #   * `qp` is not a key at all. The old `qp="19"` did nothing and this path ran
    #     plain VBR at the default ~10 Mbps target -- the 2x bitrate gap against the
    #     piped path. The quality-target key is `cq` (NVENC targetQuality).
    #   * the tuning value is `uhq`, not `ultra_high_quality`, and HQ is
    #     `high_quality`, not `hq`; anything else falls through to UNDEFINED, which
    #     surfaces as error 8 at CreateEncoder.
    #   * `temporalaq` enables on any non-empty string (even "0"); "" left it unset.
    # `aq` is a single key that both enables AQ and sets its strength.
    return dict(
        codec="hevc", preset=NVENC_PRESET.upper(),
        tuning_info={"hq": "high_quality"}.get(NVENC_TUNING, NVENC_TUNING),
        rc="vbr", cq=NVENC_CQ, multipass=NVENC_MULTIPASS, tier="high",
        aq=NVENC_AQ_STRENGTH, temporalaq="1", lookahead="32",
        bf=NVENC_BFRAMES, gop="250",
    )


def _create_hevc_encoders(count: int, width: int, height: int) -> list:
    """Up to `count` identically configured NVENC sessions, ceiling restored, same SPS.

    Every session of one job must emit byte-identical parameter sets: FFmpegMuxer strips
    the inline VPS/SPS/PPS from IDR packets (the hvcC box carries them), and the joined
    file has one hvcC, so a segment encoded under different parameters would be decoded
    against the wrong ones. Probed 2026-10-04: at the explicit level all sessions match.
    Returns fewer sessions (with a WARNING) only if a no-level fallback cannot match.
    """
    import PyNvVideoCodec as nvc

    kwargs = _nvenc_encoder_kwargs()

    def create(level: str | None):
        for attempt in range(1, NVENC_CREATE_ATTEMPTS + 1):
            try:
                encoder = nvc.CreateEncoder(
                    width, height, "P010", False, **kwargs, **({"level": level} if level else {}),
                )
                break
            except Exception as exc:  # noqa: BLE001 - PyNvVCException is not importable by name
                if "error 8" not in str(exc) or attempt == NVENC_CREATE_ATTEMPTS:
                    raise
                print(f"NVENC CreateEncoder returned error 8 (intermittent), attempt {attempt} "
                      f"of {NVENC_CREATE_ATTEMPTS}; retrying.")
        # `cq` makes PyNvVideoCodec zero maxBitRate, and the driver then caps VBR at its
        # own default (~32 Mbps at 5.1), so CQ never reached its target. Restore the
        # ceiling before the first frame -- and before GetSequenceParams(), since the
        # HRD values land in the VPS/SPS the muxer stores as extradata.
        if NVENC_MAX_MBPS > 0:
            rc = encoder.GetEncodeReconfigureParams()
            rc.maxBitRate = NVENC_MAX_MBPS * 1_000_000
            rc.vbvBufferSize = NVENC_MAX_MBPS * 1_000_000
            if not encoder.Reconfigure(rc):
                raise RuntimeError("NVENC Reconfigure() refused the bitrate ceiling.")
        return encoder

    level = _hevc_level(width, height)
    try:
        encoders = [create(level) for _ in range(count)]
    except Exception as exc:  # noqa: BLE001
        if not level or "error 8" not in str(exc):
            raise
        # An explicit level at P5+fullres is rejected (error 8) at random; six straight
        # rejections have not been observed. Fall back loudly rather than fail the job.
        print(
            f"WARNING: NVENC rejected explicit HEVC level {level} (error 8) "
            f"{NVENC_CREATE_ATTEMPTS} times in a row; using the driver-selected level. "
            "The ceiling still applies, but the stream may carry a different level_idc."
        )
        level = None
        encoders = [create(None) for _ in range(count)]

    reference = bytes(encoders[0].GetSequenceParams())
    matched = [encoders[0]]
    for encoder in encoders[1:]:
        for attempt in range(1, NVENC_CREATE_ATTEMPTS + 1):
            if bytes(encoder.GetSequenceParams()) == reference:
                matched.append(encoder)
                break
            # Without an explicit level the driver picks one per session (5.0 or 6.0
            # for the same input), so a session can disagree with the first.
            if attempt < NVENC_CREATE_ATTEMPTS:
                encoder = create(level)
    if len(matched) < count:
        print(
            f"WARNING: only {len(matched)} of {count} NVENC sessions produced matching "
            "parameter sets; encoding with fewer sessions (slower, same output format)."
        )
    return matched


def _warn_on_driver_mismatch() -> None:
    """The baked NGX library must match the host driver, which Modal upgrades on its own."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    host_version = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
    if host_version and host_version != DRIVER_VERSION:
        print(
            f"WARNING: host NVIDIA driver is {host_version} but this image bundles "
            f"libnvidia-ngx from {DRIVER_VERSION}. If super-resolution fails with "
            f"NvVFX_Load error -12, set DRIVER_VERSION = \"{host_version}\" and redeploy."
        )


def _is_video(input_name: str, mime_type: str | None) -> bool:
    extension = Path(input_name).suffix.lower()
    return extension in VIDEO_EXTENSIONS or (mime_type or "").lower().startswith("video/")


@app.cls(
    image=gpu_image,
    gpu=GPU_TYPE,
    # A feature film at ~80 fps is ~40 min of encoding; 3 h leaves room for long inputs
    # and the slower unsegmented fallbacks.
    timeout=3 * 3600,
    # NVDEC and NVENC both hand frames back through CPU-side swscale conversions
    # (nv12->rgb24 in, rgb24->p010 out) that run concurrently with inference. Those
    # conversions once capped 4K throughput at 4 cores, but that was before NVDEC decode
    # landed. Re-measured at 4K: cpu=4 and cpu=6 both hold ~12.3 fps with decode-wait 1%,
    # while cpu=12 measured *slower* (11.7 fps) and costs 15% more per frame. 6 keeps
    # margin for larger outputs, where swscale work scales with pixel count.
    cpu=WORKER_CPU,
    memory=24576,
    volumes={JOBS_DIR.as_posix(): jobs_volume},
    scaledown_window=120,
    max_containers=WORKER_MAX_CONTAINERS,
)
@modal.concurrent(max_inputs=WORKER_CONCURRENCY)
class UpscaleWorker:
    @modal.enter()
    def warm(self) -> None:
        """Pay CUDA init and encoder probing once per container, not once per request."""
        import nvvfx  # noqa: F401
        import torch

        torch.zeros(1, device="cuda")
        _warn_on_driver_mismatch()
        self.encoder = next(
            (enc for enc in ("hevc_nvenc", "h264_nvenc") if _encoder_works(enc)),
            "libx264",
        )
        if self.encoder == "libx264":
            print(
                "WARNING: no NVENC encoder available, falling back to CPU libx264. "
                "Video encoding will be much slower. Check that FFMPEG_URL points at a "
                "build whose NVENC SDK supports the host driver."
            )
        # UHQ tuning is an SDK 12.2+ HEVC feature, but whether this ffmpeg build exposes
        # the tune name varies, so probe instead of assuming.
        self.encoder_tune = None
        if self.encoder == "hevc_nvenc" and _encoder_works(self.encoder, ("-tune", NVENC_TUNING)):
            self.encoder_tune = NVENC_TUNING
        # Multi-reference frames (NVEncC's --ref) likewise depend on the build and the
        # card's NVENC generation, so probe rather than assume.
        self.encoder_refs = self.encoder.endswith("_nvenc") and _encoder_works(
            self.encoder, ("-refs", NVENC_REFS)
        )
        print(
            f"Selected video encoder: {self.encoder} (tune={self.encoder_tune or 'none'}, "
            f"preset={NVENC_PRESET}, cq={NVENC_CQ}, multipass={NVENC_MULTIPASS}, "
            f"level={NVENC_LEVEL}, maxrate={NVENC_MAX_MBPS}M, "
            f"refs={NVENC_REFS if self.encoder_refs else 'default'})"
        )
        # Cached effect instances keyed by (thread, role) -> (settings key, effect). Three
        # roles exist: the upscale pass, the optional same-resolution preprocess pass, and
        # TrueHDR for HDR10 output.
        #
        # The cache is per *thread*, not per container, because Modal runs concurrent
        # inputs as separate threads sharing one instance. An nvvfx effect must not be
        # shared across them: Run() is not reentrant, and the effect reuses one internal
        # DLPack output buffer between calls (the same trap the mandatory .clone() in
        # _upscale_rgb_batch guards against). Two jobs sharing one effect would interleave
        # into each other's output. Per-thread instances keep the reuse benefit without
        # the race.
        self._sr_cache: dict[tuple[int, str], tuple[tuple, object]] = {}
        self._sr_stacks: dict[int, contextlib.ExitStack] = {}
        self._sr_lock = threading.Lock()
        # Serializes Volume reload/commit across concurrent inputs (see run()).
        self._volume_lock = threading.Lock()
        self._active_jobs = 0
        # Worker threads of the parallel encoder. Persistent, so each keeps its VSR
        # effect in the per-thread cache above from one job to the next.
        self._segment_pool = ThreadPoolExecutor(
            max_workers=NVENC_SESSIONS, thread_name_prefix="nvenc-session"
        )
        # One segmented job at a time per container: two jobs interleaving their
        # submissions could each hold part of the pool while waiting for the rest (a
        # deadlock until the timeout), and they would compete for the same four engines.
        self._segment_job_lock = threading.Lock()

    @modal.exit()
    def teardown(self) -> None:
        self._segment_pool.shutdown(wait=True, cancel_futures=True)
        self._close_sr()

    def _close_sr(self) -> None:
        # VideoSuperRes exposes its lifecycle as a context manager, so release it that way.
        with self._sr_lock:
            stacks = list(self._sr_stacks.values())
            self._sr_stacks = {}
            self._sr_cache = {}
        for stack in stacks:
            stack.close()

    def _super_res(self, quality: str, output_width: int, output_height: int,
                   role: str = "upscale", input_size: tuple[int, int] | None = None):
        """Reuse the loaded model when consecutive jobs share quality and output size.

        `input_size` selects the RGB10A2 image encoding (10-bit sources, packed (H,W)
        uint32 in and out); None keeps the float RGB8 interface. The encoding is part of
        the cache key, so an 8-bit job never reuses a 10-bit effect or the reverse.
        """
        import nvvfx

        normalized_quality = _normalize_quality(quality)
        key = (normalized_quality, output_width, output_height, input_size)
        cache_key = (threading.get_ident(), role)
        cached = self._sr_cache.get(cache_key)
        if cached is not None and cached[0] == key:
            return cached[1]

        # Effects are cached per role, so replacing one must not tear down the other.
        if cached is not None:
            with contextlib.suppress(Exception):
                cached[1].close()

        with self._sr_lock:
            stack = self._sr_stacks.setdefault(cache_key[0], contextlib.ExitStack())

        quality_level = getattr(nvvfx.VideoSuperRes.QualityLevel, normalized_quality)
        if input_size is None:
            sr = stack.enter_context(nvvfx.VideoSuperRes(quality_level))
        else:
            sr = stack.enter_context(nvvfx.VideoSuperRes(
                quality_level, image_encoding=nvvfx.VideoSuperRes.ImageEncoding.RGB10A2,
            ))
            sr.input_width, sr.input_height = input_size
        sr.output_width = output_width
        sr.output_height = output_height
        sr.load()
        self._sr_cache[cache_key] = (key, sr)
        return sr

    def _preprocess_effect(self, preprocess: str | None, input_width: int, input_height: int,
                           ten_bit: bool = False):
        """Optional same-resolution restoration pass applied before upscaling.

        DENOISE/DEBLUR run at the input's own size, so the effect's output dimensions
        must equal its input dimensions.
        """
        if not preprocess:
            return None
        normalized = _normalize_quality(preprocess)
        if normalized not in SAME_RES_QUALITIES:
            raise ValueError(
                f"preprocess must be a same-resolution mode {sorted(SAME_RES_QUALITIES)}, "
                f"got {preprocess}."
            )
        if not getattr(self, "_logged_preprocess", False):
            print(f"Preprocess pass active: {normalized} at {input_width}x{input_height}.")
            self._logged_preprocess = True
        return self._super_res(
            normalized, input_width, input_height, role="preprocess",
            input_size=(input_width, input_height) if ten_bit else None,
        )

    def _true_hdr(self, hdr: HdrSettings, output_width: int, output_height: int):
        """This thread's TrueHDR effect for `hdr`, warmed up at the output size.

        Cached per thread like the VSR effects (an effect is not reentrant and reuses its
        DLPack output buffer). TrueHDR infers its size on the first run(), not at load(),
        so a warm-up run at the job's output size happens here -- before CreateEncoder,
        the same ordering that fixed error 8 after an effect changed size (see
        _upscale_video_gpu). Any change of settings or size rebuilds the effect (load is
        ~0.3 s): reusing one effect across output sizes was never probed.
        """
        import nvvfx
        import torch

        key = (hdr, output_width, output_height)
        cache_key = (threading.get_ident(), "truehdr")
        cached = self._sr_cache.get(cache_key)
        if cached is not None and cached[0] == key:
            return cached[1]
        if cached is not None:
            with contextlib.suppress(Exception):
                cached[1].close()
        with self._sr_lock:
            stack = self._sr_stacks.setdefault(cache_key[0], contextlib.ExitStack())
        effect = stack.enter_context(nvvfx.TrueHDR(
            contrast=hdr.contrast, saturation=hdr.saturation, middle_gray=hdr.middle_gray,
            luminance=hdr.luminance, debanding_off=0 if hdr.debanding else 1,
        ))
        effect.load()
        warm = torch.zeros((3, output_height, output_width), dtype=torch.float32, device="cuda")
        warmed = torch.from_dlpack(effect.run(warm).image).clone()
        torch.cuda.synchronize()
        if tuple(warmed.shape) != (output_height, output_width):
            raise OutputIntegrityError(
                f"TrueHDR returned {tuple(warmed.shape)} for a {output_width}x{output_height} frame."
            )
        self._sr_cache[cache_key] = (key, effect)
        return effect

    def _upscale_rgb_batch(self, sr, rgb_frames, plan: PlannedDimensions, pre=None,
                           keep_on_gpu: bool = False, hdr=None):
        """Preprocess, VSR, integrity check and hybrid resize for a batch of frames.

        Frames are (H,W,3) uint8 (host arrays, or CUDA tensors from in-process NVDEC), or
        (H,W) int32 host arrays of x2bgr10le words for 10-bit sources, which run the
        effects in RGB10A2 and come back as float with 10-bit precision.

        Returns (H,W,3) float [0,1] CUDA tensors (keep_on_gpu) or uint8 arrays -- or,
        when `hdr` (a TrueHDR effect) is given, (H,W) uint32 RGB10A2 CUDA tensors of
        full-range BT.2020 PQ codes for _hdr10_to_p010.
        """
        import numpy as np
        import torch
        import torch.nn.functional as F

        ten_bit = rgb_frames[0].ndim == 2
        if ten_bit:
            # x2bgr10le words are nvvfx's RGB10A2 layout; set the alpha bits it expects
            # (swscale writes 3 already, but the effect must never see anything else).
            batch_cuda = (torch.from_numpy(np.stack(rgb_frames, axis=0)).cuda()
                          | RGB10A2_ALPHA).view(torch.uint32)
        elif torch.is_tensor(rgb_frames[0]):
            # NVDEC decoded these straight into CUDA memory, so there is nothing to upload.
            batch_cuda = (
                torch.stack(rgb_frames, dim=0)
                .permute(0, 3, 1, 2)
                .float()
                .div(255.0)
                .contiguous()
            )
        else:
            # Upload uint8 and convert on the GPU: the float conversion used to run on the
            # CPU and push 4x the bytes over PCIe. float32 x/255 is correctly rounded on
            # both, so the values are identical.
            batch_cuda = (
                torch.from_numpy(np.stack(rgb_frames, axis=0))
                .cuda()
                .permute(0, 3, 1, 2)
                .float()
                .div(255.0)
                .contiguous()
            )

        if plan.is_hybrid and not getattr(self, "_logged_hybrid", False):
            print(
                f"Hybrid resize active: running VSR at {plan.sr_width}x{plan.sr_height}, "
                f"resampling to requested {plan.output_width}x{plan.output_height}."
            )
            self._logged_hybrid = True

        outputs = []
        for index in range(batch_cuda.shape[0]):
            inp_frame = batch_cuda[index]
            if pre is not None:
                # Same-resolution restoration pass. The SDK reuses its DLPack buffer, so
                # this clone is as mandatory as the one below.
                inp_frame = torch.from_dlpack(pre.run(inp_frame).image).clone()
                if ten_bit:
                    # DENOISE/DEBLUR in RGB10A2 return words without alpha 3 (probed
                    # 2026-10-04); restore it before the upscale pass reads them.
                    inp_frame = (inp_frame.view(torch.int32) | RGB10A2_ALPHA).view(torch.uint32)
            dlpack_out = sr.run(inp_frame).image
            # The NVIDIA VFX SDK reuses its internal DLPack output buffer on the next call.
            # We must explicitly clone the tensor to PyTorch memory before subsequent SDK operations.
            sr_tensor = torch.from_dlpack(dlpack_out).clone()
            if ten_bit:
                inp_frame = _unpack_rgb10a2(inp_frame)
                sr_tensor = _unpack_rgb10a2(sr_tensor)

            # Output integrity check (NaN/Inf and channel collapse). The numbers stay on the
            # GPU until the one read-back below, after TrueHDR has run too.
            stats = [_integrity_stats(inp_frame, sr_tensor)]
            stages = "Super-resolution"

            if plan.is_hybrid:
                out_tensor = F.interpolate(
                    sr_tensor.unsqueeze(0),
                    size=(plan.output_height, plan.output_width),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0)
            else:
                out_tensor = sr_tensor

            out_clamped = out_tensor.clamp(0.0, 1.0)
            if hdr is not None:
                # SDR -> HDR10 at output size, after VSR: the order NVEncC uses. VSR run on
                # PQ data instead overshot highlights (code 720 -> 900, ~650 -> 3200 nits).
                # TrueHDR reuses its DLPack output buffer like VSR, so the clone is
                # mandatory here too.
                hdr_frame = torch.from_dlpack(hdr.run(out_clamped.contiguous()).image).clone()
                if hdr_frame.shape != out_clamped.shape[1:]:
                    raise OutputIntegrityError(
                        f"TrueHDR returned {tuple(hdr_frame.shape)} for a "
                        f"{tuple(out_clamped.shape[1:])} frame."
                    )
                # Checked on a 1/16 subsample: a collapsed channel or a wrong word layout
                # shows in any 4x4 grid, and every full-frame pass here is GPU time that the
                # four segment threads' device-wide syncs all wait on.
                sample = hdr_frame[::4, ::4]
                stats.append(_integrity_stats(
                    out_clamped[:, ::4, ::4], _unpack_rgb10a2(sample), valid=_rgb10a2_alpha_ok(sample),
                ))
                stages += "+TrueHDR"
            _check_frame_integrity(torch.cat(stats), plan.sr_width, plan.sr_height, stages)

            if hdr is not None:
                outputs.append(hdr_frame)
                continue
            out_clamped = out_clamped.movedim(0, -1)
            if keep_on_gpu:
                # The GPU encoder path converts to P010 on-device, so the 24.9 MB/frame
                # download that dominates the piped path never happens.
                outputs.append(out_clamped)
                continue
            output_uint8 = (out_clamped.cpu().numpy() * 255.0).round().astype(np.uint8)
            outputs.append(output_uint8)

        return outputs

    def _upscale_image(
        self,
        input_path: Path,
        output_dir: Path,
        stem: str,
        resize_type: str,
        scale: float,
        width: int,
        height: int,
        quality: str,
        keep_aspect_ratio: bool = True,
        preprocess: str | None = None,
    ) -> tuple[str, str]:
        import numpy as np
        from PIL import Image

        with Image.open(input_path) as input_image:
            rgb_array = np.asarray(input_image.convert("RGB"), dtype=np.uint8)

        in_height, in_width = rgb_array.shape[:2]
        plan = plan_dimensions(
            input_width=in_width,
            input_height=in_height,
            resize_type=resize_type,
            scale=scale,
            width=width,
            height=height,
            keep_aspect_ratio=keep_aspect_ratio,
        )

        sr = self._super_res(quality, plan.sr_width, plan.sr_height)
        pre = self._preprocess_effect(preprocess, in_width, in_height)
        output_rgb = self._upscale_rgb_batch(sr, [rgb_array], plan, pre)[0]

        output_name = f"{stem}_upscaled.png"
        Image.fromarray(output_rgb).save(output_dir / output_name, format="PNG")
        return output_name, "image/png"

    def _upscale_video_gpu(
        self,
        input_path: Path,
        output_dir: Path,
        stem: str,
        resize_type: str,
        scale: float,
        width: int,
        height: int,
        quality: str,
        keep_aspect_ratio: bool = True,
        preprocess: str | None = None,
        scratch_dir: Path | None = None,
        hdr: HdrSettings | None = None,
    ) -> tuple[str, str]:
        """GPU-resident output path: VSR result goes to NVENC without touching the host.

        The piped path sends every 4K frame GPU->host->pipe->ffmpeg->swscale->NVENC, about
        31 MB per frame across three processes. That contention, not any single stage, is
        what caps it at 12 fps (see CLAUDE.md section 6). Here the frame stays in CUDA
        memory from VSR through colour conversion into the encoder, which is how NVEncC
        reaches ~50 fps on far weaker hardware.
        """
        import numpy as np
        import PyNvVideoCodec as nvc
        import torch

        # PyNvVideoCodec calls __dlpack__(stream) positionally; torch 2.13 wants it
        # keyword-only. Patch the signature rather than wrapping the tensor, so torch keeps
        # reporting the CUDA device (a wrapper makes the encoder see a CPU buffer).
        if not getattr(UpscaleWorker, "_dlpack_patched", False):
            original_dlpack = torch.Tensor.__dlpack__
            torch.Tensor.__dlpack__ = lambda self, *a, **k: original_dlpack(self)
            UpscaleWorker._dlpack_patched = True

        output_name = f"{stem}_upscaled.mp4"
        output_path = output_dir / output_name
        # Intermediates stay on container-local disk; only the final output goes to the
        # Volume, so a failed job never commits a multi-GB partial stream.
        scratch_dir = scratch_dir or output_dir
        video_only_path = scratch_dir / f"{stem}_video.mp4"
        decode_log_path = scratch_dir / "ffmpeg-decode.log"

        in_width, in_height, fps = _probe_video_metadata(input_path.as_posix())
        plan = plan_dimensions(
            input_width=in_width,
            input_height=in_height,
            resize_type=resize_type,
            scale=scale,
            width=width,
            height=height,
            keep_aspect_ratio=keep_aspect_ratio,
        )
        batch_limit = _batch_size_for_output(plan.output_width, plan.output_height)
        hints = _probe_decode_hints(input_path.as_posix())
        ten_bit = _is_ten_bit_sdr(hints)
        frame_bytes = in_width * in_height * (4 if ten_bit else 3)

        # Only consulted if the in-process decoder below is unavailable; the warning is
        # deferred until we know we actually need the piped decoder.
        use_nvdec = _nvdec_can_decode(input_path.as_posix())

        # Load this job's VSR effect *before* creating the encoder. When the cached
        # effect from the previous job had a different output size (an image job, or a
        # 4K job before an 8K one), CreateEncoder with an explicit level at P5+fullres
        # failed nvEncInitializeEncoder with error 8 -- every video job after an image
        # job in production, 2026-10-03. Swapping the effect first avoids that state.
        sr = self._super_res(quality, plan.sr_width, plan.sr_height,
                             input_size=(in_width, in_height) if ten_bit else None)
        pre = self._preprocess_effect(preprocess, in_width, in_height, ten_bit)
        # TrueHDR too: its warm-up run at the output size is part of the effect swap.
        thdr = self._true_hdr(hdr, plan.output_width, plan.output_height) if hdr else None
        sei = hdr10_sei_nals(hdr) if hdr else b""
        levels = _HdrLightLevels("cuda") if hdr else None

        # Settings, level policy, error-8 retries and the bitrate ceiling all live in the
        # shared factory, so this path and the parallel one cannot drift apart.
        encoder = _create_hevc_encoders(1, plan.output_width, plan.output_height)[0]
        fps_num, fps_den = Fraction(fps).limit_denominator(65535).as_integer_ratio()
        muxer = nvc.FFmpegMuxer(
            video_only_path.as_posix(), nvc.MP4, "hevc",
            plan.output_width, plan.output_height, fps_num, fps_den,
            1, 90000, encoder.GetSequenceParams(),
        )
        # No source container is muxed here, so Finalize() needs the per-frame tick count
        # to rebuild display-order PTS. Without it, B-frame reordering silently drops the
        # tail frames (measured: 98 of 100).
        muxer.SetUniformPtsIncrement(round(90000 * fps_den / fps_num))

        # In-process NVDEC keeps decoded frames in CUDA memory, so the last host
        # round-trip disappears and the pipeline is GPU-resident end to end. Probed at
        # 1675 fps decode-only on the reference clip, i.e. free either way -- this buys
        # architecture, not throughput.
        #
        # Deliberately the low-level demuxer/decoder pair. ThreadedDecoder is unusable in
        # PyNvVideoCodec 2.2.0 (unchanged in 2.2.3): it drains after exactly one frame in every configuration
        # probed (NATIVE/RGB/RGBP, every buffer and batch size, mkv and mp4).
        # Interlace-flagged streams (field_order tt/bb, i.e. most UK/EU HD Blu-ray and
        # broadcast) make PyNvVideoCodec create its decoder in Adaptive deinterlace mode,
        # and 2.2.x exposes no kwarg to change it (re-probed on 2.2.3). On PsF content -- progressive frames in
        # an interlaced container, the usual case for drama -- that rewrites one field's
        # rows: probed on a BBC Blu-ray, 7.8% of odd-row pixels changed by >6 levels (up to
        # 31% in a frame) and 32% of vertical detail was gone before VSR saw the frame.
        # ffmpeg's NVDEC path weaves instead (lossless for PsF), so route those inputs to
        # the piped decoder. Truly interlaced (50i/60i) video would want a real
        # deinterlacer such as bwdif ahead of VSR; that is not handled here.
        field_order = hints.get("field_order", "")
        interlaced = field_order in INTERLACED_FIELD_ORDERS
        if interlaced and GPU_DECODER and use_nvdec:
            print(
                f"WARNING: input is flagged interlaced (field_order={field_order}); "
                "in-process NVDEC would apply Adaptive deinterlacing and soften vertical "
                "detail, so decoding through ffmpeg NVDEC (woven) instead."
            )
        if ten_bit and GPU_DECODER and use_nvdec:
            print(
                f"10-bit source ({hints.get('pix_fmt')}): decoding through ffmpeg NVDEC as "
                "x2bgr10le so VSR sees all 10 bits; in-process NVDEC delivers 8-bit RGB only."
            )
        gpu_decoder = None
        if GPU_DECODER and use_nvdec and not interlaced and not ten_bit:
            try:
                demuxer = nvc.CreateDemuxer(filename=input_path.as_posix())
                gpu_decoder = nvc.CreateDecoder(
                    gpuid=0, codec=demuxer.GetNvCodecId(), usedevicememory=1,
                    outputColorType=nvc.OutputColorType.RGB,
                )
            except Exception as exc:  # noqa: BLE001 - any failure just means fall back
                print(
                    f"WARNING: in-process NVDEC unavailable for this input ({exc}); "
                    "falling back to the piped decoder."
                )
                gpu_decoder = None

        decode_log_handle = None
        decoder_proc = None
        reader = None
        frame_queue: queue.Queue = queue.Queue(maxsize=max(8, 3 * batch_limit))
        stop_reading = threading.Event()

        if gpu_decoder is not None:
            # Decode on its own thread. This is for honest accounting and headroom, not
            # speed: inline decode reported decode-wait 50% but moving it here left the
            # wall clock identical (5.1s), so that "wait" was overlapping GPU work. The
            # clone happens on the producer side because the decoder recycles its output
            # surfaces on the next Decode() call.
            def _read_cuda_frames() -> None:
                import torch

                try:
                    for packet in demuxer:
                        for decoded in gpu_decoder.Decode(packet):
                            frame = torch.from_dlpack(decoded).clone()
                            if not _put_until_stopped(frame_queue, frame, stop_reading):
                                return
                except Exception as exc:  # surfaced on the main thread below
                    _put_until_stopped(frame_queue, exc, stop_reading)
                finally:
                    _put_until_stopped(frame_queue, None, stop_reading)

            reader = threading.Thread(target=_read_cuda_frames, daemon=True)
            reader.start()
        else:
            if not use_nvdec:
                print(
                    "WARNING: NVDEC cannot decode this input (unsupported codec or pixel "
                    "format); falling back to CPU decoding, which is slower."
                )
            decode_log_handle = open(decode_log_path, "wb")
            decoder_proc = subprocess.Popen(
                _ffmpeg_decode_command(
                    input_path.as_posix(), use_nvdec,
                    pix_fmt="x2bgr10le" if ten_bit else "rgb24",
                ),
                stdout=subprocess.PIPE,
                stderr=decode_log_handle,
            )

            def _read_frames() -> None:
                try:
                    while True:
                        buffer = decoder_proc.stdout.read(frame_bytes)
                        if not buffer or len(buffer) < frame_bytes:
                            break
                        if not _put_until_stopped(frame_queue, buffer, stop_reading):
                            return
                except Exception as exc:  # surfaced on the main thread below
                    _put_until_stopped(frame_queue, exc, stop_reading)
                finally:
                    _put_until_stopped(frame_queue, None, stop_reading)

            reader = threading.Thread(target=_read_frames, daemon=True)
            reader.start()

        frame_count = 0
        packet_count = 0
        decode_seconds = 0.0
        infer_seconds = 0.0
        encode_seconds = 0.0
        started = time.perf_counter()
        rgb_batch: list = []
        sei_count = 0  # SEIs actually inserted: what patch_cll_sei must find

        def _mux(packets) -> int:
            nonlocal sei_count
            count = 0
            for packet in packets or []:
                data = bytes(packet["data"])
                if sei:
                    # HDR10 mastering display + MaxCLL on every IRAP, as NVEncC writes them.
                    size = len(data)
                    data = insert_sei_before_irap_slice(data, sei)
                    sei_count += len(data) != size
                muxer.MuxVideoPacket(data, packet["picture_type"], packet["timestamp"])
                count += 1
            return count

        held_surface: list = [None]  # last P010 surface handed to Encode(), see _flush

        def _flush(sr, pre) -> None:
            nonlocal infer_seconds, encode_seconds, packet_count
            if not rgb_batch:
                return
            infer_start = time.perf_counter()
            upscaled = self._upscale_rgb_batch(sr, rgb_batch, plan, pre, keep_on_gpu=True,
                                               hdr=thdr)
            infer_seconds += time.perf_counter() - infer_start
            encode_start = time.perf_counter()
            for frame_rgb in upscaled:
                p010 = _hdr10_to_p010(frame_rgb, levels) if hdr else _rgb_to_p010(frame_rgb)
                # NVENC reads this surface from the encoder ASIC, which takes no part in
                # CUDA stream ordering: it can start reading before the conversion kernels
                # above have run. Without this sync it encoded half-written surfaces --
                # measured as alternating-column striping on every frame, because the
                # freed uint16 buffer was recycled for the next frame's int32 `packed`
                # tensor and int32 read as uint16 is [value, 0, value, 0, ...].
                #
                # Reproduced and fixed in scripts/probe_p010_encode.py (variants I vs N):
                # same code, sync alone turns 22 corrupt frames of 30 into 0. Holding the
                # buffer alive instead does NOT work (variants K/L/M) -- the race is
                # against the kernels filling it, not against the allocator.
                torch.cuda.synchronize()
                # Encode() copies the surface into the encoder's own buffer with
                # cuMemcpy2DAsync on a non-blocking stream PyNvVideoCodec creates, and
                # returns before that copy runs. Dropping `p010` at once would hand its
                # memory back to torch's allocator, which knows nothing of that stream:
                # the next allocation could overwrite the surface before NVENC got it.
                # The device-wide sync above also waits for the previous frame's copy,
                # so the previous surface is safe to release only now.
                held_surface[0] = None
                packet_count += _mux(encoder.Encode(p010))
                held_surface[0] = p010
            encode_seconds += time.perf_counter() - encode_start
            rgb_batch.clear()

        def _decoded_frames():
            """Yield one RGB frame at a time: CUDA tensors from NVDEC, else host arrays."""
            while True:
                item = frame_queue.get()
                if isinstance(item, Exception):
                    raise RuntimeError("Video decoding failed.") from item
                if item is None:
                    return
                if isinstance(item, bytes) and ten_bit:
                    yield np.frombuffer(item, dtype="<i4").reshape(in_height, in_width)
                elif isinstance(item, bytes):
                    yield np.frombuffer(item, dtype=np.uint8).reshape(
                        in_height, in_width, 3
                    )
                else:
                    yield item

        try:
            frames = _decoded_frames()
            while True:
                decode_start = time.perf_counter()
                frame = next(frames, None)
                decode_seconds += time.perf_counter() - decode_start
                if frame is None:
                    _flush(sr, pre)
                    break
                rgb_batch.append(frame)
                frame_count += 1
                if len(rgb_batch) >= batch_limit:
                    _flush(sr, pre)
        finally:
            if decoder_proc is not None:
                with contextlib.suppress(Exception):
                    decoder_proc.stdout.close()
            _stop_reader(reader, frame_queue, stop_reading)
            if decoder_proc is not None:
                decoder_proc.wait()
            if decode_log_handle is not None:
                decode_log_handle.close()

        packet_count += _mux(encoder.EndEncode())
        torch.cuda.synchronize()
        held_surface[0] = None
        muxer.Finalize()
        # Drop the reference so the container file is closed before ffmpeg reads it.
        muxer = None

        if frame_count == 0:
            decode_tail = ""
            if decode_log_path.exists():
                decode_tail = decode_log_path.read_bytes()[-2000:].decode(
                    "utf-8", errors="replace"
                )
            raise ValueError(f"Uploaded video has no decodable frames. {decode_tail}")
        if packet_count != frame_count:
            raise RuntimeError(
                f"Encoder returned {packet_count} packets for {frame_count} frames; "
                "output would be missing frames."
            )
        if levels is not None:
            # Measured MaxCLL/MaxFALL into the local intermediate, before the copy remux.
            _signal_light_levels(hdr, [levels], [video_only_path], sei_count)

        # Remux to attach audio and write the colour VUI (bt709, or BT.2020/PQ for HDR10).
        # NVENC does not tag colour itself, and CLAUDE.md requires all three fields to be
        # set. The input here is an mp4 with real timestamps, so unlike a raw elementary
        # stream this remux is lossless.
        mux_command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_only_path.as_posix(),
            "-i", input_path.as_posix(),
            "-map", "0:v:0", *_audio_args(input_path.as_posix()),
            "-c:v", "copy", *_remux_color_args(hdr),
            "-movflags", "+faststart", output_path.as_posix(),
        ]
        result = subprocess.run(mux_command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Muxing failed: {result.stderr.strip()[-1500:]}")
        video_only_path.unlink(missing_ok=True)
        decode_log_path.unlink(missing_ok=True)

        elapsed = max(time.perf_counter() - started, 1e-6)
        print(
            f"Video done: {frame_count} frames in {elapsed:.1f}s = {frame_count / elapsed:.1f} fps "
            f"(decode-wait {100 * decode_seconds / elapsed:.0f}%, "
            f"infer {100 * infer_seconds / elapsed:.0f}%, "
            f"encode {100 * encode_seconds / elapsed:.0f}%; "
            f"nvdec={'in-process' if gpu_decoder is not None else ('yes' if use_nvdec else 'no')}, "
            f"batch={batch_limit}, gpu-encoder=yes{', 10-bit' if ten_bit else ''}"
            f"{', hdr=' + hdr.describe() if hdr else ''})"
        )
        return output_name, "video/mp4"

    def _upscale_video_segmented(
        self,
        input_path: Path,
        output_dir: Path,
        stem: str,
        segments: list[VideoSegment],
        resize_type: str,
        scale: float,
        width: int,
        height: int,
        quality: str,
        keep_aspect_ratio: bool = True,
        preprocess: str | None = None,
        scratch_dir: Path | None = None,
        hdr: HdrSettings | None = None,
    ) -> tuple[str, str]:
        """Parallel NVENC sessions over keyframe-aligned segments (see NVENC_SESSIONS).

        Each worker thread owns one VSR effect (the per-thread `_super_res` cache), one
        encoder session and its share of the segments: i, i+N, i+2N, ... in time order.
        Per segment it runs a piped ffmpeg decode (seek + trim, prefetched one segment
        ahead), VSR, P010 and Encode -- the same per-frame steps, sync included, as
        _upscale_video_gpu -- and muxes into the segment's own mp4. One session runs
        continuously across its segments: FORCEIDR starts each segment, and packets are
        routed to segment muxers by the session's input index. EndEncode() runs once,
        after the worker's last segment; flushing per segment measured 37.4 fps against
        44.0 on one worker (2026-10-04).
        """
        import numpy as np
        import PyNvVideoCodec as nvc
        import torch

        if not getattr(UpscaleWorker, "_dlpack_patched", False):
            original_dlpack = torch.Tensor.__dlpack__
            torch.Tensor.__dlpack__ = lambda self, *a, **k: original_dlpack(self)
            UpscaleWorker._dlpack_patched = True

        output_name = f"{stem}_upscaled.mp4"
        output_path = output_dir / output_name
        scratch_dir = Path(tempfile.mkdtemp(prefix="segments-", dir=scratch_dir))

        in_width, in_height, fps = _probe_video_metadata(input_path.as_posix())
        plan = plan_dimensions(
            input_width=in_width,
            input_height=in_height,
            resize_type=resize_type,
            scale=scale,
            width=width,
            height=height,
            keep_aspect_ratio=keep_aspect_ratio,
        )
        batch_limit = _batch_size_for_output(plan.output_width, plan.output_height)
        ten_bit = _is_ten_bit_sdr(_probe_decode_hints(input_path.as_posix()))
        decode_format = "x2bgr10le" if ten_bit else "rgb24"
        frame_bytes = in_width * in_height * (4 if ten_bit else 3)
        sei = hdr10_sei_nals(hdr) if hdr else b""
        use_nvdec = _nvdec_can_decode(input_path.as_posix())
        if not use_nvdec:
            print(
                "WARNING: NVDEC cannot decode this input (unsupported codec or pixel "
                "format); falling back to CPU decoding, which is slower."
            )
        fps_num, fps_den = Fraction(fps).limit_denominator(65535).as_integer_ratio()
        pts_increment = round(90000 * fps_den / fps_num)
        force_idr = int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR)

        sessions = min(NVENC_SESSIONS, len(segments))
        stop = threading.Event()
        loaded = [threading.Event() for _ in range(sessions)]
        go = threading.Event()
        encoders: list = []
        segment_frames = [0] * len(segments)
        segment_paths = [scratch_dir / f"segment-{seg.index:05d}.mp4" for seg in segments]
        stage_seconds = {"decode-wait": 0.0, "infer": 0.0, "encode": 0.0}
        effect_seconds = [0.0] * sessions
        # MaxCLL/MaxFALL accumulate per worker slot and are combined after the join.
        light_levels: list = [None] * sessions
        sei_counts = [0] * sessions  # SEIs inserted per slot, for patch_cll_sei
        stage_lock = threading.Lock()

        def worker(slot: int) -> None:
            decode_wait = infer_seconds = encode_seconds = 0.0
            decoders: list[_SegmentDecoder] = []
            try:
                # Per-thread effects, loaded before any encoder exists (the error-8
                # mitigation of _upscale_video_gpu, kept for the same reason).
                effect_start = time.perf_counter()
                sr = self._super_res(quality, plan.sr_width, plan.sr_height,
                                     input_size=(in_width, in_height) if ten_bit else None)
                pre = self._preprocess_effect(preprocess, in_width, in_height, ten_bit)
                thdr = self._true_hdr(hdr, plan.output_width, plan.output_height) if hdr else None
                levels = _HdrLightLevels("cuda") if hdr else None
                light_levels[slot] = levels
                with stage_lock:
                    effect_seconds[slot] = time.perf_counter() - effect_start
                # The first segment is this slot's whatever the session count turns out
                # to be (segments[slot::k][0] == segments[slot]), so its ffmpeg start-up
                # can overlap encoder creation.
                first = _SegmentDecoder(
                    input_path.as_posix(), use_nvdec, segments[slot], frame_bytes, scratch_dir,
                    decode_format,
                )
                decoders.append(first)
                loaded[slot].set()
                go.wait()
                if stop.is_set() or slot >= len(encoders):
                    return
                encoder = encoders[slot]
                mine = segments[slot::len(encoders)]
                # [segment, first session input index, frame count or None, muxer, packets]
                open_segments: list[list] = []
                session_frames = 0

                def route(packets) -> None:
                    for packet in packets or []:
                        index = packet["timestamp"]
                        entry = next(
                            (e for e in open_segments
                             if index >= e[1] and (e[2] is None or index < e[1] + e[2])),
                            None,
                        )
                        if entry is None:
                            raise RuntimeError(f"NVENC returned a packet for unknown input {index}.")
                        data = bytes(packet["data"])
                        if sei:
                            # HDR10 SEI on every IRAP, including each segment's FORCEIDR.
                            size = len(data)
                            data = insert_sei_before_irap_slice(data, sei)
                            sei_counts[slot] += len(data) != size
                        entry[3].MuxVideoPacket(data, packet["picture_type"], index - entry[1])
                        entry[4] += 1
                        if entry[2] is not None and entry[4] == entry[2]:
                            entry[3].Finalize()
                            open_segments.remove(entry)

                def start(position: int) -> _SegmentDecoder | None:
                    if position >= len(mine):
                        return None
                    decoder = _SegmentDecoder(
                        input_path.as_posix(), use_nvdec, mine[position], frame_bytes, scratch_dir,
                        decode_format,
                    )
                    decoders.append(decoder)
                    return decoder

                held: list = [None]  # last P010 surface handed to Encode()
                current = first
                for position, segment in enumerate(mine):
                    following = start(position + 1)  # prefetch: overlaps ffmpeg start-up
                    muxer = nvc.FFmpegMuxer(
                        segment_paths[segment.index].as_posix(), nvc.MP4, "hevc",
                        plan.output_width, plan.output_height, fps_num, fps_den,
                        1, 90000, encoder.GetSequenceParams(),
                    )
                    # See _upscale_video_gpu: Finalize() rebuilds display-order PTS.
                    muxer.SetUniformPtsIncrement(pts_increment)
                    entry = [segment, session_frames, None, muxer, 0]
                    open_segments.append(entry)
                    frame_count = 0
                    batch: list = []

                    def flush() -> None:
                        nonlocal infer_seconds, encode_seconds, frame_count
                        if not batch:
                            return
                        infer_start = time.perf_counter()
                        upscaled = self._upscale_rgb_batch(sr, batch, plan, pre, keep_on_gpu=True,
                                                           hdr=thdr)
                        infer_seconds += time.perf_counter() - infer_start
                        encode_start = time.perf_counter()
                        for frame_rgb in upscaled:
                            p010 = (_hdr10_to_p010(frame_rgb, levels) if hdr
                                    else _rgb_to_p010(frame_rgb))
                            # The striping fix (see _upscale_video_gpu). A per-thread stream
                            # sync is NOT enough: measured striped (Laplacian 52-85) with
                            # four workers, because nvvfx does not run on our stream.
                            torch.cuda.synchronize()
                            held[0] = None  # see _upscale_video_gpu: the copy is done now
                            route(encoder.Encode(p010, force_idr) if frame_count == 0
                                  else encoder.Encode(p010))
                            held[0] = p010
                            frame_count += 1
                        encode_seconds += time.perf_counter() - encode_start
                        batch.clear()

                    frames = current.frames()
                    while not stop.is_set():
                        wait_start = time.perf_counter()
                        buffer = next(frames, None)
                        decode_wait += time.perf_counter() - wait_start
                        if buffer is None:
                            break
                        batch.append(
                            np.frombuffer(buffer, dtype="<i4").reshape(in_height, in_width)
                            if ten_bit else
                            np.frombuffer(buffer, dtype=np.uint8).reshape(in_height, in_width, 3)
                        )
                        if len(batch) >= batch_limit:
                            flush()
                    if stop.is_set():
                        return
                    flush()
                    current.finish(frame_count)
                    entry[2] = frame_count
                    session_frames += frame_count
                    segment_frames[segment.index] = frame_count
                    if entry[4] == frame_count:
                        muxer.Finalize()
                        open_segments.remove(entry)
                    current = following
                encode_start = time.perf_counter()
                route(encoder.EndEncode())
                torch.cuda.synchronize()
                held[0] = None
                encode_seconds += time.perf_counter() - encode_start
                if open_segments:
                    missing = [(e[0].index, e[2], e[4]) for e in open_segments]
                    raise RuntimeError(
                        f"Encoder returned too few packets (segment, frames, packets): {missing}"
                    )
            except BaseException:
                stop.set()
                raise
            finally:
                loaded[slot].set()
                for decoder in decoders:
                    decoder.close()
                with stage_lock:
                    stage_seconds["decode-wait"] += decode_wait
                    stage_seconds["infer"] += infer_seconds
                    stage_seconds["encode"] += encode_seconds

        print(
            f"Segmented encode: {len(segments)} keyframe-aligned segments across {sessions} "
            "NVENC sessions; decoding through ffmpeg"
            f"{' NVDEC' if use_nvdec else ''} (woven, seek + trim per segment)"
            f"{', as x2bgr10le (10-bit source)' if ten_bit else ''}."
        )
        with self._segment_job_lock:
            started = time.perf_counter()
            futures = [self._segment_pool.submit(worker, slot) for slot in range(sessions)]
            try:
                for event in loaded:
                    while not event.wait(timeout=0.5):
                        if any(f.done() for f in futures):
                            break
                effects_done = time.perf_counter()
                if not stop.is_set():
                    encoders.extend(
                        _create_hevc_encoders(sessions, plan.output_width, plan.output_height)
                    )
                setup_done = time.perf_counter()
            except BaseException:
                stop.set()
                raise
            finally:
                go.set()
                done, _ = wait(futures, return_when=FIRST_EXCEPTION)
                if any(f.exception() for f in done):
                    stop.set()
                wait(futures)
                encoders.clear()
            for future in futures:
                if future.exception() is not None:
                    raise future.exception()
            encode_done = time.perf_counter()

        frame_count = sum(segment_frames)
        if frame_count == 0:
            raise ValueError("Uploaded video has no decodable frames.")
        if hdr:
            # Whole-job MaxCLL/MaxFALL, the same in every segment, patched into the local
            # segment files before the concat remux.
            _signal_light_levels(
                hdr, [level for level in light_levels if level is not None],
                [path for path in segment_paths if path.exists()], sum(sei_counts),
            )
        list_path = scratch_dir / "segments.txt"
        list_path.write_text(
            "".join(f"file '{path.as_posix()}'\n" for path in segment_paths if path.exists())
        )
        # Joins the segments without re-encoding, then the same remux as
        # _upscale_video_gpu: audio from the source and the colour VUI.
        mux_command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", list_path.as_posix(),
            "-i", input_path.as_posix(),
            "-map", "0:v:0", *_audio_args(input_path.as_posix()),
            "-c:v", "copy", *_remux_color_args(hdr),
            "-movflags", "+faststart", output_path.as_posix(),
        ]
        result = subprocess.run(mux_command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Muxing failed: {result.stderr.strip()[-1500:]}")
        shutil.rmtree(scratch_dir, ignore_errors=True)

        finished = time.perf_counter()
        elapsed = max(finished - started, 1e-6)
        encode_wall = max(encode_done - setup_done, 1e-6)
        worker_seconds = max(encode_wall * sessions, 1e-6)
        print(
            f"Video done: {frame_count} frames in {elapsed:.1f}s = {frame_count / elapsed:.1f} fps "
            f"(setup {setup_done - started:.1f}s [effects {max(effect_seconds):.1f}s, encoders "
            f"{setup_done - effects_done:.1f}s], encode {encode_wall:.1f}s = "
            f"{frame_count / encode_wall:.1f} fps, join {finished - encode_done:.1f}s; "
            f"sessions={sessions}, segments={len(segments)}; summed over sessions: "
            f"decode-wait {100 * stage_seconds['decode-wait'] / worker_seconds:.0f}%, "
            f"infer {100 * stage_seconds['infer'] / worker_seconds:.0f}%, "
            f"encode {100 * stage_seconds['encode'] / worker_seconds:.0f}%; "
            f"nvdec={'yes' if use_nvdec else 'no'}, batch={batch_limit}, gpu-encoder=yes"
            f"{', 10-bit' if ten_bit else ''}{', hdr=' + hdr.describe() if hdr else ''}; "
            f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.1f} GiB)"
        )
        return output_name, "video/mp4"

    def _upscale_video(
        self,
        input_path: Path,
        output_dir: Path,
        stem: str,
        resize_type: str,
        scale: float,
        width: int,
        height: int,
        quality: str,
        keep_aspect_ratio: bool = True,
        preprocess: str | None = None,
        scratch_dir: Path | None = None,
        hdr: HdrSettings | None = None,
    ) -> tuple[str, str]:
        import numpy as np

        gpu_path = GPU_ENCODER and self.encoder == "hevc_nvenc"
        if hdr is not None:
            # Fail before any GPU work rather than hand back SDR, or HDR made from HDR.
            if not gpu_path:
                raise HdrRequestError(
                    "HDR10 request: TrueHDR output needs the GPU encoder path "
                    f"(encoder={self.encoder}, MODAL_GPU_ENCODER={int(GPU_ENCODER)}); the piped "
                    "fallback writes SDR only."
                )
            hints = _probe_decode_hints(input_path.as_posix())
            if _is_hdr_source(hints):
                raise HdrRequestError(
                    f"HDR10 request: the source is already HDR (color_transfer="
                    f"{hints.get('color_transfer')}); TrueHDR converts SDR sources only."
                )
            in_width, in_height, _ = _probe_video_metadata(input_path.as_posix())
            plan = plan_dimensions(in_width, in_height, resize_type, scale, width, height,
                                   keep_aspect_ratio)
            if plan.output_width * plan.output_height > MAX_HDR_OUTPUT_PIXELS:
                raise HdrRequestError(
                    f"HDR10 request: output {plan.output_width}x{plan.output_height} is above "
                    f"the largest size TrueHDR was verified at ({MAX_HDR_OUTPUT_PIXELS} pixels)."
                )
        if gpu_path:
            segments = _plan_segments(input_path.as_posix(), NVENC_SESSIONS)
            if segments is not None:
                return self._upscale_video_segmented(
                    input_path, output_dir, stem, segments, resize_type, scale, width,
                    height, quality, keep_aspect_ratio, preprocess, scratch_dir, hdr,
                )
            return self._upscale_video_gpu(
                input_path, output_dir, stem, resize_type, scale, width, height,
                quality, keep_aspect_ratio, preprocess, scratch_dir, hdr,
            )

        output_name = f"{stem}_upscaled.mp4"
        output_path = output_dir / output_name
        scratch_dir = scratch_dir or output_dir
        stderr_path = scratch_dir / "ffmpeg.log"
        decode_log_path = scratch_dir / "ffmpeg-decode.log"

        in_width, in_height, fps = _probe_video_metadata(input_path.as_posix())
        plan = plan_dimensions(
            input_width=in_width,
            input_height=in_height,
            resize_type=resize_type,
            scale=scale,
            width=width,
            height=height,
            keep_aspect_ratio=keep_aspect_ratio,
        )

        command = _ffmpeg_encode_command(
            encoder=self.encoder,
            output_width=plan.output_width,
            output_height=plan.output_height,
            fps=fps,
            audio_source_path=input_path.as_posix(),
            output_path=output_path.as_posix(),
            tune=getattr(self, "encoder_tune", None),
            refs=getattr(self, "encoder_refs", False),
        )
        batch_limit = _batch_size_for_output(plan.output_width, plan.output_height)
        frame_bytes = in_width * in_height * 3

        use_nvdec = _nvdec_can_decode(input_path.as_posix())
        if not use_nvdec:
            print(
                "WARNING: NVDEC cannot decode this input (unsupported codec or pixel "
                "format); falling back to CPU decoding, which is slower."
            )

        stderr_handle = open(stderr_path, "wb")
        decode_log_handle = open(decode_log_path, "wb")
        encoder_proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=stderr_handle)
        decoder_proc = subprocess.Popen(
            _ffmpeg_decode_command(input_path.as_posix(), use_nvdec),
            stdout=subprocess.PIPE,
            stderr=decode_log_handle,
        )

        # Decode, inference and encode each run at roughly 20 fps at 4K, so they only pay
        # for themselves once if they run at the same time. A reader thread keeps the
        # decoder busy and a writer thread keeps NVENC busy; the main thread does nothing
        # but CUDA work, which it must own exclusively.
        frame_queue: queue.Queue = queue.Queue(maxsize=max(8, 3 * batch_limit))
        write_queue: queue.Queue = queue.Queue(maxsize=max(8, 3 * batch_limit))
        writer_errors: list[Exception] = []
        stop_reading = threading.Event()

        def _read_frames() -> None:
            try:
                while True:
                    buffer = decoder_proc.stdout.read(frame_bytes)
                    if not buffer or len(buffer) < frame_bytes:
                        break
                    if not _put_until_stopped(frame_queue, buffer, stop_reading):
                        return
            except Exception as exc:  # surfaced on the main thread below
                _put_until_stopped(frame_queue, exc, stop_reading)
            finally:
                _put_until_stopped(frame_queue, None, stop_reading)

        def _write_frames() -> None:
            failed = False
            while True:
                item = write_queue.get()
                if item is None:
                    break
                if failed:
                    # Keep draining so the producer never blocks on a full queue; the
                    # encoder's exit code carries the real error.
                    continue
                try:
                    encoder_proc.stdin.write(item)
                except BrokenPipeError:
                    failed = True
                except Exception as exc:
                    writer_errors.append(exc)
                    failed = True

        reader = threading.Thread(target=_read_frames, daemon=True)
        writer = threading.Thread(target=_write_frames, daemon=True)
        reader.start()
        writer.start()

        frame_count = 0
        decode_seconds = 0.0
        infer_seconds = 0.0
        encode_seconds = 0.0
        started = time.perf_counter()
        rgb_batch: list = []

        def _flush(sr, pre) -> None:
            nonlocal infer_seconds, encode_seconds
            if not rgb_batch:
                return
            infer_start = time.perf_counter()
            upscaled = self._upscale_rgb_batch(sr, rgb_batch, plan, pre)
            infer_seconds += time.perf_counter() - infer_start
            encode_start = time.perf_counter()
            for upscaled_rgb in upscaled:
                write_queue.put(upscaled_rgb.tobytes())
            encode_seconds += time.perf_counter() - encode_start
            rgb_batch.clear()

        try:
            sr = self._super_res(quality, plan.sr_width, plan.sr_height)
            pre = self._preprocess_effect(preprocess, in_width, in_height)
            while True:
                decode_start = time.perf_counter()
                item = frame_queue.get()
                decode_seconds += time.perf_counter() - decode_start
                if isinstance(item, Exception):
                    raise RuntimeError("Video decoding failed.") from item
                if item is None:
                    _flush(sr, pre)
                    break
                rgb_batch.append(np.frombuffer(item, dtype=np.uint8).reshape(in_height, in_width, 3))
                frame_count += 1
                if len(rgb_batch) >= batch_limit:
                    _flush(sr, pre)
        except BrokenPipeError:
            pass
        finally:
            with contextlib.suppress(Exception):
                decoder_proc.stdout.close()
            _stop_reader(reader, frame_queue, stop_reading)
            decoder_proc.wait()
            write_queue.put(None)
            writer.join()
            encoder_proc.stdin.close()
            stderr_handle.close()
            decode_log_handle.close()

        if writer_errors:
            raise RuntimeError("Writing frames to the encoder failed.") from writer_errors[0]

        return_code = encoder_proc.wait()
        if return_code != 0:
            stderr_tail = stderr_path.read_bytes()[-2000:].decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ffmpeg encode failed (encoder={self.encoder}, code={return_code}): {stderr_tail}"
            )
        if frame_count == 0:
            decode_tail = decode_log_path.read_bytes()[-2000:].decode("utf-8", errors="replace")
            raise ValueError(f"Uploaded video has no decodable frames. {decode_tail}")

        elapsed = max(time.perf_counter() - started, 1e-6)
        print(
            f"Video done: {frame_count} frames in {elapsed:.1f}s = {frame_count / elapsed:.1f} fps "
            f"(decode-wait {100 * decode_seconds / elapsed:.0f}%, "
            f"infer {100 * infer_seconds / elapsed:.0f}%, "
            f"encode-backpressure {100 * encode_seconds / elapsed:.0f}%; "
            f"nvdec={'yes' if use_nvdec else 'no'}, batch={batch_limit})"
        )

        stderr_path.unlink(missing_ok=True)
        decode_log_path.unlink(missing_ok=True)
        return output_name, "video/mp4"

    @modal.method()
    def run(
        self,
        job_id: str,
        input_name: str,
        mime_type: str | None,
        resize_type: str,
        scale: float,
        width: int,
        height: int,
        quality: str,
        keep_aspect_ratio: bool = True,
        preprocess: str | None = None,
        truehdr: str = "",
        master_display: str = "",
        max_cll: str = "",
    ) -> dict[str, str]:
        self._logged_hybrid = False
        self._logged_preprocess = False
        job_dir = JOBS_DIR / job_id
        input_path = job_dir / f"input{Path(input_name).suffix or '.bin'}"

        # A reload re-materializes the whole mount, so it must not run while another job
        # on this container is mid-flight: that job's scratch files under /jobs (input,
        # ffmpeg.log, partial output) are local-only until commit, and a reload wipes
        # them. Measured, not theorized — see MODAL_WORKER_CONCURRENCY in CLAUDE.md.
        # With the default concurrency of 1 this is exactly the old unconditional reload.
        with self._volume_lock:
            if self._active_jobs == 0:
                jobs_volume.reload()
            self._active_jobs += 1
        try:
            if not input_path.exists():
                raise FileNotFoundError(f"Input for job {job_id} is missing.")
        except BaseException:
            with self._volume_lock:
                self._active_jobs -= 1
            raise

        stem = Path(input_name).stem
        args = (
            input_path, job_dir, stem, resize_type, scale, width, height, quality,
            keep_aspect_ratio, preprocess,
        )
        # Video intermediates (the video-only stream, ffmpeg logs) live on local disk and
        # never reach the Volume. This is also the first step towards the scratch-off-Volume
        # prerequisite for MODAL_WORKER_CONCURRENCY > 1.
        scratch_dir = Path(tempfile.mkdtemp(prefix=f"rtx-{job_id}-"))
        try:
            # Plain strings in, parsed here as well as in the web tier, so a direct
            # run.remote() (scripts/verify_roundtrip.py) gets the same validation.
            try:
                hdr = parse_hdr_settings(truehdr, master_display, max_cll)
            except ValueError as exc:
                raise HdrRequestError(f"HDR10 request: {exc}") from None
            if _is_video(input_name, mime_type):
                output_name, media_type = self._upscale_video(
                    *args, scratch_dir=scratch_dir, hdr=hdr
                )
            elif hdr is not None:
                raise HdrRequestError("HDR10 request: TrueHDR output is for video inputs only.")
            else:
                output_name, media_type = self._upscale_image(*args)
        except BaseException:
            # /result cannot clean up after a failure (it only learns the job id from a
            # successful return), so drop the job's files here before the commit below.
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)
            input_path.unlink(missing_ok=True)
            import gc
            import torch

            gc.collect()
            torch.cuda.empty_cache()
            with self._volume_lock:
                self._active_jobs -= 1
                jobs_volume.commit()

        return {"job_id": job_id, "output_name": output_name, "media_type": media_type}


# The web tier talks to the Volume through its API, not a mount. With a mount, concurrent
# inputs share one view of the Volume: a /result reload() fails while another input has an
# upload open on it ("volume busy") and can drop other inputs' uncommitted writes. The
# API calls are atomic per file, so there is nothing to reload, commit, or serialize.
@app.function(
    image=web_image,
    timeout=900,
)
@modal.asgi_app(requires_proxy_auth=True)
@modal.concurrent(max_inputs=20)
def api():
    from urllib.parse import quote

    from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse, StreamingResponse

    web_app = FastAPI(
        title="RTX Media Upscaler",
        description="Submit an image/video, upscale it on an RTX GPU in Modal, then download the output.",
        version="2.0.0",
    )

    @web_app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "gpu": GPU_TYPE}

    @web_app.post("/upscale")
    async def upscale(
        file: UploadFile = File(...),
        resize_type: str = Form(UpscaleType.SCALE_BY.value),
        scale: float = Form(2.0),
        width: int = Form(1920),
        height: int = Form(1080),
        keep_aspect_ratio: bool = Form(True),
        # HIGHBITRATE_ULTRA rather than ULTRA: the standard family's artifact suppression
        # damages texture on sources that are soft but not heavily compressed. See CLAUDE.md.
        quality: str = Form("HIGHBITRATE_ULTRA"),
        preprocess: str = Form(""),
        # SDR->HDR10 (video only). Empty = SDR output, unchanged. "on" or NVEncC syntax
        # "contrast=102,saturation=102,middlegray=46,maxluminance=680[,debanding=off]";
        # master_display / max_cll take NVEncC's --master-display / --max-cll strings.
        # Empty max_cll = MaxCLL/MaxFALL measured per job and written into the stream.
        truehdr: str = Form(""),
        master_display: str = Form(""),
        max_cll: str = Form(""),
    ):
        try:
            if _normalize_quality(quality) not in UPSCALE_QUALITIES:
                raise ValueError(
                    f"quality must be an upscaling mode {sorted(UPSCALE_QUALITIES)}; "
                    f"same-resolution modes belong in `preprocess`."
                )
            if preprocess and _normalize_quality(preprocess) not in SAME_RES_QUALITIES:
                raise ValueError(
                    f"preprocess must be a same-resolution mode {sorted(SAME_RES_QUALITIES)}."
                )
            selected_type = _normalize_resize_type(resize_type)
            if selected_type == UpscaleType.SCALE_BY:
                if scale < 1.0:
                    raise ValueError("Scale must be >= 1.0 when resize_type is scale by multiplier.")
            else:
                if width < 1 or height < 1:
                    raise ValueError("Output dimensions must be positive.")
                if width > MAX_OUTPUT_EDGE or height > MAX_OUTPUT_EDGE:
                    raise ValueError(
                        f"Output dimension exceeds maximum supported edge of {MAX_OUTPUT_EDGE} (got {width}x{height})."
                    )
            if parse_hdr_settings(truehdr, master_display, max_cll) is not None and not _is_video(
                file.filename or "", file.content_type
            ):
                raise ValueError("truehdr (HDR10 output) is for video inputs only.")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        input_name = file.filename or "upload.bin"
        job_id = uuid.uuid4().hex
        remote_input = f"/{job_id}/input{Path(input_name).suffix or '.bin'}"

        # Starlette has already spooled the multipart body to a local temp file, so this
        # uploads from disk and a multi-GB file never lands in memory. The batch commits
        # on exit; the worker's reload() then sees it.
        upload = file.file
        upload.seek(0, os.SEEK_END)
        if upload.tell() == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        upload.seek(0)
        async with jobs_volume.batch_upload.aio() as batch:
            batch.put_file(upload, remote_input)

        call = await UpscaleWorker().run.spawn.aio(
            job_id=job_id,
            input_name=input_name,
            mime_type=file.content_type,
            resize_type=resize_type,
            scale=scale,
            width=width,
            height=height,
            quality=quality,
            keep_aspect_ratio=keep_aspect_ratio,
            preprocess=preprocess or None,
            truehdr=truehdr,
            master_display=master_display,
            max_cll=max_cll,
        )
        return {"call_id": call.object_id, "job_id": job_id}

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
            if (
                isinstance(exc, (UpscaleOnlyError, OutputDimensionExceededError, HdrRequestError))
                or "HDR10 request:" in msg
                or "Upscaling only" in msg
                or "Output dimension exceeds" in msg
                or "Unsupported resize type" in msg
                or "Output dimensions must be positive" in msg
                or "Scale must be >= 1.0" in msg
            ):
                raise HTTPException(status_code=400, detail=msg) from exc
            raise HTTPException(status_code=500, detail=f"Upscaling failed: {exc}") from exc

        output_path = f"/{job['job_id']}/{job['output_name']}"
        try:
            entries = await jobs_volume.listdir.aio(output_path)
        except (FileNotFoundError, modal.exception.NotFoundError):
            entries = []
        if not entries:
            raise HTTPException(status_code=410, detail="Output already downloaded or cleaned up.")

        # read_file prefetches up to cpu_count() 8 MiB blocks ahead of the client (17 on
        # the probed web container, ~136 MiB per download). There is no hard memory limit
        # on this function, so that is extra billed memory while a download streams, not
        # an OOM risk.
        async def _stream():
            async for chunk in jobs_volume.read_file.aio(output_path):
                yield chunk

        background_tasks.add_task(_cleanup_job_dir, job["job_id"])
        filename = job["output_name"]
        ascii_name = filename.encode("ascii", "replace").decode().replace('"', "_")
        return StreamingResponse(
            _stream(),
            media_type=job["media_type"],
            headers={
                # Content-Length lets the client detect a truncated download.
                "Content-Length": str(entries[0].size),
                "Content-Disposition": (
                    f'attachment; filename="{ascii_name}"; '
                    f"filename*=utf-8''{quote(filename)}"
                ),
            },
        )

    return web_app


async def _cleanup_job_dir(job_id: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        await jobs_volume.remove_file.aio(f"/{job_id}", recursive=True)

