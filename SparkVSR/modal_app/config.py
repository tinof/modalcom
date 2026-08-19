"""Constants and configuration for SparkVSR Modal deployment."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Modal app name
APP_NAME = "sparkvsr"

# Volume mount paths
MODEL_VOLUME_NAME = "sparkvsr-models"
IO_VOLUME_NAME = "sparkvsr-io"

MODEL_MOUNT_PATH = "/models"
IO_MOUNT_PATH = "/io"

# Model paths on Volume
SPARKVSR_MODEL_DIR = f"{MODEL_MOUNT_PATH}/SparkVSR"
EMPTY_PROMPT_EMBED_PATH = f"{MODEL_MOUNT_PATH}/empty_prompt_embedding.pt"
PISASR_DIR = f"{MODEL_MOUNT_PATH}/pisasr"
PISASR_SD21_DIR = f"{PISASR_DIR}/stable-diffusion-2-1-base"
PISASR_WEIGHTS_PATH = f"{PISASR_DIR}/pisa_sr.pkl"

# HuggingFace Repositories
SPARKVSR_HF_REPO = "JiongzeYu/SparkVSR"
# Stability delisted stabilityai/stable-diffusion-2-1-base from the Hub (404 even
# with a valid token). This community re-host carries the identical diffusers
# layout PiSASR_eval loads from (tokenizer/text_encoder/scheduler/vae/unet).
PISASR_SD21_HF_REPO = "sd2-community/stable-diffusion-2-1-base"
PISASR_WEIGHTS_FILENAME = "pisa_sr.pkl"

# PiSA-SR adapter weights are operator-supplied. Upstream (csslc/PiSA-SR) ships
# pisa_sr.pkl through Google Drive only; there is no public programmatic download
# (the ComfyUI-Spark README notes the HuggingFace path requires authentication).
# Provisioning therefore accepts an optional direct URL and otherwise instructs
# the operator to upload the file once:
#   modal volume put sparkvsr-models /path/to/pisa_sr.pkl pisasr/pisa_sr.pkl
PISASR_WEIGHTS_URL_ENV = "PISASR_WEIGHTS_URL"

# Official PiSA-SR inference scales (csslc/PiSA-SR test_pisasr.py defaults)
PISASR_LAMBDA_PIX = 1.0
PISASR_LAMBDA_SEM = 1.0

# GPU Hardware Configuration
#
# RTX PRO 6000 is the only sensible target. The NVENC support matrix rules out the
# datacenter compute cards (A100/H100/H200/B200 carry no NVENC, so the encoder ladder
# would fall back to CPU libx264 and lose the 10-bit HEVC path), and of the NVENC-capable
# cards only this one holds 4K untiled (49.1 GB peak) and offers Blackwell FP8 tensor
# cores. L40S (48 GB) fits 1080p but is Ada with roughly half the bf16 throughput.
GPU_TYPE = "RTX-PRO-6000"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- Inference optimisation knobs -------------------------------------------------------
#
# Measured baseline (12 s 1080p clip, one RTX PRO 6000): inference is 98% of wall clock at
# 1080p and 93% at 4K, while decode+upscale+blend+encode together are under 2%. Every knob
# here therefore targets the transformer forward; none of them touch I/O.
#
# All are env-overridable so a run can A/B a single knob without a code change.

# torch.compile the transformer. The single biggest same-hardware win, but it costs a
# one-off graph compile on the first window of a container (~1-3 min), which is only
# amortised when a container processes several windows.
TORCH_COMPILE = _env_flag("SPARKVSR_TORCH_COMPILE", True)
TORCH_COMPILE_MODE = os.environ.get("SPARKVSR_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs")

# Also compile the VAE decode. Smaller share of the forward, and its input shape varies
# with window length, so it recompiles more often. Off by default.
TORCH_COMPILE_VAE = _env_flag("SPARKVSR_TORCH_COMPILE_VAE", False)

# FP8 dynamic quantisation of the transformer's linear layers via torchao. Blackwell has
# native FP8 tensor cores. This is the one knob that can change output quality, so it stays
# off until an A/B on real frames justifies it.
FP8_QUANTIZE = _env_flag("SPARKVSR_FP8", False)

# Emit a torch.profiler trace for the first window, then exit that window early. Used once
# to attribute inference time across VAE-encode / transformer / attention / VAE-decode.
PROFILE_FIRST_WINDOW = _env_flag("SPARKVSR_PROFILE", False)

# --- Horizontal fan-out -----------------------------------------------------------------
#
# Windows are independent except where they overlap inside a shot, and segments are cut
# only on shot boundaries, so segments carry no cross-worker dependency at all. Fanning
# them across containers divides wall clock by the worker count at identical total cost.
PARALLEL = _env_flag("SPARKVSR_PARALLEL", True)
# Frames per parallel segment. Smaller means more workers and shorter wall clock, but each
# worker re-pays container start plus model load (~40 s), so very small segments waste GPU.
PARALLEL_SEGMENT_TARGET_FRAMES = int(os.environ.get("SPARKVSR_SEGMENT_FRAMES", "500"))
# Ceiling on simultaneous GPU containers. Modal's own plan limit (10 on Starter, 50 on
# Team) applies on top of this.
MAX_PARALLEL_CONTAINERS = int(os.environ.get("SPARKVSR_MAX_CONTAINERS", "10"))

# Container timeouts and lifecycle
CONTAINER_TIMEOUT = 3600  # 1 hour max, sized for ONE worker rendering one segment
CONTAINER_IDLE_TIMEOUT = 120  # 2 min scaledown window

# The fan-out driver blocks on every worker, so its budget is the whole job's wall clock,
# not one segment's. An episode is ~180 segments over MAX_PARALLEL_CONTAINERS workers, each
# round paying model load and possibly a compile. It is a CPU container at $0.63/hr, so a
# generous ceiling is cheap insurance against losing a paid fan-out to a driver timeout.
DRIVER_TIMEOUT = int(os.environ.get("SPARKVSR_DRIVER_TIMEOUT", str(12 * 3600)))

# A preempted or crashed worker would otherwise abort the whole map and discard every
# sibling's finished work. Segment rendering is deterministic and writes only its own file,
# so retrying one is safe.
WORKER_RETRIES = int(os.environ.get("SPARKVSR_WORKER_RETRIES", "2"))

# Container resources. Frames are streamed rather than accumulated, but decoding
# holds the source at native resolution and each window is materialised at the
# target resolution, so the worker still needs substantial host RAM.
CONTAINER_MEMORY_MB = 32768
CONTAINER_CPU_COUNT = 8

# Processing Defaults
DEFAULT_CHUNK_SIZE = 49  # 8n+1 length
DEFAULT_OVERLAP = 8
# Same-resolution restoration is the primary workflow: it recovers 7-9x the source's
# high-frequency detail (Laplacian variance 23 -> 163-211) at ~4.6x less GPU time than
# upscaling to 4K. Pass target_height=2160 explicitly to upscale.
DEFAULT_TARGET_WIDTH = 1920
DEFAULT_TARGET_HEIGHT = 1080
DEFAULT_SR_NOISE_STEP = 399
DEFAULT_REF_GUIDANCE_SCALE = 1.0

# Supported Reference Modes
REF_MODE_PISASR = "pisasr"
REF_MODE_API = "api"
REF_MODE_NO_REF = "no_ref"
VALID_REF_MODES = frozenset({REF_MODE_PISASR, REF_MODE_API, REF_MODE_NO_REF})
DEFAULT_REF_MODE = REF_MODE_PISASR

# Cut detection defaults
DEFAULT_CUT_THRESHOLD = 3.0
DEFAULT_MIN_SCENE_LEN_SEC = 0.6
MIN_REF_SPACING = 5  # References must be >4 frames apart (>=5)
MAX_REF_WINDOW_OFFSET_SEC = 0.5

# Validation Bounds
TARGET_HEIGHT_MIN = 256
TARGET_HEIGHT_MAX = 4320
TARGET_WIDTH_MIN = 256
TARGET_WIDTH_MAX = 7680

CHUNK_SIZE_MIN = 9  # 8*1 + 1 minimum
CHUNK_SIZE_MAX = 257
OVERLAP_MIN = 0
OVERLAP_MAX = 32

REF_GUIDANCE_MIN = 1.0
REF_GUIDANCE_MAX = 10.0

# IO and Storage Paths
JOBS_PREFIX = "jobs"
JOBS_DIR = f"{IO_MOUNT_PATH}/{JOBS_PREFIX}"
UPLOAD_CHUNK = 8 * 1024 * 1024  # 8 MB chunks for streaming uploads

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ts", ".flv"})


def _bounded(name: str, value: float, low: float, high: float) -> float:
    if value < low or value > high:
        raise ValueError(f"{name} must be between {low} and {high} (got {value}).")
    return value


def validate_request(
    *,
    target_height: Optional[int] = DEFAULT_TARGET_HEIGHT,
    target_width: Optional[int] = None,
    ref_mode: str = DEFAULT_REF_MODE,
    ref_guidance_scale: float = DEFAULT_REF_GUIDANCE_SCALE,
    cut_aware: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    keep_audio: bool = True,
    tile: bool = False,
    tile_size: Optional[int] = None,
    custom_ref_indices: Optional[Sequence[int]] = None,
    check_fal_key: bool = False,
) -> Dict[str, Any]:
    """Validate and normalize super-resolution request parameters before GPU invocation.

    Raises ValueError on any invalid or out-of-range parameter.
    """
    mode = (ref_mode or DEFAULT_REF_MODE).strip().lower()
    if mode not in VALID_REF_MODES:
        raise ValueError(f"ref_mode must be one of {sorted(VALID_REF_MODES)} (got {ref_mode!r}).")

    if mode == REF_MODE_API and check_fal_key:
        if not (os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY")):
            raise ValueError("ref_mode='api' requires the 'fal' Secret with FAL_KEY configured.")

    if target_height is not None:
        _bounded("target_height", int(target_height), TARGET_HEIGHT_MIN, TARGET_HEIGHT_MAX)
        target_height = int(target_height)

    if target_width is not None:
        _bounded("target_width", int(target_width), TARGET_WIDTH_MIN, TARGET_WIDTH_MAX)
        target_width = int(target_width)

    _bounded("ref_guidance_scale", float(ref_guidance_scale), REF_GUIDANCE_MIN, REF_GUIDANCE_MAX)
    ref_guidance_scale = float(ref_guidance_scale)

    _bounded("chunk_size", int(chunk_size), CHUNK_SIZE_MIN, CHUNK_SIZE_MAX)
    chunk_size = int(chunk_size)
    if (chunk_size - 1) % 8 != 0:
        raise ValueError(f"chunk_size must satisfy 8n+1 (e.g. 9, 17, 25, 33, 41, 49), got {chunk_size}.")

    _bounded("overlap", int(overlap), OVERLAP_MIN, OVERLAP_MAX)
    overlap = int(overlap)
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be strictly smaller than chunk_size ({chunk_size}).")

    # Validate custom reference spacing if provided
    if custom_ref_indices:
        sorted_refs = sorted(int(x) for x in custom_ref_indices)
        for i in range(len(sorted_refs) - 1):
            diff = sorted_refs[i + 1] - sorted_refs[i]
            if diff < MIN_REF_SPACING:
                raise ValueError(
                    f"Reference indices {sorted_refs[i]} and {sorted_refs[i + 1]} are spaced only "
                    f"{diff} frames apart. Reference spacing must be > 4 frames (>= 5 frames)."
                )

    if tile and tile_size is None:
        tile_size = 512

    return {
        "target_height": target_height,
        "target_width": target_width,
        "ref_mode": mode,
        "ref_guidance_scale": ref_guidance_scale,
        "cut_aware": bool(cut_aware),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "keep_audio": bool(keep_audio),
        "tile_size": tile_size,
    }
