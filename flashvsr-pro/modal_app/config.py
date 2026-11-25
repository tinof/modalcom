"""Constants for the FlashVSR Modal deployments.

The first part configures the original FlashVSR v1.1 service (`flashvsr-pro`).
The `PLUS_*` section at the end configures the FlashVSR+ segment worker
(`flashvsr-plus`). Everything here is plain Python: the CPU tiers and the local
tests import it without torch or Modal.
"""

import os

# Modal app name (also used by the web tier to look up deployed GPU classes)
APP_NAME = "flashvsr-pro"

# Volume mount paths
MODEL_VOLUME_NAME = "flashvsr-models"
IO_VOLUME_NAME = "flashvsr-io"
# Caches the compiled Block-Sparse-Attention wheel so the ~hour-long CUDA
# compile is paid once, not on every image rebuild.
BUILD_CACHE_VOLUME_NAME = "flashvsr-build-cache"
MODEL_MOUNT_PATH = "/models"
IO_MOUNT_PATH = "/io"

# Model file paths (relative to MODEL_MOUNT_PATH)
MODEL_DIR = f"{MODEL_MOUNT_PATH}/FlashVSR-v1.1"
DIT_PATH = f"{MODEL_DIR}/diffusion_pytorch_model_streaming_dmd.safetensors"
WAN_VAE_PATH = f"{MODEL_DIR}/Wan2.1_VAE.pth"
TCD_VAE_PATH = f"{MODEL_DIR}/TCDecoder.ckpt"
LQ_PROJ_PATH = f"{MODEL_DIR}/LQ_proj_in.ckpt"
PROMPT_TENSOR_PATH = f"{MODEL_MOUNT_PATH}/prompt_tensor/posi_prompt.pth"

# HuggingFace repo for weight downloads
HF_REPO_ID = "JunhaoZhuang/FlashVSR-v1.1"
HF_FILES = [
    ("diffusion_pytorch_model_streaming_dmd.safetensors", MODEL_DIR),
    ("Wan2.1_VAE.pth", MODEL_DIR),
    ("TCDecoder.ckpt", MODEL_DIR),
    ("LQ_proj_in.ckpt", MODEL_DIR),
]
# Prompt tensor from 1038lab/FlashVSR (safetensors format, converted to .pth at download)
PROMPT_TENSOR_HF_REPO = "1038lab/FlashVSR"
PROMPT_TENSOR_HF_FILE = "Prompt.safetensors"
PROMPT_TENSOR_DIR = f"{MODEL_MOUNT_PATH}/prompt_tensor"

# GPU configurations. All three modes run on Blackwell RTX PRO 6000: 96 GB of
# VRAM removes the 1080p->4K OOM risk that A10G (24 GB) had in tiny mode, and it
# is the only Modal GPU with a 9th-gen NVENC encoder for hardware video encode.
# NOTE: Blackwell is sm_120 -- it requires the torch 2.12 / CUDA 13 stack pinned
# in bsa_build.py, and sm_120 must be in the BSA arch list at kernel-build time.
GPU_FULL = "RTX-PRO-6000"
GPU_TINY = "RTX-PRO-6000"
GPU_TINY_LONG = "RTX-PRO-6000"

# Container timeouts (seconds)
CONTAINER_IDLE_TIMEOUT = 300  # 5 min warm
CONTAINER_TIMEOUT = 1800  # 30 min max per request

# I/O paths
IO_INPUT_DIR = f"{IO_MOUNT_PATH}/inputs"
IO_OUTPUT_DIR = f"{IO_MOUNT_PATH}/outputs"

# HTTP job workspace. `inputs/` and `outputs/` stay reserved for the
# `modal volume put/get` power-user workflow; HTTP jobs are self-cleaning and
# live under `jobs/<job_id>/` on the same volume.
JOBS_PREFIX = "jobs"
JOBS_DIR = f"{IO_MOUNT_PATH}/{JOBS_PREFIX}"
UPLOAD_CHUNK = 8 * 1024 * 1024

# ---------------------------------------------------------------------------
# Web-tier validation
#
# These are plain frozensets and pure-Python helpers on purpose: the web tier
# runs on a slim CPU image with no torch, no diffsynth, and no CUDA kernels, so
# it cannot import the pipeline to ask what a valid request looks like. Rejecting
# a bad request here costs milliseconds; discovering it inside a GPU container
# costs a cold start plus model load.
# ---------------------------------------------------------------------------
VALID_MODES = frozenset({"full", "tiny", "tiny-long"})
MODE_ALIASES = {"tiny_long": "tiny-long", "tinylong": "tiny-long", "long": "tiny-long"}
# The workers load their models once per container, in bf16. `fp32` and `fp16`
# used to be accepted and then applied to the input tensor only, which fed the
# bf16 models a tensor of another dtype. Until a worker can load another dtype,
# bf16 is the only honest answer.
VALID_DTYPES = frozenset({"bf16"})

MODE_GPUS = {"full": GPU_FULL, "tiny": GPU_TINY, "tiny-long": GPU_TINY_LONG}
MODE_CLASSES = {
    "full": "FlashVSRFull",
    "tiny": "FlashVSRTiny",
    "tiny-long": "FlashVSRTinyLong",
}

SCALE_MIN, SCALE_MAX = 1.0, 4.0
QUALITY_MIN, QUALITY_MAX = 1, 10  # imageio/ffmpeg encode quality, not a model tier
TILE_SIZE_MIN, TILE_SIZE_MAX = 64, 1024
OVERLAP_MIN, OVERLAP_MAX = 0, 256
LOCAL_RANGE_MIN, LOCAL_RANGE_MAX = 1, 64
SPARSE_RATIO_MIN, SPARSE_RATIO_MAX = 0.1, 16.0
KV_RATIO_MIN, KV_RATIO_MAX = 0.1, 16.0

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"})


def temp_sibling(path: str) -> str:
    """`<name>_temp<ext>` beside `path`, for any extension.

    The audio fallback used `path.replace(".mp4", "_temp.mp4")`. For an output
    not named `.mp4` that returned `path` itself, and the finished output was
    then deleted as the "temporary" file.
    """
    root, extension = os.path.splitext(path)
    return f"{root}_temp{extension}"


def normalize_mode(mode: str) -> str:
    """Canonicalize a mode string. Raises ValueError on anything unknown."""
    candidate = (mode or "").strip().lower()
    candidate = MODE_ALIASES.get(candidate, candidate)
    if candidate not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)} (got {mode!r}).")
    return candidate


def _bounded(name: str, value, low, high):
    if value < low or value > high:
        raise ValueError(f"{name} must be between {low} and {high} (got {value}).")
    return value


def validate_request(
    *,
    mode: str = "tiny",
    scale: float = 2.0,
    seed: int = 0,
    sparse_ratio: float = 2.0,
    kv_ratio: float = 3.0,
    local_range: int = 11,
    color_fix: bool = False,
    fps: float | None = None,
    quality: int = 10,
    keep_audio: bool = False,
    tile_dit: bool = False,
    tile_vae: bool | None = None,
    tile_size: int = 256,
    overlap: int = 24,
    dtype: str = "bf16",
) -> dict:
    """Validate and normalize inference parameters. Raises ValueError on bad input.

    Returns a kwargs dict suitable for constructing an `InferenceRequest`.
    """
    mode = normalize_mode(mode)

    # Hard constraint: DiT tiling produces visible grid artifacts on every
    # source we have tested. It is not a tradeoff, it is a broken path.
    if tile_dit:
        raise ValueError(
            "tile_dit is not supported: DiT tiling produces grid artifacts in the output. "
            "Use tile_vae for memory relief instead."
        )

    # Full mode decodes with the Wan2.1 VAE and peaks near 79 GB without VAE
    # tiling. Default it on rather than OOM-ing, but never silently override an
    # explicit request to disable it -- say why instead.
    if mode == "full":
        if tile_vae is False:
            raise ValueError(
                "mode='full' requires tile_vae=True; without it the Wan2.1 VAE decode "
                "exceeds 80 GB of VRAM and the job dies mid-decode."
            )
        tile_vae = True
    else:
        tile_vae = bool(tile_vae)

    if dtype not in VALID_DTYPES:
        raise ValueError(f"dtype must be one of {sorted(VALID_DTYPES)} (got {dtype!r}).")

    _bounded("scale", float(scale), SCALE_MIN, SCALE_MAX)
    _bounded("quality", int(quality), QUALITY_MIN, QUALITY_MAX)
    _bounded("tile_size", int(tile_size), TILE_SIZE_MIN, TILE_SIZE_MAX)
    _bounded("overlap", int(overlap), OVERLAP_MIN, OVERLAP_MAX)
    _bounded("local_range", int(local_range), LOCAL_RANGE_MIN, LOCAL_RANGE_MAX)
    _bounded("sparse_ratio", float(sparse_ratio), SPARSE_RATIO_MIN, SPARSE_RATIO_MAX)
    _bounded("kv_ratio", float(kv_ratio), KV_RATIO_MIN, KV_RATIO_MAX)

    if int(overlap) >= int(tile_size):
        raise ValueError(f"overlap ({overlap}) must be smaller than tile_size ({tile_size}).")

    if fps is not None and not (0 < float(fps) <= 240):
        raise ValueError(f"fps must be between 0 and 240 (got {fps}).")

    return {
        "mode": mode,
        "scale": float(scale),
        "seed": int(seed),
        "sparse_ratio": float(sparse_ratio),
        "kv_ratio": float(kv_ratio),
        "local_range": int(local_range),
        "color_fix": bool(color_fix),
        "fps": None if fps is None else float(fps),
        "quality": int(quality),
        "keep_audio": bool(keep_audio),
        "tile_dit": False,
        "tile_vae": bool(tile_vae),
        "tile_size": int(tile_size),
        "overlap": int(overlap),
        "dtype": dtype,
    }


# ---------------------------------------------------------------------------
# FlashVSR+ (OpenImagingLab/FlashVSR_Plus) segment worker
#
# Values that an operator may want to vary are read from the environment. They
# are read again at container import, where the deploying shell's environment
# does not exist, so `plus_image.py` bakes every one of them into the image with
# `.env()`. A knob added here and not there silently measures the default.
# ---------------------------------------------------------------------------
PLUS_APP_NAME = os.getenv("FLASHVSR_PLUS_APP_NAME", "flashvsr-plus")
PLUS_GPU = os.getenv("FLASHVSR_PLUS_GPU", "RTX-PRO-6000")
PLUS_WORKER_CPU = float(os.getenv("FLASHVSR_PLUS_CPU", "6"))
PLUS_WORKER_MEMORY_MIB = int(os.getenv("FLASHVSR_PLUS_MEMORY_MIB", "32768"))
# Upper bound on GPU containers for one app. It limits wall time, never cost:
# the same segments cost the same GPU-seconds on one container or on ten.
PLUS_MAX_CONTAINERS = int(os.getenv("FLASHVSR_PLUS_MAX_CONTAINERS", "8"))

# Upstream is pinned to a commit and cloned at image-build time. Our wrapper is
# written against the runner and decoder code at exactly this commit
# (`plus_runtime.py`), so move the pin and re-read those files together.
PLUS_REPO_URL = "https://github.com/OpenImagingLab/FlashVSR_Plus.git"
PLUS_REPO_SHA = "b515ffc6dba57fdc9ad84c66cadd87d05f27590a"
PLUS_REPO_DIR = "/opt/flashvsr_plus"

# ffmpeg for both the workers and the orchestrator: rtx-vsr's pinned BtbN n8.1
# build. Debian's ffmpeg lags on NVENC, and BtbN's `master` builds link a Video
# Codec SDK that needs driver 610+, where every NVENC encoder fails to
# initialise on Modal's 580 hosts. The checksum makes a changed artifact fail
# the image build instead of degrading at run time.
PLUS_FFMPEG_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-09-30-13-08"
    "/ffmpeg-n8.1.3-9-g29e619e767-linux64-gpl-8.1.tar.xz"
)
PLUS_FFMPEG_SHA256 = "97ce978979194b5cf7e06a5e68020dbdaa7a4f3294c5452b6a1bc347100dbb79"
PLUS_FFMPEG_INSTALL = (
    f"curl -fsSL -o /tmp/ff.tar.xz {PLUS_FFMPEG_URL}",
    f"echo '{PLUS_FFMPEG_SHA256}  /tmp/ff.tar.xz' | sha256sum -c -",
    "mkdir -p /opt/ffmpeg",
    "tar -xJf /tmp/ff.tar.xz -C /opt/ffmpeg --strip-components=1",
    "cp /opt/ffmpeg/bin/ffmpeg /opt/ffmpeg/bin/ffprobe /usr/local/bin/",
    "rm -rf /tmp/ff.tar.xz /opt/ffmpeg",
    "ffmpeg -version",
)
PLUS_NUMPY = "numpy==2.2.6"

PLUS_MODEL_VOLUME_NAME = "flashvsr-plus-models"
PLUS_IO_VOLUME_NAME = "flashvsr-plus-io"
PLUS_MODEL_MOUNT_PATH = "/models"
PLUS_IO_MOUNT_PATH = "/io"
PLUS_CKPT_DIR = f"{PLUS_MODEL_MOUNT_PATH}/ckpts"
PLUS_TINY_DECODER_PATH = f"{PLUS_CKPT_DIR}/TCDecoder.ckpt"
# The prompt tensor ships inside the upstream repository.
PLUS_CONTEXT_PATH = f"{PLUS_REPO_DIR}/prompt_tensor/posi_prompt.pth"

PLUS_HF_REPO = "ruchangyao/FlashVSR_plus"
# (file in the Hub repo, path below PLUS_CKPT_DIR)
PLUS_HF_FILES = (
    ("turbo/dit.safetensors", "turbo/dit.safetensors"),
    ("turbo/lq_proj_in.safetensors", "turbo/lq_proj_in.safetensors"),
    ("base/dit.safetensors", "base/dit.safetensors"),
    ("base/lq_proj_in.safetensors", "base/lq_proj_in.safetensors"),
    ("TCDecoder.ckpt", "TCDecoder.ckpt"),
)
# The `v1` preset loads the FlashVSR v1.1 weights under upstream's file names.
# The DiT is copied as is; the LQ projector is a torch checkpoint upstream wants
# as safetensors, so it is converted once at download time.
PLUS_V1_HF_REPO = HF_REPO_ID
PLUS_V1_DIT_FILE = "diffusion_pytorch_model_streaming_dmd.safetensors"
PLUS_V1_LQ_PROJ_FILE = "LQ_proj_in.ckpt"

PLUS_MODES = ("turbo", "base", "v1")
PLUS_ATTN_KERNELS = ("auto", "block_sparse", "triton_block_sparse", "sdpa")
PLUS_TINY_DECODER_MODES = ("auto", "serial", "parallel")
# Model canvas sides must be multiples of this (upstream INPUT_MULTIPLE).
PLUS_CANVAS_MULTIPLE = 128
# Latents per TinyDecoder call, a multiple of upstream's fixed 4-latent chunk.
# 28 latents are 112 frames: the README's own example value.
PLUS_DECODE_EVERY = 28

# Job layout on the io Volume: jobs/<job_id>/{input<ext>,plan.json,segments/,output.mkv}
PLUS_JOBS_PREFIX = "jobs"

# Segment planning (see plus_plan.py).
PLUS_SEGMENT_SECONDS = float(os.getenv("FLASHVSR_PLUS_SEGMENT_SECONDS", "60"))
PLUS_SEGMENT_SEARCH_SECONDS = float(os.getenv("FLASHVSR_PLUS_SEGMENT_SEARCH_SECONDS", "30"))
# ffmpeg scene score (0..1) a frame must reach to count as a hard cut.
PLUS_SCENE_THRESHOLD = float(os.getenv("FLASHVSR_PLUS_SCENE_THRESHOLD", "0.30"))
# Where no hard cut exists the two segments overlap by this many output frames,
# which are cross-faded, and the later segment first runs warm-up frames through
# the model so its KV cache is not cold at the seam. The cache holds 3 temporal
# windows of 2 latents, i.e. 24 frames of history; 32 is two full 16-frame
# stream chunks and covers it.
PLUS_SEAM_OVERLAP_FRAMES = 8
PLUS_SEAM_WARMUP_FRAMES = 32

# Encoder. The quality point is rtx-vsr's, chosen there by VMAF against a
# lossless encode (rtx-vsr/CLAUDE.md, "The video path"): HEVC Main10, CQ 20,
# preset p5, UHQ tuning, two-pass full resolution, AQ, 5 B-frames, a 100 Mbit/s
# ceiling with a 1 s buffer, explicit level 5.1 High tier.
PLUS_VIDEO_ENCODER = os.getenv("FLASHVSR_PLUS_VIDEO_ENCODER", "hevc_nvenc")
PLUS_NVENC_PRESET = os.getenv("FLASHVSR_PLUS_NVENC_PRESET", "p5")
PLUS_NVENC_CQ = os.getenv("FLASHVSR_PLUS_NVENC_CQ", "20")
PLUS_NVENC_MAX_MBPS = int(os.getenv("FLASHVSR_PLUS_NVENC_MAX_MBPS", "100"))
PLUS_NVENC_LEVEL = os.getenv("FLASHVSR_PLUS_NVENC_LEVEL", "5.1")
# Level 5.x MaxLumaPs; larger outputs leave the level to the driver.
PLUS_HEVC_LEVEL5_MAX_PIXELS = 8_912_896

# Every FLASHVSR_PLUS_* knob above, for plus_image.py to bake into the image.
PLUS_ENV = {
    "FLASHVSR_PLUS_APP_NAME": PLUS_APP_NAME,
    "FLASHVSR_PLUS_GPU": PLUS_GPU,
    "FLASHVSR_PLUS_CPU": str(PLUS_WORKER_CPU),
    "FLASHVSR_PLUS_MEMORY_MIB": str(PLUS_WORKER_MEMORY_MIB),
    "FLASHVSR_PLUS_MAX_CONTAINERS": str(PLUS_MAX_CONTAINERS),
    "FLASHVSR_PLUS_SEGMENT_SECONDS": str(PLUS_SEGMENT_SECONDS),
    "FLASHVSR_PLUS_SEGMENT_SEARCH_SECONDS": str(PLUS_SEGMENT_SEARCH_SECONDS),
    "FLASHVSR_PLUS_SCENE_THRESHOLD": str(PLUS_SCENE_THRESHOLD),
    "FLASHVSR_PLUS_VIDEO_ENCODER": PLUS_VIDEO_ENCODER,
    "FLASHVSR_PLUS_NVENC_PRESET": PLUS_NVENC_PRESET,
    "FLASHVSR_PLUS_NVENC_CQ": PLUS_NVENC_CQ,
    "FLASHVSR_PLUS_NVENC_MAX_MBPS": str(PLUS_NVENC_MAX_MBPS),
    "FLASHVSR_PLUS_NVENC_LEVEL": PLUS_NVENC_LEVEL,
}


def validate_plus_params(
    *,
    mode: str = "turbo",
    scale: float = 2.0,
    deliver_scale: float | None = None,
    seed: int = 1,
    attn_kernel: str = "auto",
    tiny_decoder_mode: str = "auto",
    color_correction: bool = False,
    resident: bool = True,
    rope_float32: bool = False,
    compile_dit: bool = False,
    bypass_model: bool = False,
    decode_every: int = PLUS_DECODE_EVERY,
) -> dict:
    """Validate and normalize the parameters of a FlashVSR+ job.

    `scale` is what the model renders at. `deliver_scale`, when set, is the size
    the encoder resamples to (Lanczos), so a cheaper 1.5x render can still be
    delivered at 2x. `bypass_model` replaces the model with the bicubic upscale
    it would have been conditioned on: the I/O path alone, for colour and
    frame-count checks.
    """
    mode = (mode or "").strip().lower()
    if mode not in PLUS_MODES:
        raise ValueError(f"mode must be one of {list(PLUS_MODES)} (got {mode!r}).")
    if attn_kernel not in PLUS_ATTN_KERNELS:
        raise ValueError(f"attn_kernel must be one of {list(PLUS_ATTN_KERNELS)} (got {attn_kernel!r}).")
    if tiny_decoder_mode not in PLUS_TINY_DECODER_MODES:
        raise ValueError(
            f"tiny_decoder_mode must be one of {list(PLUS_TINY_DECODER_MODES)} (got {tiny_decoder_mode!r})."
        )
    _bounded("scale", float(scale), SCALE_MIN, SCALE_MAX)
    if deliver_scale is not None:
        _bounded("deliver_scale", float(deliver_scale), SCALE_MIN, SCALE_MAX)
    decode_every = int(decode_every)
    if decode_every <= 0 or decode_every % 4:
        raise ValueError(f"decode_every must be a positive multiple of 4 (got {decode_every}).")
    return {
        "mode": mode,
        "scale": float(scale),
        "deliver_scale": None if deliver_scale is None else float(deliver_scale),
        "seed": int(seed),
        "attn_kernel": attn_kernel,
        "tiny_decoder_mode": tiny_decoder_mode,
        "color_correction": bool(color_correction),
        "resident": bool(resident),
        "rope_float32": bool(rope_float32),
        "compile_dit": bool(compile_dit),
        "bypass_model": bool(bypass_model),
        "decode_every": decode_every,
    }
