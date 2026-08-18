"""Constants for FlashVSR-Pro Modal deployment."""

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
# NOTE: Blackwell is sm_120 -- it requires the torch 2.13 / CUDA 13 stack pinned
# in image.py, and sm_120 must be in the BSA arch list at kernel-build time.
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
VALID_DTYPES = frozenset({"fp32", "fp16", "bf16"})

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
