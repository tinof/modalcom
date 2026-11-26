"""GPU image for the FlashVSR+ worker.

It reuses the stack the v1.1 image already proved on Blackwell (`bsa_build.py`):
CUDA 13 devel base, Python 3.12, torch 2.12.1, and the Block-Sparse-Attention
(BSA) kernels from the wheel cache. Upstream's requirements file asks for
`torch>=2.5,<2.8`, but its own install skips that file
(`pip install -e . --no-deps`), so the range is advice, not a constraint. It is
untested here. If upstream breaks on torch 2.12, the fallback is `torch==2.7.1`
from the cu128 index on a CUDA 12.8 base: the newest line inside upstream's
range that supports sm_120. The wheel cache key includes the torch version, so
BSA then rebuilds by itself.

Upstream is cloned at a pinned commit instead of vendored. `plus_runtime.py` is
written against that commit's runner and decoder.

Only the BSA sources are copied from this repository, so `modal deploy` must run
from the project directory (the path is relative). The working directory inside
the container is the parent of that copy, which keeps the relative path valid
when this module is imported there.
"""

import modal

from .bsa_build import (
    BSA_SRC,
    BUILD_CACHE_MOUNT,
    CUDA_BASE_IMAGE,
    PYTHON_VERSION,
    TORCH_PACKAGES,
    _build_or_install_bsa,
    build_cache_volume,
)
from .config import (
    PLUS_CKPT_DIR,
    PLUS_ENV,
    PLUS_FFMPEG_INSTALL,
    PLUS_NUMPY,
    PLUS_REPO_DIR,
    PLUS_REPO_SHA,
    PLUS_REPO_URL,
)

_WORKDIR = BSA_SRC.rsplit("/", 1)[0]

plus_image = (
    modal.Image.from_registry(CUDA_BASE_IMAGE, add_python=PYTHON_VERSION)
    .apt_install("git", "build-essential", "ninja-build", "curl", "xz-utils")
    .run_commands(*PLUS_FFMPEG_INSTALL)
    .uv_pip_install(*TORCH_PACKAGES)
    .uv_pip_install(
        *TORCH_PACKAGES,
        # Upstream's runtime dependencies (its requirements.txt), pinned.
        "einops==0.8.1",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        PLUS_NUMPY,
        "opencv-python-headless==4.11.0.86",
        "pillow==11.0.0",
        "safetensors==0.5.3",
        "tqdm==4.67.1",
        "huggingface-hub==0.34.4",
        # Build-time deps for the BSA setup.py (Python 3.12 venvs ship none of these).
        "wheel==0.45.1",
        "setuptools==75.8.0",
        "packaging==24.2",
        "ninja==1.11.1.3",
        "psutil==6.1.1",
    )
    .run_commands(
        f"git clone {PLUS_REPO_URL} {PLUS_REPO_DIR}",
        f"cd {PLUS_REPO_DIR} && git checkout --detach {PLUS_REPO_SHA}",
        f"cd {PLUS_REPO_DIR} && python -m pip install -e . --no-deps",
        # Upstream's CLI and presets look for weights under <repo>/ckpts. Point
        # that at the weights Volume, which is mounted at run time.
        f"rm -rf {PLUS_REPO_DIR}/ckpts && ln -s {PLUS_CKPT_DIR} {PLUS_REPO_DIR}/ckpts",
    )
    .workdir(_WORKDIR)
    # Block-Sparse-Attention kernels through the wheel cache (see bsa_build.py).
    .add_local_dir("Block-Sparse-Attention", BSA_SRC, copy=True)
    .run_function(
        _build_or_install_bsa,
        gpu="RTX-PRO-6000",
        cpu=16,
        memory=65536,
        timeout=5400,
        volumes={BUILD_CACHE_MOUNT: build_cache_volume},
    )
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        **PLUS_ENV,
    })
    .add_local_python_source("modal_app")
)
