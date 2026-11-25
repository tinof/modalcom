"""Modal Image definition for the FlashVSR v1.1 service.

The stack pins and the Block-Sparse-Attention build step live in `bsa_build.py`,
shared with the FlashVSR+ image (`plus_image.py`). The Blackwell target (RTX PRO
6000, sm_120) needs the CUDA 13 / torch 2.12 stack pinned there; the fallback
stack named beside each pin cannot run on that GPU at all, so falling back also
means reverting the GPU tiers in config.py to A100-80GB / A10G.

`modal_app/` is attached last as a mount, so code edits never touch the kernel
layer.
"""

import modal

from .app import build_cache_volume
from .bsa_build import (
    BLOCK_SPARSE_ATTN_CUDA_ARCHS,
    BSA_SRC,
    BUILD_CACHE_MOUNT,
    CUDA_BASE_IMAGE,
    PYTHON_VERSION,
    TORCH_PACKAGES,
    _build_or_install_bsa,
)

__all__ = [
    "BLOCK_SPARSE_ATTN_CUDA_ARCHS",
    "BSA_SRC",
    "BUILD_CACHE_MOUNT",
    "CUDA_BASE_IMAGE",
    "PYTHON_VERSION",
    "TORCH_PACKAGES",
    "flashvsr_image",
]

# Cache-efficient layer ordering: base → system → pytorch → deps → project → CUDA kernels
flashvsr_image = (
    modal.Image.from_registry(
        CUDA_BASE_IMAGE,
        add_python=PYTHON_VERSION,
    )
    # Layer 1: System packages (rarely change)
    .apt_install(
        "ffmpeg",
        "git",
        "git-lfs",
        "build-essential",
        "ninja-build",
        "wget",
        "curl",
    )
    # Layer 2: PyTorch (pinned, rarely changes)
    .uv_pip_install(*TORCH_PACKAGES)
    # Layer 3: Python dependencies (pinned versions from requirements.txt)
    .uv_pip_install(
        *TORCH_PACKAGES,
        "torchmetrics==1.7.3",
        "torchsde==0.2.6",
        "accelerate==1.8.1",
        "einops==0.8.1",
        "huggingface-hub==0.34.4",
        "matplotlib==3.10.3",
        # numpy stays on the 1.x line: diffsynth and its vendored deps assume it.
        # Migrating numpy and torch in the same change makes a failure ambiguous.
        "numpy==1.26.4",
        "opencv-python-headless==4.11.0.86",
        "peft==0.16.0",
        "pillow==11.0.0",
        "safetensors==0.5.3",
        "sentencepiece==0.2.0",
        "transformers==4.46.2",
        "pytorch-lightning==2.5.2",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "protobuf==3.20.3",
        "ftfy==6.3.1",
        "pandas==2.3.0",
        "tqdm==4.67.1",
        "datasets==3.2.0",
        "ffmpeg-python==0.2.0",
        "modelscope==1.22.3",
        # Build-time deps for the BSA setup.py (Python 3.12 venvs ship none of these)
        "wheel==0.45.1",
        "setuptools==75.8.0",
        "packaging==24.2",
        "ninja==1.11.1.3",
        "psutil==6.1.1",
    )
    # Layer 4: Copy project source onto PYTHONPATH
    .add_local_dir("diffsynth", "/workspace/FlashVSR-Pro/diffsynth", copy=True)
    .add_local_dir("utils", "/workspace/FlashVSR-Pro/utils", copy=True)
    .add_local_file("setup.py", "/workspace/FlashVSR-Pro/setup.py", copy=True)
    .add_local_file("requirements.txt", "/workspace/FlashVSR-Pro/requirements.txt", copy=True)
    .add_local_file("infer.py", "/workspace/FlashVSR-Pro/infer.py", copy=True)
    .workdir("/workspace/FlashVSR-Pro")
    # Layer 5: Block-Sparse-Attention CUDA kernels via the wheel cache
    .add_local_dir("Block-Sparse-Attention", BSA_SRC, copy=True)
    .run_function(
        _build_or_install_bsa,
        gpu="RTX-PRO-6000",
        cpu=16,
        memory=65536,
        timeout=5400,
        volumes={BUILD_CACHE_MOUNT: build_cache_volume},
    )
    # Layer 6: Environment tuning
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True",
        "PYTHONPATH": "/workspace/FlashVSR-Pro",
    })
    # Layer 7: this package, last and as a mount rather than a copied build layer,
    # so edits to modal_app/ never invalidate the kernel layer above.
    .add_local_python_source("modal_app")
)
