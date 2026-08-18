"""Modal Image definition for FlashVSR-Pro.

Stack pins live in the constants below; each names its previously verified
fallback. The Blackwell target (RTX PRO 6000, sm_120) requires the CUDA 13 /
torch 2.13 stack — the fallback stack cannot run on that GPU at all, so falling
back also means reverting the GPU tiers in config.py to A100-80GB / A10G.

The Block-Sparse-Attention CUDA kernels are the expensive part (~25 heavy
cutlass translation units). They are built once by `_build_or_install_bsa` into
a wheel cached on the `flashvsr-build-cache` Volume, keyed on source hash +
torch version + arch list. Rebuilding the image with unchanged BSA inputs
installs the cached wheel in seconds instead of recompiling for an hour.
`modal_app/` is attached last as a mount, so code edits never touch this layer.

A successful build proves the kernels compiled, not that they compute the right
thing — run `modal run -m modal_app.probe_gpu` after any stack change.
"""

import modal

from .app import build_cache_volume

# Fallback (previously verified): "nvidia/cuda:12.4.1-devel-ubuntu22.04" / "3.11"
CUDA_BASE_IMAGE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
PYTHON_VERSION = "3.12"

# Fallback (previously verified): torch 2.6.0+cu124 / torchvision 0.21.0+cu124
# with extra_index_url="https://download.pytorch.org/whl/cu124".
# torchaudio is deliberately absent: nothing in the codebase imports it (audio
# is handled by ffmpeg), and its release line has decoupled from torch's.
#
# Modal's uv resolver lands these as +cu130 builds from the PyTorch index. The
# two pins MUST stay a matching pair (torchvision 0.27.x <-> torch 2.12.x): a
# mismatch surfaces at runtime as "operator torchvision::nms does not exist"
# when transformers imports torchvision. They are also repeated in the layer-3
# install below so its joint resolution cannot drift torch to another version.
TORCH_PACKAGES = (
    "torch==2.12.1",
    "torchvision==0.27.1",
)

# sm_80 = A100, sm_86 = A10G, sm_120 = Blackwell RTX PRO 6000 (the deployed
# tier). Keeping 80/86 in the fatbin preserves portability back to the cheaper
# Ampere GPUs without another kernel build. Fallback stack supports "80;86" only.
BLOCK_SPARSE_ATTN_CUDA_ARCHS = "80;86;120"

BUILD_CACHE_MOUNT = "/build-cache"
BSA_SRC = "/workspace/FlashVSR-Pro/Block-Sparse-Attention"


def _build_or_install_bsa() -> None:
    """Image build step: install the BSA wheel from cache, or compile and cache it.

    Runs via Image.run_function with a GPU and 16 CPUs. The cache key covers the
    kernel sources, the torch version, and the arch list, so any input that
    changes the binary forces a fresh compile — and nothing else does.
    """
    import hashlib
    import os
    import shutil
    import subprocess
    import sys

    import torch

    hasher = hashlib.sha256()
    for root, dirs, files in sorted(os.walk(BSA_SRC)):
        dirs[:] = sorted(d for d in dirs if d not in {".git", "build", "dist"})
        for name in sorted(files):
            if os.path.splitext(name)[1] in {".py", ".cu", ".cpp", ".h", ".hpp", ".cuh"}:
                path = os.path.join(root, name)
                hasher.update(path.removeprefix(BSA_SRC).encode())
                with open(path, "rb") as f:
                    hasher.update(f.read())
    hasher.update(torch.__version__.encode())
    hasher.update(BLOCK_SPARSE_ATTN_CUDA_ARCHS.encode())
    key = hasher.hexdigest()[:20]

    cache_dir = os.path.join(BUILD_CACHE_MOUNT, "wheels", key)
    cached = sorted(
        os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".whl")
    ) if os.path.isdir(cache_dir) else []

    if cached:
        print(f"[bsa-build] cache HIT ({key}): installing {os.path.basename(cached[0])}")
        subprocess.run([sys.executable, "-m", "pip", "install", cached[0]], check=True)
        return

    # Each cutlass translation unit peaks at several GB of RAM in nvcc; an
    # uncapped MAX_JOBS OOM-kills the builder long before CPUs are the limit.
    jobs = max(4, min(os.cpu_count() or 4, 10))
    print(
        f"[bsa-build] cache MISS ({key}): compiling for archs "
        f"{BLOCK_SPARSE_ATTN_CUDA_ARCHS} with MAX_JOBS={jobs} — expect ~30-40 min"
    )
    env = os.environ | {
        "CXX": "g++",
        "CC": "gcc",
        "MAX_JOBS": str(jobs),
        "BLOCK_SPARSE_ATTN_CUDA_ARCHS": BLOCK_SPARSE_ATTN_CUDA_ARCHS,
    }
    subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel"], cwd=BSA_SRC, env=env, check=True
    )
    wheels = sorted(
        os.path.join(BSA_SRC, "dist", f)
        for f in os.listdir(os.path.join(BSA_SRC, "dist"))
        if f.endswith(".whl")
    )
    if not wheels:
        raise RuntimeError("BSA bdist_wheel produced no wheel in dist/")

    os.makedirs(cache_dir, exist_ok=True)
    shutil.copy2(wheels[-1], cache_dir)
    build_cache_volume.commit()
    print(f"[bsa-build] cached {os.path.basename(wheels[-1])} under wheels/{key}/")
    subprocess.run([sys.executable, "-m", "pip", "install", wheels[-1]], check=True)


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
