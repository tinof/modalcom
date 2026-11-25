"""Stack pins and the Block-Sparse-Attention build step shared by both GPU images.

Kept apart from `image.py` and `plus_image.py` so that importing one image
never evaluates the other's local-directory layers.

The Block-Sparse-Attention CUDA kernels are the expensive part of either image
(~25 heavy cutlass translation units). `_build_or_install_bsa` builds them once
into a wheel cached on the `flashvsr-build-cache` Volume, keyed on source hash +
torch version + arch list. Rebuilding an image with unchanged inputs installs
the cached wheel in seconds.

A successful build proves the kernels compiled, not that they compute the right
thing: run `modal run -m modal_app.probe_gpu` after any stack change.
"""

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
# when transformers imports torchvision. Each image repeats them in its
# dependency install so the joint resolution cannot drift torch to another version.
TORCH_PACKAGES = (
    "torch==2.12.1",
    "torchvision==0.27.1",
)

# sm_120 = Blackwell RTX PRO 6000, the only GPU either app runs on. Each extra
# arch multiplies the nvcc time of every translation unit, so the Ampere archs
# ("80;86;120": sm_80 = A100, sm_86 = A10G) are added back only together with a
# move to those GPUs. The fallback stack supports "80;86" only. Until
# 2026-11-25 this was "80;86;120"; the single-arch build has not run yet.
BLOCK_SPARSE_ATTN_CUDA_ARCHS = "120"

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
