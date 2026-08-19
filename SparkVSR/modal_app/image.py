"""Modal Images for SparkVSR GPU worker and web tier."""

import modal

FFMPEG_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
    "/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz"
)

# A CUDA *devel* base (nvcc present) rather than debian_slim, because SageAttention
# compiles its kernels from source. This pin is the one flashvsr-pro already runs on the
# same RTX PRO 6000, so the torch 2.12.1 / cu130 pairing is proved rather than assumed.
CUDA_BASE_IMAGE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
PYTHON_VERSION = "3.12"

# sm_120 only. SageAttention's setup.py reads TORCH_CUDA_ARCH_LIST and maps 12.0 -> sm_120a,
# which is what lets the kernels build on a CPU-only builder with no GPU to interrogate.
# It also hard-requires nvcc >= 12.8 for this capability, which the CUDA 13 base satisfies.
SAGE_CUDA_ARCH = "12.0"

# SageAttention 2.x is not on PyPI (PyPI cannot carry one version per torch/CUDA variant, so
# upstream publishes none) — 1.0.6 is the newest thing there and is the older, slower Triton
# implementation. Install from the v2.2.0 tag instead, pinned by commit so a moving tag
# cannot silently change what the benchmark measured.
SAGE_ATTENTION_REF = "eb615cf6cf4d221338033340ee2de1c37fbdba4a"  # tag v2.2.0

# Slim torch-free CPU image for web tier and weight provisioning
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "fastapi[standard]>=0.115.0",
        "python-multipart>=0.0.9",
        "requests>=2.31.0",
        "huggingface-hub>=0.26.0",
    )
    .add_local_python_source("modal_app")
)

# GPU image for SparkVSR inference
# Blackwell sm_120 compatible stack, matching proved pins in ngx-vsr and FlashVSR-Pro.
gpu_image = (
    modal.Image.from_registry(CUDA_BASE_IMAGE, add_python=PYTHON_VERSION)
    .apt_install(
        "curl",
        "xz-utils",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libglib2.0-0",
        "git",
        "build-essential",
        "ninja-build",
    )
    .run_commands(
        # Pinned static n8.1 ffmpeg with NVENC/NVDEC and libx264 support
        f"curl -fsSL -o /tmp/ff.tar.xz {FFMPEG_URL}",
        "mkdir -p /opt/ffmpeg",
        "tar -xJf /tmp/ff.tar.xz -C /opt/ffmpeg --strip-components=1",
        "cp /opt/ffmpeg/bin/ffmpeg /opt/ffmpeg/bin/ffprobe /usr/local/bin/",
        "rm -rf /tmp/ff.tar.xz /opt/ffmpeg",
        "ffmpeg -version",
    )
    .uv_pip_install(
        # Pinned torch / torchvision compatible with Blackwell sm_120
        "torch==2.12.1",
        "torchvision==0.27.1",
        # Core model dependencies
        "numpy==1.26.4",  # Keep numpy < 2 for diffusers/decord compatibility
        "diffusers>=0.36.0",
        "transformers>=4.46.2",
        "accelerate>=1.1.1",
        "safetensors>=0.4.5",
        "decord>=0.6.0",
        "av>=12.0.0",
        "imageio>=2.36.0",
        "imageio-ffmpeg>=0.5.1",
        "opencv-python-headless>=4.10.0",
        "scenedetect[opencv]>=0.6.4",
        "einops>=0.8.0",
        "peft>=0.13.0",
        # FP8 dynamic quantisation for the transformer (Blackwell native). Only loaded
        # when SPARKVSR_FP8 is set; the import is guarded either way.
        "torchao>=0.9.0",
        "pillow>=10.4.0",
        "tqdm>=4.66.0",
        "huggingface-hub>=0.26.0",
        "fal-client>=0.5.0",
        # Backs the diffusers "*_hub" attention backends, which pull prebuilt kernels from
        # the Hub. Cheap insurance: if the source build below produces something that will
        # not load, SPARKVSR_ATTENTION_BACKEND=sage_hub is still available.
        "kernels>=0.6.0",
        "triton>=3.0.0",
    )
    # SageAttention: INT8 QK + FP8 PV attention. Upstream's own benchmark is on
    # CogVideoX1.5-5B, this project's exact backbone, where it beats FlashAttention2 end to
    # end by ~2x on an H20. There is no PyPI wheel matching this torch/CUDA pair, so it is
    # built here — roughly 9 translation units for a single arch, a few minutes once.
    # --no-build-isolation is required: setup.py imports the installed torch to pick flags.
    .env({
        # Read by SageAttention's setup.py to pick -gencode flags. Without it the build
        # interrogates the local GPU, and the builder has none.
        "TORCH_CUDA_ARCH_LIST": SAGE_CUDA_ARCH,
        # nvcc peaks at several GB per translation unit; an uncapped job count OOM-kills
        # the builder before CPUs become the limit.
        "EXT_PARALLEL": "4",
        "MAX_JOBS": "8",
        "NVCC_APPEND_FLAGS": "--threads 8",
        # The CUDA 13 base leaves torch's cpp_extension resolving a clang++ that reports
        # itself as version 0.0.0, which then fails CUDA's minimum-compiler check. Name the
        # GNU toolchain explicitly instead.
        "CXX": "g++",
        "CC": "gcc",
    })
    # --no-build-isolation is required: setup.py imports the *installed* torch to decide
    # ABI flags and arch targets, so it cannot run in a fresh isolated build env. That also
    # means pip installs none of the build backend for it, so those go in by hand first —
    # without wheel, setup.py dies on "invalid command 'bdist_wheel'".
    .pip_install("wheel", "setuptools", "packaging", "ninja")
    .pip_install(
        f"sageattention @ git+https://github.com/thu-ml/SageAttention.git@{SAGE_ATTENTION_REF}",
        extra_options="--no-build-isolation",
    )
    .run_commands(
        "python -c \"import sageattention; print('sageattention ok', sageattention.__file__)\"",
    )
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True",
    })
    # Attach modal_app last so code edits do not invalidate dependency cache
    .add_local_python_source("modal_app")
)
