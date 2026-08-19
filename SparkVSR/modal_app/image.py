"""Modal Images for SparkVSR GPU worker and web tier."""

import modal

FFMPEG_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
    "/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz"
)

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
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "curl",
        "xz-utils",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libglib2.0-0",
        "git",
        "build-essential",
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
    )
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True",
    })
    # Attach modal_app last so code edits do not invalidate dependency cache
    .add_local_python_source("modal_app")
)
