"""One-time model weight download to Modal Volume.

Usage:
    modal run -m modal_app.download_weights
"""

import modal

from .app import app, model_volume
from .config import (
    HF_FILES,
    HF_REPO_ID,
    MODEL_MOUNT_PATH,
    PROMPT_TENSOR_DIR,
    PROMPT_TENSOR_HF_FILE,
    PROMPT_TENSOR_HF_REPO,
    PROMPT_TENSOR_PATH,
)

download_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface-hub==0.34.4",
    "safetensors",
    "torch==2.6.0+cpu",
    extra_index_url="https://download.pytorch.org/whl/cpu",
)


@app.function(
    image=download_image,
    volumes={MODEL_MOUNT_PATH: model_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def download_models():
    """Download all FlashVSR-Pro model weights from HuggingFace to the Volume."""
    import os

    from huggingface_hub import hf_hub_download

    hf_token = os.environ.get("HF_TOKEN")

    # Download main model files from JunhaoZhuang/FlashVSR-v1.1
    for filename, target_dir in HF_FILES:
        target_path = os.path.join(target_dir, filename)
        if os.path.exists(target_path):
            print(f"Already exists: {target_path}")
            continue

        os.makedirs(target_dir, exist_ok=True)
        print(f"Downloading {filename}...")
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=target_dir,
            token=hf_token,
        )
        print(f"  -> {downloaded}")

    # Download and convert prompt tensor (safetensors -> .pth)
    if os.path.exists(PROMPT_TENSOR_PATH):
        print(f"Already exists: {PROMPT_TENSOR_PATH}")
    else:
        os.makedirs(PROMPT_TENSOR_DIR, exist_ok=True)
        print(f"Downloading {PROMPT_TENSOR_HF_FILE} from {PROMPT_TENSOR_HF_REPO}...")
        downloaded = hf_hub_download(
            repo_id=PROMPT_TENSOR_HF_REPO,
            filename=PROMPT_TENSOR_HF_FILE,
            local_dir=PROMPT_TENSOR_DIR,
            token=hf_token,
        )
        print(f"  -> {downloaded}")

        # Convert safetensors to .pth format
        import torch
        from safetensors import safe_open

        safetensors_path = os.path.join(PROMPT_TENSOR_DIR, PROMPT_TENSOR_HF_FILE)
        with safe_open(safetensors_path, framework="pt") as f:
            tensor = f.get_tensor("posi_prompt")
        print(f"  Prompt tensor shape: {tensor.shape}, dtype: {tensor.dtype}")

        torch.save(tensor, PROMPT_TENSOR_PATH)
        print(f"  Converted to: {PROMPT_TENSOR_PATH}")

        # Clean up the safetensors file
        os.remove(safetensors_path)

    model_volume.commit()
    print("All weights downloaded and committed to volume.")


@app.local_entrypoint()
def main():
    download_models.remote()
