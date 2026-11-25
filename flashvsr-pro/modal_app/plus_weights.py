"""One-time FlashVSR+ weight download to the `flashvsr-plus-models` Volume.

    modal run -m modal_app.plus_weights

Fetches the turbo and base presets and the TinyDecoder from
`ruchangyao/FlashVSR_plus`, and the FlashVSR v1.1 weights for upstream's `v1`
preset, laid out the way upstream's `ckpts/` directory expects:

    ckpts/{turbo,base,v1}/dit.safetensors
    ckpts/{turbo,base,v1}/lq_proj_in.safetensors
    ckpts/TCDecoder.ckpt

About 20 GB. Existing files are kept, so a failed run can be repeated.
"""

import modal

from .config import (
    PLUS_CKPT_DIR,
    PLUS_HF_FILES,
    PLUS_HF_REPO,
    PLUS_MODEL_MOUNT_PATH,
    PLUS_V1_DIT_FILE,
    PLUS_V1_HF_REPO,
    PLUS_V1_LQ_PROJ_FILE,
)
from .plus_app import plus_app, plus_model_volume

weights_image = (
    modal.Image.debian_slim(python_version="3.12")
    # CPU torch, only to read one checkpoint. From its own index: as an extra
    # index the resolver would still take PyPI's CUDA build.
    .uv_pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cpu")
    .uv_pip_install("huggingface-hub==0.34.4", "safetensors==0.5.3")
    .add_local_python_source("modal_app")
)


@plus_app.function(
    image=weights_image,
    volumes={PLUS_MODEL_MOUNT_PATH: plus_model_volume},
    timeout=3600,
)
def download_plus_models(hf_token: str | None = None) -> list[str]:
    """Download and lay out the weights. Returns the files now on the Volume.

    Both repositories are public; `hf_token` only matters if that changes.
    """
    import os
    import shutil
    import tempfile

    from huggingface_hub import hf_hub_download

    def fetch(repo: str, filename: str, target: str) -> None:
        if os.path.exists(target):
            print(f"Already exists: {target}")
            return
        os.makedirs(os.path.dirname(target), exist_ok=True)
        print(f"Downloading {repo}:{filename} ...")
        with tempfile.TemporaryDirectory() as staging:
            downloaded = hf_hub_download(
                repo_id=repo, filename=filename, local_dir=staging, token=hf_token
            )
            # Copy under a temporary name first: a file that exists under its
            # final name is complete.
            shutil.copyfile(downloaded, f"{target}.part")
        os.replace(f"{target}.part", target)
        # Commit per file: a run that fails later keeps what it already fetched.
        plus_model_volume.commit()
        print(f"  -> {target} ({os.path.getsize(target) / 1e9:.2f} GB)")

    for filename, relative in PLUS_HF_FILES:
        fetch(PLUS_HF_REPO, filename, os.path.join(PLUS_CKPT_DIR, relative))

    v1_dir = os.path.join(PLUS_CKPT_DIR, "v1")
    fetch(PLUS_V1_HF_REPO, PLUS_V1_DIT_FILE, os.path.join(v1_dir, "dit.safetensors"))

    # Upstream loads `lq_proj_in.safetensors` by suffix with safetensors, and the
    # v1.1 projector is published as a torch checkpoint. Convert it once.
    v1_projector = os.path.join(v1_dir, "lq_proj_in.safetensors")
    if os.path.exists(v1_projector):
        print(f"Already exists: {v1_projector}")
    else:
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as staging:
            downloaded = hf_hub_download(
                repo_id=PLUS_V1_HF_REPO, filename=PLUS_V1_LQ_PROJ_FILE,
                local_dir=staging, token=hf_token,
            )
            state = torch.load(downloaded, map_location="cpu", weights_only=True)
        for container in ("state_dict", "module", "model", "model_state_dict"):
            if isinstance(state, dict) and isinstance(state.get(container), dict):
                state = state[container]
        tensors = {key: value.contiguous() for key, value in state.items() if torch.is_tensor(value)}
        if not tensors:
            raise RuntimeError(f"{PLUS_V1_LQ_PROJ_FILE} holds no tensors; cannot convert it.")
        save_file(tensors, f"{v1_projector}.part")
        os.replace(f"{v1_projector}.part", v1_projector)
        print(f"Converted {PLUS_V1_LQ_PROJ_FILE} -> {v1_projector} ({len(tensors)} tensors)")

    plus_model_volume.commit()
    present = sorted(
        os.path.relpath(os.path.join(root, name), PLUS_CKPT_DIR)
        for root, _, names in os.walk(PLUS_CKPT_DIR) for name in names
    )
    print("Weights on the Volume:", ", ".join(present))
    return present


# A unique name: two entrypoints called `main` on one App cannot be imported together.
@plus_app.local_entrypoint()
def weights_main():
    download_plus_models.remote()
