"""Weight provisioning and verification for SparkVSR.

Downloads:
1. Diffusers CogVideoX model weights: JiongzeYu/SparkVSR (~22 GB)
2. PiSA-SR SD2.1 base weights: stabilityai/stable-diffusion-2-1-base (~5 GB)
3. Precomputes and saves the empty-prompt T5 embedding, so inference never
   loads the text encoder

The PiSA-SR adapter checkpoint (pisa_sr.pkl, ~32 MB) is NOT downloaded here.
Upstream distributes it via Google Drive only and there is no public
programmatic download, so it is operator-supplied:

    modal volume put sparkvsr-models /path/to/pisa_sr.pkl pisasr/pisa_sr.pkl

If you have a direct URL, set the PISASR_WEIGHTS_URL environment variable and
this function will fetch it for you.

Usage:
    modal run -m modal_app.download_weights
"""

import os
from pathlib import Path

import modal

from .app import app, models_volume
from .config import (
    EMPTY_PROMPT_EMBED_PATH,
    MODEL_MOUNT_PATH,
    PISASR_DIR,
    PISASR_SD21_DIR,
    PISASR_SD21_HF_REPO,
    PISASR_WEIGHTS_PATH,
    PISASR_WEIGHTS_URL_ENV,
    SPARKVSR_HF_REPO,
    SPARKVSR_MODEL_DIR,
)
from .weights import missing_pisasr_weights, verify_weights

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.12.1",
        "transformers>=4.46.2",
        "diffusers>=0.36.0",
        "safetensors>=0.4.5",
        "accelerate>=1.1.1",
        "huggingface-hub>=0.26.0",
        "sentencepiece>=0.2.0",
        "requests>=2.31.0",
    )
    .add_local_python_source("modal_app")
)


@app.function(
    image=download_image,
    volumes={MODEL_MOUNT_PATH: models_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def download_weights(force: bool = False):
    """Download all model weights and precompute the empty-prompt embedding."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, T5EncoderModel

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print("=== Step 1/4: Downloading SparkVSR weights ===")
    os.makedirs(SPARKVSR_MODEL_DIR, exist_ok=True)
    spark_downloaded = snapshot_download(
        repo_id=SPARKVSR_HF_REPO,
        local_dir=SPARKVSR_MODEL_DIR,
        token=hf_token,
        max_workers=8,
    )
    print(f"SparkVSR snapshot ready at {spark_downloaded}")

    print("=== Step 2/4: Downloading PiSA-SR SD 2.1 base weights ===")
    os.makedirs(PISASR_DIR, exist_ok=True)
    os.makedirs(PISASR_SD21_DIR, exist_ok=True)
    sd21_downloaded = snapshot_download(
        repo_id=PISASR_SD21_HF_REPO,
        local_dir=PISASR_SD21_DIR,
        token=hf_token,
        max_workers=8,
        # PiSASR_eval loads the fp32 subfolder safetensors, so skip the single-file
        # checkpoints, the .bin duplicates and the fp16 variants (~11 GB of waste).
        allow_patterns=[
            "model_index.json",
            "tokenizer/*",
            "scheduler/*",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
        ],
    )
    print(f"SD 2.1 base ready at {sd21_downloaded}")

    # pisa_sr.pkl is operator-supplied unless a direct URL was provided.
    pisa_dst = Path(PISASR_WEIGHTS_PATH)
    if pisa_dst.is_file() and not force:
        print(f"PiSA-SR adapter already present at {pisa_dst}")
    else:
        pisa_url = os.environ.get(PISASR_WEIGHTS_URL_ENV)
        if pisa_url:
            import requests

            print(f"Fetching PiSA-SR adapter from {PISASR_WEIGHTS_URL_ENV}...")
            resp = requests.get(pisa_url, timeout=300, stream=True)
            resp.raise_for_status()
            with pisa_dst.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            print(f"Saved PiSA-SR adapter to {pisa_dst} ({pisa_dst.stat().st_size} bytes)")
        else:
            print(
                "NOTE: pisa_sr.pkl not present and no "
                f"{PISASR_WEIGHTS_URL_ENV} set. The 'pisasr' reference mode will be "
                "unavailable until you upload it once:\n"
                "  modal volume put sparkvsr-models /path/to/pisa_sr.pkl pisasr/pisa_sr.pkl\n"
                "Download it from the Google Drive link in https://github.com/csslc/PiSA-SR\n"
                "The 'api' and 'no_ref' reference modes work without it."
            )

    print("=== Step 3/4: Precomputing empty-prompt T5 embedding ===")
    if Path(EMPTY_PROMPT_EMBED_PATH).is_file() and not force:
        print(f"Empty-prompt embedding already exists at {EMPTY_PROMPT_EMBED_PATH}")
    else:
        tokenizer_path = os.path.join(SPARKVSR_MODEL_DIR, "tokenizer")
        text_encoder_path = os.path.join(SPARKVSR_MODEL_DIR, "text_encoder")

        print(f"Loading tokenizer from {tokenizer_path} and text encoder from {text_encoder_path}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_path,
            torch_dtype=torch.bfloat16,
        )
        text_encoder.eval()

        text_inputs = tokenizer(
            [""],
            padding="max_length",
            max_length=226,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            prompt_embeds = text_encoder(text_inputs.input_ids)[0]

        print(f"Computed prompt_embeds with shape {prompt_embeds.shape} dtype {prompt_embeds.dtype}")
        torch.save(prompt_embeds.cpu().to(torch.bfloat16), EMPTY_PROMPT_EMBED_PATH)
        print(f"Saved empty prompt embedding to {EMPTY_PROMPT_EMBED_PATH}")

        del text_encoder, tokenizer

    print("=== Step 4/4: Verifying weights ===")
    models_volume.commit()

    missing = verify_weights(require_pisasr=False)
    if missing:
        raise RuntimeError(f"Weight verification failed! Missing components: {missing}")

    pisa_missing = missing_pisasr_weights()
    if pisa_missing:
        print(f"Provisioned. PiSA-SR reference mode NOT yet available: {pisa_missing}")
    else:
        print("Provisioned. All weights including PiSA-SR verified and committed.")


@app.local_entrypoint()
def main(force: bool = False):
    download_weights.remote(force=force)
