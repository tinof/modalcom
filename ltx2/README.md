# LTX-2.3 Image-to-Video on Modal

Generate videos from images using [Lightricks LTX-2.3](https://github.com/Lightricks/LTX-2) (22B parameter DiT model) deployed on [Modal](https://modal.com) with H100 GPUs.

## Architecture

LTX-2.3 uses a **two-stage HQ pipeline** (`TI2VidTwoStagesHQPipeline`):
1. Low-resolution video generation (960×544) from text + image conditioning
2. 2x spatial upsampling for full HD (1920×1088) output

The HQ pipeline uses a second-order **Res2s sampler** (instead of Euler) for significantly improved quality, and applies per-stage distilled LoRA strengths for optimal results.

### Model Components (downloaded to Modal Volume on first run)

| Component | File | Purpose |
|---|---|---|
| Checkpoint | `ltx-2.3-22b-dev.safetensors` | Main 22B transformer (bf16) |
| Distilled LoRA | `ltx-2.3-22b-distilled-lora-384.safetensors` | Per-stage adaptation (stage 1: 0.25, stage 2: 0.5) |
| Spatial Upsampler | `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | 2x resolution enhancement |
| Text Encoder | `text_encoder/*` | Gemma 3 12B (quantized) |
| Tokenizer | `tokenizer/*` | Gemma tokenizer |

Source: [Lightricks/LTX-2.3 on HuggingFace](https://huggingface.co/Lightricks/LTX-2.3)

### Key Differences from LTX-2.0

| Aspect | LTX-2.0 | LTX-2.3 |
|---|---|---|
| Model size | 19B parameters | 22B parameters |
| Pipeline | `TI2VidTwoStagesPipeline` | `TI2VidTwoStagesHQPipeline` |
| Sampler | Euler | Res2s (second-order) |
| Resolution | 768×512 | 1920×1088 (full HD) |
| Default steps | 40 | 15 |
| Default fps | 25 | 24 |
| LoRA strength | Single (0.8) | Per-stage (0.25 / 0.5) |
| STG guidance | stg_scale=1.0, stg_blocks=[29] | Not needed (Res2s handles coherence) |

## Prerequisites

- Python 3.12
- Modal account and CLI: `pip install modal && modal setup`

## Usage

### CLI

```bash
modal run image_to_video.py \
  --prompt "A cat looking out the window at a snowy mountain" \
  --image-path /path/to/cat.jpg
```

Options:
- `--image-path` (required): local file path or URL
- `--prompt` (required): text description of desired video
- `--num-frames`: frame count (default: 121 = ~5s at 24fps)
- `--num-inference-steps`: denoising steps (default: 15)
- `--cfg-scale`: classifier-free guidance scale (default: 3.0)
- `--seed`: RNG seed for reproducibility
- `--no-twice`: skip the warm-run benchmark

### Deploy as API + Web UI

```bash
modal deploy image_to_video.py
```

This creates:
- **POST API** at `https://<your-app>--inference-web.modal.run` (add `/docs` for Swagger UI)
- **Web UI** at `https://<your-app>--ui.modal.run` (requires `frontend/` directory with Alpine.js templates)

### API Example (curl)

```bash
curl -X POST "https://<your-url>/inference-web" \
  -F "image_bytes=@photo.png" \
  -F "prompt=A bird takes flight from a rooftop" \
  -F "num_frames=121" \
  -F "seed=42"
  --output video.mp4
```

## Project Structure

```
ltx2/
├── image_to_video.py    # Main Modal app (inference, API, web UI)
├── frontend/            # Alpine.js web UI templates (optional)
│   └── index.html
└── README.md
```

## Modal Resources

| Resource | Name | Purpose |
|---|---|---|
| Volume | `ltx2.3-model-cache` | Cached model weights (~40GB) |
| Volume | `outputs` | Generated video files |
| GPU | H100 | Inference (80GB VRAM) |

## Inference Parameters

The `MultiModalGuiderParams` control video generation quality:

| Parameter | Default | Purpose |
|---|---|---|
| `cfg_scale` | 3.0 | Prompt adherence (higher = more literal) |
| `stg_scale` | 0.0 | Spatio-temporal guidance (not needed with Res2s) |
| `rescale_scale` | 0.45 | Prevents over-saturation |
| `modality_scale` | 3.0 | Audio-visual sync control |
| `skip_step` | 0 | Steps to skip in guidance |
| `stg_blocks` | [] | Transformer blocks for STG (empty for HQ pipeline) |

## Known Considerations

- **First cold start** downloads ~40GB of model weights to the volume. Subsequent starts use the cached weights.
- **torch.inference_mode workaround**: The spatial upsampler and VAE decoder use `conv3d` under `@torch.inference_mode()` decorators, which causes `RuntimeError: Inference tensors cannot be saved for backward`. We patch `torch.inference_mode = torch.no_grad` before importing `ltx_pipelines` to avoid this.
- `ltx-core` and `ltx-pipelines` are installed from git (not PyPI). Pin to a commit hash for reproducibility by changing the git URL to include `@<commit-sha>`.
- The bf16 checkpoint is used because the FP8 variant causes tensor size mismatches during LoRA fusion with the distilled LoRA weights.
- **H100 memory**: 22B params in bf16 ≈ 44GB VRAM + text encoder + upsampler + activations ≈ 60-70GB peak. Fits within H100's 80GB.

## Upstream References

- [LTX-2 GitHub](https://github.com/Lightricks/LTX-2)
- [LTX-2 Pipeline README](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-pipelines)
- [LTX-2.3 HuggingFace Model](https://huggingface.co/Lightricks/LTX-2.3)
- [Modal Documentation](https://modal.com/docs)
