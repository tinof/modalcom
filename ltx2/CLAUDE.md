# LTX-2.3 Image-to-Video on Modal

## Project Overview

Single-file Modal app (`image_to_video.py`) that deploys Lightricks LTX-2.3 (22B DiT) for image-to-video generation on H100 GPUs. Uses the HQ pipeline (`TI2VidTwoStagesHQPipeline`) with Res2s sampler for full HD output. Includes a FastAPI POST endpoint and an Alpine.js web UI.

## Project Structure

```
ltx2/
├── image_to_video.py    # Modal app: inference class, API endpoint, web UI server
├── frontend/
│   └── index.html       # Alpine.js + Jinja2 template (served by Modal ASGI function)
└── README.md
```

Everything is in one file. The `Inference` class handles model loading (`@modal.enter`) and inference (`@modal.method`). The `ui()` function serves the frontend as a separate Modal ASGI app.

## Run & Deploy

```bash
# CLI inference
modal run image_to_video.py --prompt "..." --image-path /path/to/image.png

# Deploy API + web UI
modal deploy image_to_video.py
```

Modal CLI is at `~/.local/bin/modal`.

## Critical Workarounds (DO NOT REMOVE)

### torch.inference_mode patch
The spatial upsampler and VAE decoder use `conv3d` under `@torch.inference_mode()`, causing `RuntimeError: Inference tensors cannot be saved for backward`. The fix patches `torch.inference_mode = torch.no_grad` BEFORE importing ltx packages. This happens in the `image.imports()` block.

### bf16 checkpoint (not FP8)
FP8 checkpoints cause tensor size mismatch when fusing the distilled LoRA. Always use the bf16 checkpoint (`ltx-2.3-22b-dev.safetensors`).

### Package installation order
`ltx-core` and `ltx-pipelines` are installed with `--no-deps` to avoid pulling incompatible dependency versions. `transformers` is pinned to `4.57.3` (v5.x breaks `Gemma3TextConfig.rope_local_base_freq`).

## Pipeline Details

- Uses `TI2VidTwoStagesHQPipeline` from `ltx_pipelines.ti2vid_two_stages_hq` (NOT HuggingFace diffusers)
- HQ pipeline: Res2s second-order sampler (not Euler), per-stage LoRA strengths (0.25/0.5)
- Two-stage: low-res generation (960×544) + 2x spatial upsampling → full HD (1920×1088)
- Returns `(Iterator[torch.Tensor], Audio)` — video frames + audio
- Frames are `(T, H, W, C)` uint8 tensors — no permutation needed for imageio
- Image input format: file path tuples `(path, frame_idx, strength)`
- Default: 15 steps, 24fps, 121 frames (~5s), cfg_scale=3.0, no STG

## Modal Resources

| Resource | Name | Purpose |
|----------|------|---------|
| Volume | `ltx2.3-model-cache` | Model weights (~40GB), mounted at `/models` |
| Volume | `outputs` | Generated videos, mounted at `/outputs` |
| GPU | H100 | Required (80GB VRAM, peak ~60-70GB) |

## Code Style

- Python 3.12, single-file architecture
- Follow existing patterns: type hints, `Optional` params with `None` defaults
- Ruff for formatting and linting (`ruff format . && ruff check --fix .`)
- Keep literate-programming comment style (Modal example format)
