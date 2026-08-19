## Why

Soft-looking 1080p Finnish broadcast series from 2000–2010 need detail restoration, not just resampling: the masters are low-contrast, lightly filtered, and already at 1080p, so a plain scaler adds pixels without adding information. SparkVSR (ECCV 2026, CogVideoX1.5-5B-I2V backbone) restores detail by propagating sparse high-quality keyframes across a shot, and it runs a **single** transformer forward per chunk, which makes a 5B-parameter model economically viable on serverless GPUs. There is no Modal deployment of it yet; this change builds one in the empty `SparkVSR/` workspace, reusing the two production Modal VSR services already in this monorepo.

## What Changes

- New `modal_app/` package deploying SparkVSR as a Modal service, structured after `../FlashVSR-Pro/modal_app/` (app/config/image/service/web_api split, CPU web tier in front of a GPU worker, spawn → poll job queue, Volume-backed I/O).
- Weight provisioning: one-shot Modal Function downloads `JiongzeYu/SparkVSR` (Diffusers pipeline layout, ~22 GB) plus PiSA-SR reference weights into a `sparkvsr-models` Volume, and **precomputes the empty-prompt T5 embedding** so the runtime pipeline loads with `text_encoder=None, tokenizer=None` (drops the ~9 GB T5-XXL from GPU memory and from cold start entirely).
- Vendored inference core adapted from upstream `sparkvsr_inference_script.py` (Apache-2.0, attribution retained): temporal chunking with overlap, optional spatial tiling with overlap blending, `8n+1` frame padding, VAE slicing/tiling, `CogVideoXDPMScheduler(timestep_spacing="trailing")`, bf16, single-step denoise at `sr_noise_step=399`.
- **Cut-aware planning** (a capability upstream lacks, inspired by `goodguy1963/ComfyUI-SparkVSR-CutAware`): PySceneDetect shot boundaries drive chunk windows so no temporal chunk straddles a scene cut, and every window is guaranteed a reference keyframe. TV drama cuts every few seconds; a chunk spanning a cut bleeds one scene's detail into the next.
- **Reference keyframe generation** with three modes: `pisasr` (default, open-source image SR run in-process on the same GPU), `api` (fal.ai `nano-banana-pro/edit`, needs a secret), `no_ref` (fallback baseline).
- Encode path lifted from `../ngx-vsr/modal_app.py`: probed `hevc_nvenc` Main 10 → `h264_nvenc` → `libx264`, explicit bt709 tagging, rational fps, and **audio stream-copied** from the source rather than re-encoded.
- Local CLI client (`sparkvsr_client.py`) and a `run_sample.py` smoke test against the existing `sample/jopet_*.mkv` clips.

### Assumptions recorded (clarification was requested but not answered)

1. Default reference mode is `pisasr`; `api` and `no_ref` are selectable flags.
2. Default output is 4K (3840×2160) from a 1080p source, with a `--target-height` flag and a same-resolution "refresh" mode for detail-only restoration.
3. Cut-aware planning is on by default, disableable with `--no-cut-aware`.

These are reversible flag defaults, not architecture; they can be changed during implementation without reworking the plan.

## Capabilities

### New Capabilities
- `video-upscaling`: submitting a video for SparkVSR super-resolution and retrieving the result — request validation, resolution/scale semantics, job lifecycle, output container and audio guarantees.
- `cut-aware-planning`: deriving scene-cut-respecting chunk windows and reference-frame indices from an input video.
- `reference-keyframes`: producing the high-quality keyframes SparkVSR propagates, across the three reference modes.
- `model-provisioning`: getting SparkVSR and reference-model weights, plus the precomputed empty-prompt embedding, onto a Modal Volume.

### Modified Capabilities
None — `openspec/specs/` is empty; this is the project's first change.

## Impact

- **New code**: `modal_app/` package, `sparkvsr_client.py`, `run_sample.py`, `README.md` in `SparkVSR/`. No existing files modified; `FlashVSR-Pro/` and `ngx-vsr/` are read-only references, copied from rather than imported.
- **Modal resources**: app `sparkvsr`; Volumes `sparkvsr-models` (weights) and `sparkvsr-io` (job files); GPU `RTX-PRO-6000` (96 GB, confirmed available in the Modal GPU list); Secrets `huggingface` (existing) and `fal` (new, only for `--ref-mode api`).
- **Upstream dependencies**: `taco-group/SparkVSR` inference code and `ComfyUI-Spark/sparkvsr_wrapper/pisasr_src` (both Apache-2.0, vendored with attribution); `diffusers>=0.36`, `transformers`, `decord`, `scenedetect`, `numpy<2` (upstream pins 1.26).
- **Cost**: a 5B model at one step per chunk plus VAE encode/decode; per-episode-minute cost is unknown until measured and is an explicit deliverable of the verification step.
