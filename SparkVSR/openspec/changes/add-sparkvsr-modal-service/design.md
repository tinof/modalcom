## Context

See `proposal.md` — Why. Design-relevant facts established by reading upstream source and the Modal docs:

- **The model.** `JiongzeYu/SparkVSR` is a complete Diffusers `CogVideoXImageToVideoPipeline` layout (transformer + 3D VAE + T5-XXL text encoder + scheduler), ~22 GB of F32 weights on the Hub, run in bf16. Inference is **one transformer forward per chunk**, not an iterative sampling loop: the upstream script runs the transformer once at `sr_noise_step=399` and recovers the clean latent with `scheduler.get_velocity()`. Cost per chunk is therefore one 5B forward plus a VAE encode and decode.
- **Conditioning mechanism.** A zero tensor the shape of the LQ latent is filled at `latent_index = frame_index // 4` with the VAE encoding of each HR reference frame (the frame is repeated 4× temporally before encoding), then channel-concatenated with the LQ latent to form the 32-channel transformer input. `ref_guidance_scale != 1.0` builds a second, reference-free copy of the batch and does classifier-free guidance, doubling compute. Reference indices must be more than 4 frames apart because two references landing in the same latent slot overwrite each other.
- **Upstream chunking.** `make_temporal_chunks` / `make_spatial_tiles` / `get_valid_tile_region` in `sparkvsr_inference_script.py` produce overlapping chunks and tiles and keep only each region's non-overlapping interior, with frames padded to `8n+1` and spatial dims to a multiple of 4. This machinery is sound and worth vendoring; what it lacks is any notion of a scene cut.
- **Two in-repo baselines.** `../FlashVSR-Pro/modal_app/` supplies the deployment skeleton (module split, weight download to a Volume, CPU web tier with `validate_request()` ahead of GPU spend, spawn/poll job queue, long-video orchestrator). `../ngx-vsr/modal_app.py` supplies the video I/O craft (probed encoder selection, 10-bit HEVC with explicit bt709 tagging, `Fraction(fps).limit_denominator(1001)`, audio stream-copy, threaded decode→infer→encode).
- **Platform.** Modal client 1.5.4. `RTX-PRO-6000` (96 GB) is a supported GPU string and is what both baselines run on. GPU memory snapshots exist (`enable_memory_snapshot=True` plus `experimental_options={"enable_gpu_snapshot": True}`) but the docs state plainly that they do not speed up loading weights from storage, which is exactly where this app's cold start goes.

## Goals / Non-Goals

**Goals:**
- A deployed Modal service that restores soft 1080p broadcast masters, with cut-aware chunking as a first-class feature rather than a bolt-on.
- Reuse of the two baselines by copying their proven code paths, so this repo does not depend on its siblings at import time.
- Measured cost and timing per episode-minute as a deliverable, not an afterthought.

**Non-Goals:**
- Training or fine-tuning on Yle material.
- Multi-GPU or multi-node inference.
- Batch fan-out over an episode library — the orchestrator is built so this can be added later, but it is not in this change.
- A ComfyUI node. The CutAware repo is a design reference, not a dependency.

## Decisions

### Vendor the upstream inference core rather than pip-install the repo

`taco-group/SparkVSR` has no package metadata; its inference path is a single script that assumes a repo-rooted working directory, and the reference-generation helper it imports pulls in a fal API key at module scope. Copying the ~8 functions that matter into `modal_app/pipeline/core.py` under an Apache-2.0 attribution header gives a stable import surface and lets the cut-aware planner replace `make_temporal_chunks` cleanly.

*Alternative rejected:* `git clone` in the image build and `sys.path` manipulation — reproduces the working-directory assumptions inside a container and makes the tiling code hard to substitute.

### Drop the text encoder by precomputing the empty-prompt embedding

Every job uses an empty prompt. The upstream script already supports a precomputed `empty_prompt_embedding`, so provisioning encodes it once and inference constructs the pipeline with `text_encoder=None, tokenizer=None`. This removes T5-XXL (~9 GB in bf16) from both GPU memory and the cold-start read.

*Alternative rejected:* loading T5 and encoding `""` at container start — pure waste on every cold start for a constant.

### Cut-aware planning as a separate module producing a plan, with inference as a consumer

`pipeline/cutplan.py` turns a video into a list of windows, each carrying `(shot_id, start, end, pad_before, pad_after, ref_indices)`; `service.py` executes the plan without knowing how it was derived. Disabling cut-awareness swaps the planner for fixed-length windows, exercising the same executor. The plan is serialized into the job output, which makes the cut detection independently checkable against the video — a spec requirement.

Parameters follow the CutAware reference (PySceneDetect adaptive detector, threshold 3.0, minimum scene 0.6 s; 49-frame windows; references picked no more than 0.5 s into a window), because those are the only empirically tuned values available for this model.

*Alternative rejected:* detecting cuts inside the chunk loop — makes overlap blending decisions local and untestable without a GPU.

### Reference generation runs in-process, then unloads

PiSA-SR is Stable Diffusion 2.1-based; the upstream ComfyUI wrapper already vendors a version (`ComfyUI-Spark/sparkvsr_wrapper/pisasr_src`) that runs in the same process, unlike the main repo's approach of shelling out to a second conda environment. All references for a job are generated up front, the model is unloaded, and only then does the SparkVSR pipeline load. Sequential rather than concurrent because the two models together are a poor fit even for 96 GB once 4K activations are in play, and reference generation is a small fraction of total time.

*Alternative rejected:* a separate Modal Function for reference generation — a second cold start and a second 5 GB weight load to save a few seconds of serial time.

### Take ngx-vsr's encode path wholesale

Restoring detail and then encoding it at 8-bit `yuv420p` with default settings throws away part of what was just computed. ngx-vsr's encoder command is already tuned for exactly this: 10-bit HEVC to avoid banding on gradients (common in these masters), explicit bt709 tagging to avoid the bt601 hue shift, AQ and lookahead settings, and audio stream-copy. It is lifted with its probe-don't-assume structure intact, since a hardcoded `hevc_nvenc` fails opaquely when the driver and the ffmpeg build disagree — the failure mode ngx-vsr's pinned ffmpeg build exists to prevent.

### No GPU memory snapshots in this change

The docs are explicit that GPU snapshots do not accelerate loading weights from storage, which dominates this cold start, and they interact poorly with the kind of code this app runs. Keep-warm containers are the lever if cold start proves painful; snapshots can be evaluated later against measured numbers.

### Spatial tiling off by default

At 96 GB, 1080p→4K should fit without spatial tiling, and tiling costs both time and seam risk. VAE slicing and tiling stay on (cheap, no seam risk at the VAE level). Spatial tiling is exposed as a flag so a 6K or 8K target has a path, and the OOM error message names that flag.

## Risks / Trade-offs

- **4K activations may not fit even on 96 GB.** → Measure at the smoke-test step before building anything on top; fall back to spatial tiling (already implemented, just off) or process at 2× and resample. This is the first thing to verify, not the last.
- **PiSA-SR's vendored copy may not run standalone.** → It is exercised by the ComfyUI wrapper in-process, so the risk is dependency drift rather than architecture. If it resists, the fal API mode is a working substitute and `no_ref` still ships.
- **PySceneDetect thresholds are tuned for other material.** → Yle masters from this era are soft and low-contrast, which suppresses detector confidence. The plan is written to the job output specifically so thresholds can be checked against `sample/jopet_60s.mkv` and adjusted from evidence.
- **Cost per episode-minute could be prohibitive.** → One-step inference is the reason this is plausible at all, but a 5B forward per 49 frames at 4K is not cheap. Instrumentation is a spec requirement; if the number is bad, the honest options are a lower target resolution or fewer references, and both are flags.
- **`numpy<2` pin from upstream against current torch wheels.** → Both baselines hit this and pinned it deliberately; follow their resolution rather than rediscovering it.
- **Vendored code drifts from upstream.** → Record the upstream commit in the vendored files' headers so a future diff is mechanical.

## Migration Plan

Nothing to migrate — `SparkVSR/` contains only sample clips and this OpenSpec scaffolding. Deployment order: provision weights, smoke-test on `sample/jopet_10s.mkv`, then deploy the web tier. Rollback is `modal app stop`; the weight Volume is independent of the app and survives.

## Open Questions

- Whether 1080p→4K is the right default or whether these particular masters respond better to same-resolution restoration. Answerable from the A/B in the verification step; it changes a flag default, not the design.
- Whether references benefit from being placed at more than one per shot in long dialogue scenes. A planner parameter, decided from output inspection.
