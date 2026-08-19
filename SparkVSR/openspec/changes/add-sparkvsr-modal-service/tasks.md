## 1. Project scaffolding

- [x] 1.1 Create the `modal_app/` package skeleton (`__init__.py`, `app.py`, `config.py`, `image.py`, `pipeline/__init__.py`) mirroring the module split in `../FlashVSR-Pro/modal_app/`
- [x] 1.2 Define `app.py`: `modal.App("sparkvsr")` plus Volumes `sparkvsr-models` and `sparkvsr-io`
- [x] 1.3 Define `config.py`: mount paths, GPU type `RTX-PRO-6000`, timeouts, chunk/overlap defaults (49 / 8), default target resolution 3840x2160, supported reference modes, and the parameter bounds used by validation
- [x] 1.4 Define `image.py`: GPU image with torch (Blackwell sm_120-compatible wheels, matching the pin `../ngx-vsr/modal_app.py` proved working), `diffusers>=0.36`, `transformers`, `accelerate`, `safetensors`, `decord`, `av`, `imageio[ffmpeg]`, `opencv-python-headless`, `scenedetect`, `einops`, `peft`, `numpy<2`, plus the pinned static ffmpeg build from ngx-vsr; attach `modal_app/` last so code edits do not invalidate the dependency layers
- [x] 1.5 Add a slim torch-free CPU image for the web tier

## 2. Weight provisioning

- [x] 2.1 Write `modal_app/download_weights.py`: CPU Function using the `huggingface` Secret to `snapshot_download("JiongzeYu/SparkVSR")` into the models Volume, idempotent on re-run, reporting bytes written
- [x] 2.2 Extend it to fetch the PiSA-SR SD 2.1 base weights into the same Volume. Note: `pisa_sr.pkl` has no public programmatic download (upstream ships it via Google Drive only), so it is operator-supplied via `modal volume put` or an optional `PISASR_WEIGHTS_URL`, and only the `pisasr` reference mode requires it
- [x] 2.3 Add the empty-prompt embedding step: load the text encoder once, encode `""`, save the tensor to the Volume, and log its path
- [x] 2.4 Add a `verify_weights()` helper that checks every expected file is present and complete, returning what is missing
- [x] 2.5 Run `modal run -m modal_app.download_weights` and confirm weights and the embedding land on the Volume — SparkVSR (22 GB), SD 2.1 base and `empty_prompt_embedding.pt` ([1, 226, 4096] bf16) verified on `sparkvsr-models`

## 3. Vendored inference core

- [x] 3.1 Create `modal_app/pipeline/core.py` with an Apache-2.0 attribution header naming the upstream repo and commit
- [x] 3.2 Port `preprocess_video_match` (8n+1 frame padding, spatial padding to a multiple of 4) and `remove_padding_and_extra_frames`
- [x] 3.3 Port `make_spatial_tiles` and `get_valid_tile_region` for the optional spatial-tiling path
- [x] 3.4 Port `prepare_rotary_positional_embeddings` and `process_video_ref_i2v`, changing only the prompt path to always use the stored empty-prompt embedding
- [x] 3.5 Write `load_pipeline()`: `CogVideoXImageToVideoPipeline.from_pretrained(..., text_encoder=None, tokenizer=None, torch_dtype=bfloat16)` from the Volume, `CogVideoXDPMScheduler.from_config(..., timestep_spacing="trailing")`, VAE slicing and tiling enabled, TF32 and `cudnn.benchmark` on; fail loudly if the embedding is absent
- [x] 3.6 Write the window executor: run each planned window, blend overlaps within a shot only, assemble the full output tensor

## 4. Cut-aware planning

- [x] 4.1 Write `modal_app/pipeline/cutplan.py`: PySceneDetect adaptive detection (threshold 3.0, minimum scene 0.6 s) returning shot boundaries
- [x] 4.2 Turn shots into windows: overlapping 49-frame windows within a shot, never crossing a boundary, covering every frame
- [x] 4.3 Apply model length constraints per window (pad to at least 9 frames and to an `8n+1` length, record the crop-back amounts)
- [x] 4.4 Assign reference indices: at least one per window, no more than 0.5 s into the window, spaced more than 4 frames apart
- [x] 4.5 Add the fixed-window fallback planner used when cut-awareness is disabled, producing the same plan structure
- [x] 4.6 Serialize the plan (cuts, windows, padding, reference indices) into the job output
- [x] 4.7 Add unit tests for the planner with synthetic shot lists — no GPU, no model: coverage of every frame, no window crossing a cut, every window has a reference, reference spacing greater than 4, padding cropped back to exact source length

## 5. Reference keyframe generation

- [x] 5.1 Write `modal_app/pipeline/refgen.py` with a mode-dispatching entry point and a per-job cache directory on the io Volume
- [x] 5.2 Implement `pisasr` mode: vendor `pisasr_src` from the upstream ComfyUI wrapper, load lazily, upscale each planned reference frame to the target resolution, then free GPU memory
- [x] 5.3 Implement `api` mode against `fal-ai/nano-banana-pro/edit` using the fixed restoration prompt from upstream `ref_utils.py`, reading the `fal` Secret
- [x] 5.4 Implement `no_ref` mode as a no-op returning an empty reference set
- [x] 5.5 Make reference generation failures fail the job with a diagnostic instead of substituting a bicubic upscale
- [x] 5.6 Persist generated references so a retry reuses them and a caller can inspect them

## 6. Video I/O and encoding

- [x] 6.1 Write `modal_app/pipeline/video_io.py`: decode to frame tensors, reading fps, dimensions, and audio presence
- [x] 6.2 Port the probed encoder-selection ladder from `../ngx-vsr/modal_app.py` (`hevc_nvenc` → `h264_nvenc` → `libx264`), each verified by a real one-frame encode, logging the selection
- [x] 6.3 Port the 10-bit HEVC command with its quality settings and explicit bt709 tagging
- [x] 6.4 Port rational frame-rate handling via `Fraction(fps).limit_denominator(1001)`
- [x] 6.5 Port audio stream-copy (`-c:a copy -shortest -movflags +faststart`), handling sources with no audio track

## 7. GPU service

- [x] 7.1 Write `modal_app/service.py` as an `@app.cls` on `RTX-PRO-6000` with both Volumes mounted, a 1-hour timeout, and a scaledown window
- [x] 7.2 Implement `@modal.enter()`: verify weights, load the pipeline, log model load time
- [x] 7.3 Implement the job flow: decode → plan → generate references → execute windows → assemble → encode
- [x] 7.4 Implement output sizing: bilinear pre-upscale of the source to the target resolution preserving aspect ratio, including the same-resolution refresh mode
- [x] 7.5 Add per-stage timing, peak GPU memory, and chunk count to the job logs
- [x] 7.6 Make OOM failures name the spatial-tiling flag in their message
- [x] 7.7 Add the long-input orchestrator: segment on shot boundaries, process segments, concatenate without re-encoding

## 8. Web tier and client

- [x] 8.1 Write `modal_app/web_api.py` on the slim CPU image: authenticated multipart upload streamed to the io Volume, `validate_request()` before spawning GPU work, spawn returning a job id
- [x] 8.2 Implement the result endpoint: pending status while running, file response when complete, diagnostic message on failure, job directory cleaned up afterwards
- [x] 8.3 Implement validation covering unsupported inputs, out-of-range parameters, reference indices spaced 4 frames or closer, and `api` mode without the `fal` Secret configured
- [x] 8.4 Write `sparkvsr_client.py` (ported from `../FlashVSR-Pro/flashvsr_client.py`) with `--target-height`, `--ref-mode`, `--ref-guidance`, `--no-cut-aware`, `--no-audio`, `--tile`
- [x] 8.5 Write `run_sample.py` as a `@app.local_entrypoint()` smoke test over `sample/jopet_10s.mkv`

## 9. Verification

- [x] 9.1 Run the smoke test end to end and record whether 1080p→4K fits in 96 GB without spatial tiling; if it does not, enable tiling by default and note it — 4K peaked at 49.06 GB of 96 GB with no tiling; `--tile` stays off by default
- [x] 9.2 Inspect the smoke-test output for seams at window boundaries, and `ffprobe` it to confirm bt709 tagging, 10-bit HEVC, correct fps, and a stream-copied audio track — hevc/yuv420p10le, bt709 on all three fields, 25/1, AAC stereo, 301/301 frames; temporal-difference scan at the blend edges (41/49/82/90/247/255) stays within ±9% of local behaviour, so no measurable seam
- [x] 9.3 Run `sample/jopet_60s.mkv` with and without `--no-cut-aware`; confirm the serialized cut plan matches the visible scene changes and that cut-aware output shows no cross-scene ghosting — the plan's shot boundaries (727, 836, 1195) match the clip's three hard cuts exactly, with no false positives at the default threshold of 3.0. Correlating each restored frame's residual against the outgoing scene shows ghosting without cut-awareness (+0.55 at the cut, decaying over 5–8 frames) and none with it (flat at the mid-shot baseline)
- [x] 9.4 A/B one clip across `no_ref` and `pisasr` and record the visible difference in restored detail — at 1080p, Laplacian variance is ~23 (source), 77–106 (`no_ref`), 163–211 (`pisasr`): references roughly double restored detail over blind, at +38 s of reference generation
- [x] 9.5 Deploy the web tier and complete a client round-trip: upload, poll, download — deployed to https://tinof--sparkvsr-api.modal.run; `sparkvsr_client.py` uploaded, polled and downloaded a 301-frame 1080p result through proxy auth
- [x] 9.6 Record measured cost and wall-clock time per minute of 1080p→4K output in the README, with the GPU type and settings used — ~$10.30 and ~2.8 h per output minute on RTX PRO 6000
- [x] 9.7 Write the README: architecture, provisioning, deployment, CLI usage, and tuning knobs. The cost section is explicitly marked unmeasured until 9.6 fills it in

## 10. Defects found during bring-up (2026-08-18)

Seven changes were needed before the pipeline produced a correct render. All are fixed and
verified by a real render.

- [x] 10.1 `service.py` imported `torch` and the pipeline modules at module scope, so every `modal run` died locally with `ModuleNotFoundError`. Moved into `with gpu_image.imports():`
- [x] 10.2 `service.py` pulled `verify_weights` from `download_weights`, dragging that module's `@app.local_entrypoint()` into the graph and colliding with `run_sample.py`'s `main`. Helpers extracted to `modal_app/weights.py`
- [x] 10.3 `extract_raw_reference_frames` called `.asnumpy()` on a decord batch, but `decode_video_to_uint8` sets the process-global torch bridge first, so it returns a Tensor. Now accepts either
- [x] 10.4 `preprocess_video_match` padded a 5D tensor with a 4-element pad tuple; replicate padding on 5D requires all three trailing dims. `NotImplementedError` on every window at 1080p
- [x] 10.5 `stabilityai/stable-diffusion-2-1-base` is delisted from the Hub (404 with a valid token), breaking provisioning. Repointed to `sd2-community/stable-diffusion-2-1-base` with an allow-list that skips ~11 GB of unused variants
- [x] 10.6 `-shortest` in `build_encode_command` truncated the video against the copied audio track, silently dropping trailing frames (301 in, 296 out). Removed
- [x] 10.7 The same `-shortest` in `concat_segments_no_reencode` truncated multi-segment output (1551 in, 1546 out on the 60 s clip). Removed and confirmed by a re-render: the two-segment 60 s job now writes all 1551 frames
- [x] 10.8 `CogVideoXDPMScheduler.from_config` was passed a directory path instead of the loaded config dict (deprecated, removed in diffusers 1.0). Now uses `pipeline.scheduler.config`

## 11. Performance workstream (2026-08-19, not yet run on GPU)

Applied in full; nothing below has executed on a GPU, since Modal credits are exhausted.
Measured baseline for comparison: 0.7-0.8 fps at 1080p, inference 98% of wall clock.

- [x] 11.1 Default target resolution changed to 1080p (same-resolution restore is the primary workflow; 4K remains available via `--target-height 2160`)
- [x] 11.2 `torch.compile` on the transformer with `dynamic=True` (window length varies at shot ends), plus optional VAE-decode compile; degrades to eager on failure
- [x] 11.3 Explicit SDPA fast-kernel enablement (flash / mem-efficient / cuDNN) with the math fallback retained
- [x] 11.4 FP8 dynamic quantisation via torchao, behind `SPARKVSR_FP8`, off by default because it is the one knob that can alter output
- [x] 11.5 `torch.profiler` breakdown of the first window behind `SPARKVSR_PROFILE`
- [x] 11.6 Per-window inference timings with a steady-state median that excludes the compile warmup, so a 5 s clip can settle an A/B
- [x] 11.7 Shot-level fan-out: `process_video_parallel` (CPU driver) plans, `.map()`s segments across GPU workers, concatenates with `-c copy`
- [x] 11.8 Safety guards found by review: driver Volume reload, CPU pre-flight, output-file frame probe on both paths, segment ordering and encoder-agreement assertions, memory pre-check, worker retries, separate driver timeout
- [x] 11.9 Shot boundaries clamped to the decord frame count (PySceneDetect's decoder disagrees on VFR/MKV)
- [x] 11.10 12 unit tests for segmentation and pre-flight (20 total, CPU-only)
- [ ] 11.11 **Run the validation ladder (~$4).** Profile, then baseline / compile / FP8 on `bench_5s.mkv`, then a 3-worker parallel smoke on the 10 s clip, then the 60 s clip end to end. See README "Validation budget"
- [ ] 11.12 Record the measured speedup and update the cost table; the 2-3x figure is an estimate until 11.11 runs
