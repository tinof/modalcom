# FlashVSR on Modal

Diffusion video super-resolution on [Modal](https://modal.com), as two separate apps in one package:

| App | What it is | State |
|---|---|---|
| `flashvsr-plus` | [FlashVSR+](https://github.com/OpenImagingLab/FlashVSR_Plus) as a segment worker for long videos: plan at scene cuts, render segments on several GPUs, join, carry audio and subtitles over. | Written 2026-11-25. **Never built or run on Modal.** Tested locally without the model's weights. |
| `flashvsr-pro` | FlashVSR v1.1 as an authenticated HTTP service for short clips. | Recovered from a deployed image (see Provenance). Not deployed; its Volumes no longer exist. |

Read "What is and is not tested" before you spend money on either.

## Cost first

FlashVSR is a one-step diffusion model. On a 3840x2176 canvas it runs at roughly 0.6 to 2 frames per second on an RTX PRO 6000, so a 68-minute 25 fps episode (101,950 frames) costs 14 to 47 GPU-hours. At about $3.5 per GPU-hour that is $50 to $165. `rtx-vsr` (one folder up) upscales the same episode for under $3.

These numbers are not measured here. They are scaled from the only public figures, upstream PR #3 (RTX 5090, 144 frames, 1536x2688 canvas):

| | stock upstream | with CPU copies removed |
|---|---|---|
| turbo | 1.4 fps | 2.6 fps |
| base | 1.2 fps | 2.0 fps |
| v1 | not given | 3.8 fps |

A 4K canvas has twice the pixels. So the first thing to run on new material is the look test below: it costs a few dollars and tells you whether FlashVSR+ beats `rtx-vsr` on that material at all, and what a full run will cost.

## What "use the whole GPU" means for this model

In `rtx-vsr` the model takes 3.5 ms per frame and NVENC is the bound, so the win there was a GPU-resident path and four NVENC sessions. Here the DiT is the bound and one NVENC session is 20 to 60 times faster than the model. Extra NVENC sessions buy nothing.

What does matter, and what the worker does about it:

1. **No CPU round trips inside the model's runner.** Stock upstream moves the whole DiT (5.7 GB) to the CPU and back around every decoder call, copies every decoded chunk to the CPU, and rescales it there. All of that exists to fit a 24 GB card. `plus_runtime.resident_transport` removes it for a run. `--no-resident` restores stock behaviour, so the gain can be measured.
2. **Spend the 96 GB.** The KV cache is never offloaded, nothing is tiled.
3. **Keep the model busy.** Decode runs on a reader thread and encode on a writer thread, each behind a bounded queue.
4. **More containers** shorten the wall time. They do not lower the cost.

Open, to be measured: the attention kernel, upstream's float64 rotary embeddings (`--rope-float32`), `torch.compile` (`--compile-dit`), the parallel TinyDecoder (`--tiny-decoder-mode parallel`), and a second stream on the same card (only worth trying if one stream leaves the GPU under about 85% busy). PyTorch's own attention (`--attn-kernel sdpa`) cannot run at 4K: upstream refuses the mask it would need.

## Layout

| Path | What it is |
|---|---|
| `modal_app/plus_plan.py` | Segment planner and frame arithmetic. Standard library only. |
| `modal_app/plus_video.py` | Probe, scene scores, frame-exact decode, encode, seam cross-fade, join, verification. numpy and ffmpeg, no torch, no Modal. |
| `modal_app/plus_job.py` | A job on disk: prepare, render seams, finish. No Modal. |
| `modal_app/plus_runtime.py` | Drives upstream's stream runner from our frame source and sink; the resident transport. torch. |
| `modal_app/plus_service.py`, `plus_episode.py` | The Modal wrappers: GPU worker class and CPU orchestrator. |
| `modal_app/plus_image.py`, `plus_weights.py`, `plus_probe.py`, `plus_deploy.py`, `plus_app.py` | Image, weight download, hardware probe, deploy entrypoint, App and Volumes. |
| `modal_app/bsa_build.py`, `gpu_probe.py` | Stack pins, the cached Block-Sparse-Attention build and the hardware checks, shared by both apps. |
| `client.py` | Local client of the deployed `flashvsr-plus` app. |
| `tests/` | See "What is and is not tested". |
| `modal_app/{app,image,service,web_api,orchestrator,video_io,download_weights,probe_gpu,deploy}.py` | The v1.1 service. |
| `diffsynth/`, `utils/`, `infer.py`, `setup.py`, `requirements.txt` | Vendored FlashVSR-Pro source, used by the v1.1 service only. |
| `Block-Sparse-Attention/` | Vendored CUDA kernels, compiled at image-build time for both apps. |

Run every `modal` command from this directory: the images copy `Block-Sparse-Attention/` (and, for v1.1, the vendored trees) by relative path.

## FlashVSR+: how a job runs

1. `client.py upload` puts the source on the `flashvsr-plus-io` Volume.
2. `run_job` (CPU) probes it, scores every frame for scene changes in one low-resolution pass, and plans segments of about 60 s. Each cut is the strongest scene change within 30 s of the target. Every segment restarts the model's stream, so two segments never agree on the fine detail they synthesise; on a hard cut nobody can tell. Where no cut qualifies, the two segments overlap by 8 frames that are cross-faded, and the later one first runs 32 warm-up frames through the model.
3. `PlusWorker.upscale_segment` (GPU) renders one segment per call: frame-exact decode from any start frame, bicubic upscale and reflect padding to the model's canvas on the GPU, the model, crop, HEVC Main10 through `hevc_nvenc`. The encoder settings are `rtx-vsr`'s (CQ 20, preset p5, UHQ, level 5.1 High, 100 Mbit/s ceiling, BT.709 tags).
4. `run_job` checks every part (planned frame count, an IDR first, identical codec parameters), joins them by stream copy, and copies audio, subtitles, attachments, chapters and metadata from the source into `jobs/<job>/output.mkv`.

A job is resumable: run the same job id again and only the missing segments are rendered. Nothing is repaired silently. A frame count that differs from the plan fails the segment.

Source limits: progressive, constant frame rate, SDR. Interlaced, variable-rate and HDR sources are refused with a message. 10-bit sources are read at 8 bits.

## Runbook for a first run

Nothing below has been run. Expect to fix things in step 1.

### 0. Build, weights, probe (about $3 to $6, mostly the one-time kernel compile)

```bash
modal run -m modal_app.plus_weights          # ~20 GB to the flashvsr-plus-models Volume
modal deploy -m modal_app.plus_deploy        # builds the image; first build compiles BSA, 10-40 min
modal run -m modal_app.plus_probe            # must print no FAILED line
```

The probe runs a real NVENC encode, checks the Block-Sparse-Attention kernel against an exact reference on a sparse mask, and confirms upstream imports at the pinned commit with the names `plus_runtime.py` patches.

### 1. I/O check without the model (cents)

```bash
uv run --with modal --with numpy python client.py upload /path/to/film.mkv
uv run --with modal --with numpy python client.py submit inputs/film.mkv --job io-check \
    --bypass-model --start 600 --duration 20 --wait
```

`--bypass-model` writes the bicubic upscale the model would have been conditioned on. Fetch it and compare with the source: same frames, same colours.

### 2. Look test (about $5 to $8)

```bash
uv run --with modal --with numpy python client.py look-test /path/to/film.mkv \
    --starts 120 900 2400 --seconds 8 --out /path/outside/the/repo/bench
```

Pick starts for a face close-up, a wide shot with foliage or water, and a shot with on-screen text; small text is where diffusion upscalers fail. The command cuts each excerpt locally as a lossless, frame-exact re-encode (40 to 100 MB for 8 s of 1080p), uploads it, and renders it with turbo, base and v1 at 2x and at 1.5x delivered as 2x (44% fewer pixels to render). The command prints frames per GPU-second, GPU utilisation, peak VRAM, bitrate, and the projected cost of the whole file. Then run the same excerpt files in `bench/src/` through `rtx-vsr` and compare, on the real screen and with `rtx-vsr/scripts/compare_frames.py`.

**Stop here** and decide whether FlashVSR+ is worth 20 to 60 times the cost of `rtx-vsr` on this material, and with which preset and scale.

### 3. Speed benchmark (about $10 to $15)

One container (`FLASHVSR_PLUS_MAX_CONTAINERS=1 modal deploy -m modal_app.plus_deploy`), the chosen preset, one 60 s excerpt per run (`--start S --duration 60`), each run twice, read the second:

| Run | Flags |
|---|---|
| baseline | (none) |
| stock transport | `--no-resident`, deployed with `FLASHVSR_PLUS_MEMORY_MIB=65536` |
| float32 rotary embeddings | `--rope-float32` (changes the output; compare it) |
| compiled DiT | `--compile-dit` (experimental; compare the output) |
| parallel TinyDecoder | `--tiny-decoder-mode parallel` |
| colour correction | `--color-correction` |
| all seams soft, to judge the cross-fade | `--duration 180 --scene-threshold 2` |

Knobs are read at deploy time and baked into the image. A bare `FLASHVSR_PLUS_...=x` in front of `client.py` changes nothing.

The stock transport keeps float32 frames, the decoded chunks and the offloaded DiT in host memory. At a 4K canvas that is an estimated 45 to 55 GB, more than the worker's default 32 GB, so that row needs the larger memory setting or it is killed.

Before the full run, deploy with the normal container count and run one 10-minute excerpt. Its report lists the `hosts` that rendered it. If the join passes with segments from at least two hosts, their encoders agree. If it stops with "codec parameters differ", segments from different driver generations cannot be joined by stream copy: the report names the segment, and the only remedy today is to delete that segment's `.ok` marker on the Volume and run the job again until it lands on a matching host. Settle that before paying for a full run.

**Stop here** with the measured frames per GPU-second and approve the cost per episode.

### 4. Full file

```bash
uv run --with modal --with numpy python client.py submit inputs/film.mkv --job film --mode turbo
uv run --with modal --with numpy python client.py status <call id>
uv run --with modal --with numpy python client.py fetch film /path/outside/the/repo/film.2160p.mkv
uv run --with modal --with numpy python client.py clean film
```

Expect 25 to 40 GB for a 68-minute episode at CQ 20: synthesised detail is expensive to encode. `clean` removes the segment files and keeps the joined output; it also ends the job's ability to resume.

## What is and is not tested

Local, no Modal, no weights:

```bash
pytest tests/                                   # 375 tests, needs ffmpeg with libx264 and libx265
ruff check modal_app client.py tests
```

- The planner over every case that matters (hard cut, soft seam, cut outside the window, excerpts, a 101,950-frame input), and that a plan delivers every frame exactly once.
- A real ffmpeg round trip with `libx265` standing in for `hevc_nvenc`: frame-exact decode from arbitrary start frames, segments, a cross-faded seam, the join, a resampled delivery size, and an encoder that fails to start. A synthetic source carries its frame number in the picture; the result has every frame once and in order, exactly 25/1 fps, BT.709 tags in stream and container, audio and subtitles copied with their timestamps, channel means within one 8-bit level of the source.
- The fixes to the v1.1 service listed below.

With torch and an upstream checkout at the pinned commit (see the top of `tests/test_plus_runtime.py`; 43 more tests):

- `plus_runtime.py` through upstream's real `StreamRunner`, with the DiT and the TinyDecoder replaced by stubs that pass frames through. Every frame comes back once and in order for clip lengths from 1 to 250 frames, stock and resident transport agree, the resident transport is undone after a run and after a failure, and the float32 rotary embedding matches upstream's float64 one within bf16 rounding.
- A whole job on disk: plan, render, seam, join, resume after a lost segment, no reuse after a parameter change, an excerpt without the model.
- It ran on torch 2.15 (CPU), so upstream's inference code at least imports and steps on a torch far newer than the `<2.8` in its requirements file.

Checked once by hand on a real 1080p25 episode: lossless 8 s excerpts cut the way `look-test` cuts them are constant-rate, pass the scene pass and are pixel-identical to the source frames. The planner has not yet run over a whole real file (see `HANDOFF.md`).

Not tested at all:

- The image build, including the Block-Sparse-Attention build for sm_120 alone and upstream on torch 2.12 with CUDA 13.
- The weight download and the v1.1 projector conversion.
- The real model on a GPU: correctness, speed, VRAM, the resident transport's gain.
- `hevc_nvenc` with these arguments through this ffmpeg build, on either of Modal's driver generations (580 and 610). Segments from hosts with different drivers may get different parameter sets; the join refuses them and names the segment. NVENC also refuses a session at random when a level is set (`rtx-vsr` measured 3 to 5 in 30); the encoder is restarted with the frames written so far, up to six times, which is tested with a stand-in process and not with NVENC.
- The Modal wiring: Volumes v2 under concurrent commits, `map` over the worker class, timeouts.
- `client.py` against a deployed app.

## Differences from the plan this was built from

- There is no separate benchmark module that runs upstream's own CLI. The look test goes through the same worker as a full job. Upstream's CLI centre-crops to the canvas and writes untagged 8-bit H.264 at its library's default quality, which cannot be compared with `rtx-vsr`'s output frame for frame. The price is that the look test depends on the whole worker.
- A part whose codec parameters differ is refused at the join. It is not re-encoded on a matching host.
- The v1.1 service rejects `dtype` values other than `bf16`. It does not load the models in the requested dtype.

## Upstream pin

`config.PLUS_REPO_SHA` pins `OpenImagingLab/FlashVSR_Plus`. `plus_runtime.py` replaces two module-level functions of upstream's runner by name and subclasses its `TinyDecoder`, all read at that commit. Move the pin only together with a re-read of `inference/runners/stream.py` and `inference/decoders/tiny.py`, and run `tests/test_plus_runtime.py` against the new commit.

The stack is the one the v1.1 image proved on Blackwell (CUDA 13, torch 2.12.1). Upstream's `torch>=2.5,<2.8` is in a requirements file its own install skips. If upstream breaks on 2.12, the fallback is `torch==2.7.1` from the cu128 index on a CUDA 12.8 base.

## The v1.1 service (`flashvsr-pro`)

```bash
modal deploy -m modal_app.deploy
modal run -m modal_app.probe_gpu
```

A CPU web tier (`api`) in front of three GPU classes, job I/O through Volumes. It holds a whole clip in memory (about 12 GB of VRAM and well over 25 GB of RAM for 10 seconds at 4K), so it is for short clips only.

Review of 2026-11-25. Fixed:

- Real frames were lost at the tail. The input was padded to the next 8k+1 frames, the streaming loop emits 4 fewer than it is given, and the gap was hidden by repeating the last frame. Clips of 17 frames or fewer crashed. The padding now covers the loop's loss, and a short result is an error.
- With an output not named `.mp4`, the audio fallback deleted the finished output.
- `dtype` other than `bf16` was accepted and applied to the input only, while the models are always loaded in bf16. Only `bf16` is accepted now.
- `probe_gpu` listed NVENC encoders instead of opening a session and checked the kernel only for finite output on a dense mask. It now runs real encodes and compares the sparse path with a reference.
- Docstrings and comments named GPUs and a torch version that are not the ones pinned.

Known and not fixed:

- `orchestrator.py` (the long-video path) is dead code: `deploy.py` does not register it, and it imports the GPU image on a CPU image. Use `flashvsr-plus` for long videos.
- The web tier mounts the io Volume under `@modal.concurrent` and calls `reload()`, which can fail or drop another request's upload (`rtx-vsr/CLAUDE.md` describes the hazard).
- Colour: decode and encode run through swscale's defaults with no matrix and no tags, so output is encoded BT.601 and read as BT.709. Output is 8-bit H.264 with no level or bitrate ceiling.
- The encoder pipes are written with stderr unread; an encoder that floods stderr would block.
- One call handles at most about 4,090 frames (the temporal position table has 1024 entries). Nothing checks it.
- The vendored `requirements.txt` has been bumped by Dependabot into a set that cannot resolve. The image ignores it; the pins that count are in `modal_app/bsa_build.py` and `modal_app/image.py`.

The Block-Sparse-Attention wheel is cached on the `flashvsr-build-cache` Volume, keyed on kernel-source hash, torch version and arch list. The arch list is now `120` (Blackwell) only; the two Ampere archs it used to carry tripled the compile and neither app runs there.

## Provenance of the v1.1 code

The local working copy of this project was destroyed on 2026-08-18, and the `tinof/FlashVSR-Pro` GitHub repo had already been deleted, so no git history survived. The v1.1 tree was recovered from the deployed Modal image (`flashvsr-pro` v10, deployed 2026-08-16 15:28 from commit `1adf4ba` plus uncommitted changes) by booting a sandbox on that image and extracting `/root/modal_app` and `/workspace/FlashVSR-Pro`. Any local edit made after that deploy is not here.

## Attribution

The v1.1 service is a fork of [FlashVSR-Pro](https://github.com/LujiaJin/FlashVSR-Pro), with DiffSynth-Studio-derived `diffsynth/`. Both apps use the vendored [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) (see that directory's `LICENSE`). FlashVSR+ is [OpenImagingLab/FlashVSR_Plus](https://github.com/OpenImagingLab/FlashVSR_Plus) (Apache-2.0), cloned at build time, with weights from `ruchangyao/FlashVSR_plus` and `JunhaoZhuang/FlashVSR-v1.1`. The decode and encode commands are ported from `rtx-vsr`. Upstream licenses apply to the vendored and cloned code.
