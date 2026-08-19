# CLAUDE.md — RTX Media Upscaler on Modal

Guidance for AI agents working in this repository. User-facing setup and API docs live in
[README.md](README.md); this file covers conventions, constraints, and the traps that have
already cost real debugging time.

## What this project is

A Modal service (`modal_app.py`, `modal_client.py`): NVIDIA RTX Video Super Resolution
exposed as an authenticated HTTP job queue on Modal.

It began as a port of the Comfy-Org ComfyUI custom node (see [NOTICE](NOTICE) for the
Apache-2.0 attribution), and deliberately keeps that node's resize-type handling, quality
mapping, dimension alignment (multiples of 8), and `MAX_PIXELS = 1024 * 1024 * 16` batching
heuristic. The node itself is no longer carried here. Those behaviours are now this
service's contract rather than a parity target, so changing them is a breaking API change —
say so explicitly rather than "improving" them in passing.

**This project is also the template for other Modal jobs.** The section below marks which
parts generalize and which are specific to super-resolution.

## Architecture patterns worth copying

These are the reusable parts. They are not decoration — each one fixes a concrete problem.

- **Split the web tier from the GPU tier.** `api` is a CPU-only ASGI function on a slim
  image; `UpscaleWorker` is an `@app.cls()` that owns the GPU. If the GPU is attached to
  the web function, you pay GPU rates for uploads, downloads, and idle time.
- **Job queue, not synchronous HTTP.** Modal web endpoints enforce a **150-second** request
  timeout. Anything slower must `.spawn()` and return a `call_id`, with a separate
  `GET /result/{call_id}` that returns 202 while pending. A `timeout=3600` on the function
  does not extend the HTTP window.
- **Move payloads through a `modal.Volume`, not through function arguments.** The web tier
  streams uploads to the Volume in chunks so a multi-GB file never lands in memory.
- **Warm expensive state in `@modal.enter()`.** CUDA init, model load, and encoder probing
  belong there, once per container, not once per request.
- **`requires_proxy_auth=True` on any endpoint that spends GPU money.** Modal web functions
  are public by default.
- **Let Modal handle concurrency.** Without `@modal.concurrent`, Modal sends one input per
  container and scales horizontally. Do not add locks to serialize work — the CPU-only web
  tier uses `@modal.concurrent` because it is I/O-bound; the GPU worker deliberately does not.
- **Keep GPU data on the GPU.** The single largest win here (12 → 59 fps) was not a faster
  stage; it was deleting the host round-trips between them. If frames, tensors, or images
  cross a process boundary between GPU stages, that copy — not any kernel — is very likely
  your ceiling. Decode, process, and encode in one process against CUDA memory.

## Hard constraints — do not change casually

These are pinned in `modal_app.py` and explained in code comments. Each was discovered the
expensive way.

### 1. `libnvidia-ngx.so.1` must be installed into the image (`DRIVER_VERSION`)

RTX VSR is an **NGX** feature. The `nvidia-vfx` wheel bundles the NGX snippet but
`dlopen`s the *driver-side* `libnvidia-ngx.so.1`, which Modal's container runtime does not
inject. Without it, `NvVFX_Load` fails on **every GPU type**:

```
NvVFX_Load failed: The effect has not been properly initialized (code -12)
```

The image installs that one library from the matching NVIDIA driver `.run` package.
`DRIVER_VERSION` must track the host driver from `nvidia-smi`; `_warn_on_driver_mismatch()`
logs a warning at container start when they diverge. If VSR breaks after a Modal driver
upgrade, this is the first thing to check.

Note this failure is *not* GPU-model-specific — do not go GPU shopping when you see `-12`.

### 2. ffmpeg's NVENC support is gated on the driver (`FFMPEG_URL`)

Newer is **not** better here. Each ffmpeg build links a Video Codec SDK with a minimum
driver version:

| Build | SDK | Requires driver | Works on Modal (580.95.05) |
|---|---|---|---|
| BtbN `master-latest` | 13.1.15 | 610.0+ | **No** |
| BtbN `n8.1` (pinned) | — | 570.0+ | Yes, incl. `av1_nvenc` |
| Debian apt ffmpeg 5.1 | — | — | Yes, no `av1_nvenc` |

Pointing `FFMPEG_URL` at `master-latest` makes every NVENC encoder fail to initialize and
the worker falls back to CPU `libx264` — correct output, roughly an order of magnitude
slower. Pin versioned URLs, never `latest`/`master`, which are moving targets that make
image builds unreproducible.

### 3. GPU choice and torch version are coupled

Default is `RTX-PRO-6000` (Blackwell): verified working, and the only Modal GPU with 9th-gen
NVENC. B200/B300 and A100 have **no NVENC engine at all**. Blackwell is `sm_120`, so
`torch==2.13.0` is required — torch 2.6 fails with "no kernel image is available for
execution on the device". Upgrading the GPU forces upgrading torch.

### 4. VSR 15360px limit, DLPack buffer clone, and output integrity

- **SDK Defect at 16K Output**: The NVIDIA VFX SDK has a verified bug at extreme output
  sizes. Probed on Linux + Blackwell RTX-PRO-6000:
  - At **15360x15360**: input means `[0.5, 0.5, 0.5]` -> output means `[0.4998, 0.5007, 0.5012]` (clean).
  - At **16384x16384**: input means `[0.5, 0.5, 0.5]` -> output means `[0.4998, 0.5007, 0.0]` (Blue channel completely collapsed to 0).
- **Hybrid Resize Path**: When requested output is between 15360 and 16384, the worker runs
  VSR capped at 15360 and resamples to target dimensions using GPU `torch.nn.functional.interpolate(mode="bicubic")`
  in float before uint8 conversion.
- **DLPack Explicit Clone**: `nvvfx` reuses its internal DLPack buffer across calls. We MUST
  `.clone()` the PyTorch tensor immediately after `torch.from_dlpack()`.
- **Integrity Check**: Every frame is verified on GPU for `torch.isfinite` and channel collapse
  (`_check_frame_integrity`) so corrupted SDK outputs fail loudly instead of returning bad files.
- **Do NOT remove** the hybrid path, `.clone()`, or integrity check as "dead code" or "premature optimization".

### 5. Quality modes: four families, two roles

`nvvfx.VideoSuperRes.QualityLevel` is not just LOW/MEDIUM/HIGH/ULTRA. It has 19 members in
four families, **all probed working** on `RTX-PRO-6000` / driver 580.95.05:

- `BICUBIC, LOW..ULTRA` (0-4) and `HIGHBITRATE_LOW..ULTRA` (16-19) **change resolution** —
  valid for `quality`. The standard family also suppresses compression artifacts, so it
  softens clean sources; `HIGHBITRATE_*` skips that for ProRes-grade input.
- `DENOISE_LOW..ULTRA` (8-11) and `DEBLUR_LOW..ULTRA` (12-15) are **same-resolution**
  restoration — valid only for `preprocess`, which runs a second pass at input size before
  upscaling.

`UPSCALE_QUALITIES` / `SAME_RES_QUALITIES` in `modal_app.py` encode this split, and the web
tier rejects a mismatched role with 400 before spending GPU time. `_normalize_quality()`
deliberately validates against those frozensets rather than importing `nvvfx`, because the
web tier runs on the slim CPU image where the SDK is not installed.

The preprocess pass gets its own cached effect instance (`role="preprocess"` in
`_super_res`), so the two models coexist without evicting each other between frames. Its
output needs the same mandatory `.clone()` as the upscale pass — same reused-DLPack-buffer
trap described in section 4.

**Judging output quality needs a real source check first.** A "1080p" broadcast file is
often an SD master someone else already upscaled. Downscale a frame to 540p and back: if
that reconstructs the file almost exactly (mean abs diff < 1/255), there is no real detail
for VSR to recover and the result will look near-identical to Lanczos no matter the
settings. That is the source's ceiling, not a pipeline defect.

**The default is `HIGHBITRATE_ULTRA` (19), not `ULTRA` (4)** — set in the `quality` Form
default. "High bitrate" means *few compression artifacts*, not *high resolution*. Broadcast
archive material is typically soft but cleanly encoded, so the standard family's artifact
suppression has nothing real to remove and instead damages texture, producing a blotchy
painterly look. Measured on the reference clip, mode 19 beat mode 4 on both axes: ~15% more
edge detail and ~8x less flat-region noise. Only prefer the standard family for genuinely
blocky, heavily compressed encodes.

**Two counter-intuitive facts about DEBLUR, both measured, both contradicting the names:**

1. **Intensity is not monotonic.** `DEBLUR_LOW`/`MEDIUM` yield *more* apparent detail than
   `HIGH`/`ULTRA`. The docs describe ULTRA as "maximum sharpening strength"; observed
   behaviour is the opposite, so the upper rungs evidently regularise more as well.
2. **Rank deblur by temporal flicker, not by still-frame metrics.** Flat-region noise that
   varies frame to frame is what looks bad in motion. `DEBLUR_ULTRA` adds ~50% detail for
   only +5% flicker; `DEBLUR_LOW` scores best on stills and flickers most (+22%).

So a still-frame sharpness metric will actively mislead you here. When evaluating any
preprocess change, measure flicker across consecutive frames in a static patch — the helper
in `scripts/preprocess_sweep.py` (`detail_vs_noise`) covers the spatial half; the temporal
half is a mean abs diff between consecutive frames in a static region.

`scripts/compare_frames.py` and `scripts/preprocess_sweep.py` build the side-by-side and
n-panel comparison grids used for these judgements; both take `--variants label=path` pairs.

## The scripts directory

| Script | Use it when |
|---|---|
| `compare_frames.py`, `preprocess_sweep.py` | Judging VSR quality / preprocess modes (above). |
| `probe_gpu_encoder.py` | Re-testing which `CreateEncoder` kwargs are live. One encoder session per container — see section 6. |
| `probe_gpu_pipeline.py` | Re-testing the NVDEC decoder and rate control on real content. |
| `probe_nvdec.py` | Re-testing decoder classes after a PyNvVideoCodec upgrade (is `ThreadedDecoder` fixed yet?). |
| `probe_nvdec_rgb.py` | Re-testing the NVDEC `OutputColorType.RGB` path and its surface layout. |
| `probe_p010_encode.py` | Any suspicion of corrupt GPU-encoder output — striping, stale or flat frames. See the sync bug below. |
| `verify_roundtrip.py` | Running the deployed worker on a Volume-staged job, warm, without proxy tokens. |

The probes are standalone by design (own image, no `import modal_app`) and cost a few
minutes of GPU each. Re-run them rather than reasoning about SDK behavior from documentation
— every finding in section 6 came from a probe contradicting what the docs implied.

## Modal conventions used here

- **Use `.aio()` inside async functions.** `jobs_volume.commit()` in an async endpoint
  raises `AsyncUsageWarning`; use `await jobs_volume.commit.aio()`. Same for `.reload()`,
  `.spawn()`, and `FunctionCall.get()`.
- **Pin every dependency** in the image definition. `requirements.txt` / `pyproject.toml`
  cover only the *local client* — container deps live in `modal_app.py`'s image objects and
  the two sets are intentionally separate.
- **Never let a fallback be silent.** The `libx264` fallback hid a real defect until the
  logs were read. When degrading, `print` a warning that names the cause and the fix.
- Consult the official Modal docs (or a locally vendored copy of the guide/api/examples)
  before guessing at SDK behavior; `modal changelog` covers recent changes.

## 6. Video I/O: what the throughput is actually bounded by

**Where this ended up — read this first.** The video path is fully GPU-resident: in-process
NVDEC → VSR → NVENC from CUDA memory, no rawvideo pipes, no host round-trip. **≈59 fps**
warm on the reference clip (1080p→4K, `HIGHBITRATE_ULTRA`), inference-bound at `infer 74%`.
The encoder quality point mirrors NVEncC `--qvbr 18 … --tune uhq --multipass 2pass-quarter
--aq --aq-strength 10 --bframes 5 --ref 5` at preset P4. Details and the traps are in
*Fully GPU-resident* below; the current settings live in the `NVENC_*` constants.

The subsections between here and there are **the historical record**, kept so nobody
re-runs these experiments. They describe the *piped* architecture, where contention between
processes dominated everything and the pipeline sat at 12 fps. Their conclusions about
encoder settings and preset choice are true of that architecture and mostly moot in the
current one — each is annotated where it was superseded.

### The piped era (historical) — measured 2026-08-16

Measured on `RTX-PRO-6000`, 1080p→4K, `HIGHBITRATE_ULTRA`, 62 s / 1552-frame
clip. Each stage runs about as fast as the others, so they only pay for themselves once if
they run at the same time:

| Change | fps |
|---|---|
| CPU decode (`cv2.VideoCapture`), serialized infer→encode | 9.7 |
| + NVDEC decode on a reader thread, writer thread for the encoder | 11.9 |
| + `cpu=12` (was 4) | 13.1 (12.5 on repeat) — superseded, see below |
| preset `p5` instead of `p6` | 12.1 — *slower*, i.e. within noise |

The takeaways, so nobody re-runs these experiments:

- **Decode is free now**: `decode-wait 0%`. Do not spend effort there.
- **The encoder preset is not the lever.** Standalone (nothing else on the GPU),
  `hevc_nvenc` at p6+uhq+fullres does 18.6 fps at 4K and p4 does 35 fps — but in situ p5
  measured *slower* than p6. What binds is the GPU being shared between VSR inference and
  NVENC, not encoder settings, so keep p6 and take the quality. **Superseded for the
  default:** the preset is now `p4` to match the reference NVEncC command — see the
  fully-GPU-resident section below. The finding still holds *for the piped path*, where
  preset choice buys nothing.
- **The bt709 color filter is free**: 18.6 fps with it vs 18.0 without. Never drop it for
  speed.
- Every job prints a `Video done: … fps (decode-wait …, infer …, encode-backpressure …)`
  line. Compare against the table above before believing a change helped.
- **Compare warm containers only.** On a 12 s clip the same config measures 8.0 fps cold
  and 12 fps warm — the effect load is paid once per container. Run the clip 3-4 times
  back to back in one process and read the later runs, or you will "measure" cold starts.

### GPU choice and `cpu=` — settled 2026-08-18, do not re-litigate

Measured on a 10 s reference clip (302 frames, 1080p→4K, `HIGHBITRATE_ULTRA`), warm:

| Config | warm fps | $/s (GPU+cpu+mem) | $/frame |
|---|---|---|---|
| RTX PRO 6000, `cpu=12` | 11.7 | 0.001053 | 0.0000900 |
| **RTX PRO 6000, `cpu=6`** (current) | **12.2** | **0.000974** | **0.0000798** |
| RTX PRO 6000, `cpu=4` | 12.3 | 0.000948 | 0.0000771 |
| L40S, `cpu=12` | 6.8 | 0.000753 | 0.0001107 |

- **The GPU question is closed: keep `RTX-PRO-6000`.** It is the only Modal GPU with
  9th-gen NVENC, and H100/H200/A100/B200/B300 have **zero** NVENC engines — paying more
  buys less. L40S is the only credible alternative and it loses on *both* axes: 56% of the
  throughput for 72% of the cost. It needed ≥71.5% of RTX throughput just to break even.
  L40S is not broken (NGX loads, `hevc_nvenc` selected, `nvdec=yes`); it is inference-bound
  at `infer 75%`. Do not go GPU shopping again without re-deriving that break-even.
- **`cpu=12` was over-provisioned and measured *slower* than `cpu=6`.** The old note that
  4 cores capped 4K throughput predates NVDEC decode; at 4K, `cpu=4` and `cpu=6` both hold
  `decode-wait 1%`. `cpu=6` is the default, keeping margin for larger outputs where swscale
  work scales with pixel count. Re-measure before going below 6.

### Where the 12 fps actually goes (historical, piped path) — probed 2026-08-18

"Consumer RTX VSR runs realtime on cheap GPUs" is true and is **not** in tension with our
12 fps. Each stage was measured in isolation on `RTX-PRO-6000`, 1080p→4K:

| Stage, measured alone | fps |
|---|---|
| VSR `run()`, GPU-resident, HIGHBITRATE_ULTRA | **310** |
| + mandatory `.clone()` + integrity check | 262 |
| + CPU→GPU upload and GPU→CPU download | **103** |
| bt709 color filter (vs no filter) | 214 vs 215 — **free** |
| Python→pipe→ffmpeg, no encode | 159 (≈4 GB/s) |
| `hevc_nvenc` with production args, alone | **19.5** |
| NVDEC decode | free (`decode-wait` 1%) |
| **Whole pipeline** | **12.1** |

- **Inference is not the bottleneck and never was.** VSR alone is ~26x realtime for 25 fps
  content. That is exactly why the consumer driver overlay is realtime: it does
  decode→VSR→*screen*, fully GPU-resident, with no encode and no host round-trips. We
  additionally write an archival 4K 10-bit HEVC file, which alone is a ~19.5 fps job.
- **Neither is the encoder's compute, which is the counter-intuitive part.** Standalone the
  NVENC preset is a huge lever (p6 21 fps, p5 31, p4 38.5, p4+qres 43 — `-multipass`, AQ
  and `-rc-lookahead` are all minor by comparison). **In situ every one of them measures
  12.0-12.6 fps — no change at all.** p4+`-multipass qres` is 2.05x faster standalone and
  gains nothing end-to-end. So do not "optimize" the encoder settings for speed; you would
  trade away quality for zero throughput. This confirms the older p5-vs-p6 note. All of
  this is about the **piped** path; once the frames stop crossing process boundaries the
  contention disappears and the picture changes completely (see below).
- **What binds is contention, not any single stage.** In-pipeline the inference stage runs
  at ~21 fps against 103 fps standalone — a 5x inflation from sharing the box with two
  ffmpeg processes and ~31 MB of raw frames per frame (6.2 MB in + 24.9 MB out) crossing
  process boundaries. Every stage is fast alone; together they are ~4x slower.
- **Therefore the only real lever left is architectural**: keep frames on the GPU end to
  end (NVDEC→CUDA→VSR→NVENC, no host round-trip, no rawvideo pipes). Ceiling for that is
  the NVENC ASIC, so expect ~19 fps at current settings — worth roughly 1.6x, not 10x. Do
  not attempt it expecting realtime.
- Quality rungs *do* change inference cost a lot (SDK-only: HIGHBITRATE_ULTRA 310 fps,
  ULTRA 431, HIGHBITRATE_LOW/LOW ~1235), but since inference is not binding, dropping
  quality buys nothing end-to-end either.

Every `MODAL_NVENC_*` knob (`PRESET`, `MULTIPASS`, `CQ`, `AQ_STRENGTH`, `BFRAMES`, `REFS`,
`SFE`) plus `MODAL_GPU_ENCODER` / `MODAL_GPU_DECODER` exists for re-testing this. **They
must be baked into the image with `.env()`** — they are read again at container import,
where the deploying shell's environment does not exist, so a bare
`MODAL_NVENC_PRESET=p4 modal deploy` silently measures the default. Anything added to the
constants must also be added to the `.env({...})` block, or it will appear to do nothing.
Verify any such knob took effect by deploying a deliberately invalid value once and
confirming it fails loudly — and note the worker prints the resolved encoder settings at
startup (`Selected video encoder: … preset=p4, cq=18, …`), which is the cheapest check.

### Step 1 — feed NVENC from CUDA memory (`MODAL_GPU_ENCODER`), 2.8x, measured 2026-08-18

A consumer RTX 4070 running NVEncC does this same job (1080p→4K, NGX VSR, HEVC 10-bit,
uhq) at ~50 fps. Its log shows why: `Input Buffers CUDA` — decode, VSR and NVENC all live
in GPU memory, and the only pipe carries *1080p* frames. Ours moved every **4K** frame
GPU→host→pipe→ffmpeg→swscale→NVENC, ~31 MB per frame across three processes.

**`NVEncC cannot be adopted on Linux` — but NVIDIA super-resolution itself runs fine
there, so keep the two claims separate.** Re-verified 2026-08-18 against live `master`:
`NVEncCore/rgy_version.h`'s non-WIN32 branch hard-defines `ENABLE_NVVFX 0` and
`ENABLE_NVSDKNGX 0` with no `#ifndef` and no configure override, the official option docs
say `nvvfx-superres` is "supported on x64 [Windows] version only", and the 7.15 release
notes say "Windows x64 only". So `--vpp-resize algo=ngx-vsr` / `algo=nvvfx-superres` on
Linux NVEncC is a dead end regardless of which SDKs are installed — do not retry it.

What *does* work on Linux: the NGX RTX VSR effect we use via `nvidia-vfx` (the upstream
node this service derives from documents "Windows or Linux", and this service is the
proof), and separately NVIDIA's Maxine VFX SDK, which ships a native Linux package on NGC
(`VFXSDK_linux_<ver>.tgz`, `nvvfxvideosuperres`, compute capability up to 12.0). Linux is a
first-class target there — `install_feature.sh -f nvvfxvideosuperres`, supported GPUs
include L4, VSR needs driver 570.190+/580.82+/590.44+ — and official Python bindings exist
(`nvidia-vfx`, samples at NVIDIA-Maxine/nvidia-vfx-python-samples).

**Corrected 2026-08-18: Maxine is very likely the *same* model, not a different family.**
An earlier note here claimed the `HIGHBITRATE_*` tuning would not carry over. The current
Maxine VSR filter docs list modes 0-19 — `VSR_Bicubic/Low..Ultra`, `Denoise_*`, `Deblur_*`,
`HighBitrate_Low..Ultra` — i.e. exactly the family this pipeline tunes against, over the
same BGRA/RGBA GPU-buffer interface. So it reads as the same model delivered through the
SDK instead of driver-side NGX. Nothing to do today (the NGX path works at ~59 fps), but it
is the natural fallback if the `libnvidia-ngx.so.1` install in hard constraint #1 ever
breaks, and it would remove the `DRIVER_VERSION` coupling. Pin the driver/SDK pair if
adopted: r570-era `libnvVFXVideoSuperRes.so` is documented breaking against r595 NGX.

So `_upscale_video_gpu()` replicates the architecture instead, using PyNvVideoCodec to
encode straight from a CUDA tensor. Measured on the reference clip, warm:

| Path | fps | frames out |
|---|---|---|
| piped rawvideo → ffmpeg (the default at the time) | 12.1 | 296 of 301 |
| **`MODAL_GPU_ENCODER=1`** | **34.2** | **302 of 302** |

Encoder throughput alone, 4K, GPU-fed vs through the pipe: p6+fullres **35 vs 19.5 fps**,
p4+quarter **119 vs 43**. Verified on the real output: `Main 10` / `yuv420p10le`,
3840x2160, bt709 on all three colour fields, AAC audio intact.

It also **fixes the long-standing tail frame loss**. Cause found: a raw HEVC elementary
stream carries no timestamps, so B-frame reordering makes ffmpeg's demuxer drop the last
frames (`bf=4` → 98 of 100; `bf=0` → 100 of 100). `FFmpegMuxer` + `SetUniformPtsIncrement`
rebuilds display-order PTS and keeps every frame.

Traps worth knowing before touching this path:

- PyNvVideoCodec calls `__dlpack__(stream)` positionally; torch 2.13 wants it keyword.
  Patch `torch.Tensor.__dlpack__`; do **not** wrap the tensor in a shim object, or the
  encoder decides it has a CPU buffer and fails with "incorrect usage of CPU input buffer".
- torch has no uint16 left-shift on CUDA, so P010 packing scales by 64 in int32 and casts
  last (`_rgb_to_p010`). The colour math is BT.709 limited-range and was verified by a
  decode round-trip: max channel error 3/255.
- The muxer needs `encoder.GetSequenceParams()` as extradata or the mp4 has no usable
  sample entry.

### Step 2 — fully GPU-resident (`MODAL_GPU_DECODER`) and the settled quality point, 2026-08-18

**This is the current architecture.**

Both knobs now **default to 1**. Step 1 shipped opt-in because three things were unresolved
— the output bitrate was ~2x the piped path, `uhq` looked unsupported, and no quality
comparison was possible while frame counts disagreed. All three are closed below, and the
last host round-trip (the decode pipe) is gone: NVDEC decodes in-process, so a frame is created in
CUDA memory and never leaves it until NVENC emits a packet. Warm on the reference clip:

| Path | fps | frames |
|---|---|---|
| piped decode + piped encode (original) | 12.1 | 296 of 301 |
| piped decode + GPU encode | 34.2 | 302 (over-counted) |
| **in-process NVDEC + GPU encode (current)** | **57–60** | **301 of 301** |

301 is the correct count — `ffprobe -count_frames` on the source says 301, so the piped
path's "302" was over-counting by one.

**Two silently-wrong kwarg spellings caused blockers 1 and 2.** PyNvVideoCodec's option
parser drops unknown keys into a `map<string,string>` with **no validation**, so a typo
costs quality and raises nothing:

- **`qp` is not a key at all.** The old `rc="vbr", qp="19"` therefore set no quality target
  and ran plain VBR at the default bitrate — that, not a rate-control philosophy
  difference, was the 2x bitrate gap. The correct key is **`cq`** (NVENC `targetQuality`),
  and it is monotonic in the right direction (measured, real content, 1080p: cq=14 → 7.79,
  18 → 7.44, 24 → 4.48, 32 → 1.60 Mbps).
- **The uhq tuning value is `uhq`, not `ultra_high_quality`.** The long name falls through
  to UNDEFINED, which is exactly the error 8 that made uhq look unsupported here.

Output at the settled settings: 12.7 Mbps — *below* the piped path's 19.9, not double it.

The quality point mirrors NVEncC `--qvbr 18 --codec h265 --tune uhq --output-depth 10
--profile main10 --tier high --multipass 2pass-quarter --aq --aq-strength 10 --bframes 5
--ref 5` at preset P4. It lives in the `NVENC_*` module constants and **both** encode paths
read it, so the piped fallback stays directly comparable. `main10` needs no kwarg — the
profile autoselects from the P010 input surface.

**Verifying an encoder kwarg requires one session per process.** With several
`CreateEncoder` calls in one process the measurements are worthless: the same settings
measured 17199 KiB and 7204 KiB on different runs, and every session after the first
returned a byte-identical size no matter what changed. `scripts/probe_gpu_encoder.py` runs
each variant in its own container (`single_use_containers`) for this reason. Also prefer
**real content** — on synthetic noise the `cq` response measured *inverted*.

**`ThreadedDecoder` is unusable in PyNvVideoCodec 2.2.0.** It drains after exactly one
frame in every configuration probed: NATIVE/RGB/RGBP, every buffer and batch size, mkv and
mp4. (An empty batch really does mean EOF — `SPSCBuffer::PopEntries` blocks until the batch
is available and only returns short once the producer signalled done — so this is the
decode thread stopping, not a race.) `SimpleDecoder` and the low-level
`CreateDemuxer` + `CreateDecoder` pair both deliver all 301 frames; `_upscale_video_gpu`
uses the low-level pair, which matches sequential streaming use.

Decoder traps:

- `outputColorType=OutputColorType.RGB` gives tightly packed **HWC uint8** with no pitch
  padding, and converts on GPU using the bitstream's own `matrix_coefficients` — do not
  hand-roll BT.709 math for it.
- **`.clone()` is mandatory**, same trap as nvvfx: the decoder recycles its surfaces on the
  next `Decode()`. Measured — an uncloned view's mean changed under it while a clone held.
- `torch.from_dlpack` works directly; the encoder-side positional-`__dlpack__` patch is not
  needed on the decode side.
- Gate it on the existing `_nvdec_can_decode()` probe so unsupported inputs fall back to
  the piped decoder instead of failing mid-stream.

**This path is inference-bound, and `decode-wait` lies when decode is inline.** Run inline,
in-process NVDEC reported `decode-wait 50%` despite being ~1675 fps standalone. Moving it
to a reader thread changed the attribution to `decode-wait 1%, infer 74%` and the
throughput **not at all** (5.1s either way) — the "wait" was overlapping GPU work, not a
stall. The thread is kept because the attribution is honest and it removes a real
serialization risk at larger output sizes, but do not expect fps from it. The binding stage
is now VSR inference; the encoder sits at 12%.

**Decoupled encode threading — evaluated 2026-08-18, declined.** Video Codec SDK 13.1
introduced a decoupled-queue architecture so NVDEC/CUDA/NVENC work on *different* frames
concurrently, and the obvious suggestion is to apply it here: decode is already on a reader
thread, but `_flush()` still encodes inline right after inference. The arithmetic kills it —
encode is **12%** of wall time, so perfect overlap is worth at most ~1.14x (~59 → ~66 fps).
Against that, a writer thread reintroduces packet-ordering and DLPack-buffer-lifetime
hazards on the encoder side, and the reader-thread experiment directly above showed that
this kind of "wait" is often overlapping GPU work that threading does not recover at all.
Do not spend effort here; the lever is inference, and inference is the SDK's.

### Input concurrency (`MODAL_WORKER_CONCURRENCY`) is NOT safe yet

The card is idle a third of the time (`encode-backpressure ~33%`), so packing two jobs per
container looks attractive. It does not work today, and both blockers are measured:

1. **Effect instances cannot be shared across threads.** `Run()` is not reentrant and the
   effect reuses one internal DLPack output buffer. Fixed: `_sr_cache` is now keyed by
   `(thread, role)`, so concurrent inputs get their own instances.
2. **Unfixed, and the real blocker — `jobs_volume.reload()` destroys in-flight jobs.** A
   reload re-materializes the whole mount, and every job's scratch (input, `ffmpeg.log`,
   partial output) lives on that Volume uncommitted. A second job arriving mid-flight
   either cannot see its own input or wipes the running job's working files. Observed as
   `FileNotFoundError: /jobs/<id>/ffmpeg.log`.

`run()` now only reloads when it is the sole in-flight job, which makes the default
(`concurrency=1`) byte-identical to the old behavior. **Before raising concurrency above 1,
job scratch space must move off the Volume to container-local disk**, with only the
finished output copied back. Until then the knob is for experiments only.

`MODAL_APP_NAME`, `MODAL_GPU`, `MODAL_WORKER_CPU`, `MODAL_WORKER_CONCURRENCY` and
`MODAL_WORKER_MAX_CONTAINERS` exist so a variant deploys beside production instead of
replacing it, e.g.
`MODAL_APP_NAME=rtx-bench-x MODAL_WORKER_CPU=4 modal deploy modal_app.py`. Stop bench apps
when done (`modal app stop <name> -y`). Note the autoscaler scales *out* rather than
packing a container, so measuring input concurrency requires `MODAL_WORKER_MAX_CONTAINERS=1`.

**Color tagging is not optional and needs `setparams`.** Raw RGB from the pipe carries no
color metadata, so swscale converts with its bt601 default while players read HD output as
bt709 — a visible hue shift. The `-colorspace/-color_primaries/-color_trc` output options
alone left primaries and transfer `unknown` in the encoded stream (NVENC writes its VUI
from the frame properties), so the `-vf` chain must carry `setparams` as well.

## Verification protocol

**Static checks are not evidence that this works.** Importing the app validates decorator
arguments and nothing else — the app imported cleanly for the entire period when VSR was
100% broken. A change is verified only when:

1. `uvx ruff check --select F,E9 modal_app.py modal_client.py` passes.
2. `modal deploy modal_app.py` succeeds.
3. An **image** round-trip returns correctly-scaled output.
4. A **video** round-trip returns correct dimensions, frame count (301 for the reference
   clip — verify against `ffprobe -count_frames` on the source, not against memory), and intact audio
   (`ffprobe`), and the worker log says `Selected video encoder: hevc_nvenc` — not `libx264`.
5. `ffprobe` on that output reports `Main 10` / `yuv420p10le` and bt709 for **all three** of
   `color_space`, `color_primaries`, `color_transfer` — `unknown` on any of them is a bug.
6. **Pixels are actually inspected.** Steps 4 and 5 all passed while the GPU encoder was
   emitting visibly corrupt video — see the warning below. Two cheap checks catch it:
   - **No flat frames.** Decode every frame small (`-vf scale=320:180 -f rawvideo
     -pix_fmt gray`) and count frames whose per-frame standard deviation is > 1. Anything
     below the full frame count means frames encoded as flat grey.
   - **No striping.** Mean absolute Laplacian of a 512px crop from a mid-clip frame,
     extracted with `-pix_fmt rgb24` (without it, ffmpeg writes 16-bit PNGs from the
     10-bit stream). It should land near the source's own value (~1-3 on the reference
     clip). ~190 means alternating-column striping, i.e. the P010 surface walked as 8-bit.
7. The `Video done: …` line reports `nvdec=in-process`, `gpu-encoder=yes`, and fps in line
   with the table in section 6 (~56-60 fps warm on the reference clip). `nvdec=yes` means
   the in-process decoder was skipped; `gpu-encoder` absent means the piped encoder ran.
   Either is a silent ~6x downgrade worth explaining before shipping.
8. `modal app logs rtx-media-upscaler` shows no warnings or tracebacks.
9. `/health` returns 401 without proxy-auth headers.

### NVENC does not observe CUDA stream ordering — fixed 2026-08-19

The GPU encode path spent a day disabled because it emitted alternating-column striping on
every frame (Laplacian ~192 over a 512px crop against ~1.1 piped and ~2.0 for the source;
raw pixels alternating 177, 59, 177, 61, …), plus flat grey frames whenever the inference
batch exceeded 1. **Both were one bug**, and it was not in any of the places it looked
like: `_flush` called `encoder.Encode(_rgb_to_p010(frame_rgb))` with no synchronisation
between the conversion kernels and the encoder.

NVENC reads the input surface from the encoder ASIC, which takes no part in CUDA stream
ordering, so it can start reading before the kernels filling that surface have run. The
striping is the tell: the freed uint16 P010 block gets recycled for the next frame's int32
`packed` tensor, and int32 data read as uint16 is `[value, 0, value, 0, …]`. The piped
path never showed it only because its `.cpu()` download synchronises the device every
frame. Fix is one `torch.cuda.synchronize()` before `Encode()`.

Things that were *not* the cause, each disproved by probe, so nobody re-runs them: the
`__dlpack__` shim, the flat-vs-per-plane surface handoff, the non-contiguous CHW→HWC
tensor production feeds in, batching itself, and holding the P010 buffer alive (rings of
8/48/64 all still corrupt — the race is against the kernels filling the buffer, not
against the allocator).

**`scripts/probe_p010_encode.py` is the tool for this class of bug.** It encodes a smooth
synthetic clip, muxes it exactly like production, decodes it back and reports the striping
Laplacian and a stale-frame count, one encoder session per container. Variants I (freed
temporary, production's old lifetime) and N (same plus the sync) are the before/after:
22 corrupt frames of 30 versus 0. `scripts/probe_gpu_encoder.py` cannot see any of this —
it only watches the encoded size.

The lasting lesson is verification, not encoders: for a full day this path produced
visibly broken 4K while passing frame count, `ffprobe`, a clean decode with zero errors,
and `_check_frame_integrity`. Only looking at pixels caught it — hence step 6 above.

For GPU-level questions (does this library load? does this encoder exist? which encoder
option costs the throughput?), write a small throwaway `modal run` script that probes the
real hardware and prints results. That loop is minutes, and it is the only way to answer
these questions — they cannot be reasoned out from documentation, as the `-12`, NVENC and
encoder-preset investigations all showed. Keep such a probe **standalone** (its own small
image, no `import modal_app`): a probe that imports the app module needs
`add_local_python_source` and fails confusingly without it.

A round-trip can be run without proxy tokens by putting the input on the Volume
(`modal volume put rtx-upscaler-jobs <file> /<job-id>/input.mkv`) and calling
`modal.Cls.from_name("rtx-media-upscaler", "UpscaleWorker")().run.remote(job_id=…)`
directly — this exercises the worker without going through the web tier.

## Reusing this as a template for other Modal jobs

Keep: the web/GPU split, the job-queue endpoints, the Volume-backed file handoff, proxy
auth, `@modal.enter()` warming, the client's submit-and-poll loop, the verification
protocol above, and — for any job that chains GPU stages — the GPU-resident data path and
the standalone-probe habit that produced it.

Replace: `UpscaleWorker`'s `_upscale_*` methods and the `nvvfx` specifics. Drop the
`DRIVER_VERSION` / NGX install unless the new workload also uses NGX (super-resolution,
DLSS-family effects); most Modal GPU jobs do not need it. Keep the ffmpeg pin only if the
job touches video, and re-verify NVENC against the host driver of the day.

Rename `app = modal.App("rtx-media-upscaler")` and the `rtx-upscaler-jobs` Volume, or the
new service will collide with this one in the same workspace.
