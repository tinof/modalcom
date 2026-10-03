# CLAUDE.md — RTX Media Upscaler on Modal

Conventions, constraints, and the traps that have already cost real debugging time. Setup
and API docs are in [README.md](README.md). The measured history behind the performance
conclusions here is in [docs/PERF-HISTORY.md](docs/PERF-HISTORY.md) — read it before
re-running any benchmark.

## What this project is

A Modal service (`modal_app.py`, `modal_client.py`): NVIDIA RTX Video Super Resolution
exposed as an authenticated HTTP job queue.

It began as a port of the Comfy-Org ComfyUI custom node ([NOTICE](NOTICE) has the
Apache-2.0 attribution) and keeps that node's resize-type handling, quality mapping,
dimension alignment (multiples of 8), and `MAX_PIXELS = 1024 * 1024 * 16` batching
heuristic. Those behaviours are now this service's **contract**, not a parity target.
Changing one is a breaking API change — say so explicitly rather than "improving" it in
passing.

**This project is also the template for other Modal jobs.** The next section marks which
parts generalize.

## Architecture patterns worth copying

Each one fixes a concrete problem.

- **Split the web tier from the GPU tier.** `api` is a CPU-only ASGI function on a slim
  image; `UpscaleWorker` is an `@app.cls()` that owns the GPU. Attach the GPU to the web
  function and you pay GPU rates for uploads, downloads, and idle time.
- **Job queue, not synchronous HTTP.** Modal web endpoints enforce a **150-second**
  request timeout. Anything slower must `.spawn()` and return a `call_id`, with a separate
  `GET /result/{call_id}` that returns 202 while pending. `timeout=3600` on the function
  does not extend the HTTP window.
- **Move payloads through a `modal.Volume`, not through function arguments.** The web tier
  does **not** mount the Volume: it uploads with `batch_upload.aio()` from Starlette's
  on-disk spool, serves results with `read_file.aio()`, and deletes with
  `remove_file.aio()`. A mount under `@modal.concurrent` shares one view between inputs,
  so a `/result` `reload()` fails ("volume busy") or drops another input's uncommitted
  upload. Only the GPU worker mounts it, and only the worker calls `reload()`/`commit()`.
- **Warm expensive state in `@modal.enter()`.** CUDA init, model load, and encoder probing
  belong there — once per container, not once per request.
- **`requires_proxy_auth=True` on any endpoint that spends GPU money.** Modal web
  functions are public by default.
- **Let Modal handle concurrency.** Without `@modal.concurrent`, Modal sends one input per
  container and scales horizontally. Do not add locks to serialize work. The CPU-only web
  tier uses `@modal.concurrent` because it is I/O-bound; the GPU worker deliberately does
  not.
- **Keep GPU data on the GPU.** The single largest win here (12 → 59 fps) was not a faster
  stage. It was deleting the host round-trips between stages. If frames, tensors, or
  images cross a process boundary between GPU stages, that copy is very likely your
  ceiling.

## Hard constraints — do not change casually

Each was discovered the expensive way. All are pinned in `modal_app.py`.

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
upgrade, check this first. The failure is *not* GPU-model-specific — do not go GPU
shopping when you see `-12`.

### 2. ffmpeg's NVENC support is gated on the driver (`FFMPEG_URL`)

Newer is **not** better. Each ffmpeg build links a Video Codec SDK with a minimum driver
version:

| Build | SDK | Requires driver | Works on Modal (580.95.05) |
|---|---|---|---|
| BtbN `master-latest` | 13.1.15 | 610.0+ | **No** |
| BtbN `n8.1` (pinned) | — | 570.0+ | Yes, incl. `av1_nvenc` |
| Debian apt ffmpeg 5.1 | — | — | Yes, no `av1_nvenc` |

Pointing `FFMPEG_URL` at `master-latest` makes every NVENC encoder fail to initialize and
the worker falls back to CPU `libx264` — correct output, roughly an order of magnitude
slower. Pin versioned URLs, never `latest`/`master` — and note that BtbN's
`latest` *tag* is rebuilt daily even for the `n8.1-latest` asset. `FFMPEG_URL` points at a
**month-end** `autobuild-YYYY-MM-DD-*` release (BtbN prunes the daily tags but keeps
month-end ones) and `FFMPEG_SHA256` makes the image build fail on any artifact change. To
move: pick a newer month-end tag, take the asset digest from
`gh api repos/BtbN/FFmpeg-Builds/releases/tags/<tag>`, and re-run the verification
protocol.

### 3. GPU choice and torch version are coupled

Default is `RTX-PRO-6000` (Blackwell): verified working, and the only Modal GPU with
9th-gen NVENC. B200/B300 and A100 have **no NVENC engine at all**. Blackwell is `sm_120`,
so `torch==2.13.0` is required — torch 2.6 fails with "no kernel image is available for
execution on the device". Upgrading the GPU forces upgrading torch.

### 4. VSR 15360px limit, DLPack buffer clone, and output integrity

- **SDK defect at 16K output**, probed on Linux + Blackwell RTX-PRO-6000. At
  **15360x15360** input means `[0.5, 0.5, 0.5]` give output means
  `[0.4998, 0.5007, 0.5012]` (clean). At **16384x16384** the same input gives
  `[0.4998, 0.5007, 0.0]` — the blue channel collapses to 0.
- **Hybrid resize path.** When requested output falls between 15360 and 16384, the worker
  runs VSR capped at 15360 and resamples to target with GPU
  `torch.nn.functional.interpolate(mode="bicubic")` in float, before uint8 conversion.
- **DLPack explicit clone.** `nvvfx` reuses its internal DLPack buffer across calls. We
  MUST `.clone()` the PyTorch tensor immediately after `torch.from_dlpack()`.
- **Integrity check.** `_check_frame_integrity` verifies every frame on GPU for
  `torch.isfinite` and channel collapse, so corrupted SDK output fails loudly.

**Do NOT remove** the hybrid path, the `.clone()`, or the integrity check as "dead code"
or "premature optimization".

### 5. Quality modes: four families, two roles

`nvvfx.VideoSuperRes.QualityLevel` has 19 members in four families, **all probed working**
on `RTX-PRO-6000` / driver 580.95.05:

- `BICUBIC, LOW..ULTRA` (0-4) and `HIGHBITRATE_LOW..ULTRA` (16-19) **change resolution** —
  valid for `quality`. The standard family also suppresses compression artifacts, so it
  softens clean sources; `HIGHBITRATE_*` skips that for ProRes-grade input.
- `DENOISE_LOW..ULTRA` (8-11) and `DEBLUR_LOW..ULTRA` (12-15) are **same-resolution**
  restoration — valid only for `preprocess`, which runs a second pass at input size before
  upscaling.

`UPSCALE_QUALITIES` / `SAME_RES_QUALITIES` encode this split and the web tier rejects a
mismatched role with 400 before spending GPU time. `_normalize_quality()` validates
against those frozensets rather than importing `nvvfx`, because the web tier runs on the
slim CPU image where the SDK is not installed.

The preprocess pass gets its own cached effect instance (`role="preprocess"` in
`_super_res`), so the two models coexist without evicting each other. Its output needs the
same mandatory `.clone()` — same reused-buffer trap as section 4.

**The default is `HIGHBITRATE_ULTRA` (19), not `ULTRA` (4)** — set in the `quality` Form
default. "High bitrate" means *few compression artifacts*, not *high resolution*.
Broadcast archive material is typically soft but cleanly encoded, so the standard family's
artifact suppression has nothing real to remove and instead damages texture, producing a
blotchy painterly look. Measured on the reference clip, mode 19 beat mode 4 on both axes:
~15% more edge detail and ~8x less flat-region noise. Prefer the standard family only for
genuinely blocky, heavily compressed encodes.

**Judging output quality needs a real source check first.** A "1080p" broadcast file is
often an SD master someone else already upscaled. Downscale a frame to 540p and back: if
that reconstructs the file almost exactly (mean abs diff < 1/255), there is no real detail
for VSR to recover and the result will look near-identical to Lanczos at any setting. That
is the source's ceiling, not a pipeline defect.

**Two counter-intuitive facts about DEBLUR, both measured, both contradicting the names:**

1. **Intensity is not monotonic.** `DEBLUR_LOW`/`MEDIUM` yield *more* apparent detail than
   `HIGH`/`ULTRA`. The docs describe ULTRA as "maximum sharpening strength"; observed
   behaviour is the opposite, so the upper rungs evidently regularise more as well.
2. **Rank deblur by temporal flicker, not by still-frame metrics.** Flat-region noise that
   varies frame to frame is what looks bad in motion. `DEBLUR_ULTRA` adds ~50% detail for
   only +5% flicker; `DEBLUR_LOW` scores best on stills and flickers most (+22%).

A still-frame sharpness metric will actively mislead you here. When evaluating any
preprocess change, measure flicker across consecutive frames in a static patch.
`detail_vs_noise` in `scripts/preprocess_sweep.py` covers the spatial half; the temporal
half is a mean abs diff between consecutive frames in a static region.

## The scripts directory

| Script | Use it when |
|---|---|
| `compare_frames.py`, `preprocess_sweep.py` | Judging VSR quality / preprocess modes (above). Both take `--variants label=path` pairs. |
| `probe_gpu_encoder.py` | Re-testing which `CreateEncoder` kwargs are live. One encoder session per container. |
| `probe_gpu_pipeline.py` | Re-testing the NVDEC decoder and rate control on real content. |
| `probe_nvdec.py` | Re-testing decoder classes after a PyNvVideoCodec upgrade (is `ThreadedDecoder` fixed yet?). |
| `probe_nvdec_rgb.py` | Re-testing the NVDEC `OutputColorType.RGB` path and its surface layout. |
| `probe_p010_encode.py` | Any suspicion of corrupt GPU-encoder output — striping, stale or flat frames. |
| `verify_roundtrip.py` | Running the deployed worker on a Volume-staged job, warm, without proxy tokens. `MODAL_APP_NAME` selects a bench deploy. |
| `probe_nvdec_interlace.py` | Re-testing whether in-process NVDEC still deinterlaces interlace-flagged PsF input (row-parity diff against ffmpeg's woven decode). |
| `probe_stages.py` | Per-stage timing: VSR alone, encode alone per preset, serial vs overlapped, N concurrent NVENC sessions. One encoder per single-use container. |
| `probe_encoder_quality.py` | Encoder settings vs quality: same VSR frames, one encoder per container, against a lossless encode; score the outputs locally with VMAF 4K. |

The probes are standalone by design (own image, no `import modal_app`) and cost a few
minutes of GPU each. Re-run them rather than reasoning about SDK behavior from
documentation — every video-path finding came from a probe contradicting the docs.

## Modal conventions used here

- **Use `.aio()` inside async functions.** `jobs_volume.commit()` in an async endpoint
  raises `AsyncUsageWarning`; use `await jobs_volume.commit.aio()`. Same for `.reload()`,
  `.spawn()`, and `FunctionCall.get()`.
- **Pin every dependency** in the image definition. `requirements.txt` / `pyproject.toml`
  cover only the *local client*; container deps live in `modal_app.py`'s image objects.
  The two sets are intentionally separate.
- **Never let a fallback be silent.** The `libx264` fallback hid a real defect until
  someone read the logs. When degrading, `print` a warning naming the cause and the fix.
- Consult the official Modal docs before guessing at SDK behavior; `modal changelog`
  covers recent changes.

## The video path: fully GPU-resident

NVDEC → VSR → NVENC from CUDA memory. No host round-trip for the 4K output frames.
`MODAL_GPU_ENCODER` and `MODAL_GPU_DECODER` both default to 1.

**The encoder is the binding stage, not inference** (probed 2026-10-03, RTX PRO 6000
Blackwell Server Edition): VSR `run()` is 3.5 ms/frame, VSR + integrity + P010 4.4 ms, and
one NVENC session at the default settings encodes 4K at 38.5 fps on its own. The whole pipeline runs at
the encoder-only rate; overlapping VSR with encode on another stream measured no faster.
**≈36 fps** warm end to end (1080p→4K, `HIGHBITRATE_ULTRA`, defaults), byte-identical
output across runs.

The encoder quality point (since 2026-10-03, chosen by VMAF against a lossless encode of
the same VSR frames) is NVEncC-equivalent `--qvbr 20 --preset p5 --tune uhq --tier high
--level 5.1 --max-bitrate 100000 --multipass 2pass-full --aq --aq-strength 10
--aq-temporal --bframes 5 --lookahead 32`, Main 10. It lives in the `NVENC_*` module
constants, which **both** encode paths read, so the piped fallback stays comparable. The
measurements behind each choice are in the `NVENC_*` comments in `modal_app.py` and in
[docs/PERF-HISTORY.md](docs/PERF-HISTORY.md). CQ 20 measured 78 Mbps on the 61 s Night
Manager Blu-ray sample.

### Traps in this path

- **`torch.cuda.synchronize()` before `Encode()` is mandatory.** NVENC reads the input
  surface from the encoder ASIC, which takes no part in CUDA stream ordering, so it can
  start reading before the kernels filling that surface have run. Removing the sync
  returns alternating-column striping and flat grey frames. Full post-mortem in
  [docs/PERF-HISTORY.md](docs/PERF-HISTORY.md).
- **PyNvVideoCodec silently drops unknown kwargs** into a `map<string,string>` with no
  validation, so a typo costs quality and raises nothing. The quality key is **`cq`**
  (NVENC `targetQuality`), not `qp`. Tuning values are **`uhq`** and **`high_quality`**
  (not `ultra_high_quality` / `hq`) — anything else falls through to UNDEFINED and surfaces
  as error 8. `temporalaq` enables on *any* non-empty string, even `"0"`; `""` leaves it
  unset. `numrefl0/1` force reference-list sizes (NVEncC `--multiref-l0/l1`), not
  NVEncC's `--ref` (DPB size); the GPU path leaves them to the driver.
- **`cq` silently caps the bitrate unless the ceiling is restored.** PyNvVideoCodec zeroes
  `maxBitRate` whenever `cq` is set (`NvEncoderClInterface.cpp`, after parsing
  `maxbitrate`, so passing it does not help). The driver then applies its own VBR ceiling
  (~20 Mbps at Level 5.0, ~32 at 5.1, ~48 at 6.0), and grainy 4K hits it: CQ 18 and CQ 24
  produced the same size. `_upscale_video_gpu` restores the ceiling with
  `GetEncodeReconfigureParams()` + `Reconfigure()` **before** `GetSequenceParams()` (the
  HRD values are in the VPS/SPS the muxer stores). Check it in the stream: trace_headers
  shows `bit_rate_value_minus1 = 1562499` (100 Mbps) and `cpb_size_value_minus1 = 6249999`
  (100 Mbit, a 1 s buffer).
- **Keep the VBV buffer at 1 s.** With 1.6x the ceiling (160 Mbit), an encoder created at a
  different resolution than the previous job in the same warm container failed
  `nvEncInitializeEncoder` with error 8 — 4K after 8K and 8K after 4K, reproducible —
  while each size passed on a fresh container. Bisected: removing the explicit level, the
  ceiling or P5+fullres each hid it; the 1 s buffer fixed it for 4K, 1620p and 8K in both
  orders. Any change to the level, ceiling or buffer must re-run a mixed-size sequence in
  one container (e.g. 2x, 4x, 2x, 1.5x, 2x), not just one job.
- **Explicit level + P5+fullres stays fragile after a size change; the code falls back.**
  The first production deploy of these defaults failed *every* video job after an image
  job, and later 1620p→4K failed intermittently, all with error 8 at `CreateEncoder`. What
  the failures have in common: the previous job left a VSR effect (or encoder state) at a
  different output size. Bisected: dropping the explicit level made every sequence pass;
  P4+qres also passed. Two mitigations in `_upscale_video_gpu`: the job's VSR effect is
  loaded *before* `CreateEncoder` (fixes image→video), and an error-8 rejection of the
  explicit level retries once without it, with a WARNING (`rejected explicit HEVC level`).
  In 16 mixed jobs the fallback fired 4 times; those 4K outputs carried level 5.0 or 6.0
  instead of 5.1, with the same 100 Mbps ceiling and file size. **Test image→video and
  size switches before deploying any encoder change** — the bench runs that preceded the
  failed deploy were all video-first.
- **Always set the HEVC level.** With `level` unset the driver chose 5.0 or 6.0 per
  session for identical input, each with a different default ceiling — the "same job, 2.4x
  bigger file" bug. `MODAL_NVENC_LEVEL` defaults to 5.1 and applies to outputs up to
  4096x2176; above that `_hevc_level()` leaves the level to the driver (it picked 6.0 or
  6.2 at 8K; with the ceiling restored the size stays identical).
- **Patch `torch.Tensor.__dlpack__`, do not wrap the tensor in a shim.** PyNvVideoCodec
  calls `__dlpack__(stream)` positionally and torch 2.13 wants it keyword. A shim object
  makes the encoder decide it has a CPU buffer ("incorrect usage of CPU input buffer").
- **`MAX_BATCH = 1`.** batch=1 measured 56.0 fps against batch=2's 53.2 — a larger batch
  makes the encoder wait on a longer run of inference instead of overlapping with it. The
  constant also pins the batch for a correctness reason (the cross-stream DLPack race), so
  do not raise it for speed or memory without reading the post-mortem.
- **`ThreadedDecoder` is unusable in PyNvVideoCodec 2.2.0** — it drains after exactly one
  frame in every configuration probed. 2.2.3 (pinned since 2026-10-04) did not touch it:
  its source diff against 2.2.0 is the Windows ARM64 port plus the bundled libavformat
  (FFmpeg 8.1.2 → 9.0.1), nothing in the encoder, decoder or muxer code. Use the
  low-level `CreateDemuxer` + `CreateDecoder` pair, which `_upscale_video_gpu` does.
- **`.clone()` the decoder output too.** The decoder recycles its surfaces on the next
  `Decode()`; an uncloned view's mean was measured changing under it.
- **`outputColorType=OutputColorType.RGB`** gives tightly packed **HWC uint8** with no
  pitch padding, and converts on GPU using the bitstream's own `matrix_coefficients`. Do
  not hand-roll BT.709 math for it. `torch.from_dlpack` works directly here; the
  positional-`__dlpack__` patch is encoder-side only.
- Gate the in-process decoder on the existing `_nvdec_can_decode()` probe so unsupported
  inputs fall back to the piped decoder instead of failing mid-stream.
- **Interlace-flagged input must not use the in-process decoder.** PyNvVideoCodec's
  `NvDecoder` picks `cudaVideoDeinterlaceMode_Adaptive` for any non-progressive sequence,
  with no Python kwarg to change it. On a BBC Blu-ray (PsF: 25p pictures, MBAFF-coded,
  `field_order=tt`) it rewrote the odd rows — 7.8% of odd-row pixels changed by >6 levels
  — and removed 32% of vertical detail before VSR. `_upscale_video_gpu` routes
  `field_order` tt/bb/tb/bt to the piped ffmpeg NVDEC decoder, which weaves.
  Re-test with `scripts/probe_nvdec_interlace.py`. Truly interlaced 50i/60i
  content would need a real deinterlacer (bwdif) ahead of VSR; not handled.
- **The piped decoder's swscale flags are load-bearing.** `_ffmpeg_decode_command` passes
  `in_color_matrix` (BT.709 for untagged HD, like NVDEC's own rule), `in_range`, and
  `flags=bicubic+accurate_rnd+full_chroma_int`: the default lookup-table conversion sits
  ~1.6 luma levels dark (Codex reproduced it byte for byte; a grey ramp shows it too).
  It also needs `-fps_mode passthrough`, or rawvideo output duplicates frames to fill
  timestamp gaps (306 frames from a 304-frame cut).
- **In-process NVDEC's RGB path truncates instead of rounding** (`ColorSpace.cu`
  `YuvToRgbForPixel`), about -0.43 luma levels. Known and unfixed (needs a rebuilt wheel).
- **torch has no uint16 left-shift on CUDA**, so P010 packing scales by 64 in int32 and
  casts last (`_rgb_to_p010`). The colour math is BT.709 limited-range, verified by decode
  round-trip at max channel error 3/255.
- **The muxer needs `encoder.GetSequenceParams()` as extradata**, or the mp4 has no usable
  sample entry. `FFmpegMuxer` + `SetUniformPtsIncrement` also rebuilds display-order PTS,
  which is what fixed the old tail-frame loss (a raw HEVC elementary stream carries no
  timestamps, so B-frame reordering made ffmpeg's demuxer drop the last frames).
- **Color tagging needs `setparams` in the `-vf` chain.** Raw RGB carries no color
  metadata, so swscale uses its bt601 default while players read HD output as bt709 — a
  visible hue shift. The `-colorspace`/`-color_primaries`/`-color_trc` output options
  alone leave primaries and transfer `unknown` in the stream.
- **Verifying an encoder kwarg requires one encoder session per process.** With several
  `CreateEncoder` calls in one process the measurements are worthless — every session
  after the first returned a byte-identical size no matter what changed. Prefer **real
  content**: on synthetic noise the `cq` response measured *inverted*.

### Settled — do not re-derive without reading the history first

All of these are measured; details in [docs/PERF-HISTORY.md](docs/PERF-HISTORY.md).

- **The GPU question is closed: keep `RTX-PRO-6000`.** It is the only Modal GPU with
  9th-gen NVENC, and H100/H200/A100/B200/B300 have **zero** NVENC engines. L40S is the
  only credible alternative and loses on both axes (56% of throughput for 72% of cost).
- **`cpu=6` is the default**, measured faster than `cpu=12`. Re-measure before going below
  6.
- **Superseded 2026-10-03: "the lever is inference".** Stage isolation on the current
  path shows VSR at 3.5 ms/frame and one NVENC session as the bound (encode-only: P4+qres
  65 fps, P5+fullres+UHQ 38.5, P6 27.5; end to end at the defaults ~36). The old `infer 74%` attribution included time blocked on the
  encoder. Encoder preset *is* the throughput lever now, and so is the number of NVENC
  sessions: the card has **4 NVENC engines** and independent sessions scale (4x P5+HQ
  210 fps combined, 4x P6+UHQ 66). Chunked parallel encoding is the next speed lever and
  is not implemented. PyNvVideoCodec 2.2.0 has no `splitEncodeMode` option. Overlapping
  VSR with a single encoder buys nothing. Batching still does not pay.
- **The RTX 4070 comparison is settled.** StaxRip/NVEncC on a 4070 does 71.7 fps with
  P5 + **HQ**; the same settings here measured 68.1 fps per session (encoder clock 1912 vs
  2093 MHz). Per session the cards are equal; the PRO 6000's advantage is four engines.
- **`decode-wait` lies when decode is inline.** Inline NVDEC reported `decode-wait 50%`
  while being ~1675 fps standalone. Moving it to a reader thread changed the attribution
  to `decode-wait 1%, infer 74%` and the throughput not at all. The thread stays because
  the attribution is honest.
- **NVEncC cannot be adopted on Linux for VSR** — `ENABLE_NVVFX`/`ENABLE_NVSDKNGX` are
  hard-defined to 0 in the non-WIN32 branch with no configure override, and the docs say
  Windows x64 only. Do not retry `--vpp-resize algo=ngx-vsr`.
- **NVIDIA's Maxine VFX SDK is very likely the *same* model**, delivered through the SDK
  instead of driver-side NGX (its filter docs list the same modes 0-19 over the same
  BGRA/RGBA GPU-buffer interface). Nothing to do today, but it is the natural fallback if
  the `libnvidia-ngx.so.1` install in constraint 1 ever breaks, and it would remove the
  `DRIVER_VERSION` coupling. Pin the driver/SDK pair if adopted.

### Env knobs

`MODAL_NVENC_*` (`PRESET`, `MULTIPASS`, `TUNING`, `CQ`, `LEVEL`, `MAX_MBPS`,
`AQ_STRENGTH`, `BFRAMES`, `REFS`, `SFE`; `REFS` and `SFE` affect only the piped ffmpeg
encoder),
`MODAL_GPU_ENCODER`, `MODAL_GPU_DECODER`, `MODAL_APP_NAME`, `MODAL_GPU`,
`MODAL_WORKER_CPU`, `MODAL_WORKER_CONCURRENCY`, `MODAL_WORKER_MAX_CONTAINERS`.

**They must be baked into the image with `.env()`.** They are read again at container
import, where the deploying shell's environment does not exist, so a bare
`MODAL_NVENC_PRESET=p4 modal deploy` silently measures the default. Anything added to the
constants must also be added to the `.env({...})` block or it will appear to do nothing.
The worker prints the resolved settings at startup (`Selected video encoder: … preset=p5,
cq=20, multipass=fullres, level=5.1, maxrate=100M, …`) — that is the cheapest check.

Deploy variants beside production, never over it:
`MODAL_APP_NAME=rtx-bench-<x> MODAL_GPU=<gpu> modal deploy modal_app.py`, then
`modal app stop rtx-bench-<x> -y`. Set `MODAL_WORKER_MAX_CONTAINERS=1` when measuring —
the autoscaler scales *out* rather than packing a container, hiding the effect. Compare
**warm containers only**: the same config measures 8.0 fps cold and 12 fps warm on a 12 s
clip, so run the clip 3-4 times in one container and read the later runs. Compute **cost
per frame**, not fps.

### Input concurrency (`MODAL_WORKER_CONCURRENCY`) is NOT safe yet

The card is idle a third of the time (`encode-backpressure ~33%`), so packing two jobs per
container looks attractive. Two blockers, both measured:

1. **Fixed:** effect instances cannot be shared across threads — `Run()` is not reentrant
   and the effect reuses one internal DLPack output buffer. `_sr_cache` is now keyed by
   `(thread, role)`.
2. **Partly fixed, still the blocker:** `jobs_volume.reload()` destroys in-flight jobs. A
   reload re-materializes the whole mount, and every job's scratch (input, `ffmpeg.log`,
   partial output) lives on that Volume uncommitted. A second job arriving mid-flight
   either cannot see its own input or wipes the running job's working files. Observed as
   `FileNotFoundError: /jobs/<id>/ffmpeg.log`. Since 2026-10-03 the video
   intermediates and ffmpeg logs live in a per-job local `tempfile` dir (`scratch_dir`),
   but the input and the final output are still written on the mount, so the hazard
   remains for those.

`run()` only reloads when it is the sole in-flight job, so the default (`concurrency=1`)
is byte-identical to the old behavior. **Before raising concurrency above 1, job scratch
must move off the Volume to container-local disk**, with only the finished output copied
back. Until then the knob is for experiments only.

## Verification protocol

**Static checks are not evidence that this works.** Importing the app validates decorator
arguments and nothing else — the app imported cleanly for the entire period when VSR was
100% broken. A change is verified only when:

1. `uvx ruff check --select F,E9 modal_app.py modal_client.py` passes.
2. `modal deploy modal_app.py` succeeds.
3. An **image** round-trip returns correctly-scaled output.
4. A **video** round-trip returns correct dimensions, frame count (301 for the reference
   clip — verify against `ffprobe -count_frames` on the source, not against memory), and
   intact audio, and the worker log says `Selected video encoder: hevc_nvenc` — not
   `libx264`.
5. `ffprobe` on that output reports `Main 10` / `yuv420p10le` and bt709 for **all three**
   of `color_space`, `color_primaries`, `color_transfer`. `unknown` on any of them is a
   bug. It also reports `level=153` (5.1), and two runs of the same input produce the
   same size — a size that jumps ~2.4x between runs means the level/ceiling setup broke.
6. **Pixels are actually inspected.** Steps 4 and 5 all passed while the GPU encoder was
   emitting visibly corrupt video. Two cheap checks catch it:
   - **No flat frames.** Decode every frame small (`-vf scale=320:180 -f rawvideo
     -pix_fmt gray`) and count frames whose per-frame standard deviation is > 1. Anything
     below the full frame count means frames encoded as flat grey.
   - **No striping.** Mean absolute Laplacian of a 512px crop from a mid-clip frame,
     extracted with `-pix_fmt rgb24` (without it, ffmpeg writes 16-bit PNGs from the
     10-bit stream). It should land near the source's own value (~1-3 on the reference
     clip). ~190 means alternating-column striping, i.e. the P010 surface walked as 8-bit.
7. The `Video done: …` line reports `gpu-encoder=yes` and ~36 fps warm at the default
   encoder settings (1080p→4K). `nvdec=in-process` is expected for progressive input;
   `nvdec=yes` is expected — and logged with a WARNING — for interlace-flagged input, and
   means the in-process decoder was skipped for any other input. `gpu-encoder` absent
   means the piped encoder ran, a large silent downgrade worth explaining before shipping.
8. `modal app logs rtx-media-upscaler` shows no warnings or tracebacks. To scope to the
   worker, `modal function logs 'rtx-media-upscaler/UpscaleWorker.*' --tail 200` (SDK ≥
   1.6.0); `modal app logs <app> --function-call fc-…` isolates one job.
9. `/health` returns 401 without proxy-auth headers.

**The lasting lesson of the striping bug is verification, not encoders:** for a full day
that path produced visibly broken 4K while passing frame count, `ffprobe`, a clean decode
with zero errors, and `_check_frame_integrity`. Only looking at pixels caught it — hence
step 6.

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
`DRIVER_VERSION` / NGX install unless the new workload also uses NGX; most Modal GPU jobs
do not. Keep the ffmpeg pin only if the job touches video, and re-verify NVENC against the
host driver of the day.

Rename `app = modal.App("rtx-media-upscaler")` and the `rtx-upscaler-jobs` Volume, or the
new service will collide with this one in the same workspace.
