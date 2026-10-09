# Video I/O performance history

The record of experiments already run on this pipeline, kept verbatim so nobody spends GPU
time re-deriving them. [CLAUDE.md](../CLAUDE.md) carries the operative conclusions; this
file carries the measurements and the reasoning behind them.

Most of it describes the **piped** architecture, where frames crossed process boundaries
and contention between processes dominated everything. That path sat at 12 fps. The
current path is fully GPU-resident at ~57-60 fps and inference-bound, so conclusions here
about encoder settings and preset choice are true of the old architecture and mostly moot
in the current one.

---

## The piped era (historical) — measured 2026-08-16

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

## GPU choice and `cpu=` — settled 2026-08-18 **on the piped path**; re-derive before reusing

**Read this before the next GPU benchmark.** The table below was measured when the
pipeline ran at 12 fps and was bound by contention between three processes. The current
path is fully GPU-resident and bound by *inference* (`infer 74%`, encode 11%). The
conclusion may still hold, but the numbers no longer describe the running system, so
**re-derive every row before quoting it**. What does not need re-deriving is the NVENC
engine count — that is a hardware fact, and it is what makes most Modal GPUs unusable here.

Method for the re-run, so results are comparable:

1. Deploy beside production, never over it:
   `MODAL_APP_NAME=rtx-bench-<gpu> MODAL_GPU=<gpu> modal deploy modal_app.py`.
2. Set `MODAL_WORKER_MAX_CONTAINERS=1`. The autoscaler scales *out* rather than packing a
   container, which hides the effect you are measuring.
3. Run the same clip 3-4 times in one container; read the later runs only (cold start is
   8.0 fps against 12 fps warm on a 12 s clip).
4. Record fps **and** the stage attribution from the `Video done:` line. A GPU that shifts
   the bound from inference to encode changes which optimisations matter next.
5. Run the step 6 pixel checks from the verification protocol. A fast GPU that emits
   striped frames has not won.
6. Compute **cost per frame**, not fps: `($/s for GPU + cpu + mem) / fps`.
7. `modal app stop rtx-bench-<gpu> -y` when done.

Remember the knobs are read at container import, so they must be baked in with `.env()`.
A bare `MODAL_GPU=L40S modal deploy` measures the default and tells you nothing.

Historical table (piped path, 302-frame clip, 1080p→4K, `HIGHBITRATE_ULTRA`, warm):

| Config | warm fps | $/s (GPU+cpu+mem) | $/frame |
|---|---|---|---|
| RTX PRO 6000, `cpu=12` | 11.7 | 0.001053 | 0.0000900 |
| **RTX PRO 6000, `cpu=6`** (current) | **12.2** | **0.000974** | **0.0000798** |
| RTX PRO 6000, `cpu=4` | 12.3 | 0.000948 | 0.0000771 |
| L40S, `cpu=12` | 6.8 | 0.000753 | 0.0001107 |

- **The GPU question was closed on the piped path: keep `RTX-PRO-6000`.** It is the only Modal GPU with
  9th-gen NVENC, and H100/H200/A100/B200/B300 have **zero** NVENC engines — paying more
  buys less. L40S is the only credible alternative and it loses on *both* axes: 56% of the
  throughput for 72% of the cost. It needed ≥71.5% of RTX throughput just to break even.
  L40S is not broken (NGX loads, `hevc_nvenc` selected, `nvdec=yes`); it is inference-bound
  at `infer 75%`. Do not go GPU shopping again without re-deriving that break-even.
- **`cpu=12` was over-provisioned and measured *slower* than `cpu=6`.** The old note that
  4 cores capped 4K throughput predates NVDEC decode; at 4K, `cpu=4` and `cpu=6` both hold
  `decode-wait 1%`. `cpu=6` is the default, keeping margin for larger outputs where swscale
  work scales with pixel count. Re-measure before going below 6.

## Where the 12 fps actually goes (historical, piped path) — probed 2026-08-18

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

## Step 1 — feed NVENC from CUDA memory (`MODAL_GPU_ENCODER`), 2.8x, measured 2026-08-18

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

## Step 2 — fully GPU-resident (`MODAL_GPU_DECODER`) and the settled quality point, 2026-08-18

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

**Batching does not pay — `MAX_BATCH = 1`, measured 2026-08-19.** Warm on the reference
clip, batch=1 measured **56.0 fps against batch=2's 53.2**. A larger batch makes the
encoder wait on a longer run of inference instead of overlapping with it. The constant
also pins the batch for a correctness reason — see the cross-stream DLPack race in the
encoder-sync section below — so do not raise it for either speed or memory reasons without
re-reading both.

**Decoupled encode threading — evaluated 2026-08-18, declined.** Video Codec SDK 13.1
introduced a decoupled-queue architecture so NVDEC/CUDA/NVENC work on *different* frames
concurrently, and the obvious suggestion is to apply it here: decode is already on a reader
thread, but `_flush()` still encodes inline right after inference. The arithmetic kills it —
encode is **12%** of wall time, so perfect overlap is worth at most ~1.14x (~59 → ~66 fps).
Against that, a writer thread reintroduces packet-ordering and DLPack-buffer-lifetime
hazards on the encoder side, and the reader-thread experiment directly above showed that
this kind of "wait" is often overlapping GPU work that threading does not recover at all.
Do not spend effort here; the lever is inference, and inference is the SDK's.
---

## NVENC does not observe CUDA stream ordering — fixed 2026-08-19

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

## Encoder-bound, the hidden bitrate cap, and new defaults — measured 2026-10-03

Source: a BBC Blu-ray remux (H.264 MBAFF, `field_order=tt`, untagged colour, fine grain),
1080p→4K, `HIGHBITRATE_ULTRA`, RTX PRO 6000 Blackwell Server Edition, driver 580.95.05.
Probes: `scripts/probe_stages.py`, `scripts/probe_encoder_quality.py`,
`scripts/probe_nvdec_interlace.py`. Second opinions from Codex (gpt-6-astra) on the ffmpeg
and PyNvVideoCodec sources.

### The pipeline is encoder-bound

| Stage, alone (300 frames) | Result |
|---|---|
| VSR `run()` | 3.5 ms/frame |
| VSR + integrity + P010 | 4.4 ms/frame |
| NVENC P4+qres+UHQ / P4+fullres | 65.0 / 61.3 fps |
| NVENC P5+fullres+HQ / +UHQ | 68.1 / 38.5 fps |
| NVENC P6+qres / P6+fullres / P6 single pass | 29.6 / 27.5 / 30.1 fps |
| Full pipeline, serial vs VSR overlapped on a side stream (P4 / P6) | 61.0 vs 64.5 / 26.7 vs 26.9 fps |
| Concurrent sessions, P6+fullres: 2 / 3 / 4 | 47.7 / 58.9 / 65.9 fps combined |
| Concurrent sessions: 4x P4+qres / 2x and 4x P5+HQ | 166.8 / 125.7 and 209.8 fps combined |

Encoder video clock 1912-1935 MHz under load, no throttle reasons, ~100 W of 600 W. The
reference-list kwargs (`numrefl0/1=5` vs driver default) and UHQ vs HQ at P6 changed
nothing. The 2026-08-18 "inference-bound at infer 74%" reading included time blocked on the
encoder. StaxRip/NVEncC on an RTX 4070 ran the user's job at 71.7 fps with **P5 + HQ**
(VEClock 2093 MHz); the same settings measured 68.1 fps per session here.

### The bitrate cap and the "2.4x file" bug

With `cq` set, PyNvVideoCodec zeroes `maxBitRate`; the driver substitutes a VBR ceiling
from the level. With `level` unset the driver picked Level 5.0 or 6.0 per session for the
same input. Every "normal" output was L5.0 (~17.5 Mbps), every 2.4x outlier L6.0
(~40 Mbps); at an explicit L5.1 the HRD ceiling read 32 Mbps and CQ 18 and CQ 24 gave the
same size (~27.5 Mbps). `Reconfigure()` with `maxBitRate` 100 Mbps before the first frame
fixed it: CQ 18 → ~80 Mbps, CQ 24 → ~35 Mbps, and repeated runs became byte-identical.

The first version used a 160 Mbit VBV buffer. In a warm container, an encoder created at a
different resolution than the previous job then failed `nvEncInitializeEncoder` with
error 8 (4K→8K, 8K→4K; reproducible; the original code passed 4K→8K). Bisected on three
parallel deploys: dropping the explicit level, the ceiling, or P5+fullres each made it
pass. A 1 s buffer (100 Mbit) passed 4K→8K→4K→1620p→4K and 8K→1620p→8K→4K, and the 4K
outputs stayed byte-identical. Above 4K the level is left to the driver.

### Quality matrix (ceiling lifted; VMAF 4K v0.6.1 vs a lossless encode of the same VSR frames)

The fps column is the probe's in-container decode→VSR→encode loop with up to ten probe
containers running at once, so it reads lower than the encode-only table above.

| Setting | fps (1 session) | CQ 18: Mbps / VMAF / PSNR-Y | CQ 24: Mbps / VMAF / PSNR-Y |
|---|---|---|---|
| P4+qres+UHQ | 51-56 | 80.3 / 93.45 / 42.69 | 33.9 / 90.20 / 41.02 |
| P5+fullres+HQ | 58-64 | 85.6 / 92.23 / 42.46 | 42.6 / 89.62 / 41.17 |
| **P5+fullres+UHQ** | **36** | 81.9 / 94.28 / 43.33 | 34.7 / 91.28 / 41.52 |
| P6+fullres+UHQ | 25 | 82.2 / 94.39 / 43.39 | 34.7 / 91.34 / 41.54 |
| P7+fullres+UHQ | 24 | 82.1 / 94.50 / 43.45 | 34.7 / 91.43 / 41.56 |

Defaults adopted: P5 + fullres + UHQ, CQ 20, Level 5.1 High, 100 Mbps ceiling. End to end
on the 61 s sample: 35.7 fps warm, 78 Mbps, 1526/1526 frames, byte-identical across runs.

### Decode-side fixes on this source

- In-process NVDEC deinterlaced the PsF frames (Adaptive mode, chosen by `NvDecoder.cpp`
  for any interlaced sequence): odd rows 7.8% of pixels off by >6 levels, vertical
  gradient energy 0.684x of a woven decode. Interlace-flagged inputs now use ffmpeg's NVDEC
  (weave). Downscaled back to 1080p, the woven path kept 69% more fine detail
  (Laplacian 4.33 vs 2.56) and restored the source's vertical/horizontal balance.
- swscale's default yuv→rgb24 tables read ~1.6 luma levels dark; `accurate_rnd+
  full_chroma_int` measured +0.004. With it, `BICUBIC` mode round-trips at +0.007 and
  `HIGHBITRATE_ULTRA` at -0.35 (the model's own offset).
- rawvideo output needed `-fps_mode passthrough` (306 vs 304 frames).
- The woven path cost about 12% throughput against in-process decode at the old P4
  settings (52 vs 59 fps). At the new defaults the encoder (~36 fps) is far below either
  decode path; the in-process path was not re-measured at those settings.

### Production deploy failure and fix — 2026-10-03 (evening)

The first production deploy of these defaults failed every video job with error 8 at
`CreateEncoder`; production was restored by redeploying the previous `modal_app.py`
(rollback is not on the current Modal plan). Cause: every bench sequence had started with a
video job, while production's first job was an image. Reproduced on the bench: image →
video failed every time. Bisected on four parallel deploys: without the explicit level, or
at P4+qres, it passed; tuning and the VBV ceiling were irrelevant. Loading the job's VSR
effect before `CreateEncoder` fixed image → video; 1620p → 4K still failed intermittently,
so an error-8 rejection of the explicit level now retries without it (WARNING logged).
Validation: img/4K ×2, and 16 mixed 4K/8K/1620p jobs in two warm containers: all passed,
the fallback fired 4 times (those 4K outputs carried level 5.0 or 6.0, same ceiling and
size). 1620p outputs alternated between two sizes 0.2% apart depending on the preceding
job; 4K outputs at level 5.1 stayed byte-identical.

## Four NVENC sessions on keyframe-aligned segments — measured 2026-10-04

The encoder bound above (one P5+fullres+UHQ session, ~38 fps) is per *session*; the card
has four NVENC engines. Inputs of 30 s or more are now cut at source keyframes into
segments of up to 30 s, segment *i* goes to session *i mod 4*, and each segment is muxed
to its own mp4 and joined with ffmpeg's concat demuxer. PyNvVideoCodec 2.2.3, driver
580.95.05, same quality point. Probe: `scripts/probe_parallel_encode.py`.

### Throughput

| Case | Result |
|---|---|
| Encode only, production settings incl. level + ceiling: 1 / 2 / 3 / 4 sessions | 45.1 / 73.6 / 93.0 / 103.9 fps combined |
| Full per-frame pipeline per thread (VSR, integrity, P010, device sync, Encode), 1 / 2 / 4 threads | 40.0 / 62.8 / 82.8 fps combined |
| Same, 4 threads with a per-thread CUDA stream sync instead of `torch.cuda.synchronize()` | 98.5 fps, **striped** (Laplacian 52-85, flat frames) |
| Segmented prototype, 10-min progressive clip (15026 frames, 20 x 30 s), 1 / 4 sessions | 41.7 / 78.9-81.4 fps |
| Same, 4 sessions at `cpu=12` | 81.1 fps (CPU is not the bound in steady state) |
| Production, 61 s Blu-ray sample, warm, 4 segments | 55 fps end to end: setup 3 s, encode 73.5 fps, join 4 s |
| Production, 10-min progressive clip, warm, 24 segments | **75.1-75.5 fps** end to end: setup 8-9 s, encode 82.4-82.9 fps, join 9-10 s |
| Production, full 46-min H.264 episode (69552 frames, 96 segments), first job in its container | **74.8 fps** end to end: setup 38 s (NGX init 11 s, encoders 26.6 s), encode 81.4 fps, join 37 s; peak RSS 10.0 GiB; 8.05 GB output. Verified in place: 69552 packets and decoded frames, uniform pts, level 153, bt709 x3, all 95 seams and pre-seam frames clean (max Laplacian 12.3), 130 flat frames against the source's own 130 |
| Unsegmented production path, 61 s sample, warm (same day, for reference) | 34.7-35.7 fps |

UHQ sessions do not scale linearly: each slows to ~26 fps when four run at once (encode
only), so four engines buy ~2.3x encode-only and ~2x end to end, not 4x. Per-thread
stream syncs are not an option: nvvfx does not run on the caller's stream.

### Things that had to be true, and were checked

- **Frame-exact partition.** Each segment decodes with input `-ss` one second before its
  keyframe, `-copyts`, and `trim=start_pts:end_pts` in the stream time base. framemd5 of
  the concatenated segments equals a sequential decode on mkv, mp4, ts and m2ts remuxes of
  the interlace-flagged sample (1526/1526), on a full H.264 episode (69552/69552) and on a
  full HEVC episode (86318/86318). `-ss` is relative to the container's `start_time`:
  given the absolute pts, MPEG-TS (start 1.56 s) lost 254 of 1526 frames. `trim`'s
  `end_pts` stops the decode, so a segment does not read to EOF (10 s of a 46-min file:
  2.5 s). `-enc_time_base:v demux -stats_enc_pre` logs the exact source pts of every
  delivered frame; production fails the job unless a segment's first frame is its keyframe.
- **Segment joins.** A continuous session per worker with `FORCEIDR` at each segment
  start emits packets in segment order; packets are routed to the segment muxer by the
  session's input index (`m_frameNum`, never reset). Concat output: exact frame count,
  uniform pts, keyframes exactly at the seams, `ffprobe -count_frames` equal to the
  source. `EndEncode()` per segment also works but measured 37.4 fps against 44.0 on one
  worker (the flush idles VSR).
- **Same parameter sets.** All sessions created with the explicit level emit
  byte-identical `GetSequenceParams()`, and four sessions encoding the same segments give
  byte-identical bitstreams. This matters because `FFmpegMuxer` strips the inline
  VPS/SPS/PPS from IDR packets and the joined file has one hvcC.
- **Seams are not visible.** The Laplacian spike at segment starts (e.g. 14.8 at frame
  386) is the source's own keyframes: the unsegmented output reads 18.7 at the same frame.

### Quality: a tie on average, an occasional dip on the seam frame

VMAF 4K v0.6.1 and PSNR-Y against a lossless encode of the same VSR frames (production
decode, VSR and P010), 61 s sample, 4 segments:

| Output | VMAF mean | 5th pct | min | PSNR-Y |
|---|---|---|---|---|
| Unsegmented | 91.253 | 83.03 | 72.86 | 40.888 |
| Segmented | 91.271 | 83.05 | 72.43 | 40.884 |

On the first frame of each segment (an IDR on a source keyframe) the segmented output
scored 80.2 / 87.4 / 85.0 against 86.3 / 86.5 / 90.1 unsegmented, a one-frame dip of up to
~6 at two of three seams. Neighbouring frames went both ways (frame 1129: 89.4 segmented,
78.9 unsegmented). With 30 s segments that is one such frame per 30 s per session.

### CreateEncoder error 8 is random, not state-dependent

| Series (P5+fullres+UHQ, level 5.1) | Failures |
|---|---|
| Fresh container | 3/30 |
| No level | 0/15 |
| VSR effect loaded | 5/30 |
| Three sessions alive | 3/20 |

The next attempt almost always succeeds (one double failure in 80), so the encoder
factory makes up to six attempts before the no-level fallback. That also removes most of
the single-session fallbacks seen on 2026-10-03 (4K outputs at level 6.0).
`CreateEncoder` itself costs ~0.5-0.8 s, so four sessions are ~3 s of setup per job.

### The surface-lifetime race

`Encode()` copies the P010 tensor into the encoder's own buffer with `cuMemcpy2DAsync` on
a non-blocking stream that PyNvVideoCodec creates, and returns before the copy runs.
Dropping the tensor right after `Encode()` hands its memory back to torch's allocator,
which does not know about that stream. With four threads allocating, a later allocation
can overwrite the surface before the copy reads it. This is a race in the source, not a
measured failure: the probe once showed one B-frame of 15026 encoded differently between
two runs (diffuse, max 18 10-bit levels), but those runs were in different containers and
could have been on hosts with different drivers (next section), so the observation
cannot be pinned on the race. Both paths now keep the surface alive until the next
device-wide sync, which waits for every stream. After the fix, four warm runs in one
container were byte-identical, and so was a run after an image job.

### Run-to-run identity is per host driver

During these runs Modal placed some containers on hosts with driver **610.57.04**. VSR
still worked (the NGX library is from 580.95.05; `_warn_on_driver_mismatch` fired), but
the outputs from those hosts differed in size from each other and from the 580 hosts
(592,119,197 and 592,096,538 against 592,182,118 bytes). On 580 hosts every 4K run of
the sample was identical, across containers and after image, 8K and 1620p jobs. The
10-minute clip gave 1,587,814,094 bytes three times in one warm container. Compare sizes
within one warm container, not across containers.

### The unsegmented path is unchanged in output

The 12 s interlace-flagged clip (under the 30 s threshold) gave 120,735,540 bytes twice on
the new build, and 120,735,540 on the old build's warm run, both on 580 hosts. The GPU-side
uint8→float conversion, the surface hold and the shared encoder factory therefore change
nothing. The old build's *first* run gave 120,766,902 bytes, because it hit `NVENC rejected
explicit HEVC level 5.1` and fell back to the driver-selected level. The new factory
retries instead.

### Fixed costs

The first job in a container pays NGX initialisation (~10-14 s with four effects
loading). After that the pool threads keep their effects (`effects 0.0s`). Each job still
creates four encoders (3-9 s, depending on how many error-8 retries it hits; once 26.6 s,
on the episode run, cause not visible in the logs), and joins and remuxes at the end (~4 s
for 61 s with a DTS→AAC transcode, ~9 s for 10 min, 37 s for a 46-min episode). Peak RSS
on that episode was 10.0 GiB. The unsegmented muxer would have held the 8 GB bitstream on
top of that, and a Blu-ray film at 78 Mbps is ~70 GB. Below 30 s the unsegmented path is kept, because those fixed costs are not
earned back.

## Why four sessions give ~2x, and what does not fix it — measured 2026-10-04 (evening)

Modal workspace `tinof`. Every run except two was on a 580.95.05 host; the two 610.57.04
runs are marked. Probe: `scripts/probe_parallel_encode.py`, one case per single-use
container, 61 s Night Manager sample unless noted. Run-to-run noise between containers is
about ±5%.

### Encode-only, production settings (P5 + fullres, CQ 20, level 5.1, 100 Mbps ceiling)

| Case | Aggregate fps | enc % | SM % |
|---|---|---|---|
| UHQ, 1 / 4 / 6 / 8 sessions | 43.4 / 107.4 / 118.9 / 122.9 | 29 / 77 / 88 / 92 | 5-11 |
| HQ, 1 / 4 sessions | 59.8 / 183.5 | 26 / 77 | 1-3 |
| UHQ, no ceiling (driver cap, 27 Mbps), 1 / 4 sessions | 39.4 / 110.4 | 26 / 72 | 5-10 |
| UHQ, 4 sessions in 4 processes | 109.2 (27.3-27.8 each) | 77 | 11 |

- UHQ scaling is worse than HQ at matched settings: 2.5x vs 3.1x. The old 210 fps HQ figure
  (`probe_stages.py`) had no ceiling, so it was never like-for-like.
- Not the cause:
  - the bitrate: uncapped UHQ scales the same;
  - the GIL or in-process serialization: processes match threads;
  - the temporal filter's CUDA work: SM utilisation stays near 10%.
- The engines simply sit idle part of the time with four UHQ sessions. More sessions fill
  some of it.

### Per-worker CUDA streams: nvvfx ignores `stream_ptr` for its copies

Four threads on pre-decoded GPU frames, each running VSR, integrity checks, P010 and Encode.
All four threads encode the same frames, so a correct scheme gives four byte-identical
outputs. Production's device-wide sync does.

| Scheme | fps | Outputs identical |
|---|---|---|
| Production: `torch.cuda.synchronize()` per frame | 75-82 | yes |
| Torch on stream S, nvvfx on stream 0 (the old "stream" mode) | 97 | no, striped |
| One stream S per worker for torch, nvvfx `stream_ptr=S`, `CreateEncoder(cudastream=S)`, forwarding `__dlpack__` patch (s1-s4) | 102-105 | **no** |
| Same, single thread | 39 | differs from production |
| Same, on blocking streams (`cudaStreamDefault`) | 67-69 | yes |
| nvvfx on stream 0, joined to S with events (`sz`) | 84-88 | yes |
| `sz` with nvvfx 0.2.0.0 | 87 | yes |

How the s1-s4 failure was diagnosed:
- Exact checksums of every P010 surface handed to Encode() showed frame *i* equal to
  production's frame *i−1*.
- Device syncs after `run()`, a lock around VSR, and splitting `run()` apart with a host or
  event wait before `get_output()` each left 25-30% of frames wrong.
- Only blocking streams and `sz` matched byte for byte. Both make nvvfx's input transfer and
  output copy legacy-stream work.
- The outputs that came out wrong looked healthy: no striping, Laplacian 3-7. Only the md5
  comparison across threads caught them.

What does not help on top of `sz`:
- `run(non_blocking=True)`: 86.6 fps, no change.
- Deferring the integrity read-backs: 74-76 fps, worse.

### More sessions: large in isolation, small end to end

`pipe` cases (pre-decoded frames, 300 per thread):

| Pipelines | Production sync | `sz` |
|---|---|---|
| 4 | 77-82 | 86-88 |
| 6 | 89 | 89-93 (93 on 580; 89 on 610) |
| 8 | 95.4 (580), 88.3 (610) | 96.5 / 96.1 |

The segmented prototype (`full:K:idr:30:clip10`), a 10-min progressive 1080p25 WEB-DL clip
(15152 frames, 21 segments), with production sync:

| Sessions | cpu=6 | cpu=12 |
|---|---|---|
| 4 | 70.3 (610 host); 78.9-81.4 on 2026-10-04 morning | — |
| 6 | 76.1 | 80.0 |
| 8 | 80.6 | 82.9 |

All runs were frame-exact, with clean seams and one SPS. On the real path, extra sessions
buy little or nothing over the morning's 4-session numbers. Two reasons:
- With 6-8 ffmpeg segment decoders starting together on 6 CPUs, each decoder's first frame
  takes 9-11 s, against ~2 s with four.
- 21 segments over 8 workers leave a ragged tail.

The `pipe` gains do not carry over until decode start-up and load balance change.

### CreateEncoder error 8 was sticky in some containers

First batch, no backoff between attempts: these failed ten consecutive `CreateEncoder`
attempts at level 5.1:
- `enc:1:uhq:nocap`, `enc:4:uhq:nocap` and `enc:4:hq:nocap`;
- two of the four children of `encproc:4:uhq`.

Rerun with backoff: the two UHQ cases and `encproc` passed. `enc:4:hq:nocap` failed ten
attempts again, on a 580 host.

Within an affected container, every attempt failed. The 2026-10-04 morning picture ("random,
the next attempt almost always succeeds") does not hold for every container. Production
falls back to no level after six attempts. Investigate this before raising the session count.

## TrueHDR (SDR to HDR10) and nvidia-vfx 0.2.0.0 — measured 2026-10-04

`scripts/probe_truehdr.py` and a bench deploy (`rtx-bench-hdr`, one container) on the
61 s BBC Blu-ray sample and a 273-frame progressive 1080p25 clip, 1080p to 4K,
`HIGHBITRATE_ULTRA`, TrueHDR `contrast=102,saturation=102,middlegray=46,maxluminance=680`.

### The upgrade itself (0.1.0.1 to 0.2.0.0)

- Same frames, same 580 host: VSR output is 97% identical, and the rest differs by exactly
  1/255. Mean |d| was 0.027/255 and max 1/255 on four frames.
- `pipe:1:dev:hash`:
  - Each version reproduces exactly run to run.
  - 0.2.0.0's P010 checksums were the same on a 580 and a 610 host.
  - Output size +0.04%, Laplacian unchanged, 37.7 against 37.1 fps.

### Per-stage cost at 4K (single thread, synced)

| Stage | ms/frame |
|---|---|
| VSR (`HIGHBITRATE_ULTRA`) | 3.5-3.6 |
| TrueHDR (debanding on) | 0.96-1.06 |
| `_hdr10_to_p010` incl. MaxCLL/MaxFALL, int64 planes | 1.74 |
| same, int32 planes (shipped) | 1.21 |

TrueHDR at other sizes: 0.57 ms at 2880x1620, 3.3 ms at 7680x4320, 3.4 ms at 8192x4320,
with identical channel means at every size.

### End to end (warm, `Video done:`)

| Job | Host | SDR | HDR |
|---|---|---|---|
| 273 frames, single session | 580 | 36.3 fps | 34.0 / 36.3 / 37.4 |
| 273 frames, single session | 610 | 34.2 | 35.4 / 35.4 |
| 61 s, 4 sessions, encode phase | 580 | 72.4 | 61.1 / 63.0 (int64 conversion) |
| 61 s, 4 sessions, encode phase | 610 | 67.4 | 55.5 / 55.4 (int32 conversion) |

- **Single session:** TrueHDR is free, because the encoder is the bound.
- **Four sessions:** HDR costs 14-18% of the encode phase. The GPU is not saturated (about
  35% busy). Each thread's per-frame device-wide `torch.cuda.synchronize()` waits for every
  thread's queued work, so ~1.7 ms more GPU work per frame shows up about fourfold. The
  `infer` share rose from 30-32% to 44-45%.
  - The lever is the same as for SDR: the `sz` stream scheme, which does not need the host
    sync (see "Per-worker CUDA streams").
  - Debanding off cut TrueHDR from 0.31 to 0.14 ms at 1080p, at a quality cost the user decides on.

### Correctness checks that passed

- **Bitstream:**
  - The HDR10 SEI bytes are identical to libx265's.
  - The SEI is on 8/8 keyframes of the segmented output (every FORCEIDR seam and gop-250
    IDR) and 2/2 on the short clip.
  - The `colr` nclx box is present. Tags are bt2020nc / bt2020 / smpte2084 / topleft, level 153.
- **Determinism and threads:**
  - Byte-identical sizes run to run (34.2 MB x3, 588.1 MB x2).
  - TrueHDR is stateless: forwards and backwards are identical.
  - Four threads match one thread on every P010 hash.
- **Pixels:**
  - Flat-frame and Laplacian checks on tone-mapped frames, seams included, are in the
    source range.
  - `_hdr10_to_p010` is within ±1 code of float64 (1.6e-5 of luma samples differ).
  - The encode round trip is ordinary codec loss: signed error ≈ 0 and absolute error below
    SDR's on the same frames. HDR files at CQ 20 were 28% smaller on the short clip and the
    same size on the grainy 61 s sample (100 Mbps ceiling).
- **Light levels:** measured MaxCLL 679 nits, against 680 signalled. Measured MaxFALL was
  30-52 nits on these dark drama test clips, against the 300 the StaxRip command signals.
  Full films with bright scenes measure higher, so no single constant fits; the measured
  values became the default `max_cll` (CLAUDE.md section 6). Letterbox bars count in the frame
  average, so MaxFALL reads slightly low on letterboxed sources (~1.33x for 2.39:1).
- **10-bit:**
  - `x2bgr10le` decode reads +0.5 codes against the exact rgb24 path; rgb48 reads 1-1.5 low.
  - 10-bit VSR output differs from the 8-bit path by 3 codes (0.3%) on average.
  - DENOISE/DEBLUR in RGB10A2 return alpha 0. VSR ignores it, and the code restores 3 anyway.
