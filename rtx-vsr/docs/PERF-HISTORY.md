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
