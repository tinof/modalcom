# RTX Media Upscaler on Modal

Runs NVIDIA RTX Video Super Resolution on [Modal.com](https://modal.com) as an
authenticated HTTP service, and doubles as a **template for other Modal GPU jobs**.

Working on this with an AI agent? Read [CLAUDE.md](CLAUDE.md) first — it carries the
conventions and the hardware constraints that are easy to break.

You can:
- Upload an image or video file.
- Upscale it on an NVIDIA GPU (default: RTX PRO 6000 Blackwell).
- Download the upscaled output back to your local machine.

## Architecture

Two tiers, so GPU time is only billed while frames are actually being upscaled:

- **Web tier** (`api`): a CPU-only FastAPI app. It uploads the request body to a Modal Volume through the Volume API (no mount), spawns the GPU job, and streams the finished file back. Handles up to 20 concurrent requests per container.
- **GPU worker** (`UpscaleWorker`): a GPU class whose `@modal.enter()` warms CUDA and probes the NVENC encoder once per container. The super-resolution model stays loaded between jobs that share quality and output size.

Because video jobs can run far longer than Modal's 150-second HTTP request limit, the API is asynchronous: `POST /upscale` returns a `call_id` immediately and `GET /result/{call_id}` returns 202 while pending, then the file.

- `modal_app.py`: the Modal app (web tier + GPU worker).
- `modal_client.py`: local client that uploads, polls, and saves the output.

## Two Constraints That Are Easy To Break

Both are pinned in `modal_app.py`. Read this before changing the image.

**1. The NGX driver library (`DRIVER_VERSION`).** RTX Video Super Resolution is an NGX
feature. The `nvidia-vfx` wheel bundles the NGX snippet, but at runtime it `dlopen`s the
driver-side `libnvidia-ngx.so.1` — and Modal’s container runtime does not inject that
library. Without it, `NvVFX_Load` fails on **every** GPU type with:

```
NvVFX_Load failed: The effect has not been properly initialized (code -12)
```

The image therefore installs that one library from the matching NVIDIA driver package.
`DRIVER_VERSION` must track the host driver reported by `nvidia-smi` (currently
`580.95.05`); the worker logs a warning at startup if the two diverge.

**2. The ffmpeg build (`FFMPEG_URL`).** NVENC availability depends on which NVIDIA Video
Codec SDK the ffmpeg binary was linked against, and each SDK sets a minimum driver version:

| ffmpeg build | Video Codec SDK | Requires driver | Usable here |
|---|---|---|---|
| BtbN `master-latest` | 13.1.15 | 610.0+ | No — NVENC silently unavailable |
| BtbN `n8.1` (pinned) | 570-compatible | 570.0+ | Yes, including `av1_nvenc` |
| Debian apt ffmpeg 5.1 | — | — | Yes, but no `av1_nvenc` |

Pointing `FFMPEG_URL` at `master-latest` makes every NVENC encoder fail to initialise and
the worker falls back to CPU `libx264`. Encoder preference is `hevc_nvenc` → `h264_nvenc`
→ `libx264`; `av1_nvenc` also works on the pinned build if you want it, but HEVC is the
safer default for MP4 playback.

## Video output format and throughput

Video jobs run **entirely on the GPU**: NVDEC decodes, super-resolution runs on the decoded
CUDA tensor, and NVENC encodes straight from GPU memory. Output is **HEVC Main 10**
(`yuv420p10le`, Level 5.1 High tier) tagged bt709 for matrix, primaries and transfer.

### Default encoder settings

Chosen by measurement on 2026-10-03 (BBC Blu-ray, 1080p→4K, `HIGHBITRATE_ULTRA`; details in
[docs/PERF-HISTORY.md](docs/PERF-HISTORY.md)). The NVEncC equivalent is:

    --qvbr 20 --codec h265 --preset p5 --tune uhq --output-depth 10 --profile main10
    --tier high --level 5.1 --max-bitrate 100000 --multipass 2pass-full --aq
    --aq-strength 10 --aq-temporal --bframes 5 --lookahead 32

| Setting | Default | Why |
|---|---|---|
| Preset | `p5` | P6 and P7 add only 0.06–0.15 VMAF over P5 and cost about 30% more encode time. |
| Multipass | `fullres` (2pass-full) | About 1.1 VMAF better than `qres` at the same bitrate. |
| Tuning | `uhq` | 1.7 VMAF better than `hq` with 19% fewer bits. UHQ also enables NVENC's temporal filter and deeper lookahead; PyNvVideoCodec cannot set those separately. |
| CQ | `20` | A file-size choice. CQ 18 measured about 80 Mbps (VMAF 94.3), CQ 24 about 35 Mbps (VMAF 91.3). |
| Level / ceiling | `5.1` High, 100 Mbps | See below. Above 4096x2176 the driver picks the level. |

10-bit encoding is used even for 8-bit sources because it costs nothing on NVENC and keeps
upscaled gradients from banding.

**The bitrate ceiling matters more than the preset.** PyNvVideoCodec zeroes `maxBitRate`
whenever a CQ target is set, and the driver then applies its own VBR ceiling: about
20 Mbps at Level 5.0, 32 Mbps at 5.1, 48 Mbps at 6.0. Before 2026-10-03 every 4K output
hit that ceiling, so CQ 18 and CQ 24 produced the same file size. The driver also picked
the level per session (5.0 or 6.0 for the same input), so the same job sometimes produced a
2.4x larger file. The worker now sets Level 5.1 explicitly for outputs up to 4K and
restores a 100 Mbps ceiling (1 s buffer) with `Reconfigure()` before the first frame. In
some warm-container states (a job right after one with a different output size) NVENC
rejects the explicit level; the worker then retries with the driver's level and logs a
warning. The ceiling, and so the file size, is the same either way. Expect output bitrate to follow the content:
grainy 4K at CQ 20 lands roughly between 35 and 80 Mbps.

### Interlace-flagged sources (UK/EU Blu-ray and broadcast)

Many HD discs carry progressive 25p pictures in a 1080i container (PsF). PyNvVideoCodec
creates its decoder in Adaptive deinterlace mode for any interlace-flagged stream, and that
mode rewrote one field's rows on such a disc: 32% of vertical detail was gone before VSR saw
the frame. The worker therefore decodes interlace-flagged inputs through ffmpeg's NVDEC
path, which weaves the fields losslessly, and logs a warning saying so. That path converts
to RGB with exact swscale flags (`accurate_rnd+full_chroma_int`; the default lookup tables
sit about 1.6 luma levels dark), picks BT.709 for untagged HD, and passes frames through
1:1. Truly interlaced 50i/60i video would need a real deinterlacer; that is not handled.

## Performance

Measured on RTX PRO 6000 Blackwell Server Edition, 1080p→4K, `HIGHBITRATE_ULTRA`, warm
containers. Read the measurement rules at the end of this section before you compare a
new number against them.

### The encoder is the limit

| Stage, measured alone | Speed |
|---|---|
| VSR `run()` | 3.5 ms/frame (about 286 fps) |
| VSR + integrity check + P010 conversion | 4.4 ms/frame |
| NVENC, P4 + qres + UHQ | 65.0 fps |
| **NVENC, P5 + 2pass-full + UHQ (default)** | **38.5 fps** |
| NVENC, P5 + 2pass-full + HQ | 68.1 fps |
| NVENC, P6 + 2pass-full + UHQ | 27.5 fps |
| NVENC, P7 + 2pass-full + UHQ | about 24 fps |

One encoder session is the bottleneck. With the defaults a job runs at about **36 fps** end
to end (1080p→4K, warm): roughly 42 minutes of GPU time per hour of 25 fps video, about
$2.40 at Modal's 2026-10 RTX PRO 6000 rate. On the 61-second Night Manager Blu-ray sample
CQ 20 produced 78 Mbps, i.e. about 35 GB per hour of grainy 4K. Overlapping VSR with
encode measured no faster, because the encoder is already busy all the time.

The card has **four NVENC engines** and one session uses one. Independent sessions scale:
four P5 + HQ sessions measured 210 fps combined, four P6 + UHQ sessions 66 fps. Splitting a
job into time chunks encoded in parallel is the next speed lever. It is not implemented yet.
PyNvVideoCodec 2.2.0 exposes no split-frame encoding option, so a single session cannot
spread one frame over several engines.

Each job logs a line like `Video done: 1526 frames in 42.8s = 35.7 fps (decode-wait 1%,
infer 34%, encode 54%; nvdec=yes, batch=1, gpu-encoder=yes)` (an interlace-flagged
Blu-ray, hence `nvdec=yes`; progressive input shows `nvdec=in-process`). The `infer` share
includes time spent waiting for the encoder to accept the next frame. Use it to spot a regression
or a silent fallback. `nvdec=yes` is expected for interlace-flagged inputs.

### Earlier history

The original pipeline piped rawvideo through two ffmpeg processes and ran at 12 fps.
Keeping every frame on the GPU took it to 56–60 fps at the old P4 + qres settings. The
full history, including the stage-isolation tables from that era, is in
[docs/PERF-HISTORY.md](docs/PERF-HISTORY.md).

### How to measure without fooling yourself

1. **Compare warm containers only.** The same configuration measured 8.0 fps cold and
   12 fps warm on a 12-second clip. The effect loads once per container. Run the clip
   three or four times in one container and read the later runs.
2. **Use real content.** On synthetic noise the `cq` response measured inverted.
3. **Use one encoder session per container.** With several `CreateEncoder` calls in one
   process the measurements are worthless. The same settings measured 17199 KiB and
   7204 KiB on different runs, and every session after the first returned an identical
   size whatever changed. `scripts/probe_gpu_encoder.py` runs each variant in its own
   container for this reason.
4. **Check the pixels, not only the metadata.** See the note below.

### Speed is not correctness

For a full day this path produced visibly striped 4K while passing frame count, `ffprobe`,
a clean decode with zero errors, and the built-in integrity check. Two cheap checks catch
that class of failure, and both belong in any benchmark run:

- **Flat frames.** Decode every frame small and count frames whose standard deviation is
  above 1. A count below the frame count means frames encoded as flat grey.
- **Striping.** Take the mean absolute Laplacian of a 512-pixel crop. It should land near
  the source value (1–3 on the reference clip). About 190 means column striping.

`scripts/probe_p010_encode.py` runs both checks against a synthetic clip.

Two escape hatches exist for inputs the GPU path cannot take. `MODAL_GPU_DECODER=0` falls
back to a decode subprocess (this also happens automatically for codecs NVDEC cannot
handle), and `MODAL_GPU_ENCODER=0` falls back to piping frames to ffmpeg. Both fallbacks
use the same encoder quality point, and every degradation is logged with its cause.

## Choosing a GPU

The default is `RTX-PRO-6000`. Change it with `MODAL_GPU`.

### Which Modal GPUs can run this job at all

NVENC is a separate hardware engine from the CUDA cores. Several Modal GPUs have none.

| GPU | NVENC engines | Usable for this service |
|---|---|---|
| RTX PRO 6000 (Blackwell) | Yes, 9th generation | Yes — current default |
| L40S | Yes | Yes |
| H100, H200, A100, B200, B300 | **None** | No — falls back to CPU `libx264` |

A GPU with no NVENC still produces correct output, but it encodes on the CPU at roughly a
tenth of the speed. Paying more for such a card buys less throughput, not more.

### Cost per frame, not cost per second

Throughput alone is the wrong measure. Compare cost per frame:

    cost per frame = (GPU + CPU + memory cost per second) / fps

Measured on a 302-frame clip, warm, **on the older piped architecture**:

| Configuration | fps | Cost per second | Cost per frame |
|---|---|---|---|
| RTX PRO 6000, `cpu=12` | 11.7 | $0.001053 | $0.0000900 |
| **RTX PRO 6000, `cpu=6`** (current) | **12.2** | **$0.000974** | **$0.0000798** |
| RTX PRO 6000, `cpu=4` | 12.3 | $0.000948 | $0.0000771 |
| L40S, `cpu=12` | 6.8 | $0.000753 | $0.0001107 |

L40S is cheaper per second but loses on cost per frame: 56% of the throughput for 72% of
the cost. It needed at least 71.5% of RTX throughput to break even. It is not broken — NGX
loads and `hevc_nvenc` is selected — it is inference-bound at 75%.

### Read this before the next GPU benchmark

**The table above was measured on the piped architecture, when the pipeline ran at 12 fps
and was bound by contention between processes. The current path is bound by
super-resolution inference. The ranking may therefore change, and every row needs
re-deriving before it is trusted.**

Run a benchmark like this:

1. Deploy a variant beside production instead of replacing it:
   `MODAL_APP_NAME=rtx-bench-l40s MODAL_GPU=L40S modal deploy modal_app.py`.
2. Set `MODAL_WORKER_MAX_CONTAINERS=1`. The autoscaler otherwise scales out rather than
   packing a container, which hides what you are trying to measure.
3. Run the same clip three or four times in one container. Read the later runs only.
4. Record fps, the stage attribution from the `Video done:` line, and the output size.
5. Run the two pixel checks above. A fast GPU that emits striped frames has not won.
6. Compute cost per frame from Modal's current per-second price for that GPU, plus the CPU
   and memory you requested.
7. Stop the bench app when finished: `modal app stop rtx-bench-l40s -y`.

Environment knobs exist for exactly this: `MODAL_APP_NAME`, `MODAL_GPU`,
`MODAL_WORKER_CPU`, `MODAL_WORKER_MAX_CONTAINERS`, and `MODAL_MAX_BATCH`. Note that these
are read at container import, so they must be baked into the image. A bare
`MODAL_GPU=L40S modal deploy` in the deploying shell silently measures the default.

Two constraints travel with the GPU choice. Blackwell is `sm_120` and needs
`torch==2.13.0`. Torch 2.6 fails there with "no kernel image is available". And
`DRIVER_VERSION` must match the host driver of whichever GPU you land on.

## Behaviour Contract & Safeguards

- Resize type options:
	- `scale by multiplier`
	- `target dimensions`
- **Aspect Ratio Preservation**: In `target dimensions` mode, `keep_aspect_ratio` defaults to `true` (fits the output inside the requested bounding box without distortion). Set to `false` (or use `--no-keep-aspect-ratio` in the CLI) to stretch to exact dimensions.
- Output dimensions are aligned to multiples of 8 (minimum 8).
- Quality mapping: `LOW`, `MEDIUM`, `HIGH`, `ULTRA`, plus the rest of the SDK's model
  families (see below).
- **Output Limits & Safeguards**:
	- **Upscaling Only**: Downscaling is rejected (`400`). `scale` must be $\ge 1.0$, and computed target output cannot be smaller than input dimensions.
	- **Maximum Output Edge**: Capped at 16384 pixels. Requests exceeding 16384 on any edge are rejected (`400`) before starting GPU work.
	- **Hybrid VSR Resampling**: Super-resolution runs at the trusted hardware limit ($\le 15360$ px) and resamples up to 16384 via high-precision GPU bicubic interpolation, avoiding the SDK's known blue-channel collapse defect at 16K.
	- **Integrity Checks**: Every frame is checked on GPU for non-finite values (NaN/Inf) and channel collapse before delivery.
- Batch sizing guard preserved using max-pixel heuristic:
	- `MAX_PIXELS = 1024 * 1024 * 16`
- **Attribution**: This service began as a port of the Comfy-Org ComfyUI custom node and
  keeps its resize, quality, alignment and batching behaviour. See [NOTICE](NOTICE) for the
  Apache-2.0 credits to Comfy-Org and `whmc76/ComfyUI-NVIDIA-RTX-VSR-Pro`.

## Prerequisites

- Modal CLI installed and authenticated:
	- `pip install modal`
	- `modal token new`
- NVIDIA GPU runtime in Modal (configured by deployment).
- A proxy auth token pair, since the endpoint requires authentication:
	- `modal workspace proxy-tokens create`
	- Export the pair for the client: `export MODAL_KEY=... MODAL_SECRET=...`

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Deploy to Modal

Default GPU is `RTX-PRO-6000`.

```bash
modal deploy modal_app.py
```

To choose a different GPU at deploy time, set `MODAL_GPU`:

```bash
MODAL_GPU=RTX-PRO-6000 modal deploy modal_app.py
```

GPU guidance, from testing on this app:

- **`RTX-PRO-6000` (default, recommended).** Verified working end to end, and the only
  Modal GPU with 9th-gen NVENC.
- **B200 / B300.** Data-center Blackwell has **no NVENC engine at all**, so video would
  CPU-encode at the highest GPU price on the platform.
- **A100.** No NVENC either.
- **L40S and older.** Not verified for super-resolution here. Any GPU still needs the
  `libnvidia-ngx` fix described above, and older torch wheels lack Blackwell kernels —
  the pinned `torch==2.13.0` is required for `RTX-PRO-6000` (sm_120).

### Deploy-time tuning knobs

These are read when the container image is built and **baked into it**. Setting one in your
shell at request time does nothing — it must be present on the `modal deploy` command:

| Variable | Default | Effect |
|---|---|---|
| `MODAL_GPU_ENCODER` | `1` | Encode from CUDA memory. `0` pipes rawvideo to ffmpeg instead. |
| `MODAL_GPU_DECODER` | `1` | Decode in-process with NVDEC. `0` uses a decode subprocess. |
| `MODAL_NVENC_PRESET` | `p5` | NVENC preset, `p1`–`p7`. |
| `MODAL_NVENC_CQ` | `20` | Quality target, equivalent to NVEncC `--qvbr`. Lower is higher quality and larger files. |
| `MODAL_NVENC_MULTIPASS` | `fullres` | `qres` = 2pass-quarter, `fullres` = 2pass-full. |
| `MODAL_NVENC_TUNING` | `uhq` | `uhq` or `hq`. UHQ is slower and better per bit. |
| `MODAL_NVENC_LEVEL` | `5.1` | HEVC level for outputs up to 4096x2176. Set explicitly so the driver cannot pick one per session. Empty = driver picks. |
| `MODAL_NVENC_MAX_MBPS` | `100` | VBR ceiling in Mbit/s, with a 1-second VBV buffer. `0` skips it and leaves the driver's default ceiling, which caps CQ (see above). |
| `MODAL_NVENC_AQ_STRENGTH` | `10` | Adaptive quantisation strength, 1–15. |
| `MODAL_NVENC_BFRAMES` | `5` | B-frames per GOP. |
| `MODAL_NVENC_REFS` | `5` | Reference frames for the piped ffmpeg encoder (`-refs`). The GPU encoder leaves reference lists to the driver. |
| `MODAL_GPU` | `RTX-PRO-6000` | Worker GPU type. |
| `MODAL_APP_NAME` | `rtx-media-upscaler` | Deploy a variant beside production instead of replacing it. |

To benchmark a variant without disturbing production, give it its own app name and stop it
when finished:

```bash
MODAL_APP_NAME=rtx-bench MODAL_NVENC_PRESET=p6 modal deploy modal_app.py
modal app stop rtx-bench -y
```

After deploy, Modal prints a URL similar to:

`https://<workspace>--rtx-media-upscaler-api.modal.run`

## Choosing a Quality Mode

`nvvfx.VideoSuperRes.QualityLevel` groups its models into four families. All 19 were probed
working on `RTX-PRO-6000` with driver 580.95.05. They split into two roles:

| Parameter | Accepted values | What it does |
|---|---|---|
| `quality` | `BICUBIC`, `LOW`, `MEDIUM`, `HIGH`, `ULTRA` | Upscale. Also suppresses compression artifacts, so it softens already-clean sources. |
| `quality` | `HIGHBITRATE_LOW` … `HIGHBITRATE_ULTRA` | Upscale for clean/high-bitrate sources (ProRes, high-quality H.265). Skips artifact suppression. |
| `preprocess` | `DENOISE_LOW` … `DENOISE_ULTRA` | Optional same-resolution pass before upscaling. Removes macro-blocking and mosquito noise. |
| `preprocess` | `DEBLUR_LOW` … `DEBLUR_ULTRA` | Optional same-resolution pass before upscaling. Sharpens soft or out-of-focus footage. |

The two roles are validated separately: a same-resolution mode in `quality` (or an upscaling
mode in `preprocess`) is rejected with `400` before any GPU work starts.

**Why the default is `HIGHBITRATE_ULTRA`, not `ULTRA`.** The standard family (`LOW`–`ULTRA`)
suppresses compression artifacts as part of upscaling. That helps a genuinely artifact-heavy
encode, but on a source that is *soft yet cleanly encoded* — the common case for broadcast
archive material — there are no real artifacts to remove, so the pass chews on genuine
texture and leaves a blotchy, painterly look. Measured on the reference clip, mode 19 beat
mode 4 on **both** axes at once: ~15% more edge detail and roughly 8x less flat-region noise.
Reach for `ULTRA` only when the source really is a heavily compressed, blocky encode.

**When `preprocess` earns its cost.** It adds a second inference pass per frame, so use it
only when the source is the limiting factor. Measured behaviour on a soft, already-upscaled
SD master, all paired with `HIGHBITRATE_ULTRA`:

| Preprocess | Edge detail | Flat-region noise | Temporal flicker |
|---|---|---|---|
| none | 3.7 | 0.10 | 1.89 (baseline) |
| `DEBLUR_LOW` | 6.5 | 0.11 | +22% |
| `DEBLUR_MEDIUM` | 6.6 | 0.12 | +18% |
| `DEBLUR_HIGH` | 5.9 | 0.12 | +12% |
| `DEBLUR_ULTRA` | 5.6 | 0.09 | +5% |

Two things here are worth internalising, because neither matches what the mode names imply:

- **Intensity is not monotonic.** `DEBLUR_LOW`/`MEDIUM` produce *more* apparent detail than
  `HIGH`/`ULTRA`, not less. The higher rungs appear to regularise more heavily alongside the
  sharpening. Do not assume "ULTRA = strongest".
- **Judge deblur on temporal flicker, not on stills.** Flat-region noise that changes frame to
  frame is far more objectionable in motion than in a screenshot. `DEBLUR_ULTRA` is the
  best-behaved rung for video: it still lifts detail by ~50% over no preprocess while adding
  only 5% flicker, whereas `DEBLUR_LOW` wins on paper and shimmers the most.

Default to no preprocess. If a soft source needs more bite, start at `DEBLUR_ULTRA` and
**check the result in motion** before committing to a batch run.

```bash
python modal_client.py \
	--endpoint "https://<workspace>--rtx-media-upscaler-api.modal.run" \
	--input ./soft_broadcast_source.mkv \
	--scale 2.0 --quality HIGHBITRATE_ULTRA --preprocess DEBLUR_ULTRA
```

## Comparing Results

`scripts/compare_frames.py` builds side-by-side comparison images from an input and an output
file, matching frames by index rather than timestamp:

```bash
uv run --with pillow --with numpy python scripts/compare_frames.py \
	--original ./input.mkv --upscaled ./output.mp4
```

For each frame it writes a full-frame comparison, a centre-crop (where differences are
actually visible at screen zoom), and an amplified difference map showing which pixels the
upscaler changed. The original side is Lanczos-resized to match dimensions and labelled as
such, so the comparison is against the strongest conventional resampler rather than a
flattering baseline.

## Upload and Download From Local Machine

```bash
python modal_client.py \
	--endpoint "https://<workspace>--rtx-media-upscaler-api.modal.run" \
	--input ./input.mp4 \
	--resize-type "scale by multiplier" \
	--scale 2.0 \
	--quality ULTRA
```

Or target exact output dimensions:

```bash
python modal_client.py \
	--endpoint "https://<workspace>--rtx-media-upscaler-api.modal.run" \
	--input ./input.png \
	--resize-type "target dimensions" \
	--width 1920 \
	--height 1080 \
	--quality HIGH
```

The client submits the job, polls until the worker finishes, and saves the output locally (`*_upscaled.png` or `*_upscaled.mp4`) unless `--output` is provided. Tune the wait with `--timeout` and `--poll-interval`. Transient network errors and proxy 502/503/504 answers are retried with backoff. If the client stops anyway, resume the same job with `--call-id <call_id>` (printed at submit time) instead of uploading again. The download goes to `<output>.part` and is renamed only when its size matches `Content-Length`.

## Direct API Usage

Submit a job (returns a `call_id`):

```bash
curl -X POST "https://<workspace>--rtx-media-upscaler-api.modal.run/upscale" \
	-H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" \
	-F "file=@./input.png" \
	-F "resize_type=scale by multiplier" \
	-F "scale=2.0" \
	-F "quality=ULTRA"
```

Poll for the result. While the job runs this returns HTTP 202; on completion it returns the file:

```bash
curl "https://<workspace>--rtx-media-upscaler-api.modal.run/result/fc-XXX" \
	-H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" \
	-o upscaled.png
```

The output is deleted from the Volume once it has been downloaded; a second request returns 410. A failed job's files are deleted when it fails. Errors return JSON with a `detail` message: 400 for invalid parameters, 404 when the result has expired, 500 when upscaling failed.

Health check:

```bash
curl "https://<workspace>--rtx-media-upscaler-api.modal.run/health" \
	-H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET"
```

## Using This as a Template for Other Modal Jobs

The structure here is deliberately generic; only the upscaling logic is domain-specific.

**Keep** — these solve problems any Modal GPU service hits:

- The CPU web tier / GPU worker split, so GPU time is billed only for GPU work.
- The job-queue API, which sidesteps the 150-second HTTP limit on web endpoints.
- The `modal.Volume` file handoff: the web tier uses the Volume API (`batch_upload`, `read_file`, `remove_file`), only the GPU worker mounts it.
- `requires_proxy_auth=True`, plus the client’s submit-and-poll loop.
- `@modal.enter()` warming for anything expensive to initialise.

**Replace** — `UpscaleWorker._upscale_image` / `_upscale_video` and the `nvvfx` calls.

**Reconsider per job:**

- The `DRIVER_VERSION` / `libnvidia-ngx` install is only needed for NGX features such as
  super-resolution. Most GPU jobs can drop it entirely.
- The pinned ffmpeg build matters only if the job encodes video — and re-verify NVENC
  against the host driver of the day, since that pairing is what breaks.
- Rename the app (`rtx-media-upscaler`) and the Volume (`rtx-upscaler-jobs`), or a new
  service will collide with this one in the same workspace.

## Compatibility Notes

- The runtime contract is HTTP upload/download.
- Video output is encoded with NVENC HEVC/H.264 when the GPU encoder is available (falling back to `libx264`), and **every** source audio track is kept. Each track is stream-copied when mp4 can carry its codec (AAC, AC-3, E-AC-3, Opus, ALAC, MP3, FLAC); any other codec (DTS, TrueHD, PCM, ...) is re-encoded to AAC 192k for that track only, with a warning in the worker log. (Before 2026-10-03 only the first audio track was kept, and the GPU-encoder path always re-encoded it to AAC.)
