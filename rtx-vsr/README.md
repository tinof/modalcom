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

- **Web tier** (`api`): a CPU-only FastAPI app. It streams the upload into a Modal Volume, spawns the GPU job, and serves the finished file. Handles up to 20 concurrent requests per container.
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

Video jobs run **entirely on the GPU**: NVDEC decodes in-process, super-resolution runs on
the decoded CUDA tensor, and NVENC encodes straight from GPU memory. A frame is created in
video memory and never touches the host until the finished packet comes back out. Output is
**HEVC Main 10** (`yuv420p10le`) tagged bt709 for matrix, primaries and transfer.

Encoder settings mirror the reference NVEncC invocation
`--qvbr 18 --codec h265 --tune uhq --output-depth 10 --profile main10 --tier high
--multipass 2pass-quarter --aq --aq-strength 10 --bframes 5 --ref 5` at preset **P4**.
10-bit encoding is used even for 8-bit sources because it costs nothing on NVENC and keeps
upscaled gradients from banding.

Measured on RTX PRO 6000, 1080p→4K, `HIGHBITRATE_ULTRA`, warm containers:

| Data path | fps |
|---|---|
| rawvideo pipes to/from ffmpeg (original) | 12.1 |
| GPU encode, piped decode | 34.2 |
| **fully GPU-resident (current default)** | **≈59** |

Each job logs a line like `Video done: 301 frames in 5.1s = 58.8 fps (decode-wait 1%,
infer 74%, encode 11%; nvdec=in-process, batch=2, gpu-encoder=yes)` — useful for spotting a
regression or a silent fallback. The binding stage is now super-resolution inference; the
encoder sits around 11%.

Two escape hatches exist for inputs the GPU path cannot take. `MODAL_GPU_DECODER=0` falls
back to a decode subprocess (this also happens automatically for codecs NVDEC cannot
handle), and `MODAL_GPU_ENCODER=0` falls back to piping frames to ffmpeg. Both fallbacks
use the same encoder quality point, and every degradation is logged with its cause.

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
| `MODAL_NVENC_PRESET` | `p4` | NVENC preset, `p1`–`p7`. |
| `MODAL_NVENC_CQ` | `18` | Quality target, equivalent to NVEncC `--qvbr`. Lower is higher quality. |
| `MODAL_NVENC_MULTIPASS` | `qres` | `qres` = 2pass-quarter, `fullres` = 2pass-full. |
| `MODAL_NVENC_AQ_STRENGTH` | `10` | Adaptive quantisation strength, 1–15. |
| `MODAL_NVENC_BFRAMES` | `5` | B-frames per GOP. |
| `MODAL_NVENC_REFS` | `5` | Reference frames. |
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

The client submits the job, polls until the worker finishes, and saves the output locally (`*_upscaled.png` or `*_upscaled.mp4`) unless `--output` is provided. Tune the wait with `--timeout` and `--poll-interval`.

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

The output is deleted from the Volume once it has been downloaded.

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
- The `modal.Volume` file handoff and chunked upload streaming.
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
- Video output is encoded with NVENC HEVC/H.264 when the GPU encoder is available (falling back to `libx264`), and the source audio track is copied through unchanged.
