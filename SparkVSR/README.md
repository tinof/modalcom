# SparkVSR on Modal

Serverless deployment of **SparkVSR** (ECCV 2026, CogVideoX1.5-5B-I2V backbone) on Modal cloud GPUs, purpose-built for restoring detail in Finnish broadcast television archives (e.g. Yle master tapes from 2000–2010).

SparkVSR propagates sparse high-quality reference keyframes across a video shot using a single transformer forward pass per chunk, making high-fidelity 5B-parameter super-resolution economically viable on serverless compute.

---

## Key Features & Architecture

- **Cut-Aware Temporal Planning**: Uses PySceneDetect adaptive thresholding to detect scene cuts. Temporal windows (49 frames, 8-frame overlap) are scheduled strictly within shot boundaries, preventing cross-scene detail bleed/ghosting.
- **Reference Keyframe Pipeline**:
  - `pisasr` (Default): In-process open-source image super-resolution using SD 2.1 + PiSA adapter, automatically unloaded before main inference to conserve GPU memory.
  - `api`: External restoration via `fal-ai/nano-banana-pro/edit`.
  - `no_ref`: Reference-free blind super-resolution baseline.
- **Precomputed Prompt Embeddings**: Text encoder (T5-XXL, ~9 GB) is precomputed once during provisioning, eliminating it from GPU VRAM and container cold starts.
- **Probed 10-bit HEVC Hardware Encoding**: Probes NVENC capabilities on container start (`hevc_nvenc` Main 10 → `h264_nvenc` → CPU `libx264`), preserves rational frame rates (`Fraction(fps).limit_denominator(1001)`), tags BT.709 color metadata, and stream-copies source audio without re-encoding.
- **Asynchronous Web Tier**: Slim CPU image with FastAPI streaming uploads directly to Modal IO Volume before spawning GPU workers.

---

## Directory Structure

```
SparkVSR/
├── modal_app/
│   ├── __init__.py
│   ├── app.py                 # Modal App and Volume definitions
│   ├── config.py              # Configuration constants, bounds, and validator
│   ├── deploy.py              # Modal deploy entrypoint
│   ├── download_weights.py    # One-shot model weight provisioning & T5 embed calculation
│   ├── image.py               # GPU and slim CPU Image definitions
│   ├── service.py             # GPU worker class (RTX-PRO-6000)
│   ├── web_api.py             # Authenticated FastAPI web service
│   └── pipeline/
│       ├── __init__.py
│       ├── core.py            # Vendored CogVideoX SparkVSR inference core
│       ├── cutplan.py         # PySceneDetect cut detection and window planning
│       ├── pisasr.py          # PiSA-SR reference keyframe generator
│       ├── refgen.py          # Multi-mode reference dispatcher & caching
│       └── video_io.py        # Video decoding, probed NVENC encoding, audio copy
├── tests/
│   └── test_cutplan.py        # Unit tests for cut-aware planner (CPU, no GPU needed)
├── sparkvsr_client.py         # Local CLI client for interacting with deployed API
├── run_sample.py              # Local entrypoint smoke test over sample video
└── sample/                    # Test clips (jopet_10s.mkv, jopet_60s.mkv)
```

---

## Setup & Deployment

### 1. Provision Model Weights

Downloads `JiongzeYu/SparkVSR` (~22 GB) and the PiSA-SR SD 2.1 base weights (~5 GB), and precomputes the empty-prompt T5 embedding, into the `sparkvsr-models` Volume:

```bash
modal run -m modal_app.download_weights
```

Requires a Modal Secret named `huggingface` containing `HF_TOKEN`.

SD 2.1 base comes from `sd2-community/stable-diffusion-2-1-base`, not from
Stability's own repo — `stabilityai/stable-diffusion-2-1-base` was delisted from
the Hub and now 404s even with a valid token. Only the fp32 subfolder
safetensors are pulled (`unet`, `vae`, `text_encoder`, plus the tokenizer and
scheduler configs), which is what `PiSASR_eval` loads; the single-file
checkpoints and fp16 variants in that repo are skipped.

#### PiSA-SR adapter weights (only for `--ref-mode pisasr`)

`pisa_sr.pkl` (~32 MB) cannot be fetched automatically: upstream distributes it
through Google Drive only, and the HuggingFace mirror requires authentication.
Download it once from the link in the [PiSA-SR README](https://github.com/csslc/PiSA-SR),
then upload it to the Volume:

```bash
modal volume put sparkvsr-models /path/to/pisa_sr.pkl pisasr/pisa_sr.pkl
```

If you have a direct URL, set `PISASR_WEIGHTS_URL` in a Secret on the
provisioning function and it will be fetched for you instead.

The `api` and `no_ref` reference modes need none of this — provisioning
succeeds and containers start without the adapter, and only a `pisasr` job
fails, with instructions.

### 2. Run Local Smoke Test

Execute a full end-to-end smoke test on a 10-second sample clip:

```bash
modal run run_sample.py --input-file sample/jopet_10s.mkv --target-height 2160 --ref-mode pisasr
```

### 3. Deploy Web Service

Deploy the authenticated web API and GPU service to Modal:

```bash
modal deploy -m modal_app.deploy
```

---

## CLI Client Usage

Generate proxy auth tokens via `modal workspace proxy-tokens create`, then set:

```bash
export MODAL_KEY="your-modal-key"
export MODAL_SECRET="your-modal-secret"
```

Submit a video for restoration:

```bash
python sparkvsr_client.py \
    --endpoint https://<workspace>--sparkvsr-api.modal.run \
    --input sample/jopet_10s.mkv \
    --output sample/jopet_10s_4k.mp4 \
    --target-height 2160 \
    --ref-mode pisasr
```

### CLI Options & Tuning Knobs

| Flag | Default | Description |
|---|---|---|
| `--target-height` | `2160` | Vertical resolution (e.g. 2160 for 4K, 1080 for same-res refresh) |
| `--target-width` | `None` | Horizontal resolution (aspect ratio is preserved automatically) |
| `--ref-mode` | `pisasr` | Reference generation mode: `pisasr`, `api`, or `no_ref` |
| `--ref-guidance` | `1.0` | Guidance strength (> 1.0 enables classifier-free guidance, doubling forward passes) |
| `--no-cut-aware` | `False` | Disable cut detection and fall back to fixed overlapping windows |
| `--no-audio` | `False` | Omit stream-copying audio from input video |
| `--tile` | `False` | Enable spatial tiling for extreme resolutions (> 4K) |
| `--tile-size` | `512` | Spatial tile dimension in pixels when `--tile` is active |
| `--chunk-size` | `49` | Temporal window frame length (must be `8n+1`) |
| `--overlap` | `8` | Overlap frame count between consecutive windows in the same shot |

---

## Performance & Cost Profile

- **GPU Type**: NVIDIA RTX PRO 6000 (Blackwell, 96 GB VRAM)
- **Model cost shape**: SparkVSR is single-step — one transformer forward per
  49-frame window, plus a VAE encode and decode. That is what makes a 5B model
  affordable here. Setting `--ref-guidance` above 1.0 enables classifier-free
  guidance and doubles the forward passes.

### Measured on `sample/jopet_10s.mkv`

301 frames (12.0 s) of 1920x1080 25 fps source, default `--chunk-size 49` /
`--overlap 8`, 6 windows across 3 shots, no spatial tiling. Container: 8 CPU
cores, 32 GiB RAM, one RTX PRO 6000.

| Run | Peak VRAM | Reference gen | Inference | Total | Cost |
|---|---|---|---|---|---|
| 1080p, `no_ref` | 21.5 GB | — | 376 s | 384 s | ~$0.43 |
| 1080p, `pisasr` | 25.9 GB | 38 s | 380 s | 429 s | ~$0.48 |
| 4K, `pisasr` | 49.1 GB | 122 s | 1852 s | 1992 s | ~$2.07 |

**1080p→4K costs roughly $10.30 and 2.8 hours of wall clock per minute of
output.** Inference dominates: 93% of the 4K run. Cost is GPU-bound
(RTX PRO 6000 at $0.000842/s, ~83% of the total; the rest is CPU and memory)
and excludes container start and the ~6 s model load.

4K fits in 96 GB with wide headroom at 49.1 GB peak, so `--tile` stays off by
default — spatial tiling is only needed above 4K or with a larger
`--chunk-size`. Doubling resolution cost 4.9x the inference time, so the run is
compute-bound rather than memory-bound.

### What the reference architecture buys

At 1080p, Laplacian variance measured 23 on the source, 77–106 with `no_ref`,
and 163–211 with `pisasr`. References roughly double the restored detail over
blind super-resolution, for 38 s of reference generation on a 12 s clip.

### Cut awareness

On `sample/jopet_60s.mkv` (1551 frames), the planner's shot boundaries — 727,
836 and 1195 — match the clip's three hard cuts exactly, with no false
positives at the default adaptive threshold of 3.0. Correlating each restored
frame's residual against the outgoing scene shows the difference cut awareness
makes: with `--no-cut-aware` the previous scene bleeds into the new one at
+0.55 correlation, decaying over 5–8 frames; with cut awareness the same
measure is flat at its mid-shot baseline.

---

## Performance Tuning

Inference is 98% of wall clock at 1080p (decode, upscale, blend and encode together are
under 2%), so every knob below targets the transformer forward. All are environment
variables, so a run can A/B one knob without a code change.

| Variable | Default | Effect |
|---|---|---|
| `SPARKVSR_TORCH_COMPILE` | `1` | `torch.compile` the transformer. Costs a one-off graph build on a container's first window |
| `SPARKVSR_TORCH_COMPILE_MODE` | `max-autotune-no-cudagraphs` | Compile mode |
| `SPARKVSR_TORCH_COMPILE_VAE` | `0` | Also compile VAE decode. Recompiles more often because window length varies |
| `SPARKVSR_FP8` | `0` | FP8 dynamic quantisation of the transformer's linears (torchao, Blackwell-native). The only knob that can change output quality |
| `SPARKVSR_PROFILE` | `0` | Emit a `torch.profiler` CUDA breakdown for the first window |
| `SPARKVSR_PARALLEL` | `1` | Fan segments across GPU workers |
| `SPARKVSR_SEGMENT_FRAMES` | `500` | Frames per parallel segment |
| `SPARKVSR_MAX_CONTAINERS` | `10` | Ceiling on simultaneous GPU workers |

### Parallelism

Segments are cut only on shot boundaries and no window ever straddles a shot, so segments
share no state: no halo frames, no cross-worker blending. `process_video_parallel` plans on
a **CPU** container (giving the driver a GPU would bill an idle card while it waits on
workers), fans segments out with `.map()`, and concatenates with `-c copy`.

This divides **wall clock**, not cost — ten workers cost the same as one worker running ten
times as long. The single-GPU knobs above are what reduce the bill. A 60-minute episode at
one shot per ~4 s yields ~180 segments, so the practical limit is your Modal plan's GPU
concurrency (10 on Starter, 50 on Team), not the content.

A single unbroken shot cannot be split, so it stays on one worker.

### Safety guards on the parallel path

Fan-out multiplies the cost of a silent defect, so the driver checks what it can for free
before and after spending GPU time:

- **CPU pre-flight before dispatch** — segments must tile `[0, total_frames)` contiguously,
  and every window must fall inside exactly one segment. A plan defect fails instantly at
  no cost instead of surfacing in a worker once the rest of the fleet has been billed.
- **Output frame count is probed on the finished file.** The in-flight counter only sees
  frames pushed into ffmpeg's stdin, which is upstream of every way the muxer can still
  lose them — the `-shortest` truncation that cost 5 frames twice, and B-frame or timebase
  mismatches across an episode's ~180 concat joins.
- **Segment ordering is asserted**, because a permutation is the one corruption a frame
  count cannot detect: the sum is identical for any order.
- **Encoder agreement is asserted.** Each worker probes NVENC independently and falls back
  to 8-bit libx264 on a transient failure; concatenating mixed codecs with `-c copy` yields
  a file most decoders cut short at the switch.
- **A segment too large for worker memory fails before the GPU starts**, which is what an
  unbroken shot longer than the segment target would otherwise cause.
- Workers retry twice, so one preemption does not discard the whole fleet's finished work,
  and the driver has its own multi-hour timeout since it outlives any single worker.

### Reading the timings

Every run logs `per-window inference [...] | first Xs, steady-state median Ys`. Judge
optimisations on the **steady-state median**: the first window carries the compile warmup,
which on a short clip can exceed the savings and make a real win look like a regression.

### Validating a change cheaply

`sample/bench_5s.mkv` (126 frames, 3 windows) is the A/B unit — one warmup window plus two
steady-state ones, for roughly $0.25 a run instead of $0.45 for the 12 s clip:

```bash
ffmpeg -y -i sample/jopet_10s.mkv -t 5 -c copy sample/bench_5s.mkv

# Baseline, then one knob at a time. --no-parallel isolates inference from fan-out.
SPARKVSR_TORCH_COMPILE=0 modal run run_sample.py --input-file sample/bench_5s.mkv \
    --ref-mode no_ref --no-parallel --output-file sample/bench_base.mp4
modal run run_sample.py --input-file sample/bench_5s.mkv \
    --ref-mode no_ref --no-parallel --output-file sample/bench_compile.mp4
```

Compare the steady-state medians. For `SPARKVSR_FP8=1`, also check quality against the
baseline output — Laplacian variance should stay within a few percent and there should be
no visible artefacts, since FP8 is the one knob that can alter the result.

### Validation budget

At $0.001018/s for the worker (GPU $0.000842 + 8 cores + 32 GiB = $3.66/hr) and ~63 s of
inference per 1080p window, the whole ladder costs about **$4**:

| Step | Clip | Cost | Answers |
|---|---|---|---|
| 1. Profile | bench 5 s | $0.24 | Where inference time actually goes |
| 2. Baseline | bench 5 s | $0.24 | Steady-state window time, eager |
| 3. Compile | bench 5 s | $0.36 | Compile speedup and warmup cost |
| 4. FP8 | bench 5 s | $0.36 | FP8 speedup and quality delta |
| 5. Parallel smoke | 10 s, `--segment-frames 100` | $0.52 | Fan-out correctness: 3 workers, 301/301 frames |
| 6. Full validation | 60 s | $2.28 | Best config end to end, 1551/1551 frames |

Run steps 2–4 before 5–6: there is no point validating fan-out with a configuration you are
about to change. The driver container is CPU-only at $0.63/hr, so its wait time is
negligible next to the workers.
