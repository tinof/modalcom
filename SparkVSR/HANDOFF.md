# Dev Handoff — SparkVSR Optimization & Validation

Last session: 2026-08-19. Written for whoever picks this up next, assuming no memory of
how it got here.

## What changed most recently

A research pass on how others speed up one-step VSR on a CogVideoX backbone produced three
things worth acting on, all now in the code and none yet measured:

- **SageAttention is in** and is the highest-expected-value knob, benchmarked upstream on
  this project's exact backbone. See "SageAttention is now wired in" below for the two
  non-obvious details of how it had to be integrated.
- **VAE slicing/tiling is now a knob** rather than unconditional. FastVSR (arXiv 2509.24142,
  also one-step VSR on CogVideoX1.5) reports that once the denoiser runs a single time, the
  VAE codec dominates the forward — so a memory optimisation we were paying for
  unconditionally, on a card with 70 GB of headroom at 1080p, is a prime suspect.
- **INT8 weight-only quantisation was ruled out**, not merely skipped: the CogVideoX model
  card states it *reduces* inference speed and exists to save VRAM. Not a lever here.

Also confirmed: the step-caching family (TeaCache, PAB) is structurally inapplicable, since
a one-step model has no timestep loop to cache across. Do not let it back onto the list.

The environment section at the bottom is stale in one respect and the blocker below explains
it: the `tinof` workspace is currently disabled.

## Where things stand

The pipeline **works and is verified** end to end. Earlier sessions took it from
never-executed code to real renders: 1080p and 4K on `sample/jopet_10s.mkv`, a 60 s
cut-aware run, and a client round-trip through the deployed web endpoint. Eight defects
were found and fixed along the way, all recorded in `openspec/changes/add-sparkvsr-modal-service/tasks.md`
sections 9 and 10.

The last session applied a full optimization workstream (section 11 of the same file).
**None of it has run on a GPU** — Modal credits ran out first. So the code is written,
statically validated, reviewed and unit-tested, but every performance claim in it is an
estimate.

Do not treat the optimizations as working because they were reviewed. The first
steady-state timing off a real GPU is the first evidence.

### Measured baseline (real, from GPU runs)

| Config | Wall clock | Peak VRAM | Cost |
|---|---|---|---|
| 1080p→1080p, `no_ref` | 384 s | 21.5 GB | $0.43 |
| 1080p→1080p, `pisasr` | 429 s | 25.9 GB | $0.48 |
| 1080p→4K, `pisasr` | 1992 s | 49.1 GB | $2.07 |

That is 0.7–0.8 fps at 1080p and 0.15 fps at 4K, on a 12.0 s / 301-frame clip.
Inference is **98%** of wall clock at 1080p; decode, upscale, blend and encode together
are under 2%. Worker rate is $0.001018/s ($3.66/hr) for GPU + 8 cores + 32 GiB.

**The project has now committed to 1080p→1080p restoration as the primary workflow** —
same resolution, no upscale. It recovers 7–9x the source's high-frequency detail
(Laplacian variance 23 → 163–211) at ~4.6x less GPU time than upscaling to 4K. Defaults
in `config.py` reflect this; 4K is still available via `--target-height 2160`.

## Blocker: the `tinof` workspace has exceeded its spend limit

As of 2026-08-19 no GPU run can start. `modal run` fails with:

```
Workspace ac-V3ivE6YpiVVa1aMm8dXDW8 has exceeded its spend limit
```

(An earlier attempt surfaced the same condition as `ConflictError: workspace ... is
disabled`, so both messages mean this.) It is an account state, not a code fault — raise the
limit under Settings → Spend limit in the Modal dashboard for the `tinof` workspace.

Nothing was billed for GPU: the failure lands after the image build and before any container
is allocated. Volume *reads* still succeed while in this state, so `modal volume list`
working is not evidence that the workspace can run anything — test with an actual `modal
run`, which now fails in under a minute against the cached image.

The image build itself **succeeded**, including the new SageAttention kernels
(`sageattention ok /usr/local/lib/python3.12/site-packages/sageattention/__init__.py`,
built in 148 s), so that layer is cached and the next attempt starts from a warm image.

Two ways forward:

- **Re-enable `tinof`.** Nothing to re-provision: the 22 GB SparkVSR weights, the SD 2.1
  base and the operator-supplied `pisa_sr.pkl` are already on its volumes. This is the
  cheap path and the assumed one.
- **Move to the `ilias-fotiou` workspace**, which was checked and is enabled and writable.
  This costs a re-provision: ~27 GB re-downloaded from HuggingFace via
  `modal run -m modal_app.download_weights`, plus a manual
  `modal volume put sparkvsr-models ./pisa_sr.pkl pisasr/pisa_sr.pkl` — that file has no
  public programmatic source, but a local copy exists at the repo root.

Every command below assumes `MODAL_PROFILE=tinof`; the local default profile is
`ilias-fotiou`, and running the ladder under it without re-provisioning fails on missing
volumes.

## Your first job: run the validation ladder (~$4)

Everything else is blocked on this. The ladder answers whether the optimizations help
and by how much, for about $4 of GPU time.

### Step 0 — recreate the benchmark clip (free)

`sample/bench_5s.mkv` is gitignored, so it may not exist:

```bash
cd ~/Documents/GitHub/modalcom/SparkVSR
ffmpeg -y -i sample/jopet_10s.mkv -t 5 -c copy sample/bench_5s.mkv
ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=nb_read_frames -of csv=p=0 sample/bench_5s.mkv   # expect 126
```

126 frames is 3 windows: one compile-warmup window plus two steady-state ones. That is
the whole point — it makes each A/B cost ~$0.25 instead of ~$0.45.

Also confirm the unit tests still pass (CPU, free, no Modal needed):

```bash
PYTHONPATH=. uvx --with modal --with pytest pytest tests/ -q    # expect 20 passed
```

### How to read every result

Each run logs a line like:

```
per-window inference [78.2, 61.4, 62.0] | first 78.2s, steady-state median 62.0s
```

**Judge every optimization on the steady-state median, never the total.** `torch.compile`
pays a one-off graph build on a container's first window; on a short clip that warmup can
exceed the savings and make a genuine win look like a regression. The baseline to beat is
**~62.7 s per 1080p window**.

### Step 1 — profile (~$0.24)

```bash
SPARKVSR_PROFILE=1 SPARKVSR_TORCH_COMPILE=0 modal run run_sample.py \
    --input-file sample/bench_5s.mkv --ref-mode no_ref --no-parallel \
    --output-file sample/bench_profile.mp4
```

Prints a CUDA operator table for window 0. Record how inference splits across VAE encode,
the transformer forward, attention, and VAE decode. **This decides whether the remaining
steps are worth running at all** — if attention dominates, FP8 and compile are the right
levers; if something unexpected dominates, stop and re-plan rather than working the ladder.

Note the first image build will also pull `torchao` (newly added), so this run includes a
one-off image rebuild of a few minutes. Build compute is cheap but not free.

### Step 2 — eager baseline (~$0.24)

```bash
SPARKVSR_TORCH_COMPILE=0 modal run run_sample.py \
    --input-file sample/bench_5s.mkv --ref-mode no_ref --no-parallel \
    --output-file sample/bench_base.mp4
```

Record the steady-state median. Sanity-check it lands near 62.7 s; if it is far off, the
image or hardware changed and the historical baseline no longer applies.

### Step 3 — torch.compile (~$0.36)

```bash
modal run run_sample.py --input-file sample/bench_5s.mkv \
    --ref-mode no_ref --no-parallel --output-file sample/bench_compile.mp4
```

Compile is **on by default**, so no env var is needed. Record both numbers: the first
window (warmup cost) and the steady-state median (the actual win).

Decision rule: keep compile if the steady-state median improves by more than ~5%. Also
note the warmup, because it is paid once **per container** — with fan-out across 10
workers you pay it 10 times, so a 2-minute warmup against a 3-minute segment is a bad
trade. If warmup is large, raise `SPARKVSR_SEGMENT_FRAMES` so each worker amortizes it
over more windows.

### Step 4 — VAE tiling off (~$0.30)

```bash
SPARKVSR_VAE_TILING=0 modal run run_sample.py --input-file sample/bench_5s.mkv \
    --ref-mode no_ref --no-parallel --output-file sample/bench_notile.mp4
```

VAE slicing and tiling were enabled unconditionally before anything was measured. They buy
peak memory at the cost of throughput, and 1080p peaked at 25.9 GB against this card's 96 GB
— so there was probably nothing to buy. This is also the knob FastVSR's finding points at:
in a one-step model the denoiser runs once, which pushes the VAE codec's share of the
forward up sharply.

Numerically identical output, so no quality gate. Keep it off if it is faster **and** peak
VRAM stays comfortably under 96 GB. Note that 4K peaked at 49.1 GB *with* tiling on, so this
decision may not carry from 1080p to 4K — re-check before changing the 4K default.

### Step 5 — SageAttention (~$0.36)

```bash
SPARKVSR_ATTENTION_BACKEND=sage modal run run_sample.py \
    --input-file sample/bench_5s.mkv --ref-mode no_ref --no-parallel \
    --output-file sample/bench_sage.mp4
```

Stack this on whatever steps 3–4 selected. Confirm the log line
`SageAttention installed over torch SDPA` appears — without it the kernels did not import
and the run is measuring nothing. Quality gate below applies (INT8 QK, FP8 PV).

### Step 6 — FP8 (~$0.36)

```bash
SPARKVSR_FP8=1 modal run run_sample.py --input-file sample/bench_5s.mkv \
    --ref-mode no_ref --no-parallel --output-file sample/bench_fp8.mp4
```

### Quality gate for steps 5 and 6

Both change numerics, so both need a frame-level check, not just a timing check. Compare the
run's output against `bench_base.mp4` (substitute `bench_sage.mp4` for the SageAttention
run):

```bash
python3 - <<'EOF'
import subprocess, numpy as np, cv2
W,H=1920,1080
def frames(p, idx=(10,60,100)):
    raw=subprocess.run(['ffmpeg','-v','error','-i',p,'-pix_fmt','gray','-f','rawvideo','-'],
                       capture_output=True).stdout
    f=np.frombuffer(raw,dtype=np.uint8).reshape(-1,H,W)
    if len(f)>1 and np.abs(f[1].astype(np.int16)-f[0]).mean()==0.0: f=f[1:]  # ffmpeg dup first frame
    return {i:f[i] for i in idx}
a,b=frames('sample/bench_base.mp4'),frames('sample/bench_fp8.mp4')
for i in a:
    lap=lambda x: cv2.Laplacian(x.astype(np.float32),cv2.CV_32F).var()
    print(i, 'lap base %.1f fp8 %.1f'%(lap(a[i]),lap(b[i])),
          'mean|diff| %.2f'%np.abs(a[i].astype(np.int16)-b[i]).mean())
EOF
```

Keep the knob only if Laplacian variance stays within a few percent **and** mean absolute
difference is small (single-digit on 0–255). Otherwise leave it off; the speed is not worth
degrading the detail recovery that is this project's whole point. SageAttention's paper
claims under 0.2% end-to-end metric loss, which is a reason to test it, not to trust it.

### Step 7 — parallel smoke (~$0.52)

First real exercise of the fan-out path. **Nothing in it has ever executed.**

```bash
modal run run_sample.py --input-file sample/jopet_10s.mkv \
    --ref-mode pisasr --segment-frames 100 \
    --output-file sample/out_parallel.mp4
```

`--segment-frames 100` forces the 301-frame clip to split into 3 segments at its shot
boundaries (frames 0, 152, 206), so you get real fan-out on a cheap clip. Expect 3 workers.

Then verify the output independently — the in-pipeline check is not enough on its own:

```bash
ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=nb_read_frames -of csv=p=0 sample/out_parallel.mp4   # MUST be 301
ffprobe -v error -select_streams v:0 -of default=nw=1 \
    -show_entries stream=codec_name,pix_fmt,color_space,r_frame_rate sample/out_parallel.mp4
```

Pass criteria: 301 frames, `hevc`, `yuv420p10le`, `bt709`, `25/1`, AAC audio present.
Then compare against `sample/out_1080_pisa.mp4` from the earlier session — the parallel
output should be equivalent, since segments split only on shot boundaries.

### Step 8 — full validation (~$2.28)

With whatever configuration steps 2–6 selected:

```bash
modal run run_sample.py --input-file sample/jopet_60s.mkv \
    --ref-mode pisasr --output-file sample/out_60s_optimized.mp4
```

Must produce **1551 frames**. Compare wall clock and steady-state median against the
recorded 1912 s / 1551-frame baseline from the previous session.

### Step 9 — close the loop

Update the README's cost table and "Performance & Cost Profile" with measured numbers, then
check off 11.11 and 11.12 in `openspec/changes/add-sparkvsr-modal-service/tasks.md`. After
that the change is ready for `/opsx:verify` and archiving.

## What to do if the fan-out misbehaves

The parallel path has guards that fail loudly rather than shipping a broken file. If you
hit one, this is what it means:

| Error | Cause |
|---|---|
| `Window N ... straddles a segment boundary` / `past the video's N frames` | Cut plan disagrees with the decord frame count. Pre-flight caught it on CPU, so it cost nothing. Check `plan.json` on the io volume |
| `Workers disagreed on the video encoder` | A worker's NVENC probe fell back to libx264. Re-run; if it repeats, pin the encoder in the payload |
| `Encoded output has N frames but the source has M` | Frames lost in muxing or concat — the exact class of bug that hit twice before. Do not paper over it |
| `The longest segment is N frames` | An unbroken shot too large for a worker's RAM. See the known limit below |
| `FileNotFoundError: Job input file not found` | A volume reload gap. Both known ones are fixed; a new one means a new code path reads the volume without reloading |

Useful commands:

```bash
modal app list                                   # is anything still running / burning?
modal volume ls sparkvsr-io jobs                 # per-job dirs
modal volume get sparkvsr-io jobs/<job>/plan.json .   # inspect the cut plan
pkill -f "modal run"                             # stop local drivers
modal app stop <app-id>                          # stop a running app
```

## Known limits and open risks

**A single unbroken shot cannot be parallelized.** Segment boundaries are only ever shot
boundaries, because splitting mid-shot puts a join between two sides that denoised
independently — a visible seam. Continuous takes, screen recordings and some animation
therefore get one worker and no speedup, and above ~2600 frames at 1080p they now fail
fast on CPU rather than OOM on a billed GPU. If the Yle masters contain long unbroken
takes, this needs a real fix: per-window on-demand decode via decord random access, which
removes the whole-segment uint8 hold. That is the highest-value piece of remaining work
after the ladder.

**The 2–3x speedup is an estimate.** It is the product of unverified compile, FP8 and
SageAttention assumptions. Do not quote it to anyone until steps 3–6 produce numbers.

**Parallelism divides wall clock, not cost.** Ten workers cost the same as one worker
running ten times as long. Only the single-GPU knobs reduce the bill. Worth restating
whenever someone asks why the invoice did not move.

**SageAttention is now wired in** (`SPARKVSR_ATTENTION_BACKEND=sage`), reversing the earlier
decision to skip it. The blocker was believed to be sm_120 wheel availability; that was
stale. Upstream's `setup.py` has an explicit `12.0 -> sm_120a` branch, requires nvcc >= 12.8,
and honours `TORCH_CUDA_ARCH_LIST`, so it builds on a GPU-free builder. The GPU image
therefore moved to a CUDA 13.0.1 *devel* base — the same base flashvsr-pro already runs on
this GPU — and compiles the kernels for sm_120 only.

Two things about that wiring are worth not rediscovering:

- It replaces `F.scaled_dot_product_attention`, **not** `set_attention_backend`.
  `CogVideoXAttnProcessor2_0` calls SDPA directly and never reaches diffusers' attention
  dispatcher, so setting a dispatcher backend on this transformer succeeds and does nothing.
  That failure mode is invisible — it looks exactly like "SageAttention gave no speedup".
- The replacement is scoped to the transformer forward via `sage_attention()`. PiSA-SR runs
  in the same process *after* the pipeline loads, so an unscoped patch would also quantise
  the reference keyframes and contaminate the very comparison used to judge the knob.

**Cost estimates come from Modal's published rates**, not from an invoice. Check the
billing page once real runs land.

## Environment notes

- Modal profile `tinof`; `huggingface` secret exists, no `fal` secret (so `--ref-mode api`
  is unavailable and degrades gracefully).
- Volumes `sparkvsr-models` and `sparkvsr-io` are provisioned, including the
  operator-supplied `pisa_sr.pkl`. No need to re-provision.
- SD 2.1 base comes from `sd2-community/stable-diffusion-2-1-base`; Stability delisted the
  original repo and it 404s even with a valid token.
- `modal run` needs the Modal CLI's own Python; bare `python3` lacks the `modal` module.
  Run tests with `PYTHONPATH=. uvx --with modal --with pytest pytest tests/ -q`.
- **The whole `SparkVSR/` directory is still untracked in git.** Nothing has been committed.
  Consider an initial commit before making further changes, so the optimization work is
  recoverable and reviewable as a diff.
- ffmpeg's rawvideo decode of these masters emits a **duplicate first frame** (the stream
  starts at pts 0.028). Drop frame 0 before comparing source to output, or you will
  rediscover a phantom off-by-one at scene cuts. The snippets in this file already do.

## Tuning knobs

All are environment variables read in `modal_app/config.py`, so you can A/B one without a
code change. Full table with defaults is in the README under "Performance Tuning".

`SPARKVSR_TORCH_COMPILE`, `SPARKVSR_TORCH_COMPILE_MODE`, `SPARKVSR_TORCH_COMPILE_VAE`,
`SPARKVSR_FP8`, `SPARKVSR_PROFILE`, `SPARKVSR_PARALLEL`, `SPARKVSR_SEGMENT_FRAMES`,
`SPARKVSR_MAX_CONTAINERS`, `SPARKVSR_DRIVER_TIMEOUT`, `SPARKVSR_WORKER_RETRIES`.

## Key files

| File | Role |
|---|---|
| `modal_app/service.py` | `SparkVSRService` (GPU worker), `process_segment`, `process_video_parallel` (CPU driver), `_preflight_plan` |
| `modal_app/pipeline/core.py` | Inference; `SparkVSRPipeline.load`, `_apply_inference_optimizations`, `process_video_ref_i2v` |
| `modal_app/pipeline/executor.py` | Streaming window execution, blending, per-window timing, profiler hook |
| `modal_app/pipeline/video_io.py` | Decode, encoder ladder, BT.709, audio, concat. Both `-shortest` bugs lived here |
| `modal_app/pipeline/cutplan.py` | Scene detection and window planning |
| `modal_app/config.py` | All defaults and tuning knobs |
| `tests/` | 20 CPU-only tests (cut planning, segmentation, pre-flight) |
