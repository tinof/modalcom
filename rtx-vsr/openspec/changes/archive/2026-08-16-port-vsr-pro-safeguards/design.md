## Context

See `proposal.md` — Why. Design-relevant facts about the current implementation:

- `_compute_output_size()` computes dimensions **inside the GPU worker**, after the media has
  been decoded. The web tier never sees the input dimensions, which constrains where each
  validation rule can live (see Decisions).
- `_upscale_rgb_batch()` runs the engine per frame and converts straight to CPU `uint8`.
  Its `.clamp(0.0, 1.0)` allocates a new tensor, so the SDK's DLPack buffer is currently
  copied *by accident*; changing that one call to an in-place `clamp_` would silently
  corrupt output.
- The worker is an `@app.cls()` whose container persists across jobs and caches the loaded
  super-resolution instance, so large allocations survive between requests.
- `_batch_size_for_output()` already collapses to a batch of 1 for very large outputs via the
  `MAX_PIXELS` heuristic, so the hybrid path never has to hold many huge frames at once.
- `CLAUDE.md` requires the Modal path to mirror the reference ComfyUI node's dimension
  alignment (multiples of 8). That constraint is load-bearing for the sizing design below.

## Goals / Non-Goals

**Goals:**

- Make a corrupt engine result an explicit failure rather than a delivered file.
- Deliver the planned output size even above the trusted super-resolution limit.
- Turn unservable requests into `400` responses, cheaply and as early as the information
  allows.
- Keep the DLPack copy and the between-job memory release explicit and commented, so neither
  can be removed by accident.

**Non-Goals:**

- Changing the 8-pixel alignment rule, or delivering pixel-exact unaligned sizes.
- Raising the 16384 ceiling, or supporting downscaling.
- Print-size (mm + DPI) input, per `proposal.md`.
- Reworking the batching heuristic.

## Decisions

### Split validation between the web tier and the worker

Rules that depend only on the request are validated in the `/upscale` endpoint, before
`spawn()`: invalid resize type, `scale < 1.0`, non-positive target dimensions, and a
requested edge above 16384. These return `400` with no GPU container started, which is what
the maximum-size rule is for.

The upscale-only rule depends on the *input* dimensions, which are known only after decoding
in the worker. Rather than decode twice, the worker raises a dedicated error type that
`/result` maps to `400` with the message intact. It is still a client error; it is just
detected later.

*Alternative considered*: probing dimensions in the web tier with Pillow and `ffprobe`. This
would give a uniformly early `400`, but it pulls a decoder and ffmpeg into the slim web image
and re-reads every upload. Rejected as disproportionate.

### Plan dimensions once, in one pure function

A single planner returns both the final output size and the size super-resolution should run
at. Keeping it pure (no torch, no SDK) makes every sizing rule in the spec unit-testable
without a GPU, and gives the endpoint and the worker one shared source of truth.

The planner applies, in order: mode-specific sizing, aspect-ratio fitting, 8-alignment,
then the limit checks. Alignment runs before the limit checks so an aligned-up dimension
cannot slip past the ceiling.

### Integrity check compares output against input

Checking only the output would flag legitimate images — a photo of a red wall genuinely has
a near-zero blue channel. The check therefore only reports collapse when the input channel
carried signal and the corresponding output channel did not. Statistics are reduced on the
GPU before the frame is transferred, so the cost is a few reductions per frame rather than a
second host round-trip.

*Alternative considered*: sampling only the first frame of a video. Rejected — corruption is
a property of the engine's configured output size, so it would either affect all frames or
none, but per-frame checking costs little and catches the case where that assumption is wrong.

### Hybrid resize happens on the GPU, before the uint8 conversion

When the planned output exceeds the trusted limit, the engine runs at the trusted size and
`torch.nn.functional.interpolate(mode="bicubic")` scales the float tensor up to the planned
size while it is still on the device. Resampling in float avoids a second quantisation step,
and staying on the GPU avoids moving the larger buffer over PCIe.

## Risks / Trade-offs

- **The corruption may not reproduce on Linux + Blackwell** → Task 1 probes real hardware
  before the workaround is written. If it does not reproduce, the hybrid path and the check
  ship anyway as insurance, but the tasks record that finding so nobody treats the threshold
  as tuned against observed data.
- **False positives on legitimately near-monochrome media** → mitigated by the
  input-versus-output comparison above; the threshold is chosen against probe data rather
  than guessed, and the failure message names the channel so a false positive is diagnosable.
- **Interpolating the top 6.67% is not true super-resolution** → unavoidable given the
  engine's limit, and strictly better than returning a corrupt image. Applies only above
  15360 px.
- **The hybrid path holds two large tensors briefly** (about 2.8 GB plus 3.2 GB at 16K) →
  acceptable on the 96 GB default GPU, and the existing `MAX_PIXELS` heuristic already forces
  a batch of 1 at those sizes.
- **`keep_aspect_ratio` defaulting to `true` changes existing behaviour** → called out as
  BREAKING in the proposal; `false` restores the old result exactly. The service is new and
  its only known caller is our own client.
- **Per-frame checking adds work to long videos** → reductions on already-resident GPU data;
  expected to be lost in the noise next to the upscale itself. Worth measuring in Task 1's
  probe if a long video regresses.

## Migration Plan

1. Land the planner and validation with the probe result recorded.
2. Deploy with `modal deploy modal_app.py`; the web and GPU images both rebuild.
3. Verify the regression round-trips from `CLAUDE.md`'s verification protocol still pass
   before exercising the new paths.

Rollback is a redeploy of the previous revision — no persisted state changes, and the jobs
Volume layout is untouched, so in-flight jobs drain normally.

## Open Questions

- **Channel collapse probe results (Task 1 probe on RTX-PRO-6000, Linux, driver 580.95.05):**
  - At **15360x15360** (trusted limit): Input means `[0.5, 0.5, 0.5]` -> Output means `[0.4998, 0.5007, 0.5012]`. All channels intact and clean, no NaN/Inf.
  - At **16384x16384** (untrusted limit): Input means `[0.5, 0.5, 0.5]` -> Output means `[0.4998, 0.5007, 0.0]`. The Blue channel collapses completely to `0.0` across the entire frame.
  - **Collapse threshold chosen**: If an input channel mean is $\ge 0.01$ (contains signal) and the corresponding output channel mean is $< 0.001$ (or output mean / input mean $< 0.01$), the frame is considered collapsed and raises an integrity error.

