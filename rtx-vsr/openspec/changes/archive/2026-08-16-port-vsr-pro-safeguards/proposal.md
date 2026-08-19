# Port output safeguards from ComfyUI-NVIDIA-RTX-VSR-Pro

## Why

The [whmc76/ComfyUI-NVIDIA-RTX-VSR-Pro](https://github.com/whmc76/ComfyUI-NVIDIA-RTX-VSR-Pro)
fork of our upstream ComfyUI node documents a defect in the NVIDIA VFX SDK: at large output
sizes the super-resolution effect **silently returns a corrupted image** — their report is a
near-zero blue channel on every `4096x4096 -> 16384x16384` run, while `15360x15360` stayed
clean. Our Modal service would hand that corrupted file to the caller with no error at all.

The same fork also surfaces gaps our HTTP API has today: it accepts targets *smaller* than
the input even though VSR can only upscale, it enforces **no upper bound** on output size
(so a caller can request 50000x50000 and exhaust GPU memory), and `target dimensions`
silently distorts images by stretching to a non-matching aspect ratio.

## What Changes

- Detect corrupted super-resolution output (collapsed colour channel, NaN/Inf) and fail the
  job loudly instead of returning a broken image.
- Route requests above the largest verified VSR size through a hybrid path: run VSR at the
  verified size, then resample to the exact requested dimensions.
- Reject requests VSR cannot satisfy with `400`, not a `500` or a garbage image:
  targets smaller than the input, and targets above the maximum supported edge.
- **BREAKING**: add `keep_aspect_ratio` to `POST /upscale`, **defaulting to `true`**. Callers
  using `resize_type=target dimensions` with a mismatched aspect ratio currently get a
  stretched image; they will now get an undistorted image fitted inside the requested box.
  Passing `keep_aspect_ratio=false` restores the previous behaviour.
- Copy the SDK's DLPack output buffer explicitly, and release GPU memory between jobs — our
  worker container is long-lived and caches its model, so large allocations otherwise linger.
- Add a `NOTICE` file crediting the upstream projects, as Apache-2.0 requires.

### Decisions made without user input

These were chosen when a clarifying question went unanswered; they are cheap to revisit.

- `keep_aspect_ratio` defaults to `true` (matches the Pro node, prevents silent distortion).
- The Pro node's print-size mode (mm + DPI) is **out of scope** — an HTTP caller computes
  `mm / 25.4 * dpi` in one line, so it buys API surface rather than capability.
- The corruption bug is **verified on our own hardware before** the workaround is written,
  rather than ported on faith. It was reported on Windows; we run Linux + Blackwell.

## Capabilities

### New Capabilities

- `media-upscaling/output-sizing`: how the service turns a resize request into concrete
  output dimensions — scale and target modes, aspect-ratio handling, upscale-only and
  maximum-size limits, alignment, and the choice of a VSR intermediate size.
- `media-upscaling/output-integrity`: guarantees about the pixels the service returns —
  detection of corrupted SDK output, and exact-size delivery via the hybrid resize path.

### Modified Capabilities

None. This is the first change in the repository, so no spec exists yet to modify.

## Impact

- **`modal_app.py`**: `_compute_output_size` and `_aligned_dimension` are absorbed into a
  dimension planner; `UpscaleWorker._upscale_rgb_batch` gains the integrity check and the
  hybrid resize; `_upscale_image` / `_upscale_video` pass the new parameter through; the
  `/upscale` endpoint gains a form field and maps the new errors to `400`.
- **`modal_client.py`**: a `--no-keep-aspect-ratio` flag.
- **API consumers**: the `keep_aspect_ratio` default is a behaviour change for
  `target dimensions` callers (see BREAKING above).
- **Docs**: `README.md` (new parameter, error cases) and `CLAUDE.md` (why the hybrid path
  exists, so it is not later "simplified" away).
- **Licensing**: new `NOTICE` file; both projects are Apache-2.0, so reuse is permitted with
  attribution.
- **No new dependencies** — the hybrid resize uses `torch.nn.functional.interpolate`.
