## 1. Confirm the defect on our hardware

- [x] 1.1 Write a throwaway `modal run` probe (scratchpad, not committed) that upscales a
      synthetic image with signal in all three channels to 16384x16384 on `RTX-PRO-6000`, and
      prints the per-channel means of both input and output.
- [x] 1.2 Repeat the probe at 15360x15360 to confirm the trusted limit is clean.
- [x] 1.3 Record both results (channel means, reproduced or not) in `design.md` under Open
      Questions, and pick the collapse threshold from the measured gap.

## 2. Dimension planning

- [x] 2.1 Add a pure dimension planner to `modal_app.py` returning the final output size and
      the super-resolution size, absorbing `_compute_output_size` and reusing
      `_aligned_dimension`; it must take a `keep_aspect_ratio` argument.
- [x] 2.2 Implement aspect-ratio fitting (fit inside the requested box) for target-dimension
      mode, applied before 8-alignment.
- [x] 2.3 Enforce the limits in the planner: reject a computed output smaller than the input
      in either dimension, and reject a computed edge above 16384, using distinct error types
      so callers can be mapped to the right status code.
- [x] 2.4 Return a super-resolution size capped at the 15360 trusted limit, leaving the final
      output size unchanged, so the worker can tell when a hybrid resize is needed.

## 3. Request validation and API surface

- [x] 3.1 Add `keep_aspect_ratio: bool = Form(True)` to `POST /upscale` and thread it through
      `spawn()` into `UpscaleWorker.run` and both `_upscale_image` / `_upscale_video`.
- [x] 3.2 Validate request-only rules in the endpoint before `spawn()` — resize type,
      `scale >= 1.0`, positive target dimensions, requested edge within 16384 — returning
      `400` with no GPU container started.
- [x] 3.3 Map the worker's upscale-only error to `400` in `GET /result/{call_id}`, preserving
      the message and reporting both input and requested output size; keep integrity
      failures as `500`.

## 4. Output integrity

- [x] 4.1 Add a per-frame integrity check that reduces channel statistics on the GPU and
      raises when an output channel has collapsed while the corresponding input channel
      carried signal, or when the frame contains NaN/Inf.
- [x] 4.2 Call it for every frame in `_upscale_rgb_batch`, with a message naming the affected
      channel and the output size.
- [x] 4.3 Replace the incidental copy in `_upscale_rgb_batch` with an explicit `.clone()` of
      the DLPack tensor, commented with the SDK's copy-before-next-call requirement.

## 5. Hybrid resize

- [x] 5.1 In `_upscale_rgb_batch`, when the planned output differs from the super-resolution
      size, resample the float tensor on the GPU with
      `torch.nn.functional.interpolate(mode="bicubic")` up to the planned size before the
      uint8 conversion.
- [x] 5.2 Log once per job when the hybrid path is taken, naming both sizes, so the fallback
      is never silent.
- [x] 5.3 Free GPU memory at the end of `UpscaleWorker.run` (`gc.collect()` plus
      `torch.cuda.empty_cache()`) without discarding the cached super-resolution instance.

## 6. Client and documentation

- [x] 6.1 Add a `--no-keep-aspect-ratio` flag to `modal_client.py` and send the field.
- [x] 6.2 Add a `NOTICE` file crediting whmc76/ComfyUI-NVIDIA-RTX-VSR-Pro and the upstream
      Comfy-Org node, both Apache-2.0, and reference it from `README.md`.
- [x] 6.3 Document the new parameter, the size limits, and the `400` cases in `README.md`.
- [x] 6.4 Record in `CLAUDE.md` why the hybrid path and the integrity check exist, including
      the probe result from Task 1.3, so neither is later removed as redundant.

## 7. Verification

- [x] 7.1 `uvx ruff check --select F,E9 modal_app.py modal_client.py` passes and
      `modal deploy modal_app.py` succeeds.
- [x] 7.2 Regression: the image (800x400 -> 1600x800) and video (320x240 -> 640x480, 48
      frames, audio intact, `hevc_nvenc` in the worker log) round-trips still pass.
- [x] 7.3 `keep_aspect_ratio=true` with a mismatched box returns fitted, undistorted
      dimensions; `false` reproduces the previous stretched result.
- [x] 7.4 A downscale request and an over-16384 request each return `400`, and the
      over-limit one starts no GPU container.
- [x] 7.5 If Task 1 reproduced the defect: a 16384-target job now returns an image whose
      three channel means are all non-zero — the case that previously returned corrupt output.
