# Dev Handoff: FlashVSR+ segment worker

Last session: 2026-11-25. Written for whoever picks this up next, assuming no memory of how it got here.

## Where things stand

The FlashVSR+ worker (`modal_app/plus_*.py`, `client.py`) is written, reviewed and tested locally. **It has never been built, deployed or run on Modal, and no frame has been upscaled.** That was the owner's decision for the session: code only, no Modal spend, and the test run will use a file the owner supplies. Do not upload or process the Selviytyjät S06E02 episode that the session started from.

Branch `flashvsr-plus-worker` on top of `main`, pushed, draft PR #13 (https://github.com/tinof/modalcom/pull/13). The repo has no `.claude/workflow.md`, so `/ship:land` cannot run here; the PR waits for the owner or for `/ship:roadmap-init`.

`README.md` is the reference: cost estimate, design, runbook, and an exact list of what is and is not tested. Read it first. The approved plan is `~/.claude/plans/can-you-review-our-encapsulated-melody-of-1195689c-741d-47aa-908a-63818cd43d83.md`.

What the tests prove (375 without torch, 418 with torch and an upstream checkout):

- The planner delivers every frame exactly once, for every case that matters.
- The whole video path except the model works on real ffmpeg with `libx265` standing in for NVENC: frame-exact decode from any start frame, segments, a cross-faded seam, the join, colour within one 8-bit level, audio and subtitles carried over.
- `plus_runtime.py` drives upstream's real `StreamRunner` (pinned commit `b515ffc`) correctly, with stub models in place of the weights: stock and resident transport return identical frames, and a whole job plans, renders, joins and resumes.

What they do not prove: anything on a GPU. See "Not tested at all" in the README.

## What is left, in order

### 1. One free local check that was cut short

A dry run of the planner over a whole real file. It was started on the S06E02 episode and stopped by the owner while the scene pass was still decoding (it takes several minutes on this 4-core ARM host). It costs nothing and answers a real question: **the scene threshold of 0.30 is unmeasured on real edits.** One synthetic cut scored 0.77; that says nothing about a reality-TV edit.

Run it on any real file the owner agrees to, not necessarily the test file:

```bash
mkdir -p /tmp/dry/inputs && ln -s /path/to/real.mkv /tmp/dry/inputs/ep.mkv
cd flashvsr-pro && python3 -c "
from modal_app import plus_job
from modal_app.plus_plan import Plan
job = plus_job.prepare_job('/tmp/dry', 'dry', 'inputs/ep.mkv')
plan = Plan.from_json(job['plan'])
print(len(plan.segments), 'segments,', len(plan.seams), 'soft seams')
"
```

Read from it: the scene pass frame count equals the packet scan (the call fails otherwise), how many seams are soft, and the score distribution in `inputs/ep.mkv.scenes.json`. Many soft seams on material with frequent cuts means the threshold is too high. Then put the result in the README where it says the planner has not run over a whole real file. The scene cache is written beside the symlink, not beside the real file.

### 2. The first Modal run (needs the owner's go-ahead and their test file)

Follow "Runbook for a first run" in the README step by step. It has two stops where the owner decides: after the look test (is FlashVSR+ worth 20 to 60 times the cost of `rtx-vsr` on this material, which preset and scale) and after the speed benchmark (the measured cost per episode). Expected spend up to the first stop: about $10 to $15.

Expect step 0 to fail somewhere. In likely order:

1. **Image build.** Upstream is untested by its authors on torch 2.12 (its requirements say `<2.8`; it did import and step on CPU torch 2.15 in our tests). The Block-Sparse-Attention build for arch `120` alone is new; the last build was `80;86;120`. Fallback stack is named in `plus_image.py`.
2. **`modal_app/plus_image.py` inside the container.** It calls `add_local_dir("Block-Sparse-Attention")` at import. The image's workdir is set so that relative path exists there, on the belief that Modal may resolve it at import. Unverified either way.
3. **NVENC through this ffmpeg build.** The pinned BtbN n8.1 build is proven in `rtx-vsr` on driver 580. Modal also has 610 hosts. `plus_probe` and the worker's start-up both run a real encode and fail loudly.
4. **`plus_runtime.build_pipeline`.** Weight paths, the `v1` projector conversion in `plus_weights.py`, and the attention kernel upstream selects (`plus_probe` prints it; it should be `block_sparse`).
5. **The resident transport on real tensors.** `ResidentTinyDecoder` keeps the decoded segment on the GPU: 112 frames of bf16 at a 4K canvas is about 5.6 GB, twice that during the concat. If VRAM is tight, lower `PLUS_DECODE_EVERY` in `config.py` (a multiple of 4).

`modal run -m modal_app.plus_probe` after every image build. It must print no `FAILED`.

### 3. Things to settle before a full-length run

- **Driver generations.** Segments from 580 and 610 hosts may get different HEVC parameter sets. The join refuses them and names the segment; nothing re-encodes it. The README's runbook has a gate for this (a 10-minute excerpt across at least two hosts). If it trips, the options are to drop the explicit level, pin the worker to one driver generation if Modal allows it, or re-encode the odd segments.
- **Memory for `--no-resident`.** Estimated 45 to 55 GB of host RAM at a 4K canvas, by arithmetic only. That benchmark row needs `FLASHVSR_PLUS_MEMORY_MIB=65536` at deploy.
- **The scene threshold**, from item 1.

## Decisions already made (do not reopen without a reason)

- **New app beside the old one.** `flashvsr-plus` is separate from `flashvsr-pro` (v1.1). The vendored `diffsynth/`, `utils/`, `infer.py` stay; upstream's `v1` preset covers the same weights, so removing them is a later decision for the owner.
- **Upstream is cloned at a pinned commit, not vendored.** `plus_runtime.py` replaces two module-level functions of upstream's runner by name and subclasses its `TinyDecoder`. Move `PLUS_REPO_SHA` only together with a re-read of `inference/runners/stream.py` and `inference/decoders/tiny.py`, and re-run `tests/test_plus_runtime.py`.
- **The `rtx-vsr` levers do not transfer.** There NVENC is the bound; here the DiT is, at about 1 fps, and one NVENC session is 20 to 60 times faster. No GPU-resident encode, no parallel NVENC sessions. What transfers is keeping tensors on the GPU inside the model's runner.
- **Piped ffmpeg for decode and encode**, with `rtx-vsr`'s measured flags and encoder settings.
- **Cuts on scene changes, cross-fade where there is none.** The source's GOP is fixed (2.0 s on the episode that was probed), so keyframes are not scene cuts and a segment may start on any frame.
- **No silent repair.** A frame count that differs from the plan fails the segment. No CPU encoder fallback.
- **No benchmark module that runs upstream's own CLI.** The look test goes through the worker. Upstream's CLI centre-crops and writes untagged 8-bit H.264, which cannot be compared with `rtx-vsr` frame for frame.

## Reviews done

- Plan: Gemini 4 Neon (high) through `agy`, and the built-in advisor. Both changed the plan; the plan file's last section records how.
- Diff: Gemini 4 Neon (high) through `agy`, one whole-diff pass (727 s, completed). Findings and what happened to each:
  - `plus_weights.py` image without `add_local_python_source`: fixed.
  - `SegmentEncoder.abort()` could leave the writer thread blocked: fixed, with a test.
  - CPU torch resolved from PyPI instead of the CPU index; weights committed only at the end: both fixed.
  - `overlap_frames=0` produced zero-frame seams: fixed, with a test.
  - A negative seek in `look-test`: that code is gone (see next point).
- The built-in advisor's last pass found that `look-test` cut excerpts by stream copy, which leaves timestamp holes with B-frames, and the worker then refuses the file. Reproduced on the real episode (3 of 4 starts failed). `look-test` now cuts lossless frame-exact excerpts, verified pixel-identical to the source.
- Not done: a review of the final state after those fixes.

## Environment notes

- This host has no GPU, no torch and no `modal` in its system Python. `rtx-vsr/.venv` has `modal` 1.6.0 and serves for import checks:
  `../rtx-vsr/.venv/bin/python -W error -c "import modal_app.plus_deploy, modal_app.plus_probe, modal_app.plus_weights, modal_app.deploy, modal_app.probe_gpu"`
- The torch tests need an upstream checkout; the one used last session was in a session scratch directory and is gone. Re-create it as the top of `tests/test_plus_runtime.py` describes.
- `client.py` needs `modal` and `numpy`: `uv run --with modal --with numpy python client.py ...`.
- Modal workspace state on 2026-11-25: apps `rtx-media-upscaler` and `hlg-manifest` deployed; no `flashvsr-*` app and none of its Volumes exist.
