# FlashVSR-Pro on Modal

FlashVSR-Pro video super-resolution deployed as an authenticated HTTP service on
[Modal](https://modal.com). A CPU-only web tier (`api`) fronts two GPU classes,
`FlashVSRFull` and `FlashVSRTiny`, with job I/O passed through Modal Volumes.

## Layout

| Path | What it is |
|---|---|
| `modal_app/` | The Modal service — this repo's own work. Entry points: `app.py` (App + Volumes), `image.py` (image build), `service.py` (GPU classes), `web_api.py` (HTTP tier), `orchestrator.py`, `video_io.py`. |
| `diffsynth/`, `utils/`, `infer.py`, `setup.py`, `requirements.txt` | Vendored FlashVSR-Pro upstream source. |
| `Block-Sparse-Attention/` | Vendored CUDA kernels, compiled at image-build time. |

The vendored trees are **required at deploy time**: `image.py` pulls them in with
`add_local_dir(...)` / `add_local_file(...)` using paths relative to this directory, so
`modal deploy` must run from here.

## Deploying

```bash
modal deploy -m modal_app.deploy      # from this directory
modal run -m modal_app.probe_gpu      # verify the CUDA kernels actually compute
```

The Block-Sparse-Attention wheel is cached on the `flashvsr-build-cache` Volume, keyed on
kernel-source hash + torch version + arch list. Leave the vendored kernel sources byte-identical
and rebuilds install the cached wheel in seconds; change them and you pay a ~30–40 minute
CUDA compile. A successful build proves the kernels *compiled*, not that they compute the
right thing — run `probe_gpu` after any stack change.

## Volumes

`flashvsr-models` (FlashVSR-v1.1 weights + prompt tensor), `flashvsr-io` (job I/O),
`flashvsr-build-cache` (compiled BSA wheel).

## Provenance — recovered source

The local working copy of this project was destroyed on 2026-08-18, and the
`tinof/FlashVSR-Pro` GitHub repo had already been deleted, so no git history survived.

This tree was recovered from the **deployed Modal image** (`flashvsr-pro` v10, deployed
2026-08-16 15:28 from commit `1adf4ba` plus uncommitted working-tree changes) by booting a
sandbox on that image and extracting `/root/modal_app` and `/workspace/FlashVSR-Pro`.

Consequences worth knowing:

- The recovered state is the **Aug 16 15:28 deploy snapshot**. Any local edit made after
  that deploy and never deployed is not here and is not recoverable.
- The previous `CLAUDE.md` and `README.md` were not part of the deployed source and were
  **not** recovered. This README is newly written from the recovered code.
- Git history (commits `1adf4ba`, `ac40e18`) is gone; this tree starts fresh.

## Attribution

Fork of [FlashVSR-Pro](https://github.com/LujiaJin/FlashVSR-Pro), with vendored
[Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) (see that
directory's `LICENSE`) and DiffSynth-Studio-derived `diffsynth/`. Model weights are
`JunhaoZhuang/FlashVSR-v1.1` with the prompt tensor from `1038lab/FlashVSR`. Upstream
licenses apply to the vendored trees.
