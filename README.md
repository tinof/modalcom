# modalcom

GPU video-processing services deployed on [Modal](https://modal.com), one directory per
project. Each project is self-contained: its own `README.md`, its own `CLAUDE.md` where the
constraints are non-obvious, and its own Modal app and Volumes.

| Project | What it does | Modal app |
|---|---|---|
| [`rtx-vsr/`](rtx-vsr/) | NVIDIA RTX Video Super Resolution (NGX) as an authenticated job queue. Fully GPU-resident NVDEC → VSR → NVENC path. | `rtx-media-upscaler` |
| [`flashvsr-pro/`](flashvsr-pro/) | FlashVSR-Pro diffusion video super-resolution, full and tiny tiers. | `flashvsr-pro` |
| [`ltx2/`](ltx2/) | LTX-2 text-to-video, two-stage generate-then-upscale. | `ltx2` |

## Conventions

- **One directory per project, kebab-case, matching its folder name locally and here.**
- **Deploy from the project directory** (`cd rtx-vsr && modal deploy modal_app.py`) — image
  definitions reference local paths relative to it.
- **Pin container dependencies in the image definition**, not in a root requirements file.
  Any `requirements.txt` / `pyproject.toml` in a project covers only its local client.
- **Nothing large goes in git.** Sample clips, outputs, and model weights are gitignored;
  they live in the working tree or on Modal Volumes. See the root `.gitignore`.
- **Secrets never go in git.** ggshield runs as a pre-commit and pre-push hook.

## Local-only projects

These exist in the working tree but are deliberately not committed yet: `sdr2hdr/`,
`SwiftVR/`, `SparkVSR/`. They are unpublished, not lost — but they are also not backed up
by this repo.
