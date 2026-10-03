"""Standalone probe: is the in-process NVDEC RGB surface actually what we think it is?

Process of elimination. The GPU encode path produced alternating-column striping on every
frame (mean |Laplacian| ~192 against ~2 for the source), and scripts/probe_p010_encode.py
cleared every stage downstream of the decoder: the flat uint16 P010 handoff, the
__dlpack__ shim, batching, and production's non-contiguous CHW->HWC tensor provenance all
encode and decode cleanly. The one component the *clean* piped run never touched is this
one -- it logged `nvdec=yes` (ffmpeg hwaccel), while every striped run logged
`nvdec=in-process`.

So this compares the same frames decoded two ways, with no encoder anywhere:

  * in-process NVDEC into CUDA memory, outputColorType=RGB, as _upscale_video_gpu does
  * ffmpeg to rgb24, the reference

CLAUDE.md claims the RGB output is "tightly packed HWC uint8 with no pitch padding". If
that is wrong -- a row pitch, a planar layout, an alpha channel -- this prints it.

    modal run scripts/probe_nvdec_rgb.py

Expects the clip at /jobs/probe-nvdec/input.mkv on the rtx-upscaler-jobs Volume:
    modal volume put rtx-upscaler-jobs sample/jopet_10s.mkv /probe-nvdec/input.mkv
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("torch==2.13.0", "PyNvVideoCodec==2.2.3", "numpy==2.5.2")
)

app = modal.App("rtx-probe-nvdec-rgb", image=image)
jobs_volume = modal.Volume.from_name("rtx-upscaler-jobs", create_if_missing=True)

CLIP = "/jobs/probe-nvdec/input.mkv"
CHECK_FRAME = 40


def _lap(gray):
    import numpy as np

    g = gray.astype(np.float32)
    return float(np.abs(
        4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    ).mean())


@app.function(gpu="RTX-PRO-6000", timeout=1800, volumes={"/jobs": jobs_volume})
def probe() -> None:
    import subprocess

    import numpy as np
    import PyNvVideoCodec as nvc
    import torch

    # --- reference: ffmpeg, tightly packed rgb24 -----------------------------------
    meta = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", CLIP],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    width, height = (int(x) for x in meta.split(",")[:2])
    print(f"source: {width}x{height}")

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", CLIP, "-frames:v", str(CHECK_FRAME + 1),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    ref = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)[CHECK_FRAME]
    ref_gray = ref.mean(axis=2)
    print(
        f"ffmpeg   rgb24     shape={ref.shape} mean={ref_gray.mean():7.2f} "
        f"lap={_lap(ref_gray):7.2f}"
    )

    # --- in-process NVDEC, exactly as _upscale_video_gpu builds it ------------------
    demuxer = nvc.CreateDemuxer(filename=CLIP)
    decoder = nvc.CreateDecoder(
        gpuid=0, codec=demuxer.GetNvCodecId(), usedevicememory=1,
        outputColorType=nvc.OutputColorType.RGB,
    )
    got = None
    index = 0
    for packet in demuxer:
        for decoded in decoder.Decode(packet):
            if index == CHECK_FRAME:
                got = torch.from_dlpack(decoded).clone()
                break
            index += 1
        if got is not None:
            break

    if got is None:
        print("NVDEC produced no frame at that index")
        return

    print(f"nvdec    tensor    shape={tuple(got.shape)} dtype={got.dtype}")
    arr = got.cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        nv_gray = arr[..., :3].mean(axis=2)
    else:
        nv_gray = arr.reshape(arr.shape[0], -1)[:, :width].astype(np.float32)
    print(f"nvdec    as HWC    mean={nv_gray.mean():7.2f} lap={_lap(nv_gray):7.2f}")

    # If the surface is really planar (RGBP) rather than interleaved, reading it as three
    # stacked planes is what recovers a sane image -- and this is the check that tells us.
    flat = arr.reshape(-1)
    if flat.size >= height * width * 3:
        planar = flat[: height * width * 3].reshape(3, height, width).astype(np.float32)
        print(
            f"nvdec    as planar mean={planar.mean():7.2f} "
            f"lap={_lap(planar.mean(axis=0)):7.2f}"
        )
        # A row pitch would show up as each row being offset from the last.
        for pitch in (width * 3, width * 4):
            rows = flat.size // pitch
            if rows >= height:
                view = flat[: rows * pitch].reshape(rows, pitch)[:height, : width * 3]
                g = view.reshape(height, width, 3).astype(np.float32).mean(axis=2)
                print(f"nvdec    pitch={pitch:<6} mean={g.mean():7.2f} lap={_lap(g):7.2f}")

    # Correlate against the reference so a layout that merely *looks* smooth is not
    # mistaken for the right one.
    if nv_gray.shape == ref_gray.shape:
        print(f"mean abs diff vs ffmpeg: {np.abs(nv_gray - ref_gray).mean():.3f}")


@app.local_entrypoint()
def main() -> None:
    probe.remote()
