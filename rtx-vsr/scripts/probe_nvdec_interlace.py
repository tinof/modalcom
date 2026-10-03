"""Does in-process NVDEC deinterlace interlace-flagged PsF frames?

Decodes the same clip two ways and compares row parity:
  * PyNvVideoCodec CreateDecoder(outputColorType=RGB), exactly as _upscale_video_gpu does
  * ffmpeg software decode, woven (no deinterlacer), converted with bt709/tv to rgb24

Adaptive deinterlacing rewrites one field's rows wherever its motion detector fires, so
the difference concentrates in odd (or even) rows. Pure colour-conversion rounding is
spread evenly over both parities.

    modal volume put rtx-upscaler-jobs <interlace-flagged clip>.mkv /probe-interlace/input.mkv
    modal run scripts/probe_nvdec_interlace.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("torch==2.13.0", "PyNvVideoCodec==2.2.0", "numpy==2.5.2")
)
app = modal.App("rtx-probe-interlace", image=image)
jobs_volume = modal.Volume.from_name("rtx-upscaler-jobs")
CLIP = "/jobs/probe-interlace/input.mkv"
N = 300


@app.function(gpu="RTX-PRO-6000", timeout=1200, volumes={"/jobs": jobs_volume})
def probe() -> str:
    import subprocess

    import numpy as np
    import PyNvVideoCodec as nvc
    import torch

    w, h = 1920, 1080
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", CLIP, "-frames:v", str(N),
         "-vf", "scale=in_color_matrix=bt709:in_range=tv:flags=bicubic+accurate_rnd+full_chroma_int",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    ref = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)

    demuxer = nvc.CreateDemuxer(filename=CLIP)
    decoder = nvc.CreateDecoder(gpuid=0, codec=demuxer.GetNvCodecId(), usedevicememory=1,
                                outputColorType=nvc.OutputColorType.RGB)
    got = []
    for packet in demuxer:
        for frame in decoder.Decode(packet):
            got.append(torch.from_dlpack(frame).clone().cpu().numpy())
            if len(got) >= N:
                break
        if len(got) >= N:
            break

    n = min(len(got), len(ref))
    lines = [f"frames: nvdec={len(got)} ffmpeg={len(ref)} compared={n}"]
    even, odd, big_e, big_o, vdet_n, vdet_f = [], [], [], [], [], []
    for i in range(n):
        a = got[i].astype(np.float32).mean(2)
        b = ref[i].astype(np.float32).mean(2)
        d = np.abs(a - b)
        even.append(d[0::2].mean()); odd.append(d[1::2].mean())
        big_e.append((d[0::2] > 6).mean()); big_o.append((d[1::2] > 6).mean())
        vdet_n.append(np.abs(np.diff(a, axis=0)).mean()); vdet_f.append(np.abs(np.diff(b, axis=0)).mean())
    lines.append(f"mean |diff| even rows {np.mean(even):.3f}  odd rows {np.mean(odd):.3f}")
    lines.append(f"pixels |diff|>6: even {100*np.mean(big_e):.3f}%  odd {100*np.mean(big_o):.3f}%  "
                 f"worst frame odd {100*np.max(big_o):.2f}%")
    lines.append(f"vertical gradient: nvdec {np.mean(vdet_n):.3f}  ffmpeg-weave {np.mean(vdet_f):.3f}  "
                 f"ratio {np.mean(vdet_n)/np.mean(vdet_f):.3f}")
    worst = int(np.argmax(big_o))
    lines.append(f"worst frame {worst}: even {even[worst]:.2f} odd {odd[worst]:.2f}")
    return "\n".join(lines)


@app.local_entrypoint()
def main() -> None:
    print(probe.remote())
