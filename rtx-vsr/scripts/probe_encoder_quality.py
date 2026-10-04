"""Encoder settings vs quality: same VSR frames, one encoder per container, lossless reference.

    modal volume put rtx-upscaler-jobs <clip>.mkv /probe-quality/input.mkv
    modal run scripts/probe_encoder_quality.py    # writes scripts/q_<label>.mp4, skips existing

Then score each against the reference locally, e.g.
    ffmpeg -i q_<label>.mp4 -i q_ref_lossless.mp4 -lavfi \
      "libvmaf=model=version=vmaf_4k_v0.6.1:feature=name=psnr:n_subsample=3" -f null -

Every CQ variant restores the VBV ceiling with Reconfigure(), as modal_app.py does. Without
that, the driver's default ceiling caps every variant at the same size and CQ means nothing.
The lossless reference is ~2.6 GB for 300 frames of 4K.
"""

from pathlib import Path

import modal

DRIVER_VERSION = "580.95.05"
DRIVER_RUN_URL = (
    f"https://us.download.nvidia.com/XFree86/Linux-x86_64/{DRIVER_VERSION}"
    f"/NVIDIA-Linux-x86_64-{DRIVER_VERSION}.run"
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "xz-utils", "libsm6", "libxext6", "libxrender1", "libglib2.0-0")
    .run_commands(
        f"curl -fsSL -o /tmp/nv.run {DRIVER_RUN_URL}",
        "sh /tmp/nv.run --extract-only --target /tmp/nvx",
        f"cp /tmp/nvx/libnvidia-ngx.so.{DRIVER_VERSION} /usr/lib/x86_64-linux-gnu/",
        f"ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ngx.so.{DRIVER_VERSION}"
        " /usr/lib/x86_64-linux-gnu/libnvidia-ngx.so.1",
        "ldconfig",
        "rm -rf /tmp/nv.run /tmp/nvx",
    )
    .uv_pip_install("numpy==2.5.2", "torch==2.13.0", "nvidia-vfx==0.1.0.1", "PyNvVideoCodec==2.2.3")
)
app = modal.App("rtx-probe-quality", image=image)
vol = modal.Volume.from_name("rtx-upscaler-jobs")
CLIP = "/jobs/probe-quality/input.mkv"
N = 300

BASE = dict(codec="hevc", tier="high", level="5.1", aq="10", temporalaq="1", lookahead="32", bf="5",
            gop="250", rc="vbr")
VARIANTS = {
    "ref_lossless": dict(preset="P4", tuning_info="lossless", rc="constqp", constqp="0",
                         aq=None, temporalaq=None, lookahead=None, bf="0"),
}
for cq in ("18", "24"):
    VARIANTS |= {
        f"p4q_uhq_cq{cq}": dict(preset="P4", tuning_info="uhq", multipass="qres", cq=cq),
        f"p5f_hq_cq{cq}": dict(preset="P5", tuning_info="high_quality", multipass="fullres", cq=cq),
        f"p5f_uhq_cq{cq}": dict(preset="P5", tuning_info="uhq", multipass="fullres", cq=cq),
        f"p6f_uhq_cq{cq}": dict(preset="P6", tuning_info="uhq", multipass="fullres", cq=cq),
        f"p7f_uhq_cq{cq}": dict(preset="P7", tuning_info="uhq", multipass="fullres", cq=cq),
    }


@app.function(gpu="RTX-PRO-6000", cpu=6, memory=32768, timeout=1800,
              volumes={"/jobs": vol}, single_use_containers=True)
def encode(label: str) -> tuple[str, bytes, float]:
    import os
    import time

    import nvvfx
    import PyNvVideoCodec as nvc
    import torch

    orig = torch.Tensor.__dlpack__
    torch.Tensor.__dlpack__ = lambda self, *a, **k: orig(self)

    demux = nvc.CreateDemuxer(filename=CLIP)
    dec = nvc.CreateDecoder(gpuid=0, codec=demux.GetNvCodecId(), usedevicememory=1,
                            outputColorType=nvc.OutputColorType.RGB)
    sr = nvvfx.VideoSuperRes(nvvfx.VideoSuperRes.QualityLevel.HIGHBITRATE_ULTRA)
    sr.__enter__()
    sr.output_width, sr.output_height = 3840, 2160
    sr.load()

    def to_p010(rgb):  # production's _rgb_to_p010
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        cb = (b - y) / 1.8556
        cr = (r - y) / 1.5748
        y10 = (64.0 + 876.0 * y).round().clamp(0, 1023)
        cb10 = (512.0 + 896.0 * cb.unfold(0, 2, 2).unfold(1, 2, 2).mean(dim=(-2, -1))).round().clamp(0, 1023)
        cr10 = (512.0 + 896.0 * cr.unfold(0, 2, 2).unfold(1, 2, 2).mean(dim=(-2, -1))).round().clamp(0, 1023)
        h, w = y10.shape
        packed = torch.empty((h * 3 // 2, w), dtype=torch.int32, device=rgb.device)
        packed[:h] = y10.to(torch.int32)
        uv = packed[h:].view(h // 2, w // 2, 2)
        uv[..., 0] = cb10.to(torch.int32)
        uv[..., 1] = cr10.to(torch.int32)
        return (packed * 64).to(torch.uint16)

    kw = {**BASE, **VARIANTS[label]}
    kw = {k: v for k, v in kw.items() if v is not None}
    enc = nvc.CreateEncoder(3840, 2160, "P010", False, **kw)
    rc_note = ""
    if "cq" in kw:
        # PyNvVideoCodec zeroes maxBitRate whenever cq is set, and the driver then
        # substitutes a ~32 Mbps ceiling at L5.1. Lift it before the first frame.
        rp = enc.GetEncodeReconfigureParams()
        before = (rp.maxBitRate, rp.vbvBufferSize, rp.targetQuality)
        rp.maxBitRate = 100_000_000
        rp.vbvBufferSize = 160_000_000
        ok = enc.Reconfigure(rp)
        after = enc.GetEncodeReconfigureParams()
        rc_note = f" reconfigure={ok} before={before} after={(after.maxBitRate, after.vbvBufferSize, after.targetQuality)}"
    out = f"/tmp/{label}.mp4"
    mux = nvc.FFmpegMuxer(out, nvc.MP4, "hevc", 3840, 2160, 25, 1, 1, 90000, enc.GetSequenceParams())
    mux.SetUniformPtsIncrement(90000 // 25)

    def put(pkts):
        for p in pkts or []:
            mux.MuxVideoPacket(bytes(p["data"]), p["picture_type"], p["timestamp"])

    n = 0
    t0 = time.perf_counter()
    for pkt in demux:
        for f in dec.Decode(pkt):
            if n >= N:
                break
            rgb = torch.from_dlpack(f).clone().permute(2, 0, 1).float().div(255.0).contiguous()
            o = torch.from_dlpack(sr.run(rgb).image).clone().movedim(0, -1).clamp(0.0, 1.0)
            p = to_p010(o)
            torch.cuda.synchronize()
            put(enc.Encode(p))
            n += 1
        if n >= N:
            break
    put(enc.EndEncode())
    mux.Finalize()
    fps = n / (time.perf_counter() - t0)
    mux = None  # close the file before reading it back
    data = Path(out).read_bytes()
    os.remove(out)
    return label + rc_note, data, fps


@app.local_entrypoint()
def main() -> None:
    here = Path(__file__).parent
    todo = [v for v in VARIANTS if not (here / f"q_{v}.mp4").exists()]
    for res in encode.map(todo, return_exceptions=True, wrap_returned_exceptions=False):
        if isinstance(res, BaseException):
            msg = repr(res); i = msg.find("Here is the remote traceback")
            print("FAILED:", msg[i:][-700:] if i >= 0 else msg[:700], flush=True)
            continue
        label, data, fps = res
        (here / f"q_{label.split(' ')[0]}.mp4").write_bytes(data)
        print(f"{label}: {len(data) / 1e6:.1f} MB, pipeline {fps:.1f} fps", flush=True)
