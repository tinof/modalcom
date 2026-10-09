"""Where does a frame's time go? Each stage alone, in production's serial order, and overlapped.

One CreateEncoder per container (CLAUDE.md): every case runs in its own single-use
container. Frames: 300 decoded 1080p frames of the staged clip, upscaled 2x to 4K with
HIGHBITRATE_ULTRA, encoded with production's NVENC kwargs.

    modal volume put rtx-upscaler-jobs <clip>.mkv /probe-stages/input.mkv
    modal run scripts/probe_stages.py [--cases encode:p6f,multi4:p6f]
"""

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
    .uv_pip_install("numpy==2.5.2", "torch==2.13.0", "nvidia-vfx==0.2.0.0", "PyNvVideoCodec==2.2.3")
)
app = modal.App("rtx-probe-stages", image=image)
vol = modal.Volume.from_name("rtx-upscaler-jobs")
CLIP = "/jobs/probe-stages/input.mkv"
N = 300
ENC = {
    "p4q": dict(preset="P4", multipass="qres"),
    "p6f": dict(preset="P6", multipass="fullres"),
    "p6q": dict(preset="P6", multipass="qres"),
    "p4f": dict(preset="P4", multipass="fullres"),
    "p5f_hq": dict(preset="P5", multipass="fullres", tuning="high_quality", refl=False),
    "p5f_uhq": dict(preset="P5", multipass="fullres", refl=False),
    "p6f_hq": dict(preset="P6", multipass="fullres", tuning="high_quality"),
    "p6f_norefl": dict(preset="P6", multipass="fullres", refl=False),
    "p4q_norefl": dict(preset="P4", multipass="qres", refl=False),
    "p6f_norefl_hq": dict(preset="P6", multipass="fullres", refl=False, tuning="high_quality"),
    "p6_1pass": dict(preset="P6", multipass=None),
}


@app.function(gpu="RTX-PRO-6000", cpu=6, memory=24576, timeout=1800,
              volumes={"/jobs": vol}, single_use_containers=True)
def case(name: str, enc_key: str | None) -> str:
    import queue
    import statistics
    import threading
    import time

    import nvvfx
    import PyNvVideoCodec as nvc
    import torch

    orig = torch.Tensor.__dlpack__
    torch.Tensor.__dlpack__ = lambda self, *a, **k: orig(self)

    # --- inputs: N decoded frames resident on the GPU (uint8 HWC) ---------------
    demux = nvc.CreateDemuxer(filename=CLIP)
    dec = nvc.CreateDecoder(gpuid=0, codec=demux.GetNvCodecId(), usedevicememory=1,
                            outputColorType=nvc.OutputColorType.RGB)
    frames = []
    for pkt in demux:
        for f in dec.Decode(pkt):
            frames.append(torch.from_dlpack(f).clone())
        if len(frames) >= N:
            break
    frames = frames[:N]
    host_frames = [f.cpu().numpy() for f in frames] if name == "upload" else None

    sr = nvvfx.VideoSuperRes(nvvfx.VideoSuperRes.QualityLevel.HIGHBITRATE_ULTRA)
    sr.__enter__()
    sr.output_width, sr.output_height = 3840, 2160
    sr.load()

    def vsr(rgb_u8):
        x = rgb_u8.permute(2, 0, 1).float().div(255.0).contiguous()
        out = torch.from_dlpack(sr.run(x).image).clone()
        return out.movedim(0, -1).clamp(0.0, 1.0)

    def integrity(out):  # production's per-frame checks: isfinite + 6 .item() syncs
        assert torch.isfinite(out).all()
        m = out.float().mean(dim=(0, 1))
        _ = [m[c].item() for c in range(3)]

    def to_p010(rgb):
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

    def make_encoder():
        e = ENC[enc_key]
        return nvc.CreateEncoder(
            3840, 2160, "P010", False, codec="hevc", preset=e["preset"],
            tuning_info=e.get("tuning", "uhq"), rc="vbr", cq="18",
            **({"multipass": e["multipass"]} if e["multipass"] else {}), tier="high", aq="10",
            temporalaq="1", lookahead="32", bf="5", gop="250",
            **({"numrefl0": "5", "numrefl1": "5"} if e.get("refl", True) else {}),
        )

    def ms(xs):
        xs = xs[20:]  # drop warm-up
        return f"median {statistics.median(xs):6.2f} ms  mean {statistics.mean(xs):6.2f} ms"

    for _ in range(10):  # warm the effect
        vsr(frames[0])
    torch.cuda.synchronize()
    lines = [f"[{name} {enc_key or ''}] frames={len(frames)}"]

    if name == "vsr":
        # A: VSR only, sync each frame.  B: + clone/integrity/p010.  C: VSR back to back, no sync.
        a, b = [], []
        for f in frames:
            t = time.perf_counter(); vsr(f); torch.cuda.synchronize(); a.append((time.perf_counter() - t) * 1e3)
        for f in frames:
            t = time.perf_counter(); o = vsr(f); integrity(o); to_p010(o); torch.cuda.synchronize()
            b.append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter()
        for f in frames:
            vsr(f)
        torch.cuda.synchronize()
        c = (time.perf_counter() - t) * 1e3 / len(frames)
        lines += [f"VSR alone (sync/frame):            {ms(a)}",
                  f"VSR+integrity+P010 (sync/frame):   {ms(b)}",
                  f"VSR back-to-back, no per-frame sync: {c:6.2f} ms/frame"]

    elif name == "upload":
        cur, new = [], []
        for h in host_frames:  # production's woven-path input conversion
            t = time.perf_counter()
            _ = torch.from_numpy(h).float().div(255.0).cuda().permute(2, 0, 1).contiguous()
            torch.cuda.synchronize(); cur.append((time.perf_counter() - t) * 1e3)
        pinned = torch.empty((1080, 1920, 3), dtype=torch.uint8).pin_memory()
        for h in host_frames:
            t = time.perf_counter()
            pinned.numpy()[:] = h
            _ = pinned.cuda(non_blocking=True).permute(2, 0, 1).float().div(255.0).contiguous()
            torch.cuda.synchronize(); new.append((time.perf_counter() - t) * 1e3)
        lines += [f"host frame -> GPU, current (CPU float, pageable): {ms(cur)}",
                  f"host frame -> GPU, uint8 pinned + GPU float:      {ms(new)}"]

    elif name == "encode":
        surfaces = [to_p010(vsr(f)) for f in frames[:60]]
        torch.cuda.synchronize()
        enc = make_encoder()
        import subprocess as sp
        mon = sp.Popen(["nvidia-smi", "--query-gpu=clocks.video,clocks.sm,clocks.max.sm,power.draw,power.limit,clocks_event_reasons.active",
                        "--format=csv,noheader", "-lms", "500"], stdout=sp.PIPE, text=True)
        e = []
        for i in range(N):
            s = surfaces[i % len(surfaces)]
            t = time.perf_counter(); enc.Encode(s); e.append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter(); enc.EndEncode(); tail = (time.perf_counter() - t) * 1e3
        mon.terminate(); samples = mon.stdout.read().strip().splitlines()
        lines.append("clocks (video, max video, sm, power, throttle): " + (samples[len(samples)//2] if samples else "n/a"))
        lines += [f"Encode() alone: {ms(e)}  (EndEncode {tail:.0f} ms)",
                  f"Encode-only throughput: {1000 * N / (sum(e) + tail):.1f} fps"]

    elif name.startswith("multi"):  # k independent sessions at once: do they use separate NVENC engines?
        k = int(name[5:])
        surfaces = [to_p010(vsr(f)) for f in frames[:60]]
        torch.cuda.synchronize()
        encs = [make_encoder() for _ in range(k)]

        def run(enc):
            for i in range(N):
                enc.Encode(surfaces[i % len(surfaces)])
            enc.EndEncode()

        ths = [threading.Thread(target=run, args=(e,)) for e in encs]
        t0 = time.perf_counter()
        for t in ths: t.start()
        for t in ths: t.join()
        total = time.perf_counter() - t0
        lines += [f"{k} concurrent sessions: aggregate {k * N / total:.1f} fps, "
                  f"per session {N / total:.1f} fps"]

    elif name == "serial":  # production order: VSR -> checks -> P010 -> sync -> Encode
        enc = make_encoder()
        v, e = [], []
        t0 = time.perf_counter()
        for f in frames:
            t = time.perf_counter(); o = vsr(f); integrity(o); p = to_p010(o); torch.cuda.synchronize()
            v.append((time.perf_counter() - t) * 1e3)
            t = time.perf_counter(); enc.Encode(p); e.append((time.perf_counter() - t) * 1e3)
        enc.EndEncode()
        total = time.perf_counter() - t0
        lines += [f"VSR stage in pipeline: {ms(v)}", f"Encode in pipeline:    {ms(e)}",
                  f"serial throughput: {N / total:.1f} fps"]

    elif name == "overlap":  # VSR of frame i+1 runs while frame i encodes on another thread
        enc = make_encoder()
        q: queue.Queue = queue.Queue(maxsize=4)
        e = []

        def encoder_loop():
            while (p := q.get()) is not None:
                t = time.perf_counter(); enc.Encode(p); e.append((time.perf_counter() - t) * 1e3)

        th = threading.Thread(target=encoder_loop); th.start()
        side = torch.cuda.Stream()
        v = []
        t0 = time.perf_counter()
        for f in frames:
            t = time.perf_counter()
            with torch.cuda.stream(side):
                o = vsr(f); integrity(o); p = to_p010(o)
            side.synchronize()  # surface complete before NVENC reads it (the striping fix)
            v.append((time.perf_counter() - t) * 1e3)
            q.put(p)
        q.put(None); th.join(); enc.EndEncode()
        total = time.perf_counter() - t0
        lines += [f"VSR stage while encoding: {ms(v)}", f"Encode while VSR runs:    {ms(e)}",
                  f"overlapped throughput: {N / total:.1f} fps"]

    clocks = torch.cuda.get_device_name(0)
    lines.append(f"device: {clocks}")
    return "\n".join(lines)


@app.local_entrypoint()
def main(cases: str = "vsr,upload,encode:p4q,encode:p5f_uhq,serial:p5f_uhq,overlap:p5f_uhq,multi4:p5f_uhq") -> None:
    """--cases name[:enc_key],...  e.g. --cases encode:p6f,multi3:p6f"""
    todo = [(c.split(":")[0], (c.split(":") + [None])[1]) for c in cases.split(",")]
    for out in case.starmap(todo, return_exceptions=True, wrap_returned_exceptions=False):
        print(out, "\n", flush=True)
