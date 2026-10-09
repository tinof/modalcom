"""Can one container drive all four NVENC engines, and do segmented encodes join cleanly?

Phase-0 probe for the parallel encoder in modal_app.py. Every case runs in its own
single-use container, with production's NVENC settings (P5 + fullres + UHQ, CQ 20,
level 5.1, the Reconfigure()d 100 Mbps / 1 s VBV ceiling).

Cases (`--cases a,b,...`):
  enc:K[:TUNE][:nocap]
                 K sessions, encode only, one thread each. Aggregate fps. TUNE is uhq
                 (default) or hq; nocap skips the Reconfigure()d 100 Mbps ceiling, which
                 leaves the driver's level-derived VBR cap (the old probe_stages condition).
                 Every enc/encproc case also samples nvidia-smi (SM %, encoder %, video
                 clock) over the timed window and reports VRAM per encoder session.
  encproc:K[:TUNE][:nocap]
                 Same, but each session in its own process (own CUDA context, own GIL).
                 Separates in-process serialization from the hardware.
  pipe:K:dev     K threads, each VSR -> integrity -> P010 -> torch.cuda.synchronize() ->
                 Encode with its own nvvfx effect. Aggregate fps, plus a pixel check of
                 every thread's output (striping Laplacian, flat frames).
  pipe:K:stream  Same, but each thread runs on its own CUDA stream and syncs only it.
                 nvvfx stays on stream 0 and the __dlpack__ patch drops the consumer
                 stream: the 2026-10-04 striped variant, kept as the negative control.
  pipe:K:s1..s4  One stream S per worker, connected end to end: nvvfx run(stream_ptr=S),
                 CreateEncoder(cudastream=S), and a __dlpack__ patch that forwards the
                 consumer stream. s1 = S.synchronize() before Encode() and the surface held
                 until the next sync; s2 = no hold; s3 = no explicit sync either; s4 = s3 +
                 nvvfx run(non_blocking=True). All four threads encode the same frames, so
                 their outputs must be byte-identical to each other (md5 reported), and to
                 pipe:K:dev on a host with the same driver.
  pipe:K:s1v|s1e|s1d
                 Bisect variants of s1: s1v adds torch.cuda.synchronize() right after
                 nvvfx run(); s1e creates the encoder on its own stream (no cudastream);
                 s1d syncs the whole device instead of S before Encode(); s1L runs every
                 nvvfx call under one lock across threads (no two VSR runs overlap on the
                 GPU); s0 is s1L with nvvfx on stream 0.
  pipe:K:sb1|sb3 s1/s3 on blocking streams (created with cudaStreamDefault), which take part
                 in legacy-default-stream ordering.
  pipe:K:sx|sy   s3 with nvvfx run() split open, because its get_output() copies on the
                 legacy default stream: sx waits on S from the host before that copy, sy
                 orders it with events only. Append :hash to any pipe case for exact
                 checksums of every P010 surface handed to Encode().
  join:endenc    One session, 3 segments. EndEncode() after each segment, then keep
                 encoding on the same session. Per-segment mp4s, ffmpeg concat.
  join:idr       One session, 3 segments, FORCEIDR at each segment start, packets routed
                 to segment muxers by timestamp. No EndEncode() in between.
  join4:endenc   4 sessions encode the same 3 segments concurrently: are the bitstreams
                 and GetSequenceParams() identical across sessions?
  full:K:MODE:SECONDS:CLIP[:label][:cpu12]
                 The real thing: keyframe-aligned segments of SECONDS, K workers with
                 static assignment, each segment decoded by the piped ffmpeg decoder with
                 seek + trim and an exact-pts guard, VSR, encode, per-segment mp4, concat.
                 MODE idr (continuous session, FORCEIDR per segment) or endenc (EndEncode
                 per segment). CLIP is /jobs/probe-par/<CLIP>.mkv. Checks frame count
                 against a full decode, reports fps and CPU use, and copies the joined
                 output to /jobs/probe-par/out/<case>.mp4 (delete it afterwards).
  create         CreateEncoder error-8 rates at level 5.1 vs none, fresh / with a VSR
                 effect loaded / with three sessions alive.

    modal volume put rtx-upscaler-jobs <clip>.mkv /probe-par/input.mkv
    modal run scripts/probe_parallel_encode.py --cases enc:1,enc:4,pipe:4:dev,full:4:idr:30:input
"""

import os

import modal

# Production pins nvidia-vfx 0.2.0.0 (VFX SDK 1.3.0.0); PROBE_VFX=0.1.0.1 re-tests the old wheel.
VFX_VERSION = os.environ.get("PROBE_VFX", "0.2.0.0")
DRIVER_VERSION = "580.95.05"
DRIVER_RUN_URL = (
    f"https://us.download.nvidia.com/XFree86/Linux-x86_64/{DRIVER_VERSION}"
    f"/NVIDIA-Linux-x86_64-{DRIVER_VERSION}.run"
)
FFMPEG_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-09-30-13-08"
    "/ffmpeg-n8.1.3-9-g29e619e767-linux64-gpl-8.1.tar.xz"
)
FFMPEG_SHA256 = "97ce978979194b5cf7e06a5e68020dbdaa7a4f3294c5452b6a1bc347100dbb79"
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
        f"curl -fsSL -o /tmp/ff.tar.xz {FFMPEG_URL}",
        f"echo '{FFMPEG_SHA256}  /tmp/ff.tar.xz' | sha256sum -c -",
        "mkdir -p /opt/ffmpeg",
        "tar -xJf /tmp/ff.tar.xz -C /opt/ffmpeg --strip-components=1",
        "cp /opt/ffmpeg/bin/ffmpeg /opt/ffmpeg/bin/ffprobe /usr/local/bin/",
        "rm -rf /tmp/ff.tar.xz /opt/ffmpeg",
    )
    .uv_pip_install("numpy==2.5.2", "torch==2.13.0", f"nvidia-vfx=={VFX_VERSION}", "PyNvVideoCodec==2.2.3",
                    extra_index_url="https://pypi.nvidia.com")
)
app = modal.App("rtx-probe-parallel", image=image)
vol = modal.Volume.from_name("rtx-upscaler-jobs")
CLIP = "/jobs/probe-par/input.mkv"
N = 300
N_ENC = 600  # frames per session in the enc/encproc cases: long enough for a steady state
W, H = 3840, 2160
ENC_KW = dict(codec="hevc", preset="P5", tuning_info="uhq", rc="vbr", cq="20",
              multipass="fullres", tier="high", aq="10", temporalaq="1", lookahead="32",
              bf="5", gop="250")
TUNINGS = {"uhq": "uhq", "hq": "high_quality"}
STREAM_MODES = ("s1", "s2", "s3", "s4", "s1v", "s1e", "s1d", "s1L", "s0", "sb1", "sb3", "sx", "sy", "sz", "szn", "szi")
HELD_MODES = ("s1", "s1v", "s1e", "s1d", "s1L", "s0")


def _run_case(name: str) -> str:
    import contextlib
    import hashlib
    import os
    import queue
    import subprocess
    import sys
    import tempfile
    import threading
    import time
    from fractions import Fraction

    import numpy as np
    import nvvfx
    import PyNvVideoCodec as nvc
    import torch
    import torch.nn.functional as F

    parts = name.split(":")
    kind = parts[0]
    orig = torch.Tensor.__dlpack__
    if kind == "pipe" and parts[2] in STREAM_MODES:
        # PyNvVideoCodec passes its copy stream positionally; torch 2.13 wants it as a
        # keyword, and uses it to make that stream wait for the producing stream.
        torch.Tensor.__dlpack__ = (
            lambda self, *a, **k: orig(self, stream=a[0]) if a else orig(self, **k))
    else:  # production's patch: drops the consumer stream
        torch.Tensor.__dlpack__ = lambda self, *a, **k: orig(self)
    work = tempfile.mkdtemp()
    smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    from importlib.metadata import version

    lines = [f"[{name}] PyNvVideoCodec {nvc.__version__}; nvidia-vfx {version('nvidia-vfx')}; {smi}"]
    tuning = parts[2] if kind in ("enc", "encproc", "encchild") and len(parts) > 2 else "uhq"
    cap = "nocap" not in parts

    def make_encoder(level="5.1", retries=10, stream=None):
        """(encoder, attempts). Error 8 at an explicit level is intermittent: retry it."""
        kw = dict(ENC_KW, tuning_info=TUNINGS[tuning])
        if level:
            kw["level"] = level
        if stream is not None:
            kw["cudastream"] = stream.cuda_stream
        for attempt in range(1, retries + 1):
            try:
                enc = nvc.CreateEncoder(W, H, "P010", False, **kw)
                break
            except Exception as exc:  # noqa: BLE001
                if "error 8" not in str(exc):
                    raise
                if attempt == retries:
                    raise RuntimeError(f"{smi}: {retries} CreateEncoder attempts failed: {exc}") from exc
                time.sleep(0.2 * attempt)
        if cap:
            rc = enc.GetEncodeReconfigureParams()
            rc.maxBitRate = 100_000_000
            rc.vbvBufferSize = 100_000_000
            assert enc.Reconfigure(rc), "Reconfigure refused"
        return enc, attempt

    def gpu_mem_mib():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.split()
        return int(out[0]) if out else -1

    class GpuSampler:
        """nvidia-smi every 200 ms: SM %, encoder %, video clock. Medians over a window."""

        def __init__(self):
            self.rows = []
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.encoder,clocks.video",
                 "--format=csv,noheader,nounits", "-lms", "200"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=self._read, daemon=True).start()

        def _read(self):
            for line in self.proc.stdout:
                with contextlib.suppress(ValueError):
                    self.rows.append((time.perf_counter(), *map(float, line.split(","))))

        def report(self, t0, t1):
            self.proc.terminate()
            rows = [r for r in self.rows if t0 + 1.0 <= r[0] <= t1 - 1.0] or self.rows
            if not rows:
                return "gpu: no nvidia-smi samples"
            med = [sorted(r[i] for r in rows)[len(rows) // 2] for i in (1, 2, 3)]
            return f"gpu (median of {len(rows)}): sm {med[0]:.0f}%, enc {med[1]:.0f}%, video clock {med[2]:.0f} MHz"

    def to_p010(rgb):  # modal_app._rgb_to_p010
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

    def new_sr():
        sr = nvvfx.VideoSuperRes(nvvfx.VideoSuperRes.QualityLevel.HIGHBITRATE_ULTRA)
        sr.__enter__()
        sr.output_width, sr.output_height = W, H
        sr.load()
        return sr

    def vsr_frame(sr, rgb_u8, stream_ptr=0, non_blocking=False, device_sync=False, lock=None,
                  split=None, pending=None):
        """_upscale_rgb_batch for one frame, incl. integrity checks."""
        x = rgb_u8.permute(2, 0, 1).float().div(255.0).contiguous()
        if split == "legacy":
            # nvvfx entirely on the legacy default stream, as in production, but joined to
            # this worker's stream S with events instead of a device-wide host sync.
            cur, legacy = torch.cuda.current_stream(), torch.cuda.default_stream()
            legacy.wait_event(cur.record_event())
            x.record_stream(legacy)  # S must not reuse x's memory before nvvfx has read it
            result = sr.run(x, non_blocking=non_blocking)
            cur.wait_event(legacy.record_event())
            out = torch.from_dlpack(result.image).clone()
        elif split is not None:
            # VideoSuperRes.run() split open: its get_output() copies on the legacy default
            # stream, which does not wait for a non-blocking stream S. Order that copy
            # after the inference on S, and S's clone after the copy.
            eff = sr._effect
            eff.set_input_image(x.shape[2], x.shape[1])
            eff.set_cuda_stream(stream_ptr)
            eff.transfer_input(x, stream_ptr)
            eff.run(non_blocking)
            cur, legacy = torch.cuda.current_stream(), torch.cuda.default_stream()
            if split == "host":
                cur.synchronize()  # this worker's stream only
            else:
                legacy.wait_event(cur.record_event())
            image = eff.get_output(sr.device)
            cur.wait_event(legacy.record_event())
            out = torch.from_dlpack(image).clone()
        elif lock is not None:  # s1L / s0: one VSR on the GPU at a time, across all threads
            torch.cuda.current_stream().synchronize()
            with lock:
                result = sr.run(x, stream_ptr=stream_ptr, non_blocking=non_blocking)
                out = torch.from_dlpack(result.image).clone()
                torch.cuda.current_stream().synchronize()
        else:
            result = sr.run(x, stream_ptr=stream_ptr, non_blocking=non_blocking)
            if device_sync:  # s1v: is nvvfx's output complete when run() returns on stream S?
                torch.cuda.synchronize()
            out = torch.from_dlpack(result.image).clone()
        if pending is not None:
            # szi: the same checks, read back a few frames later instead of a host sync
            # per frame.
            stats = torch.cat([torch.isfinite(out).all().float().view(1), out.float().mean(dim=(-2, -1))])
            pending.append((torch.cuda.current_stream().record_event(), stats.to("cpu", non_blocking=True)))
            while pending and pending[0][0].query():
                assert pending.pop(0)[1][0].item() == 1.0, "non-finite VSR output"
        else:
            assert torch.isfinite(out).all()
            m = out.float().mean(dim=(-2, -1))
            _ = [m[c].item() for c in range(3)]
        return out.movedim(0, -1).clamp(0.0, 1.0)

    def new_muxer(path, enc, fps=25):
        n, d = Fraction(fps).limit_denominator(65535).as_integer_ratio()
        mux = nvc.FFmpegMuxer(path, nvc.MP4, "hevc", W, H, n, d, 1, 90000, enc.GetSequenceParams())
        mux.SetUniformPtsIncrement(round(90000 * d / n))
        return mux

    def gpu_frames(count):
        demux = nvc.CreateDemuxer(filename=CLIP)
        dec = nvc.CreateDecoder(gpuid=0, codec=demux.GetNvCodecId(), usedevicememory=1,
                                outputColorType=nvc.OutputColorType.RGB)
        frames = []
        for pkt in demux:
            for f in dec.Decode(pkt):
                frames.append(torch.from_dlpack(f).clone())
            if len(frames) >= count:
                break
        return frames[:count]

    def pixel_check(path, mid_only=False, idx=()):
        """flat frame count, and striping Laplacian at the given frame indices."""
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=320:180", "-f",
                              "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
        a = np.frombuffer(raw, np.uint8).reshape(-1, 180 * 320).astype(np.float32)
        flat = int((a.std(1) <= 1).sum())
        laps = {}
        for i in (idx or [a.shape[0] // 2]):
            r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf",
                                f"select=eq(n\\,{i}),crop=512:512:{W // 2 - 256}:{H // 2 - 256}",
                                "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                               capture_output=True).stdout
            g = np.frombuffer(r[:512 * 512 * 3], np.uint8).reshape(512, 512, 3).astype(np.float32).mean(-1)
            lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
            laps[i] = round(float(np.abs(lap).mean()), 2)
        return a.shape[0], flat, laps

    def concat(paths, out):
        lst = os.path.join(work, "list.txt")
        with open(lst, "w") as f:
            f.writelines(f"file '{p}'\n" for p in paths)
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                            "-c", "copy", out], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]

    def packet_report(path):
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "packet=pts,dts,flags", "-of", "csv=p=0", path],
                           capture_output=True, text=True).stdout.split()
        pts = sorted(int(x.split(",")[0]) for x in r)
        keys = sorted(int(x.split(",")[0]) for x in r if "K" in x.split(",")[2])
        d = {pts[i + 1] - pts[i] for i in range(len(pts) - 1)}
        step = min(d) if d else 1
        return len(pts), sorted(d), [(k - pts[0]) // step for k in keys]

    def cpu_sample():
        with open("/proc/stat") as f:
            v = list(map(int, f.readline().split()[1:]))
        return sum(v), v[3] + v[4]

    torch.zeros(1, device="cuda")

    def p010_surfaces(count=60):
        sr = new_sr()
        surfaces = [to_p010(vsr_frame(sr, f)) for f in gpu_frames(count)]
        torch.cuda.synchronize()
        return surfaces

    def mbps(nbytes):  # the clip is 25 fps
        return nbytes * 8 * 25 / N_ENC / 1e6

    if kind == "enc":
        k = int(parts[1])
        surfaces = p010_surfaces()
        mem0 = gpu_mem_mib()
        encs = [make_encoder()[0] for _ in range(k)]
        mem_per = (gpu_mem_mib() - mem0) / k
        sps = {bytes(e.GetSequenceParams()) for e in encs}
        sizes = [0] * k

        def run(i):
            for j in range(N_ENC):
                sizes[i] += sum(len(p["data"]) for p in encs[i].Encode(surfaces[j % len(surfaces)]))
            sizes[i] += sum(len(p["data"]) for p in encs[i].EndEncode())

        ths = [threading.Thread(target=run, args=(i,)) for i in range(k)]
        sampler = GpuSampler()
        time.sleep(1.0)
        t0 = time.perf_counter()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        t1 = time.perf_counter()
        dt = t1 - t0
        lines.append(f"{k} sessions encode-only ({tuning}, {'100M cap' if cap else 'driver cap'}): "
                     f"aggregate {k * N_ENC / dt:.1f} fps, per session {N_ENC / dt:.1f}; "
                     f"{mbps(sizes[0]):.1f} Mbps; distinct SPS={len(sps)}; VRAM/session {mem_per:.0f} MiB")
        lines.append("  " + sampler.report(t0, t1))

    elif kind == "encchild":
        # One session in this process, started by encproc. Rendezvous through files.
        idx, sync_dir = int(parts[1]), parts[-1]
        surfaces = p010_surfaces()
        enc = make_encoder()[0]
        open(os.path.join(sync_dir, f"ready{idx}"), "w").close()
        while not os.path.exists(os.path.join(sync_dir, "go")):
            time.sleep(0.005)
        size = 0
        t0 = time.time()
        for j in range(N_ENC):
            size += sum(len(p["data"]) for p in enc.Encode(surfaces[j % len(surfaces)]))
        size += sum(len(p["data"]) for p in enc.EndEncode())
        return f"CHILD {t0} {time.time()} {size}"

    elif kind == "encproc":
        k = int(parts[1])
        sync_dir = tempfile.mkdtemp()
        loader = ("import importlib.util as u, sys; s = u.spec_from_file_location('ppe', sys.argv[1]); "
                  "m = u.module_from_spec(s); s.loader.exec_module(m); print(m._run_case(sys.argv[2]))")
        rest = ":".join(parts[2:])
        procs = [subprocess.Popen(
            [sys.executable, "-c", loader, os.path.abspath(__file__),
             f"encchild:{i}:{rest or 'uhq'}:{sync_dir}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(k)]
        deadline = time.time() + 600
        while not all(os.path.exists(os.path.join(sync_dir, f"ready{i}")) for i in range(k)):
            if time.time() > deadline or any(p.poll() not in (None, 0) for p in procs):
                break
            time.sleep(0.05)
        sampler = GpuSampler()
        time.sleep(1.0)
        t0 = time.perf_counter()
        open(os.path.join(sync_dir, "go"), "w").close()
        results = [p.communicate() for p in procs]
        t1 = time.perf_counter()
        runs = []
        for out, err in results:
            row = [x for x in out.splitlines() if x.startswith("CHILD ")]
            if not row:
                lines.append(f"  child failed: {err[-600:]}")
                continue
            a, b, size = row[-1].split()[1:]
            runs.append((float(a), float(b), int(size)))
        if runs:
            span = max(r[1] for r in runs) - min(r[0] for r in runs)
            lines.append(f"{len(runs)} processes encode-only ({tuning}, {'100M cap' if cap else 'driver cap'}): "
                         f"aggregate {len(runs) * N_ENC / span:.1f} fps, per process "
                         f"{[round(N_ENC / (r[1] - r[0]), 1) for r in runs]}; {mbps(runs[0][2]):.1f} Mbps")
        lines.append("  " + sampler.report(t0, t1))

    elif kind == "pipe":
        k, mode = int(parts[1]), parts[2]
        blocking = mode.startswith("sb")
        split = {"sx": "host", "sy": "event", "sz": "legacy", "szn": "legacy", "szi": "legacy"}.get(mode)
        deferred = mode == "szi"
        nb_legacy = mode in ("szn", "szi")
        mode = {"sb1": "s1", "sb3": "s3", "sx": "s3", "sy": "s3", "sz": "s3", "szn": "s3",
                "szi": "s3"}.get(mode, mode)
        frames = gpu_frames(N)
        torch.cuda.synchronize()
        srs = [None] * k
        ready = threading.Barrier(k + 1)
        go = threading.Event()
        encs = [None] * k
        counts = [0] * k
        errors = []
        paths = [os.path.join(work, f"pipe{i}.mp4") for i in range(k)]
        # s1..s4: one non-default stream per worker, shared by torch, nvvfx and the encoder.
        # A zero pointer would make CreateEncoder create its own stream (PyNvEncoder.cpp:224).
        streams = [torch.cuda.Stream() for _ in range(k)] if mode in STREAM_MODES else [None] * k
        if blocking:
            # nvvfx's get_output() copies on the legacy default stream, which does not wait
            # for torch's (non-blocking) streams. A blocking stream does take part in
            # legacy-stream ordering.
            from cuda.bindings import runtime as cudart

            def blocking_stream():
                err, handle = cudart.cudaStreamCreateWithFlags(cudart.cudaStreamDefault)
                assert err == cudart.cudaError_t.cudaSuccess, err
                return torch.cuda.ExternalStream(int(handle))

            streams = [blocking_stream() for _ in range(k)]
        vsr_lock = threading.Lock()
        hashing = "hash" in parts
        sums = [[] for _ in range(k)]
        weight = []

        def chk(i, p):
            """Exact checksums of the P010 surface handed to Encode() (":hash" cases only)."""
            if not hashing:
                return
            torch.cuda.synchronize()
            if not weight:
                weight.append((torch.arange(p.numel(), device=p.device, dtype=torch.int64) % 1009 + 1)
                              .view(p.shape))
            pv = p.to(torch.int64)
            sums[i].append((int(pv.sum()), int((pv * weight[0]).sum())))

        def run(i):
            try:
                srs[i] = new_sr()
                for f in frames[:5]:
                    vsr_frame(srs[i], f)
                torch.cuda.synchronize()
                ready.wait()
                go.wait()
                enc = encs[i]
                mux = new_muxer(paths[i], enc)
                stream = torch.cuda.Stream() if mode == "stream" else streams[i]
                ptr = stream.cuda_stream if mode in STREAM_MODES and mode != "s0" else 0
                lock = vsr_lock if mode in ("s1L", "s0") else None
                non_blocking = mode == "s4"
                held = None
                pending = []
                for f in frames:
                    if mode in STREAM_MODES:
                        with torch.cuda.stream(stream):
                            p = to_p010(vsr_frame(srs[i], f, ptr, non_blocking or nb_legacy, mode == "s1v", lock, split,
                                                pending if deferred else None))
                            if mode == "s1d":
                                torch.cuda.synchronize()
                                held = None
                            elif mode in ("s1", "s2", "s1v", "s1e", "s1L", "s0"):
                                stream.synchronize()
                                held = None
                            chk(i, p)
                            pkts = enc.Encode(p)
                            if mode in HELD_MODES:
                                held = p  # noqa: F841 - kept alive until the next sync
                    elif stream is not None:
                        with torch.cuda.stream(stream):
                            p = to_p010(vsr_frame(srs[i], f))
                        stream.synchronize()
                        chk(i, p)
                        pkts = enc.Encode(p)
                    else:
                        p = to_p010(vsr_frame(srs[i], f))
                        torch.cuda.synchronize()
                        chk(i, p)
                        pkts = enc.Encode(p)
                    for pkt in pkts:
                        mux.MuxVideoPacket(bytes(pkt["data"]), pkt["picture_type"], pkt["timestamp"])
                        counts[i] += 1
                for pkt in enc.EndEncode():
                    mux.MuxVideoPacket(bytes(pkt["data"]), pkt["picture_type"], pkt["timestamp"])
                    counts[i] += 1
                mux.Finalize()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"thread {i}: {exc!r}")
                with contextlib.suppress(Exception):
                    ready.abort()

        ths = [threading.Thread(target=run, args=(i,)) for i in range(k)]
        for t in ths:
            t.start()
        ready.wait()  # every effect loaded before any encoder exists (production order)
        for i in range(k):
            encs[i] = make_encoder(stream=None if mode == "s1e" else streams[i])[0]
        sampler = GpuSampler()
        time.sleep(1.0)
        t0 = time.perf_counter()
        go.set()
        for t in ths:
            t.join()
        t1 = time.perf_counter()
        dt = t1 - t0
        lines.append(f"{k} pipelines ({mode} sync): aggregate {k * N / dt:.1f} fps, per pipeline "
                     f"{N / dt:.1f}; packets {counts}; errors {errors}")
        lines.append("  " + sampler.report(t0, t1))
        digests = []
        for i, p in enumerate(paths):
            if os.path.exists(p):
                digest = hashlib.md5(open(p, "rb").read()).hexdigest()[:12]
                digests.append(digest)
                n, flat, laps = pixel_check(p, idx=[30, 150, 270])
                lines.append(f"  out{i}: frames={n} flat={flat} lap={laps} bytes={os.path.getsize(p)} md5={digest}")
        lines.append(f"  distinct outputs across threads: {len(set(digests))}")
        if hashing:
            ref = sums[0]
            lines.append(f"  P010 checksums: md5={hashlib.md5(repr(ref).encode()).hexdigest()[:12]} "
                         f"first={ref[:2]} frame150={ref[150:151]}")
            for i in range(1, k):
                diff = [j for j, (a, b) in enumerate(zip(ref, sums[i])) if a != b]
                lines.append(f"  thread {i}: {len(diff)} P010 frames differ from thread 0, first {diff[:5]}")

    elif kind in ("join", "join4"):
        mode = parts[1]
        k = 4 if kind == "join4" else 1
        src = gpu_frames(N)
        # Bicubic 4K versions of the real frames: cheap, and lets us match decoded frames
        # back to their source index.
        up = [to_p010(F.interpolate(f.permute(2, 0, 1)[None].float().div(255), size=(H, W),
                                    mode="bicubic", align_corners=False)[0].movedim(0, -1).clamp(0, 1))
              for f in src]
        torch.cuda.synchronize()
        thumbs = np.stack([F.interpolate(f.permute(2, 0, 1)[None].float(), size=(90, 160), mode="area")[0]
                           .mean(0).cpu().numpy() for f in src])
        seg_len = [100, 100, 100]
        encs = [make_encoder()[0] for _ in range(k)]
        sps = {bytes(e.GetSequenceParams()) for e in encs}
        seg_paths = [[os.path.join(work, f"s{i}_{s}.mp4") for s in range(3)] for i in range(k)]
        errors = []
        idr = nvc.NV_ENC_PIC_FLAGS.FORCEIDR

        def run(i):
            enc = encs[i]
            try:
                if mode == "endenc":
                    for s in range(3):
                        mux = new_muxer(seg_paths[i][s], enc)
                        first, n = None, 0
                        for j in range(sum(seg_len[:s]), sum(seg_len[:s + 1])):
                            pkts = enc.Encode(up[j], int(idr)) if j == sum(seg_len[:s]) else enc.Encode(up[j])
                            for pkt in pkts:
                                first = pkt["timestamp"] if first is None else min(first, pkt["timestamp"])
                                mux.MuxVideoPacket(bytes(pkt["data"]), pkt["picture_type"],
                                                   pkt["timestamp"] - s * 100)
                                n += 1
                        for pkt in enc.EndEncode():
                            mux.MuxVideoPacket(bytes(pkt["data"]), pkt["picture_type"],
                                               pkt["timestamp"] - s * 100)
                            n += 1
                        mux.Finalize()
                        if n != seg_len[s]:
                            errors.append(f"s{i} seg{s}: {n} packets for {seg_len[s]} frames")
                else:  # idr: one continuous session, packets routed by timestamp
                    muxes = [new_muxer(p, enc) for p in seg_paths[i]]
                    got = [0, 0, 0]
                    order = []

                    def route(pkts):
                        for pkt in pkts:
                            s = pkt["timestamp"] // 100
                            order.append(s)
                            muxes[s].MuxVideoPacket(bytes(pkt["data"]), pkt["picture_type"],
                                                    pkt["timestamp"] - s * 100)
                            got[s] += 1
                            if got[s] == seg_len[s]:
                                muxes[s].Finalize()
                    for j in range(N):
                        route(enc.Encode(up[j], int(idr)) if j % 100 == 0 else enc.Encode(up[j]))
                    route(enc.EndEncode())
                    inter = sum(1 for a, b in zip(order, order[1:]) if b < a)
                    if inter:
                        errors.append(f"s{i}: {inter} packets arrived after a later segment's")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"s{i}: {exc!r}")

        ths = [threading.Thread(target=run, args=(i,)) for i in range(k)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        lines.append(f"{k} session(s), mode {mode}: errors {errors}; distinct SPS {len(sps)}")
        if k > 1:
            for s in range(3):
                digests = {open(seg_paths[i][s], "rb").read() for i in range(k)
                           if os.path.exists(seg_paths[i][s])}
                lines.append(f"  seg{s}: {len(digests)} distinct bitstream(s) across {k} sessions")
        out = os.path.join(work, "joined.mp4")
        concat(seg_paths[0], out)
        n_pk, deltas, key_idx = packet_report(out)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", out, "-vf", "scale=160:90:flags=area",
                              "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
        dec = np.frombuffer(raw, np.uint8).reshape(-1, 90, 160).astype(np.float32)
        misplaced = []
        for j in range(dec.shape[0]):
            lo, hi = max(0, j - 3), min(len(thumbs), j + 4)
            best = lo + int(np.argmin([np.abs(dec[j] - thumbs[m]).mean() for m in range(lo, hi)]))
            if best != j:
                misplaced.append((j, best))
        n, flat, laps = pixel_check(out, idx=[0, 99, 100, 199, 200, 299])
        lines.append(f"  joined: packets={n_pk} decoded={dec.shape[0]} pts-deltas={deltas} "
                     f"keyframes at {key_idx} misplaced={misplaced[:10]} flat={flat} lap={laps}")

    elif kind == "create":
        # How often does CreateEncoder with an explicit level fail with error 8, and does
        # an immediate retry succeed? Also: is the driver-selected level stable without it?
        import gc
        import hashlib

        def attempt(level):
            t = time.perf_counter()
            try:
                e = make_encoder(level, retries=1)[0]
                return e, None, time.perf_counter() - t
            except Exception as exc:  # noqa: BLE001
                return None, "error 8" if "error 8" in str(exc) else repr(exc)[:80], time.perf_counter() - t

        def series(label, n, level, hold=()):
            res, times, sps = [], [], set()
            for _ in range(n):
                e, err, dt = attempt(level)
                times.append(dt)
                res.append("x" if err else ".")
                if e is not None:
                    sps.add(hashlib.md5(bytes(e.GetSequenceParams())).hexdigest()[:8])
                    del e
                    gc.collect()
            lines.append(f"{label}: {''.join(res)}  fails={res.count('x')}/{n} "
                         f"create median {sorted(times)[len(times) // 2] * 1000:.0f} ms; distinct SPS {sorted(sps)}")

        series("fresh, level 5.1", 30, "5.1")
        series("fresh, no level ", 15, None)
        sr = new_sr()
        f = gpu_frames(5)
        for x in f:
            vsr_frame(sr, x)
        torch.cuda.synchronize()
        series("VSR loaded, level 5.1", 30, "5.1")
        held = [make_encoder("5.1", retries=10)[0] for _ in range(3)]
        series("3 sessions alive, level 5.1", 20, "5.1")
        del held
        gc.collect()

    elif kind == "full":
        # full:K:MODE  MODE endenc = EndEncode() after every segment; idr = one continuous
        # session per worker, FORCEIDR at segment starts, packets routed by timestamp.
        k = int(parts[1])
        mode = parts[2] if len(parts) > 2 else "idr"
        seg_seconds = float(parts[3]) if len(parts) > 3 else 8.0
        clip = f"/jobs/probe-par/{parts[4]}.mkv" if len(parts) > 4 else CLIP
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=time_base,width,height,r_frame_rate", "-of",
                                "default=noprint_wrappers=1", clip], capture_output=True, text=True).stdout
        info = dict(line.split("=", 1) for line in probe.split())
        tb_n, tb_d = map(int, info["time_base"].split("/"))
        in_w, in_h = int(info["width"]), int(info["height"])
        fps = float(Fraction(info["r_frame_rate"]))
        st = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of",
                             "csv=p=0", clip], capture_output=True, text=True).stdout.split()
        start_time = float(st[0]) if st and st[0] != "N/A" else 0.0
        pk = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "packet=pts,flags", "-of", "csv=p=0", clip], capture_output=True, text=True).stdout.split()
        keys = sorted(int(p.split(",")[0]) for p in pk if "K" in p.split(",")[1] and not p.startswith("N/A"))
        starts = [keys[0]]
        for kk in keys:
            if (kk - starts[-1]) * tb_n / tb_d >= seg_seconds:
                starts.append(kk)
        # First segment has no lower bound and the last none upper, so the union is
        # exactly what a full decode yields (incl. any frames before the first keyframe).
        segs = [(starts[i] if i else None, starts[i + 1] if i + 1 < len(starts) else None)
                for i in range(len(starts))]
        workers = min(k, len(segs))
        frame_bytes = in_w * in_h * 3

        def decode_cmd(a, b, stats):
            seek = []
            if a is not None:
                seek = ["-copyts", "-noaccurate_seek", "-ss", f"{max(0.0, a * tb_n / tb_d - start_time - 1.0):.6f}"]
            else:
                seek = ["-copyts"]
            bounds = [f"start_pts={a}"] if a is not None else []
            bounds += [f"end_pts={b}"] if b is not None else []
            trim = f"trim={':'.join(bounds)}," if bounds else ""
            return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-hwaccel", "cuda", *seek,
                    "-i", clip, "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough",
                    "-vf", f"{trim}scale=in_color_matrix=bt709:in_range=tv"
                           ":flags=bicubic+accurate_rnd+full_chroma_int",
                    "-enc_time_base:v", "demux", "-stats_enc_pre", stats,
                    "-stats_enc_pre_fmt", "{pts}",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]

        class SegDecoder:
            def __init__(self, i):
                self.i = i
                self.a, self.b = segs[i]
                self.stats = os.path.join(work, f"stats{i}.txt")
                self.t0 = time.perf_counter()
                self.first_frame_s = None
                self.proc = subprocess.Popen(decode_cmd(self.a, self.b, self.stats),
                                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                self.q: queue.Queue = queue.Queue(maxsize=8)
                self.thread = threading.Thread(target=self._read, daemon=True)
                self.thread.start()

            def _read(self):
                while True:
                    buf = self.proc.stdout.read(frame_bytes)
                    if len(buf) < frame_bytes:
                        self.q.put(None)
                        return
                    if self.first_frame_s is None:
                        self.first_frame_s = time.perf_counter() - self.t0
                    self.q.put(buf)

            def frames(self):
                while (buf := self.q.get()) is not None:
                    yield buf
                self.proc.wait()

            def check(self, n):
                pts = [int(x) for x in open(self.stats).read().split()]
                problems = []
                if len(pts) != n:
                    problems.append(f"stats {len(pts)} != frames {n}")
                if self.a is not None and (not pts or pts[0] != self.a):
                    problems.append(f"first pts {pts[:1]} != segment start {self.a}")
                if self.b is not None and pts and max(pts) >= self.b:
                    problems.append("pts past segment end")
                return problems

        seg_paths = [os.path.join(work, f"seg{i:05d}.mp4") for i in range(len(segs))]
        seg_frames = [0] * len(segs)
        timings = {"first_frame": [], "flush": [], "create": []}
        srs, encs = [None] * workers, [None] * workers
        ready = threading.Barrier(workers + 1)
        go = threading.Event()
        errors = []
        idr = int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR)

        def claim(i):
            # Static assignment (segment i -> session i mod N): a session's rate-control
            # history then never depends on scheduling, so the output is deterministic.
            return SegDecoder(i) if i < len(segs) else None

        def worker(w):
            try:
                srs[w] = new_sr()
                ready.wait()
                go.wait()
                enc = encs[w]
                frame_num = 0  # this session's m_frameNum
                open_segs = []  # [i, base, n or None, muxer, received]

                def route(pkts):
                    for pkt in pkts:
                        ts = pkt["timestamp"]
                        seg = next(s for s in open_segs if ts >= s[1] and (s[2] is None or ts < s[1] + s[2]))
                        seg[3].MuxVideoPacket(bytes(pkt["data"]), pkt["picture_type"], ts - seg[1])
                        seg[4] += 1
                        if seg[2] is not None and seg[4] == seg[2]:
                            seg[3].Finalize()
                            open_segs.remove(seg)

                cur = claim(w)
                while cur is not None:
                    nxt = claim(cur.i + workers)  # prefetch: starts while this one runs
                    i = cur.i
                    entry = [i, frame_num, None, new_muxer(seg_paths[i], enc, fps), 0]
                    open_segs.append(entry)
                    n = 0
                    for buf in cur.frames():
                        rgb = torch.from_numpy(np.frombuffer(buf, np.uint8).reshape(in_h, in_w, 3)).cuda()
                        p = to_p010(vsr_frame(srs[w], rgb))
                        torch.cuda.synchronize()
                        route(enc.Encode(p, idr) if n == 0 else enc.Encode(p))
                        n += 1
                    timings["first_frame"].append(round(cur.first_frame_s or -1, 2))
                    errors.extend(f"seg {i}: {p}" for p in cur.check(n))
                    entry[2] = n
                    frame_num += n
                    seg_frames[i] = n
                    if mode == "endenc" or nxt is None:
                        t = time.perf_counter()
                        route(enc.EndEncode())
                        timings["flush"].append(round(time.perf_counter() - t, 2))
                    elif entry[4] == n:
                        entry[3].Finalize()
                        open_segs.remove(entry)
                    cur = nxt
                if open_segs:
                    errors.append(f"worker {w}: unfinished segments {[s[0] for s in open_segs]}")
            except Exception:  # noqa: BLE001
                import traceback
                errors.append(f"worker {w}: {traceback.format_exc()[-600:]}")
                with contextlib.suppress(Exception):
                    ready.abort()

        ths = [threading.Thread(target=worker, args=(w,)) for w in range(workers)]
        for t in ths:
            t.start()
        ready.wait()
        attempts = []
        for w in range(workers):
            encs[w], a = make_encoder("5.1", retries=10)
            attempts.append(a)
        sps = {bytes(e.GetSequenceParams()) for e in encs}
        c0 = cpu_sample()
        t0 = time.perf_counter()
        go.set()
        for t in ths:
            t.join()
        dt = time.perf_counter() - t0
        c1 = cpu_sample()
        busy = 1 - (c1[1] - c0[1]) / max(1, c1[0] - c0[0])
        total = sum(seg_frames)
        out = os.path.join(work, "joined.mp4")
        concat(seg_paths, out)
        os.makedirs("/jobs/probe-par/out", exist_ok=True)
        tag = name.replace(":", "_")
        subprocess.run(["cp", out, f"/jobs/probe-par/out/{tag}.mp4"])
        vol.commit()
        ref = subprocess.run(["ffmpeg", "-v", "error", "-i", clip, "-map", "0:v:0", "-fps_mode",
                              "passthrough", "-f", "null", "-", "-progress", "pipe:1"],
                             capture_output=True, text=True).stdout
        ref_n = int([x for x in ref.splitlines() if x.startswith("frame=")][-1].split("=")[1])
        bounds = [sum(seg_frames[:i]) for i in range(1, len(segs))]
        n_pk, deltas, key_idx = packet_report(out)
        n, flat, laps = pixel_check(out, idx=sorted({b for b in bounds} | {b - 1 for b in bounds}))
        lines.append(f"{workers} workers, mode {mode}, {len(segs)} segments ({seg_seconds}s), cpu={os.cpu_count()}: "
                     f"{total} frames in {dt:.1f}s = {total / dt:.1f} fps; CPU busy {100 * busy:.0f}%; "
                     f"create attempts {attempts}; errors {errors}; distinct SPS {len(sps)}")
        lines.append(f"  timings: decoder first frame {timings['first_frame']} flush {timings['flush']}")
        lines.append(f"  frames: segments {seg_frames} sum={total} reference decode={ref_n}; joined "
                     f"packets={n_pk} decoded={n} pts-deltas={deltas} flat={flat}")
        lines.append(f"  keyframes(first 12)={key_idx[:12]} boundaries={bounds[:12]}")
        lines.append(f"  boundary Laplacians={laps} bytes={os.path.getsize(out)}")

    return "\n".join(lines)


@app.function(gpu="RTX-PRO-6000", cpu=6, memory=32768, timeout=3600, volumes={"/jobs": vol},
              single_use_containers=True)
def case6(name: str) -> str:
    return _run_case(name)


@app.function(gpu="RTX-PRO-6000", cpu=12, memory=32768, timeout=3600, volumes={"/jobs": vol},
              single_use_containers=True)
def case12(name: str) -> str:
    return _run_case(name)


@app.local_entrypoint()
def main(cases: str = "enc:1,enc:4,pipe:4:dev,pipe:4:stream,join:endenc,join:idr,join4:endenc,full:4") -> None:
    calls = []
    for c in cases.split(","):
        fn = case12 if c.endswith(":cpu12") else case6
        calls.append((c, fn.spawn(c.removesuffix(":cpu12").removesuffix(":cpu6"))))
    for c, call in calls:
        try:
            print(call.get(), "\n", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{c}] FAILED: {exc!r}\n", flush=True)
