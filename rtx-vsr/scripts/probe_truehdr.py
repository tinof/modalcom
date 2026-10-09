"""TrueHDR (SDR -> HDR10) and the RGB10A2 path, probed before they went into modal_app.py.

Standalone image (CLAUDE.md), but the helpers under test are modal_app's own: the local
entrypoint reads their source with `inspect` and each case execs it, so the probe can
never drift from the production code. Frames come from the staged clip.

Cases (one single-use container each; only `encode` creates an encoder):
  api        TrueHDR/VSR 0.2.0.0 API surface: tunable setters, RGB10A2 for VSR and the
             DENOISE/DEBLUR preprocess, per-stage timing at 4K.
  stateless  Same frames through TrueHDR forwards and backwards: equal per-frame hashes
             means no temporal state, so segment assignment cannot change the output.
  threads    Four threads (own VSR + TrueHDR each, legacy stream, device-wide sync as in
             production) against a one-thread reference: exact P010 hashes.
  sizes      TrueHDR at 2880x1620, 3840x2160, 7680x4320 and 8192x4320: alpha, finite,
             channel means against the 4K result (the 16K VSR blue-channel trap).
  math       _hdr10_to_p010 against a float64 numpy reference on a real TrueHDR frame.
  encode     The whole HDR path on 240 frames: two FORCEIDR segments with the HDR10 SEI
             inserted at mux, concat + the HDR remux, ffprobe tags and side data on every
             keyframe, trace_headers, and a decode round trip against the P010 fed in.
  tenbit     A 10-bit HEVC SDR clip decoded as x2bgr10le (layout vs an rgb48 decode),
             VSR + preprocess in RGB10A2, against the 8-bit path.

    modal volume put rtx-upscaler-jobs <clip>.mkv /probe-par/input.mkv   # already staged
    modal run scripts/probe_truehdr.py [--cases api,stateless,threads,sizes,math,encode,tenbit]
"""

import inspect
import sys
from pathlib import Path

import modal

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
    .uv_pip_install("numpy==2.5.2", "torch==2.13.0", "nvidia-vfx==0.2.0.0", "PyNvVideoCodec==2.2.3")
)
app = modal.App("rtx-probe-truehdr", image=image)
vol = modal.Volume.from_name("rtx-upscaler-jobs")
CLIP = "/jobs/probe-par/input.mkv"

# modal_app helpers execed inside the container (pure functions, no Modal objects).
HELPERS = (
    "RGB10A2_ALPHA", "_unpack_rgb10a2", "_rgb10a2_alpha_ok", "_pq_nits_table",
    "_HdrLightLevels", "_hdr10_to_p010", "_rgb_to_p010", "insert_sei_before_irap_slice",
    "_SWS_MATRIX", "_probe_decode_hints", "_ffmpeg_decode_command", "_remux_color_args",
    "SDR_VUI_BSF", "HDR_VUI_BSF",
)


def _helper_source() -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import modal_app as m

    chunks = []
    for name in HELPERS:
        value = getattr(m, name)
        if inspect.isfunction(value) or inspect.isclass(value):
            chunks.append(inspect.getsource(value))
        else:
            chunks.append(f"{name} = {value!r}")
    hdr = m.parse_hdr_settings("on")
    assert hdr is not None
    return {
        # Postponed annotations: the helpers name types (HdrSettings, ...) not execed here.
        "source": "from __future__ import annotations\n\n" + "\n\n".join(chunks),
        "sei": m.hdr10_sei_nals(hdr),
        "encoder_kwargs": m._nvenc_encoder_kwargs(),
        "max_mbps": m.NVENC_MAX_MBPS,
        "level": m._hevc_level(3840, 2160),
        "hdr": (hdr.contrast, hdr.saturation, hdr.middle_gray, hdr.luminance),
    }


@app.function(gpu="RTX-PRO-6000", cpu=6, memory=32768, timeout=1800,
              volumes={"/jobs": vol}, single_use_containers=True)
def case(name: str, ctx: dict) -> str:
    import hashlib
    import json
    import subprocess
    import tempfile
    import threading
    import time
    import traceback

    import numpy as np
    import nvvfx
    import torch
    import torch.nn.functional as F

    ns: dict = {"subprocess": subprocess, "dataclasses": __import__("dataclasses")}
    exec(ctx["source"], ns)  # noqa: S102 - modal_app's own helpers, see module docstring
    unpack, alpha_ok, hdr10_to_p010 = ns["_unpack_rgb10a2"], ns["_rgb10a2_alpha_ok"], ns["_hdr10_to_p010"]
    decode_cmd = ns["_ffmpeg_decode_command"]
    contrast, saturation, middle_gray, luminance = ctx["hdr"]

    lines: list[str] = []

    def p(*parts) -> None:
        text = " ".join(str(x) for x in parts)
        print(f"[{name}] {text}", flush=True)
        lines.append(text)

    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    p(f"host: {smi}")

    def md5(t) -> str:
        return hashlib.md5(t.contiguous().cpu().numpy().tobytes()).hexdigest()

    def decode(n: int, pix_fmt: str = "rgb24", path: str = CLIP, width=1920, height=1080):
        bpp = 4 if pix_fmt == "x2bgr10le" else (6 if pix_fmt == "rgb48le" else 3)
        cmd = decode_cmd(path, True, pix_fmt=pix_fmt) if pix_fmt != "rgb48le" else \
            decode_cmd(path, True, pix_fmt="rgb48le")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frames = []
        for _ in range(n):
            buf = proc.stdout.read(width * height * bpp)
            if len(buf) < width * height * bpp:
                break
            frames.append(buf)
        proc.kill()
        proc.wait()
        return frames

    def to_float(buf):
        arr = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3)
        return torch.from_numpy(arr).cuda().permute(2, 0, 1).float().div(255.0).contiguous()

    def vsr(quality="HIGHBITRATE_ULTRA", w=3840, h=2160, encoding=None, in_size=None):
        kwargs = {} if encoding is None else {"image_encoding": encoding}
        sr = nvvfx.VideoSuperRes(getattr(nvvfx.VideoSuperRes.QualityLevel, quality), **kwargs)
        if in_size is not None:
            sr.input_width, sr.input_height = in_size
        sr.output_width, sr.output_height = w, h
        sr.load()
        return sr

    def truehdr(debanding_off=0):
        t = nvvfx.TrueHDR(contrast=contrast, saturation=saturation, middle_gray=middle_gray,
                          luminance=luminance, debanding_off=debanding_off)
        t.load()
        return t

    def run_sr(sr, x):
        return torch.from_dlpack(sr.run(x).image).clone()

    def run_hdr(t, x):
        return torch.from_dlpack(t.run(x.clamp(0.0, 1.0).contiguous()).image).clone()

    try:
        if name == "api":
            t = truehdr()
            for attr, value in (("contrast", 110), ("saturation", 90), ("middle_gray", 40),
                                ("luminance", 1000), ("debanding_off", 1)):
                try:
                    before = getattr(t, attr)
                    setattr(t, attr, value)
                    p(f"TrueHDR.{attr}: {before} -> {getattr(t, attr)} (settable)")
                except Exception as exc:  # noqa: BLE001
                    p(f"TrueHDR.{attr}: NOT settable ({type(exc).__name__}: {exc})")
            t.close()
            frames = [to_float(b) for b in decode(30)]
            enc = nvvfx.VideoSuperRes.ImageEncoding
            p("ImageEncoding:", {k: int(v) for k, v in enc.__members__.items()})
            for quality in ("HIGHBITRATE_ULTRA", "DENOISE_ULTRA", "DEBLUR_ULTRA"):
                same = quality.startswith(("DENOISE", "DEBLUR"))
                w, h = (1920, 1080) if same else (3840, 2160)
                for in_size in (None, (1920, 1080)):
                    try:
                        sr = vsr(quality, w, h, enc.RGB10A2, in_size)
                        x = torch.zeros((1080, 1920), dtype=torch.int32, device="cuda")
                        x |= ns["RGB10A2_ALPHA"]
                        x[:, :] |= (512 | (512 << 10) | (512 << 20))
                        out = torch.from_dlpack(sr.run(x.view(torch.uint32)).image).clone()
                        p(f"VSR {quality} RGB10A2 in_size={in_size}: out {tuple(out.shape)} {out.dtype} "
                          f"alpha_ok={bool(alpha_ok(out))} mean={unpack(out).mean(dim=(1, 2)).tolist()}")
                        sr.close()
                    except Exception as exc:  # noqa: BLE001
                        p(f"VSR {quality} RGB10A2 in_size={in_size}: FAILED {type(exc).__name__}: {exc}")
            sr = vsr()
            t = truehdr()
            levels = ns["_HdrLightLevels"]("cuda")
            for x in frames[:5]:
                hdr10_to_p010(run_hdr(t, run_sr(sr, x)), levels)
            torch.cuda.synchronize()
            stages = {"vsr": 0.0, "truehdr": 0.0, "p010": 0.0}
            for x in frames[5:]:
                t0 = time.perf_counter()
                y = run_sr(sr, x)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                packed = run_hdr(t, y)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                hdr10_to_p010(packed, levels)
                torch.cuda.synchronize()
                t3 = time.perf_counter()
                stages["vsr"] += t1 - t0
                stages["truehdr"] += t2 - t1
                stages["p010"] += t3 - t2
            n = len(frames) - 5
            p("4K ms/frame:", {k: round(1000 * v / n, 3) for k, v in stages.items()})
            p("measured MaxCLL/MaxFALL:", ns["_HdrLightLevels"].combine([levels]))

        elif name == "stateless":
            frames = [to_float(b) for b in decode(48)]
            sr, t = vsr(), truehdr()
            ups = [run_sr(sr, x) for x in frames]
            forward = [md5(run_hdr(t, u)) for u in ups]
            backward = [md5(run_hdr(t, u)) for u in reversed(ups)][::-1]
            again = md5(run_hdr(t, ups[0]))
            p(f"forward==backward per frame: {sum(a == b for a, b in zip(forward, backward))}/{len(ups)}; "
              f"frame0 after 48 others equal: {again == forward[0]}")
            fresh = truehdr()
            p(f"fresh instance frame0 equal: {md5(run_hdr(fresh, ups[0])) == forward[0]}")

        elif name == "threads":
            raw = decode(60)
            # Production order: per thread its own effects, legacy stream, a device-wide
            # sync before each "Encode" (here: hashing the P010 surface).
            def pipeline(results: list, barrier) -> None:
                sr, t = vsr(), truehdr()
                if barrier is not None:
                    barrier.wait()
                for buf in raw:
                    p010 = hdr10_to_p010(run_hdr(t, run_sr(sr, to_float(buf))))
                    torch.cuda.synchronize()
                    results.append(md5(p010))

            reference: list = []
            pipeline(reference, None)
            outs = [[] for _ in range(4)]
            barrier = threading.Barrier(4)
            threads = [threading.Thread(target=pipeline, args=(outs[i], barrier)) for i in range(4)]
            started = time.perf_counter()
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            elapsed = time.perf_counter() - started
            for i, out in enumerate(outs):
                p(f"thread {i}: {sum(a == b for a, b in zip(out, reference))}/{len(reference)} equal to reference")
            p(f"4 threads: {4 * len(raw) / elapsed:.1f} frames/s (VSR + TrueHDR + P010, no encoder)")

        elif name == "sizes":
            x = to_float(decode(1)[0])
            sizes = [(2880, 1620), (3840, 2160), (7680, 4320), (8192, 4320)]
            ref = None
            for w, h in sizes:
                try:
                    up = F.interpolate(x.unsqueeze(0), size=(h, w), mode="bicubic", align_corners=False)[0]
                    t = truehdr()
                    out = run_hdr(t, up)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(10):
                        run_hdr(t, up)
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - t0) * 100
                    rgb = unpack(out)
                    means = rgb.mean(dim=(1, 2))
                    if (w, h) == (3840, 2160):
                        ref = means
                    p(f"{w}x{h}: shape {tuple(out.shape)} alpha_ok={bool(alpha_ok(out))} "
                      f"finite={bool(torch.isfinite(rgb).all())} means={[round(v, 4) for v in means.tolist()]} "
                      f"{ms:.2f} ms/frame")
                    t.close()
                except Exception as exc:  # noqa: BLE001
                    p(f"{w}x{h}: FAILED {type(exc).__name__}: {exc}")
            p(f"4K reference means: {ref.tolist() if ref is not None else None}")

        elif name == "math":
            x = to_float(decode(1)[0])
            packed = run_hdr(truehdr(), run_sr(vsr(), x))
            p010 = hdr10_to_p010(packed).cpu().numpy().astype(np.int64) >> 6
            words = packed.view(torch.int32).cpu().numpy().astype(np.int64)
            rgb = np.stack([(words >> s) & 0x3FF for s in (0, 10, 20)]).astype(np.float64) / 1023.0
            r, g, b = rgb
            y = 0.2627 * r + 0.6780 * g + 0.0593 * b
            cb, cr = (b - y) / 1.8814, (r - y) / 1.4746
            h, w = y.shape
            def site(c):
                pad = np.pad(c, 1, mode="edge")
                across = pad[:, 0:w:2] + 2 * pad[:, 1:w + 1:2] + pad[:, 2:w + 2:2]
                return (across[0:h:2] + 2 * across[1:h + 1:2] + across[2:h + 2:2]) / 16
            ref_y = np.clip(np.round(64 + 876 * y), 0, 1023)
            ref_c = [np.clip(np.round(512 + 896 * site(c)), 0, 1023) for c in (cb, cr)]
            got_y = p010[:h]
            uv = p010[h:].reshape(h // 2, w // 2, 2)
            dy = np.abs(got_y - ref_y)
            dc = [np.abs(uv[..., i] - ref_c[i]) for i in range(2)]
            p(f"Y: max |diff| {dy.max():.0f}, frac>0 {np.mean(dy > 0):.2e}")
            p(f"Cb: max {dc[0].max():.0f} frac>0 {np.mean(dc[0] > 0):.2e}; "
              f"Cr: max {dc[1].max():.0f} frac>0 {np.mean(dc[1] > 0):.2e}")
            p(f"code ranges: Y {got_y.min()}..{got_y.max()}, C {uv.min()}..{uv.max()}")

        elif name == "encode":
            import PyNvVideoCodec as nvc

            original = torch.Tensor.__dlpack__
            torch.Tensor.__dlpack__ = lambda self, *a, **k: original(self)
            raw = decode(240)
            fps_num, fps_den = 24000, 1001
            sr, t = vsr(), truehdr()
            run_hdr(t, torch.zeros((3, 2160, 3840), device="cuda"))  # warm-up at output size
            kwargs = dict(ctx["encoder_kwargs"])
            if ctx["level"]:
                kwargs["level"] = ctx["level"]
            encoder = nvc.CreateEncoder(3840, 2160, "P010", False, **kwargs)
            rc = encoder.GetEncodeReconfigureParams()
            rc.maxBitRate = ctx["max_mbps"] * 1_000_000
            rc.vbvBufferSize = ctx["max_mbps"] * 1_000_000
            assert encoder.Reconfigure(rc)
            sei = ctx["sei"]
            work = tempfile.mkdtemp()
            seg_paths = [f"{work}/seg{i}.mp4" for i in range(2)]
            muxers = [nvc.FFmpegMuxer(path, nvc.MP4, "hevc", 3840, 2160, fps_num, fps_den,
                                      1, 90000, encoder.GetSequenceParams()) for path in seg_paths]
            for muxer in muxers:
                muxer.SetUniformPtsIncrement(round(90000 * fps_den / fps_num))
            counts = [0, 0]
            irap_with_sei = [0]
            first_packet = []

            def route(packets) -> None:
                for packet in packets or []:
                    index = packet["timestamp"]
                    seg = 0 if index < 120 else 1
                    data = bytes(packet["data"])
                    patched = insert(data)
                    if patched != data:
                        irap_with_sei[0] += 1
                    if not first_packet:
                        first_packet.append(patched)
                    muxers[seg].MuxVideoPacket(patched, packet["picture_type"], index - 120 * seg)
                    counts[seg] += 1

            insert = lambda data: ns["insert_sei_before_irap_slice"](data, sei)  # noqa: E731
            saved = {}
            held = None
            force_idr = int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR)
            started = time.perf_counter()
            for i, buf in enumerate(raw):
                p010 = hdr10_to_p010(run_hdr(t, run_sr(sr, to_float(buf))))
                torch.cuda.synchronize()
                held = None
                route(encoder.Encode(p010, force_idr) if i in (0, 120) else encoder.Encode(p010))
                held = p010
                if i in (0, 60, 120, 239):
                    saved[i] = p010.cpu().numpy().astype(np.int64) >> 6
            route(encoder.EndEncode())
            torch.cuda.synchronize()
            del held
            elapsed = time.perf_counter() - started
            for muxer in muxers:
                muxer.Finalize()
            # The mp4 is only complete once the muxer object is gone (as in production).
            muxers.clear()
            del muxer
            p(f"encoded {len(raw)} frames at {len(raw) / elapsed:.1f} fps; packets per segment {counts}; "
              f"IRAP packets given SEI: {irap_with_sei[0]}")
            listing = f"{work}/list.txt"
            Path(listing).write_text("".join(f"file '{s}'\n" for s in seg_paths))
            out = f"{work}/joined.mp4"
            remux = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listing,
                     "-map", "0:v:0", "-c:v", "copy", *ns["_remux_color_args"](True),
                     "-movflags", "+faststart", out]
            result = subprocess.run(remux, capture_output=True, text=True)
            p(f"remux rc={result.returncode} {result.stderr.strip()[-300:]}")
            stream = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                 "stream=profile,pix_fmt,level,color_space,color_primaries,color_transfer,color_range,"
                 "chroma_location,nb_read_frames", "-of", "compact", out],
                capture_output=True, text=True).stdout.strip()
            p("stream:", stream)
            frames = json.loads(subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey", "-show_frames",
                 "-show_entries", "frame=pts,key_frame:frame_side_data_list", "-of", "json", out],
                capture_output=True, text=True).stdout)["frames"]
            for fr in frames:
                kinds = [(s.get("side_data_type"), s.get("max_luminance"), s.get("max_content"),
                          s.get("max_average")) for s in fr.get("side_data_list", [])]
                p(f"keyframe pts {fr.get('pts')}: {kinds}")
            colr = Path(out).read_bytes().find(b"colrnclx")
            p(f"mp4 colr nclx box present: {colr > 0}")
            trace = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", out, "-frames:v", "1", "-c", "copy", "-bsf:v", "trace_headers",
                 "-f", "null", "-"], capture_output=True, text=True).stderr
            for key in ("max_display_mastering_luminance", "max_content_light_level",
                        "max_pic_average_light_level", "colour_primaries", "transfer_characteristics",
                        "matrix_coeffs", "chroma_sample_loc_type_top_field"):
                hits = [ln.split("]")[-1].strip() for ln in trace.splitlines() if key in ln]
                p(f"trace {key}: {hits[:2]}")
            decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", out, "-f", "rawvideo", "-pix_fmt", "p010le",
                                      "-"], capture_output=True).stdout
            per = 3840 * 2160 * 3 // 2
            for i, ref in saved.items():
                got = np.frombuffer(decoded[i * per * 2:(i + 1) * per * 2], dtype="<u2").reshape(-1, 3840)
                got = got.astype(np.int64) >> 6
                dy = np.abs(got[:2160] - ref[:2160])
                dc = np.abs(got[2160:] - ref[2160:])
                p(f"round trip frame {i}: Y max {dy.max()} mean {dy.mean():.3f}; C max {dc.max()} mean {dc.mean():.3f}")
            torch.Tensor.__dlpack__ = original

        elif name == "tenbit":
            src = "/tmp/tenbit.mkv"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", CLIP, "-frames:v", "48", "-an",
                            "-vf", "setfield=prog", "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
                            "-x265-params", "log-level=error:crf=16", src], check=True)
            hints = ns["_probe_decode_hints"](src)
            p(f"10-bit source hints: {hints}")
            packed = decode(8, "x2bgr10le", src)
            deep = decode(8, "rgb48le", src)
            words = np.frombuffer(packed[0], "<i4").reshape(1080, 1920)
            ref = np.frombuffer(deep[0], "<u2").reshape(1080, 1920, 3).astype(np.int64)
            ours = np.stack([(words >> s) & 0x3FF for s in (0, 10, 20)], -1).astype(np.int64)
            alpha = (words >> 30) & 3
            p(f"x2bgr10le frames {len(packed)}; alpha values {np.unique(alpha).tolist()}; "
              f"vs rgb48>>6 max diff {np.abs(ours - (ref >> 6)).max()}, "
              f"vs round(rgb48/64) max {np.abs(ours - np.round(ref / 64.0)).max()}")
            p(f"distinct 10-bit codes in R: {len(np.unique(ours[..., 0]))}")
            enc = nvvfx.VideoSuperRes.ImageEncoding
            sr10 = vsr(encoding=enc.RGB10A2, in_size=(1920, 1080))
            pre10 = vsr("DENOISE_ULTRA", 1920, 1080, enc.RGB10A2, (1920, 1080))
            sr8 = vsr()
            eight = decode(8, "rgb24", src)
            for i in (0, 7):
                x = torch.from_numpy(np.frombuffer(packed[i], "<i4").reshape(1080, 1920).copy()).cuda()
                x = (x | ns["RGB10A2_ALPHA"]).view(torch.uint32)
                out10 = unpack(torch.from_dlpack(sr10.run(x).image).clone())
                cleaned = torch.from_dlpack(pre10.run(x).image).clone()
                out10p = unpack(torch.from_dlpack(sr10.run(cleaned).image).clone())
                out8 = run_sr(sr8, to_float(eight[i]))
                diff = (out10 - out8).abs()
                p(f"frame {i}: 10-bit vs 8-bit VSR mean |diff| {diff.mean():.5f} (x1023 {1023 * diff.mean():.2f}), "
                  f"max {1023 * diff.max():.0f}; with DENOISE preprocess mean {(out10p - out8).abs().mean():.5f}; "
                  f"distinct R codes out10 {int(torch.unique((out10[0] * 1023).round()).numel())}")
        elif name == "tenbit_pre":
            # DENOISE/DEBLUR in RGB10A2 return words whose alpha bits are not 3: check the
            # channel order and values against the RGB8 effect on a coloured frame.
            enc = nvvfx.VideoSuperRes.ImageEncoding
            ramp = torch.linspace(0.05, 0.95, 1920, device="cuda")
            rgb = torch.stack([ramp.expand(1080, 1920), 0.6 - 0.5 * ramp.expand(1080, 1920),
                               torch.full((1080, 1920), 0.15, device="cuda")])
            codes = (rgb * 1023).round().to(torch.int32)
            packed = (codes[0] | (codes[1] << 10) | (codes[2] << 20) | ns["RGB10A2_ALPHA"]).view(torch.uint32)
            real8 = to_float(decode(1)[0])
            r8codes = (real8 * 1023).round().to(torch.int32)
            real10 = (r8codes[0] | (r8codes[1] << 10) | (r8codes[2] << 20) | ns["RGB10A2_ALPHA"]).view(torch.uint32)
            for quality in ("DENOISE_ULTRA", "DEBLUR_ULTRA", "DENOISE_LOW"):
                e10 = vsr(quality, 1920, 1080, enc.RGB10A2, (1920, 1080))
                e8 = vsr(quality, 1920, 1080)
                for label, x10, x8 in (("synthetic", packed, rgb), ("real", real10, real8)):
                    o10 = torch.from_dlpack(e10.run(x10).image).clone()
                    words = o10.view(torch.int32)
                    alphas = torch.unique((words >> 30) & 3).tolist()
                    u10 = unpack(o10)
                    o8 = run_sr(e8, x8)
                    per = (u10 - o8).abs().mean(dim=(1, 2)).tolist()
                    swapped = (u10[[2, 1, 0]] - o8).abs().mean(dim=(1, 2)).tolist()
                    p(f"{quality} {label}: alpha {alphas}; in means {x8.mean(dim=(1, 2)).tolist()}; "
                      f"10-bit means {u10.mean(dim=(1, 2)).tolist()}; 8-bit means {o8.mean(dim=(1, 2)).tolist()}; "
                      f"|10-8| per channel {[round(v, 4) for v in per]}, if R/B swapped {[round(v, 4) for v in swapped]}")
                    # Fed onward to VSR with alpha forced back to 3: same result as without?
                    sr10 = vsr(encoding=enc.RGB10A2, in_size=(1920, 1080))
                    a = unpack(torch.from_dlpack(sr10.run(o10).image).clone())
                    fixed = (o10.view(torch.int32) | ns["RGB10A2_ALPHA"]).view(torch.uint32)
                    b = unpack(torch.from_dlpack(sr10.run(fixed).image).clone())
                    p(f"  VSR on it, alpha as returned vs forced 3: max |diff| {float((a - b).abs().max()):.5f}")
                    sr10.close()
                e10.close()
                e8.close()
            # x2bgr10le vs an rgb48 decode: is the difference a bias or rounding noise?
            src = "/tmp/tenbit.mkv"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", CLIP, "-frames:v", "8", "-an",
                            "-vf", "setfield=prog", "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
                            "-x265-params", "log-level=error:crf=16", src], check=True)
            p(f"hints: {ns['_probe_decode_hints'](src)}")
            words = np.frombuffer(decode(1, "x2bgr10le", src)[0], "<i4").reshape(1080, 1920)
            ref = np.frombuffer(decode(1, "rgb48le", src)[0], "<u2").reshape(1080, 1920, 3).astype(np.float64)
            ours = np.stack([(words >> s) & 0x3FF for s in (0, 10, 20)], -1).astype(np.float64)
            diff = ours - ref / 65535.0 * 1023.0
            p(f"x2bgr10le - rgb48 (in 10-bit codes): mean {diff.mean(axis=(0, 1)).round(3).tolist()}, "
              f"mean |.| {np.abs(diff).mean(axis=(0, 1)).round(3).tolist()}, max |.| {np.abs(diff).max():.2f}")
        elif name == "decodebias":
            # Which 10-bit decode is right? Compare channel means of x2bgr10le, rgb48 and the
            # verified-exact rgb24 path (CLAUDE.md: +0.004 on a grey ramp with these flags).
            src = "/tmp/tenbit.mkv"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", CLIP, "-frames:v", "8", "-an",
                            "-vf", "setfield=prog", "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
                            "-x265-params", "log-level=error:crf=16", src], check=True)
            ramp = "/tmp/ramp10.mkv"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                            "gradients=s=1920x1080:c0=black:c1=white:x0=0:y0=0:x1=1919:y1=0:d=1:r=8,format=gray16le",
                            "-vf", "format=yuv420p10le", "-c:v", "libx265", "-x265-params",
                            "log-level=error:lossless=1", ramp], check=True)
            for label, path in (("8-bit clip", CLIP), ("10-bit clip", src), ("10-bit grey ramp", ramp)):
                words = np.frombuffer(decode(1, "x2bgr10le", path)[0], "<i4").reshape(1080, 1920)
                x2 = np.stack([(words >> sh) & 0x3FF for sh in (0, 10, 20)], -1).astype(np.float64)
                r48 = np.frombuffer(decode(1, "rgb48le", path)[0], "<u2").reshape(1080, 1920, 3) / 65535.0 * 1023
                r24 = np.frombuffer(decode(1, "rgb24", path)[0], np.uint8).reshape(1080, 1920, 3) / 255.0 * 1023
                p(f"{label}: mean codes x2bgr10le {x2.mean(axis=(0, 1)).round(2).tolist()}, "
                  f"rgb48 {r48.mean(axis=(0, 1)).round(2).tolist()}, rgb24 {r24.mean(axis=(0, 1)).round(2).tolist()}; "
                  f"x2-rgb48 {(x2 - r48).mean():.3f}, rgb48-rgb24 {(r48 - r24).mean():.3f}")
                if "ramp" in label:
                    y = np.frombuffer(subprocess.run(
                        ["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1", "-f", "rawvideo",
                         "-pix_fmt", "gray10le", "-"], capture_output=True).stdout, "<u2").reshape(1080, 1920)
                    expect = (y.astype(np.float64) - 64) / 876 * 1023
                    p(f"  ramp vs (Y-64)/876: x2bgr10le {(x2[..., 1] - expect).mean():.3f}, "
                      f"rgb48 {(r48[..., 1] - expect).mean():.3f}, rgb24 {(r24[..., 1] - expect).mean():.3f}")

        elif name.startswith("roundtrip"):
            # One encoder per container: roundtrip:hdr or roundtrip:sdr on the same frames.
            import PyNvVideoCodec as nvc

            mode = name.split(":")[1]
            original = torch.Tensor.__dlpack__
            torch.Tensor.__dlpack__ = lambda self, *a, **k: original(self)
            raw = decode(60)
            sr = vsr()
            t = truehdr() if mode == "hdr" else None
            kwargs = dict(ctx["encoder_kwargs"])
            if ctx["level"]:
                kwargs["level"] = ctx["level"]
            encoder = nvc.CreateEncoder(3840, 2160, "P010", False, **kwargs)
            rc = encoder.GetEncodeReconfigureParams()
            rc.maxBitRate = ctx["max_mbps"] * 1_000_000
            rc.vbvBufferSize = ctx["max_mbps"] * 1_000_000
            assert encoder.Reconfigure(rc)
            path = tempfile.mktemp(suffix=".mp4")
            muxer = nvc.FFmpegMuxer(path, nvc.MP4, "hevc", 3840, 2160, 25, 1, 1, 90000,
                                    encoder.GetSequenceParams())
            muxer.SetUniformPtsIncrement(3600)
            fed = {}
            held = None
            for i, buf in enumerate(raw):
                up = run_sr(sr, to_float(buf))
                p010 = hdr10_to_p010(run_hdr(t, up)) if t else ns["_rgb_to_p010"](up.clamp(0, 1).movedim(0, -1))
                torch.cuda.synchronize()
                held = None
                for pk in encoder.Encode(p010) or []:
                    muxer.MuxVideoPacket(bytes(pk["data"]), pk["picture_type"], pk["timestamp"])
                held = p010
                if i in (0, 30, 59):
                    fed[i] = p010.cpu().numpy().astype(np.int64) >> 6
            for pk in encoder.EndEncode() or []:
                muxer.MuxVideoPacket(bytes(pk["data"]), pk["picture_type"], pk["timestamp"])
            torch.cuda.synchronize()
            del held
            muxer.Finalize()
            del muxer
            decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt",
                                      "p010le", "-"], capture_output=True).stdout
            per = 3840 * 2160 * 3 // 2
            p(f"{mode}: {Path(path).stat().st_size / 1e6:.1f} MB for 60 frames, decoded {len(decoded) // (2 * per)}")
            for i, ref in fed.items():
                got = np.frombuffer(decoded[i * per * 2:(i + 1) * per * 2], "<u2").reshape(-1, 3840).astype(np.int64) >> 6
                y, ry = got[:2160], ref[:2160]
                c, rc_ = got[2160:], ref[2160:]
                shifted = np.abs(y[:, 1:] - ry[:, :-1]).mean()
                p(f"{mode} frame {i}: Y signed {(y - ry).mean():+.3f} abs {np.abs(y - ry).mean():.3f} "
                  f"(1px shift {shifted:.3f}); C signed {(c - rc_).mean():+.3f} abs {np.abs(c - rc_).mean():.3f}; "
                  f"ref Y range {ry.min()}..{ry.max()} std {ry.std():.1f}")
            torch.Tensor.__dlpack__ = original
        else:
            p("unknown case")
    except Exception:  # noqa: BLE001
        p("FAILED:\n" + traceback.format_exc())
    return "\n".join(f"[{name}] {line}" for line in lines)


@app.local_entrypoint()
def main(cases: str = "api,stateless,threads,sizes,math,encode,tenbit") -> None:
    ctx = _helper_source()
    calls = [(c, case.spawn(c, ctx)) for c in cases.split(",")]
    for c, call in calls:
        print(f"\n===== {c} =====\n{call.get()}")
