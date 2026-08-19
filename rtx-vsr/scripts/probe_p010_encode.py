"""Standalone probe: how must a P010 CUDA surface be handed to NVENC so the pixels survive?

`scripts/probe_gpu_encoder.py` proves a *kwarg* is live by watching the encoded size move.
That is blind to the failure this probe exists for: on 2026-08-18 the GPU encode path was
producing alternating-column striping on every frame (mean |Laplacian| ~192 against ~2 for
the source) while the file still decoded cleanly, carried the right frame count, and
measured a plausible size. Only decoding the output and looking at pixels catches it.

So each variant here encodes a *smooth* synthetic clip, muxes it exactly like production,
decodes it back with ffmpeg, and reports:

  * lap   -- mean |Laplacian|. Smooth source, so a correct encode stays low (~1-3).
             ~190 is the striping signature: the 16-bit surface walked as 8-bit.
  * stale -- frames whose mean brightness does not match the per-frame ramp baked into the
             source. Catches the second defect (repeated/dropped frames at batch >= 2).

One encoder session per container (`max_inputs=1`): with several sessions in one process
the numbers are self-contradictory -- see probe_gpu_encoder.py's header.

    modal run scripts/probe_p010_encode.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("torch==2.13.0", "PyNvVideoCodec==2.2.0", "numpy==2.5.2")
)

app = modal.App("rtx-probe-p010-encode", image=image)

WIDTH, HEIGHT, FRAMES, FPS = 3840, 2160, 30, 25

# Mirrors the production quality point (see NVENC_* in modal_app.py).
SETTINGS = dict(
    codec="hevc", preset="P4", tuning_info="uhq", rc="vbr", cq="18",
    multipass="qres", tier="high", aq="10", temporalaq="", lookahead="32",
    bf="5", numrefl0="5", numrefl1="5", gop="250",
)

# label, how the surface is handed to Encode(), dlpack shim, batch size, source layout.
# "chw" reproduces production's provenance: VSR yields CHW, and _upscale_rgb_batch hands
# on movedim(0, -1).clamp(...), i.e. an HWC tensor whose memory is still channel-planar.
VARIANTS = [
    ("A flat uint16 tensor + old shim (control)", "flat", "drop", 1, "hwc"),
    ("B per-plane CUDA array interface", "cai", "drop", 1, "hwc"),
    ("C flat tensor + stream-forwarding shim", "flat", "forward", 1, "hwc"),
    ("D flat tensor + no shim at all", "flat", "none", 1, "hwc"),
    ("E per-plane CAI, batch=4", "cai", "drop", 4, "hwc"),
    ("F flat uint16 tensor, batch=4", "flat", "drop", 4, "hwc"),
    ("G production layout: movedim CHW->HWC", "flat", "drop", 1, "chw"),
    ("H same, but .contiguous() first", "flat", "drop", 1, "chw_contig"),
    # I is production's real lifetime: the P010 tensor is a temporary, freed the moment
    # Encode() returns, so torch's allocator can hand the same block to the next frame's
    # conversion while NVENC is still reading it. A-H all retained every surface in a
    # list, which is why they came out clean.
    ("I P010 as a freed temporary (production)", "flat", "drop", 1, "chw_temp"),
    ("J same, but surface retained", "flat", "drop", 1, "chw"),
    # How deep does the hold need to be? NVENC's queue is roughly lookahead + bframes, so
    # a bounded ring of recent surfaces should be as good as retaining everything.
    ("K temporary, ring of 8", "flat", "drop", 1, "chw_ring8"),
    ("L temporary, ring of 48", "flat", "drop", 1, "chw_ring48"),
    ("M temporary, ring of 64", "flat", "drop", 1, "chw_ring64"),
    # NVENC is an ASIC, not a CUDA stream consumer: it can start reading the surface
    # before the conversion kernels that fill it have run. A-H and J only looked clean
    # because they built every surface up front. Sync between conversion and Encode.
    ("N temporary + sync before Encode", "flat", "drop", 1, "chw_sync"),
    ("O batch=4, temporary + sync", "flat", "drop", 4, "chw_sync"),
]


def _brightness(index: int) -> float:
    """Per-frame marker so a repeated or dropped frame is detectable after decode."""
    return 0.25 + 0.4 * index / FRAMES


@app.function(gpu="RTX-PRO-6000", timeout=1800, max_inputs=1)
def encode_once(item: tuple[str, str, str, int]) -> dict:
    import subprocess
    import tempfile
    import time
    from pathlib import Path

    import numpy as np
    import PyNvVideoCodec as nvc
    import torch

    label, surface, shim, batch, layout = item

    if shim == "drop":
        # What production does today: every argument discarded, stream included.
        original = torch.Tensor.__dlpack__
        torch.Tensor.__dlpack__ = lambda self, *a, **k: original(self)
    elif shim == "forward":
        # PyNvVideoCodec passes the stream positionally; torch 2.13 wants it keyword.
        # Forward it instead of dropping it, so the consumer gets real stream ordering.
        original = torch.Tensor.__dlpack__

        def _dlpack(self, *a, **k):
            stream = a[0] if a else k.get("stream")
            try:
                return original(self, stream=stream)
            except TypeError:
                return original(self)

        torch.Tensor.__dlpack__ = _dlpack

    def rgb_to_p010(rgb):
        """Copy of _rgb_to_p010 in modal_app.py: BT.709 limited range, 10 bits high."""
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        cb = (b - y) / 1.8556
        cr = (r - y) / 1.5748
        y10 = (64.0 + 876.0 * y).round().clamp(0, 1023)
        cb10 = (512.0 + 896.0 * cb.unfold(0, 2, 2).unfold(1, 2, 2).mean(dim=(-2, -1))).round().clamp(0, 1023)
        cr10 = (512.0 + 896.0 * cr.unfold(0, 2, 2).unfold(1, 2, 2).mean(dim=(-2, -1))).round().clamp(0, 1023)
        height, width = y10.shape
        packed = torch.empty((height * 3 // 2, width), dtype=torch.int32, device=rgb.device)
        packed[:height] = y10.to(torch.int32)
        uv = packed[height:].view(height // 2, width // 2, 2)
        uv[..., 0] = cb10.to(torch.int32)
        uv[..., 1] = cr10.to(torch.int32)
        return (packed * 64).to(torch.uint16)

    class PlaneCAI:
        """One plane of a surface, described the way NVIDIA's own samples do it.

        DeepStream Libraries' common/nvc_utils.py hands NVENC a *list* of these, one per
        plane, each carrying its own shape, byte strides and typestr. Production instead
        hands over a single flat (H*3/2, W) tensor and lets the SDK infer the layout.
        """

        def __init__(self, shape, strides, typestr, ptr):
            self.__cuda_array_interface__ = {
                "shape": tuple(int(x) for x in shape),
                "strides": tuple(int(x) for x in strides),
                "data": (int(ptr), False),
                "typestr": typestr,
                "version": 3,
            }

    def as_planes(p010):
        pitch = WIDTH * 2  # bytes per row, 16-bit samples, no padding
        base = p010.data_ptr()
        luma = PlaneCAI((HEIGHT, WIDTH, 1), (pitch, 2, 2), "<u2", base)
        chroma = PlaneCAI(
            (HEIGHT // 2, WIDTH // 2, 2), (pitch, 4, 2), "<u2", base + pitch * HEIGHT
        )
        return [luma, chroma]

    # A smooth source: correct output has a low Laplacian, so striping cannot hide. The
    # per-frame brightness ramp makes a stale or duplicated frame obvious after decode.
    ramp_x = torch.linspace(0, 1, WIDTH, device="cuda").expand(HEIGHT, WIDTH)
    ramp_y = torch.linspace(0, 1, HEIGHT, device="cuda").unsqueeze(1).expand(HEIGHT, WIDTH)
    wave = 0.15 * torch.sin(ramp_x * 6.0) * torch.cos(ramp_y * 4.0)
    ring_size = int(layout.split("ring")[1]) if "ring" in layout else 0
    sync_before_encode = layout == "chw_sync"
    retain = layout not in ("chw_temp", "chw_sync") and not ring_size
    surfaces = []
    for i in range(FRAMES):
        luma = (0.45 * ramp_x + 0.25 * ramp_y + wave).clamp(0, 1) * 0.5 + _brightness(i)
        if layout == "hwc":
            rgb = luma.clamp(0, 1).unsqueeze(-1).expand(HEIGHT, WIDTH, 3).contiguous()
        else:
            chw = luma.clamp(0, 1).unsqueeze(0).expand(3, HEIGHT, WIDTH).contiguous()
            rgb = chw.movedim(0, -1).clamp(0.0, 1.0)
            if layout == "chw_contig":
                rgb = rgb.contiguous()
        # For the temporary variant keep only the RGB and convert at Encode() time, so the
        # P010 buffer's lifetime matches production exactly.
        surfaces.append(rgb_to_p010(rgb) if retain else rgb)

    workdir = Path(tempfile.mkdtemp())
    out_path = workdir / "probe.mp4"

    try:
        encoder = nvc.CreateEncoder(WIDTH, HEIGHT, "P010", False, **SETTINGS)
    except Exception as exc:  # noqa: BLE001 - a probe reports failures, it does not raise
        return {"label": label, "error": f"CreateEncoder: {exc}"}

    muxer = nvc.FFmpegMuxer(
        out_path.as_posix(), nvc.MP4, "hevc", WIDTH, HEIGHT, FPS, 1, 1, 90000,
        encoder.GetSequenceParams(),
    )
    muxer.SetUniformPtsIncrement(round(90000 / FPS))

    def mux(packets):
        for packet in packets or []:
            muxer.MuxVideoPacket(
                bytes(packet["data"]), packet["picture_type"], packet["timestamp"]
            )

    started = time.perf_counter()
    try:
        import collections

        ring = collections.deque(maxlen=ring_size or 1)

        def feed(frame):
            if not retain:
                frame = rgb_to_p010(frame)
                if sync_before_encode:
                    torch.cuda.synchronize()
                if ring_size:
                    # Holding a reference keeps torch's allocator from handing this block
                    # to the next conversion while NVENC is still reading it.
                    ring.append(frame)
            return encoder.Encode(as_planes(frame) if surface == "cai" else frame)

        pending = []
        for item_surface in surfaces:
            pending.append(item_surface)
            if len(pending) >= batch:
                for frame in pending:
                    mux(feed(frame))
                pending = []
        for frame in pending:
            mux(feed(frame))
        mux(encoder.EndEncode())
    except Exception as exc:  # noqa: BLE001
        return {"label": label, "error": f"Encode: {type(exc).__name__}: {exc}"}
    fps = FRAMES / max(time.perf_counter() - started, 1e-6)
    muxer.Finalize()
    muxer = None  # noqa: F841 - closes the container file before ffmpeg reads it

    if not out_path.exists() or out_path.stat().st_size == 0:
        return {"label": label, "error": "muxer wrote nothing"}

    # Decode with ffmpeg rather than PyNvVideoCodec, so a decoder-side bug cannot mask an
    # encoder-side one. rgb24 matters: without it ffmpeg emits 16-bit from the 10-bit
    # stream and every downstream number is nonsense.
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", out_path.as_posix(),
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=False,
    )
    if raw.returncode != 0:
        return {"label": label, "error": f"decode: {raw.stderr.decode()[:200]}"}

    frames = np.frombuffer(raw.stdout, dtype=np.uint8)
    decoded = frames.size // (WIDTH * HEIGHT)
    if decoded == 0:
        return {"label": label, "error": "decoded 0 frames"}
    frames = frames[: decoded * WIDTH * HEIGHT].reshape(decoded, HEIGHT, WIDTH).astype(np.float32)

    laps, means = [], []
    for f in frames:
        g = f
        lap = np.abs(
            4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        )
        laps.append(float(lap.mean()))
        means.append(float(g.mean()))

    # Each source frame is brighter than the last, so decoded means must rise too. A frame
    # that repeats its predecessor (or goes flat) breaks that ordering.
    stale = sum(1 for a, b in zip(means, means[1:]) if b - a < 0.2)

    return {
        "label": label, "frames": decoded, "expected": FRAMES,
        "lap": float(np.mean(laps)), "stale": stale, "fps": fps,
        "kb": out_path.stat().st_size // 1024,
    }


@app.local_entrypoint()
def main() -> None:
    print(f"\n{'variant':<42}{'frames':>8}{'lap':>8}{'stale':>7}{'fps':>7}{'KiB':>8}   verdict")
    print("-" * 96)
    for r in encode_once.map(VARIANTS, order_outputs=True):
        if "error" in r:
            print(f"{r['label']:<42}{'FAILED':>8}   {r['error']}")
            continue
        ok_pixels = r["lap"] < 20
        ok_frames = r["frames"] == r["expected"] and r["stale"] == 0
        verdict = "CLEAN" if ok_pixels and ok_frames else (
            "STRIPED" if not ok_pixels else "FRAME LOSS/STALE"
        )
        print(
            f"{r['label']:<42}{r['frames']:>8}{r['lap']:>8.1f}{r['stale']:>7}"
            f"{r['fps']:>7.1f}{r['kb']:>8}   {verdict}"
        )
    print(
        "\nlap ~1-3 = smooth, as encoded. lap ~190 = alternating-column striping.\n"
        "stale > 0 = frames repeated or dropped (the batch >= 2 defect)."
    )
