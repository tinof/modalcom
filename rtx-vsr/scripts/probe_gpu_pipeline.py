"""Standalone probe of the fully GPU-resident path on real content.

Answers two questions that synthetic frames could not:

1. **Does in-process NVDEC decode work?** PyNvVideoCodec's `ThreadedDecoder` can hand back
   CUDA RGB frames directly, removing the last host round-trip (the rawvideo pipe from a
   decode subprocess). The risk is the `__dlpack__` signature: the binding accepts only
   `stream`, and torch 2.13 may pass `max_version=`, which would TypeError.
2. **Which rate-control key is live, and in which direction?** On synthetic noise `cq`
   moved the size but *inverted* (cq=18 smaller than cq=32). Real video is the only
   trustworthy signal, and bitrate parity with the piped path is what gates adopting this.

    modal run scripts/probe_gpu_pipeline.py
"""

from pathlib import Path

import modal

SAMPLE = Path(__file__).parent.parent / "sample" / "jopet_10s.mkv"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.13.0", "PyNvVideoCodec==2.2.3", "numpy==2.5.2")
    .add_local_file(SAMPLE.as_posix(), "/sample.mkv", copy=True)
)

app = modal.App("rtx-probe-gpu-pipeline", image=image)


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def probe_decoder() -> None:
    """Can we get CUDA RGB tensors out of NVDEC in-process, and how fast?"""
    import time

    import PyNvVideoCodec as nvc
    import torch

    # ThreadedDecoder is not usable in 2.2.0: it drains after exactly one frame in every
    # configuration probed (NATIVE/RGB/RGBP, every buffer and batch size, mkv and mp4).
    # The low-level demuxer + decoder pair streams the whole file, which is all we need.
    print("== decoder construction ==")
    dmx = nvc.CreateDemuxer(filename="/sample.mkv")
    dec = nvc.CreateDecoder(
        gpuid=0, codec=dmx.GetNvCodecId(), usedevicememory=1,
        outputColorType=nvc.OutputColorType.RGB,
    )

    print("\n== first frame -> torch ==")
    packets = iter(dmx)
    frames = []
    while not frames:
        frames = list(dec.Decode(next(packets)))
    f0 = frames[0]
    print(f"  DecodedFrame.shape={f0.shape} strides={f0.strides} dtype={f0.dtype}")

    # The documented integration risk: torch 2.13 may call __dlpack__ with kwargs the
    # pybind binding does not accept. Try the plain path first and report what happens.
    try:
        t = torch.from_dlpack(f0)
        print(f"  torch.from_dlpack OK -> {tuple(t.shape)} {t.dtype} {t.device} "
              f"stride={t.stride()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  torch.from_dlpack FAILED: {type(exc).__name__}: {exc}")
        print("  retrying via explicit __dlpack__(stream=...)")
        t = torch.utils.dlpack.from_dlpack(
            f0.__dlpack__(stream=torch.cuda.current_stream().cuda_stream)
        )
        print(f"  explicit-stream OK -> {tuple(t.shape)} {t.dtype} {t.device}")

    print(f"  mean per channel (should not be 0/uniform): "
          f"{t.float().mean(dim=(0, 1)).tolist()}")

    # Surfaces are recycled: batch k is valid only until get_batch_frames() for k+1.
    # Prove it, so the mandatory .clone() is justified by measurement, not by folklore.
    print("\n== surface reuse (is .clone() mandatory?) ==")
    before = t.float().mean().item()
    kept_clone = torch.from_dlpack(f0).clone()
    clone_before = kept_clone.float().mean().item()
    for _ in range(3):
        list(dec.Decode(next(packets)))
    after = t.float().mean().item()
    clone_after = kept_clone.float().mean().item()
    print(f"  unclonedview mean {before:.3f} -> {after:.3f} "
          f"({'OVERWRITTEN' if abs(before - after) > 0.01 else 'stable'})")
    print(f"  cloned tensor mean {clone_before:.3f} -> {clone_after:.3f} "
          f"({'CORRUPTED' if abs(clone_before - clone_after) > 0.01 else 'stable'})")

    print("\n== throughput, decode only ==")
    dmx2 = nvc.CreateDemuxer(filename="/sample.mkv")
    dec2 = nvc.CreateDecoder(
        gpuid=0, codec=dmx2.GetNvCodecId(), usedevicememory=1,
        outputColorType=nvc.OutputColorType.RGB,
    )
    count = 0
    started = time.perf_counter()
    for packet in dmx2:
        for fr in dec2.Decode(packet):
            torch.from_dlpack(fr).clone()
            count += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(f"  decoded {count} frames in {elapsed:.2f}s = {count / elapsed:.1f} fps")


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def probe_rate_control() -> None:
    """On REAL content: which key sets the quality target, and which way does it run?"""
    import time

    import PyNvVideoCodec as nvc
    import torch

    original_dlpack = torch.Tensor.__dlpack__
    torch.Tensor.__dlpack__ = lambda self, *a, **k: original_dlpack(self)

    dmx = nvc.CreateDemuxer(filename="/sample.mkv")
    dec = nvc.CreateDecoder(
        gpuid=0, codec=dmx.GetNvCodecId(), usedevicememory=1,
        outputColorType=nvc.OutputColorType.RGB,
    )
    rgb_frames = []
    for packet in dmx:
        for fr in dec.Decode(packet):
            # Mandatory: the decoder recycles its surfaces on the next Decode() call.
            rgb_frames.append(torch.from_dlpack(fr).clone())
        if len(rgb_frames) >= 150:
            break
    height, width = rgb_frames[0].shape[0], rgb_frames[0].shape[1]
    fps_meta = 25.0
    print(f"decoded {len(rgb_frames)} real frames at {width}x{height}\n")

    def to_p010(rgb: "torch.Tensor") -> "torch.Tensor":
        r, g, b = (rgb[..., i].float() for i in range(3))
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        u = (b - y) / 1.8556
        v = (r - y) / 1.5748
        y10 = (y * (219.0 / 255.0) + 16.0) * 4.0
        u10 = (u * (224.0 / 255.0) + 128.0) * 4.0
        v10 = (v * (224.0 / 255.0) + 128.0) * 4.0
        out = torch.empty((height * 3 // 2, width), dtype=torch.int32, device="cuda")
        out[:height] = (y10 * 64).to(torch.int32).clamp(0, 65535)
        uv = out[height:].view(height // 2, width // 2, 2)
        uv[..., 0] = (u10[::2, ::2] * 64).to(torch.int32).clamp(0, 65535)
        uv[..., 1] = (v10[::2, ::2] * 64).to(torch.int32).clamp(0, 65535)
        return out.to(torch.uint16).contiguous()

    p010 = [to_p010(f) for f in rgb_frames]
    torch.cuda.synchronize()
    seconds = len(p010) / fps_meta

    base = dict(
        codec="hevc", preset="P4", tuning_info="uhq", rc="vbr",
        multipass="qres", tier="high", aq="10", temporalaq="", lookahead="32",
        bf="5", numrefl0="5", numrefl1="5", gop="250",
    )

    print(f"{'settings':<30} {'size':>9} {'Mbps':>8} {'fps':>7}")
    print("-" * 58)
    for label, extra in [
        ("no quality target", {}),
        ("cq=14", {"cq": "14"}),
        ("cq=18 (target)", {"cq": "18"}),
        ("cq=24", {"cq": "24"}),
        ("cq=32", {"cq": "32"}),
        ("qp=18 (suspect no-op)", {"qp": "18"}),
    ]:
        try:
            enc = nvc.CreateEncoder(width, height, "P010", False, **{**base, **extra})
        except Exception as exc:  # noqa: BLE001
            print(f"{label:<30} REJECTED {str(exc).splitlines()[0][:30]}")
            continue
        total = 0
        started = time.perf_counter()
        for frame in p010:
            for pkt in enc.Encode(frame) or []:
                total += len(bytes(pkt["data"]))
        for pkt in enc.EndEncode() or []:
            total += len(bytes(pkt["data"]))
        elapsed = time.perf_counter() - started
        mbps = total * 8 / seconds / 1e6
        print(f"{label:<30} {total / 1024:8.0f}K {mbps:8.2f} "
              f"{len(p010) / max(elapsed, 1e-6):7.1f}")
        del enc

    print("\nMonotonic Mbps falling as cq rises = cq is the quality target, correct sense.")
    print("qp matching 'no quality target' = qp is ignored, as the source read predicted.")


@app.local_entrypoint()
def main() -> None:
    print("######## DECODER ########")
    probe_decoder.remote()
    print("\n######## RATE CONTROL (real content) ########")
    probe_rate_control.remote()
