"""Probe the real hardware after an image-stack change (FlashVSR v1.1 image).

    modal run -m modal_app.probe_gpu

Reports torch/CUDA versions, the device and driver, whether NVENC can open a
session (a real encode, H.264 and HEVC Main10), and whether the
Block-Sparse-Attention kernels compute the right thing: a dense forward pass,
and a block-sparse one compared against an exact reference. This is the go/no-go
gate for a torch or CUDA bump. The checks are in gpu_probe.py; plus_probe.py
runs the same ones on the FlashVSR+ image.
"""

from .app import app
from .config import GPU_FULL, GPU_TINY
from .gpu_probe import failures, probe
from .image import flashvsr_image


@app.function(image=flashvsr_image, gpu=GPU_TINY)
def probe_tiny() -> dict:
    return probe()


@app.function(image=flashvsr_image, gpu=GPU_FULL)
def probe_full() -> dict:
    return probe()


@app.local_entrypoint()
def main():
    probes = [(f"tiny ({GPU_TINY})", probe_tiny)]
    # Both tiers currently pin the same GPU; skip the redundant second container.
    if GPU_FULL != GPU_TINY:
        probes.append((f"full ({GPU_FULL})", probe_full))
    failed = []
    for name, fn in probes:
        report = fn.remote()
        print(f"\n=== {name} ===")
        for key, value in report.items():
            print(f"  {key}: {value}")
        failed += [f"{name}: {key}" for key in failures(report)]
    if failed:
        raise SystemExit("Probe FAILED: " + "; ".join(failed))
