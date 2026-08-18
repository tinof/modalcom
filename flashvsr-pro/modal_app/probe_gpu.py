"""Probe the real hardware after an image-stack change.

    modal run -m modal_app.probe_gpu

Runs on both GPU tiers and reports torch/CUDA versions, device capability, and
whether the Block-Sparse-Attention kernels import and produce a finite forward
pass. This is the go/no-go gate for a torch or CUDA bump: an image that builds
cleanly can still ship kernels that compute garbage, and that failure is
invisible until you look at the output.
"""

from .app import app
from .config import GPU_FULL, GPU_TINY
from .image import flashvsr_image


def _probe() -> dict:
    import subprocess

    import torch

    report = {
        # str() matters: torch.__version__ is a TorchVersion object, and pickling
        # it back to a torch-less local client fails deserialization.
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(0),
        "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
    }

    # NVENC availability gates hardware video encode in video_io.py.
    try:
        encoders = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30
        ).stdout
        report["nvenc"] = ",".join(
            e for e in ("h264_nvenc", "hevc_nvenc", "av1_nvenc") if e in encoders
        ) or "none in ffmpeg build"
    except Exception as exc:  # noqa: BLE001
        report["nvenc"] = f"probe failed: {exc}"

    try:
        import block_sparse_attn  # noqa: F401

        report["block_sparse_attn_import"] = "ok"
    except Exception as exc:  # noqa: BLE001
        report["block_sparse_attn_import"] = f"FAILED: {exc}"
        return report

    # A small attention call mirroring diffsynth's flash_attention() wrapper
    # (wan_video_dit.py): proves the kernels run on this architecture and return
    # finite values, not just that they linked.
    try:
        from block_sparse_attn import block_sparse_attn_func

        heads, seqlen, dim = 4, 256, 64
        q, k, v = (
            torch.randn(seqlen, heads, dim, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        )
        cu_seqlens = torch.tensor([0, seqlen], device="cuda", dtype=torch.int32)
        head_mask_type = torch.zeros(heads, device="cuda", dtype=torch.int32)
        out = block_sparse_attn_func(
            q, k, v,
            cu_seqlens, cu_seqlens,
            head_mask_type,
            None,  # streaming_info
            None,  # base_blockmask (dense heads)
            seqlen, seqlen,
            0.0,
            deterministic=False,
            softmax_scale=None,
            is_causal=False,
            exact_streaming=False,
            return_attn_probs=False,
        )
        report["forward"] = "ok" if torch.isfinite(out).all() else "FAILED: non-finite output"
    except Exception as exc:  # noqa: BLE001
        report["forward"] = f"FAILED: {exc}"

    return report


@app.function(image=flashvsr_image, gpu=GPU_TINY)
def probe_tiny() -> dict:
    return _probe()


@app.function(image=flashvsr_image, gpu=GPU_FULL)
def probe_full() -> dict:
    return _probe()


@app.local_entrypoint()
def main():
    probes = [(f"tiny ({GPU_TINY})", probe_tiny)]
    # Both tiers currently pin the same GPU; skip the redundant second container.
    if GPU_FULL != GPU_TINY:
        probes.append((f"full ({GPU_FULL})", probe_full))
    for name, fn in probes:
        print(f"\n=== {name} ===")
        for key, value in fn.remote().items():
            print(f"  {key}: {value}")
