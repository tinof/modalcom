"""What a GPU container can actually do, shared by both apps' probe entrypoints.

Kept free of image definitions so either image can import it. An image that
builds cleanly can still ship kernels that compute garbage or an ffmpeg whose
NVENC cannot open a session, and both failures are invisible until the output
is looked at. This probe looks.
"""

from __future__ import annotations


def _nvenc_encode(encoder: str, extra: list[str]) -> str:
    """Run a real 8-frame encode. An encoder listed by `ffmpeg -encoders` proves
    only that the build has the code; opening a session needs a driver the
    linked Video Codec SDK supports and a GPU with an NVENC engine."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=256x256:rate=25:duration=0.32",
                "-c:v", encoder, *extra, "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"FAILED to run: {exc}"
    if result.returncode == 0:
        return "ok"
    lines = result.stderr.strip().splitlines()
    return f"FAILED: {lines[-1] if lines else 'no detail'}"


def _dense_forward(torch, function) -> str:
    heads, seqlen, dim = 4, 256, 64
    q, k, v = (
        torch.randn(seqlen, heads, dim, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )
    cu_seqlens = torch.tensor([0, seqlen], device="cuda", dtype=torch.int32)
    out = function(
        q, k, v, cu_seqlens, cu_seqlens,
        torch.zeros(heads, device="cuda", dtype=torch.int32),  # all heads dense
        None, None, seqlen, seqlen, 0.0,
        deterministic=False, softmax_scale=None, is_causal=False,
        exact_streaming=False, return_attn_probs=False,
    )
    out = out[0] if isinstance(out, tuple) else out
    return "ok" if torch.isfinite(out).all() else "FAILED: non-finite output"


def _sparse_forward(torch, function) -> dict:
    """Block-sparse heads against an exact reference.

    This is the path FlashVSR uses: every head masked by a [Q_blocks, K_blocks]
    layout of 128-token blocks. The reference is PyTorch's own attention with
    that layout expanded to a token mask. `control_gap` is how far the reference
    is from unmasked attention; it must be far larger than `max_error`, or the
    comparison could not tell a kernel that ignores the mask from a correct one.
    """
    import torch.nn.functional as F

    heads, tokens, dim, block = 4, 512, 64, 128
    blocks = tokens // block
    generator = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (
        torch.randn(1, heads, tokens, dim, device="cuda", dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    )
    mask = torch.rand(1, heads, blocks, blocks, device="cuda", generator=generator) > 0.5
    # Every query block must see at least one key block.
    mask |= torch.eye(blocks, dtype=torch.bool, device="cuda")
    # Never the full layout: that would make the mask a no-op.
    mask[..., 0, blocks - 1] = False

    def flat(x):
        return x.transpose(1, 2).reshape(tokens, heads, dim).contiguous()

    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    out = function(
        flat(q), flat(k), flat(v), cu_seqlens, cu_seqlens,
        torch.ones(heads, device="cuda", dtype=torch.int32),  # all heads block-sparse
        None, mask.contiguous(), tokens, tokens, 0.0,
        deterministic=False, softmax_scale=None, is_causal=False,
        exact_streaming=False, return_attn_probs=False,
    )
    out = out[0] if isinstance(out, tuple) else out
    out = out.reshape(1, tokens, heads, dim).transpose(1, 2).float()

    token_mask = mask.repeat_interleave(block, dim=2).repeat_interleave(block, dim=3)
    reference = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=token_mask)
    unmasked = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    scale = reference.abs().max().item()
    max_error = (out - reference).abs().max().item()
    control_gap = (unmasked - reference).abs().max().item()
    # bf16 carries about three significant digits; 5% of the output range is
    # loose for rounding and tight against a wrong mask.
    tolerance = 0.05 * scale
    passed = bool(torch.isfinite(out).all()) and max_error <= tolerance < control_gap
    return {
        "result": "ok" if passed else "FAILED",
        "max_error": round(max_error, 5),
        "tolerance": round(tolerance, 5),
        "control_gap": round(control_gap, 5),
    }


def probe() -> dict:
    import subprocess

    import torch

    report: dict = {
        # str() matters: torch.__version__ is a TorchVersion object, and pickling
        # it back to a torch-less local client fails deserialization.
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(0),
        "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
        "driver": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        ).stdout.strip() or "unknown",
        "nvenc_h264": _nvenc_encode("h264_nvenc", ["-pix_fmt", "yuv420p"]),
        "nvenc_hevc_main10": _nvenc_encode(
            "hevc_nvenc", ["-profile:v", "main10", "-pix_fmt", "p010le"]
        ),
    }

    try:
        from block_sparse_attn import block_sparse_attn_func
    except Exception as exc:  # noqa: BLE001
        report["block_sparse_attn_import"] = f"FAILED: {exc}"
        return report
    report["block_sparse_attn_import"] = "ok"

    for name, check in (("dense_forward", _dense_forward), ("sparse_forward", _sparse_forward)):
        try:
            report[name] = check(torch, block_sparse_attn_func)
        except Exception as exc:  # noqa: BLE001
            report[name] = f"FAILED: {exc}"
    return report


def failures(report: dict) -> list[str]:
    """Keys whose check did not pass."""
    failed = []
    for key, value in report.items():
        text = value.get("result", "") if isinstance(value, dict) else str(value)
        if text.startswith("FAILED"):
            failed.append(key)
    return failed
