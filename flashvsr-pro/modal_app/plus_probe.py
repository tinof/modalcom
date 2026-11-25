"""Probe the FlashVSR+ image on the real GPU. Run it after every image build.

    modal run -m modal_app.plus_probe

The same hardware checks as probe_gpu.py (NVENC by a real encode, the
Block-Sparse-Attention kernels against an exact reference), plus what only this
image has: upstream imports at the pinned commit, the attention kernel it
selects, and the weights it expects on the Volume.
"""

from .config import PLUS_CKPT_DIR, PLUS_GPU, PLUS_MODEL_MOUNT_PATH, PLUS_MODES, PLUS_REPO_DIR
from .gpu_probe import failures, probe
from .plus_app import plus_app, plus_model_volume
from .plus_image import plus_image


@plus_app.function(
    image=plus_image, gpu=PLUS_GPU, volumes={PLUS_MODEL_MOUNT_PATH: plus_model_volume}
)
def probe_plus() -> dict:
    import os
    import subprocess

    report = probe()
    report["upstream_commit"] = subprocess.run(
        ["git", "-C", PLUS_REPO_DIR, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip() or "unknown"
    try:
        from inference.hardware import detect_hardware, select_kernels
        from inference.runners import stream as stream_module

        capabilities = detect_hardware("cuda:0")
        report["upstream_import"] = "ok"
        report["attention_kernel_auto"] = select_kernels(capabilities, attention="auto", sparse=True)
        # plus_runtime.py replaces these two by name; a renamed upstream breaks it.
        missing = [name for name in ("_offload_dit", "_restore_dit") if not hasattr(stream_module, name)]
        report["runtime_hooks"] = f"FAILED: missing {missing}" if missing else "ok"
    except Exception as exc:  # noqa: BLE001
        report["upstream_import"] = f"FAILED: {exc}"
    wanted = [f"{mode}/{name}" for mode in PLUS_MODES for name in ("dit.safetensors", "lq_proj_in.safetensors")]
    wanted.append("TCDecoder.ckpt")
    absent = [name for name in wanted if not os.path.exists(os.path.join(PLUS_CKPT_DIR, name))]
    report["weights"] = (
        f"FAILED: missing {absent}; run `modal run -m modal_app.plus_weights`" if absent else "ok"
    )
    return report


# A unique name: two entrypoints called `main` on one App cannot be imported together.
@plus_app.local_entrypoint()
def probe_main():
    report = probe_plus.remote()
    for key, value in report.items():
        print(f"  {key}: {value}")
    failed = failures(report)
    if failed:
        raise SystemExit("Probe FAILED: " + ", ".join(failed))
