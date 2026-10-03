"""Standalone probe: does PyNvVideoCodec actually honour our encoder kwargs?

PyNvVideoCodec's option parser drops unknown keys silently (they land in a
map<string,string> with no validation), so "CreateEncoder did not raise" proves nothing.
The only proof a key is live is that changing it changes the encoded size.

Each variant runs in its own fresh container (`max_inputs=1`). That is not paranoia: with
several encoder sessions in one process, results were self-contradictory -- the same
settings measured 17199 KiB and 7204 KiB on different runs, and every session after the
first returned a byte-identical size regardless of settings. One session per process is
the only configuration that reproduces.

Standalone by design (own image, no `import modal_app`) per CLAUDE.md's probe guidance.

    modal run scripts/probe_gpu_encoder.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.13.0", "PyNvVideoCodec==2.2.3", "numpy==2.5.2")
)

app = modal.App("rtx-probe-gpu-encoder", image=image)

WIDTH, HEIGHT, FRAMES = 3840, 2160, 60

PRODUCTION = dict(
    codec="hevc", preset="P4", tuning_info="uhq", rc="vbr", cq="18",
    multipass="qres", tier="high", aq="10", temporalaq="", lookahead="32",
    bf="5", numrefl0="5", numrefl1="5", gop="250",
)

VARIANTS = [
    ("baseline (production)", {}),
    ("cq=32 (quality target down)", {"cq": "32"}),
    ("cq removed entirely", {"cq": None}),
    ("qp=18 (the suspect key)", {"qp": "18"}),
    ("aq=1 (weak AQ)", {"aq": "1"}),
    ("preset=P1", {"preset": "P1"}),
    ("tuning_info=ultra_high_quality", {"tuning_info": "ultra_high_quality"}),
]


# max_inputs=1 retires the container after one encode, so no NVENC session state is
# carried between variants.
@app.function(gpu="RTX-PRO-6000", timeout=1800, max_inputs=1)
def encode_once(item: tuple[str, dict]) -> tuple[str, int, float]:
    import time

    import PyNvVideoCodec as nvc
    import torch

    label, overrides = item
    settings = {k: v for k, v in {**PRODUCTION, **overrides}.items() if v is not None}

    original_dlpack = torch.Tensor.__dlpack__
    torch.Tensor.__dlpack__ = lambda self, *a, **k: original_dlpack(self)

    # Moving noise over a gradient: compressible enough that rate control has room to
    # respond, detailed enough that AQ and multipass have something to do.
    torch.manual_seed(0)
    base = torch.linspace(0, 1, WIDTH, device="cuda").expand(HEIGHT, WIDTH)
    clips = []
    for i in range(FRAMES):
        noise = torch.rand(HEIGHT, WIDTH, device="cuda") * 0.35
        y = (base * 0.6 + noise + i / (4 * FRAMES)) % 1.0
        y16 = (y * 65535).to(torch.int32).clamp(0, 65535)
        p010 = torch.empty((HEIGHT * 3 // 2, WIDTH), dtype=torch.int32, device="cuda")
        p010[:HEIGHT] = y16
        p010[HEIGHT:] = 32768
        clips.append(p010.to(torch.uint16).contiguous())

    try:
        enc = nvc.CreateEncoder(WIDTH, HEIGHT, "P010", False, **settings)
    except Exception:  # noqa: BLE001 - a probe reports failures, it does not raise
        return label, -1, 0.0

    total = 0
    started = time.perf_counter()
    for clip in clips:
        for packet in enc.Encode(clip) or []:
            total += len(bytes(packet["data"]))
    for packet in enc.EndEncode() or []:
        total += len(bytes(packet["data"]))
    elapsed = time.perf_counter() - started
    return label, total, FRAMES / max(elapsed, 1e-6)


@app.local_entrypoint()
def main() -> None:
    results = list(encode_once.map(VARIANTS, order_outputs=True))
    baseline = next((size for label, size, _ in results if label.startswith("baseline")), 0)

    print(f"\n{'variant':<34} {'size':>10}  {'fps':>6}   vs baseline")
    print("-" * 74)
    for label, size, fps in results:
        if size < 0:
            print(f"{label:<34} {'REJECTED':>10}")
            continue
        delta = 100 * (size - baseline) / max(baseline, 1)
        verdict = "" if label.startswith("baseline") else (
            "LIVE" if abs(delta) > 1.0 else "NO EFFECT -- key ignored"
        )
        print(f"{label:<34} {size / 1024:9.0f}K {fps:6.1f}   {delta:+6.1f}%  {verdict}")
    print(
        "\nExpected if the research is right: cq=32 and 'cq removed' move the size, "
        "qp=18 does not."
    )
