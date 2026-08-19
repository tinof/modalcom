"""CPU tests for the attention-backend and VAE-tiling knobs.

These guard the conditions under which a paid GPU A/B measures what it claims to. A knob
that silently does nothing would show up as "no speedup" and be wrongly discarded.
"""

import importlib
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_core(monkeypatch, **env):
    """Reimport config and core with the given environment, returning the core module."""
    for key in (
        "SPARKVSR_ATTENTION_BACKEND",
        "SPARKVSR_VAE_TILING",
        "SPARKVSR_TORCH_COMPILE",
        "SPARKVSR_FP8",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    import modal_app.config as config

    importlib.reload(config)
    import modal_app.pipeline.core as core

    importlib.reload(core)
    return core


torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")


def test_sage_context_toggles_flag(monkeypatch):
    core = _load_core(monkeypatch, SPARKVSR_ATTENTION_BACKEND="sage")
    assert core._SAGE_ACTIVE is False
    with core.sage_attention():
        assert core._SAGE_ACTIVE is True
    assert core._SAGE_ACTIVE is False


def test_native_backend_leaves_sdpa_untouched(monkeypatch):
    core = _load_core(monkeypatch, SPARKVSR_ATTENTION_BACKEND="native")
    original = core.F.scaled_dot_product_attention
    core._apply_attention_backend(object())
    assert core.F.scaled_dot_product_attention is original


def test_unknown_backend_is_rejected_not_silently_accepted(monkeypatch, capsys):
    core = _load_core(monkeypatch, SPARKVSR_ATTENTION_BACKEND="sage_hub")
    original = core.F.scaled_dot_product_attention
    core._apply_attention_backend(object())
    assert core.F.scaled_dot_product_attention is original
    assert "Unknown SPARKVSR_ATTENTION_BACKEND" in capsys.readouterr().out


def test_sage_falls_through_for_inputs_it_cannot_express(monkeypatch):
    """A masked or scaled call must reach real SDPA, never the quantised kernel."""
    core = _load_core(monkeypatch, SPARKVSR_ATTENTION_BACKEND="sage")

    calls = []
    fake = types.ModuleType("sageattention")
    fake.sageattn = lambda *a, **k: calls.append("sage") or torch.zeros(1, 1, 1, 1)
    monkeypatch.setitem(sys.modules, "sageattention", fake)

    core._apply_attention_backend(object())
    patched = core.F.scaled_dot_product_attention
    assert patched is not core._ORIGINAL_SDPA

    q = k = v = torch.randn(1, 2, 4, 8)

    # Inactive: the transformer forward is not running, so PiSA-SR and the VAE are unaffected.
    patched(q, k, v)
    # Active but carrying a mask, a scale, or dropout: not expressible in sageattn.
    with core.sage_attention():
        patched(q, k, v, attn_mask=torch.zeros(1, 2, 4, 4))
        patched(q, k, v, scale=0.5)
        patched(q, k, v, dropout_p=0.1)
    assert calls == []

    # Active and plain: this is the case that must reach the quantised kernel.
    with core.sage_attention():
        patched(q, k, v)
    assert calls == ["sage"]


def test_vae_tiling_flag_reflects_env(monkeypatch):
    assert _load_core(monkeypatch, SPARKVSR_VAE_TILING="0").VAE_TILING is False
    assert _load_core(monkeypatch, SPARKVSR_VAE_TILING="1").VAE_TILING is True
    assert _load_core(monkeypatch).VAE_TILING is True
