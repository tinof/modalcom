"""Weight presence checks, importable without registering any Modal entrypoint.

These live apart from `download_weights` so that `service.py` can verify the
Volume contents without also pulling that module's `@app.local_entrypoint()`
into the app graph — two entrypoints named `main` collide at import time.
"""

from pathlib import Path

from .config import (
    EMPTY_PROMPT_EMBED_PATH,
    PISASR_SD21_DIR,
    PISASR_WEIGHTS_PATH,
    SPARKVSR_MODEL_DIR,
)


def verify_weights(require_pisasr: bool = False) -> list[str]:
    """Check that required weight files and precomputed tensors are present.

    Args:
        require_pisasr: also require the PiSA-SR reference model. Only the
            `pisasr` reference mode needs it, so this is off by default —
            a container serving `no_ref` or `api` jobs must still start.

    Returns a list of missing descriptions (empty if all verified).
    """
    missing = []

    spark_dir = Path(SPARKVSR_MODEL_DIR)
    if not spark_dir.is_dir():
        missing.append(f"SparkVSR directory missing: {SPARKVSR_MODEL_DIR}")
    else:
        for sub in ("transformer", "vae", "scheduler"):
            subdir = spark_dir / sub
            if not subdir.is_dir() or not any(subdir.iterdir()):
                missing.append(f"SparkVSR component missing or empty: {subdir}")

    embed_path = Path(EMPTY_PROMPT_EMBED_PATH)
    if not embed_path.is_file() or embed_path.stat().st_size == 0:
        missing.append(f"Empty-prompt embedding missing: {EMPTY_PROMPT_EMBED_PATH}")

    if require_pisasr:
        missing.extend(missing_pisasr_weights())

    return missing


def missing_pisasr_weights() -> list[str]:
    """Return descriptions of any absent PiSA-SR component."""
    missing = []

    sd21_dir = Path(PISASR_SD21_DIR)
    if not sd21_dir.is_dir() or not any(sd21_dir.iterdir()):
        missing.append(f"PiSA-SR SD2.1 base weights missing: {PISASR_SD21_DIR}")

    pisa_pkl = Path(PISASR_WEIGHTS_PATH)
    if not pisa_pkl.is_file() or pisa_pkl.stat().st_size == 0:
        missing.append(
            f"PiSA-SR pisa_sr.pkl missing: {PISASR_WEIGHTS_PATH} "
            "(operator-supplied; see modal_app/download_weights.py docstring)"
        )

    return missing
