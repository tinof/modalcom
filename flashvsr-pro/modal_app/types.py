"""Serializable request objects shared by the web tier and the GPU workers.

This module deliberately imports nothing but the standard library. The CPU-only
web image has neither torch nor diffsynth installed, and it must not import
`service.py` (which pulls in `image.py` and its local-directory build layers).
Keeping the wire format here lets both tiers agree on it cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferenceRequest:
    """Request object for FlashVSR-Pro inference."""

    input_path: str  # Path on io_volume (relative to IO_MOUNT_PATH)
    output_path: str  # Path on io_volume (relative to IO_MOUNT_PATH)
    mode: str = "tiny"
    scale: float = 2.0
    seed: int = 0
    sparse_ratio: float = 2.0
    kv_ratio: float = 3.0
    local_range: int = 11
    color_fix: bool = False
    fps: float | None = None
    quality: int = 10
    keep_audio: bool = False
    tile_dit: bool = False
    tile_vae: bool = False
    tile_size: int = 256
    overlap: int = 24
    dtype: str = "bf16"
    job_id: str | None = None  # set for HTTP jobs; echoed back so /result can find the file
