"""Modal App and Volume definitions for FlashVSR-Pro."""

import modal

from .config import APP_NAME, BUILD_CACHE_VOLUME_NAME, MODEL_VOLUME_NAME, IO_VOLUME_NAME

app = modal.App(APP_NAME)

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
io_volume = modal.Volume.from_name(IO_VOLUME_NAME, create_if_missing=True)
build_cache_volume = modal.Volume.from_name(BUILD_CACHE_VOLUME_NAME, create_if_missing=True)
