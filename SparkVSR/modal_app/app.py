"""Modal App and Volume declarations for SparkVSR."""

import modal

from .config import APP_NAME, IO_VOLUME_NAME, MODEL_VOLUME_NAME

app = modal.App(APP_NAME)

models_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
io_volume = modal.Volume.from_name(IO_VOLUME_NAME, create_if_missing=True)
