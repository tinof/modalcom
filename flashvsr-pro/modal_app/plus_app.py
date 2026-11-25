"""Modal App and Volumes of the FlashVSR+ segment worker.

A separate App from `flashvsr-pro` (the FlashVSR v1.1 HTTP service in `app.py`):
the two have different images, weights and entry points, and each can be
deployed or stopped without touching the other.
"""

import modal

from .config import PLUS_APP_NAME, PLUS_IO_VOLUME_NAME, PLUS_MODEL_VOLUME_NAME

plus_app = modal.App(PLUS_APP_NAME)

# Volumes v2: many GPU containers commit their own segment files at once, which
# is the workload v2 is built for.
plus_model_volume = modal.Volume.from_name(PLUS_MODEL_VOLUME_NAME, create_if_missing=True, version=2)
plus_io_volume = modal.Volume.from_name(PLUS_IO_VOLUME_NAME, create_if_missing=True, version=2)
