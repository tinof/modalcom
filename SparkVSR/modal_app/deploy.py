"""Deploy entrypoint: registers the web tier and GPU worker with Modal.

Usage:
    modal deploy -m modal_app.deploy
"""

from .app import app
from .service import SparkVSRService, process_video_parallel
from .web_api import api

__all__ = ["api", "app", "SparkVSRService", "process_video_parallel"]
