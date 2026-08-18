"""Deploy entrypoint: registers the web tier and all three GPU workers.

    modal deploy -m modal_app.deploy

A Modal deploy only registers the objects reachable from the module it imports.
`web_api` deliberately does not import `service` (it must stay off the GPU image),
so deploying either module alone would drop the other half of the app. This module
imports both.
"""

from .app import app
from .service import FlashVSRFull, FlashVSRTiny, FlashVSRTinyLong
from .web_api import api

__all__ = ["api", "app", "FlashVSRFull", "FlashVSRTiny", "FlashVSRTinyLong"]
