"""Deploy entrypoint of the FlashVSR+ app: the GPU worker and the orchestrator.

    modal deploy -m modal_app.plus_deploy      # from the project directory

A Modal deploy registers only the objects reachable from the module it imports.
`plus_episode` deliberately does not import `plus_service` (it must stay off the
GPU image), so this module imports both.
"""

from .plus_app import plus_app
from .plus_episode import run_job
from .plus_service import PlusWorker

__all__ = ["PlusWorker", "plus_app", "run_job"]
