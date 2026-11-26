"""Make `modal_app` importable when pytest runs from the project directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
