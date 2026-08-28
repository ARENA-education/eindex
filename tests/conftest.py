"""Make `tests/reference` (the vendored original implementation) importable as `reference`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
