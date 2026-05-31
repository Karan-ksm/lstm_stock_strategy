"""Pytest configuration.

Prepends the project root to sys.path so tests can `import src.*`
without requiring an editable install. Lives at the project root so
pytest auto loads it for every test session.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
