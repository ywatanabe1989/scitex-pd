"""Pytest fixtures and rootdir marker for this package.

An empty conftest.py at tests/ is the canonical SciTeX
convention (audit-project PS208) — it pins the pytest
rootdir and gives downstream fixtures a home.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `_helpers` importable from every test file under tests/.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
