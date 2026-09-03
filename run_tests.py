"""Run the complete repository regression suite from PyCharm or a terminal."""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", str(ROOT / "tests")]))
