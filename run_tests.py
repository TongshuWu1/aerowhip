"""Run the complete repository regression suite from PyCharm or a terminal."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    # Optional subsystem names keep everyday checks small without dropping
    # coverage. With no arguments, run every maintained subsystem.
    groups = {'physics', 'calibration', 'training', 'flight', 'ui'}
    requested = sys.argv[1:]
    unknown = set(requested) - groups
    if unknown:
        raise SystemExit(f'Choose test groups from {sorted(groups)}; unknown: {sorted(unknown)}')
    targets = [str(ROOT / 'tests' / name) for name in requested] or [str(ROOT / 'tests')]
    raise SystemExit(pytest.main(["-q", *targets]))
