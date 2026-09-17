"""Publish complete artifacts despite short-lived Windows reader locks."""

from __future__ import annotations

import os
from pathlib import Path
import time


def replace_with_retry(source: str | Path, destination: str | Path, *, timeout_s: float = 5.0) -> None:
    """Keep atomic replacement; retry Windows access/sharing/lock violations.

    Windows readers can temporarily deny replacement even for writable files.
    Never truncate the published destination or hide a persistent I/O failure.
    The caller retains ownership of the temporary file if publication fails.
    """
    deadline = time.monotonic() + max(0.0, timeout_s)
    delay = 0.01
    while True:
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(0.1, delay * 2)
