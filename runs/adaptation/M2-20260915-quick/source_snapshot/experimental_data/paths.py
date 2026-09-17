"""Canonical recording locations with read compatibility for older checkouts."""
from pathlib import Path


def flight_batch_roots(root):
    root = Path(root)
    seen = set()
    folders = []
    for relative in ('data/flight_batches', 'rehearsal_csv_and_result_in_real_flight'):
        folder = root / relative
        identity = folder.resolve()
        if folder.is_dir() and identity not in seen:
            folders.append(folder)
            seen.add(identity)
    return folders


def flight_batch_root(root):
    folders = flight_batch_roots(root)
    return folders[0] if folders else Path(root) / 'data/flight_batches'
