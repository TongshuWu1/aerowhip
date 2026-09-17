"""Regression for a Windows status-file lock terminating PPO after a saved update."""

from concurrent.futures import ThreadPoolExecutor
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import threading

import pytest
import torch

from simulator import artifact_io


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows file sharing semantics")
@pytest.mark.parametrize("kind", ["evaluation_json"])
def test_publication_survives_native_windows_reader_lock(tmp_path, monkeypatch, kind):
    from experimental_data.io import atomic_json

    destination = tmp_path / ("latest.pt" if kind == "checkpoint" else "status.json")
    writer = atomic_json
    writer(destination, {"episodes": 1})
    old_bytes = destination.read_bytes()

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # Allow reads/writes but deny DELETE sharing, as an ordinary reader can do.
    handle = kernel.CreateFileW(str(destination), 0x80000000, 3, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value, ctypes.get_last_error()
    blocked = threading.Event()
    original = os.replace

    def observe_native_failure(source, target):
        try:
            return original(source, target)
        except OSError as error:
            if Path(target) == destination and error.winerror in (5, 32, 33):
                blocked.set()
            raise

    monkeypatch.setattr(artifact_io.os, "replace", observe_native_failure)
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            future = pool.submit(writer, destination, {"episodes": 2})
            assert blocked.wait(2), "Did not observe the real Windows lock failure"
            assert not future.done()
            assert destination.read_bytes() == old_bytes
        finally:
            assert kernel.CloseHandle(handle)
        future.result(timeout=5)
    payload = (torch.load(destination, map_location="cpu", weights_only=True)
               if kind == "checkpoint" else json.loads(destination.read_text()))
    assert payload == {"episodes": 2}


@pytest.mark.parametrize("code", [5, 32, 33])
def test_persistent_windows_denial_is_reported_without_destroying_artifacts(tmp_path, monkeypatch, code):
    source, destination = tmp_path / "new.tmp", tmp_path / "published.json"
    source.write_text("new")
    destination.write_text("old")
    calls = []

    def denied(*args):
        calls.append(args)
        error = PermissionError("still locked")
        error.winerror = code
        raise error

    monkeypatch.setattr(artifact_io.os, "replace", denied)
    with pytest.raises(PermissionError, match="still locked"):
        artifact_io.replace_with_retry(source, destination, timeout_s=.03)
    assert len(calls) >= 2
    assert source.read_text() == "new"
    assert destination.read_text() == "old"


def test_missing_source_is_not_retried(tmp_path, monkeypatch):
    def unexpected_sleep(*args):
        pytest.fail("Unrelated I/O failure must be reported immediately")

    monkeypatch.setattr(artifact_io.time, "sleep", unexpected_sleep)
    with pytest.raises(FileNotFoundError):
        artifact_io.replace_with_retry(tmp_path / "missing", tmp_path / "destination")
