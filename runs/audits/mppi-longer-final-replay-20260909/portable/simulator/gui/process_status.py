"""Read local worker liveness for the current training workspaces."""
import ctypes
from ctypes import wintypes
import os


def process_is_running(pid: int) -> bool:
    """Return whether a local process still exists without signaling it."""

    if int(pid) <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(int(pid), 0)
        except (OSError, ProcessLookupError):
            return False
        return True
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    try:
        exit_code = wintypes.DWORD()
        return bool(
            kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            and exit_code.value == 259
        )
    finally:
        kernel32.CloseHandle(handle)
