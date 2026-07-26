"""Windows process primitives for the managed llama-server child."""

from __future__ import annotations

import ctypes
import logging
import subprocess
from typing import Any


logger = logging.getLogger(__name__)


def process_options(subprocess_module: Any = subprocess) -> dict[str, Any]:
    """Return Windows-only creation flags without exposing them to common code."""

    new_process_group = getattr(subprocess_module, "CREATE_NEW_PROCESS_GROUP", 0)
    no_window = getattr(subprocess_module, "CREATE_NO_WINDOW", 0)
    if not new_process_group or not no_window:
        raise RuntimeError("Windows subprocess creation flags are unavailable")
    return {"creationflags": new_process_group | no_window}


def attach_process_lifetime(process: subprocess.Popen[str]) -> Any | None:
    """Attach a child to a kill-on-close Job Object when Windows permits it."""

    process_handle = getattr(process, "_handle", None)
    if process_handle is None:
        return None

    from ctypes import wintypes

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JobObjectBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JobObjectExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JobObjectBasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = _kernel32()
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

    job_handle = kernel32.CreateJobObjectW(None, None)
    if not job_handle:
        logger.warning("Unable to create a Windows Job Object for llama-server")
        return None

    information = _JobObjectExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = 0x00002000
    configured = kernel32.SetInformationJobObject(
        job_handle,
        9,
        ctypes.byref(information),
        ctypes.sizeof(information),
    )
    assigned = configured and kernel32.AssignProcessToJobObject(
        job_handle,
        wintypes.HANDLE(process_handle),
    )
    if not assigned:
        error_code = _get_last_error()
        close_process_lifetime(job_handle)
        logger.warning(
            "Unable to assign llama-server to a kill-on-close Job Object (Win32 error %s)",
            error_code,
        )
        return None
    return job_handle


def close_process_lifetime(handle: Any | None) -> None:
    """Close a retained Job Object handle, terminating assigned descendants."""

    if handle is None:
        return
    from ctypes import wintypes

    kernel32 = _kernel32()
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle(handle)


def _kernel32() -> Any:
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        raise RuntimeError("Windows kernel APIs are unavailable")
    return win_dll("kernel32", use_last_error=True)


def _get_last_error() -> int:
    get_last_error = getattr(ctypes, "get_last_error", None)
    return int(get_last_error()) if get_last_error is not None else 0
