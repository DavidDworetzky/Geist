"""Bridge passwords in the operating system credential store, never in SQLite."""

from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
import sys


SERVICE = b"Geist Proton Bridge"


class EmailCredentialError(RuntimeError):
    pass


def _mac(reference: str, password: str | None, delete: bool) -> str:
    security = ctypes.CDLL("/System/Library/Frameworks/Security.framework/Security")
    core = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    ptr = ctypes.c_void_p
    uint = ctypes.c_uint32
    security.SecKeychainFindGenericPassword.argtypes = [
        ptr,
        uint,
        ctypes.c_char_p,
        uint,
        ctypes.c_char_p,
        ctypes.POINTER(uint),
        ctypes.POINTER(ptr),
        ctypes.POINTER(ptr),
    ]
    security.SecKeychainAddGenericPassword.argtypes = [
        ptr,
        uint,
        ctypes.c_char_p,
        uint,
        ctypes.c_char_p,
        uint,
        ctypes.c_char_p,
        ctypes.POINTER(ptr),
    ]
    security.SecKeychainItemModifyAttributesAndData.argtypes = [ptr, ptr, uint, ctypes.c_char_p]
    security.SecKeychainItemDelete.argtypes = [ptr]
    security.SecKeychainItemFreeContent.argtypes = [ptr, ptr]
    core.CFRelease.argtypes = [ptr]
    account = reference.encode()
    length, data, item = uint(), ptr(), ptr()
    status = security.SecKeychainFindGenericPassword(
        None,
        len(SERVICE),
        SERVICE,
        len(account),
        account,
        ctypes.byref(length),
        ctypes.byref(data),
        ctypes.byref(item),
    )
    try:
        if status == -25300:
            if delete:
                return ""
            if password is None:
                raise EmailCredentialError("Bridge password is missing; reconnect the account")
            value = password.encode()
            status = security.SecKeychainAddGenericPassword(
                None, len(SERVICE), SERVICE, len(account), account, len(value), value, None
            )
        elif status == 0:
            if delete:
                status = security.SecKeychainItemDelete(item)
            elif password is not None:
                value = password.encode()
                status = security.SecKeychainItemModifyAttributesAndData(
                    item, None, len(value), value
                )
            else:
                return ctypes.string_at(data, length.value).decode()
        if status != 0:
            raise EmailCredentialError(
                "Cannot access macOS Keychain; unlock your login keychain and retry"
            )
        return ""
    finally:
        if data:
            security.SecKeychainItemFreeContent(None, data)
        if item:
            core.CFRelease(item)


def credentials(reference: str, password: str | None = None, *, delete: bool = False) -> str:
    if re.fullmatch(r"[a-f0-9]{32}", reference) is None:
        raise EmailCredentialError("Invalid Bridge credential reference")
    if sys.platform == "darwin":
        return _mac(reference, password, delete)
    if sys.platform.startswith("linux") and (command := shutil.which("secret-tool")):
        operation = "clear" if delete else "store" if password is not None else "lookup"
        args = [command, operation]
        if operation == "store":
            args += ["--label=Geist Proton Bridge"]
        args += ["service", SERVICE.decode(), "account", reference]
        try:
            result = subprocess.run(
                args, input=password, text=True, capture_output=True, timeout=15, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise EmailCredentialError("Cannot access the desktop Secret Service") from error
        if result.returncode:
            raise EmailCredentialError(
                "Cannot access the Bridge password in the desktop Secret Service"
            )
        return result.stdout.rstrip("\n") if operation == "lookup" else ""
    raise EmailCredentialError(
        "Bridge requires macOS Keychain or Linux Secret Service (secret-tool) on the machine running Geist"
    )
