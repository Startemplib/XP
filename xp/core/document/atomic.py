"""What every save in XP owes a notebook: no leftovers, and a rename on the disk.

Two small things, both about the moment a process dies rather than about the
code being wrong:

**A hard kill leaves the temporary file.** `except BaseException` cannot run
under `taskkill /F`, so the `.partial-<pid>` a save was writing stays beside the
notebook. Nothing is lost -- the notebook is the file it was before -- but the
next save can take the debt away instead of leaving it there for a year.

**A rename is durable on the disk only after the directory is flushed.** On
POSIX that is `fsync` on the directory; Windows has no handle that can be
flushed for a directory, so the honest answer there is "asked, not available",
which is why this returns a bool rather than pretending.
"""
from __future__ import annotations

import os
import errno
import gzip
import tempfile
from contextlib import contextmanager
from pathlib import Path


def sweep_partials(target, keep_pid: int | None = None) -> list:
    """Remove `.partial-<pid>` files a previous run left beside `target`.

    Only files named after **this** target are looked at, and the current
    process's own temporary file is kept -- it may be the one being written.
    A failure to remove one is not a failure to save: the notebook is fine and
    a leftover is a leftover.
    """
    target = Path(target)
    folder = target.parent
    if not folder.is_dir():
        return []
    keep = os.getpid() if keep_pid is None else keep_pid
    prefix = target.name + '.partial-'
    removed = []
    for child in folder.iterdir():
        if not child.name.startswith(prefix):
            continue
        tail = child.name[len(prefix):]
        pid_text = tail.split('-', 1)[0]
        if not pid_text.isdecimal() or int(pid_text) <= 0:
            continue
        if int(pid_text) == keep or _process_may_be_alive(int(pid_text)):
            continue
        try:
            child.unlink()
            removed.append(child)
        except OSError:
            pass
    return removed


def fsync_directory(folder) -> bool:
    """Flush the directory entry a rename changed, where the system has one.

    Returns whether it could be done: True on POSIX, False on Windows (a
    directory cannot be opened for flushing there -- `os.fsync` on a directory
    handle raises), and False if the folder could not be opened at all. The
    caller is saving either way; this is about surviving a power cut, not about
    correctness of the bytes.
    """
    try:
        handle = os.open(str(folder), os.O_RDONLY)
    except OSError:
        return False
    try:
        os.fsync(handle)
        return True
    except OSError:
        return False
    finally:
        os.close(handle)


def _process_may_be_alive(pid: int) -> bool:
    """Unknown/access-denied means keep the file, not permission to delete it."""
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER: absent PID
        kernel.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno != errno.ESRCH
    return True


@contextmanager
def atomic_gzip(path, compresslevel=1):
    """Close the gzip trailer, sync the complete file, then replace the target.

    Directory syncing is best-effort on platforms that support it. This does
    not promise power-loss durability on every filesystem.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sweep_partials(target)
    fd, name = tempfile.mkstemp(prefix=target.name + '.partial-%d-' % os.getpid(),
                                dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as raw:
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb',
                               compresslevel=compresslevel) as stream:
                yield stream
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, target)
        fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
