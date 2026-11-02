"""Which code is this window running?

The question came from clicking the program and seeing the old behaviour: "I open the
program by mouse click, it didn't update". The shortcut points at `xp_launcher.py` and no
stale XP process was running, so the code *should* be current -- but there was no way to
tell from the window itself, and "it should be" is not something a person can check.

`code_version()` answers it in the title bar: the short git SHA of the tree, or the newest
modification time under `xp/` when git is not available.
"""
from __future__ import annotations

import os
import subprocess
import time
from functools import lru_cache
from pathlib import Path


def _root() -> Path:
    from xp.runtime.paths import ROOT

    return Path(ROOT)


@lru_cache(maxsize=1)
def code_version() -> str:
    """A short string that changes whenever the tree changes.

    `git rev-parse --short HEAD`, and `+` when the working tree has uncommitted changes,
    so a dirty tree cannot be mistaken for the commit. Without git: the newest mtime under
    `xp/`, to the minute.
    """
    root = _root()
    try:
        sha = subprocess.run(['git', '-C', str(root), 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        if sha:
            dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'],
                                   capture_output=True, text=True, timeout=5).stdout.strip()
            return sha + ('+' if dirty else '')
    except Exception:
        pass
    newest = 0.0
    try:
        for path in (root / 'xp').rglob('*.py'):
            newest = max(newest, path.stat().st_mtime)
    except Exception:
        pass
    stamp = time.strftime('%m-%d %H:%M', time.localtime(newest)) if newest else 'unknown'
    return 'tree ' + stamp


def title(base: str) -> str:
    """The window title with the code version on it."""
    return '%s  ·  %s' % (base, code_version())
