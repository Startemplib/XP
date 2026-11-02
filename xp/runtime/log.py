"""One log file, because the user does not want a terminal.

The user's order: "delete that stupid terminal ... if it has an error then just write in the log. I don't
need that." So there is no console window and no console widget: everything that used to be typed
into the dashboard's console or printed into it goes to one file, and that file is the only record.

The location is `%LOCALAPPDATA%\\XP\\xp.log` -- outside the repository, because a log is not source,
and outside `Temp`, because it has to survive a cleanup the user does.
"""
from __future__ import annotations

import os
import time
from pathlib import Path


def log_path() -> Path:
    base = os.environ.get('LOCALAPPDATA') or os.environ.get('APPDATA') or os.path.expanduser('~')
    return Path(base) / 'XP' / 'xp.log'


def write(text: str) -> None:
    """Append `text` to the log. Never raises: a log that can break a run is worse than no log."""
    if not text:
        return
    try:
        target = log_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('a', encoding='utf-8') as handle:
            handle.write(text if text.endswith('\n') else text + '\n')
    except OSError:
        pass


def stamp(text: str) -> None:
    """A line with the time in front, for things that happen once per action."""
    write('%s %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), text))
