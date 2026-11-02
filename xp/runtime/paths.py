"""Where the project is, found rather than counted.

`ROOT` used to be `Path(__file__).resolve().parents[1]` in two places, which
was right while every module sat directly in `xp/`. The worker-centric layout
moved those modules one level down, and `parents[1]` quietly became `xp/`
instead of the checkout -- so the installer would have registered Explorer
commands pointing at `C:\\...\\XP\\xp\\xp_launcher.py`, a file that does not
exist. It would have looked like a clean install and done nothing when
clicked, which is the exact failure the installer's own comments were written
about after 2026-09-26.

So it is searched for instead of counted to: walk up until the launcher is
found beside the package. Counting levels is a number that has to be corrected
every time a file moves; a search is not.
"""
from __future__ import annotations

from pathlib import Path

# The file that marks the checkout. It is the Explorer entry point, so if it
# is not there the registration would be meaningless anyway.
MARKER = 'xp_launcher.py'


def project_root(start: Path = None) -> Path:
    """The directory holding `xp_launcher.py` and the `xp` package.

    Falls back to the package's own parent, which is what an installed
    distribution without the launcher looks like.
    """
    here = (start or Path(__file__)).resolve()
    for folder in here.parents:
        if (folder / MARKER).is_file() and (folder / 'xp').is_dir():
            return folder
    # `xp/runtime/paths.py` -> xp -> the thing that contains it
    package = Path(__file__).resolve().parent.parent
    return package.parent


ROOT = project_root()
