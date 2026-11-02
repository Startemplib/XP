"""Where the sample material is, without anyone hardcoding it twice.

The notebooks and PDFs a conversion is tested against are the only test data
that counts, and they are large enough to get moved when a tree is tidied. So
the folder is asked for rather than assumed: walk up from wherever XP is
installed looking for it, knowing the names it may live under, so an older
checkout still works.

`XP_MATERIAL` names the folder outright and always wins.
"""
from __future__ import annotations

import os
from pathlib import Path

# In order of preference; the first that exists wins.
CANDIDATES = (
    ('doing',),                      # the folder XP's own tree keeps them in
    ('material',),                   # a plain name beside the project
    ('samples',),                    # the other plain name
)


def material(start: Path = None) -> Path:
    """The folder holding the sample notebooks and PDFs.

    Raises rather than returning something plausible: a test that silently
    runs against no material passes without having tested anything.
    """
    override = os.environ.get('XP_MATERIAL')
    if override and Path(override).is_dir():
        return Path(override)
    here = (start or Path(__file__)).resolve()
    for folder in [here] + list(here.parents):
        for tail in CANDIDATES:
            candidate = folder.joinpath(*tail)
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError(
        "cannot find the sample material; set XP_MATERIAL to the folder "
        "holding the .xopp notebooks and PDFs")


def maybe_material(start: Path = None):
    """The same, but None instead of raising -- for tests that skip politely."""
    try:
        return material(start)
    except FileNotFoundError:
        return None
