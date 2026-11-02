"""Learning material: one folder per conversion, the three things that made it.

The user asked for the converter's Save As to be able to keep a sample -- the PDF the user
converted, the parameters that were actually used, and the `.xopp` that came
out -- so that a page the user tuned by hand is not thrown away when the window
closes. Nothing here reads or rewrites the notebook: it copies three files
into `<XP>/LearningMaterial/Converter/Sample_NNNN/` and writes the JSON.

It is its own module for one reason: *what* a sample is can then be tested
without a window, a pipeline or a 300dpi trace, and *when* it happens stays in
the window (`_offer_to_save`).

The folder is inside the XP tree and not somewhere in the system, so a sample
is found again by whoever picks this project up -- and it is gitignored, so a
20 MB notebook never lands in a commit.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import tempfile
from pathlib import Path

FOLDER = 'LearningMaterial'
PROGRAM = 'xp.converter'


def root(folder=None) -> Path:
    """Where samples go: `<XP>/LearningMaterial/Converter`, unless told."""
    if folder is not None:
        return Path(folder)
    # xp/workers/converter/archive.py -> xp -> <XP>
    return Path(__file__).resolve().parents[3] / FOLDER / 'Converter'


def next_sample(folder) -> Path:
    """The next free `Sample_NNNN`, without creating anything."""
    folder = Path(folder)
    number = 1
    while (folder / ('Sample_%04d' % number)).exists():
        number += 1
    return folder / ('Sample_%04d' % number)


def parameters_of(profile, source, page=None) -> dict:
    """The settings this conversion really ran with, as JSON-able values.

    `asdict` and not a hand-written list: a setting added to the profile later
    must not be silently missing from every sample, or the archive would be
    describing a run it did not record.
    """
    if dataclasses.is_dataclass(profile):
        values = dataclasses.asdict(profile)
    else:                                   # a plain object, still recorded
        values = {name: getattr(profile, name) for name in dir(profile)
                  if not name.startswith('_')
                  and not callable(getattr(profile, name))}
    return {
        'program': PROGRAM,
        'source': Path(source).name,
        'page': page,
        'profile': values,
    }


def archive_sample(source, output, profile, page=None, folder=None) -> Path:
    """Copy the three things into a new `Sample_NNNN` and return its path.

    Staged in `_incoming` and renamed at the end, so a copy that fails leaves
    no half-sample behind for the next reader to mistake for a real one, and
    so two conversions running at once cannot pick the same number. Raises on
    failure; the caller decides what to say -- the saved `.xopp` is already on
    disk and must not be touched by any of this.
    """
    source, output = Path(source), Path(output)
    for path, what in ((source, 'the source PDF'), (output, 'the converted notebook')):
        if not path.is_file():
            raise FileNotFoundError('%s is not there: %s' % (what, path))
    place = root(folder)
    place.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='_incoming-', dir=str(place)))
    try:
        shutil.copy2(source, staging / 'source.pdf')
        shutil.copy2(output, staging / 'output.xopp')
        (staging / 'parameters.json').write_text(
            json.dumps(parameters_of(profile, source, page),
                       indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8')
        target = next_sample(place)
        os.replace(staging, target)
        return target
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
