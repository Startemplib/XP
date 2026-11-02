"""Learning material: one folder per merge, the three things that made it.

The user's order (2026-10-26): "Keep `Ctrl+S` as quick save and `Ctrl+Shift+S` as Save
As; keep a `Save as Learning Material` option, **checked by default**, archived
in the background -- the original PDF, the parameters, and the final XOPP. A
failed archive must not damage the file that was saved."

The converter has done this for its own samples since `converter/archive.py`;
this is the same idea for the merge, and it is a separate module for the same
reason: what a sample *is* can be tested without a window, and *when* it is made
stays in the window (`MergeApp._write_in_the_background`).

What a merge sample holds:

- `left.xopp` / `right.pdf` (or whichever way round the user merged) -- the sources,
  under their own extensions, so a person can tell a PDF from a notebook;
- `merged.xopp` -- the file this save just wrote;
- `parameters.json` -- where each side was placed, which pages paired with
  which, how many pieces each side had, and how many decisions were kept.

The folder lives inside the XP tree (`LearningMaterial/Merger/Sample_NNNN`) so
it is found again by whoever picks the project up, and it is gitignored, so a
20 MB notebook never lands in a commit.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

FOLDER = 'LearningMaterial'
PROGRAM = 'xp.merger'


def root(folder=None) -> Path:
    """Where samples go: `<XP>/LearningMaterial/Merger`, unless told."""
    if folder is not None:
        return Path(folder)
    # xp/workers/merger/archive.py -> xp -> <XP>
    return Path(__file__).resolve().parents[3] / FOLDER / 'Merger'


def next_sample(folder) -> Path:
    """The next free `Sample_NNNN`, without creating anything."""
    folder = Path(folder)
    number = 1
    while (folder / ('Sample_%04d' % number)).exists():
        number += 1
    return folder / ('Sample_%04d' % number)


def parameters_of(session, left_path, right_path, output) -> dict:
    """What this merge really did, as JSON-able values.

    Read from the session, not from the window: the window knows what to show,
    the session knows where each side was placed and which pages were paired.
    """
    fits = {}
    pairs = []
    decisions = 0
    for page_index in range(session.page_count):
        geometry = session.geometry(page_index)
        for side, fit in geometry.fits.items():
            fits.setdefault(side, []).append({
                'page': page_index,
                'scale': round(float(fit.scale), 6),
                'dx': round(float(fit.dx), 6),
                'dy': round(float(fit.dy), 6),
                'score': round(float(fit.score), 6),
                'scale_x': None if fit.sx is None else round(float(fit.sx), 6),
                'scale_y': None if fit.sy is None else round(float(fit.sy), 6),
                'reason': fit.reason,
            })
        page = session.page_at(page_index)
        pairs.append({'page': page_index, 'left': page.left, 'right': page.right})
        decisions += len(getattr(session.decisions(page_index), 'kept', {}))
    return {
        'program': PROGRAM,
        'left': Path(left_path).name,
        'right': Path(right_path).name,
        'output': Path(output).name,
        'pages': session.page_count,
        'pairing': pairs,
        'fits': fits,
        'decisions': decisions,
    }


def archive_merge(session, left_path, right_path, output, folder=None) -> Path:
    """Copy the three things into a new `Sample_NNNN` and return its path.

    Staged in `_incoming` and renamed at the end, so a copy that fails leaves no
    half-sample behind and two saves running at once cannot pick the same
    number. Raises on failure; the caller decides what to say -- the merged
    `.xopp` is already on disk and nothing here touches it.
    """
    left_path, right_path, output = (Path(left_path), Path(right_path),
                                     Path(output))
    for path, what in ((left_path, 'the left file'),
                       (right_path, 'the right file'),
                       (output, 'the merged notebook')):
        if not path.is_file():
            raise FileNotFoundError('%s is not there: %s' % (what, path))
    place = root(folder)
    place.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='_incoming-', dir=str(place)))
    try:
        shutil.copy2(left_path, staging / ('left' + left_path.suffix.lower()))
        shutil.copy2(right_path, staging / ('right' + right_path.suffix.lower()))
        shutil.copy2(output, staging / 'merged.xopp')
        (staging / 'parameters.json').write_text(
            json.dumps(parameters_of(session, left_path, right_path, output),
                       indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8')
        target = next_sample(place)
        os.replace(staging, target)
        return target
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def archive_in_the_background(session, left_path, right_path, output,
                              folder=None, on_done=None) -> 'threading.Thread':
    """The same archive, on a thread, reporting its outcome and never raising.

    The user's order: archived **in the background**, and a failed archive must not
    damage the saved file. So the caller starts this after the write succeeded,
    gets called back on whatever thread, and nothing that happens here can reach
    the file that is already on disk.
    """
    import threading

    def work():
        try:
            where: Optional[Path] = archive_merge(session, left_path, right_path,
                                                  output, folder=folder)
            if on_done is not None:
                on_done(where, None)
        except Exception as exc:            # reported, never fatal
            if on_done is not None:
                on_done(None, exc)

    thread = threading.Thread(target=work, daemon=True,
                              name='xp-learning-material')
    thread.start()
    return thread
