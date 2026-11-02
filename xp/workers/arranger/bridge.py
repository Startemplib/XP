"""Start Arranger, which is in this tree and is still its own process.

Arranger used to be a separate project beside XP, and this module was the
adapter that went looking for it. The program moved in on 2026-10-21; what
stayed is the process boundary, because that is the part that was earning its
keep: Arranger has its own Tk root, and the dashboard's loop is already busy.
A fault in one still cannot take the other down, and XP still does not wait
for it or learn what it wrote.

So this is now a small thing: `pythonw -m xp.workers.arranger <paths>`, run
from the checkout, detached, with nothing piped.

What it is for: taking whole pages out of any number of `.xopp` files,
reordering them, and writing one new notebook. A different job from merging
two documents' strokes, so **Arranger is its own verb**, never guessed at from
the file type -- the same `.xopp` can perfectly well be merged instead.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time


def arranger_dir(start: Path = None) -> Path:
    """The directory the program is started from: the checkout.

    It is `-m xp.workers.arranger` now, so what the child needs on its path is
    the folder holding the `xp` package, not a project of its own. Kept under
    the old name because the launcher, the dashboard and the tests all ask for
    it by that name, and it still answers the same question: where does this
    get run from.

    `XP_ARRANGER` still overrides, for an XP installed somewhere unusual.
    """
    override = os.environ.get('XP_ARRANGER')
    if override and (Path(override) / 'xp').is_dir():
        return Path(override)
    from xp.runtime.paths import project_root
    return project_root(start)


def available() -> bool:
    """Whether Arrange can be offered at all, so a menu can leave it out."""
    try:
        arranger_dir()
        return True
    except FileNotFoundError:
        return False


def _pythonw() -> Path:
    """The interpreter with no console attached.

    A plain `python` child opens a console window, which is the one thing the user
    has said the user does not want from a shell verb. Arranger hides its own console
    at startup as well, but the flash is avoidable from this side.
    """
    candidate = Path(sys.executable).with_name('pythonw.exe')
    return candidate if candidate.is_file() else Path(sys.executable)


def notebooks_only(paths):
    """The paths Arranger can actually take: `.xopp` files, and folders.

    The menu offers the same three tools on both file types, so a PDF can
    reach the arranger. Arranger would open, skip it, and sit there empty --
    which looks like a broken program rather than a wrong choice. The caller
    uses this to say so instead.

    Judged by **type, not by existence**: a path that is missing is a
    different complaint, and Arranger makes it itself ("skipped: ..."). Saying
    "this is a PDF" about a notebook that merely is not there would be a
    confident wrong answer.
    """
    wanted = []
    for value in paths:
        path = Path(value).expanduser()
        if path.suffix.lower() == '.xopp' or path.is_dir():
            resolved = path.resolve()
            if resolved not in wanted:
                wanted.append(resolved)
    return wanted


def gather_from_shell(directory, settle: float = 0.7, clock=None, sleep=None):
    """Collect the `arrange` requests Explorer has just queued, as one list.

    Explorer invokes a verb once per selected file (MultiSelectModel is
    Document), so five pages chosen together arrive as five separate processes
    a few tenths of a second apart. Arranger wants them in one call, and this
    is the collection -- without Tk, because Arrange opens no XP window and
    importing Tk to wait 0.7 s would open one.

    Returns `(paths, served, others)`: the selection in the order Explorer
    queued it, the request files that may now be deleted, and how many
    requests for *other* actions are still waiting -- the caller must not eat
    those, and if there are any it has to go on and open the dashboard that
    will serve them.
    """
    import json

    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    directory = Path(directory)
    deadline = clock() + settle
    while clock() < deadline:
        sleep(min(0.05, max(0.0, deadline - clock())))

    # Read first, then order by the timestamp the request carries. Ordering by
    # the file's own mtime looked equivalent and is not: Explorer's per-file
    # invocations land microseconds apart, several share one mtime tick, and
    # the tie is then broken by a random uuid filename. For the arranger that
    # is the page order.
    queued = []
    for path in directory.glob('*.json'):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        queued.append((data.get('at') or path.stat().st_mtime_ns, path, data))

    # Ties are the normal case, not the exception: Windows' wall clock moves in
    # ~15 ms steps, so five invocations a few hundred microseconds apart all
    # carry the same `at`. Measured: ordering on the timestamp alone put the
    # pages out of order in 40 of 200 runs, because the tie fell through to a
    # random uuid filename. Settling a tie by path makes it 0 in 400.
    #
    # **What that promises, and what it does not.** It promises a determinate
    # order -- alphabetical by path. It does NOT promise the order the files
    # appear in the user's Explorer window, and nothing here could: Explorer may be
    # sorting by date, size or type, and with MultiSelectModel=Document it
    # starts one process per file without guaranteeing that the starting order
    # matches the displayed one. There is nothing truer available to sort on.
    # Reordering pages is what the arranger is for, so a predictable order the user
    # can then drag is the honest contract, and the one the README states.
    def arrived(row):
        at, _path, data = row
        return (at, [str(p) for p in (data.get('paths') or [])])

    paths, served, others = [], [], 0
    for _at, path, data in sorted(queued, key=arrived):
        if data.get('action') != 'arrange':
            others += 1
            continue
        for source in data.get('paths', []):
            if source not in paths:
                paths.append(source)
        served.append(path)
    return paths, served, others


def launch_arranger(paths, start: Path = None) -> subprocess.Popen:
    """Start Arranger on these files or folders and return immediately.

    Not waited on and not piped, both deliberately: the user arranges, saves, and
    Arranger exits on its own, and a pipe nobody drains fills up and blocks the
    child. Arranger prints no output path, so if XP ever needs the result it
    watches the folder for a new `*-ArraN.xopp` rather than reading stdout.
    """
    folder = arranger_dir(start)
    flags = 0
    if os.name == 'nt':
        # DETACHED_PROCESS so closing XP does not take the arranger with it,
        # and CREATE_NO_WINDOW for the case where only `python.exe` was found.
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(
        [str(_pythonw()), '-m', 'xp.workers.arranger', *(str(p) for p in paths)],
        cwd=str(folder),
        creationflags=flags,
        close_fds=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
