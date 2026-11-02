"""Collect Explorer's per-file invocations into one reviewed merge selection.

The stdlib-only launcher submits atomic requests before importing imaging/Tk.
An OS file lock elects one dashboard; no service survives after it closes.
"""
import json
import os
from pathlib import Path
import time
import uuid

# 'arrange' is here rather than short-circuited in the launcher because
# Explorer invokes the verb once per selected file (MultiSelectModel is
# Document), and this queue is the only thing that puts the selection back
# together. Arranger wants the whole selection in one call.
ACTIONS = {'open', 'merge', 'diff', 'auto', 'live', 'arrange'}


def submit(directory, action, paths):
    if action not in ACTIONS:
        raise ValueError('Unknown Explorer action.')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (uuid.uuid4().hex + '.json')
    temporary = target.with_suffix('.tmp')
    # `at` is what orders the selection, and it has to be written here.
    # Explorer invokes the verb once per file, microseconds apart, and the
    # file's own mtime is too coarse to tell those apart -- three requests can
    # share a tick, and then the order is whatever the directory enumerates,
    # which is a random uuid. For the arranger that is the page order, so it
    # showed up as pages arriving shuffled (and as a test that passed alone
    # and failed under load, 2026-10-21).
    temporary.write_text(json.dumps({'action': action, 'at': time.time_ns(),
                                     'paths': [str(Path(p).resolve()) for p in paths]}), encoding='utf-8')
    os.replace(temporary, target)
    return target


def try_lock(directory):
    import msvcrt
    path = Path(directory) / 'dashboard.lock'
    handle = path.open('a+b')
    if path.stat().st_size == 0:
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        return None
    return handle


def newest_source(root, patterns=('xp/**/*.py', 'xp_launcher.py')) -> float:
    """When the newest line of the program was written, as a timestamp.

    **A dashboard is a long-lived process.** Explorer's menu hands every later
    click to whichever dashboard holds the lock, so one window left open all
    evening serves the evening's code: the user's words were "the XP I use, click
    menu, arrives same, didn't update", and that is exactly why -- the window
    answering the menu had been open since 19:50, before the geometry work.
    """
    newest = 0.0
    base = Path(root)
    for pattern in patterns:
        for path in base.glob(pattern):
            try:
                newest = max(newest, path.stat().st_mtime)
            except OSError:
                continue
    return newest


def in_the_order_they_were_made(pending):
    """Requests oldest first, by the time they were made.

    The files are named by UUID, and sorting by name put `first.xopp` after
    `second.xopp` about half the time -- which decides the order the arranger
    combines them in and which side of a merge is the left one. The request
    carries the time it was made; the name is only a tie-break.
    """
    return sorted(pending.items(),
                  key=lambda item: (item[1].get('at') or 0.0, item[0].name))


class ShellInbox:
    """What serves Explorer's clicks while this dashboard is the one running.

    `started_at` is when this process was built; `on_stale` is called once if a
    request arrives for code that has been rewritten since. The dashboard
    cannot run the new code without restarting, so it stops serving, gives the
    lock back and says so, rather than answering with what it was built from.
    """

    def __init__(self, app, directory, settle=.7, started_at=None,
                 source_root=None, on_stale=None):
        self.app, self.directory, self.settle = app, Path(directory), settle
        self.pending = {}
        self.changed = 0
        self.last_merge = None
        self.started_at = started_at if started_at is not None else time.time()
        self.source_root = source_root
        self.on_stale = on_stale
        self.stale = False

    def code_is_newer(self) -> bool:
        if self.source_root is None:
            return False
        try:
            return newest_source(self.source_root) > self.started_at + 1.0
        except Exception:
            return False

    def _go_stale(self) -> None:
        if self.stale:
            return
        self.stale = True
        if self.on_stale is not None:
            try:
                self.on_stale()
            except Exception:
                pass

    def poll(self):
        if self.stale:
            # Nothing more is served from this process; the lock has been
            # given back so the next click starts a fresh one.
            return
        # **Only a click can make it stale**, not a timer. A window being used
        # while the code is being edited must not interrupt with a dialog about
        # a change it cannot see yet; the moment a *request* arrives and the
        # code is newer, the request waits on disk for a fresh process.
        if self.pending or any(self.directory.glob('*.json')):
            if self.code_is_newer():
                self._go_stale()
                return
        for path in sorted(self.directory.glob('*.json'), key=lambda p: p.name):
            if path in self.pending:
                continue
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                if data['action'] not in ACTIONS or not isinstance(data['paths'], list):
                    raise ValueError('Invalid shell request')
                self.pending[path] = data
                self.changed = time.monotonic()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.app._print(f'Could not read Explorer request: {exc}\n')
                path.rename(path.with_suffix('.invalid'))
        if self.pending and time.monotonic() - self.changed >= self.settle and not self.app._busy:
            groups = {}
            # **In the order the user clicked.** The files are named by UUID, and
            # sorting by name put `first.xopp` after `second.xopp` about half
            # the time -- which decides the order the arranger combines them in
            # and which side of the merge is the left one. The request carries
            # the time it was made; that is the order that means something.
            for path, request in in_the_order_they_were_made(self.pending):
                group = groups.setdefault(request['action'], [])
                for source in request['paths']:
                    if source not in group:
                        group.append(source)
            for action, paths in groups.items():
                reused = False
                if action == 'merge' and self.last_merge is not None:
                    try:
                        reused = self.last_merge.add_sources(paths)
                    except Exception:
                        self.last_merge = None
                if not reused:
                    opened = self.app.open_sources(paths, action=action)
                    if action == 'merge':
                        self.last_merge = opened
            for path in self.pending:
                path.unlink(missing_ok=True)
            self.pending.clear()
        self.app.root.after(150, self.poll)

