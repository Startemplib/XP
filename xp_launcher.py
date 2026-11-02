"""Absolute-path GUI entry for shortcuts / Explorer, independent of cwd."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback


def _say(message, title='XP'):
    """One plain sentence in a box, with no dashboard behind it.

    The menu offers the same three tools on every file type, so two
    combinations can be asked for that cannot work: a PDF handed to the
    Arranger, and a notebook handed to the Converter. Saying so is the whole
    response -- opening the dashboard to show an error would be a second
    window nobody asked for.
    """
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(title, message, parent=root)
    root.destroy()


def _claim_the_voice(requests, shell):
    """The right to show one message, for the first of Explorer's per-file
    invocations only -- so three selected files give one box, not three.

    It borrows the dashboard lock, because whoever holds that is already "the
    one process talking to the user". The handle is **returned, not discarded**:
    letting it fall out of scope closes the file and releases the lock at
    once, and then every sibling invocation would claim the voice too.

    Returns something truthy to speak with (the caller must keep it alive for
    as long as the box is up), or None to stay quiet. Outside the shell there
    are no siblings, so there is nothing to arbitrate.
    """
    if not shell:
        return True
    try:
        from xp.runtime.shellqueue import try_lock
        requests.mkdir(parents=True, exist_ok=True)
        return try_lock(requests)
    except Exception:
        return True            # cannot tell: better one message than silence


def main():
    # **Before any Tk window exists.** See `xp.render.appearance.be_dpi_aware`:
    # without it, a display scaled to 150 % gives fonts that are 150 % and rows
    # that are 100 %, which is what made the merge window's toolbar overlap.
    try:
        from xp.runtime.dpi import be_dpi_aware

        be_dpi_aware()
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--shell', action='store_true')
    parser.add_argument('--action',
                        choices=['open', 'merge', 'diff', 'auto', 'live', 'arrange'],
                        default='open')
    parser.add_argument('files', nargs='*')
    args = parser.parse_args()
    state = Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'XP'
    state.mkdir(parents=True, exist_ok=True)
    # pythonw has no stdout/stderr; preserve diagnostic errors in a local log.
    log = None
    if sys.stderr is None or sys.stdout is None:
        log = (state / 'launcher.log').open('a', encoding='utf-8', buffering=1)
        sys.stdout = sys.stderr = log
    lock = None
    try:
        # Arrange is a different program, and it is fire and forget: **no XP
        # window is opened for it**, which is the whole difference between a
        # verb and a pane. Nothing below this is imported for it -- not
        # `xp.app`, not even Tk.
        #
        # From Explorer it still has to go through the queue, because Explorer
        # invokes a verb once per selected file and Arranger wants the whole
        # selection in one call. Three cases, and all three end without a
        # dashboard unless something else needs one:
        #   * we win the lock           -> collect the sibling calls, launch once
        #   * somebody else holds it    -> that process serves ours; just leave
        #   * other actions are waiting -> fall through and open the dashboard,
        #                                  because those are not ours to eat
        if args.action in ('auto', 'live') and args.files:
            # Converter on a notebook: it is already one. Say so and stop,
            # rather than opening the dashboard to put an error box on it.
            if not any(str(f).lower().endswith('.pdf') for f in args.files):
                lock = _claim_the_voice(state / 'requests', args.shell)
                if lock:
                    _say('Converter turns a PDF into a .xopp notebook, and '
                         'this is already a notebook.\n\nUse Merger to combine '
                         'it with another document, or Arranger to reorder its '
                         'pages.', 'XP - Converter')
                return
        if args.action == 'arrange':
            from xp.workers.arranger.bridge import gather_from_shell, launch_arranger, notebooks_only
            if args.files and not notebooks_only(args.files):
                # Arranger on a PDF: it reads .xopp pages. Without this it
                # would open, skip the file and sit there empty, which looks
                # like a broken program rather than a wrong choice.
                lock = _claim_the_voice(state / 'requests', args.shell)
                if lock:
                    _say('Arranger reorders the pages of .xopp notebooks, and '
                         'this is a PDF.\n\nConvert it first with Converter, '
                         'then arrange its pages.', 'XP - Arranger')
                return
            if not args.shell:
                launch_arranger(notebooks_only(args.files) or args.files)
                return
            from xp.runtime.shellqueue import submit, try_lock
            requests = state / 'requests'
            submit(requests, 'arrange', args.files)
            lock = try_lock(requests)
            if lock is None:
                return
            selection, served, others = gather_from_shell(requests)
            if selection:
                launch_arranger(selection)
                for path in served:
                    path.unlink(missing_ok=True)
            if not others:
                lock.close()
                return
            # Somebody also asked for a merge or a conversion. Those are not
            # ours to eat, so go on and open the dashboard that serves them --
            # keeping the lock we already hold, because asking for it twice
            # from one process is refused and would strand them.
        elif args.shell:
            from xp.runtime.shellqueue import submit, try_lock
            requests = state / 'requests'
            submit(requests, args.action, args.files)
            lock = try_lock(requests)
            if lock is None:
                return
        from xp.runtime.app import App
        started_at = time.time()
        app = App()
        if args.shell:
            from xp.runtime.shellqueue import ShellInbox

            def stale():
                # Give the lock back so the next click is served by a fresh
                # process (this one cannot run code written after it started),
                # and say why the window is not answering.
                try:
                    if hasattr(lock, 'close'):
                        lock.close()
                except Exception:
                    pass
                try:
                    from tkinter import messagebox

                    messagebox.showinfo(
                        'XP was updated',
                        'XP has been changed since this window opened, so this '
                        'window is running an older build.\n\nClose it, then '
                        'use the menu again -- the click will open the new one.',
                        parent=app.root)
                except Exception:
                    pass

            inbox = ShellInbox(app, requests, started_at=started_at,
                               source_root=Path(__file__).resolve().parent,
                               on_stale=stale)
            app.root.after(50, inbox.poll)
        elif args.files:
            # **One window, not two.** The user's report: "I still see two windows when I use click menu to
            # activate the converter, this should not happen, it should only be one". The launcher
            # built the dashboard (which is a window) and then opened the converter beside it, so the
            # click produced an empty dashboard plus the tool. The dashboard is withdrawn here and
            # brought back only if the tool refuses the file, and the launcher ends with the tool.
            app.root.withdraw()
            app.root.after(50, lambda: app.open_sources(args.files, args.action))
        app.run()
    except Exception as exc:
        traceback.print_exc()
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror('XP could not start', f'{exc}\n\nDetails: {state / "launcher.log"}', parent=root)
        root.destroy()
        raise
    finally:
        # `_claim_the_voice` can hand back a plain True when there is nothing
        # to arbitrate, so this closes a handle and shrugs at anything else.
        if hasattr(lock, 'close'):
            lock.close()


if __name__ == '__main__':
    main()
