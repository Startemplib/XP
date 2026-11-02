"""Alt+Q ends XP, from any window and at any stage.

Not "close this window" -- **stop the program**. The user asked for one key that
shuts the whole thing down wherever the user happens to be: the dashboard, the
converter mid-trace, the merge window, the page map, a file chooser. Before
this there was no such key anywhere in XP. The arranger had its own quit
key, but the arranger is a separate process and could only ever close itself.

**Alt+Q, not Ctrl+Q**, since 2026-10-28, at the user's instruction: one key for this
in all three windows. The two cases are both bound for the reason Alt+W/S/A/D
cost a round to learn -- with Alt held, Windows sends the keysym **upper
case** (`Q`, state `0x20002`), so a lone `<Alt-q>` is a key that works in
every test and never once under the user's finger.

Three things make it work from anywhere:

  * **`bind_all`, not `bind`.** A binding on one widget only fires when that
    widget has focus. `bind_all` is per interpreter, so one call covers every
    `Toplevel` sharing a root -- which is every XP window except the arranger.
  * **armed in each window's constructor**, because a window opened straight
    from the CLI (`python -m xp merge a b`) makes its own `Tk()` and has no
    dashboard root to have been armed.
  * **it stops asking.** A quit key that opens "are you sure?" is not a quit
    key; this is the one that ends the program, and `Ctrl+S` is the one that
    saves. The window's own close button keeps whatever asking it had.

What it cannot do, said plainly: if the main thread is inside a long
synchronous call, Tk is not reading the keyboard and nothing bound here can
run. XP's conversions already work on a thread; anything new that blocks the
loop for seconds will make this key feel dead, and that is a property of the
blocking, not of the binding.
"""
from __future__ import annotations

import os
import sys

SEQUENCES = ('<Alt-q>', '<Alt-Q>',
             '<Alt-KeyPress-q>', '<Alt-KeyPress-Q>')


def _root_of(window):
    """The toplevel that owns everything, from any widget."""
    try:
        return window.winfo_toplevel().master or window.winfo_toplevel()
    except Exception:
        return window


def terminate(window=None) -> None:
    """End XP. Tries to leave Tk tidily, then makes sure the process goes.

    The fallbacks are not paranoia: a half-destroyed Tk can raise on the way
    out, and a window that cannot be destroyed must not be able to keep the
    program alive -- that would be exactly the state the user wants the key for.
    """
    try:
        if window is not None:
            root = window.winfo_toplevel()
            # Walk to the application's own root, so this ends XP rather than
            # the one window the key happened to be pressed in.
            while True:
                parent = root.master
                if parent is None:
                    break
                root = parent.winfo_toplevel()
            root.quit()
            root.destroy()
    except Exception:
        pass
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(0)


def arm(window, name: str = '') -> None:
    """Bind the terminate key for every window of this interpreter."""
    def quit_now(_event=None):
        terminate(window)
        return 'break'

    for sequence in SEQUENCES:
        try:
            window.bind_all(sequence, quit_now, add='+')
        except Exception:
            pass


def fill_the_screen(window) -> None:
    """Open maximised, which is what the user means by full screen for a tool.

    `zoomed` and not `-fullscreen`: a true fullscreen window hides the task
    bar and its own title bar, so there is no way back out of it with the
    mouse, and these are windows the user works in beside other things rather than
    presents from. Maximised fills the screen and still has a title bar.

    **The screen's rectangle first, the window-manager state second.** Two
    rounds of this were spent on `wm state zoomed`: on a not-yet-mapped Toplevel
    it reports `zoomed` while the window is still 1x1, and after the first
    `update()` it reports `normal` again -- a request the manager drops, twice
    described as done. Sizing the window to the screen is the part that cannot
    be dropped; the state is asked for as well, because a maximised window still
    has the restore button that a merely large one does not.
    """
    try:
        window.deiconify()
    except Exception:
        pass
    try:
        width = window.winfo_screenwidth()
        height = window.winfo_screenheight()
        window.geometry('%dx%d+0+0' % (width, height))
        window.update_idletasks()
    except Exception:
        pass
    try:
        window.state('zoomed')
    except Exception:
        pass
