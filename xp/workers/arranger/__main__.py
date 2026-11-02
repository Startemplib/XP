"""`python -m xp.workers.arranger [documents...]` -- the page arranger on its own.

This is the standalone door: it builds its own Tk root, so it needs neither
XP's dashboard nor XP's CLI. It is also how XP starts it (`bridge.py`), as a
detached process with its own crash domain. Laying it into the merge window
would be a later step, and it would call `ArrangerApp(master=..., on_finish=...)`
directly instead of going through here.
"""

import sys

from .arrangergui import ArrangerApp


def _hide_console() -> None:
    """No black console window behind the arranger on Windows."""
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        handle = ctypes.windll.kernel32.GetConsoleWindow()
        if handle:
            ctypes.windll.user32.ShowWindow(handle, 0)
    except Exception:
        pass


def main(argv=None) -> int:
    _hide_console()
    files = list(sys.argv[1:] if argv is None else argv)
    app = ArrangerApp()
    if files:
        app.root.after(60, lambda: app.add_paths(files))
    app.run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
