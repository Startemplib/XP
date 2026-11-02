"""DPI awareness, in a module that does not import Tk.

**Why it has its own file.** The launcher must be able to say "this verb cannot
work" without starting Tk at all -- `test_arrange_verb` holds that ("tk imported:
False"), because importing Tk is what makes the Explorer path a window. The call
was put in `xp.render.appearance` first, which imports tkinter for the palette,
so every launcher run imported Tk and the test that guards it went red. This
module is ctypes only.
"""
from __future__ import annotations

import sys


def be_dpi_aware() -> bool:
    """Tell Windows this process lays itself out in real pixels.

    **The toolbar rows overlapped on the user's screen and not on mine**, and that is
    what a DPI-unaware Tk process looks like: Tk reads the system DPI and scales
    its fonts by it, while the ttk theme's row metrics stay at 96 dpi, so every
    row's text comes out taller than the row. Under 150 % scaling the merge
    window's rows ran into each other and the left edge sat outside the screen.

    Returns whether the call was made. Windows only; elsewhere it is a no-op, and
    a failure is never fatal -- a blurry UI is better than no UI.
    """
    if sys.platform != 'win32':
        return False
    try:
        import ctypes

        # 2 = per-monitor aware (Windows 8.1+); 1 = system aware (Vista+).
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
        return True
    except Exception:
        return False
