"""Give a window the keyboard on Windows, not just Tk's idea of the focus.

The question: "why the focus is automatically in the terminal not that GUI?" `focus_force()` moves
the focus **inside** Tk and `lift()` raises the window, but neither makes it the *foreground window*
on Windows -- the process that launched it keeps the keyboard. So a key goes to the terminal, the
converter never sees it, and every binding looks broken.

`SetForegroundWindow` is the call that actually does it, and Windows refuses it from a process that
is not already in the foreground unless the caller attaches to the foreground thread's input queue.
That is what the `AttachThreadInput` dance below is for; it is the documented workaround, not a
trick of my own.

No Tk import here, so a caller can use it outside a window, and so this module stays testable.
"""
from __future__ import annotations


def take_the_foreground(window) -> bool:
    """Make a Tk window the Windows foreground window. False when it could not be done.

    `window` is a Tk widget (a Toplevel or the root). Its Windows handle is `wm_frame()` -- the
    frame that carries the caption and receives the keyboard -- falling back to `winfo_id()`.
    """
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return False
    try:
        handle = int(window.wm_frame(), 16) if isinstance(window.wm_frame(), str) else int(window.wm_frame())
    except Exception:
        try:
            handle = int(window.winfo_id())
        except Exception:
            return False
    if not handle:
        return False

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    try:
        user32.ShowWindow(wintypes.HWND(handle), 9)          # SW_RESTORE
        foreground = user32.GetForegroundWindow()
        if foreground and foreground != handle:
            # Attach to the foreground thread so the call below is allowed to succeed.
            their_thread = user32.GetWindowThreadProcessId(wintypes.HWND(foreground), None)
            my_thread = kernel32.GetCurrentThreadId()
            attached = bool(user32.AttachThreadInput(wintypes.DWORD(their_thread),
                                                     wintypes.DWORD(my_thread), True))
            try:
                user32.BringWindowToTop(wintypes.HWND(handle))
                user32.SetForegroundWindow(wintypes.HWND(handle))
            finally:
                if attached:
                    user32.AttachThreadInput(wintypes.DWORD(their_thread),
                                             wintypes.DWORD(my_thread), False)
        else:
            user32.SetForegroundWindow(wintypes.HWND(handle))
    except Exception:
        return False
    try:
        return int(user32.GetForegroundWindow()) == handle
    except Exception:
        return False
