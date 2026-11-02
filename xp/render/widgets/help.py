"""The Chinese user guide, as a window opened from XP itself.

The user asked for a complete guide that can be opened **from the XP GUI**. One
markdown file (`xp/docs/HELP.zh.md`) is the source of truth, and this shows it in
a read-only text window -- no help program of its own, no second copy of the
text.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

GUIDE_PATH = Path(__file__).resolve().parents[2] / 'docs' / 'HELP.zh.md'


def guide_text(path: Path = GUIDE_PATH) -> str:
    """The guide as text. A missing file is said, not silently blank."""
    try:
        return path.read_text(encoding='utf-8')
    except OSError as exc:
        return 'The guide could not be read:\n\n%s\n\n%s' % (path, exc)


def open_help(master=None, path: Path = GUIDE_PATH):
    """A window with the guide in it. Returns the window."""
    window = tk.Toplevel(master) if master is not None else tk.Tk()
    window.title('XP - 使用说明')

    frame = ttk.Frame(window)
    frame.pack(fill=tk.BOTH, expand=True)
    text = tk.Text(frame, wrap='word', padx=18, pady=14, relief='flat',
                   font=('Microsoft YaHei UI', 10), spacing1=2, spacing3=4)
    bar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
    text.configure(yscrollcommand=bar.set)
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    bar.pack(side=tk.RIGHT, fill=tk.Y)

    text.insert('1.0', guide_text(path))
    text.configure(state='disabled')          # read-only, but selectable

    footer = ttk.Frame(window)
    footer.pack(fill=tk.X)
    ttk.Button(footer, text='关闭 (Esc)', command=window.destroy).pack(
        side=tk.RIGHT, padx=8, pady=6)
    ttk.Label(footer, text=str(path)).pack(side=tk.LEFT, padx=8)
    window.bind('<Escape>', lambda _event: window.destroy())
    # **Maximised once it has something in it.** Asking before the widgets exist
    # is asking an unmapped window: it answers `zoomed` to nobody and comes up at
    # its default size. The guide is 210 lines of Chinese; 900x760 showed a fifth.
    from xp.runtime.shutdown import fill_the_screen

    try:
        window.transient(master)
    except tk.TclError:
        pass
    # **After `transient`, not before.** A transient window is placed by the
    # window manager relative to its master, and asking for the screen first is
    # a request it drops: the guide came up at cascade size (695x667 at -8,-8)
    # however it was asked.
    window.update_idletasks()
    fill_the_screen(window)
    return window
