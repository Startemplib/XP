"""The window's own look: one palette, one face, ttk styles.

These are the colours the arranger was drawn with while it lived inside XP's
tree; they are constants here rather than an import, because this package
must open without XP installed. The face is the same one for the same reason
-- a window that looks like XP's other windows is easier to move into them
later.
"""

from __future__ import annotations

from tkinter import ttk

ABYSS = '#101923'      # the ground
INK = '#8fc9ff'        # Xournal ink, the heavy stroke
PALE = '#eef5fb'       # the hairline
PAPER = '#4cddd5'      # PDF cyan, used for "this is the thing"
TEXT = '#d8e5ef'
MUTED = '#96adbf'

FACE = 'Times New Roman'
MONO = 'Consolas'
BASE = 10


def font(size: int = BASE, weight: str = 'normal', mono: bool = False):
    return (MONO if mono else FACE, size, weight)


def use_the_face(root) -> None:
    """Set the named fonts Tk hands to widgets, so one face wins everywhere."""
    import tkinter.font as tkfont

    for name, size in (('TkDefaultFont', BASE), ('TkTextFont', BASE),
                       ('TkMenuFont', BASE), ('TkHeadingFont', BASE),
                       ('TkCaptionFont', BASE), ('TkSmallCaptionFont', BASE - 1),
                       ('TkIconFont', BASE), ('TkTooltipFont', BASE - 1)):
        try:
            held = tkfont.nametofont(name, root=root)
        except Exception:
            continue
        held.configure(family=FACE, size=size)


def decorate(root) -> None:
    root.configure(background=ABYSS)
    use_the_face(root)
    style = ttk.Style(root)
    for name in ('Lab.TFrame', 'Lab.TLabelframe'):
        style.configure(name, background=ABYSS)
    style.configure('Lab.TLabel', background=ABYSS, foreground=TEXT, font=font())
    style.configure('Muted.Lab.TLabel', background=ABYSS, foreground=MUTED,
                    font=font())
    style.configure('Title.Lab.TLabel', background=ABYSS, foreground=PAPER,
                    font=font(18, 'bold'))
    style.configure('Lab.Horizontal.TProgressbar', troughcolor=ABYSS,
                    background=PAPER, bordercolor=ABYSS, lightcolor=PAPER,
                    darkcolor=PAPER, thickness=14)
    for name in ('TButton', 'TEntry', 'TCombobox', 'TSpinbox', 'TNotebook.Tab',
                 'TLabelframe.Label', 'TMenubutton'):
        try:
            style.configure(name, font=font())
        except Exception:
            pass
