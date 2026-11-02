# -*- coding: utf-8 -*-
"""A button that shows the window's keys while the pointer is on it.

The user's request, 2026-10-28: *"for Arranger, Merger and the Converter add a
button named hotkey, where when my mouse hovers it I can see the hotkey
setup in the GUI."* And then, having seen the first one: *"the panel is too
narrow and long, it wastes a lot of vertical space, some text is cut off,
and there are a lot of duplicated hotkeys. Keep the UI compact."*

So the panel is **a reference table, not a document**: one row per action,
the action on the left and the key on the right, grouped by what the keys
are for.

    HOTKEYS -- Converter

    Navigation
      Previous stage        Alt+W
      Next stage            Alt+S
      Previous parameter    Alt+A
      Next parameter        Alt+D

    Pages
      Previous page         PageUp
      Next page             PageDown

A row is `(action, key, sequence)`. `sequence` is the Tk binding the key
really is, or None when the key is dispatched some other way (a keysym table
inside a `<KeyPress>` handler, say). `tests/test_the_hotkey_button_tells_
the_truth.py` asks Tk which sequences are bound and fails on any row that is
only advertised -- a panel of keys copied by hand is wrong within a week and
then teaches the wrong key, which is worse than no panel at all.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from xp.render.appearance import ABYSS, MUTED, PAPER, TEXT, font

LABEL = 'Hotkeys'
EDGE = 12            # inside the panel's border
ROW_PAD = 1          # between one row and the next: tight on purpose
GROUP_PAD = 7        # above a group's name
COLUMN_GAP = 28      # between the action and its key
SCREEN_EDGE = 8      # the panel never touches the screen's edge
SPLIT_AT = 16        # past this many lines, deal into two columns
COLUMN_PAD = 54      # between the left column's keys and the right's actions


def pretty(sequence: str) -> str:
    """`<Control-Down>` -> `Ctrl+Down`, as a person writes it.

    Kept for the rows that give a Tk sequence and no display text of their
    own, and for the tests.
    """
    text = sequence.strip('<>')
    words = []
    for part in text.split('-'):
        low = part.lower()
        if low == 'control':
            words.append('Ctrl')
        elif low in ('alt', 'shift'):
            words.append(part.capitalize())
        elif low in ('key', 'keypress'):
            continue
        elif low == 'next':
            words.append('PageDown')
        elif low == 'prior':
            words.append('PageUp')
        elif low == 'return':
            words.append('Enter')
        elif len(part) == 1:
            words.append(part.upper())
        else:
            words.append(part)
    return '+'.join(words)


def rows_of(groups):
    """Every `(action, key, sequence)` in `groups`, flat."""
    out = []
    for _name, rows in groups:
        out.extend(rows)
    return out


class HotkeyButton(ttk.Button):
    """`Hotkeys`, with the window's key table under the pointer.

    `groups` is `((group name, ((action, key, sequence), ...)), ...)`.
    """

    def __init__(self, parent, groups, title: str = '', **kw):
        kw.setdefault('text', LABEL)
        kw.setdefault('takefocus', False)
        super().__init__(parent, **kw)
        self.groups = tuple((name, tuple(rows)) for name, rows in groups)
        self.title = title
        self._panel = None
        self.bind('<Enter>', self._show, add='+')
        self.bind('<Leave>', self._hide, add='+')
        self.bind('<Button-1>', self._toggle, add='+')
        self.bind('<Destroy>', self._hide, add='+')

    # -- the panel ---------------------------------------------------------
    def columns(self):
        """The groups split into side-by-side columns.

        The user's complaint about the first panel: *"too narrow and long, it wastes
        a lot of vertical space."* A single column of 26 rows -- the merge
        window -- is most of the height of the user's screen and reads as a scroll.
        So past `SPLIT_AT` rows the groups are dealt into two columns of
        about equal height, which halves it and uses the width the panel has
        anyway.

        Groups are never broken apart: a group's rows stay together in one
        column, because a `Selection` heading in one column and half its keys
        in the other is worse than an uneven pair of columns.
        """
        total = sum(len(rows) + 1 for _name, rows in self.groups)
        if total <= SPLIT_AT or len(self.groups) < 2:
            return [list(self.groups)]
        half, left, running = total / 2.0, [], 0
        for index, (name, rows) in enumerate(self.groups):
            tall = len(rows) + 1
            # Keep at least one group for the right column.
            if running and running + tall / 2.0 > half                     and index < len(self.groups):
                break
            left.append((name, rows))
            running += tall
        right = list(self.groups)[len(left):]
        return [left, right] if right else [left]

    def _fill(self, frame, groups, column):
        """One column of the table: actions left, keys right."""
        action_at, key_at = column * 2, column * 2 + 1
        # A column after the first is pushed clear of the one before it. The
        # space has to sit on these labels rather than on the grid column,
        # because a group's heading spans both cells and eats a column pad.
        lead = COLUMN_PAD if column else 0
        row = 0
        for name, rows in groups:
            if name:
                tk.Label(frame, text=name, background=ABYSS, fg=MUTED,
                         font=font(9)).grid(row=row, column=action_at,
                                            columnspan=2, sticky='w',
                                            padx=(lead, 0),
                                            pady=(GROUP_PAD, ROW_PAD))
                row += 1
            for action, key, _sequence in rows:
                tk.Label(frame, text=action, background=ABYSS, fg=TEXT,
                         font=font(9), anchor='w').grid(
                    row=row, column=action_at, sticky='w',
                    padx=(lead + EDGE, COLUMN_GAP), pady=ROW_PAD)
                tk.Label(frame, text=key, background=ABYSS, fg=PAPER,
                         font=font(9), anchor='e').grid(
                    row=row, column=key_at, sticky='e', pady=ROW_PAD)
                row += 1
        return row

    def _build(self, frame):
        columns = self.columns()
        width = len(columns) * 2
        head = 0
        if self.title:
            tk.Label(frame, text='%s  --  %s' % (LABEL.upper(), self.title),
                     background=ABYSS, fg=PAPER, font=font(9)).grid(
                row=0, column=0, columnspan=width, sticky='w',
                pady=(0, GROUP_PAD))
            head = 1
        body = tk.Frame(frame, background=ABYSS)
        body.grid(row=head, column=0, columnspan=width, sticky='nw')
        for index, groups in enumerate(columns):
            self._fill(body, groups, index)
            # Each key column keeps to the right of its own actions.
            body.grid_columnconfigure(index * 2, weight=1)
            body.grid_columnconfigure(index * 2 + 1, weight=0)

    def _show(self, _event=None):
        if self._panel is not None or not self.groups:
            return
        try:
            panel = tk.Toplevel(self)
            panel.withdraw()
            panel.overrideredirect(True)
            panel.configure(background=MUTED)
            frame = tk.Frame(panel, background=ABYSS, padx=EDGE, pady=EDGE)
            frame.pack(padx=1, pady=1)
            self._build(frame)
            panel.update_idletasks()
            panel.geometry('+%d+%d' % self._where(panel))
            panel.deiconify()
            panel.lift()
            self._panel = panel
        except Exception:
            # A window is more important than its key list.
            self._panel = None

    def _where(self, panel):
        """Top right of the button, and never off the screen.

        The button lives in the top right corner of its window, so the panel
        hangs below it and is pulled left until it fits -- a table that runs
        off the edge of the screen is a table with its key column missing.
        """
        want_w = panel.winfo_reqwidth()
        want_h = panel.winfo_reqheight()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = self.winfo_rootx() + self.winfo_width() - want_w
        y = self.winfo_rooty() + self.winfo_height() + 4
        if y + want_h > screen_h - SCREEN_EDGE:
            above = self.winfo_rooty() - want_h - 4
            y = above if above >= SCREEN_EDGE else max(
                SCREEN_EDGE, screen_h - SCREEN_EDGE - want_h)
        x = max(SCREEN_EDGE, min(x, screen_w - SCREEN_EDGE - want_w))
        return int(x), int(y)

    def _hide(self, _event=None):
        panel, self._panel = self._panel, None
        if panel is not None:
            try:
                panel.destroy()
            except Exception:
                pass

    def _toggle(self, _event=None):
        if self._panel is None:
            self._show()
        else:
            self._hide()

    # -- for the tests -----------------------------------------------------
    @property
    def showing(self) -> bool:
        return self._panel is not None

    @property
    def panel(self):
        return self._panel

    def rows(self):
        """Every `(action, key, sequence)` the panel shows."""
        return rows_of(self.groups)
