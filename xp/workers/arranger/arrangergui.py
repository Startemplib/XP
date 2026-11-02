"""The Arranger window: pages as a grid you reorganise.

    "a page manager like PDF Arranger, but inside XP, and it has to keep the
     brush merge."

The window is a view over `arrangement.Arrangement` -- it owns no page
content of its own. Every operation it offers (select, drag, copy, paste,
duplicate, delete, undo, redo) is one call on that object, which is what
keeps the page model testable without a window and keeps this file to
drawing, hit-testing and keyboard.

Three behaviours are deliberate, and each one is a cost paid somewhere else
if it is done the obvious way.

**Thumbnails are rendered off the UI thread, one at a time, for the cards on
screen only.** Drawing a page means parsing its strokes and rasterizing
them; on a page of the user's that is 1.7 million strokes, and doing it for every
page when the window opens is a window that opens after minutes. So a card
is drawn empty, the visible ones are queued, and a worker thread renders
them and hands back PIL images (PhotoImage is built on the main thread --
Tk is not thread-safe). Scrolling enqueues what it reveals and nothing
else.

**A page with more strokes than `DENSE_LIMIT` is not drawn at all.** A
thumbnail of a page of 1.7 million strokes is a grey rectangle anyway, and
rasterizing it costs what rasterizing the page costs. The card says how
many strokes it has instead of pretending to show them. That is a stated
limit, not a hidden one, and it is the first thing the GPU pass in the
project's section 7 can remove.

**A page's identity is its uid, never its slot.** Selection, drag, undo and
"delete these four" all travel by uid; a slot changes under every drag, and
a window that remembered slots would move the wrong pages the moment the
order moved. The cards are drawn in slot order; nothing else reads it.

Licences, for the record: the interaction ideas come from pdfarranger
(GPL-3.0-or-later) and NeXopp (GPL-2.0-or-later), read but not copied -- this file is written against the .xopp format, and no
line of theirs is here.
"""

from __future__ import annotations

import queue
import re
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk
from typing import Callable, Dict, List, Optional, Tuple

from PIL import ImageTk

from . import look
from .arrangement import Arrangement, PageRef
from .xopp import Notebook

# The window is painted from the one palette every XP window uses -- by value,
# not by import: this package opens with XP absent.
GROUND = look.ABYSS
PAPER = "#f7f4ec"
CARD_EDGE = "#31465b"
SELECT_EDGE = look.PAPER
TEXT = look.TEXT
MUTED = look.MUTED
ACCENT = look.INK
INSERT = look.PAPER

# The selection's neon: a dark outer halo, a blue glow, and a pale core. Three
# rectangles rather than one because Tk canvases have no blur: the halo is
# what reads as "lit" from a distance, the core is what is crisp up close.
GLOW_HALO = "#0b4bd1"
GLOW_BLUE = "#39b6ff"
GLOW_CORE = "#d8f7ff"
HOVER_EDGE = "#5d8bb0"
HOVER_FILL = "#fbf8f1"
FOCUS_BAR = "#39d7ff"      # the one bar on the focused page's left edge

MOUSE = 'MOUSE'
KEYBOARD = 'KEYBOARD'


def _expand(paths):
    """Files, folders and both -> the `.xopp` files to open, and what was not.

    A folder contributes its `.xopp` files in name order; a name that is not
    a notebook is reported rather than silently ignored.
    """
    wanted, skipped = [], []
    for raw in paths:
        if not str(raw):
            continue
        path = Path(raw)
        if path.is_dir():
            found = sorted((p for p in path.iterdir()
                            if p.is_file() and p.suffix.lower() == '.xopp'),
                           key=lambda p: p.name.lower())
            if found:
                wanted.extend(found)
            else:
                skipped.append(f'{path.name} (no .xopp inside)')
        elif path.suffix.lower() == '.xopp':
            wanted.append(path)
        else:
            skipped.append(path.name)
    return wanted, skipped

# Four thumbnail widths, halving the work per step rather than a continuous
# slider: a slider redraws the grid on every motion event of the drag.
THUMB_WIDTHS = (96, 136, 188, 248)
DEFAULT_THUMB = 1

PAD = 16
GAP = 14
CARD_PAD = 6
LABEL_H = 20
ASPECT_CAP = 1.9          # cell height = thumb width * this

# Above this many strokes a page is not rasterized for its card. See the
# module docstring; the number is where a 140k-stroke page still renders in
# about a second and a 1.7M one would not finish a screenful.
DENSE_LIMIT = 120_000

CACHE_MAX = 240           # thumbnails kept; a PhotoImage is ~110 KB at 136px
POLL_MS = 40

CTRL = 0x0004
SHIFT = 0x0001
ALT = 0x0008


# **The key table the Hotkeys button shows.** The user's design, 2026-10-28: one
# row per action, the key on the right, and the arrow keys merged with the
# letters that do the same thing -- "they are the same navigation, do not
# list them twice". Each row carries the Tk sequence it really is, and the
# test asks Tk whether that sequence is bound.
#
# Not here, deliberately: Ctrl+arrow. It is bound to a handler that does
# nothing on purpose ("Ctrl+arrow is not a second movement scheme"), and a
# key that does nothing has no business in a key table.
ARRANGE_HOTKEYS = (
    ('Navigation  --  moves the focus', (
        ('Up', '↑ / W', '<Up>'),
        ('Down', '↓ / S', '<Down>'),
        ('Left', '← / A', '<Left>'),
        ('Right', '→ / D', '<Right>'),
    )),
    ('Moving pages  --  moves the pages themselves', (
        ('Move up', 'Alt+↑ / Alt+W', '<Alt-Up>'),
        ('Move down', 'Alt+↓ / Alt+S', '<Alt-Down>'),
        ('Move left', 'Alt+← / Alt+A', '<Alt-Left>'),
        ('Move right', 'Alt+→ / Alt+D', '<Alt-Right>'),
    )),
    ('Selection', (
        ('Extend the selection', 'Shift+↑ ↓ ← →', '<Shift-Up>'),
        ('Drop the selection', 'Escape', '<Escape>'),
        ('Delete the selected pages', 'Delete', '<Delete>'),
    )),
    ('Looking at a page', (
        ('View the focused page', 'Space', '<space>'),
        ('View the selected page', 'Enter', '<Return>'),
    )),
    # The user's note, 2026-10-28: *"by the way, don't forget the Ctrl+D for
    # duplicate for the arranger"*, and then, asked about the rest of the
    # clipboard: *"just list this addon, the others are trivial, no need to
    # mention them."* So Ctrl+D alone. It carries no Tk sequence because
    # `_on_any_key` dispatches it from a dict keyed by keysym, which is how
    # it stayed out of a table built by reading `bind` calls.
    ('Editing', (
        ('Duplicate the selected pages', 'Ctrl+D', None),
    )),
    ('The view', (
        ('Bigger', 'Ctrl++', '<Control-plus>'),
        ('Smaller', 'Ctrl+-', '<Control-minus>'),
    )),
    ('The window', (
        ('Quit XP', 'Alt+Q', '<Alt-q>'),
    )),
)
# The rows above with no sequence of their own, named in one place so the
# panel test can tell "dispatched by keysym" from "advertised and not bound".
ARRANGE_PLAIN_KEYS = ('Ctrl+D',)


class ArrangerApp:
    """The page grid. `arrangement` may be passed in, or built empty here."""

    def __init__(self, arrangement: Optional[Arrangement] = None,
                 master: Optional[tk.Misc] = None,
                 on_finish: Optional[Callable[[bool], None]] = None,
                 on_open_page: Optional[Callable[[PageRef], None]] = None):
        self.arrangement = arrangement if arrangement is not None else Arrangement()
        self.master = master
        self.on_finish = on_finish
        self.on_open_page = on_open_page
        self._owns_root = master is None
        self._finished = False
        self._closing = False

        self.thumb_index = DEFAULT_THUMB
        self._cache: "Dict[Tuple[int, int, int], Tuple[str, object]]" = {}
        self._displayed: set = set()
        self._pending: set = set()
        self._cards: Dict[int, dict] = {}
        # The interaction state, kept apart on purpose:
        #   input_control  who moves the focus bar      (MOUSE | KEYBOARD)
        #   focused        the one page the focus bar is on (uid, or None)
        #   arrangement.selection   every selected page  (uids)
        #   _shift_anchor/_shift_base  the live Shift range (mouse)
        # ONE move rule: no selection moves the focus, a selection moves the
        # whole of it (see nudge). A modifier never changes that.
        self.input_control = MOUSE
        self.focused: Optional[int] = None
        self._keyboard_focus: Optional[int] = None     # remembered across Ctrl+L
        self._ctrl_target: Optional[int] = None        # what a lone Ctrl would toggle
        self._pdf_cache: Dict[Tuple[str, str], Optional[Path]] = {}
        self._save_path: Optional[Path] = None
        self._resize_job = None
        self._last_size = None
        self._in_redraw = False
        self._scrollregion = None
        self.exit_after_save = True
        self._log: List[str] = []
        self._shift_anchor: Optional[int] = None       # Shift range session
        self._shift_base: set = set()                  # selections before the range
        self._ctrl_down = False
        self._ctrl_used = False
        self._ctrl_at = 0.0
        self._hover: Optional[int] = None
        self._preview = None                           # a PagePreview while open
        self._press = None                 # (uid, x, y) of the button press
        self._drag_block: set = set()      # the pages a drag will move
        self._dragging = False
        self._dragged_at = 0.0             # last drag end, for double-click quiet
        self._insert_at: Optional[int] = None
        self._insert_line = None
        self._message = ""
        self._message_at = 0.0
        self._after_job = None

        self._open_queue: "queue.Queue" = queue.Queue()
        self._opened: "queue.Queue" = queue.Queue()
        self._render_queue: "queue.Queue" = queue.Queue()
        self._rendered: "queue.Queue" = queue.Queue()
        self._opening = 0
        self._open_total = 0
        self._open_done = 0
        self._progress_mode = None
        self._save_queue = None
        self._save_running = False

        self.root = tk.Tk() if master is None else tk.Toplevel(master)
        self.root.title("XP (arranger)")
        self.root.geometry("1080x720")
        # Opened maximised, like the converter and the merge window: the user
        # works across a whole page of cards, and 1080x720 made the user
        # resize it every time. `fill_the_screen` prefers the window
        # manager's own zoomed state over true fullscreen, so the title
        # bar and the task bar stay where they are.
        from xp.runtime.shutdown import arm, fill_the_screen
        fill_the_screen(self.root)
        # The arranger runs as its own process, so it was the one window
        # Alt+Q could not reach. It arms its own.
        arm(self.root)
        look.decorate(self.root)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self._caption_font = tkfont.Font(family=look.FACE, size=9)

        self._build_toolbar()
        self._build_canvas()
        self._build_overlay()
        self._bind_keys()

        self._open_thread = threading.Thread(target=self._open_loop, daemon=True)
        self._open_thread.start()
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._render_thread.start()

        self.root.bind('<Configure>', self._on_resize, add='+')
        self._after_job = self.root.after(POLL_MS, self._poll)
        self._redraw()

        # A toolbar that does not fit is a button that silently is not there
        # (the merge window's own history), so the window keeps at least the
        # width its buttons ask for -- capped by the screen, because a
        # minimum wider than the display is a window with no visible edge.
        self.root.update_idletasks()
        wanted = self.root.winfo_reqwidth()
        self.root.minsize(min(wanted, max(600, self.root.winfo_screenwidth() - 80)), 420)

    # ---------------- construction ----------------

    def _build_toolbar(self) -> None:
        bar = tk.Frame(self.root, bg=look.ABYSS, highlightthickness=0)
        bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 2))
        # Save is anchored right so a narrow window squeezes the editing
        # buttons rather than the one that writes the file.
        ttk.Button(bar, text="Save as...", command=self.save_as).pack(
            side=tk.RIGHT, padx=2)
        for label, action in (("Add files...", self.ask_add_files),
                              ("Delete", self.delete_selected),
                              ("Duplicate", self.duplicate_selected),
                              ("Copy", self.copy_selected),
                              ("Cut", self.cut_selected),
                              ("Paste", self.paste_clipboard),
                              ("Undo", self.undo),
                              ("Redo", self.redo)):
            ttk.Button(bar, text=label, command=action).pack(side=tk.LEFT, padx=2)
        # The key map, under the pointer. It takes no focus, so resting on it
        # cannot take the keyboard away from the page grid.
        from xp.render.widgets.hotkeys import HotkeyButton

        self.hotkey_button = HotkeyButton(bar, ARRANGE_HOTKEYS,
                                          title='Arranger')
        # Top right: the user looked for it there and it was at the end of the
        # left-hand run of buttons, which is why the user did not find it.
        self.hotkey_button.pack(side=tk.RIGHT, padx=(10, 4))
        # One bar for state and messages: the mode/focus/counts on the left,
        # the last messages on the right (errors red). A plain tk.Frame,
        # because the Windows ttk theme ignores a Frame's background and the
        # gap between the two labels would show as a white square.
        bar = tk.Frame(self.root, bg=look.ABYSS, highlightthickness=0)
        bar.pack(side=tk.TOP, fill=tk.X, padx=10)
        self.status = ttk.Label(bar, text="", style='Muted.Lab.TLabel',
                                anchor='w')
        self.status.pack(side=tk.LEFT)
        self.log = ttk.Label(bar, text="    log: ready",
                             style='Muted.Lab.TLabel', anchor='w')
        self.log.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _build_canvas(self) -> None:
        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.canvas = tk.Canvas(frame, bg=GROUND, highlightthickness=0,
                                takefocus=True)
        vbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Double-Button-1>", self._on_double)
        self.canvas.bind("<Motion>", self._on_hover)
        self.canvas.bind("<Leave>", lambda e: self._set_hover(None))
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        self.canvas.bind("<Configure>", lambda e: self._on_resize(e))
        self._hint = ttk.Label(self.root, text=self._hint_text(),
                               style='Muted.Lab.TLabel',
                               wraplength=1000, justify='left')
        self._hint.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 6))

    def _pump_open_progress(self) -> None:
        if self._progress_mode == 'open':
            if self._open_total and self._open_done >= self._open_total \
                    and self._opening == 0:
                if self._pending:
                    self._progress_mode = 'thumbs'
                    self._overlay_show("Rendering thumbnails...", None,
                                       f"{len(self._pending)} left")
                else:
                    self._progress_mode = None
                    self._overlay_hide()
        elif self._progress_mode == 'thumbs':
            if not self._pending:
                self._progress_mode = None
                self._overlay_hide()
            else:
                self._overlay_pulse(f"{len(self._pending)} thumbnail(s) left")

    def _pump_save_progress(self) -> None:
        if not self._save_running or self._save_queue is None:
            return
        while True:
            try:
                kind, *payload = self._save_queue.get_nowait()
            except queue.Empty:
                return
            if kind == 'n':
                done, total = payload
                self._overlay_set(done, total, f"page {done} of {total}")
            elif kind == 'ok':
                self._save_running = False
                self._progress_mode = None
                self._overlay_hide()
                self.close()
                return
            elif kind == 'err':
                self._save_running = False
                self._progress_mode = None
                self._overlay_error("Could not save", str(payload[0]))
                return

    def _build_overlay(self) -> None:
        """The centred progress panel, a child of the canvas, hidden at rest."""
        panel = tk.Frame(self.canvas, bg=look.ABYSS,
                         highlightbackground=look.PAPER,
                         highlightthickness=1)
        self._overlay = panel
        self._overlay_title = tk.Label(
            panel, text="", bg=look.ABYSS, fg=look.TEXT,
            font=look.font(12, 'bold'), anchor='w', justify='left')
        self._overlay_title.pack(anchor='w', padx=18, pady=(14, 4))
        self._overlay_bar = ttk.Progressbar(
            panel, style='Lab.Horizontal.TProgressbar', length=430,
            mode='determinate', maximum=100, value=0)
        self._overlay_bar.pack(fill=tk.X, padx=18, pady=4)
        self._overlay_detail = tk.Label(
            panel, text="", bg=look.ABYSS, fg=look.MUTED,
            font=look.font(9), anchor='w', justify='left', wraplength=430)
        self._overlay_detail.pack(anchor='w', padx=18, pady=(2, 12))
        self._overlay_close = tk.Button(
            panel, text="Close", command=self._overlay_hide,
            bg=look.ABYSS, fg=look.TEXT, relief='flat',
            activebackground=look.PAPER, activeforeground=look.ABYSS,
            font=look.font(9), borderwidth=0)
        panel.place_forget()

    def _overlay_show(self, title: str, maximum=None,
                      detail: str = "starting...") -> None:
        self._overlay_close.pack_forget()
        self._overlay_title.configure(text=title, fg=look.TEXT)
        self._overlay_detail.configure(text=detail, fg=look.MUTED)
        bar = self._overlay_bar
        bar.stop()
        if not bar.winfo_ismapped():
            bar.pack(fill=tk.X, padx=18, pady=4,
                     before=self._overlay_detail)
        if maximum:
            bar.configure(mode='determinate', maximum=max(1, maximum), value=0)
        else:
            bar.configure(mode='indeterminate', value=0)
            bar.start(12)
        self._overlay.place(relx=0.5, rely=0.5, anchor='center')
        self._overlay.lift()

    def _overlay_set(self, done: int, total: int, detail: str = "") -> None:
        total = max(1, total)
        done = max(0, min(done, total))
        percent = int(round(done * 100.0 / total))
        self._overlay_bar.configure(mode='determinate', maximum=total,
                                    value=done)
        line = f"{done} / {total}  ·  {percent}%"
        self._overlay_detail.configure(
            text=f"{detail}\n{line}" if detail else line)

    def _overlay_pulse(self, detail: str) -> None:
        self._overlay_detail.configure(text=detail)

    def _overlay_error(self, title: str, message: str) -> None:
        self._overlay_bar.stop()
        self._overlay_bar.pack_forget()
        self._overlay_title.configure(text=title, fg='#ff7b72')
        self._overlay_detail.configure(text=message, fg='#ff7b72')
        self._overlay_close.pack(anchor='e', padx=18, pady=(0, 12))
        self._overlay.place(relx=0.5, rely=0.5, anchor='center')
        self._overlay.lift()

    def _overlay_hide(self) -> None:
        try:
            self._overlay_bar.stop()
        except tk.TclError:
            pass
        self._overlay_bar.pack_forget()
        self._overlay_close.pack_forget()
        self._overlay.place_forget()

    def _bind_keys(self) -> None:
        on = self.root
        # Movement lives on Alt so the old Ctrl commands stay where they were:
        #   plain direction   move the focus
        #   Alt+direction     move the whole selection
        #   Shift+direction   grow the range
        # Ctrl+A/S/D/S are Select All, Duplicate and Save.
        # Letter commands and WASD are dispatched in `_on_any_key`, not by
        # per-key bindings: CapsLock changes `s` to `S`, and matching keysym
        # case in Tk bindings is how "Ctrl+S does nothing" happens.
        on.bind('<Control-equal>', self._shortcut(self.bigger))
        on.bind('<Control-plus>', self._shortcut(self.bigger))
        on.bind('<Control-minus>', self._shortcut(self.smaller))
        # No Ctrl+0: the user asked for it gone, 2026-10-28, with no key in its
        # place. Ctrl++ and Ctrl+- stay.
        on.bind('<KeyPress-Control_L>', self._on_ctrl_press)
        on.bind('<KeyRelease-Control_L>', self._on_ctrl_release)
        on.bind('<KeyPress-Control_R>', self._on_ctrl_press)
        on.bind('<KeyRelease-Control_R>', self._on_ctrl_release)
        on.bind('<KeyPress>', self._on_any_key, add='+')
        on.bind('<Delete>', lambda e: self.delete_selected())
        on.bind('<BackSpace>', lambda e: self.delete_selected())
        on.bind('<Escape>', lambda e: self._on_escape())
        on.bind('<space>', self._on_space)
        # Arrows: plain moves the focus, Alt moves the selection, Shift grows
        # the range. Ctrl+arrow is not a second movement scheme.
        for key, dx, dy in (('<Up>', 0, -1), ('<Down>', 0, 1),
                            ('<Left>', -1, 0), ('<Right>', 1, 0)):
            on.bind(key, lambda e, dx=dx, dy=dy: self.focus_move(dx, dy))
            on.bind(f'<Alt-{key[1:-1]}>',
                    lambda e, dx=dx, dy=dy: self.move_selection(dx, dy))
            on.bind(f'<Shift-{key[1:-1]}>',
                    lambda e, dx=dx, dy=dy: self.extend_selection(dx, dy))
            on.bind(f'<Control-{key[1:-1]}>', lambda e: None)
        # WASD is dispatched in `_on_any_key`, but Alt needs explicit
        # bindings: Tk does not deliver a modifier-free `<KeyPress-w>` for
        # Alt+W. Both cases, because CapsLock changes the keysym.
        for key, dx, dy in (('w', 0, -1), ('a', -1, 0),
                            ('s', 0, 1), ('d', 1, 0)):
            for case in (key, key.upper()):
                on.bind(f'<Alt-KeyPress-{case}>',
                        lambda e, dx=dx, dy=dy: self.move_selection(dx, dy))
        on.bind('<Return>', lambda e: self.preview_selected())
        on.bind('<KP_Enter>', lambda e: self.preview_selected())

    # ---------------- sources ----------------

    def ask_add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Add documents", filetypes=[("Xournal++ notebook", "*.xopp")],
            parent=self.root)
        if paths:
            self.add_paths(paths)

    def add_paths(self, paths) -> None:
        """Open each file in the background, then add its pages to the order.

        `paths` may name files, folders (every `.xopp` directly inside is
        taken, in name order), or both -- a folder is what arrives when one
        is dropped on `Arranger.cmd`, and expanding it here means there is
        one way in rather than a second launcher.

        Opening is the expensive half -- a 495 MB notebook unpacks to a
        scratch file in about nine seconds -- and it happens on the worker,
        because doing it here would freeze the window before its first card.
        """
        if paths:
            self._save_path = None
        wanted, skipped = _expand(paths)
        if skipped:
            self._say("skipped: " + ", ".join(skipped))
        if not wanted:
            if not skipped:
                self._say("nothing to open in that selection")
            return
        if wanted:
            if self._progress_mode != 'open':
                self._open_total = 0
                self._open_done = 0
            self._open_total += len(wanted)
            self._progress_mode = 'open'
            self._overlay_show(
                f"Opening {self._open_total} document(s)...",
                self._open_total,
                f"0 of {self._open_total} document(s)")
        self._opening += len(wanted)
        self._say(f"opening {len(wanted)} notebook(s)...")
        for path in wanted:
            self._open_queue.put(path)
        self._update_status()

    def _open_loop(self) -> None:
        while True:
            path = self._open_queue.get()
            if path is None:
                return
            try:
                side = Notebook(path)
                self._opened.put((path, side, None))
            except Exception as exc:              # a bad file is not a crash
                self._opened.put((path, None, exc))

    def _drain_opened(self) -> bool:
        added = False
        while True:
            try:
                path, side, error = self._opened.get_nowait()
            except queue.Empty:
                return added
            self._opening = max(0, self._opening - 1)
            self._open_done += 1
            if self._progress_mode == 'open' and self._open_total:
                self._overlay_set(
                    self._open_done, self._open_total,
                    f"{Path(path).name}")
            if error is not None:
                self._say(f"could not open {Path(path).name}: {error}")
                continue
            source = self.arrangement.add_source(side, path, Path(path).stem)
            self._say(f"added {source.label}: {source.pages} page(s)")
            added = True
        return added

    # ---------------- thumbnails ----------------

    def _cell(self) -> Tuple[int, int]:
        width = THUMB_WIDTHS[self.thumb_index]
        height = int(width * ASPECT_CAP)
        return (width + 2 * CARD_PAD + 8, height + LABEL_H + 2 * CARD_PAD)

    def _layout(self) -> Tuple[int, int, int]:
        cell_w, cell_h = self._cell()
        view = max(1, self.canvas.winfo_width())
        columns = max(1, (view - 2 * PAD + GAP) // (cell_w + GAP))
        return int(columns), cell_w, cell_h

    def card_rect(self, position: int) -> Tuple[int, int, int, int]:
        """The card's rectangle in canvas coordinates (for hit tests/tests)."""
        columns, cell_w, cell_h = self._layout()
        row, col = divmod(position, columns)
        x = PAD + col * (cell_w + GAP)
        y = PAD + row * (cell_h + GAP)
        return x, y, x + cell_w, y + cell_h

    def _uid_at(self, x: float, y: float) -> Optional[int]:
        for uid, box in self._cards.items():
            left, top, right, bottom = box['rect']
            if left <= x <= right and top <= y <= bottom:
                return uid
        return None

    def _visible_positions(self) -> List[int]:
        if not len(self.arrangement):
            return []
        top = self.canvas.canvasy(0)
        bottom = top + max(1, self.canvas.winfo_height())
        columns, cell_w, cell_h = self._layout()
        row_h = cell_h + GAP
        first = max(0, int((top - PAD) // row_h) - 1)
        last = int((bottom - PAD) // row_h) + 2
        lo = first * columns
        hi = min(len(self.arrangement), (last + 1) * columns)
        return list(range(max(0, lo), max(0, hi)))

    def _enqueue_visible(self) -> None:
        width = THUMB_WIDTHS[self.thumb_index]
        queue_size = self._render_queue.qsize()
        if queue_size > 48:
            return                        # scrolling faster than rendering
        for position in self._visible_positions():
            ref = self.arrangement[position]
            key = (ref.source, ref.index, width)
            if key in self._cache or key in self._pending:
                continue
            self._pending.add(key)
            self._render_queue.put((key, self.arrangement.source(ref.source),
                                    ref.index, width))

    def _render_loop(self) -> None:
        while True:
            task = self._render_queue.get()
            if task is None:
                return
            key, source, index, width = task
            try:
                made = self._render_one(source, index, width)
            except Exception as exc:          # one bad page is not the window
                made = ('error', repr(exc))
            self._rendered.put((key, made))

    @staticmethod
    def _stroke_count(source, index: int) -> int:
        """How many strokes a page has, without parsing it.

        `Notebook.stroke_count` counts `<stroke` in the page's own bytes,
        which is a scan; parsing the page to count them would build the very
        tree this is deciding not to draw. A page of the user's is up to 223 MB of
        XML for 1.7 million strokes, so the difference is the whole feature.
        """
        return source.side.stroke_count(index)

    @staticmethod
    def _render_one(source, index: int, width: int):
        """One page as a small PIL image, or a count when it is too big.

        Runs off the main thread: no Tk, no PhotoImage, only the page's own
        `<stroke>` elements drawn at the card's size.
        """
        page_w, page_h = source.page_box(index)
        height = int(width * ASPECT_CAP)
        scale = min(width / max(page_w, 1e-6), height / max(page_h, 1e-6))
        if source.kind == 'xopp':
            count = ArrangerApp._stroke_count(source, index)
            if count > DENSE_LIMIT:
                return ('dense', count)
        return ('image', source.render(index, scale))

    def _drain_rendered(self) -> bool:
        changed = False
        while True:
            try:
                key, made = self._rendered.get_nowait()
            except queue.Empty:
                return changed
            self._pending.discard(key)
            kind = made[0]
            if kind == 'error':
                self._cache[key] = ('error', None)
                continue
            if kind == 'dense':
                self._cache[key] = ('dense', made[1])
                self._show_dense(key)
                changed = True
                continue
            try:
                photo = ImageTk.PhotoImage(made[1], master=self.root)
            except Exception:
                continue
            self._cache[key] = ('image', photo)
            self._show_photo(key, photo)
            self._trim_cache()
            changed = True

    def _trim_cache(self) -> None:
        """Drop the oldest thumbnails once the cache is over its bound.

        A card that is on screen is never dropped: its canvas item holds an
        image name, and deleting the PhotoImage that name belongs to leaves
        a blank card rather than a page.
        """
        if len(self._cache) <= CACHE_MAX:
            return
        for key in list(self._cache):
            if len(self._cache) <= CACHE_MAX:
                return
            if key in self._displayed:
                continue
            del self._cache[key]

    # ---------------- drawing ----------------

    def _key_for(self, ref: PageRef):
        return (ref.source, ref.index, THUMB_WIDTHS[self.thumb_index])

    def _redraw(self) -> None:
        if self._in_redraw:
            return
        self._in_redraw = True
        try:
            self._do_redraw()
        finally:
            self._in_redraw = False

    def _do_redraw(self) -> None:
        self.canvas.delete('all')
        self._cards.clear()
        self._displayed.clear()
        columns, cell_w, cell_h = self._layout()
        total = len(self.arrangement)
        rows = max(1, -(-total // columns))
        region = (0, 0, self.canvas.winfo_width(),
                  PAD * 2 + rows * (cell_h + GAP))
        if region != self._scrollregion:
            self._scrollregion = region
            self.canvas.configure(scrollregion=region)

        if not total:
            self._draw_empty()
            self._update_status()
            return

        for position in range(total):
            ref = self.arrangement[position]
            x0, y0, x1, y1 = self.card_rect(position)
            # Painted back to front: a dark halo outside the card, a blue glow
            # on its rim, the paper, then the pale core line. `_style_card`
            # turns the three states on and off; drawing them all and hiding
            # the two that do not apply keeps redraw a matter of colour.
            halo = self.canvas.create_rectangle(
                x0 - 5, y0 - 5, x1 + 5, y1 + 5, outline=GLOW_HALO, width=10,
                state='hidden')
            glow = self.canvas.create_rectangle(
                x0 - 1, y0 - 1, x1 + 1, y1 + 1, outline=GLOW_BLUE, width=5,
                state='hidden')
            box = self.canvas.create_rectangle(x0, y0, x1, y1, fill=PAPER,
                                               outline='')
            edge = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline=CARD_EDGE, width=1)
            label = self.canvas.create_text(
                x0 + cell_w / 2, y1 - LABEL_H / 2 + 2,
                text=self._caption(ref, cell_w - 16), fill='#22303c',
                font=self._caption_font)
            # The focus bar: on the focused page's left edge, and on nothing
            # else. It is not the selection and does not follow it -- that
            # separation is the whole point of Ctrl+L.
            bar = self.canvas.create_rectangle(
                x0 - 6, y0, x0 - 1, y1, fill=FOCUS_BAR, outline='',
                state='hidden')
            self._cards[ref.uid] = {'rect': (x0, y0, x1, y1), 'halo': halo,
                                    'glow': glow, 'box': box, 'edge': edge,
                                    'label': label, 'bar': bar, 'image': None,
                                    'dense': None, 'ref': ref}
            self._style_card(ref.uid)
            self._paint_thumb(ref, x0, y0, x1, y1)

        self._update_status()
        self._enqueue_visible()

    def _caption(self, ref: PageRef, pixels: int) -> str:
        """`p12 · document`, shortened to the card's width.

        The page number comes first because it is the half that must survive
        the ellipsis: two cards from the same document are told apart by it.
        """
        source = self.arrangement.source(ref.source)
        return self._fit(f"p{ref.index + 1} · {source.label}", pixels)

    def _fit(self, text: str, pixels: int) -> str:
        try:
            if self._caption_font.measure(text) <= pixels:
                return text
            trimmed = text
            while trimmed and self._caption_font.measure(trimmed + "…") > pixels:
                trimmed = trimmed[:-1]
            return (trimmed + "…") if trimmed else "…"
        except tk.TclError:
            return text

    def _paint_thumb(self, ref: PageRef, x0, y0, x1, y1) -> None:
        card = self._cards.get(ref.uid)
        if card is None:
            return
        key = self._key_for(ref)
        held = self._cache.get(key)
        centre_x = (x0 + x1) / 2
        centre_y = y0 + CARD_PAD + (y1 - y0 - LABEL_H - 2 * CARD_PAD) / 2
        if held is None:
            if card.get('text') is None:
                card['text'] = self.canvas.create_text(
                    centre_x, centre_y, text='...', fill='#9aa8b4',
                    font=look.font(10))
            return
        kind, value = held
        if kind == 'image':
            if card.get('image') is None:
                if card.get('text') is not None:
                    self.canvas.delete(card['text'])
                    card['text'] = None
                card['image'] = self.canvas.create_image(centre_x, centre_y,
                                                         image=value)
                self._displayed.add(key)
            else:
                self.canvas.itemconfigure(card['image'], image=value)
        elif kind == 'dense':
            if card.get('dense') is None:
                if card.get('text') is not None:
                    self.canvas.delete(card['text'])
                    card['text'] = None
                card['dense'] = self.canvas.create_text(
                    centre_x, centre_y,
                    text=f"{value:,} strokes\n(not drawn at this size)",
                    fill='#4a5a68', font=look.font(9), justify='center')
        # an 'error' card keeps its dots, and the status says a page failed

    def _show_photo(self, key, photo) -> None:
        for ref in self.arrangement:
            if self._key_for(ref) != key:
                continue
            card = self._cards.get(ref.uid)
            if card is None:
                continue
            x0, y0, x1, y1 = card['rect']
            self._paint_thumb(ref, x0, y0, x1, y1)

    def _show_dense(self, key) -> None:
        for ref in self.arrangement:
            if self._key_for(ref) != key:
                continue
            card = self._cards.get(ref.uid)
            if card is None:
                continue
            x0, y0, x1, y1 = card['rect']
            self._paint_thumb(ref, x0, y0, x1, y1)

    def _draw_empty(self) -> None:
        self.canvas.create_text(
            max(200, self.canvas.winfo_width() // 2), 120,
            text="No pages yet.\n\nAdd files... to open one or more .xopp notebooks,\n"
                 "then drag their pages into the order you want.",
            fill=MUTED, font=look.font(11), justify='center')

    def _update_status(self) -> None:
        selected = len(self.arrangement.selection)
        text = self.input_control
        if self.focused is not None:
            ref = self.arrangement.ref(self.focused)
            if ref is not None:
                text += f" · focus {self.arrangement.describe(ref)}"
        text += " · " + self.arrangement.summary()
        if selected:
            text += f" · {selected} selected"
        if self._opening:
            text += f" · opening {self._opening}"
        self.status.configure(text=text)

    def _say(self, message: str) -> None:
        """One line about what happened, kept in the log half of the bar.

        The state half stays state (mode, focus, counts); messages live on the
        right and keep the last few, with errors in red.
        """
        self._message = message
        self._message_at = time.monotonic()
        stamp = time.strftime('%H:%M:%S')
        self._log.append(f'{stamp} {message}')
        self._log = self._log[-3:]
        error = any(word in message.lower() for word in
                    ('error', 'refused', 'could not', 'missing', 'not found',
                     'nothing to save', 'no longer'))
        try:
            self.log.configure(text='    ' + '   |   '.join(reversed(self._log)),
                               foreground='#ff7b72' if error else '')
        except (tk.TclError, AttributeError):
            pass
        self._update_status()

    def _hint_text(self) -> str:
        return ("Ctrl+L: mouse/keyboard · WASD/arrows focus · Alt+WASD move "
                "selection · Ctrl+click/Space · Space live view (WASD pan, "
                "Shift+↑/↓ zoom) · Ctrl+S save & quit · Ctrl+Shift+S save as "
                "· Alt+Q quit · Ctrl+C/X/V · Del · Ctrl+Z/Y")

    # ---------------- focus, selection, and the order ----------------
    #
    # Three things that are easy to conflate and must not be:
    #
    #   focus     one page, shown by the bar on its left edge
    #   selection any number of pages, shown by the neon border
    #   block     the contiguous *selected* run that contains the focus
    #
    # The arrows move the focus, a lone Ctrl flips one page's selection,
    # Shift+arrows grows a range from an anchor, and WASD or a mouse drag
    # moves the block. The order is the arrangement's, and every change to it
    # is one undo step.

    def _shortcut(self, action):
        """Wrap a Ctrl shortcut so a lone Ctrl is never mistaken for it."""
        def run(event=None):
            self._ctrl_used = True
            action()
        return run

    def _range_uids(self, first: int, last: int):
        a = self.arrangement.position(first)
        b = self.arrangement.position(last)
        if a is None or b is None:
            return []
        lo, hi = min(a, b), max(a, b)
        return [ref.uid for ref in self.arrangement.pages[lo:hi + 1]]

    def _visible_first_uid(self):
        if not len(self.arrangement):
            return None
        positions = self._visible_positions()
        return self.arrangement[positions[0]].uid if positions else self.arrangement[0].uid

    def set_input_control(self, mode: str) -> None:
        """Ctrl+L: who decides where the focus bar is. Never touches selection."""
        if mode == self.input_control:
            return
        if mode == KEYBOARD:
            start = self._hover
            if start is None or self.arrangement.position(start) is None:
                start = self.focused
            if start is None or self.arrangement.position(start) is None:
                start = self._keyboard_focus
            if start is None or self.arrangement.position(start) is None:
                start = self._visible_first_uid()
            self.input_control = KEYBOARD
            self._set_focus(start)
            self._scroll_to_uid(start)
        else:
            self.input_control = MOUSE
            self._keyboard_focus = self.focused
        self._shift_anchor = None
        self._shift_base = set()
        self._refresh_selection()

    def toggle_input_control(self) -> None:
        self.set_input_control(MOUSE if self.input_control == KEYBOARD else KEYBOARD)

    def _set_focus(self, uid) -> None:
        if uid == self.focused:
            return
        previous, self.focused = self.focused, uid
        if previous is not None:
            self._style_card(previous)
        if uid is not None:
            self._style_card(uid)
        if self.input_control == KEYBOARD:
            self._keyboard_focus = uid
        self._update_status()

    def _toggle_focused(self) -> None:
        """A lone Ctrl: flip the page the Ctrl was pressed over, nothing else.

        The target is fixed at key-down: hovering other pages while Ctrl is
        held must not move the toggle to them.
        """
        uid = self._ctrl_target
        if uid is None or self.arrangement.position(uid) is None:
            return
        self.arrangement.toggle(uid)
        self._shift_anchor = None
        self._shift_base = set()
        self._refresh_selection()

    def _letter_move(self, event, dx: int, dy: int):
        """One handler for W/A/S/D in either case, split by modifier."""
        if event.state & CTRL:
            # Ctrl+letter belongs to the command bindings; Ctrl+W is nothing.
            return 'break'
        if event.state & ALT:
            self.move_selection(dx, dy)
        elif event.state & SHIFT:
            self.extend_selection(dx, dy)
        else:
            self.focus_move(dx, dy)
        return 'break'

    def _on_space(self, event=None):
        """Space opens the live view of the focused page, and closes it."""
        if isinstance(getattr(event, 'widget', None), (ttk.Button, tk.Button)):
            return None
        self.toggle_preview()
        return 'break'

    def toggle_preview(self) -> None:
        if self._preview is not None:
            self.close_preview()
            return
        if not len(self.arrangement):
            return
        uid = self.focused
        if uid is None or self.arrangement.position(uid) is None:
            uid = self._hover
        if uid is None or self.arrangement.position(uid) is None:
            uid = self.arrangement.selected_positions()
            uid = self.arrangement[uid[0]].uid if uid else self.arrangement[0].uid
        ref = self.arrangement.ref(uid)
        if ref is not None:
            self.open_preview(ref)

    def focus_move(self, dx: int, dy: int) -> None:
        """A plain direction key: the cursor moves, nothing else does.

        This is true however many pages are selected. `W`/`A`/`S`/`D` and the
        arrow keys are the same call.
        """
        if self._preview is not None:
            return
        if self._ctrl_down:
            self._ctrl_used = True
        if not len(self.arrangement):
            return
        if self.focused is None or self.arrangement.position(self.focused) is None:
            self._set_focus(self._visible_first_uid())
        columns = self._layout()[0]
        at = self.arrangement.position(self.focused) + dx + dy * columns
        at = max(0, min(at, len(self.arrangement) - 1))
        self._set_focus(self.arrangement[at].uid)
        self._scroll_to_uid(self.focused)
        # A plain move ends any Shift range; the next Shift starts at the focus.
        self._shift_anchor = None
        self._shift_base = set()

    def move_selection(self, dx: int, dy: int) -> None:
        """Ctrl + direction: move the whole selection, as one ordered whole.

        Nothing happens without a selection. Horizontal steps are one page,
        vertical steps one grid row; a step that would leave the row or the
        grid stops rather than wrapping. The selection stays selected, and the
        move is one undo step.
        """
        if self._preview is not None:
            return
        if self._ctrl_down:
            self._ctrl_used = True
        if not len(self.arrangement):
            return
        selection = self.arrangement.selection
        if not selection:
            return
        columns = self._layout()[0]
        selected = self.arrangement.selected_positions()
        first, count = selected[0], len(selected)
        last_slot = len(self.arrangement) - count
        if dy:
            target = first + dy * columns
            if target < 0 or target > last_slot:
                return
        else:
            column = first % max(1, columns)
            if (dx > 0 and column == columns - 1) or (dx < 0 and column == 0):
                return
            target = first + dx
            if target < 0 or target > last_slot:
                return
        self._shift_anchor = None
        self._shift_base = set()
        if self.arrangement.move_units(selection, target):
            self._after_edit()
            self._scroll_to_uid(self.focused)

    def extend_selection(self, dx: int, dy: int) -> None:
        """Shift + direction: move the cursor and grow the range from an anchor.

        The anchor is the cursor where the Shift range started, never an older
        Ctrl pick: pages selected discretely before stay selected, and the
        range is recomputed from the anchor on every step, so moving back
        shrinks it.
        """
        if self._preview is not None:
            return
        if not len(self.arrangement):
            return
        if self.focused is None or self.arrangement.position(self.focused) is None:
            self._set_focus(self._visible_first_uid())
        if (self._shift_anchor is None
                or self.arrangement.position(self._shift_anchor) is None):
            self._shift_anchor = self.focused
            self._shift_base = set(self.arrangement.selection)
        columns = self._layout()[0]
        at = self.arrangement.position(self.focused) + dx + dy * columns
        at = max(0, min(at, len(self.arrangement) - 1))
        self._set_focus(self.arrangement[at].uid)
        self.arrangement.set_selection(
            set(self._shift_base)
            | set(self._range_uids(self._shift_anchor, self.focused)))
        self._refresh_selection()
        self._scroll_to_uid(self.focused)

    def _scroll_to_uid(self, uid) -> None:
        if uid is None:
            return
        position = self.arrangement.position(uid)
        if position is None:
            return
        columns, _cell_w, cell_h = self._layout()
        row = position // max(1, columns)
        rows = max(1, -(-len(self.arrangement) // max(1, columns)))
        row_px = cell_h + GAP
        top = self.canvas.canvasy(0)
        bottom = top + max(1, self.canvas.winfo_height())
        y = PAD + row * row_px
        if y < top:
            self.canvas.yview_moveto(max(0.0, (y - GAP) / (rows * row_px)))
        elif y + row_px > bottom:
            self.canvas.yview_moveto(
                max(0.0, (y + row_px - max(1, self.canvas.winfo_height()) + GAP)
                    / (rows * row_px)))

    # ---------------- mouse ----------------

    def _on_press(self, event) -> None:
        if self._preview is not None:
            return
        self.canvas.focus_set()
        if event.state & CTRL:
            self._ctrl_used = True
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        uid = self._uid_at(x, y)
        self._press = (uid, x, y)
        self._dragging = False
        self._drag_block = set()
        if uid is None:
            if not (event.state & CTRL):
                self.arrangement.clear_selection()
                self._shift_anchor = None
                self._shift_base = set()
                self._refresh_selection()
            return
        self._set_focus(uid)
        if event.state & SHIFT:
            # Shift+click starts or extends a range. Without a live anchor the
            # first one only adds the target and anchors there -- a Ctrl pick
            # is not a range start.
            if self._shift_anchor is None:
                self._shift_base = set(self.arrangement.selection)
                self._shift_anchor = uid
            self.arrangement.set_selection(
                set(self._shift_base)
                | set(self._range_uids(self._shift_anchor, uid)))
        elif event.state & CTRL:
            # Discrete: toggle one page, and end any live shift range.
            self.arrangement.toggle(uid)
            self._shift_anchor = None
            self._shift_base = set()
        else:
            # A plain click is the traditional range start. A page that is
            # already selected keeps the selection until the release, so a
            # drag can carry the whole of it.
            if not self.arrangement.is_selected(uid):
                self.arrangement.select_only(uid)
            self._shift_anchor = uid
            self._shift_base = set()
        self._drag_block = set(self.arrangement.selection) or {uid}
        self._refresh_selection()

    def _on_motion(self, event) -> None:
        if self._preview is not None:
            return
        if self._press is None or self._press[0] is None:
            self._press = None
            return
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        if not self._dragging:
            dx, dy = x - self._press[1], y - self._press[2]
            if abs(dx) + abs(dy) < 8:
                return
            self._dragging = True
        self._insert_at = self._insertion_at(x, y)
        self._draw_insert_line()

    def _on_release(self, event) -> None:
        if self._preview is not None:
            self._press = None
            return
        press, self._press = self._press, None
        if press is None:
            return
        if self._dragging:
            self._dragging = False
            self._dragged_at = time.monotonic()
            index = self._insert_at
            self._insert_at = None
            self._clear_insert_line()
            if index is not None:
                chosen = self._drag_block or ({press[0]} if press[0] else set())
                slot = self.arrangement.slot_for_index(chosen, index)
                if self.arrangement.move_units(chosen, slot):
                    self._after_edit()
                    self._scroll_to_uid(self.focused)
        elif press[0] is not None and (event.state & (CTRL | SHIFT)) == 0:
            # A plain click on one of several selected cards means "just this
            # one" -- after the release, so a drag can still carry the whole
            # selection it started on.
            if len(self.arrangement.selection) > 1:
                self.arrangement.select_only(press[0])
                self._drag_block = {press[0]}
                self._refresh_selection()

    def _on_double(self, event) -> None:
        # A modifier-click pair is a selection gesture, not a double-click on
        # a page: Ctrl A then Ctrl A quickly is "toggle off", not "preview".
        # Tk sends the second press only as Double-Button-1, so the click has
        # to be replayed here or the toggle is lost.
        if event.state & (CTRL | SHIFT):
            self._on_press(event)
            self._on_release(event)
            return
        # A drag that ended where it started is not half of a double-click;
        # without this, dragging a page and clicking it again quickly opens
        # the preview on top of the arrangement.
        if time.monotonic() - self._dragged_at < 0.6:
            return
        if self._preview is not None:
            return
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        uid = self._uid_at(x, y)
        if uid is None:
            return
        ref = self.arrangement.ref(uid)
        if ref is not None:
            self.open_preview(ref)

    # ---------------- Ctrl: lone press versus shortcut ----------------

    def _on_ctrl_press(self, event=None) -> None:
        self._ctrl_down = True
        self._ctrl_used = False
        self._ctrl_at = time.monotonic()
        # A lone Ctrl toggles the **cursor**: the focused page. The pointer
        # may rest over another card (the arrow keys can move the focus while
        # the mouse stays still), so hovering is only the fallback when there
        # is no focus at all. The target is fixed now, so moving the pointer
        # during the hold cannot change it.
        self._ctrl_target = (self.focused if self.focused is not None
                             else self._hover)

    def _on_ctrl_release(self, event=None) -> None:
        lone = (self._ctrl_down and not self._ctrl_used
                and time.monotonic() - self._ctrl_at < 1.2)
        self._ctrl_down = False
        self._ctrl_used = False
        if lone:
            self._toggle_focused()

    def _on_any_key(self, event=None):
        """Letters in every case: Ctrl commands, Alt moves, Shift ranges,
        plain focus. Dispatching on `event.keysym.lower()` is what makes the
        keys work with CapsLock on, where Tk reports `S` instead of `s`.
        """
        if event is None:
            return None
        keysym = (event.keysym or '').lower()
        state = event.state
        if state & CTRL:
            if keysym not in ('control_l', 'control_r'):
                self._ctrl_used = True
            commands = {
                'c': self.copy_selected,
                'x': self.cut_selected,
                'v': self.paste_clipboard,
                'z': self.redo if state & SHIFT else self.undo,
                'y': self.redo,
                'l': self.toggle_input_control,
                'a': self.select_all,
                'd': self.duplicate_selected,
                's': self.save_as if state & SHIFT else self.quick_save,
                # No 'q' here. The quit key is Alt+Q, armed by
                # xp.runtime.shutdown, and this entry was the reason a real
                # Ctrl+Q still shut the arranger's window after every
                # `bind`-level Ctrl+Q in XP had been removed: a command
                # dispatched by keysym inside a <KeyPress> handler is
                # invisible to anything that asks Tk what is bound.
            }
            action = commands.get(keysym)
            if action is not None:
                action()
                return 'break'
            return None
        if keysym in ('w', 'a', 's', 'd'):
            directions = {'w': (0, -1), 'a': (-1, 0),
                          's': (0, 1), 'd': (1, 0)}
            dx, dy = directions[keysym]
            if state & ALT:
                self.move_selection(dx, dy)
            elif state & SHIFT:
                self.extend_selection(dx, dy)
            else:
                self.focus_move(dx, dy)
            return 'break'
        return None

    def _on_escape(self) -> None:
        if self._dragging:
            # A drag in flight is cancelled: nothing moves, nothing is lost.
            self._dragging = False
            self._press = None
            self._insert_at = None
            self._clear_insert_line()
            return
        if self._preview is not None:
            self.close_preview()
            return
        self._clear_selection()

    # ---------------- preview ----------------

    def preview_selected(self) -> None:
        if self._preview is not None or not self.arrangement.selection:
            return
        ref = None
        if self.focused is not None and self.arrangement.is_selected(self.focused):
            ref = self.arrangement.ref(self.focused)
        if ref is None:
            ref = self.arrangement.selected()[-1]
        self.open_preview(ref)

    def open_preview(self, ref: PageRef) -> None:
        """The big in-window preview of one page (see preview.py)."""
        if self._preview is not None:
            return
        if self.on_open_page is not None:
            self.on_open_page(ref)
            return
        from .pageart import PageArt
        from .preview import PagePreview

        source = self.arrangement.source(ref.source)
        art = PageArt(source.side.page_element(ref.index),
                      source.path.parent if source.path else None,
                      resolve_pdf=self._resolve_missing_pdf)
        missing = art.background_missing()
        note = (f'PDF background not found: {Path(missing).name}'
                if missing else '')
        title = f'p{ref.index + 1} · {source.label}'
        self._preview = PagePreview(self.root, art, title,
                                    on_close=self._preview_closed, note=note)

    def _preview_closed(self) -> None:
        self._preview = None
        try:
            self.canvas.focus_set()
        except tk.TclError:
            pass

    def close_preview(self) -> None:
        if self._preview is not None:
            self._preview.close()
            self.canvas.focus_set()

    def _insertion_at(self, x: float, y: float) -> int:
        columns, cell_w, cell_h = self._layout()
        total = len(self.arrangement)
        if not total:
            return 0
        row = int(max(0, (y - PAD)) // (cell_h + GAP))
        col = int(max(0, (x - PAD)) // (cell_w + GAP))
        index = row * columns + min(col, columns)
        left, _top, right, _bottom = self.card_rect(min(index, total - 1))
        if index < total and x > left + (right - left) * 0.55:
            index += 1
        return max(0, min(index, total))

    def _draw_insert_line(self) -> None:
        self._clear_insert_line()
        index = self._insert_at
        if index is None:
            return
        columns, cell_w, cell_h = self._layout()
        total = len(self.arrangement)
        if not total:
            return
        row, col = divmod(index, columns)
        if index >= total:
            last_row, last_col = divmod(total - 1, columns)
            row, col = (last_row, last_col + 1) if last_col + 1 < columns else (last_row + 1, 0)
        x = PAD + col * (cell_w + GAP) - GAP / 2
        y0 = PAD + row * (cell_h + GAP)
        y1 = y0 + cell_h
        self._insert_line = self.canvas.create_line(
            x, y0, x, y1, fill=INSERT, width=4, capstyle='round')

    def _clear_insert_line(self) -> None:
        if self._insert_line is not None:
            self.canvas.delete(self._insert_line)
            self._insert_line = None

    def _style_card(self, uid: int) -> None:
        """Focused, selected, hovered -- the one place a card's look is set.

        Focus and selection are independent: a focused page gets the bar on
        its left edge whether or not it is selected, and a selected page glows
        whether or not it is focused. The three states can all be on at once.
        """
        card = self._cards.get(uid)
        if card is None:
            return
        selected = self.arrangement.is_selected(uid)
        focused = uid == self.focused
        hovered = uid == self._hover and not selected and not focused
        self.canvas.itemconfigure(card['halo'],
                                  state='normal' if selected else 'hidden')
        self.canvas.itemconfigure(card['glow'],
                                  state='normal' if selected else 'hidden')
        self.canvas.itemconfigure(card['bar'],
                                  state='normal' if focused else 'hidden')
        self.canvas.itemconfigure(
            card['edge'],
            outline=GLOW_CORE if selected else (HOVER_EDGE if hovered else CARD_EDGE),
            width=2 if selected else 1)
        self.canvas.itemconfigure(card['box'],
                                  fill=HOVER_FILL if hovered else PAPER)

    def _refresh_selection(self) -> None:
        for uid in self._cards:
            self._style_card(uid)
        self._update_status()

    def _set_hover(self, uid: Optional[int]) -> None:
        if uid == self._hover:
            if self.input_control == MOUSE and uid != self.focused:
                self._set_focus(uid)
            return
        previous, self._hover = self._hover, uid
        if previous is not None:
            self._style_card(previous)
        if uid is not None:
            self._style_card(uid)
        # In mouse control the pointer is the focus: the bar follows it, and
        # following it is not selecting it. While Ctrl is held the focus is
        # frozen, because the release toggles the page it was pressed over.
        if self.input_control == MOUSE and not self._ctrl_down:
            self._set_focus(uid)

    def _on_hover(self, event) -> None:
        if self._preview is not None:
            return
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self._set_hover(self._uid_at(x, y))

        return True

    def _clear_selection(self) -> None:
        self.arrangement.clear_selection()
        self._shift_anchor = None
        self._shift_base = set()
        self._ctrl_target = None
        self._refresh_selection()

    def select_all(self) -> None:
        if self._preview is not None:
            return
        self.arrangement.select_all()
        if self.focused is None and len(self.arrangement):
            self._set_focus(self.arrangement[0].uid)
        self._refresh_selection()

    # ---------------- commands ----------------

    def delete_selected(self) -> None:
        if self._preview is not None:
            return
        if not self.arrangement.selection:
            self._say("nothing selected")
            return
        gone = len(self.arrangement.selection)
        self.arrangement.remove(self.arrangement.selection)
        self._say(f"deleted {gone} page(s)")
        self._after_edit()

    def duplicate_selected(self) -> None:
        if self._preview is not None:
            return
        if not self.arrangement.selection:
            self._say("nothing selected")
            return
        made = self.arrangement.duplicate(self.arrangement.selection)
        self._say(f"duplicated {len(made)} page(s)")
        self._after_edit()

    def copy_selected(self) -> None:
        if self._preview is not None:
            return
        if not self.arrangement.selection:
            self._say("nothing selected")
            return
        self._say(f"copied {self.arrangement.copy(self.arrangement.selection)} page(s)")

    def cut_selected(self) -> None:
        if self._preview is not None:
            return
        if not self.arrangement.selection:
            self._say("nothing selected")
            return
        count = self.arrangement.cut(self.arrangement.selection)
        self._say(f"cut {count} page(s)")
        self._after_edit()

    def paste_clipboard(self) -> None:
        if self._preview is not None:
            return
        made = self.arrangement.paste()
        if not made:
            self._say("clipboard is empty")
            return
        self._say(f"pasted {len(made)} page(s)")
        self._after_edit()

    def undo(self) -> None:
        if self._preview is not None:
            return
        if self.arrangement.undo():
            self._say("undo")
            self._after_edit()

    def redo(self) -> None:
        if self._preview is not None:
            return
        if self.arrangement.redo():
            self._say("redo")
            self._after_edit()

    def _ask_use_pdf(self, original: Path, found: Path) -> bool:
        """One question per missing background: use the file found nearby?"""
        return messagebox.askyesno(
            "PDF background not found",
            f"The background PDF was not found:\n{original}\n\n"
            f"Use this file instead?\n{found}",
            parent=self.root)

    def _resolve_missing_pdf(self, original: Path, folder) -> Optional[Path]:
        """Find a same-named PDF next to the notebook, and ask before using it.

        The answer (yes or no) is cached per folder and file name, so the
        dialog appears once, not once per page.
        """
        here = Path(folder) if folder is not None else original.parent
        key = (str(here), original.name.lower())
        if key in self._pdf_cache:
            return self._pdf_cache[key]
        found = None
        for where in (here, original.parent):
            try:
                if not where.is_dir():
                    continue
                for child in where.iterdir():
                    if (child.is_file()
                            and child.name.lower() == original.name.lower()):
                        found = child
                        break
            except OSError:
                continue
            if found is not None:
                break
        answer = None
        if found is not None and self._ask_use_pdf(original, found):
            answer = found
        self._pdf_cache[key] = answer
        return answer

    def _close_soon(self) -> None:
        """Quit after the current callback, so a write finishes first."""
        try:
            self.root.after(1, self.close)
        except tk.TclError:
            pass

    def _save_ready(self) -> bool:
        """Shared checks for both save keys: nothing to write, or a PDF page."""
        if self._preview is not None:
            self.close_preview()
        if not len(self.arrangement):
            self._say("nothing to save: the order is empty")
            return False
        pdfs = [source.label for source in self.arrangement.sources
                if source.kind != 'xopp' and self.arrangement.pages_of(source.uid)]
        if pdfs:
            self._say("a PDF page has no strokes to write yet (needs the "
                      "converter): " + ", ".join(pdfs))
            return False
        return True

    def quick_save(self) -> None:
        """Ctrl+S: write the default name without asking.

        The first save of a session goes to `<document>-Arra1.xopp` (or the
        next free number). After that, Ctrl+S updates the same file, which is
        what a save key is expected to do; Ctrl+Shift+S renames.
        """
        if not self._save_ready():
            return
        path = self._save_path or self.suggested_save_name()
        self._say(f"saving {len(self.arrangement)} page(s)...")
        self.root.update_idletasks()
        self._start_save(path)

    def save_as(self) -> None:
        """Ctrl+Shift+S: open the name dialog, prefilled with the default."""
        if not self._save_ready():
            return
        suggested = self.suggested_save_name()
        path = filedialog.asksaveasfilename(
            title="Save the arranged document", defaultextension=".xopp",
            initialdir=str(suggested.parent), initialfile=suggested.name,
            filetypes=[("Xournal++ notebook", "*.xopp")], parent=self.root)
        if not path:
            return
        self._say(f"saving {len(self.arrangement)} page(s)...")
        self.root.update_idletasks()
        self._start_save(path)

    def suggested_save_name(self) -> Path:
        """The name proposed in the save dialog: `<document>-Arra<N>.xopp`.

        The number is the lowest one not taken yet, so pressing Enter in the
        dialog never overwrites an earlier arrangement. A stem that already
        ends in `-ArraN` is reduced to its base first, so saving an arranged
        file again gives `-Arra2`, not `-Arra1-Arra1`.
        """
        sources = [source for source in self.arrangement.sources
                   if source.path is not None]
        if sources:
            folder = sources[0].path.parent
            stem = sources[0].path.stem
        else:
            folder = Path.home()
            stem = "Arranger"
        stem = re.sub(r'-\s*Arra\d+$', '', stem) or "Arranger"
        number = 1
        while (folder / f"{stem}-Arra{number}.xopp").exists():
            number += 1
        return folder / f"{stem}-Arra{number}.xopp"

    def _start_save(self, path) -> None:
        """A test saves inline; the window saves behind the overlay."""
        self._save_path = Path(path)
        if not self.exit_after_save:
            self.save_to(path)
            return
        total = max(1, len(self.arrangement))
        self._progress_mode = 'save'
        self._save_running = True
        self._save_queue = queue.Queue()
        self._overlay_show(f"Saving {len(self.arrangement)} page(s)...",
                           total, f"0 of {total} page(s)")
        messages = self._save_queue

        def work():
            from .writer import build
            try:
                made = build(self.arrangement, path,
                             progress=lambda done, all_pages:
                             messages.put(('n', done, all_pages)))
                messages.put(('ok', made))
            except Exception as exc:       # noqa: BLE001 - shown in the overlay
                messages.put(('err', exc))

        threading.Thread(target=work, daemon=True).start()

    def save_to(self, path) -> bool:
        """Write to `path`; the dialog's answer arrives here, and a test's."""
        from .writer import Refused, build

        try:
            made = build(self.arrangement, path)
        except Refused as exc:
            self._say(str(exc))
            return False
        except OSError as exc:
            messagebox.showerror("Could not write the file", str(exc),
                                 parent=self.root)
            return False
        self._say(f"wrote {made.pages} page(s) from {made.documents} "
                  f"document(s) to {made.path.name}")
        if made.missing_backgrounds:
            self._say(f"wrote {made.pages} page(s) to {made.path.name} · "
                      f"{len(made.missing_backgrounds)} missing PDF background(s)")
            messagebox.showwarning(
                "Saved, but Xournal++ will refuse it",
                "These pages name a PDF background that is not on disk:\n\n"
                + "\n".join(made.missing_backgrounds)
                + "\n\nThe file is written and the pages are intact; the "
                  "source has the same missing file, so Xournal++ refuses "
                  "the source too.",
                parent=self.root)
        return True

    def bigger(self) -> None:
        if self._preview is not None:
            return
        self.set_thumb_width(self.thumb_index + 1)

    def smaller(self) -> None:
        if self._preview is not None:
            return
        self.set_thumb_width(self.thumb_index - 1)

    def set_thumb_width(self, index: int) -> None:
        """Change the thumbnail size, keeping the page under the pointer.

        The user's instruction, 2026-10-28: the view should be about where the mouse
        is pointing -- "the detail the mouse points at should stay the point
        of the zoom". The converter and the merge window already zoom about
        the pointer; here "zoom" reflows a grid, so the equivalent is to put
        the page that was under the pointer back under the pointer once the
        new layout exists. Without it the grid reflows from the top and the
        page the user was looking at slides away, which is the hunt the user described.

        When the pointer is not over a page -- it is on a button, or off the
        window -- there is nothing to anchor to and the top-left corner stays
        put, which is what it did before.
        """
        if self._preview is not None:
            return
        index = max(0, min(len(THUMB_WIDTHS) - 1, int(index)))
        if index == self.thumb_index:
            return
        anchor = self._page_under_the_pointer()
        self.thumb_index = index
        self._redraw()
        if anchor is not None:
            self._put_back_under_the_pointer(*anchor)

    def _page_under_the_pointer(self):
        """`(uid, how far down the window the pointer is)`, or None."""
        try:
            px = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
            py = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
            if not (0 <= px < self.canvas.winfo_width()
                    and 0 <= py < self.canvas.winfo_height()):
                return None
            uid = self._uid_at(self.canvas.canvasx(px), self.canvas.canvasy(py))
            return None if uid is None else (uid, py)
        except Exception:
            return None

    def _put_back_under_the_pointer(self, uid, py) -> None:
        """Scroll so `uid` sits `py` pixels down the window again."""
        try:
            position = self.arrangement.position(uid)
            if position is None:
                return
            columns, _cell_w, cell_h = self._layout()
            row = position // max(1, columns)
            want_top = PAD + row * (cell_h + GAP) - py
            # The fraction yview_moveto takes is of the scroll region, which
            # _redraw has just set, so ask the canvas rather than recompute it.
            region = [float(v) for v in str(
                self.canvas.cget('scrollregion')).split()]
            total = max(1.0, (region[3] - region[1]) if len(region) == 4
                        else float(self.canvas.winfo_height()))
            self.canvas.yview_moveto(max(0.0, min(1.0, want_top / total)))
            self._enqueue_visible()
        except Exception:
            pass

    # ---------------- events ----------------

    def _after_edit(self) -> None:
        self._redraw()
        self._scroll_to_selection()

    def _scroll_to_selection(self) -> None:
        selection = self.arrangement.selected()
        if not selection:
            return
        position = self.arrangement.position(selection[0].uid)
        if position is None:
            return
        _columns, _cell_w, cell_h = self._layout()
        total = len(self.arrangement)
        height = max(1, total * (cell_h + GAP))
        self.canvas.yview_moveto(max(0.0, min(1.0, position * (cell_h + GAP) / height)))

    def _on_wheel(self, event) -> None:
        if self._preview is not None:
            return
        steps = int(-event.delta / 120) or (-1 if event.delta > 0 else 1)
        self.canvas.yview_scroll(steps * 3, 'units')
        self._enqueue_visible()

    def _on_resize(self, event=None) -> None:
        """A real size change redraws; nothing here changes layout again.

        Older versions updated the hint's wraplength here, which asked for a
        new height, delivered another Configure, and could keep `update()`
        busy forever. The hint has a fixed wrap now, so this only redraws.
        """
        if event is not None and event.widget not in (self.root, self.canvas):
            return
        self._redraw()

    def _poll(self) -> None:
        if self._closing:
            return
        changed = self._drain_opened()
        if changed:
            self._redraw()
        else:
            self._drain_rendered()
            self._enqueue_visible()
        self._pump_open_progress()
        self._pump_save_progress()
        self._update_status()
        try:
            self._after_job = self.root.after(POLL_MS, self._poll)
        except tk.TclError:
            pass

    # ---------------- lifecycle ----------------

    def close(self) -> None:
        """The one way out, so the launcher hears about it exactly once."""
        if self._closing:
            return
        self._closing = True
        if self._preview is not None:
            try:
                self._preview.close()
            except tk.TclError:
                pass
            self._preview = None
        if self._after_job is not None:
            try:
                self.root.after_cancel(self._after_job)
            except tk.TclError:
                pass
            self._after_job = None
        self._open_queue.put(None)
        self._render_queue.put(None)
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        self.arrangement.close()
        self._finished = True
        if self.on_finish is not None:
            self.on_finish(True)

    def run(self) -> None:
        if self._owns_root:
            self.root.mainloop()

    def __repr__(self):
        return f'<ArrangerApp {len(self.arrangement)} pages>'
