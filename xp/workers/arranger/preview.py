"""The large preview: one page, in the window, sharp at every zoom.

Double-clicking a card opens this over the grid -- a panel about 70% of the
window, not a second application window. It draws the page from its own
elements (`pageart.PageArt`), so the content at 8x is the strokes and text at
8x, not a 136 px thumbnail enlarged until it is a blur.

**Moving the mouse pans it**, which is the thing the user asked for and then had
to point at twice: *"I mean inside here, also use visual.py, that thing. Not
outside -- outside shouldn't."* The first attempt put it on the grid of
cards, where the user never wanted it; this is the view the user reads a page in, and
this is where the page should come to the pointer.

It is `vp()` out of the user's own visual.py, in the same three pieces: a callback
that only records the pointer, an `adjust_to_mouse_center` that reads the
record, and a loop that calls the adjust every frame whether or not anything
moved. The viewer already had `vp()`'s state under different names -- `scale`
is `zoom_factor`, `ox/oy` is the crop's origin and the canvas is
`constant_ws` -- so the follow is `vp()`'s arithmetic over the names that
were here.

The wheel still zooms about the pointer and `Esc` or the close button goes
back. Nothing here writes to the arrangement: the preview has no reference to
it at all, which is why opening, panning and closing cannot change a page's
order or the selection.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional

from PIL import ImageTk

from . import look

GROUND = '#0b1219'          # the space around the page
PANEL = look.ABYSS
EDGE = '#1f7ae0'
TITLE = look.TEXT
MUTED = look.MUTED

MARGIN = 0.98               # fit leaves a hair of the page's edge visible
ZOOM_STEP = 1.2
FOLLOW_MS = 30              # `display_image`'s loop: a frame, not an event
MAX_PIXELS = 28_000_000     # one preview image is held; past this it is slow
MAX_SCALE = 20.0
DENSE_STROKES = 400_000     # past this, zoom is capped rather than waiting
DENSE_MAX_FACTOR = 8.0


class PagePreview:
    """A page over the grid: fit, pan, zoom, close."""

    def __init__(self, parent: tk.Misc, art, title: str,
                 on_close: Optional[Callable[[], None]] = None,
                 note: str = ''):
        self.art = art
        self.on_close = on_close
        self.title = title
        self.note = note
        self.scale = 1.0
        self.ox = 0.0
        self.oy = 0.0
        self._images: Dict[float, ImageTk.PhotoImage] = {}
        self._photo = None
        self._press = None
        self._fitted = False
        self._user_view = False
        # `vp()`'s `param`: where the pointer was last seen. Written by the
        # callback, read by the loop, acted on in neither place but the
        # adjust.
        self._pointer = {'x': None, 'y': None}
        self._follow_job = None

        self.frame = tk.Frame(parent, bg=PANEL, highlightthickness=2,
                              highlightbackground=EDGE)
        self.frame.place(relx=0.5, rely=0.5, anchor='center',
                         relwidth=0.72, relheight=0.86)
        self.frame.lift()

        head = tk.Frame(self.frame, bg=PANEL)
        head.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(6, 2))
        self.title_label = ttk.Label(head, text=title, style='Lab.TLabel')
        self.title_label.pack(side=tk.LEFT)
        self.zoom_label = ttk.Label(head, text='', style='Muted.Lab.TLabel')
        self.zoom_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(head, text='Close  (Esc)', command=self.close).pack(side=tk.RIGHT)
        if note:
            # Its own line: packed beside the title it was cut off by the
            # close button, and a truncated path is not a warning.
            ttk.Label(self.frame, text=note, style='Muted.Lab.TLabel',
                      wraplength=900, justify='left').pack(
                side=tk.TOP, fill=tk.X, padx=10, pady=(0, 2))

        self.canvas = tk.Canvas(self.frame, bg=GROUND, highlightthickness=0,
                                takefocus=True)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6,
                         pady=(2, 6))

        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<MouseWheel>', self._on_wheel)
        self.canvas.bind('<Motion>', self._record_the_mouse)
        self.canvas.bind('<Leave>', self._forget_the_mouse)
        self.canvas.bind('<Escape>', self._esc)
        self.frame.bind('<Escape>', self._esc)
        self.canvas.bind('<space>', self._space)
        self.frame.bind('<space>', self._space)
        for key, dx, dy in (('w', 0, -1), ('a', -1, 0),
                            ('s', 0, 1), ('d', 1, 0)):
            for case in (key, key.upper()):
                self.canvas.bind(f'<KeyPress-{case}>',
                                 lambda e, dx=dx, dy=dy: self._pan_key(dx, dy))
                self.frame.bind(f'<KeyPress-{case}>',
                                lambda e, dx=dx, dy=dy: self._pan_key(dx, dy))
        for widget in (self.canvas, self.frame):
            widget.bind('<Shift-Up>', lambda e: self._zoom_key(1.25))
            widget.bind('<Shift-Down>', lambda e: self._zoom_key(0.8))
        self.frame.bind('<Configure>', self._on_resize)
        self.frame.after(30, self._take_focus)

        self.canvas.focus_set()
        self.frame.update_idletasks()
        self.fit()
        self.frame.after(30, self.fit)
        self._follow_loop()

    # ---------------- view ----------------

    def _canvas_size(self):
        width = max(120, self.canvas.winfo_width())
        height = max(120, self.canvas.winfo_height())
        return width, height

    def _max_scale(self) -> float:
        width, height = self.art.width, self.art.height
        by_pixels = (MAX_PIXELS / max(1.0, width * height)) ** 0.5
        limit = min(MAX_SCALE, by_pixels)
        if self.art.stroke_count > DENSE_STROKES:
            limit = min(limit, self._fit_scale() * DENSE_MAX_FACTOR)
        return max(0.1, limit)

    def _fit_scale(self) -> float:
        cw, ch = self._canvas_size()
        return min((cw - 8) / self.art.width,
                   (ch - 8) / self.art.height) * MARGIN

    def fit(self) -> None:
        cw, ch = self._canvas_size()
        self.scale = max(0.05, min(self._fit_scale(), self._max_scale()))
        self.ox = (self.art.width - cw / self.scale) / 2.0
        self.oy = (self.art.height - ch / self.scale) / 2.0
        self._fitted = True
        self._show()

    def _clamp(self) -> None:
        cw, ch = self._canvas_size()
        view_w, view_h = cw / self.scale, ch / self.scale
        if view_w >= self.art.width:
            self.ox = (self.art.width - view_w) / 2.0
        else:
            self.ox = min(max(0.0, self.ox), self.art.width - view_w)
        if view_h >= self.art.height:
            self.oy = (self.art.height - view_h) / 2.0
        else:
            self.oy = min(max(0.0, self.oy), self.art.height - view_h)

    def _page_image(self):
        held = self._images.get(self.scale)
        if held is not None:
            return held
        made = self.art.render(self.scale)
        photo = ImageTk.PhotoImage(made, master=self.canvas)
        # Two scales are kept: zooming in and straight back out is common, and
        # re-rendering the page to go back is the slow half.
        if len(self._images) >= 2:
            self._images.pop(next(iter(self._images)))
        self._images[self.scale] = photo
        return photo

    def _show(self) -> None:
        self._clamp()
        self._photo = self._page_image()
        self.canvas.delete('page')
        self.canvas.create_image(-self.ox * self.scale, -self.oy * self.scale,
                                 image=self._photo, anchor='nw', tags='page')
        self.zoom_label.configure(text=f'{self.scale * 100 / max(0.05, self._fit_scale()):.0f}%'
                                       f'  ·  move the mouse to pan · wheel to zoom')

    def _on_resize(self, event=None) -> None:
        # Until the user has panned or zoomed, the page is kept fitted: the
        # first sizes a Toplevel reports are not the ones it settles on, and
        # a preview that opened at the wrong scale is a blurry first look.
        if not self._fitted or not self._user_view:
            self.fit()
            return
        self._show()

    # ---------------- the view follows the mouse ----------------
    #
    # `vp()`, kept in its three pieces. The loop is the piece an event
    # handler cannot stand in for: bound to `<Motion>` alone the view only
    # moves when Tk happens to deliver an event, so it stalls between events
    # and after a zoom. `vp()` never asks whether the mouse moved -- it
    # redraws from `param["mouse_x"]` forever.

    def _record_the_mouse(self, event) -> None:
        """`mouse_callback`: store the pointer, and do no view work here."""
        self._pointer['x'] = event.x
        self._pointer['y'] = event.y

    def _forget_the_mouse(self, _event=None) -> None:
        """Pointer off the page: hold the last view rather than jumping."""
        self._pointer['x'] = None
        self._pointer['y'] = None

    def _follow_loop(self) -> None:
        """`display_image`'s `while True`: adjust from the record, forever."""
        if not self.frame.winfo_exists():
            self._follow_job = None
            return
        try:
            self._adjust_to_mouse_center()
        except Exception:
            # A view that cannot follow holds still; it does not take the
            # window down with it.
            pass
        try:
            self._follow_job = self.frame.after(FOLLOW_MS, self._follow_loop)
        except Exception:
            self._follow_job = None

    def _adjust_to_mouse_center(self) -> None:
        """`adjust_to_mouse_center`, over this viewer's names.

        In `vp()`:

            relative_mouse_x = mouse_x / constant_ws[0]
            start_x = relative_mouse_x * new_w - constant_ws[0] // 2

        `new_w` is the page at the current zoom (`art.width * scale`) and
        `start_x` is the crop's left edge in screen pixels; this viewer keeps
        that origin in page units as `ox`, so the same line divided through
        by `scale` is:

            ox = relative * art.width - (cw / scale) / 2

        `_clamp` then holds the crop inside the page, which is `vp()`'s
        `max(0, ...)` and `min(..., new_w)` without the white padding -- the user
        asked for no large blank areas.

        It stands down during a drag: `_press` is set between a press and its
        release, and a page that slides out from under a drag is not a page
        anyone can drag.
        """
        if self._press is not None or not self._fitted:
            return
        px, py = self._pointer['x'], self._pointer['y']
        if px is None or py is None:
            return
        cw, ch = self._canvas_size()
        if not (0 <= px < cw and 0 <= py < ch):
            return
        before = (self.ox, self.oy)
        self.ox = (min(1.0, max(0.0, px / float(cw))) * self.art.width
                   - (cw / self.scale) / 2.0)
        self.oy = (min(1.0, max(0.0, py / float(ch))) * self.art.height
                   - (ch / self.scale) / 2.0)
        self._clamp()
        if (self.ox, self.oy) != before:
            self._user_view = True
            self._show()

    # ---------------- mouse ----------------

    def _on_press(self, event) -> None:
        self._user_view = True
        self._press = (event.x, event.y, self.ox, self.oy)

    def _on_drag(self, event) -> None:
        if self._press is None:
            return
        px, py, ox, oy = self._press
        self.ox = ox - (event.x - px) / self.scale
        self.oy = oy - (event.y - py) / self.scale
        self._show()

    def _on_release(self, _event) -> None:
        self._press = None

    # ---------------- keyboard: pan and zoom the live view ----------------

    def _take_focus(self) -> None:
        if self.canvas.winfo_exists():
            self.canvas.focus_set()

    def pan_by(self, dx_px: float, dy_px: float) -> None:
        self._user_view = True
        self.ox += dx_px / self.scale
        self.oy += dy_px / self.scale
        self._clamp()
        self._show()

    def _pan_key(self, dx: int, dy: int):
        self.pan_by(dx * 60, dy * 60)
        return 'break'

    def zoom_by(self, factor: float) -> None:
        new_scale = max(0.05, min(self.scale * factor, self._max_scale()))
        if abs(new_scale - self.scale) < 1e-6:
            return
        self._user_view = True
        cw, ch = self._canvas_size()
        cx, cy = cw / 2.0, ch / 2.0
        page_x = self.ox + cx / self.scale
        page_y = self.oy + cy / self.scale
        self.scale = new_scale
        self.ox = page_x - cx / self.scale
        self.oy = page_y - cy / self.scale
        self._clamp()
        self._show()

    def _zoom_key(self, factor: float):
        self.zoom_by(factor)
        return 'break'

    def _space(self, event=None):
        self.close()
        return 'break'

    def _on_wheel(self, event) -> None:
        if not event.delta:
            return
        self._user_view = True
        steps = event.delta / 120.0
        new_scale = self.scale * (ZOOM_STEP ** steps)
        new_scale = max(0.05, min(new_scale, self._max_scale()))
        if abs(new_scale - self.scale) < 1e-6:
            return
        # Keep the page point under the pointer where it is.
        page_x = self.ox + event.x / self.scale
        page_y = self.oy + event.y / self.scale
        self.scale = new_scale
        self.ox = page_x - event.x / self.scale
        self.oy = page_y - event.y / self.scale
        self._show()

    # ---------------- lifecycle ----------------

    def _esc(self, event=None):
        """Esc closes and stops the key here.

        The grid binds Escape on the toplevel too; without the 'break' it runs
        after this closes the preview and clears the selection the preview was
        opened over.
        """
        self.close()
        return 'break'

    def close(self) -> None:
        if not self.frame.winfo_exists():
            return
        if self._follow_job is not None:
            try:
                self.frame.after_cancel(self._follow_job)
            except Exception:
                pass
            self._follow_job = None
        self.canvas.destroy()
        self.frame.destroy()
        if self.on_close is not None:
            self.on_close()
