"""tkinter review/merge GUI -- the WinMerge-like part.

Shows the current page's diff (left vs right) immediately on load, muting
strokes that roughly match on both sides so real differences stand out. You
then paint with a three-color brush -- Keep Left / Take Right / Take Both --
to decide what survives; a stroke is only captured once every one of its
points is inside the painted area (or inside a closed loop you draw -- see
`_on_release`). Nothing here needs the mouse to be precise: overshoot the
brush freely, only full coverage counts.
"""

from __future__ import annotations

import tkinter as tk
import os
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from xp.workers.merger import views
from xp.workers.merger.brush import BrushMask
from xp.workers.merger.compose import LEFT, RIGHT, Ref
from xp.workers.merger.merge import (MAX_CLASSIFY_DPI, MIN_CLASSIFY_DPI,
                                     TAKE_BOTH, TAKE_LEFT, TAKE_NONE,
                                     TAKE_RIGHT, MergeSession)
from xp.render.appearance import ABYSS, decorate
from xp.render.drawing import BACKGROUND_COLOR, draw_strokes
from xp.workers.merger.views import DECIDED, LEFT_ONLY, OVERLAP, RIGHT_ONLY

# How wide a label whose text comes from the document is allowed to get before
# it wraps. Three of them carry a page's geometry reason, a fit ratio and a
# split note -- sentences of no fixed length. A label that never wraps asks
# the window to be as wide as the longest sentence it may ever hold, and Tk
# obeys: the row then pushes the buttons at the end of it past the edge, which
# is the "text and buttons cut off" the user reported. Text wraps; windows do not.
WRAP_LONG = 420

NONE_BRUSH_COLOR = (120, 120, 120)   # the "neither side" brush tint
BOTH_BRUSH_COLOR = (170, 170, 170)   # the "take both" brush tint
BRUSH_TINT = {TAKE_LEFT: LEFT_ONLY, TAKE_RIGHT: RIGHT_ONLY,
              TAKE_BOTH: BOTH_BRUSH_COLOR, TAKE_NONE: NONE_BRUSH_COLOR}

# How far past the window the page is drawn, as a share of the window.
# Panning inside the margin costs nothing; a longer drag draws once.
# A share and not a pixel count, so it means the same at every zoom.
OVERSCAN = 0.35

# **How long a sharpen may take before it is given up on.** The render is of
# the window, not of the page, so a few hundred milliseconds is the normal case
# and about a second at 4x. This is a watchdog for the case where the worker
# never comes back at all -- a lock held by a pan's render, a page the
# allocator will not give, a driver that stalls. Without it the window polls
# every 30 ms for ever and the picture stays blurred with nothing said, which
# looks exactly like "the render never finished". Generous on purpose: giving
# up on a render that would have landed is worse than waiting, and a missed
# one is retried.
SHARPEN_DEADLINE = 15.0
# How many times over the sharp render may sample, and the ceiling on the
# pixels that buys. See `MergeApp._supersample`: this is where the user's dpi field
# finally reaches the picture, and where the strokes stop being hard-edged.
# How long after the user's hand leaves the pen before a background render may
# start. See `_sharpen`: a render is interaction-sized work now, and one
# begun between two strokes is still running during the next.
BRUSH_GRACE = 0.25              # seconds
SUPERSAMPLE_FLOOR = 2           # coverage: the rasteriser has none of its own
SUPERSAMPLE_MAX = 3
# The budget is on the **samples taken**, not on the rectangle asked for.
# Counting the rectangle and then flooring the square root is how a full
# screen silently lost its antialiasing: the user's window wants about 4.6 M pixels,
# sqrt(12M / 4.6M) is 1.6, and `int` made that 1 -- the label said `x1` and
# the edges came back. 24 M samples is 2x on a full screen (about 70 MB of
# RGB while the render runs) and 3x on a smaller one.
SUPERSAMPLE_SAMPLES = 24_000_000


def _render_job(session, job, box) -> None:
    """Render on a worker and leave the answer in `box`. Touches no Tk.

    Module level and not a method, so a thread that outlives its window holds
    the session and a dict rather than the window itself. See the call site:
    a bound method here was a crash on close.
    """
    MergeApp._stand_back()
    token, page_index, mode, scale, matched, window = job[:6]
    over = job[7] if len(job) > 7 else 1
    drawn = None
    for attempt, use_match in ((1, matched), (2, None)):
        try:
            drawn = sharp_render(session, page_index, mode, scale,
                                 use_match, window, over)
            break
        except Exception as exc:
            print('[merge] the sharpen render raised %s: %s%s'
                  % (type(exc).__name__, exc,
                     '' if attempt == 1 else ' -- the rough picture stays'),
                  flush=True)
    box['result'] = (job, drawn)


def sharp_render(session, page_index, mode, scale, matched, window, over):
    """**The one definition of what belongs on the screen.**

    The strokes drawn `over` times over and shrunk back by the plain mean of
    the samples -- which is what pixel coverage is, and therefore what an
    antialiased edge is. `MergeApp._supersample` says why this exists at all:
    the rasteriser has no intermediate pixel of its own, so without this the
    XOPP side reaches the screen in two colours.

    The worker calls it and so do the tests, because a display pipeline that
    the tests do not share is a display pipeline whose output nothing checks.
    BOX and not LANCZOS: a sharpening kernel invents edge values the strokes
    never had, and in this view a colour is a decision about which side owns
    the ink.
    """
    if over <= 1:
        return views.render(session, page_index, mode, scale=scale,
                            matched=matched, window=window)
    asked = tuple(int(round(v * over)) for v in window)
    drawn = views.render(session, page_index, mode, scale=scale * over,
                         matched=matched, window=asked)
    if drawn is None:
        return None
    return drawn.resize((int(window[2]), int(window[3])), Image.BOX)

# **What a failed sharpen does.** Two attempts already happen inside the
# worker (the second drops `matched`, the one argument that can be stale after
# a decision); these are what happens when those two both raised. 300 / 900 /
# 2700 ms: a transient allocation failure or a lock held by a pan is over by
# then. After three the blurred stand-in is left up with a line in the log --
# the honest answer, and no longer silence.
SHARPEN_RETRIES = 3
SHARPEN_RETRY_DELAY = 300        # ms, multiplied by three on each retry

#: Above this zoom the stand-in frame interpolates instead of repeating cells.
GENTLE_STRETCH = 4.0


def _stand_in_resample(zoom: float):
    """The filter for the frame shown *while* the sharp render is being made.

    **NEAREST only while it is a gentle stretch.** It was chosen because it lands closer to the
    drawn stroke than a smoothing filter does at 2-4x -- measured, 0.66 grey levels against 0.83 at
    4x -- and that is still true there. At 15x it is a different picture: every source pixel becomes
    a 15-pixel square, so the user's PDF's lines came out as staircases of hard blocks with `DPI: 800` and
    `4292 dpi on screen` in the status line. Beyond `GENTLE_STRETCH` the stand-in interpolates: it
    cannot invent detail, but it is blurry rather than stepped, and the sharp render replaces it
    either way.
    """
    from PIL import Image

    if zoom <= GENTLE_STRETCH:
        return Image.Resampling.NEAREST
    return Image.Resampling.BICUBIC


def _covers(origin, size, want):
    """Is the rectangle `want` inside the piece already drawn?"""
    if origin is None:
        return False
    ox, oy = origin
    wx, wy, ww, wh = want
    return (ox <= wx and oy <= wy
            and ox + size[0] >= wx + ww and oy + size[1] >= wy + wh)


# The page's own edge, as a GUI aid and nothing else: one pixel of silver on
# the paper's real border, so a white page on a near-white background has a
# visible end. Silver, because the ink already means four things -- black,
# red, blue, green -- and the Qbase template draws its own blue frame *inside*
# the page, which this must never be mistaken for. It is drawn on the way to
# the screen only: not into the strokes, the mask, the matching or the .xopp.
SILVER = (192, 192, 192)
EDGE_TAG = 'page-edge'
# The Q base reference, as a guide. One grey **dashed** line where the printed
# border sits -- the user asked for it back, and for the blue border itself never to
# be shown. A shade darker than the paper's edge, because they mean different
# things: the edge is where the paper stops, this is where the template's
# writing area is. The geometry is shared with the converter (`widgets.guide`).
from xp.render.widgets.guide import GUIDE, GUIDE_DASH, guide_segments
from xp.render.widgets.progress import Progress

FRAME_TAG = 'page-frame-guide'


def _on_the_pictures_grid(absolute, picture_start):
    """Where something lands in the picture, on the grid the picture itself is placed on.

    The picture is placed at `picture_start` and Tk rounds that to a whole pixel; a feature drawn
    *inside* the picture has to be rounded the same way, or the two roundings disagree by one and the
    guide slides against the ink as the zoom changes. Subtracting the two rounded numbers leaves exactly
    one rounding -- the absolute position's -- so the feature's final place on the screen is the same
    number whatever the zoom is.
    """
    return int(round(absolute)) - int(round(picture_start))


def _guide_dashes(box, dash=GUIDE_DASH, low=None, high=None):
    """The dashes of a rectangle in **whole pixels**, walked from its own corner.

    `guide_segments` yields the same rectangle in the box's own units, which is right for a box measured
    in page units; this is the version for the crop's pixels, where a fractional coordinate makes
    `draw.line` round each end on its own and turns one dash into five pixels and the next into seven.
    Walking in integers from the corner keeps every dash exactly `on` long and every gap exactly `off`
    -- at any zoom, wherever the crop begins.

    `low`/`high` bound the walk to the part of the image that exists, so a page zoomed to 8x does not
    spend its time on dashes outside the picture.
    """
    x0, y0, x1, y1 = (int(round(value)) for value in box)
    on, off = dash
    step = max(1, on + off)
    segments = []
    for vertical in (False, True):
        start, end = (y0, y1) if vertical else (x0, x1)
        near, far = (x0, x1) if vertical else (y0, y1)
        if end <= start:
            continue
        first = start
        if low is not None:
            # Start at the first dash that can land in the picture, keeping the corner's phase.
            first = start + max(0, (low - start) // step) * step
        last = end
        if high is not None:
            last = min(end, high + step)
        at = first
        while at < last:
            # `draw.line` paints both endpoints, so a dash of `on` pixels is a segment of `on - 1`.
            stop = min(at + on - 1, end)
            if vertical:
                segments.append((near, at, near, stop))
                segments.append((far, at, far, stop))
            else:
                segments.append((at, near, stop, near))
                segments.append((at, far, stop, far))
            at += step
    return segments


def the_guide_dash(zoom):
    """The dash at this zoom, in whole pixels: the guide's own dash, measured on the page.

    `GUIDE_DASH` is what it is at zoom 1, and zooming multiplies it -- a guide drawn on the page grows
    with the page instead of keeping a screen-sized pattern whose phase slides as the zoom changes.
    """
    on, off = GUIDE_DASH
    # A floor, so a page zoomed far out still reads as a dashed line rather than a dotted one: at 0.15
    # the proportional dash is 0.9 on and 0.6 off, which rounds to 1 and 1.
    return (max(2, int(round(on * zoom))), max(1, int(round(off * zoom))))


def page_frame_guide(img, box, colour=GUIDE, width=1, dash=None):
    """The same dashes painted into the pixels of a shown crop, in whole pixels."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in _guide_dashes(box, dash=dash or GUIDE_DASH,
                                       low=0, high=max(img.size)):
        draw.line((x0, y0, x1, y1), fill=colour, width=width)
    return img


def _edge_pixel(edge, origin, extent):
    """Which pixel of a picture holds the page edge, or None if it is outside.

    `edge - origin` landing exactly on `extent` means the page's edge *is* the
    picture's far border -- the picture ends where the page ends -- so the last
    pixel of the picture is the one to mark. One pixel further out and the
    edge is genuinely not in this picture and nothing is drawn; that
    distinction is what keeps the line on the page's edge instead of on the
    border of every crop.
    """
    at = int(round(edge - origin))
    if at == extent:
        return extent - 1 if extent else None
    if 0 <= at < extent:
        return at
    return None


def page_edges(img, at_x, at_y, page_w, page_h, colour=SILVER):
    """Draw the page's edge into a picture of part of it, one pixel wide.

    `at_x`/`at_y` is where this picture's top-left corner sits in the page's
    canvas coordinates, so the four edges are found by mapping the page's
    rectangle through them. That is what makes the line land on the real edge
    at any zoom and after any pan, instead of on the border of the picture --
    which is the same thing only when the page exactly fills the window.
    """
    # `paste`, not `ImageDraw`: a Draw object and two coordinate lists per
    # frame is allocation in the one path that runs on every brush segment, and
    # it showed up as a *tail* -- `test_brush_is_local` measured a P95 of 65.8
    # ms on a 10k page with the drawing and 6.0 ms with it taken out, while the
    # average cost of either was about a millisecond. Four pastes are four
    # calls into C and make nothing for the collector to look at.
    width, height = img.size
    for x in (0, page_w):
        column = _edge_pixel(x, at_x, width)
        if column is not None:
            img.paste(colour, (column, 0, column + 1, height))
    for y in (0, page_h):
        row = _edge_pixel(y, at_y, height)
        if row is not None:
            img.paste(colour, (0, row, width, row + 1))
    return img


# **24, not 8.** The user asked to go about three times deeper to look at detail, and nothing in the camera
# path pays for the number: `_rough_at_this_zoom` renders only the visible rectangle (an affine
# transform into the window's own size), and a sharp render is a window of the page, capped by
# `MAX_WINDOW_PIXELS` whatever the zoom asks. What 24 does cost is the *source*: past its own
# resolution an embedded picture or a PDF page is magnified, which is physics and not a setting.
MIN_ZOOM, MAX_ZOOM = 0.2, 24.0
REDRAW_INTERVAL_MS = 16  # ~60fps cap; coalesces bursts of motion events
LOOP_CLOSE_FACTOR = 3.0  # drag end within (radius * this) of its start closes the loop
# Alt+1..Alt+9. Roughly geometric rather than evenly spaced: the small end
# is where the work is (a traced pen stroke is only 1-3px wide on screen, so
# anything coarser cannot pick out one letter of a word), and the large end
# is for clearing whole regions, where the difference between 40 and 48px
# was never worth a key of its own.
BRUSH_SIZE_PRESETS = [1, 2, 3, 5, 8, 14, 24, 42, 72]
# What a shifted digit actually reports as its keysym, in Shift+1..9 order.
# On Windows, Shift+1 is `exclam`, not `1`: a binding written as
# `<Shift-KeyPress-1>` can look right and never fire there.
SHIFTED_DIGIT_KEYSYMS = ('exclam', 'at', 'numbersign', 'dollar', 'percent',
                         'asciicircum', 'ampersand', 'asterisk', 'parenleft')
DEFAULT_BRUSH_SIZE_INDEX = 3  # 5px: fine enough to aim, wide enough to sweep
BRUSH_TINT_ALPHA = 0.30       # the paint is 30% opaque over the page
RATIO_STEP = 1.02             # one press of ratio +/-: 2%, fine enough to land on
OFFSET_STEP = 2.0             # page units per press of the offset buttons
REPEAT_DELAY_MS = 350         # hold Ctrl+Z/Y this long before it starts repeating
REPEAT_INTERVAL_MS = 60       # ...then one step this often


# **The key table the Hotkeys button shows.** Read off this window's own
# `_bind_hotkeys`, not copied from the converter's: the user's instruction was that
# the merger's keys must be taken from the merger's code. Each row carries
# the Tk sequence it really is, and the test asks Tk whether it is bound.
MERGE_HOTKEYS = (
    ('Pages', (
        ('Previous page', '← / PageUp', '<Left>'),
        ('Next page', '→ / PageDown', '<Right>'),
        ('Move this page back', 'Ctrl+PageUp', '<Control-Prior>'),
        ('Move this page on', 'Ctrl+PageDown', '<Control-Next>'),
        ('The page map', 'M', 'm'),
    )),
    ('What you are looking at', (
        ('The left document', 'Ctrl+1', '<Control-Key-1>'),
        ('The right document', 'Ctrl+2', '<Control-Key-2>'),
        ('Both, overlaid', 'Ctrl+3', '<Control-Key-3>'),
        ('The result', 'P', 'p'),
        ('Back to editing', 'Escape', '<Escape>'),
    )),
    ('The brush', (
        ('Take the left ink', '1', '<KeyPress-1>'),
        ('Take the right ink', '2', '<KeyPress-2>'),
        ('Take both', '3', '<KeyPress-3>'),
        ('Take neither', '4', '<KeyPress-4>'),
        ('Brush size', 'Alt+1 ... Alt+9', '<Alt-KeyPress-1>'),
        ('Start this page again', 'R', 'r'),
    )),
    ('The page itself', (
        ('Fit it to the sheet', 'F', 'f'),
        ('Split it in two', 'Ctrl+4', '<Control-Key-4>'),
    )),
    ('The view', (
        ('Zoom in', 'Ctrl++', '<plus>'),
        ('Zoom out', 'Ctrl+-', '<minus>'),
    )),
    ('Undo and saving', (
        ('Undo', 'Ctrl+Z', '<Control-z>'),
        ('Redo', 'Ctrl+Y', '<Control-y>'),
        ('Save and close', 'Ctrl+S', '<Control-s>'),
        ('Save as... and close', 'Ctrl+Shift+S', '<Control-S>'),
        ('The guide', 'F1', '<F1>'),
    )),
    ('The window', (
        ('Quit XP', 'Alt+Q', '<Alt-q>'),
    )),
)


class PaperProgress(Progress):
    """`Progress`, painted for paper: the merger's panel, and nothing else's.

    The base class paints the dark ground of every XP window and the arranger uses it as it stands, so the
    paper version lives here rather than as a change to the shared widget. It recolours what the base
    class builds, re-colours the three states it sets, and paints its own bar in its own accent.
    """

    GROUND = '#ffffff'
    EDGE = '#c4cfd8'
    INK = ABYSS
    MUTED_INK = '#54677a'
    TRACK = '#e6ebef'
    FAILED = '#b3261e'

    def __init__(self, parent, width: int = 430):
        super().__init__(parent, width=width)
        self._recolour()

    def _recolour(self) -> None:
        self.panel.configure(bg=self.GROUND, highlightbackground=self.EDGE)
        for widget in (self.title, self.bar, self.detail, self.more, self.more_button,
                       self.close_button):
            try:
                widget.configure(bg=self.GROUND)
            except Exception:
                continue
        self.title.configure(fg=self.INK)
        self.detail.configure(fg=self.MUTED_INK)
        self.more.configure(fg=self.MUTED_INK)
        for button in (self.more_button, self.close_button):
            button.configure(fg=self.MUTED_INK, activebackground=self.EDGE,
                             activeforeground=self.GROUND)
        self._paint_bar()

    def show(self, title, maximum=None, detail='starting...', details=''):
        super().show(title, maximum, detail, details)
        self._recolour()

    def step(self, done, total, detail=''):
        super().step(done, total, detail)
        self.detail.configure(fg=self.MUTED_INK)

    def pulse(self, detail):
        super().pulse(detail)
        self.detail.configure(fg=self.MUTED_INK)

    def failed(self, title, message, details=''):
        super().failed(title, message, details)
        self.title.configure(fg=self.FAILED)
        self.detail.configure(fg=self.FAILED)
        self.more.configure(fg=self.FAILED)
        for button in (self.more_button, self.close_button):
            button.configure(fg=self.FAILED, activebackground=self.EDGE,
                             activeforeground=self.GROUND)

    def _paint_bar(self) -> None:
        """The base class's bar, on paper: a pale track and an ink share."""
        import tkinter as tk

        try:
            width = max(1, self.bar.winfo_width())
        except tk.TclError:
            return
        self.bar.configure(bg=self.GROUND)
        self.bar.delete('all')
        middle = 3
        self.bar.create_rectangle(0, middle - 1, width, middle + 2, fill=self.TRACK, outline='')
        if self._maximum is None:
            span = max(24, int(width * 0.18))
            start = int((width - span) * self._pulse)
            self.bar.create_rectangle(start, middle - 1, start + span, middle + 2,
                                      fill=self.INK, outline='')
            return
        share = 0.0 if not self._maximum else self._value / float(self._maximum)
        share = max(0.0, min(1.0, share))
        if share:
            self.bar.create_rectangle(0, middle - 1, max(2, int(width * share)), middle + 2,
                                      fill=self.INK, outline='')


class MergeCue:
    """The small loading window while a merge is being opened.

    The user's instruction: "opening the merger should also show the same small loading
    interface as a page turn ... show it as early as possible, rather than a
    window that is unresponsive for a long time". This is that window: it exists
    before the documents are read, it names the file being opened (which is real
    work, not a timer), and the merge window closes it once the first page has
    been drawn -- so it covers opening the sides, building the session and
    preparing the first display, and it never invents a percentage.
    """

    def __init__(self, master=None, title='Opening the merge'):
        self.progress = None
        self.root = None
        self._master = master
        self._title = title
        # **Once closed, it stays closed.** `close()` sets `root` to None, and
        # `_make()` builds a window whenever `root` is None -- so a report
        # that arrived after the close did not report into a closing window,
        # it **built a new one**, lifted it over the merge, and left it there
        # with nothing running that would ever close it. That is the box the user
        # found sitting at 100% on 2026-10-27, and the reason the user's clicks were
        # not reaching the page: "IT ALREDY 100 IT SHODUE OEPON SOON , AN DI
        # SJUET FIND THE BRUSH CNAT WORK NOW". A late report is ignored now.
        self._closed = False

    def _make(self):
        if self._closed:
            return
        if self.root is not None:
            return
        if self._master is None:
            return
        import tkinter as tk

        from xp.render.widgets.progress import Progress

        self.root = tk.Toplevel(self._master)
        self.root.title('XP')
        self.root.resizable(False, False)
        # **As big as its words, not a fixed box.** 470x150 was measured for one
        # font at 100 %; at 150 % the title and the detail line need more room
        # than the box has, which is the "the loading is too large / does not
        # match" the user saw. Tk is asked for the size the widgets want, and the
        # window is centred over the window it belongs to.
        self.root.geometry('')
        try:
            self.root.transient(self._master)
        except tk.TclError:
            pass
        # **The cue is the first thing the user sees while a merge opens**, and it imports the paper palette
        # and did not pass it: 438 x 80 of #101923 over the desktop, the same block in the same place.
        self.progress = PaperProgress(self.root, width=400)
        self.progress.show(self._title, None, detail='starting...')
        self.root.update_idletasks()
        want_w = max(360, self.root.winfo_reqwidth())
        want_h = self.root.winfo_reqheight()
        try:
            parent = self._master.winfo_rootx(), self._master.winfo_rooty()
            parent_w = self._master.winfo_width()
            left = parent[0] + max(0, (parent_w - want_w) // 2)
            top = parent[1] + 90
        except Exception:
            left, top = 120, 120
        self.root.geometry('%dx%d+%d+%d' % (want_w, want_h, left, top))
        self.root.deiconify()
        self.root.lift()

    def _fit_its_words(self) -> None:
        """Grow the box to whatever the words need, **after** they arrive.

        It was measured once, at 'starting...', and then pinned by `geometry(...)`. Every later line
        is longer than that one -- "Loading page 2 of 2", "composing the page less..." -- so the text
        was clipped at the top and at the right, which is the box in the user's screenshot: a loading window
        whose own words do not fit. Never shrinks, so it does not jump about while it works, and stays
        centred over the window it belongs to.
        """
        if self.root is None:
            return
        try:
            self.root.update_idletasks()
            # **Ask the panel, not the toplevel.** The panel is `place`d, so the toplevel's requested
            # size does not include it: measuring the window gave 360x200 while the panel inside was
            # larger, and the words were clipped by the frame -- the top of the title cut off and the
            # detail line running past the right edge, which is the box in the user's screenshot.
            panel = getattr(self.progress, 'panel', None)
            if panel is not None:
                panel.update_idletasks()
                need_w = max(360, panel.winfo_reqwidth() + 2)
                need_h = max(120, panel.winfo_reqheight() + 2)
            else:
                need_w = max(360, self.root.winfo_reqwidth())
                need_h = self.root.winfo_reqheight()
            now_w, now_h = self.root.winfo_width(), self.root.winfo_height()
            if need_w <= now_w and need_h <= now_h:
                return
            want_w, want_h = max(need_w, now_w), max(need_h, now_h)
            try:
                parent_x = self._master.winfo_rootx()
                parent_w = self._master.winfo_width()
                left = parent_x + max(0, (parent_w - want_w) // 2)
                top = self.root.winfo_y()
            except Exception:
                left, top = self.root.winfo_x(), self.root.winfo_y()
            self.root.geometry('%dx%d+%d+%d' % (want_w, want_h, left, top))
        except Exception:
            pass

    def pulse(self, detail: str) -> None:
        self._make()
        if self.progress is None:
            return
        self.progress.pulse(detail)
        self.root.update_idletasks()
        self._fit_its_words()

    def step(self, done: int, total: int, detail: str = '') -> None:
        self._make()
        if self.progress is None:
            return
        self.progress.step(done, total, detail)
        self.root.update_idletasks()
        self._fit_its_words()

    def failed(self, title: str, message: str) -> None:
        self._make()
        if self.progress is None:
            return
        self.progress.failed(title, message)

    def close(self) -> None:
        self._closed = True
        # **Every callback this change scheduled is cancelled first.** A worker finishes after the window has gone and
        # its collector then runs against a dead Tk: `RuntimeError: main thread is not in main loop`, measured on the
        # whole of `test_brush_colour_convention` once the sharpen moved off the UI thread.
        for name in ('_collect_job', '_sharpen_job', '_repeat_job'):
            job = getattr(self, name, None)
            if job is None:
                continue
            try:
                self.root.after_cancel(job)
            except Exception:
                pass
            setattr(self, name, None)
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None
            self.progress = None


def _build_from_dotgit(start: str) -> str:
    """The short hash straight out of `.git`, so the title does not depend on running git.

    The user launches through the menu, which is `pythonw`, and two of the user's windows printed
    `[merge] build unknown` (2026-10-31) while `git rev-parse` answers perfectly from a pythonw
    process on this machine -- the cause of those two was never found, and a title that says
    "unknown" is the one thing the hash was added to prevent. The answer is in two files; reading
    them cannot fail for whatever reason the subprocess did.
    """
    import pathlib

    here = pathlib.Path(start)
    for folder in (here, *here.parents):
        dot = folder / '.git'
        if not dot.exists():
            continue
        try:
            base = dot
            if dot.is_file():                       # a worktree: "gitdir: <path>"
                text = dot.read_text(encoding='utf-8', errors='replace').strip()
                if not text.startswith('gitdir:'):
                    return ''
                base = (folder / text.split(':', 1)[1].strip()).resolve()
            head = (base / 'HEAD').read_text(encoding='utf-8', errors='replace').strip()
            if not head.startswith('ref:'):
                return head[:7]                     # detached: HEAD is the hash
            ref = head.split(':', 1)[1].strip()
            loose = base / ref
            if loose.exists():
                return loose.read_text(encoding='utf-8', errors='replace').strip()[:7]
            packed = base / 'packed-refs'
            if packed.exists():
                for line in packed.read_text(encoding='utf-8',
                                             errors='replace').splitlines():
                    if line.startswith(('#', '^')):
                        continue
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == ref:
                        return parts[0][:7]
            return ''
        except OSError:
            return ''
    return ''


def _the_build() -> str:
    """The checkout's short hash, read once, for the title and the startup line.

    Two rounds were spent on reports from a window launched before the fixes they were about, so the build is put where
    the user can see it. Empty when neither `.git` nor git can answer -- a copied tree with no history.
    """
    kept = globals().get('_BUILD')
    if kept is not None:
        return kept
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    # **The files first.** No subprocess, no PATH, no window handle to inherit.
    answer = _build_from_dotgit(root)
    if not answer:
        try:
            import subprocess
            answer = subprocess.run(['git', '-C', root, 'rev-parse', '--short', 'HEAD'],
                                    capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            answer = ''
    globals()['_BUILD'] = answer
    return answer


class MergeApp:
    def __init__(self, session: MergeSession, read_only: bool = False,
                 on_commit: Optional[Callable[[], None]] = None,
                 master: Optional[tk.Misc] = None,
                 on_swap: Optional[Callable[[], None]] = None,
                 on_finish: Optional[Callable[[bool], None]] = None,
                 output: Optional[str] = None,
                 cue: Optional['MergeCue'] = None):
        """`master`: an existing Tk widget to open this as a Toplevel under
        (e.g. from app.py's dashboard, which owns the one real Tk() root and
        mainloop). Leave as None for standalone use (`python -m xp
        diff/merge ...`), which creates and owns its own Tk() root/mainloop.

        `on_swap`: open this merge the other way round, if whoever launched the
        window can. The window cannot swap in place -- the session holds a
        parsed left and a parsed right, and the merged page's size is the
        left's -- so it hands the job back and closes. That is the old
        *Choose sides* window's **Swap**, moved into the window that does the
        work instead of a form that only chose.

        `on_finish(clean)`: tell the launcher this window is done, once, when it
        is saved or when the user closes it. `clean` is False only for a save that
        raised, which is the one case where the launcher must stay behind with
        the reason.

        `output`: where this merge goes, when the launcher already knows
        (`xp merge a b -o out.xopp`). **The launcher names the file and this
        window does the writing**, which is the opposite of how it was: the
        launcher used to hand over a closure that wrote, and a closure cannot
        be run on a worker thread -- the ones in `cli.py` and `app.py` print
        to the console, and `App._print` touches Tk. So that save ran on the
        Tk thread with the window frozen and no panel at all, which is
        exactly what Ctrl+S looked like from the chair: nothing happened.
        With a path instead, every save in this window goes the one way --
        `_write_in_the_background`, with the panel on screen -- and
        `on_commit` is called afterwards, on the Tk thread, purely to report.
        """
        self.session = session
        self.read_only = read_only
        self.on_commit = on_commit
        self.on_swap = None if read_only else on_swap
        self.on_finish = on_finish
        # Where the merge goes, once it is known. None means nobody has said
        # yet -- and that is the normal case when this window was opened by
        # hand rather than by `xp merge ... -o`. It is filled in when the user
        # commits and the dialog answers, and it stays None if the user cancels.
        self.output: Optional[str] = output
        # Close once the file is on disk. Both save keys do it; the arranger
        # does the same. Tests that need the window afterwards set it False.
        self.exit_after_save = True
        self.scale = session.scale
        self._owns_root = master is None

        self.page_index = 0
        # Which view is on screen. Editing is the diagnostic one you brush
        # in; the others are read-only previews of the sources and of the
        # result, each rebuilt from scratch (see views.py).
        self.view_mode = views.EDIT
        self.active_color = TAKE_RIGHT
        self.zoom = 1.0
        self.brush_radius = BRUSH_SIZE_PRESETS[DEFAULT_BRUSH_SIZE_INDEX]
        self.mask: Optional[BrushMask] = None
        self._dragging = False
        self._last_drag_end = time.monotonic()   # the sharpen waits on this
        self._drag_last: Optional[Tuple[float, float]] = None
        self._drag_path: List[Tuple[float, float]] = []
        # **A surface that did not present is left out of the next frame.** A GL surface can show
        # its clear colour -- the window's dark ground -- where the page is, without raising: a
        # locked screen or a dropped remote session does exactly that. `_the_surface_presented`
        # reads one pixel to find out; this is what it sets when the answer is no.
        self._surface_lost = False
        # **Ask the front buffer only when a frame is new.** Reading it is a synchronous stall and a
        # redraw happens on every brush motion; a page load or a zoom is when the surface is
        # re-presented and when a black frame could appear, and that is what `_the_surface_presented`
        # is for. Set here and by `_set_zoom`/`_load_page`, cleared once the frame has been checked.
        self._probe_the_surface = True
        self._photo = None  # keep a reference so PhotoImage isn't GC'd
        # The one canvas item that shows `_photo`, kept so a zoom re-points it
        # instead of deleting the canvas and building a new picture. Measured
        # before this: creating both per frame was the whole cost of a zoom.
        self._photo_item = None
        # The piece of the page that has been drawn, what zoom it was drawn
        # at, and where its top-left sits in the zoomed page. Only the part
        # the user can see is ever drawn; see `_display_image`.
        self._zoomed = None
        self._zoomed_at = None
        self._zoomed_box = None
        # Whether `_zoomed` is the stretched stand-in rather than the drawn
        # page, the zoom it belongs to, and the render still coming for it.
        self._zoomed_is_rough = False
        self._zoom_token = 0
        self._sharpen_job = None
        # One render at a time, and a lock so the worker's render and a pan's synchronous render cannot
        # interleave over the session's caches.
        self._sharp_worker = None
        self._sharp_wanted = None
        self._render_lock = threading.Lock()
        self._sharp_result = None
        self._collect_job = None
        self._base_image: Optional[Image.Image] = None
        self._redraw_pending = False
        self._repeat_job = None

        self.root = tk.Tk() if master is None else tk.Toplevel(master)
        from xp.runtime.shutdown import arm
        arm(self.root)
        from xp.runtime.shutdown import fill_the_screen
        fill_the_screen(self.root)
        # **The build, in the title.** XP (merge) - 1a2b3c4.
        self.root.title("XP" + (" (diff, read-only)" if read_only else " (merge)")
                        + ((" \u00b7 %s" % _the_build()) if _the_build() else ""))
        # **The same window as the other two.** The user's report, 2026-10-30: the
        # merger's GUI should look like the converter's and the arranger's --
        # dark ground, the panel's blue, Times New Roman. It never called
        # `decorate`: it took `ABYSS` for the canvas and left everything else
        # in the system face on the system grey, so the toolbar was a light
        # band across a dark window. One call, the same one the other two
        # make, and it also brings the XP mark on the title bar.
        decorate(self.root)
        # Closing the window is finishing with it, and the launcher has to hear
        # about it -- otherwise a launcher whose task was only ever *looked at*
        # waits for a window that is gone. Bound before anything else can fail,
        # so the protocol is in place for every way the window can end.
        self.root.protocol('WM_DELETE_WINDOW', self._closed_by_user)
        # A render queued for a zoom outlives a window that is destroyed some
        # other way than through the close button -- a parent going away, or a
        # test tearing the root down -- and Tcl then prints `invalid command
        # name` for a callback whose function has been collected. Nothing is
        # broken by it, but it is noise from a job with no page left to draw
        # on, so the window drops it on the way out.
        self.root.bind('<Destroy>', self._dropped_pending_work, add='+')

        # **Checked by default**, the user's order: a merge worth keeping is a merge
        # worth learning from, and the option must not need finding. See
        # `xp/workers/merger/archive.py` for what a sample holds. Made before
        # the toolbar, because the toolbar holds its tick.
        self.save_learning = tk.BooleanVar(self.root, True)
        self._build_toolbar()
        self._say_the_resolution()

        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        # `highlightthickness=0`, because a focused canvas draws Tk's focus
        # ring **inside its own edges** -- 2px here, in `SystemWindowFrame`,
        # which is a dark line along the top of the page area that does not
        # belong to the page, does not move with it, and looks like the view
        # failing to clip. It is the focus ring: the ring is the only thing
        # that changes when the canvas takes focus, and keys work without it
        # (they are bound on the toplevel). The white background is the paper
        # colour the page is composed on; the page's own edge is drawn in
        # silver, so the canvas no longer has to pretend the page fills it.
        self.canvas = tk.Canvas(canvas_frame, bg=BACKGROUND_COLOR,
                                highlightthickness=0)
        # The scrollbars scroll the canvas directly, which moves what is drawn
        # but asks for nothing. With the document presented as a canvas item
        # (see `tkgl`), the item would then scroll away with the page instead of
        # showing a different part of it, so both scrollbar commands go through
        # here and ask for the redraw they imply.
        vbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self._scroll_vertical)
        hbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self._scroll_horizontal)
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        # Keyboard shortcuts are bound on this window, and key events only
        # reach it if something inside it holds focus. Opened from the
        # dashboard, focus otherwise stays in the console's entry box, and
        # every shortcut here silently does nothing.
        self.canvas.configure(takefocus=True)
        from xp.render.widgets.progress import Progress
        # **Paper, not the window ground.** The panel sits over the page, and a dark panel on a white
        # sheet is a hole in it -- the user called it a black block twice.
        self.progress = PaperProgress(self.canvas)
        self._saving = False
        self._save_job = None
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        self.canvas.bind("<Button-1>", lambda e: self.canvas.focus_set(), add="+")
        self.root.after(100, self._take_focus)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        # Middle-mouse-drag pans the view (left button stays the brush).
        self.canvas.bind("<ButtonPress-2>", self._scan_mark)
        self.canvas.bind("<B2-Motion>", self._pan_and_redraw)
        # Wheel = scroll; Shift+wheel = horizontal scroll; Ctrl+wheel = zoom.
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self._scroll_sideways)
        self.canvas.bind("<Control-MouseWheel>", self._on_wheel_zoom)
        # A resized window is a different rectangle of the page, and the
        # surface's widget is a canvas item -- it is as big as it was last
        # placed, so without this the document keeps its old size until
        # something else happens to redraw. The redraw is coalesced, so a drag
        # of the window's corner does not compose the page per pixel event.
        # **The canvas only has a size after it is laid out**, so the opening fit is repeated on the
        # first resizes until it has really been done. Measured: asked too early, the canvas answers
        # 1x1, the fit comes out as the minimum zoom (0.2), and the page opens as a sliver in a corner
        # -- "why the merge open with the page not in the center".
        self._needs_the_whole_page = True

        def resized(_event=None):
            if self._needs_the_whole_page:
                self._show_the_whole_page()
                if not self._needs_the_whole_page:
                    self._redraw()
            self._request_redraw()

        self.canvas.bind("<Configure>", resized)

        self._bind_hotkeys()
        # The opening is where a big pair actually costs: on the sample files the
        # first page needs 2.9 s from window to picture. The panel is up before
        # that starts and names the two real steps it is waiting for.
        self._cue = cue
        if self._cue is not None:
            self._cue.step(1, 3, 'reading the first page')
        self._load_page(0)
        # **The cue stays up until the page is actually drawn.** It used to close after
        # `_load_page`, before the redraw -- and the redraw is the expensive part: the window then sat
        # blank for the whole first composition with no loading on it, which is what the user reported as
        # "why cant i see loading when the merge is taking time".
        if self._cue is not None:
            self._cue.step(2, 3, 'placing the page')
            self._cue.root.update_idletasks()
        self._show_the_whole_page()
        try:
            self._redraw()
        except Exception:
            pass
        if self._cue is not None:
            self._cue.step(3, 3, 'the first page is up')
            self._cue.close()
            self._cue = None
        # **Nothing is converted here, and that is the whole point.** This used
        # to schedule a background pass that traced every embedded picture into
        # strokes before the user had looked at a page -- a minute of work the user did not
        # ask for, spent on content the user might refuse. The user's specification: during
        # editing a picture in a XOPP is simply part of that side's image, and
        # only the part finally selected is converted, at export. The page is
        # drawn with its pictures (`drawing.draw_pictures`), compared as an
        # image, and `export` is the one place that turns them into strokes.

        # Tk doesn't wrap a row of buttons, it cuts it off -- so don't let the
        # window get narrower than the toolbar needs. The **rows** were made
        # short enough to fit (`_build_toolbar`) and the three labels whose
        # text comes from the document wrap, which is what stops text and
        # buttons being cut off without asking for a wider and wider minimum.
        #
        # The height stays a constant, and deliberately. Taking it from the
        # content was tried: `minsize` moves the window, moving the window
        # changes the canvas, and a changed canvas is a redrawn page -- so
        # panning inside the margin stopped being free, which is exactly what
        # `tests/test_merge_zoom` was written to hold. A window that resizes
        # itself to fit its own repaint is a window that repaints for ever.
        self.root.update_idletasks()
        self.root.minsize(self.root.winfo_reqwidth(), 360)

    def _build_toolbar(self) -> None:
        # Two rows up top -- brushes, then whole-page actions -- and the key
        # to colors/shortcuts in a bar along the bottom that wraps with the
        # window. One row held everything and the right end simply fell off
        # the screen, Save button included.
        tools = ttk.Frame(self.root)
        tools.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

        self.page_label = ttk.Label(tools, text="", width=12)
        self.page_label.pack(side=tk.LEFT, padx=6)
        ttk.Button(tools, text="< Prev", command=self.prev_page).pack(side=tk.LEFT)
        ttk.Button(tools, text="Next >", command=self.next_page).pack(side=tk.LEFT)

        if not self.read_only:
            ttk.Button(tools, text="Save & close (Ctrl+S)",
                       command=self.commit).pack(side=tk.RIGHT, padx=6)
            ttk.Button(tools, text="Save as... (Ctrl+Shift+S)",
                       command=self.commit_as).pack(side=tk.RIGHT)
            # **Checked by default, next to the two save buttons.** The user's order:
            # keep Ctrl+S and Ctrl+Shift+S, and keep a *Save as Learning
            # Material* option that is on unless the user turns it off. It sits on the
            # window rather than inside the save dialog because the system's
            # save dialog cannot hold a tick, and a tick the user can see all the time
            # is one the user can decide about before the user presses either key.
            ttk.Checkbutton(tools, text="Learning material",
                            variable=self.save_learning).pack(
                side=tk.RIGHT, padx=(0, 8))

            ttk.Separator(tools, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=2)
            ttk.Label(tools, text="Brush:").pack(side=tk.LEFT)
            self.brush_var = tk.StringVar(value=TAKE_RIGHT)
            for text, value in (("Left (1)", TAKE_LEFT), ("Right (2)", TAKE_RIGHT),
                                ("Both (3)", TAKE_BOTH), ("Neither (4)", TAKE_NONE)):
                ttk.Radiobutton(tools, text=text, variable=self.brush_var, value=value,
                                command=self._on_brush_change).pack(side=tk.LEFT, padx=(4, 0))
            self.size_label = ttk.Label(tools, text=f"size {self.brush_radius}px")
            self.size_label.pack(side=tk.LEFT, padx=(10, 0))

            ttk.Separator(tools, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=2)
            ttk.Button(tools, text="Undo", command=self._undo).pack(side=tk.LEFT)
            ttk.Button(tools, text="Redo", command=self._redo).pack(side=tk.LEFT)
            # **Top right, where the user looked for it.** It was at the end of the
            # left-hand run of buttons and the user never saw it. It takes no focus,
            # so resting on it cannot take the keyboard away from the canvas.
            from xp.render.widgets.hotkeys import HotkeyButton

            self.hotkey_button = HotkeyButton(tools, MERGE_HOTKEYS,
                                              title='Merger')
            self.hotkey_button.pack(side=tk.RIGHT, padx=(10, 4))

            page = ttk.Frame(self.root)
            page.pack(side=tk.TOP, fill=tk.X, pady=(2, 2))
            # **No whole-page buttons.** The user's order, 2026-11-02: *"删除 All Left、All Right、All Both
            # 三个整页选择按钮及其专用操作逻辑 ... Merger 中所有区域状态的修改，都只能通过 Brush 完成。"*
            # They were `_apply_page(TAKE_LEFT/RIGHT/BOTH)` -- a page-wide decision that bypassed the
            # brush, which is the one editor the region map is meant to have. `Clear page` went the
            # same way the day before. The four states are all on the brush: Left, Right, Both,
            # Neither (keys 1-4).

            # The page's own structure, on a row of its own.
            #
            # These five used to share the row above, which made it eleven
            # widgets wide and 1,296 pixels at this screen's scaling -- wider
            # than the window's minimum, so dragging the window to its
            # smallest size put "Reset page" and "Page map" past the right edge
            # and cut them off. A row of buttons does not wrap, so the fix is
            # rows that are short enough to fit, not a wider minimum: a
            # minimum that grows to whatever the widest row happens to be is a
            # window that cannot be made small.
            structure = ttk.Frame(self.root)
            structure.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
            # Splitting makes a page there and then, so it is a button and
            # not a setting: the merge gets longer, and the two halves are
            # ordinary pages you can brush, move and reset.
            self.split_button = ttk.Button(structure, text="Split into 2 pages (Ctrl+4)",
                                           command=self._toggle_split)
            self.split_button.pack(side=tk.LEFT, padx=(6, 0))
            ttk.Button(structure, text="Move up", width=9,
                       command=lambda: self.move_page(-1)).pack(side=tk.LEFT, padx=(6, 0))
            ttk.Button(structure, text="Move down", width=11,
                       command=lambda: self.move_page(1)).pack(side=tk.LEFT)
            self.split_note = ttk.Label(structure, text="", foreground="#666666",
                                        wraplength=WRAP_LONG)
            self.split_note.pack(side=tk.LEFT, padx=(4, 0))
            ttk.Button(structure, text="Reset page (R)",
                       command=self.reset_page).pack(side=tk.LEFT, padx=(14, 0))
            ttk.Button(structure, text="Page map... (M)",
                       command=self.edit_page_map).pack(side=tk.LEFT, padx=(14, 0))
            # The old chooser's Swap, where the work is. Only when a launcher
            # can actually reopen the merge the other way round -- a window
            # started by `xp merge a b` on its own command line has nobody to
            # hand the job back to, and a button that closed the window and did
            # nothing else would be worse than no button.
            if self.on_swap is not None:
                ttk.Button(structure, text="Swap sides",
                           command=self._swap_sides).pack(side=tk.LEFT, padx=(14, 0))
                ttk.Label(structure, text="(left: %s   right: %s)"
                          % (self.session.left_side.path.name
                             if getattr(self.session.left_side, 'path', None) else '?',
                             self.session.right_side.path.name
                             if getattr(self.session.right_side, 'path', None) else '?'),
                          foreground="#666666",
                          wraplength=WRAP_LONG).pack(side=tk.LEFT, padx=(8, 0))

        preview = ttk.Frame(self.root)
        preview.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
        ttk.Label(preview, text="Preview:").pack(side=tk.LEFT, padx=(6, 4))
        self.view_var = tk.StringVar(value=views.EDIT)
        for text, mode in (("Editing (Esc)", views.EDIT),
                           ("Left page (Ctrl+1)", views.SOURCE_LEFT),
                           ("Right page (Ctrl+2)", views.SOURCE_RIGHT),
                           ("Both raw (Ctrl+3)", views.OVERLAY)):
            ttk.Radiobutton(preview, text=text, variable=self.view_var, value=mode,
                            command=lambda m=mode: self.set_view(m)).pack(side=tk.LEFT, padx=(4, 0))
        # The question this answers is "if I export right now, what do I
        # get?" -- so it is a button of its own, not one more radio button
        # in a row of source views.
        ttk.Button(preview, text="Final preview (P)",
                   command=lambda: self.set_view(views.FINAL)).pack(side=tk.LEFT, padx=(14, 0))
        self.view_label = ttk.Label(preview, text="", foreground="#444444",
                                    wraplength=WRAP_LONG)
        self.view_label.pack(side=tk.LEFT, padx=(10, 0))

        # **The resolution the page is identified at, and the user types it.** The
        # three classes are decided once, at this DPI (`classification.py`), and
        # a zoom samples that answer instead of making its own -- so this is the
        # one place the answer's quality is chosen. It is not the zoom: the zoom
        # only says how big the page looks.
        ttk.Separator(preview, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10, pady=2)
        ttk.Label(preview, text="DPI:").pack(side=tk.LEFT)
        self.dpi_var = tk.StringVar(value=str(int(round(self.session.classify_dpi))))
        self.dpi_entry = ttk.Entry(preview, textvariable=self.dpi_var, width=6)
        self.dpi_entry.pack(side=tk.LEFT, padx=(4, 0))
        self.dpi_entry.bind("<Return>", self._apply_dpi)
        self.dpi_entry.bind("<FocusOut>", self._apply_dpi)
        self.dpi_label = ttk.Label(preview, text="", foreground="#666666")
        self.dpi_label.pack(side=tk.LEFT, padx=(6, 0))

        if not self.read_only:
            # **The machine fits it; the user does not.** The ratio and offset
            # buttons are gone: getting the two pages to sit on each other is
            # the program's job, and asking the user to nudge by eye was asking the user
            # to do the arithmetic the frames already answer. What is left is
            # one button that measures the files again.
            fit = ttk.Frame(self.root)
            fit.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
            ttk.Button(fit, text="Fit", width=8, command=self.fit).pack(
                side=tk.LEFT, padx=(6, 6))
            ttk.Button(fit, text="Help", width=8, command=self.show_help).pack(
                side=tk.LEFT, padx=(0, 6))
            ttk.Label(
                fit, foreground="#666666",
                text="the two files are placed by their own Q base frames; "
                     "Fit measures them again for this page pair",
            ).pack(side=tk.LEFT)
            # No widget any more, but `adjust_fit` still honours it and a
            # caller (a script, a test) may still ask for every page.
            self.fit_all = tk.BooleanVar(value=True)

        # **One short line, not a paragraph.** The colour legend and the whole
        # key list used to be a wrapping block along the bottom -- two or three
        # rows of text that took the height the page needed. The colours are
        # what the page is for, so they stay; the keys are in the guide.
        legend_bar = ttk.Frame(self.root)
        legend_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.fit_label = ttk.Label(legend_bar, text="", foreground="#444444")
        self.fit_label.pack(side=tk.RIGHT, padx=(10, 4))
        # **What the merge does not handle, said out loud.** Text, images and
        # layers above the first are carried through from the left document and
        # are not transformed by the fit; a right side's are not brought over.
        # Silence here is how it ends up discovered in the exported file.
        self.limit_label = ttk.Label(legend_bar, text="", foreground="#8a6d3b")
        self.limit_label.pack(side=tk.RIGHT, padx=(10, 4))
        legend = ttk.Label(
            legend_bar, justify=tk.LEFT,
            text="black = both   red = left only   blue = right only"
                 "   green = kept   |   F1 / Help: 使用说明",
        )
        legend.pack(side=tk.LEFT, padx=6, pady=2)
        legend_bar.bind("<Configure>", lambda e: legend.configure(
            wraplength=max(200, e.width - 220)))
        # Now that the label exists, say what this merge cannot handle.
        self.note_what_is_not_merged()

    def edit_page_map(self) -> None:
        """Re-pair the pages without leaving the merge.

        Which left page faces which right page is a question that only comes
        up once you are looking at them, so it can be answered here rather
        than only before the merge starts. Decisions already made stay made;
        what changes is which page each one is compared against.
        """
        if self.read_only or self._saving:
            # The map can re-pair the pages; re-pairing during a save would
            # change what the writer is exporting, page by page.
            return
        from xp.workers.merger.gui.mapgui import ManualAlignment, PageMapApp

        # **No alignment means "find one", not "assume page i with page i".**
        # This used to hand the editor a page-for-page mapping when the session
        # had none, so the first glance at the map showed identity links --
        # which read as "auto match matched nothing", and was the other half of
        # "auto match is unreliable". The editor computes an automatic
        # alignment itself when it is given none, by whatever both sides really
        # have (strokes, or a render for a PDF), which is the pass this was
        # trying to avoid.
        alignment = self.session.alignment
        if alignment is None:
            # **Show the pairing that is in force, not an empty map.** With no
            # alignment the session pairs page i with page i -- that is the
            # default the user asked for -- and the map used to open with no links at
            # all, which reads as "nothing is paired" while the window behind it
            # was showing 1 with 1. The editor still does **not** run a match on
            # open; this is the identity it already has.
            pairs = {}
            for page in self.session._pages:
                if page.left is not None and page.right is not None:
                    pairs[page.left] = page.right
            alignment = ManualAlignment(pairs)
        editor = PageMapApp(self.session.left, self.session.right,
                            alignment=alignment, master=self.root,
                            left_side=self.session.left_side,
                            right_side=self.session.right_side)
        editor.run()
        # **Three outcomes, three meanings.** Cancel: nothing changes.
        # Done with an empty map: the user cleared the links and meant it, so every
        # page stands alone (`ManualAlignment({})` answers *no partner* for
        # every left page). Done with links: that pairing. The editor used to
        # return only the dict, so "I opened this and changed nothing" and
        # "unlink everything" were the same empty dict, and Clear links did
        # nothing at all.
        outcome = getattr(editor, 'outcome', 'mapped' if editor.mapping else 'unchanged')
        # **What the editor hands back is what the user drew.** A close that does not go through the
        # Done button used to leave `outcome` at 'unchanged', so a changed mapping was thrown
        # away -- the merger kept the previous pairing. The editor now treats its close button
        # as Done, and this compares the mapping with the pairing in force as well, so any
        # other path that returns a changed mapping takes effect too.
        handed = dict(editor.mapping)
        in_force = {page.left: page.right for page in self.session._pages
                    if page.left is not None and page.right is not None}
        changing = (outcome == 'cleared'
                    or outcome == 'mapped'
                    or (outcome == 'unchanged' and handed != in_force))
        if not changing:
            self._take_focus()
            return
        # **Re-pairing is slow work, so it says so.** `set_alignment` rebuilds
        # the page list and every pair's placement, and the page that follows
        # is a full load; both ran with the window simply frozen. The user's section 9
        # asks for these two by name: *"Updating page mapping..."* while the
        # pairing is rebuilt, *"Loading paired pages..."* while the page it
        # lands on is drawn.
        self.progress.show('Updating page mapping', 2,
                           detail='re-pairing the pages...')
        self.root.update_idletasks()
        try:
            if outcome == 'cleared':
                self.session.set_alignment(ManualAlignment({}))
            else:
                self.session.set_alignment(ManualAlignment(handed))

            total = self.session.page_count
            self.progress.step(1, 2, 'loading paired pages...')
            self.root.update_idletasks()
            if total:
                # The page the user was on may not be a pair any more, so the view
                # lands on the nearest one that is.
                self._load_page(min(self.page_index, total - 1))
            else:
                # Nothing is paired: there is no page to load, and `_redraw`
                # draws the empty state that says so. Without this the old
                # line asked for page -1.
                self.page_index = 0
                self._redraw()
            self.progress.step(2, 2, 'done')
        except Exception as exc:                    # reported, never swallowed
            self.progress.failed(
                'The new page mapping did not load',
                '%s: %s' % (type(exc).__name__, exc))
            self._take_focus()
            return
        self.progress.hide()
        self._take_focus()

    def adjust_fit(self, ratio: float = 1.0, dx: float = 0.0, dy: float = 0.0) -> None:
        """Nudge where the other document's ink sits, and redraw.

        Nothing about the decisions changes: this is where the ink is, not
        what is kept. The ratio turns about the middle of the page, so it
        does not also walk the writing off the edge.
        """
        if self.read_only:
            return
        side = self.session.moving_side(self.page_index)
        pages = None if self.fit_all.get() else self.page_index
        self.session.adjust_fit(pages, side, ratio=ratio, dx=dx, dy=dy)
        self._reload_placement()

    def note_what_is_not_merged(self) -> None:
        """Say what the merge does not handle, once, in the window and the log."""
        try:
            found = self.session.unhandled()
        except Exception:
            found = []
        if not found:
            self.limit_label.configure(text="")
            return
        parts = []
        for page_index, side, counts in found:
            what = []
            if counts.get('text'):
                what.append('text')
            if counts.get('image'):
                what.append('images')
            if counts.get('other_layers'):
                what.append('extra layers')
            parts.append('page %d %s: %s' % (page_index + 1, side,
                                             ', '.join(what)))
        # **Carried, not compared.** The left side's objects are transformed
        # by the same fit as its ink now; what they are not is compared with
        # the other side or brushable, and a right side's copy is not brought
        # over. The line says that rather than "not merged", which was true
        # before they followed the fit.
        self.limit_label.configure(
            text="not compared: " + '; '.join(parts))
        print('[merge] not compared or brushable (the left side is carried and '
              'fitted with its ink, a right side is dropped): '
              + '; '.join(parts), flush=True)

    def show_help(self) -> None:
        """The Chinese user guide, opened from the window it describes."""
        from xp.render.widgets.help import open_help

        open_help(self.root)

    def _show_the_whole_page(self) -> None:
        """Open on the whole page, fitted to the window -- not at 1:1 from the top-left corner.

        The user's report: "why the merge open with the page not in the center". At `zoom = 1.0` a Q base page
        is taller than the window, so it was drawn from the top-left and the page the user was looking at was
        its corner. Fitting first means the page the user opens on is the page, whole and centred.
        """
        try:
            width = max(1, self.canvas.winfo_width())
            height = max(1, self.canvas.winfo_height())
            page_width, page_height = self.session.page_size(self.page_index)
            if page_width <= 1 or page_height <= 1:
                return
            if width < 60 or height < 60:
                # **Not laid out yet.** A 1x1 canvas gives a fit of 0.005, which the clamp turns into
                # the minimum zoom and -- worse -- counts as done, so the page opened as a sliver and
                # the real fit never happened.
                return
            fits = min(width / page_width, height / page_height)
            if fits > 0:
                self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, fits))
                self._needs_the_whole_page = False
        except Exception:
            pass

    def fit(self) -> None:
        """Measure this page pair's placement again, and show the result at once.

        The user's instruction: one `Fit` button, no ratio and offset knobs, and the
        view refreshes itself when it is done. What it recomputes is the *file*
        calibration -- both sides' frames, one transform each -- and it never
        touches the page pairing: which pages face each other is the user's decision.
        """
        if self.read_only or self._saving:
            # **A save is reading this session right now.** Brushing was already
            # refused during a save (`locked()`); Fit, split, a page move and
            # the page map were not, and any of them clears caches or reorders
            # pages *while the writer walks them*. One rule for every entry
            # point that changes the session.
            return
        total = self.session.page_count
        # **The panel appears before the work starts, and it moves while the work runs.** Two steps with one long
        # blocking call between them was feedback that never showed: the bar sat at 1 / 2, the window did not
        # repaint, and a Fit that took seconds looked like a Fit that did nothing.
        self.progress.show('Fitting page %d of %d' % (self.page_index + 1, total),
                           None, detail='measuring the two files')
        self.root.update()
        try:
            self.progress.pulse('measuring the two files')
            self.root.update()
            self.session.refit(self.page_index, report=self._fit_stage)
            self.progress.pulse('reloading the page and drawing it')
            self.root.update()
            self._load_page(self.page_index)
            self._redraw()
            self.root.update_idletasks()
        except Exception as exc:                  # reported, never swallowed
            self.progress.failed('The page could not be fitted',
                                 '%s: %s' % (type(exc).__name__, exc))
            return
        self.progress.hide()
        self._take_focus()

    def _dpi_stage(self, label: str) -> None:
        """One stage of a DPI change, said in the panel and painted before the next one starts."""
        self.progress.pulse(label)
        self.root.update()

    def _fit_stage(self, label: str) -> None:
        """One stage of a Fit: said in the panel, and painted before the next one starts."""
        self.progress.pulse(label)
        self.root.update()

    def refit(self) -> None:
        """The old name for `fit`, kept for anything that still calls it."""
        self.fit()

    def _reload_placement(self) -> None:
        """The ink moved, so everything derived from where it is goes too."""
        self._load_page(self.page_index)
        self._take_focus()

    def _update_fit_label(self) -> None:
        """A short word in the bar, the whole diagnostic in the terminal.

        The full message is a sentence -- "right side x0.183 at (+0.0, +70.4) -
        page boxes (the inks did not match)" -- and parked in the toolbar it
        took the width of the whole row and pushed the buttons off the edge.
        The bar gets the verdict; stdout gets the reason, which lands in the
        launcher's console because the window's stdout is redirected there.
        """
        if not hasattr(self, "fit_label"):
            return
        side = self.session.moving_side(self.page_index)
        fit = self.session.fit_for(self.page_index, side)
        if fit.is_identity:
            self.fit_label.config(text=f"{side}: unchanged")
            return
        # No offset here. The bar has to hold a row of buttons as well,
        # and with the numbers on it the label ran off the right edge --
        # visible in a screenshot of the real window. The offset and the
        # reason are in the terminal, which has room for a sentence.
        self.fit_label.config(text=f"{side} side x{fit.scale:.3f}")
        print(f"[merge] {side} side x{fit.scale:.3f} at "
              f"({fit.dx:+.1f}, {fit.dy:+.1f}) - {fit.reason}", flush=True)

    def _take_focus(self) -> None:
        try:
            self.root.lift()
            self.root.focus_force()
            self.canvas.focus_set()
        except tk.TclError:
            pass  # window already closed

    def _bind_hotkeys(self) -> None:
        self.root.bind("<Left>", lambda e: self.prev_page())
        self.root.bind("<Right>", lambda e: self.next_page())
        # The same two on PageUp/PageDown, which is what the converter uses
        # and what the user asked for. `<Control-Prior>` / `<Control-Next>` already
        # *move* a page and are left alone.
        self.root.bind("<Prior>", lambda e: self.prev_page())
        self.root.bind("<Next>", lambda e: self.next_page())
        self.root.bind("<plus>", lambda e: self._zoom_by(1.25))
        self.root.bind("<equal>", lambda e: self._zoom_by(1.25))
        self.root.bind("<minus>", lambda e: self._zoom_by(0.8))
        # No actual-size key: the user asked for Ctrl+0 gone, 2026-10-28, with
        # nothing in its place. This row also used to say Ctrl+0 while
        # binding a bare 0, which is the drift the panel test exists for.
        self.root.bind("m", lambda e: self.edit_page_map())
        if not self.read_only:
            self.root.bind("<Control-Prior>", lambda e: self.move_page(-1))
            self.root.bind("<Control-Next>", lambda e: self.move_page(1))
        # Previews are read-only, so they work in a read-only diff too.
        self.root.bind("<Control-Key-1>", lambda e: self.set_view(views.SOURCE_LEFT))
        self.root.bind("<Control-Key-2>", lambda e: self.set_view(views.SOURCE_RIGHT))
        self.root.bind("<Control-Key-3>", lambda e: self.set_view(views.OVERLAY))
        self.root.bind("p", lambda e: self.set_view(views.FINAL))
        self.root.bind("<Escape>", lambda e: self.set_view(views.EDIT))
        if not self.read_only:
            self.root.bind("<KeyPress-1>", lambda e: self._set_active_color(TAKE_LEFT))
            self.root.bind("<KeyPress-2>", lambda e: self._set_active_color(TAKE_RIGHT))
            self.root.bind("<KeyPress-3>", lambda e: self._set_active_color(TAKE_BOTH))
            self.root.bind("<KeyPress-4>", lambda e: self._set_active_color(TAKE_NONE))
            self.root.bind("r", lambda e: self.reset_page())
            self.root.bind("<Control-Key-4>", lambda e: self._toggle_split())
            # The fit, from the keyboard: one key, and it measures. The ratio
            # and offset nudges that used to sit on `[`/`]` and Alt+arrows went
            # with the nudge controls (705f814, "One Fit button, no nudge
            # controls"), so nothing binds those keysyms any more -- the comment
            # that still promised them is why this note says so.
            self.root.bind("f", lambda e: self.refit())
            self.root.bind("<F1>", lambda e: self.show_help())
            # Ctrl+S writes the default name and closes; Ctrl+Shift+S asks,
            # prefilled with that name, and closes too. The arranger's two
            # save keys have always worked this way, and the user's rule is that the
            # three tools answer the same keys the same way. Ctrl+Shift+Z is
            # redo, so Ctrl+Shift+S was free.
            # **Shift comes from `state`, never from the letter's case.**
            # Caps Lock inverts the case, so with it on `S` means no Shift and
            # `s` means Shift is down. Measured on the user's machine, before this:
            # Ctrl+S fired nothing at all, and Ctrl+Shift+S matched
            # `<Control-s>` and did a plain save -- one dead key and one doing
            # the wrong thing silently. The user met it in the converter first; this
            # window had it too.
            for case in ('s', 'S'):
                self.root.bind("<Control-%s>" % case, self._save_key)
            # Undo/redo repeat while held. Windows' own key auto-repeat
            # stutters and stops under a modifier, so the repeat is driven
            # here: the first press fires immediately, then a timer keeps
            # firing until the key comes back up.
            # Undo and redo are told apart the same way, and for the same
            # reason: with Caps Lock on, Ctrl+Z was dead and Ctrl+Shift+Z
            # undid instead of redoing.
            for case in ('z', 'Z'):
                self.root.bind("<Control-%s>" % case, self._undo_key)
                self.root.bind("<KeyRelease-%s>" % case, lambda e: self._stop_repeat())
            for case in ('y', 'Y'):
                self.root.bind("<Control-%s>" % case,
                               lambda e: self._start_repeat(self._redo))
                self.root.bind("<KeyRelease-%s>" % case, lambda e: self._stop_repeat())
            for i, size in enumerate(BRUSH_SIZE_PRESETS, start=1):
                self.root.bind(f"<Alt-KeyPress-{i}>", lambda e, s=size: self._set_brush_size(s))
            # Shift+1..9 too, bound through the keysym each shifted digit
            # actually reports. A Windows layout sends `exclam` for Shift+1,
            # so `<Shift-KeyPress-1>` would never fire -- which is exactly how a
            # correct-looking Shift shortcut ends up doing nothing.
            for i, size in enumerate(BRUSH_SIZE_PRESETS, start=1):
                keysym = SHIFTED_DIGIT_KEYSYMS[i - 1]
                # Both spellings. A Windows layout reports `exclam` for Shift+1
                # on most layout/version combinations, but not for all of them,
                # and `#` in particular did not match its shifted keysym in a
                # driven test while the other eight did. Binding the digit as
                # well costs nothing and covers the layouts that report it.
                for name in (keysym, str(i)):
                    self.root.bind(f"<Shift-KeyPress-{name}>",
                                   lambda e, s=size: self._set_brush_size(s))

    def run(self) -> None:
        """Only actually blocks on mainloop() when this app owns the root
        (standalone use). When opened as a Toplevel under another app's
        root (see `master` above), that app's own mainloop already drives
        this window -- calling mainloop() again here would nest event loops."""
        if self._owns_root:
            self.root.mainloop()

    # ---------------- page handling ----------------

    def splits_here(self, index=None) -> bool:
        """Whether this page is one half of a page that was split."""
        index = self.page_index if index is None else index
        return self.session.is_split(index)

    def locked(self) -> bool:
        """Whether the brush is inert right now.

        Only in a preview: a view that shows you something must not be a view
        that changes it. Both halves of a split are ordinary pages -- the
        work was done on them, and it goes on being possible.
        """
        return (self.read_only or self._saving
                or self.view_mode != views.EDIT)

    def _apply_dpi(self, event=None) -> None:
        """Read the DPI field and re-identify the page at that resolution.

        The three classes are a property of the page, decided once at this DPI
        (`classification.py`); this is the one place it is chosen. What it is
        not is a zoom: the page looks the same size before and after, and only
        the quality of the classification -- how finely a boundary between
        black, red and blue is drawn -- changes.

        An unreadable or out-of-range value is put back rather than clamped
        silently: a field that answers with a different number than the one
        typed is a field that lies about what it did.
        """
        if not hasattr(self, 'dpi_var'):
            return
        typed = (self.dpi_var.get() or '').strip()
        try:
            wanted = float(typed)
        except ValueError:
            wanted = None
        if wanted is None or not (MIN_CLASSIFY_DPI <= wanted <= MAX_CLASSIFY_DPI):
            self.dpi_var.set(str(int(round(self.session.classify_dpi))))
            self.dpi_label.config(
                text="%g-%g" % (MIN_CLASSIFY_DPI, MAX_CLASSIFY_DPI))
            return
        before = self.session.render_dpi
        # **One DPI for the whole base pipeline, and it is work.** A resolution change re-renders the source, takes
        # the ink out of the new raster, classifies again and rebuilds the layers; it used to happen in one silent
        # block, so the field said 576 while the picture stayed as it was for as long as that took.
        self.progress.show('Rendering at %g dpi' % wanted, None,
                           detail='the page is being rasterised again')
        self.root.update()
        try:
            self.progress.pulse('discarding what was rendered at %g dpi' % before)
            self.root.update()
            self.session.set_render_dpi(wanted, report=self._dpi_stage)
            self.dpi_var.set(str(int(round(self.session.render_dpi))))
            self._say_the_resolution()
            self.progress.pulse('drawing the page at %.1f pixels per unit'
                                % self.session.render_scale())
            self.root.update()
        except Exception as exc:                # reported, never swallowed
            self.progress.failed('The page could not be re-rendered',
                                 '%s: %s' % (type(exc).__name__, exc))
            self.dpi_var.set(str(int(round(before))))
            return
        # **What changed is the DPI, not the size of a cache.** The setter
        # returns how many cached classifications it dropped, and a page nobody
        # has drawn yet drops none -- reading that as "nothing changed" left the
        # old picture on screen after a real change.
        if self.session.render_dpi == before:
            self.progress.hide()
            return
        try:
            self._rebuild_view()
            self._update_page_label()
            self.progress.pulse('drawing the page at %.1f pixels per unit'
                                % self.session.render_scale())
            self.root.update()
            self._redraw()
        except Exception as exc:                # reported, never swallowed
            self.progress.failed('The page could not be re-rendered',
                                 '%s: %s' % (type(exc).__name__, exc))
            return
        self.progress.hide()

    def set_view(self, mode: str) -> None:
        """Show one of the views, built from scratch (views.py).

        Nothing is carried over from the view before it -- no image, no mask,
        no tint -- so the same mode always draws the same thing no matter
        what was on screen first, and no preview can change a decision.
        """
        self.view_mode = mode
        if hasattr(self, 'view_var'):
            self.view_var.set(mode if mode != views.FINAL else views.FINAL)
        if self.mask is not None:
            self.mask.clear()
        self._dragging = False
        self._last_drag_end = time.monotonic()   # the sharpen waits on this
        self._rebuild_view()
        self._update_page_label()
        self._redraw()

    def _washed(self, image, mask_array):
        """The brush's wash on the window, on the card where there is one.

        `gpurender.wash` is the same blend the window was doing with PIL,
        integer for integer -- the alpha truncated (`int(v * 0.30)`, so a full
        pixel is 76 and not 76.5) and the mix `(over*a + base*(255-a) + 127)
        // 255`, read off PIL rather than guessed at. `tests/test_gpu_render.py`
        compares the two, so they cannot drift apart quietly.

        It is worth moving because this is the only drawing in the window that
        happens on a mouse-motion event. A stroke is drawn once, when the
        gesture ends; this is drawn on every event in between.
        """
        from xp.gpu import painter as gpurender
        colour = BRUSH_TINT[self.active_color]
        if gpurender.Painter.available():
            try:
                mixed = gpurender.wash(np.asarray(image.convert('RGB')),
                                       mask_array, colour, BRUSH_TINT_ALPHA)
                return Image.fromarray(mixed, 'RGB')
            except Exception:
                pass
        overlay = Image.new('RGB', image.size, colour)
        alpha = Image.fromarray(np.asarray(mask_array, np.uint8)).point(
            lambda v: int(v * BRUSH_TINT_ALPHA))
        return Image.composite(overlay, image, alpha)

    def _swap_sides(self) -> None:
        """Close this and let the launcher open it the other way round.

        Not swapped in place. The session holds a parsed left and a parsed
        right, the merged page's size is the left's, and every cache in this
        window is keyed on them -- the brush mask's size included. Re-opening
        is the honest version of "the other way round", and it is exactly what
        the old *Choose sides* window's Swap did; it has only moved.
        """
        swap, self.on_swap = self.on_swap, None
        if swap is None:
            return
        self.root.destroy()
        swap()

    def _rebuild_view(self) -> None:
        """The current view, drawn fresh from the sources and decisions.

        At `session.scale` and no other, because this image is the logical
        space: the brush mask, the stroke boxes and `_canvas_xy` are all in
        its pixels. What the canvas is shown is a different question, and
        `_display_image` answers it.
        """
        self._view_image = views.render(self.session, self.page_index, self.view_mode,
                                        matched=getattr(self, '_matched', None))
        self._forget_zoom()
        if self.view_mode == views.EDIT:
            self._edit_colors = views.edit_colors(self.session, self.page_index, self._matched)

    def _show_the_empty_state(self) -> None:
        """"No paired pages", and the way back to the mapping.

        The user's 8.2, word for word: *"Merger main view should show a simple empty
        state ... no need to create fake left and right pages, no need to
        generate blank merge pages."* So nothing is rendered and nothing is
        invented; the canvas says what is true and names the key that fixes
        it.
        """
        self._left_strokes = []
        self._right_strokes = []
        self._matched = None
        self._edit_colors = {}
        # **Not wrapped in a bare `except`.** The first version of this was,
        # and it hid two mistakes of its own: `SILVER` is an RGB tuple, not a
        # Tk colour, and `look` is not imported here. Both raised, both were
        # swallowed, and the canvas stayed blank while the method reported
        # itself as having run. A message that cannot be drawn should say so.
        self.canvas.delete('all')
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.create_text(
                width // 2, height // 2, anchor='center', justify='center',
                fill='#c0c0c0', font=('TkDefaultFont', 12),
                text='No paired pages.\nPress M to configure Page Mapping.')
        # `_print` is given to the window by whatever launched it, so it
        # is not always there -- a merge opened straight from a test has
        # none, and the canvas already says this anyway.
        say = getattr(self, '_print', None)
        if callable(say):
            try:
                say('No paired pages -- press M to configure Page Mapping.')
            except Exception:
                pass

    def _load_page(self, index: int, say=None) -> None:
        # Every load is a new page state; work started for an older one must not
        # write over it. (It was the picture conversion that taught us this: a
        # pass that finished after a page turn wrote its result onto whichever
        # page was on screen.)
        self._page_epoch = getattr(self, '_page_epoch', 0) + 1
        if self.session.page_count == 0:
            # **Nothing is paired, so there is nothing to load.** The user's 8.2: a
            # merge with no pairs shows a plain empty state and invents no
            # pages. Everything below asks the session for page `index` and
            # would raise `IndexError` on an empty list -- which is how the
            # window used to die the moment the user unpaired the last page.
            self._show_the_empty_state()
            return
        """Load a page. `say` hears each real step, for the turning panel.

        The steps are the ones that actually cost something -- parsing the
        strokes, matching the two sides, composing the page -- so a bar that
        moves is reporting work that happened, not a timer. Callers that are
        not a page turn pass nothing and get the old silent behaviour.
        """
        self.page_index = max(0, min(index, self.session.page_count - 1))

        # session.strokes_for_page caches the parsed strokes (shared with
        # brush_capture's own hot path), and the editing colours cost a
        # stroke-matching pass over the page, so both are held here rather
        # than redone on every redraw -- which happens on every single
        # mouse-motion event while brushing.
        if say is not None:
            say('reading the strokes')
        self._left_strokes, self._right_strokes = self.session.strokes_for_page(self.page_index)
        if say is not None:
            say('matching the two sides')
        self._matched = views.matched_sides(self.session, self.page_index)
        self._edit_colors = views.edit_colors(self.session, self.page_index, self._matched)

        if say is not None:
            say('composing the page')
        w_px, h_px = views.page_pixels(self.session, self.page_index)
        self.mask = BrushMask(w_px, h_px, radius=self.brush_radius)
        self._update_page_label()

        self._left_bounds = self._bounds(self._left_strokes)
        self._right_bounds = self._bounds(self._right_strokes)

        self._surface_lost = False               # a new page is a new chance for the surface
        self._probe_the_surface = True           # and a new frame to ask the front buffer about
        self._rebuild_view()
        self._redraw()

    def _turn_to(self, index: int) -> None:
        """Turn to a page, with the panel naming the page that is coming.

        The user said the wait is acceptable and asked for feedback, not speed, so
        nothing here is faster: the page is loaded the same way it always was.
        What is new is that the turn says which page it is fetching before the
        work starts, and reports each real step while it runs. It is
        synchronous on purpose -- that is what makes "a slow page must never
        overwrite the page the user last asked for" true by construction rather than
        by a generation counter that someone has to remember to check.
        """
        total = self.session.page_count
        index = max(0, min(index, total - 1))
        if index == self.page_index:
            return
        previous = self.page_index
        self.progress.show('Loading page %d of %d' % (index + 1, total), 3,
                           detail='page %d, coming in...' % (index + 1))
        self.page_label.config(text='Page %d / %d  -  loading...' % (index + 1, total))
        self.root.update_idletasks()

        steps = [0]

        def say(text):
            steps[0] += 1
            self.progress.step(steps[0], 3, text)
            # Draw it now: the load below runs on this thread, so this is the
            # only moment the panel can be painted before it is over.
            self.root.update_idletasks()

        try:
            self._load_page(index, say=say)
        except Exception as exc:                    # reported, never swallowed
            # **The picture and the state have to come back together.** `_load_page` has already
            # replaced the strokes, the matched sets and the colour map by the time
            # `_rebuild_view` raises, while the canvas still shows the page the user was on -- so
            # putting `page_index` back alone left a window showing one page and editing another.
            self.page_index = previous
            try:
                self._load_page(previous)
            except Exception:
                # The page the user came from cannot be rebuilt either: leave its picture up and say
                # so, rather than drawing a half-page over it.
                self._update_page_label()
            self.progress.failed('Page %d did not load' % (index + 1),
                                 '%s: %s' % (type(exc).__name__, exc))
            return
        self.progress.hide()

    def next_page(self) -> None:
        self._turn_to(self.page_index + 1)

    def prev_page(self) -> None:
        self._turn_to(self.page_index - 1)

    def reset_page(self) -> None:
        """Everything done to this page, undone -- including its split."""
        if self.locked():
            return
        self._load_page(self.session.reset_page(self.page_index))

    def _update_page_label(self) -> None:
        origin = self.session.page_origin(self.page_index)
        number = self.page_index + 1
        label = f"Page {number} / {self.session.page_count}"
        if origin != 'both':
            label += f"  -  {origin}"
        self.page_label.config(text=label)
        if hasattr(self, "split_button"):
            can_split = self.session.can_split(self.page_index)
            is_half = self.session.is_split(self.page_index)
            self.split_button.config(
                text="Split into 2 pages (Ctrl+4)" if can_split else "Join back (Ctrl+4)",
                state=tk.NORMAL if (can_split or is_half) else tk.DISABLED)
            if is_half:
                note = "(one half of a split; Move up/down sets which comes first)"
            elif can_split:
                note = ""
            else:
                note = "(nothing to split: this page has one side)"
            self.split_note.config(text=note)
        self._update_fit_label()
        if hasattr(self, "view_label"):
            geometry = self.session.geometry(self.page_index)
            note = views.MODE_LABELS.get(self.view_mode, '')
            if not geometry.compatible:
                note += ('   (pages not scaled: %s)' % geometry.reason)
            self.view_label.config(text=note)

    def _repaint(self, refs) -> None:
        """Take in what a decision changed and put the view back in step."""
        self._apply_changes(refs)
        self._update_page_label()
        self._redraw()

    def _apply_changes(self, refs) -> None:
        if not refs:
            return
        # The colours are derived from the decisions, so they are brought up
        # to date before anything is drawn -- a stroke that has just been
        # decided about has to come out black, not in the colour it had a
        # moment ago. Only the strokes this decision touched, though: the map
        # has one entry per stroke on the page, and walking all of them on
        # every gesture is the cost `views.recolor` exists to avoid.
        # `tests/test_brush_is_quick.py` holds that, and holds that the
        # incremental map still equals a rebuilt one.
        if self._edit_colors is None:
            self._edit_colors = views.edit_colors(self.session, self.page_index,
                                                  self._matched)
        else:
            views.recolor(self._edit_colors, self.session, self.page_index,
                          refs, self._matched)
        self._repair(refs)

    # Undo/redo report success by whether the stack had anything on it, not
    # by whether strokes came back: undoing a split changes no strokes, and
    # treating that as "nothing to undo" would stop a held Ctrl+Z there.
    def _undo(self) -> bool:
        if self.locked() or not self.session.can_undo(self.page_index):
            return False
        layered = self._by_layers()
        self._repaint(self.session.undo(self.page_index))
        if layered:
            self._redraw_layers()
        return True

    def _redo(self) -> bool:
        if self.locked() or not self.session.can_redo(self.page_index):
            return False
        layered = self._by_layers()
        self._repaint(self.session.redo(self.page_index))
        if layered:
            self._redraw_layers()
        return True

    def _redraw_layers(self) -> None:
        """Rebuild the page from the layers as they now stand.

        For undo, redo and reset -- which are rare, and whose extent is the
        whole of a gesture rather than one segment of it. The brush itself
        never comes through here: it repairs the box it changed.
        """
        self._forget_zoom()
        self._rebuild_view()
        self._redraw()

    # ---------------- whole-page actions ----------------

    def _apply_page(self, active: str) -> None:
        """Every stroke on the page, one side (or both), as one undo step."""
        if self.locked():
            return
        layered = self._by_layers()
        self._repaint(self.session.apply_page(self.page_index, active))
        if layered:
            # A whole-page action really does cover the page, so this is a
            # full redraw and should be. The brush is the thing that must
            # not come through here.
            self._redraw_layers()

    def _toggle_split(self) -> None:
        """Split this page in two, or put a split back together.

        The merge is one page longer straight away -- these are pages of the
        document now, not a note to be acted on at save time -- so the view
        moves to the half that keeps the left side and the page count you see
        is the page count you will get.

        **It says so while it does it, and it redraws itself when it is done.**
        The user's report: the split gave no feedback, and the window kept showing the
        old page until the user pressed PageUp and PageDown. The page list, the
        strokes, the label and the picture are all reloaded here -- the two
        steps are the two real ones, so the count is work that happened.
        """
        if self.read_only or self._saving:
            return
        splitting = self.session.can_split(self.page_index)
        joining = self.session.is_split(self.page_index)
        if not (splitting or joining):
            messagebox.showinfo(
                "XP", "This page has one side only, so there is nothing to split off.",
                parent=self.root)
            return

        total = self.session.page_count
        title = ('%s page %d of %d'
                 % ('Splitting' if splitting else 'Joining', self.page_index + 1, total))
        self.progress.show(title, 2,
                           detail='one page in, two pages out' if splitting
                           else 'two pages in, one page out')
        self.root.update_idletasks()

        def say(done: int, text: str) -> None:
            self.progress.step(done, 2, text)
            # The work below runs on this thread, so this is the only moment the
            # panel can be painted before it is over.
            self.root.update_idletasks()

        try:
            say(1, 'rebuilding the page list')
            index = (self.session.split_page(self.page_index) if splitting
                     else self.session.join_page(self.page_index))
            say(2, 'reloading the page and drawing it')
            self._load_page(index)
            # Drawn now, not at the next page turn: the surface and the canvas
            # paths both go through `_redraw`, and the label was updated by
            # `_load_page`.
            self._redraw()
            self.root.update_idletasks()
        except Exception as exc:                  # reported, never swallowed
            self.progress.failed(
                'The page %s did not finish' % ('split' if splitting else 'join'),
                '%s: %s' % (type(exc).__name__, exc))
            # The session may already have changed; the window has to be
            # showing a page that exists, whatever happened.
            try:
                self._load_page(min(self.page_index, self.session.page_count - 1))
            except Exception:
                pass
            return
        self.progress.hide()
        self._take_focus()

    def move_page(self, delta: int) -> None:
        """Move this page up or down the merged document."""
        if self.read_only or self._saving:
            return
        self._load_page(self.session.move_page(self.page_index, delta))

    # ---------------- hold-to-repeat ----------------

    def _start_repeat(self, action) -> None:
        """Run `action` now, then keep running it while the key is held.
        Stops on its own once the action reports there's nothing left."""
        self._stop_repeat()
        if not action():
            return

        def again(delay: int) -> None:
            if not action():
                self._repeat_job = None
                return
            self._repeat_job = self.root.after(delay, lambda: again(REPEAT_INTERVAL_MS))

        self._repeat_job = self.root.after(REPEAT_DELAY_MS, lambda: again(REPEAT_INTERVAL_MS))

    def _stop_repeat(self) -> None:
        if self._repeat_job is not None:
            try:
                self.root.after_cancel(self._repeat_job)
            except tk.TclError:
                pass
            self._repeat_job = None

    def _on_brush_change(self) -> None:
        self.active_color = self.brush_var.get()

    def _set_active_color(self, color: str) -> None:
        self.active_color = color
        self.brush_var.set(color)

    def _set_brush_size(self, size: int) -> None:
        self.brush_radius = size
        if self.mask is not None:
            self.mask.radius = size
        if hasattr(self, "size_label"):
            self.size_label.config(text=f"size {size}px")

    def _pan_and_redraw(self, event) -> None:
        """Panning moves the window, and what is drawn follows the window.

        The overscan means most drags cost nothing: a redraw only happens
        when the window has moved off the piece that was drawn.
        """
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self._viewport_changed()
        self._request_redraw()

    def _scroll_sideways(self, event) -> None:
        self.canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")
        self._viewport_changed()
        self._request_redraw()

    # ---------------- zoom ----------------

    def _forget_zoom(self) -> None:
        self._completed_picture = None
        self._zoomed = self._zoomed_at = self._zoomed_box = None
        self._zoomed_is_rough = False
        # **A new zoom is a fresh chance for a render that kept failing.** The
        # count is of failures at *this* zoom; carrying it over would give up
        # on the third zoom of a session even though each one is a new request.
        self._sharpen_failures = 0
        # A sharper picture that is still coming belongs to the zoom that
        # asked for it. Moving on makes it stale, and a stale frame must not
        # land on the page the user is looking at now.
        self._zoom_token = getattr(self, '_zoom_token', 0) + 1
        job = getattr(self, '_sharpen_job', None)
        if job is not None:
            try:
                self.root.after_cancel(job)
            except Exception:
                pass
            self._sharpen_job = None

    def _viewport_changed(self):
        """Invalidate requests, not the last completed image; debounce every input."""
        self._zoom_token = getattr(self, '_zoom_token', 0) + 1
        self._input_stopped_at = time.monotonic() + 0.100
        self._sharpen_failures = 0
        job = getattr(self, '_sharpen_job', None)
        if job is not None:
            self.root.after_cancel(job)
        self._sharpen_job = self.root.after(
            100, lambda t=self._zoom_token: self._sharpen(t))
        self._zoomed_is_rough = True
        self._zoomed_at = None

    def _rough_at_this_zoom(self):
        """Transform cached pixels only; allocate at most the visible rectangle.

        The full-page base covers newly revealed areas; the last completed sharp
        tile supplies its higher resolution wherever it overlaps this viewport.
        Neither cache is discarded by a camera operation.
        """
        base = self._view_image
        if base is None:
            return None
        visible = self._visible_box() or (0, 0, *self._page_size_at_zoom())
        left, top, width, height = map(int, visible)
        size = max(1, width), max(1, height)
        z = self.zoom
        rough = base.transform(size, Image.Transform.AFFINE,
                               (1/z, 0, left/z, 0, 1/z, top/z),
                               resample=_stand_in_resample(z), fillcolor='white')
        completed = getattr(self, '_completed_picture', None)
        if completed is not None:
            image, old_zoom, (ox, oy) = completed
            ratio = old_zoom / z
            x0 = max(0, int(np.ceil(ox / ratio - left)))
            y0 = max(0, int(np.ceil(oy / ratio - top)))
            x1 = min(size[0], int((ox + image.width) / ratio - left))
            y1 = min(size[1], int((oy + image.height) / ratio - top))
            if x1 > x0 and y1 > y0:
                tile = image.transform((x1-x0, y1-y0), Image.Transform.AFFINE,
                    (ratio, 0, (left+x0)*ratio-ox,
                     0, ratio, (top+y0)*ratio-oy),
                    resample=Image.Resampling.NEAREST)
                rough.paste(tile, (x0, y0))
        return rough

    def _sharpen(self, token: int) -> None:
        """Start the high-resolution render **on a worker**, and let the UI thread go.

        Waiting for the render inside the UI thread is not progressive rendering: the picture on screen is blurry and the
        window cannot repaint, take another wheel tick, or answer anything. This decides what to render here and hands
        the work over; `_the_sharp_picture` puts the result up on the UI thread when it arrives.
        """
        self._sharpen_job = None
        if getattr(self, '_closed', False) or getattr(self, 'root', None) is None:
            return
        current = getattr(self, '_zoom_token', 0)
        remaining = getattr(self, '_input_stopped_at', 0) - time.monotonic()
        if remaining > 0:
            self._sharpen_job = self.root.after(
                max(1, int(remaining * 1000) + 1), lambda t=current: self._sharpen(t))
            return
        if token != current:
            # **A wheel tick landed while this was queued.** The picture on screen is still the rough one, so asking
            # again is the whole point of a progressive render -- returning here left it blurred until the next zoom,
            # which is exactly what "I didn't see more detail" was.
            if getattr(self, '_zoomed_is_rough', False):
                self._sharpen_job = self.root.after(
                    100, lambda t=current: self._sharpen(t))
            return
        if not getattr(self, '_zoomed_is_rough', False):
            return
        # **Not while the user is drawing, and not between two strokes either.**
        # The brush's own frames render at the view's scale and are what the user is
        # looking at; a sharpen beside them halves the CPU and the drag goes
        # over its budget (measured: 80.8 ms P50 against a 50 ms target on a
        # 40,000-stroke page). The `_dragging` half of this was already here.
        #
        # The grace is the other half, and the supersample is what made it
        # necessary: a render is four to nine times the pixels it used to be,
        # so one started in the gap between two strokes is still going when
        # the next begins -- and a render cannot be called back. Brushing is
        # interaction, and the user's rule for interaction is that the sharpening
        # waits for it to stop, not for one stroke of it to stop.
        resting = time.monotonic() - getattr(self, '_last_drag_end', 0.0)
        if getattr(self, '_dragging', False) or resting < BRUSH_GRACE:
            self._sharpen_job = self.root.after(120, lambda t=token: self._sharpen(t))
            return
        visible = self._visible_box()
        left, top, width, height = self._wanted_box(visible)
        job = (token, self.page_index, self.view_mode,
               self.session.scale * self.zoom, getattr(self, '_matched', None),
               (left, top, width, height), float(self.zoom),
               self._supersample(self.session.scale * self.zoom,
                                 (left, top, width, height)))
        worker = getattr(self, '_sharp_worker', None)
        if worker is not None and worker.is_alive():
            # **One render at a time, and the newest request wins.** Python cannot stop a running call here, so the old
            # one is left to finish and its result is dropped when it is seen to be stale.
            self._sharp_wanted = job
            # **The result is picked up by the poller, so one has to be there.** This used to assume it; a poller that
            # had already consumed a result left `_sharp_wanted` set and never read -- the newest request recorded and
            # never rendered, which from the outside is "it stopped sharpening".
            if getattr(self, '_collect_job', None) is None:
                self._collect_job = self.root.after(30, self._collect_the_sharp_picture)
            return
        # **The thread must not be able to reach Tk.** It used to run a bound
        # method, so its only reference was the whole `MergeApp` -- every Tk
        # variable in the window with it. When the window went away while a
        # render was in flight, the app was collected *on the worker thread*,
        # `tkinter.Variable.__del__` ran there, and Tcl ended the process with
        # `Tcl_AsyncDelete: async handler deleted by the wrong thread`. It
        # surfaced the moment the supersample made renders long enough to
        # still be running at teardown. The worker is handed the session, the
        # job and a plain box to put the answer in, and nothing else.
        self._sharp_result = None
        self._sharp_box = box = {}
        thread = threading.Thread(target=_render_job,
                                  args=(self.session, job, box), daemon=True)
        thread.start()
        self._sharp_worker = thread
        self._sharp_started = time.monotonic()
        self._sharp_slow_reported = False
        self._sharp_wanted = None
        self._collect_job = self.root.after(30, self._collect_the_sharp_picture)

    def _collect_the_sharp_picture(self) -> None:
        """On the UI thread: take the worker's result when it is there, and keep asking until it is."""
        self._collect_job = None
        root = getattr(self, 'root', None)
        if root is None or getattr(self, '_closed', False):
            return
        try:
            if not root.winfo_exists():
                return
        except Exception:
            return
        # The worker leaves its answer in a plain box it was handed; see
        # `_render_job` for why it may not be given the window.
        box = getattr(self, '_sharp_box', None)
        result = box.pop('result', None) if box else None
        if result is None:
            result = getattr(self, '_sharp_result', None)
        if result is not None:
            self._sharp_result = None
            self._the_sharp_picture(result[0], result[1])
            return
        worker = getattr(self, '_sharp_worker', None)
        if worker is not None and worker.is_alive():
            # **A worker that never comes back is a failure, not a slow render.** Without this the poll below ran every
            # 30 ms for the life of the window, the picture stayed stretched, and nothing anywhere said why.
            started = getattr(self, '_sharp_started', None)
            if started is not None and time.monotonic() - started > SHARPEN_DEADLINE:
                # A timed-out thread still owns the render slot. Never launch a
                # second writer into the session caches or shared result slot.
                if not getattr(self, '_sharp_slow_reported', False):
                    print('[merge] sharpen is slow; keeping cached image', flush=True)
                    self._sharp_slow_reported = True
            try:
                self._collect_job = self.root.after(30, self._collect_the_sharp_picture)
            except Exception:
                self._collect_job = None
            return
        if worker is not None:
            # **Gone and left nothing.** The worker writes its result and only then ends, so a thread that ended with
            # no result either raised outside the guarded call or had its result cleared by a newer request. Silence
            # here is the other half of "the picture stayed rough for ever".
            self._sharp_worker = None
            self._sharpen_failed(getattr(self, '_zoom_token', 0), 'the render ended without a picture')

    def _say_the_resolution(self) -> None:
        """Put the dpi the picture really has beside the one the user asked for.

        The user's rule: *"if the resolution is lowered by a memory budget, say the
        number it actually landed on -- do not show 8000 in the field and
        render at 288 without saying so."* That is what this label is for.
        It reads `800 -> 1728 dpi` : the field, and what the frame on screen
        was sampled at once the zoom and the supersample are counted.
        """
        if not hasattr(self, 'dpi_label'):
            return
        try:
            over = self._supersample(self.session.scale * self.zoom,
                                     self._wanted_box(self._visible_box()))
            got = self.effective_dpi(over)
            asked = float(self.session.render_dpi)
            short = 'x%d  %.0f dpi on screen' % (over, got)
            if got + 1 < asked:
                # Below what the user asked for, so it says so rather than implying
                # the field was honoured.
                short += ' (asked %g, capped)' % asked
            self.dpi_label.config(text=short)
        except Exception:
            pass

    def _supersample(self, scale, window):
        """How many times over the display render samples, and the two reasons.

        **One resolution story, not three.** The user's complaint was that
        `session.scale`, `render_scale()` and the supersample were three
        unrelated numbers, and the user was right. They are one chain:

            screen samples the content at   session.scale * zoom * over
            the user's dpi field asks for          render_scale()
            the decisions live at           session.scale        (unmoved)

        `session.scale` is the space every mask, brush stroke and layer is
        recorded in. Moving it would re-resample every decision the user has made,
        so it does not move -- that is the invariant, not an oversight.
        Everything else is display, and `over` is what carries the user's dpi into it.

        Two separate reasons to sample over, and the larger wins:

        1. **Coverage.** `paint_flat` is flat in the literal sense -- there is
           not one mention of coverage in `raster.py` or `kernels.py` -- so the
           XOPP side reaches the screen in two colours, pure black and pure
           white, against the 238 greys PyMuPDF gives the PDF beside it.
           Shrinking a larger render by the plain mean of its samples *is*
           coverage, so the floor is two whatever the dpi says. This reason
           does not go away at high zoom: it is about the screen's pixels.

        2. **The user's dpi.** When the screen samples the content less finely than
           the user's field asks, the difference is made up here. At zoom 1 the user's 800
           dpi wants 11.111 px/unit against the screen's 4, so 2.78 times over;
           by 4x zoom the screen is already at 16 and asks nothing extra.

        The budget caps the product, and **what it actually achieved is
        reported** -- `effective_dpi` -- because a field reading 8000 over a
        picture drawn at 288 is the thing the user objected to.
        """
        try:
            on_screen = float(self.session.scale) * float(self.zoom)
            asked = float(self.session.render_scale())
        except Exception:
            return SUPERSAMPLE_FLOOR
        for_dpi = asked / on_screen if on_screen > 0 else 1.0
        if not np.isfinite(for_dpi):
            for_dpi = 1.0
        # **Up, not down.** Truncating 2.78 to 2 handed back 576 dpi where the user
        # asked for 800 and then reported the shortfall -- a shortfall this
        # could simply have covered. Rounding up reaches the user's number; the cap
        # below is the only thing allowed to take it away.
        want = max(SUPERSAMPLE_FLOOR, int(np.ceil(for_dpi - 1e-9)))
        # A cap on the pixels, not only on the factor: the window grows with
        # the zoom, and three times over an already large rectangle is how a
        # background render becomes a memory spike.
        cap = SUPERSAMPLE_MAX
        try:
            pixels = float(window[2]) * float(window[3])
            if pixels > 0:
                # The largest factor whose *samples* fit the budget. `over`
                # squares the work, so this is the quantity that matters and
                # the one the budget is written in.
                while cap > 1 and pixels * cap * cap > SUPERSAMPLE_SAMPLES:
                    cap -= 1
        except Exception:
            pass
        return max(1, min(want, cap))

    def effective_dpi(self, over=None):
        """The dpi the picture on screen was really sampled at.

        The user's rule, and it is a fair one: if the budget lowers the resolution,
        say the number it landed on. A field reading 8000 above a page drawn
        at 288 is the fault the user reported, and it was silent.
        """
        try:
            if over is None:
                over = self._supersample(self.session.scale * self.zoom,
                                         self._wanted_box(self._visible_box()))
            return float(self.session.scale) * float(self.zoom) * float(over) * 72.0
        except Exception:
            return 0.0

    @staticmethod
    def _stand_back() -> None:
        """Put this worker below the window in the scheduler's order.

        **The user's principle, and the scheduler has to be told.** *"Interaction
        and high-resolution rendering should run independently and must not
        block each other."* A Python thread is not polite by default: the
        sharpen is numpy and MuPDF holding the GIL in long stretches, and a
        brush stroke on the UI thread waits behind it. Measured when the
        supersample started rendering four to nine times the pixels:
        `test_brush_is_local` went from passing to a P95 of 153 ms against
        its 100 ms target -- on a page with no picture in it, so none of
        that cost was the new code's own work, all of it was contention.

        Below-normal and not lowest: lowest can be starved outright on a
        busy machine, and a sharpen that never finishes is the "it stays
        blurry" the user spent three rounds on.
        """
        try:
            import ctypes

            handle = ctypes.windll.kernel32.GetCurrentThread()
            ctypes.windll.kernel32.SetThreadPriority(handle, -1)
        except Exception:
            pass                            # not Windows, or not permitted

    def _render_for_sharpening(self, job) -> None:
        """The old in-process worker, kept only for callers that still name it.

        `_sharpen` runs `_render_job` on a thread instead: a bound method
        here means the thread's one reference is the whole window, Tk
        variables included, and a render still in flight when the window goes
        away is collected on the worker thread -- which Tcl ends the process
        over. This writes to the same box.
        """
        self._stand_back()
        box = getattr(self, '_sharp_box', None)
        if box is None:
            box = self._sharp_box = {}
        _render_job(self.session, job, box)
        return
        token, page_index, mode, scale, matched, window = job[:6]
        over = self._supersample(scale, window)
        drawn = None
        for attempt, use_match in ((1, matched), (2, None)):
            try:
                drawn = sharp_render(self.session, page_index, mode, scale,
                                     use_match, window, over)
                break
            except Exception as exc:
                # **Said, then tried again.** Swallowing this left the rough picture up for ever, and from the outside
                # that is "the progressive render does not work". The second attempt drops the match, which is the one
                # argument that can be stale after a decision.
                print('[merge] the sharpen render raised %s: %s%s'
                      % (type(exc).__name__, exc,
                         '' if attempt == 1 else ' -- the rough picture stays'), flush=True)
        # **Not `root.after` from here.** Tk is not thread safe, and a cross-thread `after` is dropped silently -- which
        # is how the first version of this left the picture rough for ever. The result is left here and the UI thread
        # picks it up: `_collect_the_sharp_picture`.
        self._sharp_result = (job, drawn)

    def _the_sharp_picture(self, job, drawn) -> None:
        """On the UI thread: put the rendered picture up if it is still the view the user is looking at."""
        self._sharp_worker = None
        token = job[0]
        current = getattr(self, '_zoom_token', 0)
        same_zoom = abs(float(getattr(self, 'zoom', 0.0)) - float(job[6])) < 1e-9
        if token != current or self.page_index != job[1] or self.view_mode != job[2] or not same_zoom:
            wanted = getattr(self, '_sharp_wanted', None)
            self._sharp_wanted = None
            if wanted is not None:
                self._sharpen(wanted[0])
            elif getattr(self, '_zoomed_is_rough', False):
                self._sharpen_job = self.root.after(
                    100, lambda t=current: self._sharpen(t))
            return
        if drawn is None:
            # **Checked for staleness first, and for failure second.** A render that raised is a failure of *this*
            # view, and it used to return here in silence -- no retry, no flag cleared, no word to the window. The
            # stretched picture is the right thing to keep; keeping it for ever without asking again was not.
            self._sharpen_failed(token, 'the render raised')
            return
        # **The swap.** Nothing else changes: the picture, the zoom it belongs to, the box it was cut from, and the flag
        # that says it is no longer rough.
        left, top = job[5][0], job[5][1]
        self._zoomed, self._zoomed_at = drawn, self.zoom
        self._zoomed_box = (left, top)
        self._zoomed_is_rough = False
        self._completed_picture = (drawn, self.zoom, (left, top))
        self._sharpen_failures = 0
        # **Ask the screen about this frame too.** The probe that reads the front buffer ran only after a zoom, so the
        # frame that carried the sharp picture was never checked -- and the sharpen is exactly the frame the user reports as
        # "it never arrived". One read, on the frame that is supposed to prove itself.
        self._probe_the_surface = True
        self._redraw()

    def _sharpen_failed(self, token, why: str) -> None:
        """A sharpen did not land: keep the picture up, ask again a few times, then say it once.

        **Not asking again was the bug.** `_the_sharp_picture` returned on a result of `None` and nothing else
        happened: one failed render left the stretched picture on screen until the next zoom -- for ever, if the user
        never zoomed again -- and the only trace was a line from a worker in a log nobody was reading. The picture
        staying up is right. Staying *silently* and *permanently* is not.
        """
        if getattr(self, '_closed', False) or getattr(self, 'root', None) is None:
            return
        if not getattr(self, '_zoomed_is_rough', False):
            return                      # something sharp is already up; there is nothing to recover
        tries = getattr(self, '_sharpen_failures', 0) + 1
        self._sharpen_failures = tries
        if tries <= SHARPEN_RETRIES:
            delay = SHARPEN_RETRY_DELAY * (3 ** (tries - 1))
            print('[merge] the sharpen failed (%s); asking again in %d ms (try %d of %d)'
                  % (why, delay, tries, SHARPEN_RETRIES), flush=True)
            self._sharpen_job = self.root.after(delay, lambda t=token: self._sharpen(t))
            return
        print('[merge] the sharpen gave up after %d tries (%s); the page stays at the zoomed '
              'picture until the next zoom' % (tries, why), flush=True)

    def _set_zoom(self, zoom: float, anchor=None) -> None:
        """Change the zoom, keeping the place the user was looking at.

        **The scroll position has to move before the picture is drawn.**
        `_redraw` asks `_visible_box` which part of the page to render, and
        `_visible_box` reads `canvasx(0)` -- the scroll offset. The canvas's
        `scrollregion` is only widened to the new zoom *inside* `_redraw`,
        after that question has been answered, so the first frame after a
        zoom was drawn for where the view used to be. Away from the top of
        the page that means rendering a rectangle the user is not looking at, and
        what the user sees where the user *is* looking is blank paper, until something
        else happens to redraw it.

        Re-anchoring is the other half, and it is why zooming threw the page
        about: the point in the middle of the view stays in the middle.
        `anchor` holds a different point still -- the pointer, for a wheel
        zoom.
        """
        old = self.zoom
        new = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
        if abs(new - old) < 1e-9:
            return
        try:
            view_w = max(1, int(self.canvas.winfo_width()))
            view_h = max(1, int(self.canvas.winfo_height()))
            hold_x, hold_y = (view_w / 2.0, view_h / 2.0) if anchor is None else anchor
            # The page point under that spot, in the zoom-free space the
            # strokes live in. The page's origin comes off first: it is where the page sits on
            # the canvas at the *old* zoom.
            origin_x, origin_y = self._page_origin()
            page_x = (self.canvas.canvasx(0) - origin_x + hold_x) / old
            page_y = (self.canvas.canvasy(0) - origin_y + hold_y) / old
        except Exception:
            hold_x = hold_y = 0.0
            page_x = page_y = 0.0

        self.zoom = new
        self._surface_lost = False              # a new size is a new chance for the surface
        self._probe_the_surface = True          # and a new frame to ask the front buffer about
        self._viewport_changed()                # keep the completed image while the camera moves
        self._say_the_resolution()              # the zoom changes what the frame is really sampled at
        try:
            page_w, page_h = self._page_size_at_zoom()
            # **At least the window, so a small page can still be placed.** With a scrollregion
            # of the page's own size and a page narrower than the window, the scroll fraction is
            # negative and Tk clamps it to 0 -- the page cannot be moved off the corner at all.
            span_x = max(page_w, view_w)
            span_y = max(page_h, view_h)
            self.canvas.configure(scrollregion=(0, 0, span_x, span_y))
            origin_x, origin_y = self._page_origin()
            left = origin_x + page_x * new - hold_x
            top = origin_y + page_y * new - hold_y
            self.canvas.xview_moveto(min(1.0, max(0.0, left / max(1.0, span_x))))
            self.canvas.yview_moveto(min(1.0, max(0.0, top / max(1.0, span_y))))
        except Exception:
            pass
        self._redraw()

    def _zoom_by(self, factor: float) -> None:
        self._set_zoom(self.zoom * factor)

    def _on_wheel(self, event) -> None:
        self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        self._viewport_changed()
        self._request_redraw()

    def _on_wheel_zoom(self, event) -> None:
        """Ctrl+wheel zooms about the pointer, not about the middle.

        Zooming about the centre means the thing under the cursor slides away
        as it grows, so getting closer to one word is a zoom followed by a
        hunt. The pointer is the only point the user has told us the user cares about.

        **This was reversed once and put back.** A round on 2026-10-29 changed
        it to the centre of the view, quoting the user as *"for the merger it should
        be the same as the converter -- zoom about the centre"*. The user's words were
        *"same as converter, zooming in and out centralized as the cursor"* --
        the cursor, not the centre -- and the converter zooms about the cursor:
        `convertgui._on_wheel` says so in as many words, "put the page point
        that was under it back under it". So "same as the converter" means this
        line, and the same round had already made the converter zoom about the
        centre and been told to undo it.

        If an axis has no room to move -- the paper narrower than the window --
        the clamp pins that axis and the point under the cursor cannot be held
        there. That is a fact about the clamp on that one axis; it is not a
        reason to give up the anchor on the axis that does have room.
        """
        self._set_zoom(self.zoom * (1.1 if event.delta > 0 else 1 / 1.1),
                       anchor=(event.x, event.y))

    # ---------------- drawing ----------------

    def _color_of(self, ref: Ref):
        """The editing colour for one stroke, or None if it is not drawn.

        views.edit_colors is the single source of this, so a repaint of one
        area colours its strokes exactly as a repaint of the whole page
        would, and neither can drift from the other.
        """
        if self._by_layers():
            pieces = self.piece_colours(ref.side, ref.index)
            if not pieces:
                return None
            colours = {colour for _o, _c, colour in pieces}
            # One answer only when the stroke really has one: a stroke the user
            # brushed part of is two colours now, and saying it is one of
            # them would be a lie in whichever direction it leaned.
            return colours.pop() if len(colours) == 1 else None
        return self._edit_colors.get((ref.side, ref.index))

    def piece_colours(self, side: str, index: int):
        """Every piece of this stroke as (offset, count, colour).

        `colour` is None where the piece is not drawn. This is the layered
        page's real answer: a decision is about a piece, so "what colour is
        this stroke" has as many answers as it has pieces.
        """
        if not self.session.uses_layers(self.page_index):
            colour = self._edit_colors.get((side, index))
            return [(0, -1, colour)]
        held = self.session.layers_for_page(self.page_index)[side]
        out = []
        for at in held.pieces_of(index):
            colour = (tuple(int(v) for v in held.colour[at])
                      if held.shown[at] else None)
            out.append((int(held.starts[at]) - held._base_of(index),
                        int(held.counts[at]), colour))
        return sorted(out)

    def _bounds(self, strokes):
        """Each stroke's box on screen, for finding what runs through an area."""
        boxes = np.zeros((len(strokes), 4), dtype=np.float64)
        for i, s in enumerate(strokes):
            x0, y0, x1, y1 = s.bbox
            pad = (s.widths[0] if s.widths else 1.0) * self.scale / 2 + 1
            boxes[i] = (x0 * self.scale - pad, y0 * self.scale - pad,
                        x1 * self.scale + pad, y1 * self.scale + pad)
        return boxes

    def _repair(self, refs) -> None:
        """Rebuild the area those strokes occupy, with the layered painter.

        **One painter, or the repair disagrees with the full render.** This used
        to draw the crossing strokes itself, coloured by `_color_of` -- one
        colour per stroke. That matched the view while the colour was decided
        per stroke; now that black is local (only where the ink coincides), the
        repair would paint a whole stroke in the colour of its first piece, and
        `test_view_repair` caught it as 90,459 pixels of drift. So the box is
        drawn by `views.layer_patch`, which is the full view with a window.
        """
        if self.view_mode != views.EDIT:
            return
        boxes = np.array([b for b in (self._box_of(ref) for ref in refs) if b is not None])
        if not len(boxes):
            return
        width, height = self._view_image.size
        x0 = max(0, int(np.floor(boxes[:, 0].min())))
        y0 = max(0, int(np.floor(boxes[:, 1].min())))
        x1 = min(width, int(np.ceil(boxes[:, 2].max())) + 1)
        y1 = min(height, int(np.ceil(boxes[:, 3].max())) + 1)
        # **The whole piece, not just the brush.** A decision changes the colour
        # of every point of the piece it touched, so clipping the repair to the
        # brush's own box left the rest of a long stroke in its old colour --
        # `test_view_repair` measured 43,680 pixels of drift. Bounding the
        # repair by the brush is only correct once pieces are cut where the
        # colour changes, which they are not yet.
        #
        # **A few pixels of bleed.** A stroke's own edge is drawn with round
        # caps and a width: the patch cuts that edge at its first column, and
        # the full render does not, which left three columns of the stroke's
        # edge different -- 96 pixels, `test_view_repair` again.
        bleed = 3
        x0 = max(0, x0 - bleed)
        y0 = max(0, y0 - bleed)
        x1 = min(width, x1 + bleed)
        y1 = min(height, y1 + bleed)
        if x1 <= x0 or y1 <= y0:
            return
        self._repair_box(x0, y0, x1, y1)

    def _repair_box(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Redraw one rectangle of the page, in page pixels.

        The whole update for a brush segment: the box the layers say changed,
        drawn by the same `views.render` that drew the page, and the zoomed
        copy repaired in step so the screen cannot keep an older answer.
        Nothing else on the page is read, recoloured or redrawn.
        """
        if self._view_image is None:
            return
        width, height = self._view_image.size
        bx0 = max(0, int(np.floor(x0)))
        by0 = max(0, int(np.floor(y0)))
        bx1 = min(width, int(np.ceil(x1)) + 1)
        by1 = min(height, int(np.ceil(y1)) + 1)
        if bx1 <= bx0 or by1 <= by0:
            return
        patch = views.edit_view(self.session, self.page_index,
                                self.session.scale,
                                matched=getattr(self, '_matched', None),
                                window=(bx0, by0, bx1 - bx0, by1 - by0))
        self._view_image.paste(patch, (bx0, by0))
        self._completed_picture = None  # content changed, unlike a camera move
        self._repair_zoomed(bx0, by0, bx1, by1)

    def _repair_zoomed(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """The same repair, on the zoomed copy -- or the screen keeps the old
        answer.

        **`_display_image` prefers `_zoomed` whenever the zoom is not 1**, and
        its cache test asks only whether the zoom still matches and whether
        the window is still covered. Neither of those changes when the brush
        decides something, so a decision repaired `_view_image`, was never
        drawn into `_zoomed`, and the screen went on showing what it showed
        before -- until the zoom changed, the page changed, or the user panned past
        the edge of the cached rectangle.

        The work was done and thrown away. From the chair that is
        indistinguishable from the brush being slow, which is what the user
        reported and what no timing of mine could find.

        Repaired rather than discarded: dropping `_zoomed` would mean a full
        re-render of the visible page for every motion event (measured 0.022s
        at 2x, 0.092s at 4x, 0.379s at 8x), turning a display bug into a real
        stall. Only the damaged rectangle is redrawn, at the zoomed scale,
        exactly as the base image's repair does.
        """
        if self._zoomed is None or self._zoomed_box is None or self.zoom == 1.0:
            return
        if getattr(self, '_zoomed_is_rough', False):
            # The cached picture is the stretched stand-in, and a sharp patch
            # is already on its way for the whole of it. Pasting one crisp
            # rectangle into a blurry page would only make the seam visible.
            return
        at_x, at_y = self._zoomed_box
        # The damaged box is in base-image pixels; the zoomed copy is the same
        # page at `self.zoom` times that, offset by its own window.
        zx0 = max(int(np.floor(x0 * self.zoom)), at_x)
        zy0 = max(int(np.floor(y0 * self.zoom)), at_y)
        zx1 = min(int(np.ceil(x1 * self.zoom)), at_x + self._zoomed.size[0])
        zy1 = min(int(np.ceil(y1 * self.zoom)), at_y + self._zoomed.size[1])
        if zx1 <= zx0 or zy1 <= zy0:
            return              # the change is outside what is cached
        try:
            # **One painter, two callers.** This used to ask `layer_patch` for a
            # page with layers and `render` for one without, and the two agree
            # only while both derive the colours the same way. They no longer
            # do: the three classes are now a page fact decided once
            # (`classification.py`) and `edit_view` samples it, while
            # `layer_patch` still colours piece by piece -- measured, 1,952
            # pixels at 2x kept the pre-gesture colour and this test caught it.
            # The patch is the same rectangle of the same view, so it is drawn
            # by the function that draws the whole of it.
            patch = views.render(self.session, self.page_index,
                                 self.view_mode,
                                 scale=self.session.scale * self.zoom,
                                 matched=getattr(self, '_matched', None),
                                 window=(zx0, zy0, zx1 - zx0, zy1 - zy0))
        except Exception:
            # A stale copy is worse than paying for a fresh one: drop it and
            # let `_display_image` rebuild.
            self._forget_zoom()
            return
        self._zoomed.paste(patch, (zx0 - at_x, zy0 - at_y))

    def _box_of(self, ref: Ref):
        bounds = self._left_bounds if ref.side == LEFT else self._right_bounds
        return bounds[ref.index] if 0 <= ref.index < len(bounds) else None

    def _compose_image(self) -> Image.Image:
        """The logical page with the brush tint on it, in the page's own
        pixels and at no zoom.

        Not the same question as what the canvas is shown, which is one
        rectangle of the page at the current zoom -- `_redraw` answers that
        one. This is the page as the decisions have left it, which is what a
        measurement wants: the brush tint is a cursor drawn mid-gesture and
        cleared on release, so after a stroke this shows the change and not
        the paint that made it.
        """
        if self._dragging and self.mask is not None and not self.mask.is_empty():
            return self._washed(self._view_image, self.mask.as_overlay_array())
        # ImageTk.PhotoImage copies pixel data at construction time, so it's
        # safe to hand out the persistent view directly here (no active
        # brush tint to composite) without copying it first.
        return self._view_image

    def _display_image(self) -> Image.Image:
        """The page at the size it is being shown, drawn and not enlarged.

        Zooming used to resize the one bitmap with NEAREST, so at 4x every
        pixel became a 4x4 block -- the staircase the user asked about. These are
        strokes: they can be drawn at any scale, and `views.render` has taken
        a scale all along.

        Kept until the zoom or the page changes, because panning must not
        pay for it. Measured, re-rendering is 0.022s at 2x, 0.092s at 4x and
        0.379s at 8x against 0.004 / 0.020 / 0.080 for the resize -- four or
        five times, once per zoom change, for the difference between reading
        the user's handwriting and reading a mosaic of it.

        Smoothing the resize instead does not work and was tried: against
        what the strokes really draw at that scale, LANCZOS is *further* away
        than NEAREST (0.83 grey levels to 0.66 at 4x). Blur does not put back
        an edge that was never in the bitmap.
        """
        if self.zoom == 1.0:
            self._zoomed_box = (0, 0)
            return self._view_image
        visible = self._visible_box()
        # What must be covered is what the user can see; what gets drawn is that
        # plus the margin. Those are two different rectangles, and confusing
        # them makes the margin buy nothing.
        if (self._zoomed is not None and self._zoomed_at == self.zoom
                and visible is not None
                and _covers(self._zoomed_box, self._zoomed.size, visible)):
            return self._zoomed
        # **The stretched picture now; the drawn one in a moment.** See
        # `_rough_at_this_zoom`: the render is 293 ms at 2x and over a second
        # at 4x, and doing it here made the zoom wait for it.
        rough = self._rough_at_this_zoom()
        if rough is not None:
            self._zoomed, self._zoomed_at = rough, self.zoom
            # **The box the piece was cut from**, not (0, 0): the rough picture is the visible rectangle now, and a
            # cached picture that says it starts at the page corner would be accepted for the wrong region.
            visible = self._visible_box()
            self._zoomed_box = ((int(round(visible[0])), int(round(visible[1])))
                                if visible is not None else (0, 0))
            self._zoomed_is_rough = True
            token = getattr(self, '_zoom_token', 0)
            if getattr(self, '_sharpen_job', None) is None:
                # A beat later, so a wheel still turning replaces this
                # request instead of queueing a render behind every tick.
                self._sharpen_job = self.root.after(
                    100, lambda t=token: self._sharpen(t))
            return rough
        try:
            # **The same lock as the worker.** A pan's render and a sharpen's render must not run over the session's
            # caches at the same time; they are 50-200 ms each, so serialising them costs nothing visible.
            drawn = views.render(self.session, self.page_index, self.view_mode,
                                 scale=self.session.scale * self.zoom,
                                 matched=getattr(self, '_matched', None),
                                 window=self._wanted_box(visible))
        except Exception:
            return self._view_image
        left, top, _w, _h = self._wanted_box(visible)
        self._zoomed, self._zoomed_at = drawn, self.zoom
        self._zoomed_box = (left, top)
        self._zoomed_is_rough = False
        return drawn

    def _page_size_at_zoom(self):
        return (max(1, round(self._view_image.width * self.zoom)),
                max(1, round(self._view_image.height * self.zoom)))

    def _the_surface_presented(self, view, picture, box):
        """Did the picture really reach the screen?

        A GL surface can fail to present without raising -- a locked screen or a dropped remote
        session leaves its clear colour, `(15, 26, 41)`, where the page is. That is the "black
        rectangle": the page's own rectangle, painted with the window's dark ground.

        One pixel of the front buffer answers it, at a point the composed picture says is **paper**.
        Only a positive answer acts: a read that fails, a surface without `read_front`, or a picture
        with no paper to compare against, all trust the surface. Returns True when it cannot tell.
        """
        frame = getattr(view, 'frame', None)
        read_front = getattr(frame, 'read_front', None)
        if read_front is None or picture is None:
            return True
        # **Nine paper points, not one.** A block the size of a paragraph passes a single-pixel test if that pixel
        # happens to be outside it, and the probe is switched off after the first success -- so one lie kept being
        # trusted. Most of these have to agree before the surface is believed.
        points = []
        for fx in (0.5, 0.12, 0.88, 0.3, 0.7):
            for fy in (0.5, 0.12, 0.88, 0.3, 0.7):
                x = int(round(picture.width * fx))
                y = int(round(picture.height * fy))
                if 0 <= x < picture.width and 0 <= y < picture.height:
                    if min(picture.getpixel((x, y))[:3]) >= 240:
                        points.append((x, y))
                if len(points) >= 9:
                    break
            if len(points) >= 9:
                break
        if not points:
            return True
        agreed = 0
        asked = 0
        for x, y in points:
            try:
                seen = read_front(x + box[0], y + box[1])
                if seen is None:
                    continue
                got = tuple(int(v) for v in seen[:3])
                want = tuple(int(v) for v in picture.getpixel((x, y))[:3])
            except Exception:
                continue
            asked += 1
            if all(abs(got[at] - want[at]) <= 8 for at in range(3)):
                agreed += 1
        if asked == 0:
            return True                 # nothing was readable; trusting it is the only honest answer
        return agreed * 2 >= asked      # most of the page is what the picture says

    def _page_origin(self):
        """Where the page's top-left sits on the canvas, in canvas pixels.

        **A page smaller than the window has no scroll range.** The canvas can only place it by
        scrolling, so without an origin it is pinned to the top-left corner: measured, a 376x782
        page at zoom 0.5 sat at (188,391) in a 2543x1352 viewport, and every zoom then appeared
        to grow out of that corner. The page is centred instead. When it is larger than the window
        the origin is (0, 0) and nothing about the old behaviour changes.
        """
        try:
            page_w, page_h = self._page_size_at_zoom()
            view_w = max(1, int(self.canvas.winfo_width()))
            view_h = max(1, int(self.canvas.winfo_height()))
        except Exception:
            return 0, 0
        return max(0, (view_w - page_w) // 2), max(0, (view_h - page_h) // 2)

    def _visible_box(self):
        """The rectangle the user can actually see, in the pixels of this zoom.

        This is what has to be covered. Asking whether the *padded* box is
        covered would make the margin worth nothing: the padded box moves
        with the window, so it leaves the drawn piece the instant the user pans a
        single pixel, and every drag would redraw.
        """
        page_w, page_h = self._page_size_at_zoom()
        try:
            view_w = max(1, int(self.canvas.winfo_width()))
            view_h = max(1, int(self.canvas.winfo_height()))
            origin_x, origin_y = self._page_origin()
            at_x = int(self.canvas.canvasx(0)) - origin_x
            at_y = int(self.canvas.canvasy(0)) - origin_y
        except Exception:
            return None
        if view_w <= 1 or view_h <= 1:
            return None
        left, top = max(0, at_x), max(0, at_y)
        right = min(page_w, at_x + view_w)
        bottom = min(page_h, at_y + view_h)
        return (left, top, max(1, right - left), max(1, bottom - top))

    def _wanted_box(self, visible=None):
        """The rectangle to draw: what the user can see, grown by the margin.

        The margin is a share of the window and not a count of pixels, so it
        means the same thing at every zoom: a drag of less than that costs
        nothing, and a longer one draws once.
        """
        page_w, page_h = self._page_size_at_zoom()
        visible = self._visible_box() if visible is None else visible
        if visible is None:
            return (0, 0, page_w, page_h)
        at_x, at_y, view_w, view_h = visible
        pad_x, pad_y = int(view_w * OVERSCAN), int(view_h * OVERSCAN)
        left = max(0, at_x - pad_x)
        top = max(0, at_y - pad_y)
        right = min(page_w, at_x + view_w + pad_x)
        bottom = min(page_h, at_y + view_h + pad_y)
        return (left, top, max(1, right - left), max(1, bottom - top))

    def _surface(self):
        """Merger presents cached pixels on Tk; GPU rendering is independent."""
        return False

    def _create_view(self):
        """Build the surface, preferring the one that draws inside the window.

        Both take the same four calls -- `place`, `place_at`, `present`,
        `close` -- so `_redraw` does not know which one it has. A module that
        cannot start is skipped; if both are impossible the answer is False and
        the processor path below draws the page.
        """
        order = []
        if os.environ.get('XP_GL_TK', '1') != '0':
            order.append('render.surface.tkgl')
        if os.environ.get('XP_GL_QT', '1') != '0':
            order.append('render.surface.glsurface')
        for name in order:
            try:
                module = __import__('xp.' + name, fromlist=['Surface', 'available'])
                if not module.available():
                    continue
                # **Paper, not the dark ground.** The merger draws pages; a frame that clears before its texture is
                # ready must be paper. 	kgl already clears to paper; this asks the Qt surface for the same, and its
                # default stays the dark ground for the windows that want it.
                extra = {'clear': (1.0, 1.0, 1.0)} if name.endswith('glsurface') else {}
                view = module.Surface(self.canvas, self.canvas.winfo_width(),
                                      self.canvas.winfo_height(), **extra)
            except Exception:
                # Anything at all: the next route, or the processor path below.
                # A display that cannot start must not stop the window.
                continue
            if name.endswith('tkgl'):
                self._bind_the_document_widget(view)
            # **Which build is drawing it.** Two rounds were spent on reports from a window launched before the fixes
            # they were about; the hash ends that, and it is in the title as well.
            print('[merge] build %s' % (_the_build() or 'unknown'), flush=True)
            # **Which surface is drawing.** A report of a black block is answered first by knowing this: `tkgl` clears
            # to paper, the Qt one takes a colour and is given paper, and a build before those cleared dark.
            try:
                print('[merge] surface %s (%s)' % (name.split('.')[-1], view.renderer), flush=True)
            except Exception:
                pass
            return view
        return False

    def _bind_the_document_widget(self, view) -> None:
        """Let the document widget answer the canvas's own bindings.

        The widget sits on top of the canvas and is a sibling, not a child, so
        Tk delivers its events to it and the canvas's ten bindings never fire.
        These are the *same bound methods*, not `event_generate`: the widget is
        pinned to the viewport's top-left corner, so its local `e.x`/`e.y` are
        already canvas-widget-local, which is what every handler below expects.
        """
        frame = view.frame
        frame.bind('<ButtonPress-1>', self._on_press)
        frame.bind('<B1-Motion>', self._on_drag)
        frame.bind('<ButtonRelease-1>', self._on_release)
        frame.bind('<ButtonPress-2>', self._scan_mark)
        frame.bind('<B2-Motion>', self._pan_and_redraw)
        frame.bind('<MouseWheel>', self._on_wheel)
        frame.bind('<Shift-MouseWheel>', self._scroll_sideways)
        frame.bind('<Control-MouseWheel>', self._on_wheel_zoom)
        # Keys are bound on the toplevel, and a click on the document lands on
        # this widget -- so it has to keep the canvas focused, exactly as the
        # canvas's own `<Enter>` does.
        frame.bind('<Enter>', lambda e: self.canvas.focus_set())

    def _scan_mark(self, event) -> None:
        self.canvas.scan_mark(event.x, event.y)

    def _scroll_vertical(self, *args) -> None:
        self.canvas.yview(*args)
        self._viewport_changed()
        self._request_redraw()

    def _scroll_horizontal(self, *args) -> None:
        self.canvas.xview(*args)
        self._viewport_changed()
        self._request_redraw()

    def _redraw(self) -> None:
        if self.session.page_count == 0:
            # **The empty state is drawn here too, or it does not last.**
            # `_load_page` puts it up, and the `<Configure>` that follows the
            # first layout calls this, which would wipe it and leave a blank
            # canvas with nothing said on it.
            self._show_the_empty_state()
            return
        # **Nothing to draw yet is not an error.** Before the first page is built `_view_image` is
        # None, and `_display_image` reads it at zoom 1; a `<Configure>` that arrives first used to
        # raise inside Tk's callback. Leaving the screen as it is beats blanking it.
        if getattr(self, '_view_image', None) is None:
            return
        # A surface that did not present last time is left out of this frame; the zoom and the page
        # turn put it back in the running (see `_the_surface_presented`).
        # The merger presents cached images through one persistent Canvas item.
        # GPU rasterisation remains in views; a GL child window is not needed
        # for image presentation and can lose its front buffer on resize.
        view = None
        img = self._display_image()
        # The brush's alpha for the surface to blend, when the surface is the
        # one showing the page. None means "the picture below is already
        # finished, show it as it is".
        wash_alpha = None
        # One decision, read once: who applies the brush's wash. It used to be
        # asked twice, two lines apart, with two different defaults for
        # `XP_GL_ONLY` -- so with nothing set the host washed the picture *and*
        # the surface was told the picture was finished, and the round trip
        # `present` says it removed was still being made.
        on_the_card = bool(view) and os.environ.get('XP_GL_ONLY', '1') != '0'
        if self._dragging and self.mask is not None and not self.mask.is_empty():
            # The brush tint, over whichever picture is being shown. The mask
            # is in the logical space, so at a zoom it is enlarged -- with
            # NEAREST, on purpose: this is a cursor showing where the brush
            # is, and a blocky cursor is a cursor. Re-rendering a mask to
            # soften the edge of a brush would be work spent on the wrong
            # thing.
            # A 256-entry table, not a lambda. `point(lambda v: ...)` calls a
            # Python function once PER PIXEL, and the mask is the whole page --
            # four and a half million pixels, tens of millions of interpreter
            # calls a second while a stroke is running. Measured at 25.3ms a
            # frame. The table is exactly the same arithmetic, truncated the
            # same way, done once in C: same pixels, and the cost stops
            # depending on how big the page is.
            tint = [int(v * BRUSH_TINT_ALPHA) for v in range(256)]
            # Crop FIRST, while this is still a numpy array. The mask is the
            # whole page -- four and a half million pixels -- and building a
            # PIL image out of all of it only to throw most of it away is what
            # made this cost 23.5ms a frame. Slicing is cheap; the conversion
            # and the table application are not, so they only ever see the
            # window now.
            overlay = self.mask.as_overlay_array()
            if overlay.shape[1] != img.width or overlay.shape[0] != img.height:
                # The picture is a rectangle of the page, not the page: crop to
                # the piece being shown before enlarging. Scaling the whole
                # page's mask into one window's worth of pixels would put the
                # brush somewhere the user did not paint.
                at_x, at_y = self._zoomed_box or (0, 0)
                x0 = at_x / self.zoom
                y0 = at_y / self.zoom
                x1 = x0 + img.width / self.zoom
                y1 = y0 + img.height / self.zoom
                top = max(0, int(y0))
                bottom = max(0, int(round(y1)))
                left = max(0, int(x0))
                right = max(0, int(round(x1)))
                overlay = overlay[top:bottom, left:right]
            alpha = Image.fromarray(overlay).point(tint)
            if alpha.size != img.size:
                alpha = alpha.resize(img.size, Image.NEAREST)
            # The alpha is already the window's, so the wash is handed it
            # rather than the page's mask: the crop above is PIL's and stays
            # PIL's, because it is a cursor being put in the right place, not
            # a drawing being made.
            if on_the_card:
                # The surface blends this into the texture, on the card. The
                # alpha is the window's, not the page's -- `wash` needs it to
                # match the frame or the wash lands in the wrong place, quietly.
                wash_alpha = np.asarray(alpha)
            else:
                img = self._washed(img, np.asarray(alpha))
        # The picture, and the one canvas item that shows it. Both are reused:
        # building a `PhotoImage` and a canvas item per frame is most of what a
        # zoom costs, and the cost was measured rather than assumed --
        #
        #     zoom   render   upload   present   total
        #     1.0x   0.0ms    13.5ms   28.7ms    42.2ms   24fps
        #     2.0x   0.0ms     5.7ms   12.5ms    18.2ms   55fps
        #
        # where `render` is the drawing (already on the card) and `upload` is
        # `PhotoImage(...)` and `present` is Tk's redraw. The drawing is not
        # what costs; the two lines that carry it to the screen are.
        page_w, page_h = self._page_size_at_zoom()
        at_x, at_y = self._zoomed_box or (0, 0)
        origin_x, origin_y = self._page_origin()
        # At least the window: a page smaller than it must still have somewhere to sit, or the
        # scroll clamps to 0 and the page stays in the corner (`_page_origin` says why).
        self.canvas.configure(scrollregion=(0, 0, max(page_w, self.canvas.winfo_width()),
                                            max(page_h, self.canvas.winfo_height())))
        # The silver page edge. Two ways to show it, because there are two ways
        # to show a page: canvas items when the canvas is the picture, pixels in
        # the frame when a surface is (a child window always covers the items).
        self._page_edge_items(page_w, page_h, show=not view)
        self._frame_guide_items(page_w, page_h, show=not view)
        # The surface, when this window has one. `XP_GL=0` turns it off and the
        # processor path below draws the page; a surface that cannot start is
        # `view = False` and does the same. Both are decided once, in
        # `_surface()`, and the wash above already knows the answer.
        if view:
            view.place(self.canvas)
            # `convert('RGB')` copies the whole window, and the picture the
            # window composes is already RGB, so it was a copy to satisfy a
            # mode that was already right. `np.asarray` on an RGB image is a
            # view, not a copy -- so this is one copy fewer per frame.
            # Only the visible rectangle goes to the card. At the
            # fitted zoom `img` is the whole page -- 1600x2240 --
            # and the window is 1153x498, so every frame was
            # uploading 6.3x more image than could be seen. That was
            # the stall: 19.5ms a frame against 3.3ms for the
            # window's own pixels, measured.
            vx = int(self.canvas.canvasx(0))
            vy = int(self.canvas.canvasy(0))
            cw = max(1, self.canvas.winfo_width())
            ch = max(1, self.canvas.winfo_height())
            left = max(0, vx - at_x)
            top = max(0, vy - at_y)
            box = (left, top, min(img.width, left + cw),
                   min(img.height, top + ch))
            frame = img.crop(box)
            # The edge goes into the pixels the surface is about to show: a
            # canvas item would be painted *under* the surface, which is a
            # child window and always wins. The crop is already a copy, so
            # this costs a draw and not a second copy of the page.
            frame = page_edges(frame, at_x + box[0], at_y + box[1],
                               page_w, page_h)
            guide = self._frame_guide_box(page_w, page_h)
            if guide is not None:
                # **On the picture's own grid** (`_on_the_pictures_grid`): one rounding for both ends
                # of the subtraction, so the guide's place on the screen does not move by a pixel as
                # the zoom changes.
                picture_x = origin_x + at_x + box[0]
                picture_y = origin_y + at_y + box[1]
                dash = the_guide_dash(self.zoom)
                frame = page_frame_guide(
                    frame,
                    (_on_the_pictures_grid(origin_x + guide[0], picture_x),
                     _on_the_pictures_grid(origin_y + guide[1], picture_y),
                     _on_the_pictures_grid(origin_x + guide[2], picture_x),
                     _on_the_pictures_grid(origin_y + guide[3], picture_y)),
                    dash=dash)
            if wash_alpha is not None:
                # The alpha has to be cut the same way or the wash
                # lands somewhere the user did not paint.
                wash_alpha = wash_alpha[box[1]:box[3], box[0]:box[2]]
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            # **The picture first, the widget second.** `place_at` maps the window, and a mapped
            # `tkgl` frame draws itself: `redraw` clears and only then draws a texture
            # `if self.rgb is not None`. Placing first left a moment with nothing to draw -- which
            # used to be a dark rectangle the size of the page, and is paper now (the clear colour
            # is white since 2026-10-30). Presenting first is still the right order: paper is
            # better than a dark hole, and the picture is better than paper.
            pixels = np.asarray(frame)
            tint = BRUSH_TINT[self.active_color] if wash_alpha is not None else None
            view.present(pixels, wash_alpha, tint)
            # The surface covers exactly the frame, so the texture and
            # the widget are the same shape and nothing is stretched.
            # **Only when the rectangle is not the one it already has.** Resizing the item and the widget on every
            # frame, and flushing Tk after it, is a synchronous round trip a drag cannot afford: measured, the
            # input-to-picture P50 at 4x went from under 55 ms to 71.4 with it on every frame.
            rectangle = (origin_x + at_x + box[0], origin_y + at_y + box[1],
                         frame.width, frame.height)
            if getattr(self, '_surface_rectangle', None) != rectangle:
                self._surface_rectangle = rectangle
                view.place_at(self.canvas, rectangle[0], rectangle[1],
                              rectangle[2], rectangle[3])
                # **The resize paints over the picture that was just put up.** `place_at` resizes the GL child window,
                # and the `<Configure>` that follows moves the viewport and draws nothing; `pyopengltk` sets the
                # widget's own background to '' so Tk never paints it either. A surface that has just changed size
                # therefore shows whatever the window system leaves in that buffer -- a black page -- until something
                # draws into it. That is the user's "zoom out and the whole page goes black for a moment", and it is why it
                # only happens once the page is smaller than the window: until then the surface's rectangle is the
                # canvas and does not change size at all.
                #
                # Presenting again after the resize is the whole fix: the last thing that happens is a draw at the
                # size the widget now has. `_display` drains the idletasks queue first, so the viewport is already
                # the new one by the time the texture goes up -- this is not two pictures, it is one picture drawn
                # again in the rectangle it now occupies.
                view.present(pixels, wash_alpha, tint)
            # **The flush stays unconditional.** It is what makes the frame that was just presented visible; skipping it
            # when the rectangle had not changed left the surface showing the previous frame (measured: 9,681 stale
            # pixels at 2x after a gesture). Only the resize is conditional.
            try:
                self.canvas.update_idletasks()
            except Exception:
                pass
            # **Did it really reach the screen?** If not, the surface is taken off the canvas and the
            # processor path below draws this frame; the dark ground is never left where the page is.
            if (on_the_card and self._probe_the_surface
                    and not self._the_surface_presented(view, frame, box)):
                self._surface_lost = True
                try:
                    self.canvas.itemconfigure(view.item, state='hidden')
                except Exception:
                    pass
            # `XP_GL_ONLY=0` draws both, which is a diagnosis and not the
            # design: it tells apart "the surface eats the pointer" from "the
            # test needs the canvas to have been drawn". With the return in
            # place the canvas holds no picture at all, which is what a real
            # GPU display does -- the card *is* the showing.
            if on_the_card:
                self._probe_the_surface = False
            if on_the_card and not getattr(self, '_surface_lost', False):
                return
        if self._photo is not None and self._photo.width() == img.width \
                and self._photo.height() == img.height:
            # Same size: paste the new pixels into the image Tk already holds,
            # instead of making it a new one and destroying the old.
            self._photo.paste(img)
            if self._photo_item is None:
                self._photo_item = self.canvas.create_image(
                    origin_x + at_x, origin_y + at_y, anchor="nw", image=self._photo)
            else:
                self.canvas.coords(self._photo_item, origin_x + at_x, origin_y + at_y)
            self.canvas.tag_raise(EDGE_TAG)
            self.canvas.tag_raise(FRAME_TAG)
            return
        replacement = ImageTk.PhotoImage(img)
        if self._photo_item is None:
            self._photo_item = self.canvas.create_image(
                origin_x + at_x, origin_y + at_y, anchor="nw", image=replacement)
        else:
            self.canvas.itemconfigure(self._photo_item, image=replacement)
            self.canvas.coords(self._photo_item, origin_x + at_x, origin_y + at_y)
        self._photo = replacement  # keep old pixels alive until the item references the new image
        self.canvas.tag_raise(EDGE_TAG)

    def _page_edge_items(self, page_w, page_h, show: bool) -> None:
        """The silver edge for the path where the canvas itself is the picture.

        Items, because the picture the canvas holds is the page's own cached
        bitmap: painting the line into it would either bake the line into the
        page (and then into the next frame, and into the saved file's preview)
        or cost a full copy of a 1600x2240 image on every frame. Items scroll
        with the page, are clipped by the canvas, and sit above the picture.

        Also removed whenever a surface is showing the page: a child window
        covers items, so leaving them there would be a line that appears and
        disappears for no reason.
        """
        origin_x, origin_y = self._page_origin()
        ox, oy = origin_x, origin_y
        coords = [(ox, oy, ox, oy + page_h), (ox + page_w, oy, ox + page_w, oy + page_h),
                  (ox, oy, ox + page_w, oy), (ox, oy + page_h, ox + page_w, oy + page_h)]
        items = getattr(self, '_edge_items', None) or []
        if not show:
            if items:
                self.canvas.delete(EDGE_TAG)
                self._edge_items = []
            return
        if len(items) == 4:
            for item, box in zip(items, coords):
                self.canvas.coords(item, *box)
            return
        self.canvas.delete(EDGE_TAG)
        self._edge_items = [
            self.canvas.create_line(*box, fill='#c0c0c0', width=1,
                                    tags=(EDGE_TAG, 'gui-aid'))
            for box in coords]

    def _frame_guide_box(self, page_w, page_h):
        """The Q base reference on this page, in **this zoom's page pixels**.

        `session.frame_box` is the calibrated position: the file's frame placed on the merged page by
        the same fit the ink uses, so the dashes move with zoom and agree with the writing rather than
        with the template in the abstract.

        **No canvas origin.** This is the space the picture is cropped in: the surface path subtracts
        the crop's own origin from these numbers and paints the dashes into the pixels, and the canvas
        path adds the page's origin when it makes its items. Adding it here as well put the guide one
        origin away from the page -- on the surface path almost all of it fell outside the crop and
        what survived was a single dashed line down the middle.
        """
        try:
            page_units_w, _page_units_h = self.session.page_size(self.page_index)
            if page_units_w <= 0:
                return None
            box = self.session.frame_box(self.page_index)
        except Exception:
            return None
        k = page_w / page_units_w
        return (box[0] * k, box[1] * k, box[2] * k, box[3] * k)

    def _frame_guide_items(self, page_w, page_h, show: bool) -> None:
        """The dashed Q base reference, for the canvas path."""
        box = self._frame_guide_box(page_w, page_h)
        dash = the_guide_dash(self.zoom)
        items = getattr(self, '_guide_items', None) or []
        if not show or box is None:
            if items:
                self.canvas.delete(FRAME_TAG)
                self._guide_items = []
            return
        # The box is in the page's own pixels; the canvas draws it where the page is.
        origin_x, origin_y = self._page_origin()
        # The box is in the page's own pixels; the canvas draws it where the page is, with the dash
        # the zoom asks for -- the same rule the surface path paints into its picture.
        origin_x, origin_y = self._page_origin()
        segments = [(x0 + origin_x, y0 + origin_y, x1 + origin_x, y1 + origin_y)
                    for x0, y0, x1, y1 in _guide_dashes(box, dash=dash)]
        if items and len(items) == len(segments):
            for item, (x0, y0, x1, y1) in zip(items, segments):
                self.canvas.coords(item, x0, y0, x1, y1)
                self.canvas.itemconfigure(item, dash=dash)
            return
        self.canvas.delete(FRAME_TAG)
        self._guide_items = [
            self.canvas.create_line(x0, y0, x1, y1, fill='#969696', width=1,
                                    dash=dash, tags=(FRAME_TAG, 'gui-aid'))
            for x0, y0, x1, y1 in segments]

    def _request_redraw(self) -> None:
        """Coalesces redraws to ~60fps instead of doing one full compose per
        mouse-motion event, which can fire faster than that."""
        if self._redraw_pending:
            return
        self._redraw_pending = True

        def fire():
            self._redraw_pending = False
            self._redraw()

        self.root.after(REDRAW_INTERVAL_MS, fire)

    # ---------------- mouse / brush ----------------

    def _canvas_xy(self, event) -> Tuple[float, float]:
        """Canvas coordinates, converted back through the current zoom into
        the fixed logical pixel space the mask/strokes are defined in.

        The page's origin is taken off first: it is where the page sits on the canvas, and a
        click on the paper has to be measured from the paper, not from the corner of the window.
        """
        origin_x, origin_y = self._page_origin()
        return ((self.canvas.canvasx(event.x) - origin_x) / self.zoom,
                (self.canvas.canvasy(event.y) - origin_y) / self.zoom)

    def _by_layers(self) -> bool:
        """Whether this page composes from layers of pieces (`layers.py`)."""
        return (self.view_mode == views.EDIT
                and self.session.uses_layers(self.page_index))

    def _on_press(self, event) -> None:
        if self.locked():
            return
        x, y = self._canvas_xy(event)
        self._dragging = True
        self._drag_last = (x, y)
        self._drag_path = [(x, y)]
        self.mask.stamp(x, y, x, y)
        # Everything captured until _on_release groups into one undo step,
        # so Ctrl+Z undoes "that stroke", not one motion-event's sliver of it.
        self.session.begin_action(self.page_index)
        # One snapshot for the whole drag, taken here and not per motion event.
        self.session.begin_gesture(self.page_index)
        if self._by_layers():
            self._paint_segment(x, y, x, y)
        else:
            self._capture_and_redraw()

    def _on_drag(self, event) -> None:
        if self.locked() or self._drag_last is None:
            return
        x, y = self._canvas_xy(event)
        x0, y0 = self._drag_last
        self.mask.stamp(x0, y0, x, y)
        self._drag_last = (x, y)
        self._drag_path.append((x, y))
        if self._by_layers():
            self._paint_segment(x0, y0, x, y)
        else:
            self._capture_and_redraw()

    def _paint_segment(self, x0, y0, x1, y1) -> None:
        """This segment of the drag, applied now, and only here repainted.

        The decision lands as the brush moves -- not at mouse-up, not once a
        whole stroke has been covered, and with nothing carried over between
        gestures. What comes back is the box that actually changed, which is
        usually a few dozen pixels, so the repaint is that box and the page
        is neither rescanned nor redrawn.
        """
        box = self.session.brush_segment(self.page_index, x0, y0, x1, y1,
                                         self.mask.radius, self.active_color)
        if box is None:
            # Nothing changed: the same place in the same state. The brush
            # tint still has to follow the mouse, though.
            self._request_redraw()
            return
        self._repair_box(*box)
        self._request_redraw()

    def _on_release(self, _event) -> None:
        if self.locked():
            return
        # If the drag ended back near where it started, treat it as a drawn
        # loop: fill its interior into the mask too, so strokes fully
        # enclosed by the loop are selected even where the brush's own
        # (fixed-width) path never actually touched them.
        if len(self._drag_path) >= 3 and not self._by_layers():
            # The loop fill belongs to the area protocol, which is what a
            # picture side still uses. A layered page has already applied
            # every segment as it arrived, so there is nothing left here to
            # decide -- and asking again at mouse-up is exactly the waiting
            # this change exists to remove.
            x0, y0 = self._drag_path[0]
            x1, y1 = self._drag_path[-1]
            if ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 <= self.mask.radius * LOOP_CLOSE_FACTOR:
                self.mask.fill_polygon(self._drag_path)
                self._apply_changes(
                    self.session.brush_capture(self.page_index, self.mask, self.active_color))

        # **The gesture's polygon goes into the region map here on a layered page.** A stroke page
        # applies each drag segment as it arrives and never calls `brush_capture`, so this is the only
        # place its gesture becomes geometry; a picture page goes through `brush_capture`, which
        # writes the same thing itself, and doing it twice would be two polygons for one drag.
        if self._by_layers():
            try:
                self.session.capture_region(self.page_index, self.mask, self.active_color)
            except Exception as exc:                  # a decision is not lost to a shape
                print('[merge] the gesture outline could not be taken: %s: %s'
                      % (type(exc).__name__, exc), flush=True)

        self.session.end_action(self.page_index)
        self.session.end_gesture(self.page_index)
        self._dragging = False
        self._last_drag_end = time.monotonic()   # the sharpen waits on this
        self._drag_last = None
        self._drag_path = []

        # The paint is a cursor, not a result: wipe it once the gesture is
        # over so what's left on screen is the page as it now stands. Leaving
        # it up meant the colored wash sat on top of the very change you
        # just made, which looked like nothing had happened.
        self.mask.clear()
        self._redraw()

    def _capture_and_redraw(self) -> None:
        self._apply_changes(
            self.session.brush_capture(self.page_index, self.mask, self.active_color))
        self._request_redraw()

    def default_output(self) -> str:
        """Where Ctrl+S writes: the launcher's name, or the default one.

        `<left document>_merged.xopp`, beside the left document -- the name
        the chooser has always offered. The arranger's Ctrl+S already works
        this way (`quick_save`: "write the default name without asking"), and
        the three tools answer the same keys the same way.
        """
        if self.output:
            return self.output
        beside = Path(self.session.left_side.path)
        return str(beside.parent / (beside.stem + '_merged.xopp'))

    def commit(self) -> None:
        """Ctrl+S: write the default name, without asking, and close.

        No dialog. The user pressed Save, and a save key that stops to ask is not
        a save key -- the arranger's Ctrl+S has never asked, and this one
        asked every time. `commit_as` is the one that asks, prefilled, and
        that split is the whole difference between the two.

        The window closes once the file is on disk (`exit_after_save`),
        which is also what the arranger does: the merge is finished, and
        leaving it open means the next thing the user does is close it.

        If the brush took ink that is not strokes yet, convert it
        first -- in the converter, not in a window of its own.

        This is the deferred conversion, and it is where it belongs: by now
        the brush has said what the result needs, so there is something to
        ask the converter about. Opening the merge asked it nothing, which
        is why a PDF side costs seconds to look at instead of twenty minutes.

        The window that opens is `convertgui.ConversionWindow` -- the same
        one, with the same stages, the same settings and the same Autotune.
        The only difference is its last button: it hands the settings back
        instead of writing a document, because where this result is going
        was decided before it opened.
        """
        if self.read_only or getattr(self, '_saving', False):
            # A second Ctrl+S while the first is still writing would build the
            # document twice and race on the file. `commit_as` has always had
            # this guard; this one did not, so the second press fell through
            # to a writer that refused it without saying so.
            #
            # `getattr`, because `tests/test_selection_is_respected.py` drives
            # this method on a deliberately bare object -- three attributes
            # and nothing else -- to check which of its three branches runs.
            # That is a fair way to test a branch, and it should not have to
            # know which flags the branch happens to read.
            return
        decided = self._convert_then_save() if self._needs_converting() else 'nothing'
        if decided == 'cancelled':
            # The user said no to converting. Writing now would save a merge that is
            # missing the ink the user selected -- quietly, and looking finished.
            return
        if decided == 'converted':
            return              # the converter is open; it finishes the save
        self._save_now()

    def save_as_name(self):
        """Ask where, prefilled with the default name. None if the user cancels.

        **A dialog of our own, because of the tick.** `asksaveasfilename` is the
        system's and cannot hold one, and the user's order puts *Save as Learning
        Material* in the save: checked by default, archived in the background.
        So the name and the tick are asked together, and a cancel still costs
        nothing -- nothing has been written when this returns None.
        """
        suggested = Path(self.default_output())
        chosen = filedialog.asksaveasfilename(
            parent=self.root, title='Save merged notebook',
            initialdir=str(suggested.parent), initialfile=suggested.name,
            defaultextension='.xopp',
            filetypes=[('Xournal++ notebook', '*.xopp')])
        if not chosen:
            return None             # a pause, not a loss
        if Path(chosen).resolve() in self._source_paths():
            messagebox.showerror(
                'Choose a new notebook',
                'Save to a separate .xopp file to preserve both originals.',
                parent=self.root)
            return None
        if Path(chosen).suffix.lower() != '.xopp':
            messagebox.showerror(
                'Choose a new notebook',
                'A merge is a Xournal++ notebook, so it is saved as .xopp.',
                parent=self.root)
            return None
        return chosen

    def _needs_converting(self) -> bool:
        pictures = getattr(self.session, 'picture_sides', lambda: [])()
        if not pictures:
            return False
        for page_index in range(self.session.page_count):
            decisions = self.session.decisions(page_index)
            for side in pictures:
                taken = decisions.area_taken(side)
                if taken is not None and taken.any():
                    return True
        return False

    def _taken_pages(self, side: str):
        """The 1-based source pages the brush took from on this side."""
        out = []
        for page_index in range(self.session.page_count):
            taken = self.session.decisions(page_index).area_taken(side)
            if taken is None or not taken.any():
                continue
            page = self.session.page_at(page_index)
            source = page.left if side == LEFT else page.right
            if source is not None:
                out.append(source + 1)
        return out

    def _convert_then_save(self) -> str:
        """What converting first decided: `'nothing'`, `'cancelled'`, `'converted'`.

        Three answers and not a boolean, because they mean different things and
        a boolean made two of them look the same. Cancelling the question below
        returned False, which is also what "there is nothing to convert"
        returns -- and `commit` read False as "carry on", so declining to
        convert **saved the merge without the ink the user had just selected**. It
        wrote a file that looked finished and was missing the thing the user asked
        for, and said nothing.
        """
        from xp.workers.converter.gui.convertgui import ConversionWindow

        pictures = [side for side in self.session.picture_sides()
                    if self._taken_pages(side)]
        if not pictures:
            return 'nothing'
        side = pictures[0]
        held = (self.session.left_side if side == LEFT
                else self.session.right_side)
        pages = self._taken_pages(side)
        if not messagebox.askokcancel(
                'Convert what you took',
                f'You took ink from {held.path.name}, which is a PDF '
                f'-- it has to be traced before it can go in.'
                + chr(10) + chr(10) +
                f'{len(pages)} page(s) to convert. The converter '
                f'opens next, with its stages and settings; the '
                f'merge is saved when you are happy with them.',
                parent=self.root):
            return 'cancelled'

        def settled(profile):
            # Not converted here. `convert_everything_taken` traces every taken
            # page, which is the minutes-long part, and doing it on this thread
            # is what kept the window frozen with no panel to look at -- the
            # save that followed got its panel, the tracing never did. The
            # profile travels with the call; the worker does both.
            self._save_now(profile=profile)

        # `live`, not `auto`: opening in auto starts an autotune, and the user
        # would sit through it before the button the user came for went live. The user
        # said Autotune is one of the things the user *can* do here, so it stays a
        # button rather than a toll.
        ConversionWindow(held.path, master=self.root, profile=None,
                         mode='live', settle=settled, pages=pages)
        return 'converted'

    def _source_paths(self) -> List[Path]:
        """The two documents that are open. A merge never writes over one."""
        out = []
        for side in (self.session.left_side, self.session.right_side):
            path = getattr(side, 'path', None)
            if path is not None:
                out.append(Path(path).resolve())
        return out

    def wait_for_save(self, seconds: float = 300.0) -> bool:
        """Pump Tk until the background save has finished. True if it did.

        The save moved onto a worker so the window keeps painting, which
        means a caller that wants the file on disk has to wait for it. The
        GUI never calls this -- it is for tests and for anything driving the
        window without a person in front of it.
        """
        import time as _time
        job = self._save_job
        if job is None:
            return True
        deadline = _time.monotonic() + seconds
        while not job.finished and _time.monotonic() < deadline:
            self.root.update()
            _time.sleep(0.01)
        return bool(job.finished)

    SHIFT_HELD = 0x0001

    def _save_key(self, event=None):
        """Ctrl+S commits; Ctrl+Shift+S asks where first. Read from `state`."""
        if getattr(event, 'state', 0) & self.SHIFT_HELD:
            self.commit_as()
        else:
            self.commit()
        return 'break'

    def _undo_key(self, event=None):
        """Ctrl+Z undoes; Ctrl+Shift+Z redoes. Read from `state`."""
        if getattr(event, 'state', 0) & self.SHIFT_HELD:
            self._start_repeat(self._redo)
        else:
            self._start_repeat(self._undo)
        return 'break'

    def commit_as(self) -> None:
        """Save As: ask where, even if this merge already has a home.

        Ctrl+S writes the default name and closes; this is the one that
        asks, prefilled with that same name. It is the split the arranger
        already has (`quick_save` and `save_as`), and it closes too: the
        difference between the two keys is the name, not what happens after.
        """
        if self.read_only or self._saving:
            return
        chosen = self.save_as_name()
        if chosen is None:
            return
        self.output = chosen
        self.commit()

    def _save_now(self, profile=None) -> None:
        """Write the merge. Where is already settled by the time this runs.

        Nothing asks here any more. `commit` uses the default name and
        `commit_as` has already asked, so this function's one job is the
        write -- which is the only way the cancellation can stay a *pause*:
        a dialog answered before anything is written means a cancel costs
        the user nothing, and the window, the decisions and any conversion
        already done all survive it.

        Nothing asks before this point either. Opening the window asked
        nothing and brushing asked nothing, because "where does this go" has
        no answer the user can give until there is a merged document in hand.
        """
        target = self.output or self.default_output()
        if Path(target).resolve() in self._source_paths():
            # The default name should never be a source, but a launcher can
            # name anything, and writing over one of the user's two originals is
            # the one mistake there is no undo for.
            messagebox.showerror(
                'Choose a new notebook',
                'Save to a separate .xopp file to preserve both originals.',
                parent=self.root)
            return
        self._write_in_the_background(target, profile)

    def _write_in_the_background(self, target, profile=None) -> None:
        """Build and write the merge on a worker, with the panel on screen.

        This used to be one synchronous call. On a dense notebook it is
        minutes -- every page of the merge is built from its decisions, and a
        PDF side's kept ink is traced first -- and for all of it the window was
        frozen with nothing to look at but the mouse pointer. Tk was not
        painting, so it could not have shown anything even if it had wanted to.

        What the panel reports is what the work really is, step by step:
        tracing the ink taken from a picture side (`page n of m`, from the
        converter's own callback), then building the pages (`page n of m`,
        from `export`), then the write itself, which is one call into gzip and
        can only say that it is happening. Nothing here estimates.
        """
        if self._saving:
            # A second Ctrl+S while the first is still writing would run two
            # writers at the same file. The panel is already up saying so.
            return
        self._saving = True
        session = self.session

        def work(say):
            pictures = [side for side in session.picture_sides()
                        if self._taken_pages(side)]
            if pictures:
                def traced(page_index, side):
                    say(page_index, session.page_count,
                        'tracing the ink taken from the %s page' % side)
                session.convert_everything_taken(profile=profile, progress=traced)
            # **One build, not two.** `session.commit()` builds the whole
            # merged document and this line threw it away; `save()` then built
            # it again, page by page for a big document -- which for a big
            # document meant paying for the full build *and* the page-by-page
            # one. `save()` is the only path that produces the file.
            session.save(target, progress=lambda done, total:
                         say(done, total, 'building the merged pages'))
            return target

        def done(_result):
            self._saving = False
            self.output = target
            # The launcher's report, on the Tk thread and only once the file
            # is really on disk. It used to be the writer; now it is the
            # receipt.
            if self.on_commit is not None:
                self._write(self.on_commit)
            self._task_done(True)
            self._keep_learning_material(session, target)
            if self.exit_after_save:
                # Save and close, both keys. The merge is finished; leaving
                # the window up means the next thing the user does is close it.
                # The arranger has worked this way all along
                # (`exit_after_save` there too), and a test that wants to
                # look at the window afterwards turns it off.
                self.progress.hide()
                self.root.destroy()
                return
            messagebox.showinfo("XP", "Merged and saved.", parent=self.root)

        def failed(error):
            self._saving = False
            self._task_done(False)

        from xp.render.widgets.progress import Job
        title = ('Tracing and saving' if self._needs_converting()
                 else 'Saving the merge')
        self._save_job = Job(self.root, self.progress, title, work,
            on_done=done, on_fail=failed, maximum=max(1, self.session.page_count),
            details=('worker: MergeSession.convert_everything_taken -> '
                     'MergeSession.save\ntarget: %s' % target)).start()

    def _keep_learning_material(self, session, target) -> None:
        """Archive the sample, after the file, on a thread, and never hurt it.

        The user's order: archived **in the background**, and a failed archive must not
        damage the saved file. Both hold here: this is called after the write
        returned, the copy runs off the Tk thread, and whatever it says is a
        line in the status -- nothing is rewritten.
        """
        try:
            wanted = bool(self.save_learning.get())
        except Exception:
            wanted = False
        if not wanted:
            return
        from xp.workers.merger import archive

        left = getattr(self.session.left_side, 'path', None)
        right = getattr(self.session.right_side, 'path', None)
        if not left or not right:
            return

        def said(where, error):
            def tell():
                try:
                    if error is not None:
                        self.page_label.config(
                            text='Saved. Learning material was not kept: %s'
                                 % (error,))
                    else:
                        self.page_label.config(
                            text='Saved. Learning material in %s' % (where,))
                except Exception:
                    pass
            try:
                self.root.after(0, tell)
            except Exception:
                pass

        archive.archive_in_the_background(session, left, right, target,
                                          on_done=said)

    def _write(self, write) -> bool:
        """Do the write, and tell the launcher how it went.

        A save that raises is the one thing a launcher must not be told went
        well: the merge is still on screen, the reason is in the message box
        and then in the console, and the launcher has to stay so it can be
        read. A failure does **not** spend the one report -- a second,
        successful save still tells the launcher the task is done, which is what
        the user would do next.
        """
        try:
            write()
        except Exception as exc:
            self._task_done(False)
            messagebox.showerror("XP", "Could not save that:\n%s" % (exc,),
                                 parent=self.root)
            return False
        self._task_done(True)
        return True

    def _task_done(self, clean: bool) -> None:
        """Tell whoever launched this window that it has finished. Once only.

        Once, because a save that succeeds and then a window that closes are
        one thing and not two: a launcher counting them as two would close
        itself in the middle of a job. This is also how a window opened from its
        own command line behaves exactly as before -- `on_finish` is None and
        nothing is told to anybody.

        A **failure report does not spend the one report**: the launcher is told
        to stay, and a second, successful save still tells it the task is done,
        which is what the user would do next.
        """
        if clean:
            done, self.on_finish = self.on_finish, None
            if done is not None:
                done(True)
        elif self.on_finish is not None:
            self.on_finish(False)

    def _dropped_pending_work(self, event=None) -> None:
        """Let go of work queued for a window that is going away."""
        if event is not None and event.widget is not self.root:
            return              # a child widget, not the window itself
        try:
            self._forget_zoom()
        except Exception:
            pass
        # **Every job, not only the sharpen.** `_forget_zoom` cancels the queued sharpen; the 30 ms
        # collector re-arms itself and was left behind, so Tcl printed
        # `invalid command name "..._collect_the_sharp_picture"` into whatever printed next -- a
        # destroyed window was still polling for a picture it could no longer show.
        for name in ('_sharpen_job', '_collect_job', '_repeat_job'):
            job = getattr(self, name, None)
            if job is None:
                continue
            try:
                self.root.after_cancel(job)
            except Exception:
                pass
            setattr(self, name, None)

    def _closed_by_user(self) -> None:
        """The window is going away because the user closed it.

        Looking at two documents and closing them is a *finished* task, not an
        abandoned one, so this reports clean: the launcher's job is over, and
        staying behind would be the orphan the user complained about. A save that
        failed never reaches here as clean, because `_save_now` reports first
        and reports badly.
        """
        self._task_done(True)
        # A render queued for a zoom outlives the window otherwise, and Tcl
        # prints `invalid command name` for the callback it can no longer
        # reach. Nothing is broken by that, but it is noise from a job that
        # has no page left to draw on.
        self._forget_zoom()
        self.root.destroy()


def run_review(session: MergeSession, read_only: bool = False,
               on_commit: Optional[Callable[[], None]] = None,
               master: Optional[tk.Misc] = None,
               on_swap: Optional[Callable[[], None]] = None,
               on_finish: Optional[Callable[[bool], None]] = None,
               output: Optional[str] = None,
               cue: Optional['MergeCue'] = None) -> "MergeApp":
    app = MergeApp(session, read_only=read_only, on_commit=on_commit,
                   master=master, on_swap=on_swap, on_finish=on_finish,
                   output=output, cue=cue)
    app.run()
    return app
