"""Manual page mapping: the two documents' pages in two columns, with a line
between each matched pair.

The automatic alignment (pagealign.py) is a guess. When it guesses wrong --
or when you simply disagree with it -- this is where you say so: click a
page on either side, then its counterpart on the other, and they're linked;
click an existing link to break it, or click a picked page again to drop it.
Unlinked pages are the ones with no counterpart.

Each row shows a thumbnail of the page and its ink coverage, so you can tell
pages apart without opening them.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageTk

from xp.render.appearance import ABYSS, INK, MUTED, PALE, decorate, font
from xp.workers.merger.matching.pagealign import Alignment, align_pages
from xp.render.drawing import render_page
from xp.core.document.xopp_io import XoppDocument

ROW_HEIGHT = 74
THUMB_W, THUMB_H = 44, 60
COLUMN_X = {"left": 150, "right": 470}
# On the ground, not on white: the link was a mid blue chosen against a
# white canvas and it sank into the dark one. These are the panel's own inks.
LINK_COLOR = INK
PICK_COLOR = "#ff7a6b"       # the pick still has to be the one warm thing
FRAME_COLOR = "#2b3d4e"      # a page's edge, a shade up from the ground


class PageMapApp:
    def __init__(self, left: XoppDocument, right: XoppDocument,
                 alignment: Optional[Alignment] = None,
                 on_done: Optional[Callable[[Dict[int, int]], None]] = None,
                 master: Optional[tk.Misc] = None,
                 left_side=None, right_side=None):
        if left is None and left_side is not None:
            from xp.workers.merger.merge import _stand_in

            left = _stand_in(left_side)
        if right is None and right_side is not None:
            from xp.workers.merger.merge import _stand_in

            right = _stand_in(right_side)
        self.left = left
        self.right = right
        # The *sides* are what can really be looked at: a PDF side has no
        # strokes, so its document is a stand-in and rendering it draws blank
        # pages -- blank thumbnails, and an auto match that measures nothing.
        # When the caller has them, thumbnails and auto match come from here.
        self.left_side = left_side
        self.right_side = right_side
        self.on_done = on_done
        self._owns_root = master is None

        # **No auto match on open.** The user's instruction: opening the merger or the
        # map must not run the similarity pass -- on the sample pair it costs
        # 6.6 s of signatures before the window can draw -- so the editor opens
        # with whatever pairing is already in force (nothing, the first time)
        # and matches only when the user asks for it. Manual linking is the default
        # way to work here.
        self.mapping: Dict[int, int] = {}
        # `unchanged` until Done says otherwise: Cancel or a closed window
        # leaves it so, and the merge window then applies nothing.
        self.outcome = 'unchanged'
        # Whether anything was actually done to the links. An untouched map and
        # a cleared one both hold nothing; only one of them is a decision.
        self.dirty = False
        self._cancelled = False
        if alignment is not None:
            for l, r, _ in alignment.matched:
                self.mapping[l] = r
        self._matched_by = 'by hand' if alignment is None else 'from the map'

        # The page picked first, as (side, index) -- either side may start
        # the pairing, because which one you notice first is not up to us.
        self._pending: Optional[Tuple[str, int]] = None
        self._thumbs: List[ImageTk.PhotoImage] = []  # keep references alive

        self.root = tk.Tk() if master is None else tk.Toplevel(master)
        from xp.runtime.shutdown import arm
        arm(self.root)
        self.root.title("XP - page mapping")
        # The Page Mapping window had no appearance at all: the system face on
        # the system grey, opened from a dark one. Same call as every other XP
        # window, for the same reason the user gave -- it should look like the rest
        # of the program, not like a different one.
        decorate(self.root)

        bar = ttk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text="Auto Match", command=self._auto).pack(side=tk.LEFT, padx=(6, 4), pady=4)
        ttk.Button(bar, text="Clear links", command=self._clear).pack(side=tk.LEFT)
        ttk.Button(bar, text="Done", command=self._done).pack(side=tk.RIGHT, padx=6)
        # **The status gets its own row and may wrap.** It used to share the
        # button row, so anything longer than "12 linked, 0 left unmatched" was
        # clipped at the window's edge with an ellipsis -- the counts and the
        # hint were both cut in half.
        status_row = ttk.Frame(self.root)
        status_row.pack(side=tk.TOP, fill=tk.X)
        self.status = ttk.Label(status_row, text="", anchor='w', justify='left')
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=(0, 4))
        self.root.bind('<Configure>', self._rewrap_status)
        # **Closing the window is Done, not cancel.** The user changes the links, closes the window
        # the way a window is closed -- and the merge went on showing the old pairing, because
        # `outcome` was only ever set on the Done button and stayed 'unchanged'. The X now
        # means "apply what I drew"; Escape is the explicit way to throw the changes away.
        self.root.protocol('WM_DELETE_WINDOW', self._done)
        self.root.bind('<Escape>', lambda _event: self._cancel())

        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)
        # The thumbnails are pages, and a page is paper -- but the gaps
        # between them are window, not page, so the canvas takes the ground
        # rather than a white sheet the pictures sit on.
        self.canvas = tk.Canvas(frame, bg=ABYSS, width=700,
                                 height=min(700, max(len(left.pages), len(right.pages)) * ROW_HEIGHT + 40))
        vbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.bind("<Button-1>", self._on_click)

        # **No overlap preview.** It was here: a thumbnail of the two pages' ink
        # on each other and "N% of the ink is in the same place", recomputed on
        # every link. The user's order (2026-10-26): "I do not need that live content
        # overlap preview, nor a percentage each time a link is made -- that
        # preview wastes time and makes manual linking slow." A link is a link:
        # clicking two pages pairs them and draws the line, nothing is measured.

        # **`M` closes it, and closes it as Done.** The user's order (2026-10-27):
        # "click M to close the page map or click Done to go back to the merge
        # GUI -- only then should there be any reloading". So both keys end the
        # same way: the pairing the user made is applied, once, on the way out. Nothing
        # in this window reloads the merge while it is open.
        self.root.bind('<m>', lambda _event: self._done())
        self.root.bind('<M>', lambda _event: self._done())

        # **Maximised, like the window it is opened from.** It came up at the
        # size of its two columns (about 450 wide), which wrapped its own status
        # line into two lines and left a scrollbar on a window with two pages in
        # it. The user's order for the windows: mostly full screen.
        from xp.runtime.shutdown import fill_the_screen

        self.root.update_idletasks()
        fill_the_screen(self.root)

        self._build_thumbs()
        self._redraw()

    # ---------------- thumbnails ----------------

    def _automatic_alignment(self, progress=None):
        """Auto match, by whatever both sides really have."""
        if self.left_side is not None and self.right_side is not None:
            from xp.workers.merger.matching.pagealign import align_sides

            return align_sides(self.left_side, self.right_side,
                               progress=progress)
        return align_pages(self.left, self.right, progress=progress)

    def _build_thumbs(self) -> None:
        """Fill the thumbnails after the window is up, a few per turn.

        **They are the reason the map took 5.8 s to open.** Rendering 63 pages
        -- two of them a 43,677-stroke page -- before the first draw is time the user
        spends looking at nothing. The window is drawn at once with empty
        frames, and the pictures arrive as they are made.
        """
        self.left_thumbs = [None] * len(self.left.pages)
        self.right_thumbs = [None] * len(self.right.pages)
        self._thumb_jobs = ([('left', i) for i in range(len(self.left.pages))]
                            + [('right', i) for i in range(len(self.right.pages))])
        self.root.after(1, self._fill_thumbs)

    def _visible_rows(self):
        """The first and last row index on screen, as a pair."""
        try:
            top = self.canvas.canvasy(0)
            height = max(1, self.canvas.winfo_height())
        except Exception:
            return 0, 12
        first = max(0, int((top - 24) // ROW_HEIGHT) - 1)
        last = int((top + height - 24) // ROW_HEIGHT) + 1
        return first, last

    def _next_thumb_job(self):
        """The pending thumbnail nearest to what the user is looking at.

        **Not simply the next one in the list.** The canvas is about 700 px
        tall and a row is 74, so roughly nineteen pictures are on screen at
        once -- and the user's pair has 129. Walking the list in order rendered a
        hundred and ten pictures nobody could see before reaching the ones the user
        was looking at. Measured on that pair: 33 batches, 7.56 s in total,
        each batch holding the GUI for 0.23 to 0.54 s.
        """
        if not self._thumb_jobs:
            # Asked with nothing left to do. `_fill_thumbs` never does this,
            # but a helper that answers `min()` on an empty list is a trap
            # for the next caller.
            return None
        first, last = self._visible_rows()
        middle = (first + last) / 2.0

        def distance(job):
            _side, index = job
            if first <= index <= last:
                return (0, abs(index - middle))
            return (1, abs(index - middle))

        best = min(range(len(self._thumb_jobs)),
                   key=lambda at: distance(self._thumb_jobs[at]))
        return self._thumb_jobs.pop(best)

    def _fill_thumbs(self) -> None:
        # **A time budget, not a count.** Four at a time meant a batch could
        # hold the window for half a second; a budget keeps each slice short
        # whatever a page costs to draw, which is what keeps the map usable
        # while it fills.
        import time as _time

        until = _time.perf_counter() + 0.06
        done = 0
        while self._thumb_jobs and (done < 1 or _time.perf_counter() < until):
            job = self._next_thumb_job()
            if job is None:
                break
            side, index = job
            document = self.left if side == 'left' else self.right
            held = self.left_side if side == 'left' else self.right_side
            picture = self._thumb(document.pages[index], held, index)
            (self.left_thumbs if side == 'left' else self.right_thumbs)[index] = picture
            done += 1
        self._redraw()
        if self._thumb_jobs:
            self.root.after(1, self._fill_thumbs)

    def _thumb(self, page, side=None, index: int = 0) -> ImageTk.PhotoImage:
        # A small render is enough to recognise a page, and cheap even for
        # documents with thousands of strokes per page. A picture side is
        # rendered through the side, which is the only thing that knows how.
        from PIL import Image

        if side is not None:
            image = side.render(index, 0.35)
            img = Image.fromarray(image[:, :, ::-1])
        else:
            img = render_page(page, scale=0.6)
        img.thumbnail((THUMB_W, THUMB_H))
        canvas = Image.new("RGB", (THUMB_W, THUMB_H), "white")
        canvas.paste(img, ((THUMB_W - img.width) // 2, (THUMB_H - img.height) // 2))
        photo = ImageTk.PhotoImage(canvas)
        self._thumbs.append(photo)
        return photo

    # ---------------- drawing ----------------

    def _row_y(self, index: int) -> int:
        return 24 + index * ROW_HEIGHT + ROW_HEIGHT // 2

    def _redraw(self) -> None:
        self.canvas.delete("all")
        # **Written on the ground, in the panel's face.** These were the
        # system default -- black, and a face that is not the program's --
        # drawn when this canvas was white. On the dark ground they were
        # near-invisible, which is half of what the user meant by the merger not
        # looking like the rest.
        for side in ("left", "right"):
            self.canvas.create_text(COLUMN_X[side], 10, text=side,
                                    fill=PALE, font=font(10, 'bold'))

        for side, doc, thumbs in (("left", self.left, self.left_thumbs),
                                   ("right", self.right, self.right_thumbs)):
            x = COLUMN_X[side]
            for i, page in enumerate(doc.pages):
                y = self._row_y(i)
                selected = self._pending == (side, i)
                self.canvas.create_rectangle(
                    x - 60, y - THUMB_H // 2 - 4, x + 60, y + THUMB_H // 2 + 4,
                    outline=PICK_COLOR if selected else FRAME_COLOR,
                    width=2 if selected else 1,
                    tags=(f"{side}:{i}",),
                )
                if thumbs[i] is not None:
                    self.canvas.create_image(x, y, image=thumbs[i],
                                             tags=(f"{side}:{i}",))
                else:
                    # Not made yet: the frame and the number are enough to
                    # start clicking, and the picture lands in a moment.
                    self.canvas.create_text(x, y, text='...',
                                            fill=MUTED, font=font(),
                                            tags=(f"{side}:{i}",))
                self.canvas.create_text(x - 78, y, text=str(i + 1), fill=PALE,
                                        font=font(), tags=(f"{side}:{i}",))

        for l, r in sorted(self.mapping.items()):
            self.canvas.create_line(
                COLUMN_X["left"] + 62, self._row_y(l),
                COLUMN_X["right"] - 62, self._row_y(r),
                fill=LINK_COLOR, width=2, tags=("link", f"link:{l}"),
            )

        height = 24 + max(len(self.left.pages), len(self.right.pages)) * ROW_HEIGHT + 24
        self.canvas.configure(scrollregion=(0, 0, 700, height))

        unmatched_left = [i for i in range(len(self.left.pages)) if i not in self.mapping]
        mapped_right = set(self.mapping.values())
        unmatched_right = [i for i in range(len(self.right.pages)) if i not in mapped_right]
        # **Short, because it has to fit.** The old line ran to ~200
        # characters, wrapped to three lines inside a two-line row, and the
        # middle of it was cut off -- the counts and the hint both. A dozen
        # words say the same thing and stay on one line.
        hint = "Click a left page, then a right one."
        blank_note = ''
        blank_left = len(getattr(self, '_blank_left', ()))
        blank_right = len(getattr(self, '_blank_right', ()))
        if blank_left or blank_right:
            blank_note = ' · %d/%d blank' % (blank_left, blank_right)
        self.status.config(
            text='%s · %d linked · %d left, %d right unmatched%s · %s'
                 % (self._matched_by, len(self.mapping), len(unmatched_left),
                    len(unmatched_right), blank_note, hint)
        )
        self._wrap_status()

    def _wrap_status(self, _event=None) -> None:
        """Let the status text use the window's width instead of being cut.

        A ttk.Label with no wraplength clips at the window's edge with an
        ellipsis, which is what the user saw: the hint and the counts both ran off.
        """
        try:
            # **The window's width, not the label's.** The label is packed with
            # `fill=X`, so its width follows the wraplength -- asking it is
            # circular, and it answered "as wide as the text", which is how the
            # first line ended up running past the window's edge and the third
            # line being clipped away.
            width = self.root.winfo_width()
            if width <= 1:
                width = 700
            # 90px of margin: the row's padding plus the border. A wraplength
            # wider than the label is how the last words kept being cut.
            self.status.configure(wraplength=max(240, width - 90))
        except tk.TclError:
            pass

    def _rewrap_status(self, event) -> None:
        if event.widget is self.root:
            self._wrap_status()

    # ---------------- interaction ----------------

    def _on_click(self, event) -> None:
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        items = self.canvas.find_overlapping(x - 2, y - 2, x + 2, y + 2)

        for item in reversed(items):
            for tag in self.canvas.gettags(item):
                if tag.startswith("link:"):
                    self.dirty = True
                    del self.mapping[int(tag.split(":")[1])]
                    self._redraw()
                    return
                if tag.startswith("left:"):
                    self._pick("left", int(tag.split(":")[1]))
                    return
                if tag.startswith("right:"):
                    self._pick("right", int(tag.split(":")[1]))
                    return

    def _pick(self, side: str, index: int) -> None:
        """Pick a page, or link it to the one already picked on the other
        side. Picking the same page again lets it go."""
        if self._pending == (side, index):
            self._pending = None
        elif self._pending is None or self._pending[0] == side:
            self._pending = (side, index)
        else:
            left, right = (index, self._pending[1]) if side == 'left' else (self._pending[1], index)
            # A page can only be used once on each side, so drop whatever
            # either of these two was linked to before.
            self.dirty = True
            for previous_left, previous_right in list(self.mapping.items()):
                if previous_left == left or previous_right == right:
                    del self.mapping[previous_left]
            self.mapping[left] = right
            self._pending = None
        self._redraw()

    def _auto(self) -> None:
        """Match every left page against every right page, on the user's request.

        **This is the expensive one, and it runs on a worker.** On the sample pair
        (2 pages against 61) the signatures alone cost 6.6 s and the whole pass
        12.7 s; doing that on the Tk thread painted the panel once with
        `update_idletasks` and then left the window deaf -- a Loading panel is
        not a responsive window. The pass now runs off the Tk thread and reports
        through a queue the main thread drains, so the panel updates, the window
        answers, and pressing the button again supersedes the run in flight
        (`_auto_request`): a result from an older request is dropped instead of
        being applied to a map the user has since changed.
        """
        import queue
        import threading

        from xp.render.widgets.progress import Progress

        if not hasattr(self, 'progress'):
            self.progress = Progress(self.root)
        self._auto_request = getattr(self, '_auto_request', 0) + 1
        version = self._auto_request
        # The map must not be submitted while a pass is in flight: Done would
        # carry the mapping the run has not produced yet.
        self._auto_running = True
        total = len(self.left.pages) + len(self.right.pages)
        self.progress.show('Auto matching pages', total,
                           detail='measuring the pages, %d to go' % total)
        self.root.update_idletasks()
        results = queue.Queue()

        def work():
            def say(done, of):
                results.put(('step', done, of))
            try:
                alignment = self._automatic_alignment(progress=say)
            except Exception as exc:
                results.put(('failed', type(exc).__name__, str(exc)))
                return
            results.put(('done', alignment))

        threading.Thread(target=work, daemon=True).start()
        self._pump_auto(version, results)

    def _pump_auto(self, version, results) -> None:
        """Drain the worker's queue on the Tk thread, one tick at a time."""
        import queue as _queue

        if getattr(self, '_auto_request', 0) != version:
            self._auto_running = False
            return                     # the user asked again; this answer is stale
        while True:
            try:
                message = results.get_nowait()
            except _queue.Empty:
                break
            kind = message[0]
            if kind == 'step':
                _kind, done, of = message
                self.progress.step(done, of, 'measuring page %d of %d' % (done, of))
                continue
            if kind == 'failed':
                _kind, name, text = message
                self._auto_running = False
                self.progress.failed('Auto match did not finish',
                                     '%s: %s' % (name, text))
                return
            if kind == 'done':
                self._auto_running = False
                self.progress.hide()
                alignment = message[1]
                self.mapping = {l: r for l, r, _ in alignment.matched}
                self.dirty = True
                self._pending = None
                self._matched_by = 'auto match'
                self._last_alignment = alignment
                if self.mapping:
                    first_left = sorted(self.mapping)[0]
        
                self._redraw()
                return
        try:
            self.root.after(80, lambda: self._pump_auto(version, results))
        except Exception:
            return                     # the window went away in the meantime

    def _clear(self) -> None:
        self.dirty = True
        self.mapping.clear()
        self._pending = None
        self._redraw()

    def _done(self) -> None:
        if getattr(self, '_auto_running', False):
            # The pass is still measuring; Done would submit a mapping that has
            # not arrived. The panel is up and the button can wait.
            return
        # **"Done" with an empty map is a decision**: the user cleared the links and
        # pressed Done, and that means every page stands alone. It used to be
        # indistinguishable from "I opened this and changed nothing", because
        # the only thing the caller received was the dict -- and an empty dict
        # was read as "leave it as it was", so Clear links did nothing.
        if self.mapping:
            self.outcome = 'mapped'
        elif self.dirty:
            self.outcome = 'cleared'          # the user cleared them and meant it
        else:
            self.outcome = 'unchanged'        # the user only looked
        if self.on_done:
            self.on_done(dict(self.mapping))
        self.root.destroy()

    @property
    def was_cancelled(self) -> bool:
        return self.outcome == 'cancelled'

    def _cancel(self) -> None:
        self.outcome = 'cancelled'
        self.root.destroy()

    def run(self) -> Dict[int, int]:
        """Blocks until the editor is closed, then returns the mapping.

        Standalone that means running our own mainloop; opened as a Toplevel
        under someone else's root it means wait_window(), which keeps their
        event loop pumping but doesn't return until this window is gone --
        without it the caller would carry on with an unedited mapping the
        moment the editor appeared.
        """
        if self._owns_root:
            self.root.mainloop()
        else:
            self.root.grab_set()
            self.root.wait_window(self.root)
        return self.mapping


class ManualAlignment:
    """An Alignment-compatible view over a hand-made {left: right} mapping,
    so MergeSession can consume it the same way."""

    def __init__(self, mapping: Dict[int, int]):
        self.mapping = dict(mapping)

    def right_for_left(self, left_index: int) -> Optional[int]:
        return self.mapping.get(left_index)

    @property
    def matched(self) -> List[Tuple[int, int, float]]:
        return [(l, r, 1.0) for l, r in sorted(self.mapping.items())]

    def summary(self) -> str:
        pairs = ", ".join(f"{l + 1}->{r + 1}" for l, r in sorted(self.mapping.items()))
        return f"manual mapping: {pairs or '(none)'}"


def edit_mapping(left: XoppDocument, right: XoppDocument,
                  alignment: Optional[Alignment] = None,
                  master: Optional[tk.Misc] = None,
                  left_side=None, right_side=None) -> ManualAlignment:
    """Open the mapping editor and return what the user settled on."""
    app = PageMapApp(left, right, alignment=alignment, master=master,
                     left_side=left_side, right_side=right_side)
    app.run()
    return ManualAlignment(app.mapping)
