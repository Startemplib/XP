"""The merge as three layers of ink, and one composition state per piece.

What was wrong before this file existed
---------------------------------------
Selection was a **coverage collector**. The brush painted into a pixel mask
and `brush_capture` asked whether that mask *fully* covered a stroke; the
mask was wiped at the end of every gesture. So:

  * brushing part of a long stroke did nothing at all;
  * lifting the pen halfway along it did nothing, twice -- the first half was
    gone from the mask before the second half was painted;
  * zoomed in, a stroke wider than the window could not be chosen by any
    gesture, because no single gesture could cover it.

Measured directly against the session: left half -> 0 decided, right half ->
0 decided, left half again -> 0 decided, whole stroke in one pass -> 1
decided. That is the whole of the "the second brush over the same place does
nothing" report.

What this is instead
--------------------
Three base layers of ink, which do not change as the user brushes:

    COMMON      ink both documents have          always black
    LEFT_ONLY   ink only the left document has   red by default
    RIGHT_ONLY  ink only the right document has  blue by default

and, separately, a **composition state** saying which layers win where. The
state is not a pixel plane and not a history of gestures: it lives on
*pieces* of ink.

A piece is a contiguous run of points of one source stroke. Every stroke
starts as a single piece covering all of it. When the brush touches part of a
stroke, that stroke's piece is **split** at the boundary and only the touched
piece takes the new state. That is why no coverage threshold is needed: there
is nothing to accumulate, because the geometry itself is divided.

    D(piece) = the action of the last brush that touched it

Last-write-wins falls out of that: a later gesture overwrites `state` on the
pieces it touches and reads nothing about what came before. LEFT then RIGHT
then LEFT leaves LEFT, with no replay and no undo stack consulted.

Why the renderer needs no changes
---------------------------------
`views.arrays_of` already hands drawing a side as flat arrays --
`(xs, ys, starts, counts, widths, own)` -- where a stroke is a `(start,
count)` slice of one point array and `own` is its colour. A piece is the
*same* shape of thing: a different `(start, count)` into the **same**
`xs`/`ys`. So splitting a stroke copies no point data, and the three layers
are drawn by exactly the code that drew strokes, with one colour per piece.

Growing happens at the end of the arrays: a split rewrites the original
entry as the first part and appends the rest, so the cost of a gesture is
the number of pieces it touched and nothing else. Nothing walks the page.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from xp.workers.merger.compose import (KEEPS, LEFT, RIGHT, TAKE_BOTH,
                                       TAKE_LEFT, TAKE_NONE, TAKE_RIGHT)

# The three base layers. A piece's layer never changes.
COMMON = 0
EXCLUSIVE = 1

# The composition state of a piece: the last decision that touched it.
DEFAULT = 0
PICK_LEFT = 1
PICK_RIGHT = 2
PICK_BOTH = 3
PICK_NONE = 4

STATE_OF = {TAKE_LEFT: PICK_LEFT, TAKE_RIGHT: PICK_RIGHT,
            TAKE_BOTH: PICK_BOTH, TAKE_NONE: PICK_NONE}

# The colours, which are the convention and not a choice made here.
BLACK = (0, 0, 0)
RED = (155, 17, 30)
BLUE = (62, 166, 227)
GREEN = (24, 154, 70)

# How many pieces a side's arrays have room for before they are grown. A
# split appends, and appending into spare room is free; doubling when it runs
# out keeps the amortised cost of a gesture proportional to what it touched.
SPARE = 64

# Page units per cell of the stroke index. A Qbase page is 188 x 391 pt, so
# this is a grid of about 31 x 65 cells -- and the size matters more than it
# looks: at 64 pt the whole page was 3 x 7 cells, so a cell held a couple of
# thousand strokes and the index bought nothing at all. Measured on forty
# thousand strokes, one brush segment then tested 2,700 pieces point by
# point, which is why the cost still followed the page after the index was
# added. The brush is a few page units across, so a cell of six means a
# gesture asks about the strokes under it and almost nothing else.
_NEXT_BOOK_ID = 0
GRID = 6.0


def colour_of(side: str, layer: int, state: int):
    """What colour this piece is drawn in, or None if it is not drawn.

    The table in full, and the only rule this module has:

        layer       state       left piece      right piece
        COMMON      DEFAULT     black           black
        COMMON      PICK_LEFT   black           black
        COMMON      PICK_RIGHT  black           black
        COMMON      PICK_BOTH   black           black
        COMMON      PICK_NONE   hidden          hidden
        EXCLUSIVE   DEFAULT     red             blue
        EXCLUSIVE   PICK_LEFT   green           hidden
        EXCLUSIVE   PICK_RIGHT  hidden          green
        EXCLUSIVE   PICK_BOTH   green           green
        EXCLUSIVE   PICK_NONE   hidden          hidden

    **Neither hides common ink too.** It used to return black for a COMMON piece
    before it ever looked at the state, so "Neither" over a shared line left the
    line on the page -- the one thing that button is for. Neither is not a
    deletion and does not touch the ink: it brings the white background to the
    front there, and the background is in front of *everything*, common ink
    included. Choosing between the sides, on the other hand, is a question about
    ink that only one of them has, so a COMMON piece stays black through
    PICK_LEFT and PICK_RIGHT.

    Hidden means hidden **here**, on this piece: the rest of the same stroke is
    untouched.
    """
    if state == PICK_NONE:
        return None
    if layer == COMMON:
        return BLACK
    if state == DEFAULT:
        return RED if side == LEFT else BLUE
    if state == PICK_BOTH:
        return GREEN
    if state == PICK_NONE:
        return None
    mine = (state == PICK_LEFT) if side == LEFT else (state == PICK_RIGHT)
    return GREEN if mine else None


def keeps_of(side: str, layer: int, state: int, base: str = LEFT) -> bool:
    """Whether this piece is **written to the file**. Not the same question.

    Drawing and writing part company on common ink, and conflating them was
    a real fault: `colour_of` says common ink is black whatever is decided
    about it, which is right on screen -- both documents have that ink, so
    no decision there changes what the user sees. But the file has to hold *one*
    copy of it, and "take right" over common ink means the right document's
    copy is the one written and the left's gives way. Reading the screen's
    answer as the file's answer kept both.

    `base` is the side an undecided piece is kept from: the notebook being
    merged into, which `MergeSession.base_side` names (and which is the
    right side when the left one is a picture).
    """
    if state == PICK_BOTH:
        # **Both means both sides' own ink, not two copies of the shared ink.** The user's rule: common
        # content "cannot be output twice and made bolder". On a COMMON piece both documents hold the
        # same ink in the same place, so one copy goes in the file -- the base side's, exactly as an
        # undecided piece is written. On an EXCLUSIVE piece the two sides are different ink, so both
        # are written, which is what Both is for.
        #
        # Measured on the user's pair: 395 of 400 sampled right strokes are exactly 100 % on the left's ink,
        # and writing both sides put 17,030 strokes in the file for 765 strokes of content.
        if layer == COMMON:
            return side == base
        return True
    if state == PICK_NONE:
        return False
    if state == PICK_LEFT:
        return side == LEFT
    if state == PICK_RIGHT:
        return side == RIGHT
    return side == base


class SideLayers:
    """One side's ink, as pieces, with the arrays drawing already wants.

    `starts`/`counts` index the flat point arrays that came from
    `views.arrays_of`, so a piece costs two integers and no points. `origin`
    says which source stroke each piece belongs to, which is what export
    needs to slice the real stroke back out.
    """

    def __init__(self, side: str, arrays, matched, base: str = LEFT):
        xs, ys, starts, counts, widths, own = arrays
        self.side = side
        self.base = base
        # **An identity that is not an address.** `id()` was used to tell which
        # book an undo record was made on, and a released object's address is
        # the first thing a new one gets: measured, a rebuilt book landed on
        # the same id and the guard did not fire. This counter never repeats.
        global _NEXT_BOOK_ID
        _NEXT_BOOK_ID += 1
        self.book_id = _NEXT_BOOK_ID
        self.xs, self.ys = xs, ys
        many = len(starts)
        room = many + SPARE
        self._many = many
        # The ink this book was built from, in points. Pieces are cut so that
        # neighbours share one point, so the sum of the *piece* counts is one
        # per cut more than this -- which is why this is what a new copy of the
        # arrays is matched against.
        self._ink = int(np.sum(counts))
        # How many *strokes* (not pieces) the ink came from. A book is built
        # with one piece per stroke and splits them later, so `_many` grows;
        # what has to stay equal for a reattach is the stroke count.
        self._strokes = many
        self.starts = np.zeros(room, np.int64)
        self.counts = np.zeros(room, np.int64)
        # **Where the piece sits in its source stroke, in point-index space.**
        # `starts`/`counts` are where its points are in the flat arrays, which
        # is what the painter draws; these are the span of the stroke it means,
        # which is what export slices. They differ once a cut falls inside a
        # segment (a two-point line brushed in the middle: the piece is from
        # 0.25 to 0.75 and neither end is a point of the file).
        self.t0 = np.zeros(room, np.float64)
        self.t1 = np.zeros(room, np.float64)
        # Points that belong to a cut piece's interpolated ends. The arrays are
        # `_points` long when they arrive and grow past that; everything that
        # reduces over "the rest of the array" must stop at `_points`.
        self._points = int(len(xs))
        # Points written so far; the arrays may have more room than that.
        self._point_used = self._points
        self._stroke_start = np.zeros(many, np.int64)
        self.widths = np.zeros(room, np.float64)
        self.origin = np.zeros(room, np.int64)
        self.layer = np.zeros(room, np.uint8)
        self.state = np.zeros(room, np.uint8)
        self.colour = np.zeros((room, 3), np.uint8)
        self.shown = np.zeros(room, np.bool_)
        # Drawn, and written: two answers, because common ink is black on
        # screen whichever side wins it and yet only one side's copy of it
        # goes into the file. See `keeps_of`.
        self.kept = np.zeros(room, np.bool_)
        if many:
            self.starts[:many] = starts
            self.counts[:many] = counts
            self._stroke_start[:] = starts
            self._stroke_count = np.asarray(counts, np.int64).copy()
            # **Stroke-local**, not offsets into the flat arrays: `t` is a place
            # in one stroke's own polyline (`2.5` is half way from its point 2
            # to its point 3), and the array offset is `_stroke_start + t`.
            self.t0[:many] = 0
            self.t1[:many] = counts - 1
            self.widths[:many] = widths
            self.origin[:many] = np.arange(many)
            self.layer[:many] = np.where(np.asarray(matched, bool), COMMON,
                                         EXCLUSIVE)
            self._recolour(np.arange(many))
        self._index = None
        self._joins = []
        self._by_stroke = None
        self._boxes = {}
        # Bumped whenever a piece is split, re-stated or re-coloured. The
        # drawing cache keys on the identity of the arrays tuple, so the
        # tuple is held and rebuilt only when this moves -- otherwise every
        # frame would hand the card a "new" page and upload it again.
        self.revision = 0
        self._held = None
        # While a gesture is open this is a list, and every state change is
        # written into it as (piece, before, after). Undo is the only thing
        # that reads it: the picture is computed from the pieces' current
        # state alone, never by replaying this.
        self.journal = None
        self.snapshot = None

    # ---- what the renderer asks for -----------------------------------

    def __len__(self):
        return self._many

    def reattach(self, arrays, common=None) -> bool:
        """Take a new copy of the same ink, **keeping the cuts and the states**.

        A brush decision is not derived from anything: it is what the user did. The
        arrays are derived -- they are the fitted coordinates -- so a change of
        fit rebuilds them, and this is how the book survives that: same strokes,
        same point counts, so the piece structure and every piece's state stay
        valid and only what the pieces point at is replaced.

        Returns False when the ink is not the same ink, in which case the book
        must be built again from the stroke-level decisions.
        """
        xs, ys, starts, counts, widths, own = arrays
        many = len(starts)
        if many != self._strokes:
            return False
        if int(np.sum(counts)) != self._ink:
            return False
        self.xs, self.ys = xs, ys
        # The arrays came fresh: the interpolated ends of any cut piece are
        # gone with the old ones and are written again below, at the new
        # coordinates but the same spans -- a fit changes where the ink is, not
        # what was decided about it.
        self._points = int(len(xs))
        self._point_used = self._points
        self._index = None
        self._by_stroke = None
        self._boxes = {}
        self._joins = []
        for at in range(self._many):
            s0, s1 = float(self.t0[at]), float(self.t1[at])
            fractional = (abs(s0 - round(s0)) > 1e-9
                          or abs(s1 - round(s1)) > 1e-9)
            if not fractional and int(self.starts[at]) < self._points:
                continue          # a plain range of the stroke: still valid
            start, count = self._place_span(int(self.origin[at]), s0, s1)
            self.starts[at] = start
            self.counts[at] = count
        if common is not None:
            # **Which ink is common depends on the pairing, so it is refreshed
            # with the coordinates.** The cuts and the states do not.
            flags = np.asarray(common, dtype=bool)
            if len(flags) == self._strokes:
                many = self._many
                origins = self.origin[:many].astype(np.int64)
                inside = origins < len(flags)
                self.layer[:many][inside] = np.where(
                    flags[origins[inside]], COMMON, EXCLUSIVE)
                self.layer[:many][~inside] = EXCLUSIVE
        self.revision += 1
        self._recolour(np.arange(self._many))
        return True

    def arrays(self):
        """This side's pieces in `views.arrays_of` shape, ready to draw.

        The same tuple object for as long as nothing has changed: the card's
        resident-page key is `id(arrays)`, so handing back a fresh tuple
        every frame would re-upload the page on every frame.
        """
        held = self._held
        if held is not None and held[0] == self.revision:
            return held[1]
        at = self._many
        made = (self.xs, self.ys, self.starts[:at], self.counts[:at],
                self.widths[:at], self.colour[:at])
        self._held = (self.revision, made)
        return made

    def visible(self):
        return self.shown[:self._many]

    def _recolour(self, which) -> None:
        self.revision = getattr(self, 'revision', 0) + 1
        self._held = None
        for at in np.atleast_1d(which):
            layer, state = int(self.layer[at]), int(self.state[at])
            rgb = colour_of(self.side, layer, state)
            self.shown[at] = rgb is not None
            self.colour[at] = rgb if rgb is not None else (0, 0, 0)
            self.kept[at] = keeps_of(self.side, layer, state, self.base)

    def _stroke_points(self, origin: int):
        """The source stroke's own points, as the arrays hold them now."""
        start = int(self._stroke_start[origin])
        count = int(self._stroke_count[origin])
        return self.xs[start:start + count], self.ys[start:start + count]

    def _span_geometry(self, origin: int, t0: float, t1: float):
        """The piece's polyline: the stroke between two **fractional** indexes.

        `t0`/`t1` are in point-index space, so `2.5` is half way along the
        segment from point 2 to point 3. Interpolating there is what lets a cut
        fall inside a segment: the two ends are real points of the drawn line
        and neither of them is a point of the file.
        """
        px, py = self._stroke_points(origin)
        count = len(px)
        if count == 0:
            return np.zeros(0), np.zeros(0)
        if count == 1:
            return np.asarray(px, float), np.asarray(py, float)

        def at(t: float):
            t = max(0.0, min(float(t), count - 1))
            index = min(int(t), count - 2)
            fraction = t - index
            return (px[index] + (px[index + 1] - px[index]) * fraction,
                    py[index] + (py[index + 1] - py[index]) * fraction)

        low = int(np.floor(t0))
        high = int(np.ceil(t1))
        xs = [at(t0)[0]]
        ys = [at(t0)[1]]
        for index in range(low + 1, high):
            if index > t0 + 1e-9 and index < t1 - 1e-9:
                xs.append(px[index])
                ys.append(py[index])
        if t1 > t0 + 1e-9:
            x, y = at(t1)
            xs.append(x)
            ys.append(y)
        return np.asarray(xs, float), np.asarray(ys, float)

    def _grow_points(self, count: int) -> None:
        """Room in the point arrays for `count` more interpolated ends."""
        used = getattr(self, '_point_used', self._points)
        need = used + count
        if need <= len(self.xs):
            return
        room = max(need, int(need * 1.5) + 64)
        kept = len(self.xs)
        xs = np.zeros(room, np.float64)
        ys = np.zeros(room, np.float64)
        xs[:kept], ys[:kept] = self.xs, self.ys
        self.xs, self.ys = xs, ys
        self._held = None

    def _place_span(self, origin: int, t0: float, t1: float):
        """Put a piece's polyline in the arrays and answer (start, count)."""
        xs, ys = self._span_geometry(origin, t0, t1)
        count = len(xs)
        self._grow_points(count)
        start = getattr(self, '_point_used', self._points)
        self.xs[start:start + count] = xs
        self.ys[start:start + count] = ys
        self._point_used = start + count
        return start, count

    def _span_polyline(self, origin: int, t0: float, t1: float):
        """The piece's drawn polyline **and where each of its points sits**.

        The `t` values matter as much as the coordinates: the cutter has to
        answer in `t`, and for a piece that was cut before the first and last
        drawn points are interpolated ends at `t0` and `t1` while the ones
        between are the stroke's own vertices.
        """
        import math

        px, py = self._stroke_points(origin)
        count = len(px)
        if count == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        if count == 1:
            return np.asarray(px, float), np.asarray(py, float), np.asarray([0.0])

        def at(t: float):
            t = max(0.0, min(float(t), count - 1))
            index = min(int(t), count - 2)
            fraction = t - index
            return (px[index] + (px[index + 1] - px[index]) * fraction,
                    py[index] + (py[index + 1] - py[index]) * fraction)

        xs, ys, ts = [], [], []
        low, high = int(math.floor(t0)), int(math.ceil(t1))
        x, y = at(t0)
        xs.append(x)
        ys.append(y)
        ts.append(float(t0))
        for index in range(low + 1, high):
            if index > t0 + 1e-9 and index < t1 - 1e-9:
                xs.append(px[index])
                ys.append(py[index])
                ts.append(float(index))
        if t1 > t0 + 1e-9:
            x, y = at(t1)
            xs.append(x)
            ys.append(y)
            ts.append(float(t1))
        return (np.asarray(xs, float), np.asarray(ys, float),
                np.asarray(ts, float))

    def _capsule_spans(self, origin: int, t0: float, t1: float,
                       x0, y0, x1, y1, radius):
        """Where the brush's capsule crosses this piece, in `t` space.

        **The intersection, not a sample of it.** The distance from a point of
        one segment to the brush's segment is convex along that segment, so the
        part of the segment inside the capsule is a single interval: minimised
        by golden section, then each boundary bisected. Sampling was tried and
        is wrong in a way that matters -- a radius-0.5 brush at the middle of a
        100-unit segment falls between any sane number of samples and did
        nothing, which is exactly the complaint that started this.
        """
        px, py, ts = self._span_polyline(origin, t0, t1)
        count = len(px)
        if count == 0:
            return []
        if count == 1:
            # **Distance to the brush's segment, not to a box around it.** The
            # first version clamped the point into the brush's axis-aligned
            # rectangle, which is a different question: a point at (0,10) with
            # a brush from (0,0) to (10,10) is 7.07 away from the line and was
            # reported as covered, because it sits inside the box the line
            # spans. `_clamp_to_segment` answers the question the brush asks.
            near_x, near_y = _clamp_to_segment(px[0], py[0], x0, y0, x1, y1)
            distance = float(np.hypot(px[0] - near_x, py[0] - near_y))
            return [] if distance > radius else [(float(ts[0]), float(ts[0]))]

        spans = []
        for index in range(count - 1):
            interval = _segment_in_capsule(px[index], py[index],
                                           px[index + 1], py[index + 1],
                                           x0, y0, x1, y1, radius)
            if interval is None:
                continue
            low, high = interval
            start_t = ts[index] + (ts[index + 1] - ts[index]) * low
            stop_t = ts[index] + (ts[index + 1] - ts[index]) * high
            if spans and start_t <= spans[-1][1] + 1e-9:
                spans[-1] = (spans[-1][0], max(spans[-1][1], stop_t))
            else:
                spans.append((start_t, stop_t))
        return [span for span in spans if span[1] - span[0] > 1e-9]

    def _room_for(self, more: int) -> None:
        if self._many + more <= len(self.starts):
            return
        want = max(len(self.starts) * 2, self._many + more + SPARE)
        for name in ('starts', 'counts', 't0', 't1', 'widths', 'origin',
                     'layer', 'state', 'shown', 'kept'):
            held = getattr(self, name)
            grown = np.zeros((want,) + held.shape[1:], held.dtype)
            grown[:self._many] = held[:self._many]
            setattr(self, name, grown)
        grown = np.zeros((want, 3), np.uint8)
        grown[:self._many] = self.colour[:self._many]
        self.colour = grown

    # ---- the stroke index, so a gesture never walks the page -----------

    def _cells(self):
        """Which grid cells each *source stroke* falls in, worked out once.

        Built from the pieces as they are at first, which is one per stroke,
        and keyed on the source stroke rather than the piece: splitting a
        stroke never moves it, so the index stays correct for the life of the
        page and a gesture never rebuilds it.
        """
        if self._index is not None:
            return self._index
        many = self._many
        boxes = np.empty((many, 4), np.float64)
        if many:
            # One reduction for the whole page, not one slice per stroke:
            # `np.minimum.reduceat` is how `views._boxes_from` does the same
            # job, and on forty thousand strokes the Python loop it replaces
            # was the slowest thing about opening the page.
            at = np.asarray(self.starts[:many], dtype=np.int64)
            # `_points`, not the end of the array: the arrays grow when a cut
            # piece needs an interpolated end, and reducing past `_points`
            # would pull that end into whichever piece happened to be last.
            xs, ys = self.xs[:self._points], self.ys[:self._points]
            room = int(len(xs))
            # **A piece can start at the end of the array.** `_place_span`
            # records `start` and then writes `count` points, so a span that
            # came out empty leaves a piece whose start *is* the length --
            # and `np.minimum.reduceat` refuses that index. It raised
            # `IndexError: index 85043 out-of-bounds` out of `_on_drag` on the user's
            # page: the whole drag died, every segment after it died with it,
            # and nothing was said. A piece with no points has no box, so it
            # is reduced at a legal index here and marked empty below with the
            # pieces whose count is zero.
            if room:
                inside = at < room
                safe = np.where(inside, at, room - 1)
                boxes[:, 0] = np.minimum.reduceat(xs, safe)
                boxes[:, 1] = np.minimum.reduceat(ys, safe)
                boxes[:, 2] = np.maximum.reduceat(xs, safe)
                boxes[:, 3] = np.maximum.reduceat(ys, safe)
            else:
                inside = np.zeros(many, dtype=bool)
                boxes[:] = (np.inf, np.inf, -np.inf, -np.inf)
            empty = (self.counts[:many] <= 0) | ~inside
            if empty.any():
                boxes[empty] = (np.inf, np.inf, -np.inf, -np.inf)
        pad = self.widths[:many] * 0.5
        # Empty pieces carry +-inf in `boxes`, and the divisions below turn
        # that into nan. They are skipped by the `isfinite` test in the loop,
        # so the warning is noise -- and noise in a log is what makes a real
        # one unreadable.
        with np.errstate(invalid='ignore'):
            lo_x = np.floor_divide(boxes[:, 0] - pad, GRID)
            hi_x = np.floor_divide(boxes[:, 2] + pad, GRID)
            lo_y = np.floor_divide(boxes[:, 1] - pad, GRID)
            hi_y = np.floor_divide(boxes[:, 3] + pad, GRID)
        cells = {}
        for at in range(many):
            if not np.isfinite(lo_x[at]):
                continue
            origin = int(self.origin[at])
            for cx in range(int(lo_x[at]), int(hi_x[at]) + 1):
                for cy in range(int(lo_y[at]), int(hi_y[at]) + 1):
                    cells.setdefault((cx, cy), []).append(origin)
        self._index = (cells, boxes)
        return self._index

    def _near(self, x0, y0, x1, y1, radius) -> List[int]:
        """The source strokes whose cells the brush segment passes through."""
        return self._near_box(min(x0, x1) - radius, min(y0, y1) - radius,
                              max(x0, x1) + radius, max(y0, y1) + radius)

    def _near_box(self, x0, y0, x1, y1) -> List[int]:
        cells, _boxes = self._cells()
        found = set()
        for cx in range(int(x0 // GRID), int(x1 // GRID) + 1):
            for cy in range(int(y0 // GRID), int(y1 // GRID) + 1):
                held = cells.get((cx, cy))
                if held:
                    found.update(held)
        return sorted(found)

    def piece_box(self, piece: int):
        """One piece's box in page units, padded for its width.

        Held, because this is asked far more often than it changes: the
        repaint asks it of every piece near the brush, and a piece's extent
        only moves when that piece is split. Worked out fresh each time it
        was numpy min/max over the points, several hundred times a motion
        event.
        """
        held = self._boxes.get(piece)
        if held is not None:
            return held
        start, count = int(self.starts[piece]), int(self.counts[piece])
        if count <= 0:
            return None
        px = self.xs[start:start + count]
        py = self.ys[start:start + count]
        if not len(px):
            # A piece whose points are not there is a bug in the cutting, not a
            # picture: say nothing and draw nothing rather than raising in the
            # middle of a repaint.
            return None
        pad = float(self.widths[piece]) * 0.5 + 1.0
        box = (px.min() - pad, py.min() - pad, px.max() + pad, py.max() + pad)
        self._boxes[piece] = box
        return box

    def shown_crossing(self, x0, y0, x1, y1):
        """The visible pieces in this box, grouped by the colour they draw.

        One pass over the box for all four colours. `layer_patch` used to
        ask once per colour per side -- eight grid queries and eight rounds
        of box tests to draw one patch -- which is eight times the work for
        an answer that does not depend on the colour being asked about.
        """
        out = {}
        for piece in self.pieces_crossing(x0, y0, x1, y1):
            if not self.shown[piece]:
                continue
            out.setdefault(tuple(int(v) for v in self.colour[piece]),
                           []).append(piece)
        return out

    def pieces_crossing(self, x0, y0, x1, y1) -> List[int]:
        """Which pieces have ink inside this box, in page units.

        Through the cell index, so the answer costs what is there rather
        than what the page holds. This is what makes a local repaint local:
        the drawing it feeds sees only the pieces crossing the damaged
        rectangle, and a page of forty thousand strokes hands it the same
        handful as a page of one thousand.
        """
        found = []
        for stroke_index in self._near_box(x0, y0, x1, y1):
            # **A snapshot, not the live list.** The cutter appends the pieces
            # it makes; iterating the list it appends to would paint those new
            # pieces again inside the same segment, and a piece cut exactly at
            # the brush's edge would be re-cut and end up decided.
            for piece in list(self.pieces_of(stroke_index)):
                box = self.piece_box(piece)
                if box is None:
                    continue
                if box[2] < x0 or box[0] > x1 or box[3] < y0 or box[1] > y1:
                    continue
                found.append(piece)
        return found

    # ---- the brush -----------------------------------------------------

    def pieces_of(self, stroke_index: int) -> List[int]:
        """Which pieces belong to this source stroke.

        Held as a list per stroke and appended to as pieces are made. It was
        `np.flatnonzero(origin == stroke_index)`, which reads the whole
        page's worth of pieces to answer a question about one stroke --
        measured at 4.6 ms a brush segment on a thousand strokes and
        **159 ms on forty thousand**, which is the cost following the page
        rather than following the brush. Now a gesture costs what it touches.
        """
        held = self._by_stroke
        if held is None:
            held = self._by_stroke = {}
            for at in range(self._many):
                held.setdefault(int(self.origin[at]), []).append(at)
        return held.get(stroke_index, ())

    def _stroke_box(self, stroke_index: int):
        """The whole of this stroke's box, padded for its width.

        The unit of damage is the **stroke**, not the piece. Three narrower
        rules were tried and each left ink behind, because changing one
        piece changes more than that piece: the spans its siblings cover
        shift, a new boundary draws a cap that was not there, and a piece
        that shrinks gives up a span whose new owner reports no change of
        colour at all. Measured, the last of those left 81 pixels of a page
        white where a fresh render had green -- every one of them inside a
        rectangle the repair had already been given, so the box was right
        about where the colour moved and wrong about where the ink did.

        A stroke's own box is provably enough: nothing about how it is
        divided can put ink outside it. It is also what the repair this
        replaced always used, and that one was quick enough for every motion
        event of a drag, because a gesture touches the strokes near the
        brush and no others.
        """
        _cells, boxes = self._cells()
        if not (0 <= stroke_index < len(boxes)):
            return (np.inf, np.inf, -np.inf, -np.inf)
        box = boxes[stroke_index]
        pieces = self.pieces_of(stroke_index)
        pad = 2.0
        if len(pieces):
            pad = max(float(self.widths[at]) for at in pieces) * 0.5 + 2.0
        return (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)

    def _also_dirty(self, box) -> None:
        """A rectangle that has to be redrawn although no colour moved there.

        Splitting a piece shrinks it, and the span it gives up is taken over
        by a sibling whose colour did not change -- so nothing reports it,
        and the repair paints the background over ink it never puts back.
        Measured as 81 pixels of a page, white where a fresh render had
        green, **every one of them inside a rectangle the repair had already
        been given**: the box was right about where the colour changed and
        wrong about where the ink did.

        So a split makes the whole of the piece it divided dirty. The pieces
        it leaves partition exactly that span, which is why nothing smaller
        is enough. It costs the bbox of one piece, once, when that piece is
        first divided; the segments after it divide something smaller.
        """
        if box is not None and box[0] <= box[2]:
            self._joins.append(box)

    def paint(self, x0, y0, x1, y1, radius, state) -> Optional[Tuple]:
        """Set `state` on every piece of ink inside this brush segment.

        The segment is the brush's own geometry -- a round-capped line of
        `radius`, exactly what the existing brush draws -- and the test is
        per point, so a piece is split at the first point that falls outside
        it. Returns the box this changed, in page units, or None.

        Only the strokes in the cells the segment passes through are looked
        at. The size of the page does not enter into it.
        """
        boxes = []
        # The segment's own box, padded by the brush. A piece whose ink is
        # outside it cannot be inside the brush, and rejecting it by two
        # comparisons is far cheaper than testing its points -- which was
        # 162,000 point tests for sixty segments before this was here.
        lo_x, hi_x = min(x0, x1) - radius, max(x0, x1) + radius
        lo_y, hi_y = min(y0, y1) - radius, max(y0, y1) + radius
        for stroke_index in self._near(x0, y0, x1, y1, radius):
            changed = []
            # **A snapshot, not the live list.** The cutter appends the pieces
            # it makes; iterating the list it appends to painted those new
            # pieces again inside the same segment, and a piece that began
            # exactly at the brush's edge was re-cut into a hidden sliver.
            for piece in list(self.pieces_of(stroke_index)):
                box = self.piece_box(piece)
                if box is None or (box[2] < lo_x or box[0] > hi_x
                                   or box[3] < lo_y or box[1] > hi_y):
                    continue
                changed.extend(self._paint_piece(piece, x0, y0, x1, y1,
                                                 radius, state))
            if changed:
                boxes.append(self._stroke_box(stroke_index))
        if not boxes:
            return None
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    def _paint_piece(self, piece: int, x0, y0, x1, y1, radius, state):
        """Set `state` on the part of this piece the brush really covers.

        **Whether the brush lands on a vertex or between two of them is the
        same question**: the piece's polyline either passes through the capsule
        or it does not, and where it does is an interval. The vertex-run
        splitting this replaces could only cut at points, so it widened a
        one-point hit into two whole segments -- a radius-0.5 brush at a vertex
        of a 4-unit line hid 60..68 instead of 59.5..60.5.

        Returns the pieces whose drawn colour changed, which is what the window
        repaints -- a piece already in this state, or one the brush missed, is
        not reported.
        """
        if int(self.state[piece]) == state:
            return []
        first, last = float(self.t0[piece]), float(self.t1[piece])
        spans = self._capsule_spans(int(self.origin[piece]), first, last,
                                    x0, y0, x1, y1, radius)
        if not spans:
            return []
        was = int(self.state[piece])
        whole = (len(spans) == 1
                 and abs(spans[0][0] - first) < 1e-9
                 and abs(spans[0][1] - last) < 1e-9)
        if whole:
            self._note(piece, was, state)
            self.state[piece] = state
            self._recolour(piece)
            return [piece]
        return self._split_by_spans(piece, spans, state, was)

    def _split_by_spans(self, piece: int, spans, state: int, was: int) -> list:
        """Cut a piece at the brush's edges and state what is inside."""
        first, last = float(self.t0[piece]), float(self.t1[piece])
        wanted = [(first, last, was)]
        for low, high in spans:
            inside = []
            for s0, s1, held in wanted:
                if high <= s0 or low >= s1:
                    inside.append((s0, s1, held))
                    continue
                if low > s0:
                    inside.append((s0, low, held))
                inside.append((max(low, s0), min(high, s1), state))
                if high < s1:
                    inside.append((high, s1, held))
            wanted = inside
        # neighbours that ended up in the same state are one piece again
        merged = []
        for s0, s1, held in wanted:
            if s1 - s0 <= 1e-9:
                continue
            if merged and merged[-1][2] == held and abs(merged[-1][1] - s0) < 1e-9:
                merged[-1] = (merged[-1][0], s1, held)
            else:
                merged.append((s0, s1, held))

        changed = []
        for index, (s0, s1, held) in enumerate(merged):
            target = piece if index == 0 else self._new_piece_like(piece)
            before = int(self.state[target]) if target == piece else was
            self._set_span(target, s0, s1, held)
            if before != held or target != piece:
                self._note(target, before, held)
            changed.append(target)
        return changed

    def _new_piece_like(self, piece: int) -> int:
        """Another piece of the same stroke: same pen, same layer, no state yet."""
        self._room_for(1)
        target = self._many
        self._many += 1
        origin = int(self.origin[piece])
        self.origin[target] = origin
        self.widths[target] = self.widths[piece]
        self.layer[target] = self.layer[piece]
        self.t0[target] = self.t0[piece]
        self.t1[target] = self.t1[piece]
        if self._by_stroke is not None:
            self._by_stroke.setdefault(origin, []).append(target)
        return target

    def _local_t(self, piece: int, index: int) -> float:
        """Where point `index` of this piece's **drawn** polyline sits in the stroke.

        For a whole piece those are the stroke's own indexes. For a piece that
        was itself cut, its polyline starts at the interpolated `t0` and ends at
        `t1`, so the first and last drawn points are those fractions and the
        ones between are the stroke's vertices, shifted by one.
        """
        import math

        t0, t1 = float(self.t0[piece]), float(self.t1[piece])
        if abs(t0 - round(t0)) < 1e-9 and abs(t1 - round(t1)) < 1e-9:
            return t0 + index
        count = int(self.counts[piece])
        if index <= 0:
            return t0
        if index >= count - 1:
            return t1
        return float(int(math.floor(t0)) + index)

    def _set_span(self, piece: int, s0: float, s1: float, state: int) -> None:
        """Point a piece at the stroke's span `s0 .. s1` and give it a state.

        An integral span is a range of the arrays, which costs nothing. A
        fractional one -- a cut inside a segment -- gets its own interpolated
        polyline in the array's tail, because the painter draws a piece from
        `starts[piece]` for `counts[piece]` points and those points have to be
        next to each other.
        """
        origin = int(self.origin[piece])
        integral = (abs(s0 - round(s0)) < 1e-9 and abs(s1 - round(s1)) < 1e-9)
        if integral:
            start = int(self._stroke_start[origin]) + int(round(s0))
            count = int(round(s1)) - int(round(s0)) + 1
        else:
            start, count = self._place_span(origin, s0, s1)
        self.starts[piece] = start
        self.counts[piece] = count
        self.t0[piece] = s0
        self.t1[piece] = s1
        self._boxes.pop(piece, None)
        self.state[piece] = state
        self._recolour(piece)

    def _cut_inside_segments(self, piece: int, x0, y0, x1, y1, radius,
                             state) -> list:
        """Split a piece where the **segments** cross the brush.

        The vertex test is blind to a segment whose ends are both outside the
        capsule: a two-point straight line brushed in the middle has no vertex
        in the brush at all, and the old code left it exactly as it was -- on
        screen and in the file.
        """
        origin = int(self.origin[piece])
        first, last = float(self.t0[piece]), float(self.t1[piece])
        spans = self._capsule_spans(origin, first, last, x0, y0, x1, y1, radius)
        if not spans:
            return []
        was = int(self.state[piece])
        wanted = [(first, last, was)]
        for low, high in spans:
            out = []
            for s0, s1, held in wanted:
                if high <= s0 or low >= s1:
                    out.append((s0, s1, held))
                    continue
                if low > s0:
                    out.append((s0, low, held))
                out.append((max(low, s0), min(high, s1), state))
                if high < s1:
                    out.append((high, s1, held))
            wanted = out
        changed = []
        for index, (s0, s1, held) in enumerate(wanted):
            if s1 - s0 <= 1e-9:
                continue
            target = piece if index == 0 else self._new_piece_like(piece)
            before = int(self.state[target])
            self._set_span(target, s0, s1, held)
            if before != held or target != piece:
                self.journal_note(target, before, held)
            if held == state or held != was:
                changed.append(target)
        return changed

    def journal_note(self, piece: int, before: int, after: int) -> None:
        """Record a state change for undo, if a gesture is being recorded."""
        self._note(piece, before, after)

    def _note(self, piece: int, before: int, after: int) -> None:
        if self.journal is not None:
            self.journal.append((int(piece), int(before), int(after)))

    def put_back(self, snapshot, entries, forward: bool) -> None:
        """Undo or redo one gesture. The same code both ways.

        Undo restores the snapshot for every piece that existed when the
        gesture began, and then walks the record for the pieces it created;
        redo simply replays the record forward.
        """
        if forward:
            for piece, _before, after in entries:
                self.state[piece] = after
                self._recolour(piece)
            return
        if len(snapshot) >= 7:
            many, states, starts, counts, origins, firsts, lasts = snapshot
        else:                        # a snapshot taken before spans existed
            many, states, starts, counts, origins = snapshot
            firsts = lasts = None
        keep = min(many, self._many)
        self.state[:keep] = states[:keep]
        # The pieces the gesture created are not in the snapshot, and their
        # own record is not enough either: a piece born of a **second** split
        # within the same gesture recorded a "before" that was already the
        # gesture's own doing, so undoing it put that back and left the ink
        # decided. Measured as one piece of the right side still hidden
        # after a full undo.
        #
        # So they are looked up by where they are: a piece belongs to one
        # source stroke and covers a range of its points, and exactly one
        # piece of the snapshot covered that range before the gesture began.
        for piece in range(keep, self._many):
            origin = int(self.origin[piece])
            if firsts is not None:
                # **By the span, not by where its points sit.** A piece can
                # begin and end between two points now, so its `starts` is a
                # position in the arrays and comparing positions asks the wrong
                # question: a piece inside a hidden 40..60 was attributed to
                # whatever snapshot piece happened to start at that array
                # offset, and the rest of the hidden span reappeared in the
                # export.
                s0, s1 = float(self.t0[piece]), float(self.t1[piece])
                found = np.flatnonzero((origins == origin)
                                       & (firsts <= s0 + 1e-9)
                                       & (s1 - 1e-9 <= lasts))
            else:
                at = int(self.starts[piece])
                found = np.flatnonzero((origins == origin) & (starts <= at)
                                       & (at < starts + counts))
            if len(found):
                self.state[piece] = states[found[0]]
            else:
                self.state[piece] = DEFAULT
        self._recolour(np.arange(self._many))

    def _box_of(self, pieces) -> Tuple[float, float, float, float]:
        lo_x = lo_y = np.inf
        hi_x = hi_y = -np.inf
        for piece in pieces:
            start, count = int(self.starts[piece]), int(self.counts[piece])
            if count <= 0:
                continue
            px = self.xs[start:start + count]
            py = self.ys[start:start + count]
            if not len(px):
                continue
            pad = float(self.widths[piece]) * 0.5 + 1.0
            lo_x = min(lo_x, px.min() - pad)
            lo_y = min(lo_y, py.min() - pad)
            hi_x = max(hi_x, px.max() + pad)
            hi_y = max(hi_y, py.max() + pad)
        return (lo_x, lo_y, hi_x, hi_y)

    def reset(self) -> None:
        """Every piece back to undecided. The splits are left alone: they
        are geometry, and geometry is not a decision."""
        self.state[:self._many] = DEFAULT
        self._recolour(np.arange(self._many))

    # ---- what export asks for ------------------------------------------

    def kept_pieces(self):
        """(stroke_index, first, last) for every drawn piece, in `t` space.

        Fractions, not integer offsets: a piece cut inside a segment starts or
        ends between two points of the file, and the slice that writes it out
        interpolates there.
        """
        out = []
        for at in range(self._many):
            if not self.kept[at]:
                continue
            out.append((int(self.origin[at]), float(self.t0[at]),
                        float(self.t1[at])))
        return out

    def _base_of(self, stroke_index: int) -> int:
        held = getattr(self, '_bases', None)
        if held is None:
            # Only the pieces in use: the arrays carry spare room, and its
            # zeroed rows would otherwise claim to be stroke 0 at offset 0.
            held = self._bases = {}
            for at in range(self._many):
                origin = int(self.origin[at])
                if origin not in held:
                    held[origin] = int(self.starts[at])
                else:
                    held[origin] = min(held[origin], int(self.starts[at]))
        return held.get(stroke_index, 0)


def _segment_in_capsule(px0, py0, px1, py1, ax, ay, bx, by, radius):
    """The part of segment P0->P1 inside the brush's capsule, or None.

    The distance from a point of the segment to the brush's segment is convex
    along the segment (the capsule is convex), so there is at most one interval:
    find its minimum by golden section, then bisect each boundary. Exact to the
    last place the bisection reaches, and it costs no sampling.

    Cheap in the cases that matter: a segment outside the brush's box is
    rejected by two comparisons, and a segment with both ends inside is
    answered without any search at all.
    """
    import math

    # The brush's own box, padded by its radius.
    lo_x, hi_x = (ax if ax < bx else bx) - radius, (ax if ax > bx else bx) + radius
    lo_y, hi_y = (ay if ay < by else by) - radius, (ay if ay > by else by) + radius
    if (px0 < lo_x and px1 < lo_x) or (px0 > hi_x and px1 > hi_x):
        return None
    if (py0 < lo_y and py1 < lo_y) or (py0 > hi_y and py1 > hi_y):
        return None

    dx, dy = bx - ax, by - ay
    length = dx * dx + dy * dy

    def distance(u: float) -> float:
        x = px0 + (px1 - px0) * u
        y = py0 + (py1 - py0) * u
        if length <= 1e-12:
            return math.hypot(x - ax, y - ay)
        t = ((x - ax) * dx + (y - ay) * dy) / length
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        return math.hypot(x - (ax + t * dx), y - (ay + t * dy))

    inside_start = distance(0.0) <= radius
    inside_stop = distance(1.0) <= radius
    if inside_start and inside_stop:
        return (0.0, 1.0)

    # Twenty-four golden steps leave the minimum to 1e-6 of the segment, and
    # twenty-four bisections put each boundary inside that; a page unit is
    # four pixels at the merger's scale, so this is far below what the user can see.
    low, high = 0.0, 1.0
    for _ in range(24):
        one = low + (high - low) * 0.382
        two = low + (high - low) * 0.618
        if distance(one) < distance(two):
            high = two
        else:
            low = one
    middle = (low + high) / 2.0
    if distance(middle) > radius:
        return None

    def boundary(outside: float, within: float) -> float:
        for _ in range(24):
            half = (outside + within) / 2.0
            if distance(half) <= radius:
                within = half
            else:
                outside = half
        return within

    start = 0.0 if inside_start else boundary(0.0, middle)
    stop = 1.0 if inside_stop else boundary(1.0, middle)
    if stop <= start:
        return None
    return (start, stop)


def _clamp_to_segment(px, py, x0, y0, x1, y1):
    """The point of the segment nearest (px, py), for one point."""
    dx, dy = x1 - x0, y1 - y0
    length = dx * dx + dy * dy
    if length <= 1e-12:
        return x0, y0
    t = ((px - x0) * dx + (py - y0) * dy) / length
    t = max(0.0, min(1.0, t))
    return x0 + t * dx, y0 + t * dy


def _distance_to_capsule(px, py, x0, y0, x1, y1) -> np.ndarray:
    """Distance from each point to the brush's segment, as a vector."""
    dx, dy = x1 - x0, y1 - y0
    length = dx * dx + dy * dy
    if length <= 1e-12:
        return np.hypot(px - x0, py - y0)
    t = ((px - x0) * dx + (py - y0) * dy) / length
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def _inside_capsule(px, py, x0, y0, x1, y1, radius) -> np.ndarray:
    """Which of these points lie within `radius` of the segment.

    This is the brush's shape, not a new one: `BrushMask.stamp` draws a line
    of thickness `2*radius` with a round cap at the end, which is the set of
    points within `radius` of the segment. Nothing about the brush's size,
    its keys or its interpolation changes -- only what being inside it does.
    """
    dx, dy = x1 - x0, y1 - y0
    length = dx * dx + dy * dy
    if length <= 0.0:
        t = np.zeros(len(px))
    else:
        t = ((px - x0) * dx + (py - y0) * dy) / length
        np.clip(t, 0.0, 1.0, out=t)
    ox = px - (x0 + t * dx)
    oy = py - (y0 + t * dy)
    return (ox * ox + oy * oy) <= radius * radius


def _tidied(runs):
    """Runs with none shorter than two points.

    A piece of one point draws nothing, and a split that makes one is a
    split that loses ink. So a one-point run is given to a neighbour instead
    of becoming a piece of its own: the brush decides down to a point, and
    the geometry it leaves behind is still made of drawable pieces.
    """
    if len(runs) <= 1:
        return runs
    out = [list(run) for run in runs]
    at = 0
    while at < len(out):
        begin, end, value = out[at]
        if end - begin >= 2 or len(out) == 1:
            at += 1
            continue
        if at > 0:
            out[at - 1][1] = end          # into the run before it
            del out[at]
            at = max(0, at - 1)
        else:
            out[1][0] = begin             # into the run after it
            del out[0]
    return [tuple(run) for run in out]


def _runs(flags: np.ndarray):
    """`flags` as (begin, end, value) runs, half-open."""
    out = []
    begin = 0
    for at in range(1, len(flags)):
        if flags[at] != flags[begin]:
            out.append((begin, at, bool(flags[begin])))
            begin = at
    out.append((begin, len(flags), bool(flags[begin])))
    return out


class PageLayers:
    """Both sides of one page, as layers of pieces.

    Built once when the page is loaded. A gesture changes pieces; nothing
    here is rebuilt, re-matched or re-classified per motion event.
    """

    def __init__(self, arrays_left, arrays_right, matched_left, matched_right,
                 base: str = LEFT):
        self._open = False
        # How many times a gesture was snapshotted. One per drag is the point;
        # it used to be one per motion event.
        self.snapshots = 0
        self.sides = {
            LEFT: SideLayers(LEFT, arrays_left, matched_left, base),
            RIGHT: SideLayers(RIGHT, arrays_right, matched_right, base),
        }

    @classmethod
    def from_sides(cls, left_book, right_book, base: str = LEFT):
        """A page out of two books that may already have been used before."""
        made = cls.__new__(cls)
        made._open = False
        made.snapshots = 0
        made.sides = {LEFT: left_book, RIGHT: right_book}
        for side in (LEFT, RIGHT):
            made[side].base = base
        for side in (LEFT, RIGHT):
            made[side]._recolour(np.arange(len(made[side])))
        return made

    def reattach(self, left_arrays, right_arrays, base: str = LEFT) -> bool:
        """Both sides at once: same ink, new coordinates, same decisions."""
        ok = (self[LEFT].reattach(left_arrays)
              and self[RIGHT].reattach(right_arrays))
        if not ok:
            return False
        for side in (LEFT, RIGHT):
            book = self[side]
            if book.base != base:
                book.base = base
                book._recolour(np.arange(len(book)))
        return True

    def __getitem__(self, side):
        return self.sides[side]

    def paint(self, x0, y0, x1, y1, radius, decision):
        """One brush segment, applied to both sides at once.

        Both sides, because which side gives way is the whole question a
        decision answers: `L` over a place where both have ink keeps the
        left's there and hides the right's there, and a record of one side
        alone cannot say the second half.

        Returns the page-unit box that changed, or None if nothing did.
        """
        state = STATE_OF[decision]
        boxes = [side.paint(x0, y0, x1, y1, radius, state)
                 for side in self.sides.values()]
        boxes = [box for box in boxes if box is not None]
        if not boxes:
            return None
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    def reset(self) -> None:
        for side in self.sides.values():
            side.reset()

    def begin(self) -> None:
        """Start recording a gesture -- **once**, however many segments it has.

        A gesture **creates** pieces as well as restating them, and a record
        made only of (piece, before, after) cannot put back a piece that did
        not exist when the record started: measured, one undo left a piece of
        the right side still hidden. So the pieces are snapshotted whole.

        That snapshot used to be taken by `brush_segment`, i.e. **on every
        motion event**: on a page of 40,000 pieces that is 2 MB copied per
        event, and a drag that took a hundred events made a hundred snapshots
        -- each one also a "before" that already contained the previous
        gesture's doing, which is how an undo chain could leave ink decided.
        The window opens the gesture on press (`begin_gesture`) and a second
        `begin` while one is open does nothing.
        """
        if self._open:
            return
        self._open = True
        self.snapshots += 1
        for side in self.sides.values():
            side.journal = []
            side.snapshot = (side._many, side.state[:side._many].copy(),
                             side.starts[:side._many].copy(),
                             side.counts[:side._many].copy(),
                             side.origin[:side._many].copy(),
                             side.t0[:side._many].copy(),
                             side.t1[:side._many].copy())

    def finish(self) -> None:
        """The gesture is over (the mouse came up): the next one snapshots again."""
        self._open = False
        for side in self.sides.values():
            side.journal = None
            side.snapshot = None

    def end(self):
        """What the last segment changed, per side, **without closing** the gesture.

        The snapshot stays until `finish()`: every segment of one drag shares
        the state the page was in when the mouse went down, which is what makes
        an undo of the drag restore the page as it was *before* the drag.
        """
        out = {}
        for name, side in self.sides.items():
            if side.journal:
                # **Which book this was recorded on.** Piece indices only mean
                # something inside one book; a page whose ink had to be built
                # again (a re-pairing with a different partner, a different
                # document) has its own numbering, and applying an old record to
                # it would change pieces that have nothing to do with the
                # gesture -- which is how an undo could paint the wrong ink.
                out[name] = (side.snapshot, list(side.journal), side.book_id)
                side.journal = []
        return out

    def set_states(self, by_side, forward: bool):
        """Apply an undo/redo record. Returns the box it changed, or None.

        `None` means **nothing was applied** -- every side in the record was
        made on a book this page no longer has -- and the caller must not treat
        that as a gesture that happened.
        """
        boxes = []
        applied = False
        for name, record in by_side.items():
            side = self.sides[name]
            snapshot, entries = record[0], record[1]
            made_on = record[2] if len(record) > 2 else side.book_id
            if made_on != side.book_id:
                # The book was built again since the gesture; its indices are
                # not this book's indices. Nothing to apply, and nothing to
                # damage.
                continue
            side.put_back(snapshot, entries, forward)
            applied = True
            boxes.append(side._box_of([piece for piece, _b, _a in entries]))
        if not applied:
            return None
        boxes = [b for b in boxes if b and b[0] < b[2]]
        if not boxes:
            return None
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    def state_at(self, x: float, y: float, radius: float = 1.0):
        """What is composed at this point: {side: (layer, state, shown)}.

        For tests and for answering "what is here", without going through
        pixels.
        """
        out = {}
        for name, side in self.sides.items():
            found = []
            for stroke_index in side._near(x, y, x, y, radius):
                for piece in side.pieces_of(stroke_index):
                    start = int(side.starts[piece])
                    count = int(side.counts[piece])
                    if count <= 0:
                        continue
                    if _inside_capsule(side.xs[start:start + count],
                                       side.ys[start:start + count],
                                       x, y, x, y,
                                       radius + float(side.widths[piece])).any():
                        found.append((int(side.layer[piece]),
                                      int(side.state[piece]),
                                      bool(side.shown[piece])))
            out[name] = found
        return out
