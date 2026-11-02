"""Every stroke of a page as a few parallel arrays, on either side of the bus.

The user's instruction: *all strokes stored as contiguous arrays, not repeatedly
built as `Segment` Python objects, flattened, uploaded and downloaded.*

That is what this is. A page is two hundred thousand ribs, and as objects
each one was walked four separate times -- once to be built, once to be
flattened for a kernel, once to be uploaded, once to be formatted for the
file -- and none of those four walks did any geometry. Measured on the user's page
that is 2.9 of 15.4 seconds spent turning objects into arrays and back.

    xs, ys      float64[N]   every point of every stroke, end to end
    starts      int64[S]     where each stroke's points begin
    counts      int64[S]     how many it has
    widths      float64[S]   in image pixels, as drawn
    tint        int32[S]     which colour, as an index
    palette     list[C]      the colours themselves, on the host

The colours stay a host list on purpose. There are 6,534 of them on the user's page
against 200,000 strokes, they are what gets written into the file as text,
and a kernel that needs them numerically wants them as an array anyway --
`lab()` builds that once, on whichever side is asking.

`xp` is numpy or cupy. One class serves both, and the same code runs on
either, which is what makes "keep it resident" a property of where the
arrays happen to live rather than a second implementation.
"""

from typing import List, Optional

import numpy as np


class StrokeBuffer:
    """A page's strokes, as arrays. Immutable in use: `take` and `concat`
    return new buffers rather than editing one, because every stage either
    keeps all of its strokes or selects a subset and the selections compose.
    """

    __slots__ = ('xp', 'xs', 'ys', 'starts', 'counts', 'widths', 'tint',
                 'palette', '_lab')

    def __init__(self, xp, xs, ys, starts, counts, widths, tint, palette):
        self.xp = xp
        self.xs = xs
        self.ys = ys
        self.starts = starts
        self.counts = counts
        self.widths = widths
        self.tint = tint
        self.palette = palette
        self._lab = None

    # ---- making one ----------------------------------------------------

    @classmethod
    def empty(cls, xp=np, palette=None):
        return cls(xp, xp.zeros(0, xp.float64), xp.zeros(0, xp.float64),
                   xp.zeros(0, xp.int64), xp.zeros(0, xp.int64),
                   xp.zeros(0, xp.float64), xp.zeros(0, xp.int32),
                   list(palette or ()))

    @classmethod
    def of_pairs(cls, xp, points, widths, tint, palette):
        """`points` is (S, K, 2) -- every stroke the same length, which is
        what a level of ribs is: all two points, or all one."""
        many, each = points.shape[0], points.shape[1]
        flat = points.reshape(-1, 2)
        counts = xp.full(many, each, xp.int64)
        starts = xp.arange(many, dtype=xp.int64) * each
        return cls(xp, xp.ascontiguousarray(flat[:, 0]),
                   xp.ascontiguousarray(flat[:, 1]), starts, counts,
                   widths, tint, palette)

    @classmethod
    def of_segments(cls, segments, xp=np, default_width=1.0):
        """From the object form, for the stages that still speak it."""
        if not segments:
            return cls.empty(xp)
        held = [np.asarray(one.points, np.float64).reshape(-1, 2)
                for one in segments]
        counts = np.fromiter((len(one) for one in held), np.int64, len(held))
        both = np.concatenate(held)
        starts = np.zeros(len(counts), np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        widths = np.fromiter((default_width if one.width is None else one.width
                              for one in segments), np.float64, len(segments))
        seen, tint = {}, np.empty(len(segments), np.int32)
        for index, one in enumerate(segments):
            at = seen.get(one.color)
            if at is None:
                at = seen[one.color] = len(seen)
            tint[index] = at
        palette = [None] * len(seen)
        for colour, at in seen.items():
            palette[at] = colour
        return cls(xp, xp.asarray(np.ascontiguousarray(both[:, 0])),
                   xp.asarray(np.ascontiguousarray(both[:, 1])),
                   xp.asarray(starts), xp.asarray(counts),
                   xp.asarray(widths), xp.asarray(tint), palette)

    # ---- what it is ----------------------------------------------------

    def __len__(self):
        return int(self.starts.shape[0])

    @property
    def points_total(self) -> int:
        return int(self.xs.shape[0])

    def points(self, index):
        """One stroke's points, as a view -- no copy, no new object."""
        begin = int(self.starts[index])
        end = begin + int(self.counts[index])
        return self.xp.stack((self.xs[begin:end], self.ys[begin:end]), axis=1)

    def colour_of(self, index):
        return self.palette[int(self.tint[index])]

    def ends(self):
        """Each stroke's first and last point, as (S, 2, 2).

        Flat and in one go: this is what every spatial search starts from,
        and building it a stroke at a time is how the join used to spend a
        tenth of a second before it had looked at anything.
        """
        last = self.starts + self.counts - 1
        out = self.xp.empty((len(self), 2, 2), self.xp.float64)
        out[:, 0, 0] = self.xs[self.starts]
        out[:, 0, 1] = self.ys[self.starts]
        out[:, 1, 0] = self.xs[last]
        out[:, 1, 1] = self.ys[last]
        return out

    def lab(self):
        """The palette in Lab, for Delta-E, built once and kept.

        A colour that is a palette name rather than a measured triple has no
        place in Lab: its row is NaN, and `measured` says so. Two names are
        the same pen or they are not -- there is no distance between the word
        "black" and the word "blue".
        """
        if self._lab is None:
            import cv2

            table = np.full((len(self.palette), 3), np.nan)
            for at, colour in enumerate(self.palette):
                if (isinstance(colour, tuple) and len(colour) == 3
                        and all(isinstance(part, (int, float, np.integer,
                                                  np.floating))
                                for part in colour)):
                    patch = np.array([[list(colour)]], np.uint8)
                    table[at] = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[0, 0]
            self._lab = table
        return self._lab

    # ---- changing one --------------------------------------------------

    def take(self, keep):
        """The subset `keep` names, as a new buffer.

        `keep` is a boolean mask or an index array. The points come with it:
        one gather, built from the run lengths, with no Python loop over the
        strokes that are leaving.
        """
        xp = self.xp
        if keep.dtype == bool:
            keep = xp.nonzero(keep)[0]
        if int(keep.shape[0]) == len(self):
            return self
        counts = self.counts[keep]
        starts = xp.zeros(len(counts), xp.int64)
        if len(counts) > 1:
            xp.cumsum(counts[:-1], out=starts[1:])
        total = int(counts.sum())
        if not total:
            return StrokeBuffer.empty(xp, self.palette)
        mine = xp.repeat(xp.arange(len(counts), dtype=xp.int64), counts)
        within = xp.arange(total, dtype=xp.int64) - starts[mine]
        where = self.starts[keep][mine] + within
        held = StrokeBuffer(xp, self.xs[where], self.ys[where], starts, counts,
                            self.widths[keep], self.tint[keep], self.palette)
        held._lab = self._lab
        return held

    def with_widths(self, widths):
        held = StrokeBuffer(self.xp, self.xs, self.ys, self.starts, self.counts,
                            widths, self.tint, self.palette)
        held._lab = self._lab
        return held

    def to(self, xp):
        """The same strokes on the other side of the bus."""
        if xp is self.xp:
            return self
        move = (xp.asarray if xp is not np
                else (lambda a: a.get() if hasattr(a, 'get') else np.asarray(a)))
        held = StrokeBuffer(xp, move(self.xs), move(self.ys), move(self.starts),
                            move(self.counts), move(self.widths),
                            move(self.tint), self.palette)
        held._lab = self._lab
        return held

    # ---- the boundary with the object form ------------------------------

    def to_segments(self, Segment, widths_told=None) -> List:
        """The object form, for the stages that still speak it.

        `widths_told` keeps a width of None as None where the older encoders
        meant "the profile's own": joining and filtering are bookkeeping and
        may not change what a stroke says about itself.
        """
        host = self.to(np)
        # One block for the page, and every Segment a view into it. Building
        # an array per stroke here put back exactly the cost the buffer was
        # made to remove: 1.22s at the one place the objects are still
        # needed, which was more than the filling it came out of.
        both = np.empty((host.points_total, 2), np.float64)
        both[:, 0] = host.xs
        both[:, 1] = host.ys
        starts = host.starts.tolist()
        counts = host.counts.tolist()
        widths = host.widths.tolist()
        tint = host.tint.tolist()
        palette = host.palette
        if widths_told is None:
            widths_told = widths
        return [Segment(both[begin:begin + count], palette[colour], width)
                for begin, count, colour, width
                in zip(starts, counts, tint, widths_told)]


def concat(parts: List[StrokeBuffer], xp=None) -> StrokeBuffer:
    """Several buffers as one, with one shared palette.

    The palettes are merged rather than assumed equal, because a level's
    ribs and the dots after them are built separately and need not have met
    the same colours.
    """
    parts = [one for one in parts if len(one)]
    if not parts:
        return StrokeBuffer.empty(xp or np)
    if len(parts) == 1:
        return parts[0]
    xp = xp or parts[0].xp
    palette, where = [], []
    seen = {}
    for part in parts:
        mine = np.empty(len(part.palette), np.int32)
        for at, colour in enumerate(part.palette):
            to = seen.get(colour)
            if to is None:
                to = seen[colour] = len(palette)
                palette.append(colour)
            mine[at] = to
        where.append(xp.asarray(mine))
    counts = xp.concatenate([one.counts for one in parts])
    starts = xp.zeros(len(counts), xp.int64)
    xp.cumsum(counts[:-1], out=starts[1:])
    return StrokeBuffer(
        xp,
        xp.concatenate([one.xs for one in parts]),
        xp.concatenate([one.ys for one in parts]),
        starts, counts,
        xp.concatenate([one.widths for one in parts]),
        xp.concatenate([table[one.tint] for one, table in zip(parts, where)]),
        palette)
