"""How far a traced page is from the page it was traced from.

The reference is the page itself: a pixel is ink when it's far enough from
the measured paper color (the same cut probe.py uses), which doesn't depend
on the threshold settings being judged. Both sides are compared on one
preview-sized grid, with a small tolerance so a line traced half a pixel off
isn't reported as a difference.

Besides the share of ink lost or invented, it counts whole *marks* -- ink
blobs at least the size of a pen dot -- that vanish or appear. A full stop is
a tiny share of a page's ink, so a percentage alone would happily trade every
full stop for a few fewer fragments; counting marks is what stops that.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from xp.core import geometry
from xp.workers.converter.probe import INK_DELTA
from xp.workers.converter.profiles import classify_bgr

GRID_MAX_DIM = 2200   # longest side of the comparison grid (and of the previews)
# <~ INK_DELTA lived here too, with a comment saying "as in probe.py".
#    It is in probe.py now, once. ~>
TOLERANCE_PX = 2      # on the grid; absorbs anti-aliasing and sub-pixel offsets
MARK_AREA_FRACTION = 0.02   # bigger than this is not a pen stroke, as in probe.py
MARK_CONTRAST_SHARE = 0.25  # a mark has to be this dark, against the page's own ink


def grid_size(width, height):
    ratio = min(1, GRID_MAX_DIM / max(width, height))
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def coverage(mask, size):
    """Shrink a boolean mask onto the grid, keeping any grid pixel that had
    ink anywhere in its footprint -- nearest-neighbour would skip thin lines,
    and a plain average would fade them below any threshold."""
    return cv2.resize(mask.view(np.uint8) * np.uint8(255), size, interpolation=cv2.INTER_AREA) > 0


def pen_marks(mask):
    """Drop the page's own edge, which is not something anyone wrote.

    "Ink" here means "far from the paper colour", and the white margin a
    scanner leaves around a tinted page is exactly that: on one green-paper
    scan it is a third of the page. Left in, the reference asks the tracer to
    reproduce the scanner's border and then reports 90% of the ink as lost,
    which drowns out every real difference and misleads the tuner.

    The test is probe.py's, which drops the same regions before measuring a
    pen: a region has to be both bigger than any pen stroke and touching the
    edge of the image. Size alone would throw away a large drawing in the
    middle of the page, which is ink like any other.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.view(np.uint8), 8)
    if count <= 1:
        return mask
    height, width = mask.shape
    x, y = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    touches_edge = ((x == 0) | (y == 0)
                    | (x + stats[:, cv2.CC_STAT_WIDTH] >= width)
                    | (y + stats[:, cv2.CC_STAT_HEIGHT] >= height))
    drop = (stats[:, cv2.CC_STAT_AREA] > MARK_AREA_FRACTION * mask.size) & touches_edge
    drop[0] = True                         # label 0 is the paper itself
    return geometry.pick(~drop, labels, mask)


def source_ink(raw, paper_bgr, size, margins=True):
    """Ink in the page render itself. Summed per channel in uint16 to avoid a
    full-page int16/int64 copy of an 800-DPI render.

    `margins` says whether this image is a whole page. It is what decides
    whether `pen_marks` runs, and it is not a convenience: the margin filter
    drops a blob that touches the edge and covers more than two percent of
    the image, which is the scan's border on a page and is **the writing** on
    a crop. On the 768-pixel crop the bridge searches, every stroke runs off
    every edge and two percent is 11,796 pixels, which any letter exceeds --
    so the whole crop came back empty, 114,782 pixels of ink reduced to none,
    and every caller that measures a crop got a reference with nothing in it.

    Pass False for a crop. There is no margin inside a page.
    """
    total = np.zeros(raw.shape[:2], np.uint16)
    for channel, value in zip(cv2.split(raw), paper_bgr):
        total += cv2.absdiff(channel, np.full_like(channel, int(value)))
    ink = coverage(total > INK_DELTA, size)
    return pen_marks(ink) if margins else ink


def contrast(raw, paper_bgr, size):
    """How far from the paper each grid pixel is, on the comparison grid.

    Used to tell a mark from a blotch: a phone photo's paper is never quite
    one colour, and the compression leaves specks that clear the ink cut.
    """
    small = cv2.resize(raw, size, interpolation=cv2.INTER_AREA).astype(np.int16)
    return np.abs(small - np.array(paper_bgr, np.int16)).sum(axis=2)


def traced_ink(render, frame, size):
    """Ink in a rendered traced page, moved onto the source grid.

    `frame` (normalized x, y, w, h) is where the source page lands on the
    output page, and it is not always inside it. Two ways round:

    - fitted as a whole, the output page is letterboxed around the content
      and the frame sits inside it;
    - aligned on the template's blue frame, the writing comes out at the size
      it was written at and the source's *margins* can fall off the paper --
      on the sample document the frame maps the render to x -52.0..240.3 of a
      page 187.8 wide.

    So the rectangle is **padded, not clamped**. Clamping cropped at the
    paper's edge and then resampled that to the source's full width, which
    slides every row sideways and stretches it: on the user's page it reported 41
    whole marks lost where nothing had been lost at all. What is off the
    paper is blank, which is the truth -- the source had margin there and the
    tracing has nothing to show for it.
    """
    rgb = np.asarray(render, dtype=np.int16)
    ink = np.abs(255 - rgb).sum(axis=2) > INK_DELTA
    th, tw = ink.shape
    ox, oy, sx, sy = frame
    left, top = int(round(ox * tw)), int(round(oy * th))
    wide, high = max(1, int(round(sx * tw))), max(1, int(round(sy * th)))
    crop = np.zeros((high, wide), bool)
    x0, x1 = max(0, left), min(tw, left + wide)
    y0, y1 = max(0, top), min(th, top + high)
    if x1 > x0 and y1 > y0:
        crop[y0 - top:y1 - top, x0 - left:x1 - left] = ink[y0:y1, x0:x1]
    return coverage(np.ascontiguousarray(crop), size)


def tolerance_kernel():
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOLERANCE_PX + 1,) * 2)


def mark_area(pen_width_px, grid_ratio):
    """Smallest blob, in grid pixels, that counts as a mark: half a pen dot.
    A full stop is about one pen width across; scan dust is smaller."""
    return max(2.0, 0.5 * (pen_width_px * grid_ratio) ** 2)


@dataclass
class Difference:
    missing: np.ndarray      # source ink with no traced ink nearby
    extra: np.ndarray        # traced ink with no source ink nearby
    lost_share: float        # of all source ink
    extra_share: float       # of all traced ink
    marks: int               # marks on the page
    marks_lost: int          # marks with nothing traced anywhere near them
    marks_extra: int         # traced blobs with no source ink anywhere near them


class Reference:
    """The page's side of the comparison, computed once and reused for every
    tracing compared against it."""

    def __init__(self, source, min_mark_area, page_contrast=None):
        self.source = source
        self.kernel = tolerance_kernel()
        self.near = cv2.dilate(source.view(np.uint8), self.kernel) > 0
        count, self.labels, stats, _ = cv2.connectedComponentsWithStats(source.view(np.uint8), connectivity=8)
        self.label_count = count
        is_mark = stats[:, cv2.CC_STAT_AREA] >= min_mark_area
        is_mark[0] = False
        if page_contrast is not None and is_mark.any():
            is_mark &= self._visible(page_contrast)
        self.mark_ids = np.flatnonzero(is_mark)
        self.total = int(source.sum())

    def _visible(self, page_contrast):
        """Which blobs are dark enough to be marks rather than blotches.

        Marks decide the search: auto-tune keeps the settings that lose the
        fewest of them, before it looks at anything else. On a phone scan the
        paper's own noise clears the ink cut in a few hundred places, and
        every one of them counted as a full stop -- so the search was steered
        by specks, and chose settings that picked up more of them. The cut
        scales with the page: a quarter of the contrast its own ink reaches,
        never below the ink cut itself, so a page written in pale pencil
        still has marks.
        """
        peak = np.zeros(self.label_count, np.int32)
        # Only the ink carries a label, and np.maximum.at costs per element,
        # so walk the ink rather than the whole grid: 2% of the work.
        where = np.flatnonzero(self.source.ravel())
        values = page_contrast.ravel()[where]
        labels = self.labels.ravel()[where]
        np.maximum.at(peak, labels, values)
        # The page's ink level: each blob counted by how dark its darkest
        # pixel is, which is what a reader sees, and weighted by how much of
        # the page it covers, so that a few hundred specks cannot outvote the
        # writing the way counting blobs alone would.
        area = np.bincount(labels, minlength=self.label_count)
        order = np.argsort(peak)
        covered = np.cumsum(area[order])
        ink_level = float(peak[order][min(len(order) - 1,
                                          int(np.searchsorted(covered, 0.75 * covered[-1])))])
        return peak >= max(INK_DELTA, MARK_CONTRAST_SHARE * ink_level)

    @property
    def marks(self):
        return len(self.mark_ids)

    def compare(self, traced):
        near_traced = cv2.dilate(traced.view(np.uint8), self.kernel) > 0
        missing = self.source & ~near_traced
        extra = traced & ~self.near
        reached = np.bincount(self.labels[near_traced & self.source], minlength=self.label_count)
        marks_lost = int(np.count_nonzero(reached[self.mark_ids] == 0))
        count, labels = cv2.connectedComponents(traced.view(np.uint8), connectivity=8)
        grounded = np.bincount(labels[traced & self.near], minlength=count)
        marks_extra = int(np.count_nonzero(grounded[1:] == 0))
        traced_total = int(traced.sum())
        return Difference(
            missing, extra,
            float(missing.sum()) / self.total if self.total else 0.0,
            float(extra.sum()) / traced_total if traced_total else 0.0,
            self.marks, marks_lost, marks_extra,
        )


def ink_classes(raw, source, profile, size):
    """Which palette color the page's own ink is, per grid pixel (-1 where
    there's no ink). Only ink pixels are classified -- a whole page of paper
    would be a few million pointless classifications."""
    small = cv2.resize(raw, size, interpolation=cv2.INTER_AREA).reshape(-1, 3)
    flat = source.ravel()
    labels = np.full(len(flat), -1, np.int32)
    where = np.flatnonzero(flat)
    if len(where):
        classified, names = classify_bgr(profile, small[where])
        labels[where] = classified
    else:
        names = list(profile.hsv_ranges if profile.mode == 'hand' else profile.palette_bgr)
    return labels.reshape(source.shape), names


def class_reach(labels, kernel):
    """Per palette color, where ink of that color is, spread by the diff
    tolerance -- so a stroke traced a pixel off its ink still counts as
    sitting on it."""
    return {int(k): cv2.dilate((labels == k).view(np.uint8), kernel) > 0
            for k in np.unique(labels) if k >= 0}


class NothingWasClassified(ValueError):
    """`color_error` was handed no classified ink, so there is no share.

    It used to answer `0.0` for this, which reads as a perfect page -- and
    that is why Board card Open(1)'s baseline could not be reproduced.
    `segment_classes` matches by palette **name**, and with `colour_5d` on,
    which is `Profile.named('hand')`'s own default, a segment's colour is the
    measured BGR tuple instead. Measured on the user's Quantum notes p1 at 300 dpi:
    3,205 segments, 3,078 carrying a tuple and 127 carrying None, so **every
    one was skipped**, nothing was counted, and the answer came back
    0.0000% -- a perfect score for a measurement that never happened.

    Both callers inside the program already avoid it: the window reads the
    Colour error stage's own number when `colour_5d` is on, and the search
    turns `colour_5d` off before measuring. So this cannot fire in normal
    use. It fires for whoever measures by hand with a default profile, which
    is exactly who was misled.
    """


def color_error(traced_by_class, reach):
    """Share of traced ink drawn in a color that isn't on the page under it.

    This is what a color split fixes: two pen strokes that cross are one
    contour, so without splitting the whole thing takes the majority color
    and the minority's half is drawn in the wrong one.

    Raises `NothingWasClassified` when nothing could be counted: a metric
    that reports a perfect score for an empty measurement is worse than one
    that refuses to answer.
    """
    wrong = total = 0
    for klass, mask in traced_by_class.items():
        count = int(mask.sum())
        total += count
        near = reach.get(int(klass))
        wrong += count if near is None else count - int((mask & near).sum())
    if not total:
        raise NothingWasClassified(
            'no traced ink was classified into a palette colour, so there is '
            'no share to report. With colour_5d on a segment carries the '
            'measured BGR tuple rather than a palette name and '
            'segment_classes skips it -- measure with colour_5d off, or read '
            'the Colour error stage instead.')
    return wrong / total


def segment_classes(segments, names, image_size, grid):
    """Grid mask per palette color for the strokes a tracing would draw.
    `segments` are (points in image pixels, color name), as
    convert.traced_segments returns them."""
    width, height = image_size
    gw, gh = grid
    index = {name: i for i, name in enumerate(names)}
    masks = {}
    for segment in segments:
        points, klass = np.asarray(segment.points), index.get(segment.color)
        if klass is None:
            continue
        rows = points[:, 1].astype(np.int64) * gh // height      # centreline points are fractional
        columns = points[:, 0].astype(np.int64) * gw // width
        flat = rows * gw + columns
        mask = masks.get(klass)
        if mask is None:
            mask = masks[klass] = np.zeros(gw * gh, bool)
        mask[flat] = True
    return {k: m.reshape(gh, gw) for k, m in masks.items()}
