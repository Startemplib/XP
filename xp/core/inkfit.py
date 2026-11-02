"""Fitting one page's ink onto another's, by the ink rather than by the page.

A page's declared size is not always the truth. A PDF picked up from
somewhere else can be any size, in any aspect ratio, with the writing
sitting in some margin of its own -- so scaling by the ratio of the two page
boxes puts the ink in the wrong place and at the wrong size, and the merge
view shows two copies of the same notes that never line up.

What does hold is that the same handwriting, written once, differs between
two documents by a scale and a shift. `fit` searches one uniform scale; `fit_axes`
lets x and y be scaled separately, which is what a scan cropped to a different
aspect needs -- a uniform scale then cannot put the writing on the writing at
all, and the Q base border it was printed with comes back stretched with it.
Both score by how much of the two inks actually coincide, measured on the ink
itself.

The search is deliberately small: a handful of candidate ratios taken from
the ink's own extents, each scored by how much of the two inks actually
coincide once the centres are lined up. It is a fit, not an optimisation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

GRID = 320          # longest side of the grid the overlap is measured on
TRIM = 0.02         # ignore this share of the ink at each end when boxing it
TOLERANCE_PX = 2    # on the grid: ink this close counts as coinciding
MIN_POINTS = 8      # below this there is nothing to fit to


@dataclass(frozen=True)
class Fit:
    """How to put one side's ink into the other's coordinates.

    **The two axes may be scaled differently.** A PDF that was printed, cropped
    or exported at a different aspect is not the same document stretched
    uniformly, and forcing one scale on it leaves the writing stretched in one
    direction only. `sx` and `sy` default to `scale`, so every uniform fit is
    still one number; a frame whose two sides disagree by more than the
    tolerance sets them apart, and `reason` says so.
    """

    scale: float = 1.0
    dx: float = 0.0
    dy: float = 0.0
    score: float = 0.0       # share of the two inks that coincide, 0-1
    reason: str = 'identity'
    sx: Optional[float] = None
    sy: Optional[float] = None

    @property
    def scale_x(self) -> float:
        return self.scale if self.sx is None else self.sx

    @property
    def scale_y(self) -> float:
        return self.scale if self.sy is None else self.sy

    @property
    def pen_scale(self) -> float:
        """How much a stroke's width grows: the mean of the two axes."""
        return (self.scale_x + self.scale_y) / 2

    @property
    def uniform(self) -> bool:
        return abs(self.scale_x - self.scale_y) <= 1e-9

    @property
    def is_identity(self) -> bool:
        return (self.scale_x == 1.0 and self.scale_y == 1.0
                and self.dx == 0.0 and self.dy == 0.0)

    def apply(self, x: float, y: float) -> Tuple[float, float]:
        return x * self.scale_x + self.dx, y * self.scale_y + self.dy


def _points(items) -> np.ndarray:
    """The ink, as one (N, 2) array, from strokes **or from points already**.

    The merge's Manual Fit asks both sides for their actual content, and a side
    whose content is a picture answers with sampled ink rather than with stroke
    objects (`MergeSession._content_points`). Both are point sets and the fit
    cannot tell them apart, so it does not have to: an array is taken as it is.
    """
    if isinstance(items, np.ndarray):
        if items.size == 0:
            return np.zeros((0, 2))
        return np.asarray(items, dtype=np.float64).reshape(-1, 2)
    blocks = [np.asarray(s.points, dtype=np.float64) for s in items if len(s.points)]
    if not blocks:
        return np.zeros((0, 2))
    return np.concatenate(blocks)


def ink_box(points: np.ndarray, trim: float = TRIM):
    """Where the ink actually is, ignoring the outermost `trim` of it.

    A percentile rather than a plain bounding box: one stray mark in a margin
    (a scanner's edge, a page number, a smudge) would otherwise set the size
    of the whole page and throw the ratio off.
    """
    if len(points) < MIN_POINTS:
        return None
    low = np.percentile(points, trim * 100, axis=0)
    high = np.percentile(points, 100 - trim * 100, axis=0)
    if np.any(high - low <= 0):
        return None
    return float(low[0]), float(low[1]), float(high[0]), float(high[1])


def _raster(points: np.ndarray, scale: float, offset, size, grid_scale) -> np.ndarray:
    """The ink as a small binary image, for scoring an overlap."""
    width, height = size
    image = np.zeros((height, width), np.uint8)
    if not len(points):
        return image
    x = (points[:, 0] * scale + offset[0]) * grid_scale
    y = (points[:, 1] * scale + offset[1]) * grid_scale
    inside = (x >= 0) & (y >= 0) & (x < width) & (y < height)
    if not inside.any():
        return image
    image[y[inside].astype(np.int32), x[inside].astype(np.int32)] = 255
    return image


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    """How much of the two inks coincide, allowing a couple of pixels.

    Symmetric on purpose: a scale that shrinks one side into a corner of the
    other would score well on one direction alone.
    """
    if not a.any() or not b.any():
        return 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOLERANCE_PX + 1,) * 2)
    near_a = cv2.dilate(a, kernel) > 0
    near_b = cv2.dilate(b, kernel) > 0
    a_hit = float((b.astype(bool) & near_a).sum()) / float(b.astype(bool).sum())
    b_hit = float((a.astype(bool) & near_b).sum()) / float(a.astype(bool).sum())
    return min(a_hit, b_hit)


def _candidates(source_box, target_box, initial: float) -> List[float]:
    sx0, sy0, sx1, sy1 = source_box
    tx0, ty0, tx1, ty1 = target_box
    by_width = (tx1 - tx0) / (sx1 - sx0)
    by_height = (ty1 - ty0) / (sy1 - sy0)
    base = [initial, by_width, by_height, (by_width + by_height) / 2]
    out = []
    for value in base:
        if not np.isfinite(value) or value <= 0:
            continue
        for factor in (1.0, 0.97, 1.03):
            candidate = round(value * factor, 6)
            if 0.05 <= candidate <= 20 and candidate not in out:
                out.append(candidate)
    return out or [1.0]


def fit(source_strokes: Sequence, target_strokes: Sequence, page_size,
        initial: float = 1.0) -> Fit:
    """The uniform scale and shift that put `source` onto `target` best.

    `initial` is what the page boxes alone suggest; it is one candidate among
    those taken from the ink, and it wins only if it actually overlaps best.
    """
    source_points, target_points = _points(source_strokes), _points(target_strokes)
    source_box, target_box = ink_box(source_points), ink_box(target_points)
    if source_box is None or target_box is None:
        return Fit(initial, 0.0, 0.0, 0.0, 'not enough ink to fit')

    width, height = page_size
    grid_scale = GRID / max(width, height)
    size = (max(1, int(round(width * grid_scale))), max(1, int(round(height * grid_scale))))
    target_image = _raster(target_points, 1.0, (0.0, 0.0), size, grid_scale)

    target_centre = ((target_box[0] + target_box[2]) / 2, (target_box[1] + target_box[3]) / 2)
    source_centre = ((source_box[0] + source_box[2]) / 2, (source_box[1] + source_box[3]) / 2)

    best = Fit(initial, 0.0, 0.0, 0.0, 'not fitted')
    for scale in _candidates(source_box, target_box, initial):
        # Line the ink's centres up, then let a short search take care of the
        # rest: what is left after centring is a shift of a few grid pixels.
        dx = target_centre[0] - source_centre[0] * scale
        dy = target_centre[1] - source_centre[1] * scale
        for shift_x, shift_y in _shifts(grid_scale):
            offset = (dx + shift_x, dy + shift_y)
            score = _overlap(_raster(source_points, scale, offset, size, grid_scale), target_image)
            if score > best.score:
                best = Fit(scale, offset[0], offset[1], score, 'fitted to the ink')
    return best


def fit_axes(source_strokes: Sequence, target_strokes: Sequence, page_size,
             initial: float = 1.0) -> Fit:
    """Like `fit`, but the two axes may be scaled **independently**.

    The user's order: "the freedom is not enough -- it does not allow x and y to be
    scaled differently", about PDFs whose Q base border comes back stretched.
    A scan cropped to a different aspect is exactly that, and a uniform scale can
    then never put the writing on the writing.

    The search stays small: for each axis, three ratios taken from the ink's own
    extents (width, height, the uniform candidate), each nudged by 3 %, then the
    best pair by overlap with a short shift search. The centre is lined up first,
    so what is left is a shift of a couple of grid pixels.
    """
    source_points, target_points = _points(source_strokes), _points(target_strokes)
    source_box, target_box = ink_box(source_points), ink_box(target_points)
    if source_box is None or target_box is None:
        return Fit(initial, 0.0, 0.0, 0.0, 'not enough ink to fit')

    width, height = page_size
    grid_scale = GRID / max(width, height)
    size = (max(1, int(round(width * grid_scale))),
            max(1, int(round(height * grid_scale))))
    target_image = _raster(target_points, 1.0, (0.0, 0.0), size, grid_scale)
    target_centre = ((target_box[0] + target_box[2]) / 2,
                     (target_box[1] + target_box[3]) / 2)
    source_centre = ((source_box[0] + source_box[2]) / 2,
                     (source_box[1] + source_box[3]) / 2)

    def spread(values):
        out = []
        for value in values:
            if not np.isfinite(value) or value <= 0:
                continue
            for factor in (1.0, 0.97, 1.03):
                candidate = round(float(value) * factor, 6)
                if 0.05 <= candidate <= 20 and candidate not in out:
                    out.append(candidate)
        return out or [1.0]

    widths = spread(_candidates(source_box, target_box, initial))
    heights = spread([(target_box[3] - target_box[1]) / (source_box[3] - source_box[1]),
                      (target_box[2] - target_box[0]) / (source_box[2] - source_box[0]),
                      initial])
    best = Fit(initial, 0.0, 0.0, 0.0, 'not fitted')
    for scale_x in widths:
        for scale_y in heights:
            dx = target_centre[0] - source_centre[0] * scale_x
            dy = target_centre[1] - source_centre[1] * scale_y
            for shift_x, shift_y in _shifts(grid_scale):
                offset = (dx + shift_x, dy + shift_y)
                image = _raster_axes(source_points, scale_x, scale_y, offset,
                                     size, grid_scale)
                score = _overlap(image, target_image)
                if score > best.score:
                    best = Fit(scale_x, offset[0], offset[1], score,
                               'fitted to the ink, each axis separately',
                               sx=scale_x, sy=scale_y)
    return best


def _raster_axes(points: np.ndarray, scale_x: float, scale_y: float, offset,
                 size, grid_scale) -> np.ndarray:
    width, height = size
    image = np.zeros((height, width), np.uint8)
    if not len(points):
        return image
    x = (points[:, 0] * scale_x + offset[0]) * grid_scale
    y = (points[:, 1] * scale_y + offset[1]) * grid_scale
    inside = (x >= 0) & (y >= 0) & (x < width) & (y < height)
    if not inside.any():
        return image
    image[y[inside].astype(np.int32), x[inside].astype(np.int32)] = 255
    return image


def _shifts(grid_scale: float):
    """Small offsets around the centred position, in page units."""
    step = 2.0 / grid_scale
    return [(0.0, 0.0)] + [(x * step, y * step)
                           for x in (-1, 0, 1) for y in (-1, 0, 1) if (x, y) != (0, 0)]
