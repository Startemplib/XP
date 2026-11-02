"""The freehand two-color brush: drag arbitrarily to paint a coverage mask,
and a stroke is only "captured" once every one of its points falls inside
the painted area -- partial overlap does nothing. That's the whole selection
mechanism this tool uses (no rectangles, no per-stroke clicking).
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from xp.render.drawing import to_pixels
from xp.core.document.xopp_io import Stroke

DEFAULT_RADIUS_PX = 8


class BrushMask:
    """One coverage mask per page. Call `stamp` for every drag segment as the
    mouse moves; call `fully_covers` to test a stroke against the accumulated
    paint so far."""

    def __init__(self, width_px: int, height_px: int, radius: int = DEFAULT_RADIUS_PX):
        self.width_px = max(1, width_px)
        self.height_px = max(1, height_px)
        self.radius = radius
        self._mask = np.zeros((self.height_px, self.width_px), dtype=np.uint8)

    def stamp(self, x0: float, y0: float, x1: float, y1: float) -> None:
        cv2.line(
            self._mask, (round(x0), round(y0)), (round(x1), round(y1)),
            color=255, thickness=self.radius * 2, lineType=cv2.LINE_8,
        )
        cv2.circle(self._mask, (round(x1), round(y1)), self.radius, 255, -1)

    def clear(self) -> None:
        self._mask[:] = 0

    def is_empty(self) -> bool:
        return not self._mask.any()

    def outline(self, scale: float = 1.0):
        """The painted area as polygons, in **page units**.

        `findContours` follows the mask exactly, so this is the same shape the brush painted -- a
        freehand band, a closed loop, several separate patches from one drag -- and not an
        approximation of it. That is what lets the decision be geometry: the region map stores these
        outlines and stops depending on the pixel grid the mask was drawn at.

        Holes are dropped (outer contours only): a brush decision is "this area takes this side", and
        a hole inside the paint would be a fifth state by the back door. `scale` converts the mask's
        pixels to page units.
        """
        if not self._mask.any():
            return []
        contours, _ = cv2.findContours(self._mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for contour in contours:
            points = [(float(x) / scale, float(y) / scale) for x, y in contour.reshape(-1, 2)]
            if len(points) >= 3:
                out.append(points)
        return out

    def fully_covers_points(self, points_px: List[Tuple[float, float]]) -> bool:
        if not points_px:
            return False
        xs = np.clip(np.round([p[0] for p in points_px]).astype(int), 0, self.width_px - 1)
        ys = np.clip(np.round([p[1] for p in points_px]).astype(int), 0, self.height_px - 1)
        return bool(self._mask[ys, xs].all())

    def fully_covers_stroke(self, stroke: Stroke, scale: float) -> bool:
        return self.fully_covers_points(to_pixels(stroke.points, scale))

    def fully_covers_batch(self, flat_points_px: np.ndarray, seg_starts: np.ndarray) -> np.ndarray:
        """Batched version of fully_covers_stroke for many strokes at once.

        `flat_points_px`: (N,2) pixel coords for every point of every stroke,
        concatenated in stroke order. `seg_starts`: (K,) the start index of
        each of K strokes within flat_points_px (non-decreasing, starts at 0).
        Returns a length-K bool array: whether *every* point of stroke i is
        covered.

        This exists because checking thousands of strokes one at a time (one
        fully_covers_stroke call each) on every single mouse-motion event
        during a drag was measurably slow on dense real pages -- the same
        "many tiny per-item numpy calls in a Python loop" pattern fixed in
        convert.py, fixed the same way: one big gather instead of one per
        stroke, then a segmented reduction (this is what np.minimum.reduceat
        does: reduces flags with the same 0/1 semantics as `all()`,
        per-segment, without a Python-level loop over segments).
        """
        k = len(seg_starts)
        if k == 0:
            return np.zeros(0, dtype=bool)
        if len(flat_points_px) == 0:
            return np.zeros(k, dtype=bool)

        xs = np.clip(np.round(flat_points_px[:, 0]).astype(np.int64), 0, self.width_px - 1)
        ys = np.clip(np.round(flat_points_px[:, 1]).astype(np.int64), 0, self.height_px - 1)
        covered = (self._mask[ys, xs] > 0).astype(np.uint8)

        return np.minimum.reduceat(covered, seg_starts).astype(bool)

    def as_overlay_array(self) -> np.ndarray:
        """RGBA-ish alpha layer (0/uint8) suitable for compositing a paint
        preview on top of the diff render."""
        return self._mask

    def fill_polygon(self, points_px: List[Tuple[float, float]]) -> None:
        """Fill the interior of an arbitrary closed loop into the mask, so
        drawing a loop around a cluster of strokes selects everything inside
        it -- not just the pixels the brush stroke itself touched."""
        if len(points_px) < 3:
            return
        pts = np.array([[int(round(x)), int(round(y))] for x, y in points_px], dtype=np.int32)
        cv2.fillPoly(self._mask, [pts], 255)
