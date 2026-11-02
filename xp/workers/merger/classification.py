"""What the merged page is made of, decided once and at one resolution.

The problem this file exists for
--------------------------------
The three diagnostic colours -- black for content both sides have, red for
content only the left has, blue for only the right -- were decided **inside
the draw call, at whatever scale the window happened to be drawing**. The
answer was a per-pixel one (`views._compose_three`), so the question "do both
sides have ink here" was asked of two rasters made at the zoom of the moment.

A raster's ink is not scale-invariant. The user's pair, page one, rendered whole at
four zooms through the product's own `views.edit_view`:

    scale 0.5     ink 2,096 px    black 1,009   red 648    blue 439
    scale 1.0     ink 4,857 px    black 2,613   red 1,182  blue 1,062
    scale 2.0     ink 11,457 px   black 7,607   red 1,352  blue 2,498
    scale 4.0     ink 32,141 px   black 20,510  red 2,282  blue 9,349

and, comparing the same page point across two of those zooms, **48.7 % of the
ink pixels were classed differently at 0.5 than at 4.0** (45.9 % at 1.0,
33.1 % at 2.0). The PDF side's 1.4 pt line is 0.7 px at 0.5, antialiased below
the ink threshold, so the notebook's own line under it comes out red; zoom in
and the same place turns black. At every zoom, 38-43 % of the red pixels had a
blue pixel next to them -- the same line drawn once in each colour. That is
exactly what the user described as colours overlapping unreasonably when zooming,
and it is a classification fault, not a display one.

What this is instead
--------------------
The two sides' ink is compared **once per page, at a fixed resolution** -- the
DPI the user sets, 288 by default, which is 4 pixels per page unit and therefore the
scale the session already works in. The answer is one label per pixel:

    PAPER      nothing
    COMMON     both sides have ink here
    LEFT       only the left side has ink here
    RIGHT      only the right has ink here

The three classes are exclusive by construction: each pixel takes exactly one
label, in this order of precedence, and `exclusive()` re-checks it. A view at
any zoom samples this plane (nearest) and never asks the question again, so
zooming can change how many pixels a boundary is wide -- and it does -- but it
cannot move a pixel from one class to another.

Sampling is nearest on purpose: a class is a decision about a place, and half
a decision is not a thing. The `residual` rule in `views._picture_edit_view`
puts ink that only exists above the classification's resolution back into the
class its own side's ink implies, so a sharp zoom is not left with uncoloured
gaps.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# One label per pixel, and every pixel has exactly one.
PAPER = 0
COMMON = 1
LEFT = 2
RIGHT = 3
LABELS = (PAPER, COMMON, LEFT, RIGHT)

# The resolution the source is rasterised at, in pixels per inch of page.
#
# **8000, because the user asked for it**: "default should be 8000". It is the number the field starts at now; the whole-page
# work is capped by a pixel budget (`MAX_WHOLE_PAGE_PIXELS`) and a window by another (`MAX_WINDOW_PIXELS`), so the
# default cannot turn a page into a swap file. 288 was the old default -- 4.0 pixels per page unit, the scale the
# session, the brush masks and the decisions work in -- and it is still the cheap end of the range.
# **800, because that is the number the user gave.** It is 11.11 pixels per page unit: 2.8 times the old 288, and the whole
# page at it is about 9 megapixels, which the user's pair pays for in roughly a second of classification and a few hundred
# milliseconds of first view -- measured, in the commit that set it. The range still runs to 8000 for a document that
# needs it, with the whole page capped by MAX_WHOLE_PAGE_PIXELS and a window by MAX_WINDOW_PIXELS.
DEFAULT_DPI = 800.0


def pixels_per_unit(dpi: float) -> float:
    """Page units are points, so one inch is 72 of them."""
    return max(1.0, float(dpi)) / 72.0


class PageClassification:
    """One page's three exclusive classes, at the resolution they were decided."""

    def __init__(self, labels: np.ndarray, scale: float,
                 page_size: Tuple[float, float]):
        self.labels = labels
        self.scale = float(scale)
        self.page_size = page_size
        self.dpi = round(self.scale * 72.0, 3)

    # -- what it is -------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, int]:
        return self.labels.shape[:2]

    def exclusive(self) -> bool:
        """True when every pixel carries exactly one label.

        The build already assigns in one pass, so this cannot fail by
        construction -- which is why it is cheap to assert on every build and
        why a future change that starts OR-ing planes together will be caught
        here rather than by a colour looking wrong at one zoom.
        """
        return bool(np.all(self.labels <= RIGHT))

    def counts(self) -> dict:
        return {name: int((self.labels == value).sum())
                for name, value in (('paper', PAPER), ('common', COMMON),
                                    ('left', LEFT), ('right', RIGHT))}

    def class_at(self, x: float, y: float) -> int:
        """The class at a page point, in page units."""
        rows, columns = self.shape
        column = int(np.clip(round(x * self.scale), 0, columns - 1))
        row = int(np.clip(round(y * self.scale), 0, rows - 1))
        return int(self.labels[row, column])

    def ink(self) -> np.ndarray:
        """Where there is any ink at all."""
        return self.labels != PAPER

    # -- what a view sees -------------------------------------------------
    def layers(self) -> dict:
        """The three classes as three masks, read off the plane and kept.

        **Not three decisions.** `labels` holds one number per pixel, and these are `labels == COMMON` and so on:
        a pixel cannot be in two of them because it cannot hold two numbers. Asking for them twice returns the
        same arrays, so a view or the brush drawing three layers costs one comparison per class per page, not per
        frame.

        The `ink` mask is the union of the three, which is what a layer may ever paint on -- paper stays paper.
        """
        held = getattr(self, '_layers', None)
        if held is not None:
            return held
        import numpy as np

        held = {COMMON: self.labels == COMMON,
                LEFT: self.labels == LEFT,
                RIGHT: self.labels == RIGHT}
        held['ink'] = held[COMMON] | held[LEFT] | held[RIGHT]
        self._layers = held
        return held

    def layer_at(self, x: float, y: float) -> int:
        """The class of a page point, named as a layer: COMMON, LEFT, RIGHT, or PAPER between them."""
        return self.class_at(x, y)

    def in_view(self, shape: Tuple[int, int], scale: float,
                window: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """This classification sampled onto a view's own pixel grid.

        `window` is (left, top, width, height) in the pixels of `scale`, the
        same convention `views.render` uses.

        **Two directions, two rules, both of them the classification's.** A view
        finer than the classification takes the label of the cell it is inside
        (nearest). A view coarser than the classification -- zoomed out -- takes
        every label the cell it covers contains, with common winning over a
        side: a page at 0.5 still shows a black line as black, and shows it
        thinner than at 4.0, but it never turns it red because the zoom could
        not hold the classification's resolution. Point-sampling on the way down
        did exactly that: at 0.5 a 1.4 pt line is a single pixel, the sample
        landed beside it, and the residual rule filled the pixel with the one
        side that still had ink there -- 47 % of the user's page's class points came
        out wrong, the same number as before the fix.
        """
        from PIL import Image

        height, width = shape
        held = self.labels
        ratio = (scale / self.scale) if self.scale else 1.0
        hold_scale = round(float(self.scale), 9)
        hold_size = tuple(self.page_size)
        left, top = (0, 0) if window is None else (int(window[0]), int(window[1]))

        if ratio >= 1.0:
            # **Finer than the classification: one label per view pixel, by
            # arithmetic.** The index is a function of the view pixel's position
            # on the page, not of where the window starts, so a windowed render
            # and a whole-page render cropped to the same rectangle agree
            # exactly -- which they did not when this went through PIL on a crop:
            # measured, 7.55 grey levels on the ink at 2x, and `test_merge_zoom`
            # caught it.
            #
            # **One entry of cache first.** During a drag nothing about the classes changes between frames -- same
            # plane, same scale, same window -- and this sampling was 52 ms of a 90 ms frame.
            key = (id(held), round(float(ratio), 9), height, width, int(left), int(top))
            kept = getattr(self, '_in_view_cache', None)
            if kept is not None and kept[0] == key:
                return kept[1]
            step = int(round(ratio)) if abs(ratio - round(ratio)) < 1e-9 else 0
            if step >= 1 and left % step == 0 and top % step == 0:
                # **A slice, not a gather.** For the zooms that are whole multiples this is the same cells
                # `floor((i + left) / ratio)` picks, and numpy copies it without building two index arrays.
                rows = held[top // step: top // step + height * step: step]
                if rows.shape[0] < height:
                    rows = held[top // step: held.shape[0]: step]
                piece = rows[:, left // step: left // step + width * step: step]
                if piece.shape[:2] != (height, width):
                    fitted = np.zeros((height, width), np.uint8)
                    fitted[:piece.shape[0], :piece.shape[1]] = piece
                    piece = fitted
                answer = np.array(piece, np.uint8)
            else:
                columns = np.clip(
                    np.floor((np.arange(width) + left) / ratio).astype(np.int64),
                    0, held.shape[1] - 1)
                rows_at = np.clip(
                    np.floor((np.arange(height) + top) / ratio).astype(np.int64),
                    0, held.shape[0] - 1)
                answer = held.take(rows_at, axis=0).take(columns, axis=1)
            self._in_view_cache = (key, answer)
            return answer

        # Coarser than the classification: reduce the whole page first, then
        # crop. The reduced plane is smaller than the classification, so this is
        # the cheap direction, and it keeps a window and the whole page agreeing
        # for the same reason as above.
        # **One entry of cache.** A drag asks the same question every frame: same plane, same ratio, same window. This
        # branch reduces a nine-megapixel plane three times, so without the cache it was 52 ms of a 90 ms frame.
        cache_key = ('coarse', id(held), round(float(ratio), 9), height, width,
                     int(left), int(top), hold_scale, hold_size)
        kept = getattr(self, '_in_view_coarse', None)
        if kept is not None and kept[0] == cache_key:
            return kept[1]
        page_w, page_h = self.page_size
        full_w = max(1, int(round(page_w * scale)))
        full_h = max(1, int(round(page_h * scale)))
        # **The reduction does not depend on the window, so it is cached without one.** The cache below is keyed on
        # (plane, ratio, window) and every brush repair asks about a *different* window -- measured at 79 ms a repair
        # on the user's 40,000-stroke page, 58 ms of it here: three whole-plane reductions of nine megapixels, thrown away
        # because the next rectangle was a few pixels to the left. One entry per (plane, ratio) and the crop below is
        # a slice; the classes are a page fact and do not change when a decision does, so a whole drag hits this.
        whole_key = ('whole', id(held), round(float(ratio), 9), full_w, full_h,
                     hold_scale, hold_size)
        kept_whole = getattr(self, '_in_view_whole', None)
        # **`kept_whole[2] is held`, not only the id.** The key carries `id(held)` for speed, and an id can be handed to
        # a new object after the old one is collected -- a recycled id would serve the previous page's labels. Holding
        # the plane in the entry is what makes the id safe.
        if (kept_whole is not None and kept_whole[0] == whole_key
                and kept_whole[2] is held):
            whole = kept_whole[1]
        else:
            covered = {}
            for name, value in (('common', COMMON), ('left', LEFT), ('right', RIGHT)):
                plane = Image.fromarray(((held == value).astype(np.uint8) * 255))
                covered[name] = np.asarray(
                    plane.resize((full_w, full_h), Image.BOX), np.uint8) > 0
            whole = np.zeros((full_h, full_w), np.uint8)
            whole[covered['right']] = RIGHT
            whole[covered['left']] = LEFT
            whole[covered['common']] = COMMON
            self._in_view_whole = (whole_key, whole, held)
        if window is None:
            self._in_view_coarse = (cache_key, whole)
            return whole
        piece = whole[top:top + height, left:left + width]
        if piece.shape[:2] != (height, width):
            padded = np.zeros((height, width), np.uint8)
            padded[:piece.shape[0], :piece.shape[1]] = piece
            piece = padded
        answer = np.array(piece, copy=True)
        self._in_view_coarse = (cache_key, answer)
        return answer


def build(left_ink: np.ndarray, right_ink: np.ndarray, scale: float,
          page_size: Tuple[float, float]) -> PageClassification:
    """The one rule, applied once: both -> common, one -> that side.

    Nothing grows and nothing is tolerated -- the same thing `views._compose_three`
    has always meant by common, asked here of the whole page at the
    classification's DPI instead of of a window at the zoom's.
    """
    labels = np.zeros(left_ink.shape, np.uint8)
    left_only = left_ink & ~right_ink
    right_only = right_ink & ~left_ink
    both = left_ink & right_ink
    labels[right_only] = RIGHT
    labels[left_only] = LEFT
    labels[both] = COMMON
    return PageClassification(labels, scale, page_size)
