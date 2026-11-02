"""How two pages' strokes sit on each other, colour ignored.

The user's words: "i want a heatmap for page map, don't care colour, it is not
necessary, just consider the match of the strokes".

So this measures *where the ink is* on each page -- never what colour it is --
and says how much of the two agree. It is what the page map shows for a pair:
dark where both pages have ink, one colour where only one does, and a number
that is the share of the union they agree on (Jaccard), which is 1.0 for two
identical pages and 0 for two blank ones.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

GRID_X = 60
GRID_Y = 120


def _mask(page, grid=(GRID_X, GRID_Y)) -> np.ndarray:
    """Ink occupancy of one page, on a fixed grid, in the writing area."""
    from xp.workers.merger.matching.pagealign import page_ink, reference_box

    x0, y0, x1, y1 = reference_box(page)
    width = max(x1 - x0, 1e-6)
    height = max(y1 - y0, 1e-6)
    mask = np.zeros((grid[1], grid[0]), dtype=bool)
    blocks = [np.asarray(s.points, dtype=float) for s in page_ink(page)
              if len(s.points)]
    if not blocks:
        return mask
    points = np.concatenate(blocks)
    gx = np.clip(((points[:, 0] - x0) / width * grid[0]).astype(int), 0, grid[0] - 1)
    gy = np.clip(((points[:, 1] - y0) / height * grid[1]).astype(int), 0, grid[1] - 1)
    mask[gy, gx] = True
    return mask


def _mask_of_picture(picture, box, grid=(GRID_X, GRID_Y)) -> np.ndarray:
    """Ink occupancy of a **rendered** page, on the same grid.

    A PDF side has no strokes until something converts it, so asking it for its
    ink gave an empty stand-in and called every pair 0%. Its ink is in its
    pixels, and the same test the page signature uses finds it: darker than
    0.62 of the paper's own level.
    """
    mask = np.zeros((grid[1], grid[0]), dtype=bool)
    if picture is None:
        return mask
    grey = np.asarray(picture, dtype=np.float32)
    if grey.ndim == 3:
        grey = grey.mean(axis=2)
    if not grey.size:
        return mask
    paper = float(np.percentile(grey, 90)) or 255.0
    dark = grey < 0.62 * paper
    if not dark.any():
        return mask
    ys, xs = np.nonzero(dark)
    height, width = grey.shape
    # The picture covers the box the caller named, so a pixel maps to a cell by
    # its share of the picture's own width and height.
    gx = np.clip((xs / width * grid[0]).astype(int), 0, grid[0] - 1)
    gy = np.clip((ys / height * grid[1]).astype(int), 0, grid[1] - 1)
    mask[gy, gx] = True
    return mask


def stroke_overlap(left_page, right_page, grid=(GRID_X, GRID_Y),
                   left_picture=None, right_picture=None, box=None):
    """`(score, image)` for a pair: the share they agree on, and the picture.

    The image is RGB: white where neither page has ink, dark grey where both
    do, light red where only the left does and light blue where only the right
    does -- the same convention the merge window uses for its strokes, so the
    map and the page it opens read the same way.
    """
    if left_picture is not None or right_picture is not None:
        from xp.workers.merger.matching.pagealign import reference_box

        area = box or reference_box(left_page)
        left = _mask_of_picture(left_picture, area, grid)
        right = _mask_of_picture(right_picture, area, grid)
    else:
        left = _mask(left_page, grid)
        right = _mask(right_page, grid)
    both = left & right
    left_only = left & ~right
    right_only = right & ~left
    union = left | right
    score = float(both.sum()) / float(union.sum()) if union.any() else 0.0
    image = np.full((grid[1], grid[0], 3), 255, dtype=np.uint8)
    image[left_only] = (222, 120, 120)
    image[right_only] = (120, 150, 222)
    image[both] = (70, 70, 70)
    return score, image
