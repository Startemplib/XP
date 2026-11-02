"""Which strokes on one side are *the same writing* as on the other.

The old matcher compared stroke centres (`diffmatch.match_strokes`, 2pt apart
and 35% in size). That works when both sides hold the same *strokes* -- copy a
notebook, touch a few lines -- and it fails completely for the case this merger
exists for: one side was traced out of a PDF, so its "strokes" are thousands of
short pieces of raster contour, while the other side still has the pen strokes
they came from. Their centres and sizes have nothing to do with each other, so
almost nothing was classified as common and a correctly aligned page showed red
and blue where it should have shown black.

What is actually true after a good alignment is *where the ink is*: a piece of
the traced side lying on top of the other side's ink is the same writing. So
this marks each side's ink on a shared grid, gives the other side a small
dilation to absorb the half-point a calibration leaves behind, and calls a
stroke common when enough of its own ink lands on the other side's.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

# **Cells per page unit: two, i.e. half a unit per cell.** The user's pages are
# 188 x 391 units, so this is ~376 x 782 cells -- finer than a pen is wide.
# It used to be 0.5, which is *half a cell per unit* -- a cell two units
# across, 94 x 196 in all, and with the slack rounded up it called ink up to
# 3.8pt apart common. The name was right and the number was the reciprocal.
CELLS_PER_UNIT = 2.0
# The share of a stroke's own points that must land on the other side's ink.
# A third is enough for a piece that crosses a gap in the other side, and far
# above what a stroke in the wrong place gets.
COMMON_SHARE = 0.34
# **Common itself does not grow** -- common_strokes compares the two sides' ink
# masks as they stand, which is what the image path means by common. SLACK is
# what point_coverage uses for a different job: comparing one side's *points* to
# the other side's ink, where the two samplings can be half a cell out of phase.
# Measured with the point comparison strict, a pair of identical lines scored
# 63.68% instead of 100% -- so that growth absorbs a sampling phase, not the
# definition of common, and it stays.
SLACK = 1.0
# Sampling a very long traced chain at every point costs more than it can tell
# at half a unit per cell. What matters is that the count is chosen by the
# stroke's **length**, not by how many points the format happens to carry: a
# two-point line and a chain of two thousand points along the same line must
# look the same, or the two sides are judged by different rulers.
MAX_POINTS_PER_STROKE = 128
# One sample every quarter of a page unit, and never more than the cap above.
SAMPLE_STEP = 0.5


def _grid_shape(left_pages, right_pages) -> Tuple[float, float]:
    width = max(max(p.width for p in left_pages + right_pages), 1.0)
    height = max(max(p.height for p in left_pages + right_pages), 1.0)
    return width, height


def _samples(strokes: Sequence):
    """Every stroke's points on one array, with where each stroke starts.

    Returns `(points, starts, kept, radii)`: `radii` is each sample's pen half
    width in page units, so the mask can be the pen and not its centre line -- a
    picture side is compared as a *rendered* page, where a 1.4 pt pen covers
    three cells of the 0.5-unit grid.
    """
    """Every stroke's points on one array, with where each stroke starts.

    One concatenation and one `reduceat` instead of a Python loop per stroke:
    the user's page has 42,420 strokes, and a loop over them cost a second per page.

    **And which stroke each block came from.** A stroke with no points has
    nothing to measure, so it is skipped -- which makes the n'th block the
    n'th *non-empty* stroke, not the n'th stroke. The caller returns these
    indexes to `edit_colors`, which looks every stroke up by its position in
    the page's own list, so one empty stroke early on shifted the colour of
    every stroke after it by one: common ink drawn red, exclusive ink drawn
    black. The mapping is carried rather than assumed.
    """
    blocks = []
    starts = [0]
    kept = []
    radii = []
    for index, stroke in enumerate(strokes):
        points = np.asarray(getattr(stroke, 'points', ()), dtype=float)
        if len(points) == 0:
            continue
        sampled = _evenly_spaced(points)
        blocks.append(sampled)
        starts.append(starts[-1] + len(sampled))
        kept.append(index)
        widths = [float(value) for value in (getattr(stroke, 'widths', ()) or ())]
        half = max(widths) / 2.0 if widths else 0.7
        radii.append(np.full(len(sampled), max(0.25, half)))
    if not blocks:
        empty = np.zeros(0, dtype=np.int64)
        return (np.zeros((0, 2)), empty, empty, np.zeros(0))
    return (np.concatenate(blocks), np.asarray(starts[:-1], dtype=np.int64),
            np.asarray(kept, dtype=np.int64), np.concatenate(radii))


def _evenly_spaced(points: np.ndarray) -> np.ndarray:
    """Points along this stroke at one spacing, whatever the format stored.

    **The density of the file is not the density of the ink.** A two-point
    straight line carries two points; the same line traced out of a PDF
    carries hundreds. Measuring "the share of my points that landed on the
    other side" with those two rulers gave the two-point line a share of 0 or
    1 and the traced one a smooth number -- the same writing, judged
    differently on each side, which is how one side came out common and the
    other did not.

    So both are resampled along their own length: `SAMPLE_STEP` apart, capped
    at `MAX_POINTS_PER_STROKE`, endpoints kept. Cost is bounded by the cap and
    no longer depends on how the stroke was produced.
    """
    if len(points) == 1:
        return points
    lengths = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    walked = np.concatenate([[0.0], np.cumsum(lengths)])
    total = walked[-1]
    if total <= 0.0:
        return points[:1]
    wanted = int(min(MAX_POINTS_PER_STROKE,
                     max(2, np.ceil(total / SAMPLE_STEP) + 1)))
    at = np.linspace(0.0, total, wanted)
    return np.column_stack([np.interp(at, walked, points[:, 0]),
                            np.interp(at, walked, points[:, 1])])


def _cells(points: np.ndarray, width: float, height: float,
           shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    rows, columns = shape
    cols = np.clip((points[:, 0] * CELLS_PER_UNIT).astype(int), 0, columns - 1)
    rows_at = np.clip((points[:, 1] * CELLS_PER_UNIT).astype(int), 0, rows - 1)
    return rows_at, cols


def _shape(width: float, height: float) -> Tuple[int, int]:
    return (max(1, int(round(height * CELLS_PER_UNIT))),
            max(1, int(round(width * CELLS_PER_UNIT))))


def _mask(points: np.ndarray, width: float, height: float,
          radii=None) -> np.ndarray:
    mask = np.zeros(_shape(width, height), dtype=bool)
    if not len(points):
        return mask
    rows_at, cols = _cells(points, width, height, mask.shape)
    if radii is None:
        mask[rows_at, cols] = True
        return mask
    # **The pen, not the centre line.** A sample marks the cells its nib covers --
    # the same footprint the rasterised side draws -- so two pens of the same width
    # on the same line are common to the pixel, and one that is half a cell out is
    # still common because the nibs overlap. Nothing is grown beyond the nib.
    steps = np.maximum(0, np.rint(np.asarray(radii, dtype=float)
                                  * CELLS_PER_UNIT)).astype(np.int64)
    tallest = int(steps.max()) if len(steps) else 0
    rows_at = rows_at.astype(np.int64)
    cols = cols.astype(np.int64)
    tall, wide = mask.shape
    for dy in range(-tallest, tallest + 1):
        for dx in range(-tallest, tallest + 1):
            if dx * dx + dy * dy > tallest * tallest:
                continue
            take = steps >= max(abs(dx), abs(dy))
            if not take.any():
                continue
            moved_rows = np.clip(rows_at[take] + dy, 0, tall - 1)
            moved_cols = np.clip(cols[take] + dx, 0, wide - 1)
            mask[moved_rows, moved_cols] = True
    return mask


def _grown(mask: np.ndarray, slack: float = SLACK) -> np.ndarray:
    """The mask with a little room around it: a shift-OR, not a filter."""
    steps = max(1, int(round(slack * CELLS_PER_UNIT)))
    if steps <= 0 or not mask.any():
        return mask
    grown = mask.copy()
    for _ in range(steps):
        moved = grown.copy()
        moved[1:, :] |= grown[:-1, :]
        moved[:-1, :] |= grown[1:, :]
        moved[:, 1:] |= grown[:, :-1]
        moved[:, :-1] |= grown[:, 1:]
        grown = moved
    return grown


def _covered(points: np.ndarray, starts: np.ndarray, mask: np.ndarray,
             width: float, height: float) -> np.ndarray:
    """The share of each stroke's own points that lands on `mask`."""
    if len(points) == 0 or len(starts) == 0:
        return np.zeros(0)
    rows_at, cols = _cells(points, width, height, mask.shape)
    hit = mask[rows_at, cols].astype(np.float64)
    counts = np.diff(np.append(starts, len(points)))
    sums = np.add.reduceat(hit, starts) if len(starts) else np.zeros(0)
    return sums / np.maximum(counts, 1)


def sample_along(points, step: float = SAMPLE_STEP) -> np.ndarray:
    """`points` re-spaced evenly along their own length, `step` apart.

    The same ruler `_evenly_spaced` gives every stroke, exposed because the
    Manual Fit needs it for a different reason: a stroke's vertices are two dots
    where its ink is a line, and fitting two dots against the other side's
    *ink* compares two different things. Resampling both sides makes them the
    same kind of object.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points
    lengths = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    walked = np.concatenate([[0.0], np.cumsum(lengths)])
    total = walked[-1]
    if total <= 0.0:
        return points[:1]
    wanted = int(max(2, np.ceil(total / max(step, 1e-6)) + 1))
    at = np.linspace(0.0, total, wanted)
    return np.column_stack([np.interp(at, walked, points[:, 0]),
                            np.interp(at, walked, points[:, 1])])


def ink_mask(strokes: Sequence, width: float, height: float):
    """One side's ink as a boolean grid, in page units.

    The grid is the same one `common_strokes` measures on, so a colour decision
    and a "is this point on the other side's ink" question cannot disagree.
    """
    points, _starts, _kept, _radii = _samples(strokes)
    if not len(points):
        return None
    mask = _mask(points, width, height)
    return mask if mask.any() else None


def point_coverage(points, other_mask, width: float, height: float,
                   slack: float = SLACK):
    """For each point, whether the other side's ink is within `slack` of it.

    **This is what makes the colour local.** `common_strokes` answers "is this
    *stroke* common", which is the wrong question for a stroke that crosses the
    other side's writing: it turns the whole stroke black, including the ends
    that lie on nothing. The user's order (2026-10-26): "do not make a whole stroke or a
    connected structure black because two strokes cross -- only what really
    coincides is black."

    `other_mask` is `ink_mask` of the other side, already placed in the merged
    page's coordinates. Returns an array of booleans, one per point.
    """
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    if other_mask is None or not len(points):
        return np.zeros(len(points), dtype=bool)
    # **The argument used to be ignored.** `grown` was built with the module
    # default whatever the caller passed, so every measurement of "how far is
    # too far" was measuring 1.0 pt -- and a test that passed 0.5 measured 1.0
    # and agreed. The number a caller asks for is the number they get.
    grown = _grown(other_mask, slack) if slack > 0.0 else other_mask
    cells = CELLS_PER_UNIT
    columns = np.rint(points[:, 0] * cells).astype(np.int64)
    rows = np.rint(points[:, 1] * cells).astype(np.int64)
    height_cells, width_cells = grown.shape
    inside = ((columns >= 0) & (rows >= 0)
              & (columns < width_cells) & (rows < height_cells))
    answer = np.zeros(len(points), dtype=bool)
    answer[inside] = grown[rows[inside], columns[inside]]
    return answer


def common_strokes(left: List, right: List, width: float, height: float,
                   share: float = COMMON_SHARE):
    """`(matched_left, matched_right)`: indexes of the strokes that are common.

    A stroke on either side is common when enough of its ink lands on the other
    side's ink. Nothing is forced: a stroke that sits where the other side has
    nothing stays exclusive, which is what red and blue are for.
    """
    if not left or not right:
        return set(), set()
    left_points, left_starts, left_kept, left_radii = _samples(left)
    right_points, right_starts, right_kept, right_radii = _samples(right)
    if len(left_points) == 0 or len(right_points) == 0:
        return set(), set()
    left_mask = _mask(left_points, width, height, left_radii)
    right_mask = _mask(right_points, width, height, right_radii)
    if not left_mask.any() or not right_mask.any():
        return set(), set()
    # **No growth.** The other side's mask as it stands, which is what the image
    # path means by common ('left_ink & right_ink'). The growth that used to be
    # here made a notebook pair 1.0 pt apart common on this path and different on
    # the image path -- one rule, or the two views disagree about the same page.
    left_on_right = right_mask
    right_on_left = left_mask
    left_share = _covered(left_points, left_starts, left_on_right, width, height)
    right_share = _covered(right_points, right_starts, right_on_left, width, height)
    # Back to the page's own stroke numbering: `_samples` dropped the empty
    # strokes, so position `at` in the shares is stroke `kept[at]` on the page.
    matched_left = {int(left_kept[at]) for at, value in enumerate(left_share)
                    if value >= share}
    matched_right = {int(right_kept[at]) for at, value in enumerate(right_share)
                     if value >= share}
    return matched_left, matched_right
