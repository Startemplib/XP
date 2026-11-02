"""Ink as geometry: the sub-pixel boundary of a region, and the curves inside it.

Everything before this module is image processing -- denoise, sharpen, an
adaptive threshold, an opening and a closing -- and it ends with a clean
binary region. What this module does is the last bridge: given that region
as a set, lay curves down that reproduce it.

The point is that a tracing should not be read off the pixel grid. A mask's
`findContours` boundary runs along pixel *centres* in whole pixels, so at
800 DPI every slope comes back as a one-pixel staircase, and a ring drawn on
it is visibly serrated where the writing is smooth. And peeling a region by
thresholding its distance transform and tracing the result again asks the
grid the same question once per layer, so every layer inherits the same
staircase and breaks into short pieces wherever the staircase turns.

So the region is treated as the level set of a continuous field instead:

    F(x) = d(x, paper) - d(x, ink)     positive inside the ink

`F = 0` is the region's own boundary, and `F = t` is everything at least `t`
inside it -- the inward offset curve at distance `t`, which is what an onion
fill wants. Both come out of the same routine, `iso_loops`, at sub-pixel
accuracy and as whole closed loops however the shape branches: a curve that
splits in two as it moves inwards simply comes back as two loops at that
level, and one that has shrunk to nothing comes back as no loop. That is the
topology handling a per-layer `findContours` has to be told about.

The remaining staircase -- half a pixel, since a crossing between an ink
pixel and a paper pixel is interpolated to the midpoint between their
centres -- is taken off by `constrain_smooth`, which is a smoothing with a
hard bound: no point may end up further than `max_shift` from where the
field put it. A plain moving average has no such bound and will round off a
sharp corner of the writing; this cannot, so it can be used on the outer
boundary, where fidelity is the whole point.
"""
from __future__ import annotations

import cv2
import numpy as np

from xp.gpu import backend as gpu
# The four corners of a cell, in cyclic order, as (dy, dx) from its top-left:
# top-left, top-right, bottom-right, bottom-left. Edge k joins corner k to
# corner k + 1, so edge 0 is the top, 1 the right, 2 the bottom, 3 the left.
CORNERS = ((0, 0), (0, 1), (1, 1), (1, 0))


def _segment_table():
    """For each of the 16 corner patterns, which cell edge leads to which.

    Walking a cell's boundary in the cyclic order above, the contour leaves
    the region at an edge whose first corner is inside and whose second is
    not, and re-enters at an edge the other way round. Those alternate, so
    pairing each leaving edge with the next entering edge after it gives the
    segments through this cell -- with the region always on the same side of
    the direction of travel, which is what makes a loop's signed area tell
    an outer boundary from a hole.

    A cell with two opposite corners inside (a saddle) has four crossings and
    two possible pairings; both are returned, and `iso_loops` picks by the
    value at the middle of the cell -- joined when the middle is inside,
    separated when it is not.
    """
    table = {}
    for pattern in range(16):
        inside = [(pattern >> k) & 1 for k in range(4)]
        leaving = [k for k in range(4) if inside[k] and not inside[(k + 1) % 4]]
        entering = [k for k in range(4) if not inside[k] and inside[(k + 1) % 4]]
        if not leaving:
            table[pattern] = ((), ())
            continue
        after = tuple((k, min((e for e in entering if e > k), default=entering[0]))
                      for k in leaving)
        before = tuple((k, max((e for e in entering if e < k), default=entering[-1]))
                       for k in leaving)
        table[pattern] = (after, before)
    return table


SEGMENTS = _segment_table()


def pick(table, labels, inside):
    """`table[labels]`, asked of the ink instead of of the whole page.

    Every "keep these components" step in the pipeline ends the same way: a
    small boolean table, one entry per connected component, indexed by the
    label image. At 800 DPI the label image is 691 MB and the answer is 173 MB,
    and the question is about the two percent of the page that is ink:

        table[labels]                 0.501s
        np.take(table, labels)        0.832s   (worse)
        this                          0.147s   the same array
        on the card                   0.006s, and 0.100s to upload the labels
                                      -- so not until the labelling is there

    It is the same array and not an equivalent one. `inside` is the mask that
    was labelled, so every pixel of it carries a label of 1 or more, and
    `table[0]` is False at every one of these sites -- label 0 is the paper.
    So `table[labels]` is already False everywhere outside `inside`, and
    filling only `inside` leaves the rest where it already was.
    """
    out = np.zeros(labels.shape, bool)
    out[inside] = table[labels[inside]]
    return out


def signed_distance(mask, smooth: float = 0.0):
    """`d(x, paper) - d(x, ink)`: positive inside the ink, zero on its edge.

    The page is padded with a pixel of paper first. Without it, ink running
    off the edge of the image has no background to measure to and the
    distance transform reports it as deep inside the region -- a stroke at
    the margin came back as a solid block.

    `smooth` blurs the field, which takes the grid out of the boundary at the
    cost of moving it by about sigma squared times its curvature -- nothing
    at handwriting scale, but it is a parameter rather than a constant
    because it is a fidelity trade and belongs to whatever is measuring it.
    """
    mask = np.ascontiguousarray(mask).astype(bool)
    padded = cv2.copyMakeBorder(mask.view(np.uint8), 1, 1, 1, 1,
                                cv2.BORDER_CONSTANT, value=0)
    field = _both_ways(padded)
    if smooth > 0:
        size = 2 * int(round(3 * smooth)) + 1
        field = cv2.GaussianBlur(field, (size, size), smooth,
                                 borderType=cv2.BORDER_REPLICATE)
    return field[1:-1, 1:-1]


def _both_ways(padded):
    """The two transforms, on the card when there is one.

    This is the single most expensive per-pixel step of a page -- 2.1s of
    cv2 against 0.062s of `cupyx`, measured on the user's 173-million-pixel page.

    Both sides compute the *same* number, and that matters more than the
    speed. cv2's `DIST_L2` over a 5x5 mask is an approximation, out by up to
    1.37 px against the true Euclidean distance -- and this field is what
    every stroke's width and every legality test is read from, so a page
    traced with a card and the same page traced without one came out with
    different strokes. `DIST_MASK_PRECISE` is exact, bit for bit identical to
    `cupyx`'s on a page of discs, and costs 1.5x the approximation. So the
    card makes this faster and nothing else: it cannot make it different.
    """
    if gpu.GPU_AVAILABLE:
        try:
            import cupyx.scipy.ndimage as cnd

            on = gpu.xp.asarray(padded.view(np.uint8) > 0)
            # float32, and one transform at a time. Two reasons, both
            # measured. `distance_transform_edt` answers in float64 by
            # default, and a page is 173 million pixels: handing that back as
            # it stands is 1.38 GB across the bus, which turned a 2.1s step
            # into a 6.0s one. And holding two of them at once is 2.76 GB on
            # the card, which is what made the same page take 9.5s one run
            # and 141.8s the next -- the card had the memory, and then it did
            # not. float32 is not an approximation here: the distance is the
            # square root of an integer, and float32 holds those exactly to
            # well past a page's diagonal.
            field = cnd.distance_transform_edt(on, float64_distances=False)
            away = cnd.distance_transform_edt(~on, float64_distances=False)
            field -= away
            del away, on
            out = gpu.to_host(field)
            del field
            gpu.xp.get_default_memory_pool().free_all_blocks()
            return out
        except Exception:
            pass
    inside = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform(1 - padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return inside - outside


def iso_loops(field, level: float, min_points: int = 4):
    """The closed curves where `field` crosses `level`, in pixel coordinates.

    Marching squares, with each crossing placed on its grid edge by linear
    interpolation -- so a boundary between an ink pixel and a paper pixel
    lands exactly between their centres, not on one of them. A crossing sits
    on a grid edge and a grid edge belongs to at most two cells, so every
    crossing has one segment arriving and one leaving: the segments link into
    closed loops with no choices to make and no seam to leave.

    Returns a list of (n, 2) float arrays of (x, y), each closed -- the first
    point repeated as the last -- so that stroking one leaves no gap where it
    began. `field[i, j]` is the value at (x=j, y=i).
    """
    field = np.ascontiguousarray(field, dtype=np.float32)
    height, width = field.shape
    if height < 2 or width < 2:
        return []
    inside = field > level
    # Bool is one byte, so a bool slice viewed as uint8 is the same memory --
    # the corners cost nothing to read and the pattern is built in place.
    pattern = np.zeros((height - 1, width - 1), np.uint8)
    for bit, (dy, dx) in enumerate(CORNERS):
        pattern |= inside[dy:height - 1 + dy, dx:width - 1 + dx].view(np.uint8) << bit
    pattern = pattern.ravel()
    # A cell matters when some corners are inside and some are out, i.e. the
    # pattern is neither 0 nor 15. Subtracting one in uint8 wraps 0 round to
    # 255 and leaves 15 at 14, so one comparison finds them all.
    active = np.flatnonzero(np.subtract(pattern, 1, dtype=np.uint8) < 14)
    if not len(active):
        return []

    cells = width - 1
    cell_row, cell_col = active // cells, active % cells
    # A saddle is joined when the middle of the cell is inside the level.
    middle = np.zeros(len(active), bool)
    saddle = (pattern[active] == 5) | (pattern[active] == 10)
    if saddle.any():
        rows, columns = cell_row[saddle], cell_col[saddle]
        average = (field[rows, columns] + field[rows, columns + 1]
                   + field[rows + 1, columns + 1] + field[rows + 1, columns]) / 4.0
        middle[saddle] = average > level

    # An edge's identity, so the two cells that share it agree: horizontal
    # edges are numbered by their left end, vertical ones by their top end.
    plane = height * width

    def key_of(row, column, edge):
        if edge == 0:
            return row * width + column
        if edge == 2:
            return (row + 1) * width + column
        if edge == 1:
            return plane + row * width + column + 1
        return plane + row * width + column

    leaving, entering = [], []
    for value in range(1, 15):
        chosen = pattern[active] == value
        if not chosen.any():
            continue
        after, before = SEGMENTS[value]
        rows, columns = cell_row[chosen], cell_col[chosen]
        if value in (5, 10):
            joined = middle[chosen]
            for pairs, take in ((after, ~joined), (before, joined)):
                if not take.any():
                    continue
                for out_edge, in_edge in pairs:
                    leaving.append(key_of(rows[take], columns[take], out_edge))
                    entering.append(key_of(rows[take], columns[take], in_edge))
            continue
        for out_edge, in_edge in after:
            leaving.append(key_of(rows, columns, out_edge))
            entering.append(key_of(rows, columns, in_edge))
    if not leaving:
        return []
    leaving = np.concatenate(leaving)
    entering = np.concatenate(entering)

    keys, index = np.unique(np.concatenate([leaving, entering]), return_inverse=True)
    index = index.ravel()
    count = len(leaving)
    successor = np.full(len(keys), -1, np.int32)
    successor[index[:count]] = index[count:]

    # Where each crossing sits on its own grid edge.
    vertical = keys >= plane
    flat = np.where(vertical, keys - plane, keys)
    rows, columns = flat // width, flat % width
    here = field[rows, columns]
    other = np.where(vertical, field[np.minimum(rows + 1, height - 1), columns],
                     field[rows, np.minimum(columns + 1, width - 1)])
    span = other - here
    step = np.where(span != 0, (level - here) / np.where(span != 0, span, 1.0), 0.0)
    step = np.clip(step, 0.0, 1.0)
    x = np.where(vertical, columns, columns + step).astype(np.float64)
    y = np.where(vertical, rows + step, rows).astype(np.float64)

    following = successor.tolist()
    seen = bytearray(len(following))
    loops = []
    for start in range(len(following)):
        if seen[start] or following[start] < 0:
            continue
        walk = []
        node = start
        while node >= 0 and not seen[node]:
            seen[node] = 1
            walk.append(node)
            node = following[node]
        if len(walk) < min_points:
            continue
        order = np.array(walk, np.int64)
        loop = np.empty((len(order) + 1, 2), np.float64)
        loop[:-1, 0], loop[:-1, 1] = x[order], y[order]
        loop[-1] = loop[0]                     # closed: no seam to stroke over
        loops.append(loop)
    return loops


def signed_area(loop) -> float:
    """Shoelace area. Loops from `iso_loops` are consistently wound, so an
    outer boundary and a hole inside it come back with opposite signs."""
    points = np.asarray(loop, dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def constrain_smooth(loop, max_shift: float = 0.35, passes: int = 3):
    """Take the grid off a closed loop without letting it move far.

    Each pass is one step of Laplacian smoothing around the loop, and after
    each one every point that has drifted further than `max_shift` from where
    the field put it is pulled straight back to that distance. So the
    staircase goes -- it is half a pixel of zig-zag, well inside the bound --
    while a corner of the writing, which would need to move much further to
    be rounded off, cannot be. The bound is on the displacement of each
    point, not on the average, so no part of the boundary can quietly walk
    away over many passes.
    """
    points = np.asarray(loop, dtype=np.float64)
    if len(points) < 5 or max_shift <= 0 or passes <= 0:
        return points
    origin = points[:-1]                        # the loop without its repeat
    current = origin.copy()
    for _ in range(passes):
        neighbours = np.roll(current, 1, axis=0) + np.roll(current, -1, axis=0)
        current = 0.5 * current + 0.25 * neighbours
        drift = current - origin
        distance = np.hypot(drift[:, 0], drift[:, 1])
        far = distance > max_shift
        if far.any():
            current[far] = origin[far] + drift[far] * (max_shift / distance[far, None])
    out = np.empty_like(points)
    out[:-1], out[-1] = current, current[0]
    return out


def simplify(loop, epsilon: float):
    """Drop the points that sit on a straight run, and nothing else.

    Sub-pixel by design: `epsilon` is a fraction of a pixel, so this takes
    out the redundancy a dense marching-squares loop has along a straight
    edge without tidying the writing. A loop that would collapse below three
    points is handed back as it was.
    """
    points = np.asarray(loop, dtype=np.float32).reshape(-1, 1, 2)
    if epsilon <= 0 or len(points) < 4:
        return np.asarray(loop, dtype=np.float64)
    reduced = cv2.approxPolyDP(points, float(epsilon), True).reshape(-1, 2)
    if len(reduced) < 3:
        return np.asarray(loop, dtype=np.float64)
    out = np.empty((len(reduced) + 1, 2), np.float64)
    out[:-1], out[-1] = reduced, reduced[0]
    return out


def offset_levels(field, first: float, spacing: float, limit: int):
    """The distances to take inward offset curves at, for this field.

    From the ink itself rather than from a setting: the deepest point of the
    region says how many curves there is room for, so a hairline gets none
    and a broad stroke gets as many as it needs. A fixed count would leave
    thick strokes hollow and spend curves on thin ones.
    """
    if spacing <= 0:
        return []
    deepest = float(field.max()) if field.size else 0.0
    levels = []
    level = first
    while level <= deepest and len(levels) < limit:
        levels.append(level)
        level += spacing
    return levels
