"""Trace pen strokes down their middle instead of around their edge.

Contour tracing draws the *boundary* of the ink, so every pen stroke comes
back as two lines with a hollow middle. At handwriting size those two lines
sit a pixel or two apart, merge into a blob, and fill in the counters of the
letters -- the writing stops being readable, which is exactly what it looked
like on real notes.

Tracing the middle instead gives one line per pen stroke, drawn at the width
the pen actually had. It also samples colour from the middle of the ink
rather than its anti-aliased edge, where a pale stroke reads as grey.

The skeleton comes from Zhang-Suen thinning, run at a scale where the pen is
about TARGET_PEN_PX across -- thinning costs a pass per pixel of half-width,
so thinning a 20px pen at full resolution is many passes over a huge image
for a result no more accurate than thinning a 3px one.
"""
from __future__ import annotations

import cv2
import numpy as np
from xp.core import geometry
# Clockwise from north; the reverse of direction k is direction (k + 4) % 8.
STEPS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
TARGET_PEN_PX = 2.0
# However wide the pen, never thin at less than a third of full size: past
# that, small characters collapse into each other before they are traced
# (a scanned page went from 17% of its ink unreached to 26%).
MAX_REDUCE = 3
MIN_POINTS = 2


def pen_width(mask):
    """Stroke width from area and perimeter: a stroke of length L and width w
    covers L*w and its outline runs 2L, so twice the area over the perimeter
    is w. Cheaper than thinning and, unlike a percentile of the distance
    transform, not dragged upwards by blobs and filled shapes."""
    contours, _ = cv2.findContours(mask.view(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    return 2 * float(mask.sum()) / perimeter if perimeter else 1.0


def _peel_tables():
    """For each of the 256 neighbourhoods, may this pixel be peeled?

    Built once. The bit order is the order of STEPS, so bit i is the
    neighbour STEPS[i], which is what makes the crossing count -- transitions
    around the ring -- a rotation of the bits.
    """
    values = np.arange(256)
    bits = (values[:, None] >> np.arange(8)) & 1
    neighbours = bits.sum(axis=1)
    crossings = ((bits == 0) & (np.roll(bits, -1, axis=1) == 1)).sum(axis=1)
    common = (neighbours >= 2) & (neighbours <= 6) & (crossings == 1)
    even = ((bits[:, 0] * bits[:, 2] * bits[:, 4] == 0)
            & (bits[:, 2] * bits[:, 4] * bits[:, 6] == 0))
    odd = ((bits[:, 0] * bits[:, 2] * bits[:, 6] == 0)
           & (bits[:, 0] * bits[:, 4] * bits[:, 6] == 0))
    return (common & even), (common & odd)


PEEL_EVEN, PEEL_ODD = _peel_tables()


def thin(mask):
    """Zhang-Suen thinning: peel boundary pixels that don't break the shape,
    until what's left is one pixel wide.

    Only the pixels that can change are looked at. Whether a pixel may be
    peeled depends on itself and its eight neighbours and on nothing else, so
    a pixel whose neighbourhood did not change in the last round cannot have
    become peelable in this one. The work is therefore a shrinking list of
    candidates, not a sweep.

    That matters more than it sounds. The user's page is 0.6% ink inside a bounding
    box of 170 million pixels and takes about ninety rounds; sweeping the box
    spent all of it on paper, and this one function was 45 of the 85 seconds
    the whole shape stage cost. The answer is identical -- the rule is
    untouched, and `test_centerline` checks the two agree pixel for pixel.
    """
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return np.zeros_like(mask, dtype=bool)
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    height, width = bottom - top, right - left
    padded = np.zeros((height + 2, width + 2), np.uint8)
    padded[1:-1, 1:-1] = mask[top:bottom, left:right]
    flat = padded.ravel()
    stride = width + 2
    offsets = np.array([dy * stride + dx for dy, dx in STEPS], np.int64)

    active = (np.flatnonzero(padded[1:-1, 1:-1].ravel()) // width + 1) * stride         + (np.flatnonzero(padded[1:-1, 1:-1].ravel()) % width + 1)
    tables = (PEEL_EVEN, PEEL_ODD)
    while len(active):
        touched = []
        for step in (0, 1):
            if not len(active):
                break
            # The eight neighbours packed into one byte, then a 256-entry
            # table says whether that arrangement may be peeled. The rule is
            # identical to sweeping the page; what changes is how many
            # pixels are asked.
            code = np.zeros(len(active), np.uint8)
            for index, offset in enumerate(offsets):
                code |= flat[active + offset] << index
            drop = (flat[active] == 1) & tables[step][code]
            if drop.any():
                dropped = active[drop]
                flat[dropped] = 0
                touched.append(dropped)
                if step == 0:
                    # A drop can make a neighbour peelable in the very next
                    # half-pass, so those join the list before it runs.
                    grown = (dropped[:, None] + offsets[None, :]).ravel()
                    active = np.union1d(active[~drop], grown[flat[grown] == 1])
                else:
                    active = active[~drop]
        if not touched:
            break
        changed = np.concatenate(touched)
        grown = (changed[:, None] + offsets[None, :]).ravel()
        active = np.unique(grown[flat[grown] == 1])

    out = np.zeros_like(mask, dtype=bool)
    out[top:bottom, left:right] = padded[1:-1, 1:-1].astype(bool)
    return out


def _edges(skeleton):
    """Which of its eight neighbours each skeleton pixel connects to.

    A diagonal step is dropped when the two pixels are already joined through
    a shared side neighbour: in a thinned staircase every pixel would
    otherwise look like a junction, and the walk below would shatter each
    stroke into two-pixel pieces.

    Read as slices of a padded copy, not with `np.roll`. Rolling makes a copy
    of the page for every direction and there are two dozen of them here --
    on an 800-DPI page that was three seconds of memory traffic to answer a
    question about one pixel in fifty. The padding also settles the edge of
    the image: a roll wraps, so a pixel on the last row came back joined to
    one on the first and `paths` followed that edge off the end of the array
    (`IndexError: index 1048614 is out of bounds for axis 1 with size
    1048576`, on a crop whose ink ran to its edge). A ring of blank cannot
    wrap, so the trimming that used to guard it is gone with the rolls.
    """
    height, width = skeleton.shape
    padded = np.zeros((height + 2, width + 2), bool)
    padded[1:-1, 1:-1] = skeleton

    def shifted(dy, dx):
        return padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]

    present = [shifted(dy, dx) & skeleton for dy, dx in STEPS]
    for index, (dy, dx) in enumerate(STEPS):
        if dy and dx:
            # Already joined through a shared side neighbour: not a step.
            present[index] = present[index] & ~(shifted(dy, 0) | shifted(0, dx))
    return np.stack([p.ravel() for p in present])


def paths(skeleton, min_points=MIN_POINTS):
    """Walk the skeleton into polylines, each edge exactly once, cutting at
    junctions and ends -- so a pen stroke comes back as one path.

    The eight directions are packed into one byte per pixel, for the edges
    that exist and again for the edges already walked. The walk then asks
    "which way can I still go" with two scalar reads and a bit trick instead
    of sixteen numpy lookups, and it asks it once per point of every path on
    the page -- a few million times. That packing was five of the shape
    stage's thirty-five seconds.
    """
    height, width = skeleton.shape
    edges = _edges(skeleton)
    degree = edges.sum(axis=0)
    offsets = np.array([dy * width + dx for dy, dx in STEPS])
    steps = offsets.tolist()

    code = np.zeros(edges.shape[1], np.uint8)
    for direction in range(8):
        code |= edges[direction].astype(np.uint8) << direction
    walked = np.zeros_like(code)
    found = []

    def walk(start, direction):
        path = [start]
        current, heading = start, direction
        while True:
            back = (heading + 4) % 8
            nxt = current + steps[heading]
            walked[current] |= 1 << heading
            walked[nxt] |= 1 << back
            current = nxt
            path.append(current)
            if degree[current] != 2:
                return path
            open_ways = int(code[current]) & ~int(walked[current]) & 0xFF
            if not open_ways:
                return path
            heading = (open_ways & -open_ways).bit_length() - 1
        return path

    pixels = np.flatnonzero(skeleton.ravel())
    ends = pixels[degree[pixels] != 2]
    for order in (ends, pixels):          # ends and junctions first, then loops
        for pixel in order.tolist():
            open_ways = int(code[pixel]) & ~int(walked[pixel]) & 0xFF
            while open_ways:
                lowest = open_ways & -open_ways
                found.append(walk(pixel, lowest.bit_length() - 1))
                open_ways = int(code[pixel]) & ~int(walked[pixel]) & 0xFF
    found = [p for p in found if len(p) >= min_points]
    # A full stop thins to a single pixel with nothing to walk to. Dropping
    # those lost every dot on the page -- 205 marks on one scanned page.
    found.extend([[int(pixel)] for pixel in pixels[degree[pixels] == 0]])
    return [np.array(p) for p in found]


def trace(mask, pen=None, target=TARGET_PEN_PX, width_from=None):
    """Centrelines of everything in `mask`, in the mask's own pixels.

    Returns [(points as (n, 2) float x/y, width in pixels)] -- one entry per
    pen stroke, each with the width its own ink had.

    `width_from` is the mask the *widths* are measured on, when that is not
    the same as the one being traced. It is for the leftovers of a fill: what
    is handed here is then the part of a mark no curve reached, and measuring
    a stroke's width on that gives the width of the gap rather than the width
    of the ink -- so the line drawn down the middle of a mark's core is too
    thin to meet the band already drawn around its edge, and the mark comes
    back hollow. Measured on the user's phone scan, that was 10.5% of the ink more
    than two pixels deep. Twice the distance to the paper is the ink's own
    width there, so a line drawn at it reaches the edge and no further.
    """
    pen = pen or pen_width(mask)
    measured = mask if width_from is None else width_from
    reduce = min(MAX_REDUCE, max(1, int(round(pen / target))))

    # Marks no bigger than a couple of pen widths -- full stops, accents, the
    # dot on an i -- are traced as single points before anything is reduced.
    # Thinning can't help them (they are already a dot) and reducing can lose
    # them outright, which is how a scanned page lost 334 of its marks.
    dots = []
    if mask.any():
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.view(np.uint8), connectivity=8)
        widest = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
        is_dot = np.zeros(count, bool)
        is_dot[1:] = widest[1:] <= max(2.0, 2 * pen)
        if is_dot.any():
            for index in np.flatnonzero(is_dot):
                area = float(stats[index, cv2.CC_STAT_AREA])
                dots.append((centroids[index].reshape(1, 2), 2 * float(np.sqrt(area / np.pi))))
            mask = geometry.pick(~is_dot, labels, mask)

    if reduce > 1:
        size = (max(1, mask.shape[1] // reduce), max(1, mask.shape[0] // reduce))
        small = cv2.resize(mask.view(np.uint8) * 255, size, interpolation=cv2.INTER_AREA) > 127
    else:
        small = mask
    if not small.any():
        return dots
    skeleton = thin(small)
    # Geometry can come from the reduced image, but width can't: a thin pen
    # measured there rounds up to whole reduced pixels and the writing comes
    # out bold. Measure it on the full-size mask instead.
    half_width = cv2.distanceTransform(measured.view(np.uint8), cv2.DIST_L2, 5)
    if reduce > 1:
        # A point from the reduced skeleton lands anywhere within `reduce`
        # pixels of the true middle -- possibly on the ink's edge, where the
        # distance to the outside is 1 and a 6px stroke measures 2px. Take the
        # largest half-width in that neighbourhood instead.
        span = 2 * reduce + 1
        half_width = cv2.dilate(half_width, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (span, span)))
    height, width = mask.shape
    stride = small.shape[1]

    # Thinning peels a blob from every side at once, and a small, round one
    # can come off entirely: a full stop just too wide to be caught by the
    # dot rule above left no skeleton at all, and 45 of them vanished from a
    # page of typed notes. Nothing in the mask is allowed to disappear, so
    # whatever lost its skeleton comes back as the dot it always was.
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(small.view(np.uint8), connectivity=8)
    if count > 1:
        alive = np.zeros(count, bool)
        alive[labels[skeleton]] = True
        for index in np.flatnonzero(~alive[1:]) + 1:
            centre = centroids[index] * reduce + (reduce - 1) / 2
            row = min(height - 1, max(0, int(centre[1])))
            column = min(width - 1, max(0, int(centre[0])))
            dots.append((centre.reshape(1, 2), float(2 * half_width[row, column])))

    out = []
    for path in paths(skeleton):
        y, x = path // stride, path % stride
        # Back to the original pixels, from the middle of the reduced pixel.
        points = np.stack([x, y], axis=1).astype(np.float64) * reduce + (reduce - 1) / 2
        rows = np.clip(points[:, 1].astype(np.int64), 0, height - 1)
        columns = np.clip(points[:, 0].astype(np.int64), 0, width - 1)
        out.append((points, float(np.median(2 * half_width[rows, columns]))))
    return out + dots
