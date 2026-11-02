"""The user's last step: join the strokes that are near each other and one colour.

    "It can merge at the last step, just connect the nearby one, I can set a
    para for it -- you know if colour as 1 point then connect."

Two numbers and no more:

    merge_gap     how close two ends have to be, in pen widths
    merge_colour  how close two colours have to be, as Delta-E

A joined stroke passes through exactly the points the two strokes had. The
only ink it adds is the short segment that bridges two ends, and that segment
is held to the same rule as every rib -- `d >= w/2 + slack` at every sample
along it -- so rule 4 holds after the join as it did before.

It does move ink by a pixel, and the reason is worth knowing: a band of even
width cannot be centred on a row of pixels, so the rasterizer gives the extra
row to one side, and which side depends on the direction the segment is drawn
in. Walking up and down a level's ribs reverses every other one, and that
moves its band one row. Twenty-nine pixels of the user's page came back bare that
way. It is not a flaw in the join and no join rule avoids it -- so the join
runs *before* the residual dots, and rule 5, whose whole job is "cover
whatever the lines left", covers them.

**The search is over the page, not over the list.** The first version only
ever asked whether the *next* stroke in the emission order would join, which
is cheap and wrong: measured on the user's page, 96.9% of the single-point strokes
that survived had a same-width neighbour a single pixel away, and the join
had never looked at it. Ends go into a grid of cells one gap across and every
end asks its nine neighbouring cells, so two strokes that touch are found
whether they were emitted a moment apart or fifty thousand strokes apart. The
pairs are taken shortest first, each end used once and no chain allowed to
close on itself, which makes the answer the same on every run.
"""

from typing import List

import cv2
import numpy as np

from xp.gpu import backend as gpu
from xp.gpu import kernels
from xp.core.document.strokes import StrokeBuffer

# How finely the bridge between two ends is checked: the same half-pixel the
# ribs were grown with (`orthofill.MARCH_STEP`), because it is the same
# question asked of the same field. Written out rather than imported so this
# module stays below `orthofill`, which calls it.
SAMPLE_STEP = 0.5
COARSE = 16                # the stride of the host's rejecting pass. Not a
                           # dial on the answer -- the samples it looks at are
                           # a subset of the samples the full test looks at,
                           # so it can only agree with it. Sixteen because a
                           # way back that is illegal is illegal early.
BUDGET = 1 << 22           # samples held at once on the host, about 4 million
                           # -- 32 MB of float64 per temporary. A ceiling on
                           # the memory of one chunk, not on the answer.
SLACK = 0.5                  # `orthofill.RASTER_SLACK`, for the same reason


def _measured(colour):
    """Is this a colour that was measured, or a name out of a palette?

    `Segment.color` is either a BGR triple read off the page (color5d, and
    what ortho-fill carries) or a palette name the older encoders hand over
    as a string. A distance is only meaningful for the first kind: two names
    are the same pen or they are not, and there is no Delta-E between the
    word "black" and the word "blue".
    """
    return (isinstance(colour, tuple) and len(colour) == 3
            and all(isinstance(part, (int, float, np.integer, np.floating))
                    for part in colour))


def _labs(colours):
    """Each distinct measured colour once, in Lab, so Delta-E is a
    subtraction.

    Lab because that is what the clustering measured in: the user's tolerance is in
    the same unit as the Delta-E 29.4 the cluster seams were reported at, and
    a tolerance in BGR would mean something different in the dark than in the
    light.
    """
    seen = {}
    for colour in colours:
        if not _measured(colour) or colour in seen:
            continue
        patch = np.array([[list(colour)]], np.uint8)
        seen[colour] = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)
    return seen


def _colours_of(segments):
    """Each stroke's colour as numbers: an identity, and a place in Lab.

    Two arrays rather than one key, because the tolerance is **not** an
    equivalence -- A near B and B near C does not make A near C -- so it
    cannot be turned into a group without deciding something it does not say.
    It is asked of each pair instead, which is what it means. The identity is
    there for the colours that have no distance: a palette name is the same
    pen or it is not.

    Every question about a colour is asked once per *colour*, not once per
    stroke. A page is two hundred thousand strokes and six thousand colours,
    and `_measured` alone was being asked four hundred thousand times.
    """
    seen, who = {}, np.empty(len(segments), np.int64)
    for index, segment in enumerate(segments):
        colour = segment.color
        at = seen.get(colour)
        if at is None:
            at = seen[colour] = len(seen)
        who[index] = at
    table = np.full((len(seen), 3), np.nan)
    for colour, at in seen.items():
        if _measured(colour):
            patch = np.array([[list(colour)]], np.uint8)
            table[at] = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[0, 0]
    return who, table, ~np.isnan(table[:, 0])


def _bridge_is_legal(field, ax, ay, bx, by, width, slack=SLACK):
    """Rule 4 for the bridge, in bulk: one array of samples for every pair.

    A round-capped stroke of width w along a segment covers everything within
    w/2 of it, so it stays inside the region exactly when `d >= w/2` all the
    way along -- and the slack is the renderer's own rounding, the same
    half-pixel the ribs were placed with. Nothing here is a new rule; it is
    the rule the ribs already obey, asked about a segment that was not drawn
    yet.
    """
    if not len(ax):
        return np.zeros(0, bool)
    # `sqrt` of the sum of squares and not `hypot`, which is more accurate:
    # the kernel computes it the plain way, and where the two differ in the
    # last bit `ceil` can differ by one sample -- which came out as one
    # stroke in forty-six thousand deciding differently on the card.
    reach = (bx - ax) ** 2 + (by - ay) ** 2
    span = np.sqrt(reach)
    out = np.zeros(len(ax), bool)
    cuts = np.maximum(1, np.ceil(span / SAMPLE_STEP).astype(np.int64))
    # By length, in bands, and inside a band each bridge is still sampled at
    # its *own* spacing. Two reasons, and the second is the one that matters:
    # a bridge of one pixel needs three samples where one of seventeen needs
    # thirty-six, and nearly every gap is a single pixel -- but above all the
    # kernel in `kernels.bridges` samples each pair at its own length, and if
    # this sampled them any other way the card would decide a borderline
    # bridge differently from the host, which came out as 527 strokes.
    edges = [2.0, 8.0, 32.0, float(span.max()) + 1.0]
    # Below zero, not at it: two ends in the same place are a bridge of no
    # length, which is a perfectly good bridge -- and `span > 0.0` dropped
    # every one of them out of all three bands and called them illegal. Ten
    # of four and a half million candidates on the user's page, which came out as
    # one stroke the card kept and the host did not.
    low = -1.0
    for high in edges:
        here = np.nonzero((span > low) & (span <= high))[0]
        low = high
        if not len(here):
            continue
        steps = int(cuts[here].max()) + 1
        # And in row chunks inside the band, because a band is as wide as its
        # longest bridge and the detour can now offer a long one: a source may
        # be any length, and its far end was found 4,472 px from the junction,
        # which is 8,945 samples times however many pairs are in the band --
        # 30 GiB, where the card just loops. The chunking is bookkeeping and
        # not arithmetic: `t` already divides by each row's own `cuts`, so a
        # bridge is sampled at its own spacing whatever it is grouped with,
        # and the answer does not know the chunks are there.
        rows = max(1, int(BUDGET // steps))
        for at in range(0, len(here), rows):
            part = here[at:at + rows]
            # Coarsely first, where coarse is worth it. The detour's way back
            # may be hundreds of pixels long and nearly all of those are
            # illegal within a few steps of leaving, so asking every COARSE'th
            # sample first and only then asking all of them is 148s of the
            # host's reference path against 40.
            #
            # It is exact, not an approximation: `t` is `i / cuts` and the
            # coarse indices are a SUBSET of the fine ones, so a coarse sample
            # that rule 4 refuses is a fine sample that rule 4 refuses. Only
            # rejection is taken from it -- a pair that survives the coarse
            # pass is still asked the whole question.
            if steps > 2 * COARSE:
                live = _sampled(field, ax, ay, bx, by, width, slack, part,
                                cuts, steps, COARSE)
                part = part[live]
                if not len(part):
                    continue
            out[part] = _sampled(field, ax, ay, bx, by, width, slack, part,
                                 cuts, steps, 1)
    return out


def _sampled(field, ax, ay, bx, by, width, slack, part, cuts, steps, stride):
    """Rule 4 asked of every `stride`th sample of each bridge in `part`.

    `stride` of 1 is the whole question. Anything larger is a subset of the
    same samples -- the last index is kept so the far end is always asked --
    so a False here is a False there, and that is the only thing a coarse
    answer is used for.
    """
    mine = cuts[part][:, None]
    want = np.arange(0, steps, stride)
    if stride > 1 and want[-1] != steps - 1:
        want = np.append(want, steps - 1)
    t = np.minimum(want[None, :], mine) / mine
    xs = np.rint(ax[part, None] + (bx[part] - ax[part])[:, None] * t).astype(np.intp)
    ys = np.rint(ay[part, None] + (by[part] - ay[part])[:, None] * t).astype(np.intp)
    np.clip(xs, 0, field.shape[1] - 1, out=xs)
    np.clip(ys, 0, field.shape[0] - 1, out=ys)
    room = field[ys, xs]
    return (room >= (width[part] / 2.0 + slack)[:, None]).all(axis=1)


def _legal_bridges(field, ax, ay, bx, by, width, slack, side=np):
    """Rule 4 for every candidate bridge, on the side the pairs are on."""
    if side is not np and kernels.ready():
        return kernels.bridges(field, ax, ay, bx, by, width, slack,
                               SAMPLE_STEP).view(bool)
    try:
        if kernels.ready():
            return gpu.to_host(kernels.bridges(field, ax, ay, bx, by, width,
                                               slack, SAMPLE_STEP)).view(bool)
    except Exception:
        pass
    return _bridge_is_legal(field, ax, ay, bx, by, width, slack)


def _grid_over(points, cell, xp):
    """The uniform grid the two searches share.

    A grid and not a tree: there is no scipy here, the cell is the search
    radius so the nine cells around a point hold every candidate, and the
    whole thing is `searchsorted` plus arithmetic -- no Python loop over a
    hundred and fifty thousand ends.

    The sort is stable on both sides. Two ends in one cell have equal keys,
    and which of them comes first decides the order of the pairs that follow,
    which decides which of two equally short joins is taken first; stable
    means equal keys keep their index order, so the card and the host agree
    about it rather than each being internally consistent.
    """
    at = xp.floor(points / cell).astype(xp.int64)
    low_y = int(at[:, 1].min())
    span = int(at[:, 1].max()) - low_y + 3
    base = at[:, 0] * span + (at[:, 1] - low_y + 1)
    order = xp.argsort(base, kind='stable')
    return order, base[order], span, low_y


def _cells_of(queries, cell, span, low_y, xp):
    asked = xp.floor(queries / cell).astype(xp.int64)
    return asked[:, 0] * span + (asked[:, 1] - low_y + 1)


OFFSETS = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
PER_PAIR = 96          # bytes a candidate pair costs while it is being made:
                       # two int64 indices, the run-length arithmetic that
                       # builds them, the squared distance and its mask. A
                       # guess on the generous side, because the cost of
                       # guessing low is the crash this exists to prevent.


def _search(queries, points, cell, sieve=None, budget=None):
    """Every point within `cell` of each query, in chunks that fit.

    THE CRASH THIS EXISTS TO PREVENT. The search used to build every
    candidate pair of the page at once and hand the lot to the caller, which
    then threw almost all of them away. On the sample page that is 15.4 million
    pairs; on a page with a thicker pen it is far more, and the one
    allocation for the index arrays asked for 5.02 GiB on a 12 GiB card. The
    conversion died with cupy's OutOfMemoryError -- in a dialog, with a raw
    byte count in it, which is not a thing a user should ever be shown.

    So the queries are walked in chunks whose pair count fits the memory
    there actually is (`gpu.room`), and `sieve(mine, theirs)` -- the caller's
    own filter chain -- runs on each chunk. What accumulates is the
    survivors, which are a ten-thousandth of what is built.

    IT IS THE SAME ANSWER. The pairs are the same pairs and carry the same
    index values; only the order they are emitted in changes, and nothing
    downstream reads that order: both callers sort by (span, mine, theirs),
    and two distinct pairs cannot tie on all three. The chunking is
    bookkeeping about memory, and the answer does not know it is there.
    """
    xp = _side(points)
    if not len(queries) or not len(points):
        return xp.zeros(0, xp.int64), xp.zeros(0, xp.int64)
    order, sorted_base, span, low_y = _grid_over(points, cell, xp)
    base = _cells_of(queries, cell, span, low_y, xp)
    # How many candidates each query has, before any of them are built: nine
    # `searchsorted` pairs, which is cheap, and it is what lets the chunks be
    # chosen to a size rather than guessed at.
    lows, counts = [], xp.zeros(len(base), xp.int64)
    for dx, dy in OFFSETS:
        want = base + dx * span + dy
        lo = xp.searchsorted(sorted_base, want, 'left')
        hi = xp.searchsorted(sorted_base, want, 'right')
        lows.append((lo, hi - lo))
        counts += hi - lo
    room = budget if budget is not None else gpu.room(xp)
    most = max(1, int(room // PER_PAIR))
    left, right = [], []
    for begin, end in _chunk_edges(counts, most, xp):
        mine, theirs = _one_chunk(queries, points, cell, order, lows,
                                  begin, end, xp)
        if mine is None:
            continue
        if sieve is not None:
            mine, theirs = sieve(mine, theirs)
            if mine is None or not len(mine):
                continue
        left.append(mine)
        right.append(theirs)
    if not left:
        return xp.zeros(0, xp.int64), xp.zeros(0, xp.int64)
    if len(left) == 1:
        return left[0], right[0]
    return xp.concatenate(left), xp.concatenate(right)


def _chunk_edges(counts, most, xp):
    """Query ranges whose candidate pairs each fit in `most`.

    One query's own nine cells can exceed the budget by themselves -- a page
    with every stroke on top of every other would do it -- and then it gets a
    chunk of its own and the budget is simply missed. Refusing to make
    progress would be worse than a large allocation: the point of the budget
    is that the *usual* page stops asking for gigabytes, not that an
    impossible one is made possible.
    """
    total = int(counts.sum())
    if total <= most:
        return [(0, int(counts.shape[0]))]
    running = xp.cumsum(counts)
    running = np.asarray(gpu.to_host(running) if xp is not np else running)
    edges, begin, taken = [], 0, 0
    for at in range(len(running)):
        if running[at] - taken > most and at > begin:
            edges.append((begin, at))
            begin, taken = at, int(running[at - 1])
    edges.append((begin, len(running)))
    return edges


def _one_chunk(queries, points, cell, order, lows, begin, end, xp):
    """The pairs of one range of queries, over all nine cells."""
    left, right = [], []
    for lo, many in lows:
        many = many[begin:end]
        total = int(many.sum())
        if not total:
            continue
        # Expand the ranges without a loop: one row per (query, candidate).
        mine = xp.repeat(xp.arange(begin, end, dtype=xp.int64), many)
        within = xp.arange(total, dtype=xp.int64) - xp.repeat(
            xp.cumsum(many) - many, many)
        theirs = order[xp.repeat(lo[begin:end], many) + within]
        close = ((points[theirs, 0] - queries[mine, 0]) ** 2
                 + (points[theirs, 1] - queries[mine, 1]) ** 2) <= cell * cell
        if not bool(close.any()):
            continue
        left.append(mine[close])
        right.append(theirs[close])
    if not left:
        return None, None
    if len(left) == 1:
        return left[0], right[0]
    return xp.concatenate(left), xp.concatenate(right)


def _neighbours(points, cell, sieve=None, budget=None):
    """Every pair of points within `cell` of each other, as two arrays.

    The distance is checked **here**, not by the caller. Nine cells of one
    gap each is a 3x3 box, and a box of side 3r has three times the area of
    the circle of radius r inside it -- so two candidates in three were
    always going to be thrown away, after being carried through the width
    test, the colour test and the sort.
    """
    return _search(points, points, cell, sieve, budget)


def _near(queries, points, cell, sieve=None, budget=None):
    """Every point within `cell` of each query, as (query index, point index).

    The same grid and the same arithmetic as `_neighbours`, built over
    `points` and walked from `queries`. The cells are numbered from the
    points' own extent so that a query lands in the cell it would have landed
    in had it been one of them, which it is: the detour's queries are the
    ends, and an end is a point.
    """
    return _search(queries, points, cell, sieve, budget)


def _side(array):
    """Which array module this array belongs to.

    One implementation, two sides: the search is a floor, a sort, two
    `searchsorted`, a repeat and a squared distance, and not one of those
    rounds. So the card is held to the same *answer* here, not to an
    equivalent one.
    """
    if gpu.GPU_AVAILABLE and isinstance(array, gpu.xp.ndarray):
        return gpu.xp
    return np


def _in_order(points, cell, sieve=None, budget=None):
    """Only the four ways each stroke can meet the next one in the list.

    A level's ribs are emitted in the order their seeds were walked along the
    boundary, so the rib beside a rib is very often the next rib -- and
    finding that costs one array instead of a grid. It is not a substitute
    for the search over the page: it is what makes that search cheap, because
    it hands it chains instead of ribs.
    """
    xp = _side(points)
    n = len(points) // 2
    if n < 2:
        return xp.zeros(0, xp.int64), xp.zeros(0, xp.int64)
    i = xp.arange(n - 1)
    mine = xp.concatenate([2 * i, 2 * i, 2 * i + 1, 2 * i + 1])
    theirs = xp.concatenate([2 * i + 2, 2 * i + 3, 2 * i + 2, 2 * i + 3])
    close = ((points[theirs, 0] - points[mine, 0]) ** 2
             + (points[theirs, 1] - points[mine, 1]) ** 2) <= cell * cell
    mine, theirs = mine[close], theirs[close]
    # Four pairs per stroke and no more, so there is nothing here to chunk;
    # the sieve is taken all the same, because both providers answer the same
    # question and a caller should not have to know which one it got.
    if sieve is not None and len(mine):
        mine, theirs = sieve(mine, theirs)
        if mine is None:
            xp_ = _side(points)
            return xp_.zeros(0, xp_.int64), xp_.zeros(0, xp_.int64)
    return mine, theirs


def join(segments, field, Segment, gap_px, colour_tolerance, max_points=1246,
         slack=SLACK, default_width=1.0) -> List:
    """The object form, for the three older encoders that still speak it."""
    if gap_px <= 0.0 or len(segments) < 2:
        return list(segments)
    held = StrokeBuffer.of_segments(segments, np, default_width)
    joined, origin = join_buffer(held, field, gap_px, colour_tolerance,
                                 max_points, slack, with_origin=True)
    if len(joined) == len(held):
        return list(segments)
    # A chain says what its first stroke said. The older encoders leave a
    # width of None to mean "the profile's own", and joining is bookkeeping:
    # it may reorder and reverse, it may not put a number where the stroke
    # said it had none.
    told = [segments[int(i)].width for i in origin]
    return joined.to_segments(Segment, widths_told=told)


def join_buffer(buffer, field, gap_px, colour_tolerance, max_points=1246,
                slack=SLACK, with_origin=False):
    """Connect strokes that touch, wherever on the page they are.

    Twice: once along the list, then once over the page. The first pass is an
    array and no grid, and it collapses a level's ribs into the chains they
    obviously form; the second then searches the page with a tenth of the
    ends to search among. Measured on the user's page, that is not a trade -- it is
    faster *and* it ends with fewer strokes, because the cheap pass makes
    exactly the joins the expensive one would have made anyway and leaves it
    free to find the ones it could not:

        over the page only         4.83s   32,513 strokes
        along the list, then over  2.00s   31,070 strokes

    `gap_px` of nought (or nothing to join) returns the strokes untouched, so
    the step is a no-op when the user dials it off rather than a different code
    path. `max_points` is `render.path_limit`: a chain longer than one
    `<stroke>` can carry would be cut up again on the way out, which would
    give back the strokes this step just saved.
    """
    if gap_px <= 0.0 or len(buffer) < 2:
        return (buffer, np.arange(len(buffer))) if with_origin else buffer
    once, first = _pass(buffer, field, gap_px, colour_tolerance, max_points,
                        slack, _in_order)
    twice, second = _pass(once, field, gap_px, colour_tolerance, max_points,
                          slack, _neighbours)
    # Which stroke of the *original* each chain started from, through both
    # passes: `with_origin` is for the callers that have to carry something
    # of their own alongside -- a width the stroke stated as None, say.
    return (twice, first[second]) if with_origin else twice


def _pass(buffer, field, gap_px, colour_tolerance, max_points, slack, pairs):
    """One round of joining, over whatever candidates `pairs` offers."""
    count = len(buffer)
    if count < 2:
        return buffer, np.arange(count)
    host = buffer.to(np)
    xs, ys = host.xs, host.ys
    starts, counts, widths = host.starts, host.counts, host.widths
    who = host.tint
    lab = host.lab()
    measured = ~np.isnan(lab[:, 0])

    # The search runs where the page is. Every step of it -- the grid, the
    # two `searchsorted`, the squared distance, the width test -- is integer
    # or exact arithmetic, so the card is held to the same answer and not an
    # equivalent one. Only the pairs that survive come back, and they are a
    # fraction of the ends.
    # Every filter on the side the page is on, and only the survivors cross.
    # Moving the search alone was *worse* than leaving it where it was: the
    # candidates went up, one test ran there, and the whole list came back
    # for the next two. A boundary in the middle of a chain of filters costs
    # more than it saves.
    side = gpu.xp if (gpu.GPU_AVAILABLE and kernels.ready()) else np
    there = buffer.to(side)
    flat = side.ascontiguousarray(there.ends().reshape(-1, 2))
    hold = there.widths
    tints = side.asarray(who)
    table = side.asarray(np.nan_to_num(lab, nan=0.0))
    known = side.asarray(measured)

    def sieve(mine, theirs):
        """Every filter, on one chunk of candidates, before the next is made.

        This is the same chain in the same order it was in when it ran after
        the whole search; what changed is where it runs. The search hands it
        a chunk, it hands back the survivors, and only those are kept -- so
        the peak is a chunk rather than the page. On the user's page that is 15.4
        million candidates against 12,436 survivors.
        """
        a, b = mine // 2, theirs // 2
        keep = (a < b) & (hold[a] == hold[b])
        if not bool(keep.any()):
            return None, None
        # The user's rule -- "if colour as 1 point then connect" -- asked of each
        # pair, which is where a tolerance belongs. Identical is always near
        # enough, whatever kind of colour it is; a distance only applies
        # where there is one to measure.
        one, two = tints[a], tints[b]
        near = one == two
        both = known[one] & known[two]
        apart = side.sqrt(((table[one] - table[two]) ** 2).sum(axis=1))
        keep &= near | (both & (apart <= colour_tolerance))
        if not bool(keep.any()):
            return None, None
        mine, theirs, a = mine[keep], theirs[keep], a[keep]
        ax, ay = flat[mine, 0], flat[mine, 1]
        bx, by = flat[theirs, 0], flat[theirs, 1]
        near = side.hypot(bx - ax, by - ay) <= gap_px
        if not bool(near.any()):
            return None, None
        mine, theirs, a = mine[near], theirs[near], a[near]
        ax, ay, bx, by = ax[near], ay[near], bx[near], by[near]
        legal = _legal_bridges(field, ax, ay, bx, by, hold[a], slack, side)
        if not bool(legal.any()):
            return None, None
        return mine[legal], theirs[legal]

    mine, theirs = pairs(flat, gap_px, sieve)
    if not len(mine):
        return buffer, np.arange(count)
    span = side.hypot(flat[theirs, 0] - flat[mine, 0],
                      flat[theirs, 1] - flat[mine, 1])

    # Shortest first, ties broken by the ends' own numbering, so that two
    # runs over the same page build the same chains. Sorted where the arrays
    # are; the matching below is a sequential walk with a union-find in it,
    # which is not a parallel job, so this is where the survivors cross.
    order = side.lexsort(side.stack((theirs, mine, span)))
    mine, theirs = mine[order], theirs[order]
    if side is not np:
        mine, theirs = gpu.to_host(mine), gpu.to_host(theirs)
    order = np.arange(len(mine))
    first_end = mine.tolist()
    second_end = theirs.tolist()
    link = [-1] * (2 * count)
    root = list(range(count))
    size = counts.tolist()

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    for one, two in zip(first_end, second_end):
        if link[one] >= 0 or link[two] >= 0:
            continue
        head, tail = find(one // 2), find(two // 2)
        if head == tail:              # a chain is a path; it may not close
            continue
        total = size[head] + size[tail]
        if total > max_points:
            continue
        link[one], link[two] = two, one
        root[head] = tail
        size[tail] = total

    # Walk each chain from an end nothing is joined to. A chain is a path, so
    # it has two of those; starting from the lower-numbered one keeps the
    # output the same whichever end was met first.
    #
    # The walk records *pieces* -- which stroke, and which way round -- and
    # the points are gathered afterwards in one indexing pass. A chain used
    # to be built by stacking arrays as it went, which is a copy per link.
    piece_of, flip_of, owner_of, lead = [], [], [], []
    seen = bytearray(count)
    for index in range(count):
        if seen[index] or (link[2 * index] >= 0 and link[2 * index + 1] >= 0):
            continue
        entry = 2 * index if link[2 * index] < 0 else 2 * index + 1
        here, mark = index, len(lead)
        lead.append(index)
        while True:
            seen[here] = 1
            piece_of.append(here)
            flip_of.append(entry != 2 * here)
            owner_of.append(mark)
            nxt = link[2 * here + 1 - (entry - 2 * here)]
            if nxt < 0:
                break
            here, entry = nxt // 2, nxt

    piece = np.asarray(piece_of, np.int64)
    flip = np.asarray(flip_of, bool)
    owner = np.asarray(owner_of, np.int64)
    lead = np.asarray(lead, np.int64)
    held = counts[piece]
    total = int(held.sum())
    begins = np.zeros(len(held), np.int64)
    np.cumsum(held[:-1], out=begins[1:])
    within = np.arange(total, dtype=np.int64) - np.repeat(begins, held)
    low = np.repeat(starts[piece], held)
    high = np.repeat(starts[piece] + held - 1, held)
    source = np.where(np.repeat(flip, held), high - within, low + within)
    out_counts = np.bincount(owner, weights=held, minlength=len(lead)).astype(np.int64)
    out_starts = np.zeros(len(lead), np.int64)
    np.cumsum(out_counts[:-1], out=out_starts[1:])
    return (StrokeBuffer(np, xs[source], ys[source], out_starts, out_counts,
                         widths[lead], who[lead], host.palette), lead)
