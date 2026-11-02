"""The user's first step: use a stroke that is already there, rather than make one.

    "Do not lightly create an independent Stroke for a small amount of new
    coverage. Prefer to modify the polyline of a nearby existing Stroke."

    A -> B -> C   becomes   A -> B -> P -> B -> C

The join connects **ends**. This connects an end to anywhere along another
stroke, which is a different question with a very different answer, and the
measurement says why:

    after the join has converged, on the user's page
      end-to-end bridges still legal          5
      end-to-any-point bridges still legal    98,966, over 13,405 strokes

An end sits on the rim of the legal set -- that is where a seed is put -- so
a bridge between two ends has to cross the gap *outside* the ink and rule 4
refuses it. A point in the middle of a stroke is deep in the ink, and a
bridge from a rim to a depth stays inside. That is the whole of it.

**What it costs: one point, whatever the length.** The source's points are
*moved* into the host, not copied -- the host leaves its path at M, covers
the source, and comes back -- so the only new point is the one that comes
back. One point against one `<stroke>` element, 18 bytes against 76, and
there is nothing for a length threshold to decide. I had one, derived from
`(k + 1) * point < stroke`, which counted the moved points as if they were
new; the user read the run's own numbers back to me -- 1,929 strokes absorbed for
1,929 points, one apiece -- and that was that.

**And both bridges are checked.** The host leaves at M and must come back to
M from the source's *other* end, so a detour is two bridges and rule 4 has
to be asked about each. Only the outbound one used to be asked. It did no
visible harm because the threshold had left almost nothing but single points
to absorb, where the two ends are the same place and the return is the
outbound bridge reversed -- but for anything longer it was ink drawn without
being asked for.
"""

import numpy as np

from xp.gpu import backend as gpu
from xp.gpu import kernels
from xp.workers.converter.fields import strokejoin
from xp.core.document.strokes import StrokeBuffer


def absorb(buffer: StrokeBuffer, field, gap_px, colour_tolerance,
           slack=strokejoin.SLACK, max_points=1246) -> StrokeBuffer:
    """Fold the strokes that can reach one into the strokes they reach.

    A stroke is folded in out-and-back -- the host leaves its own path at one
    of its own points, covers the other stroke from whichever end the bridge
    reaches, and comes back to where it left. The ink is the host's plus the
    folded stroke's plus the two bridges, and both bridges are held to rule 4
    exactly as a rib is, so nothing lands outside that did not before.

    Nothing is folded into something that is itself being folded: hosts and
    sources are kept apart, so one pass cannot build a chain of detours whose
    order would decide the answer. A chain is built by calling this again --
    the host of one pass is the source of the next, and on the user's page it
    converges in six. See ABSORB_ROUNDS in orthofill.
    """
    count = len(buffer)
    if count < 2 or gap_px <= 0.0:
        return buffer
    host = buffer.to(np)
    counts, starts, widths = host.counts, host.starts, host.widths

    # The search runs where the page is, as the join's does: every step of it
    # is exact arithmetic, so the card is held to the same answer. Only the
    # pairs that pass all of width, colour and rule 4 come back.
    side = gpu.xp if (gpu.GPU_AVAILABLE and kernels.ready()) else np
    there = buffer.to(side)
    points = side.stack([there.xs, there.ys], axis=1)
    owner = side.repeat(side.arange(count), there.counts)
    # Every stroke offers its two ends as a source and all of its points as a
    # host: the first `count` of `heads` are first points and the next `count`
    # are last points, which is what `far` reads to find the other end.
    ends = side.concatenate([there.starts,
                             there.starts + there.counts - 1])
    from_stroke = side.concatenate([side.arange(count), side.arange(count)])
    heads = points[ends]
    # Only the ends are asked about, and only the points are offered. The
    # question used to be "every pair of points within the gap of each other",
    # which built the grid over the ends AND the points and walked nine cells
    # around all 439,035 of them, to keep the pairs whose first index was one
    # of the 61,600 ends. Same pairs, same index values, same answer -- see
    # `_near`.
    lab = host.lab()
    who = side.asarray(host.tint)
    table = side.asarray(np.nan_to_num(lab, nan=0.0))
    known = side.asarray(~np.isnan(lab[:, 0]))

    def sieve(mine, theirs):
        """Every filter, on one chunk of candidates, before the next is made.

        The same chain in the same order; what changed is that it runs inside
        the search, so a chunk is reduced to its survivors before another is
        built. The search used to hand back every candidate pair of the page
        at once -- 15.4 million on the sample page, and enough on a page with a
        thicker pen to ask for 5.02 GiB in one allocation and kill the
        conversion.
        """
        source = from_stroke[mine]
        into = owner[theirs]
        keep = source != into
        if not bool(keep.any()):
            return None, None
        mine, theirs = mine[keep], theirs[keep]
        source, into = source[keep], into[keep]
        one, two = who[source], who[into]
        near = one == two
        pair = known[one] & known[two]
        apart = side.sqrt(((table[one] - table[two]) ** 2).sum(axis=1))
        keep = near | (pair & (apart <= colour_tolerance))
        keep &= there.widths[source] == there.widths[into]
        if not bool(keep.any()):
            return None, None
        mine, theirs, source = mine[keep], theirs[keep], source[keep]
        # Both ways. The host leaves its path at M, covers the source from
        # the end the bridge reaches, and comes back to M from the *other*
        # end -- so there are two bridges, not one, and rule 4 has to be
        # asked about each. Only the outbound one used to be asked, which was
        # ink drawn without being asked for everything longer than a point.
        #
        # The way back is held to rule 4 and to nothing else. I bounded it by
        # the gap first, which cost 3,232 strokes: the gap is how a candidate
        # is *found*, and a long way back that stays inside the ink band the
        # whole way is legal geometry. Measured, the bound buys nothing
        # either -- 10.89s bounded against 10.86s not, twice each.
        far = side.where(mine < count, mine + count, mine - count)
        legal = strokejoin._legal_bridges(
            field, heads[mine, 0], heads[mine, 1],
            points[theirs, 0], points[theirs, 1],
            there.widths[source], slack, side)
        legal &= strokejoin._legal_bridges(
            field, heads[far, 0], heads[far, 1],
            points[theirs, 0], points[theirs, 1],
            there.widths[source], slack, side)
        if not bool(legal.any()):
            return None, None
        return mine[legal], theirs[legal]

    mine, theirs = strokejoin._near(heads, points, gap_px, sieve)
    if not len(mine):
        return buffer
    source = from_stroke[mine]
    into = owner[theirs]
    span = side.hypot(points[theirs, 0] - heads[mine, 0],
                      points[theirs, 1] - heads[mine, 1])

    # Shortest bridge first, and each source folded once. A source may not be
    # a host and a host may not be a source, so the result does not depend on
    # which was decided first. The walk is sequential, so this is where the
    # survivors cross.
    order = side.lexsort(side.stack((theirs, mine, span)))
    mine, theirs = mine[order], theirs[order]
    source, into = source[order], into[order]
    if side is not np:
        mine, theirs = gpu.to_host(mine), gpu.to_host(theirs)
        source, into = gpu.to_host(source), gpu.to_host(into)
    order = np.arange(len(mine))
    taken = np.zeros(count, bool)        # already folded away
    hosting = np.zeros(count, bool)      # has taken somebody in
    room = counts.astype(np.int64).copy()
    fold = {}                            # host -> [(point index, source, flipped)]
    for at in order.tolist():
        s, h = int(source[at]), int(into[at])
        if taken[s] or hosting[s] or taken[h]:
            continue
        added = int(counts[s]) + 1   # its own points, moved, plus the way back
        if room[h] + added > max_points:
            continue
        taken[s] = True
        hosting[h] = True
        room[h] += added
        # Which end of the source the bridge reaches, so it is traversed from
        # there and comes back.
        flipped = int(mine[at]) >= count
        fold.setdefault(h, []).append((int(theirs[at]), s, flipped))
    if not fold:
        return buffer

    # Rebuild: every stroke that is neither folded away nor a host keeps its
    # points untouched, and a host's points are its own with each folded
    # stroke's inserted after the point the bridge reached.
    pieces, owners = [], []
    out_index = {}
    for stroke in range(count):
        if taken[stroke]:
            continue
        here = len(out_index)
        out_index[stroke] = here
        begin, length = int(starts[stroke]), int(counts[stroke])
        held = fold.get(stroke)
        if not held:
            pieces.append((begin, length, False))
            owners.append(here)
            continue
        held.sort()
        cut = begin
        for where, s, flipped in held:
            pieces.append((cut, where + 1 - cut, False))
            owners.append(here)
            pieces.append((int(starts[s]), int(counts[s]), flipped))
            owners.append(here)
            pieces.append((where, 1, False))      # back to where it left
            owners.append(here)
            cut = where + 1
        pieces.append((cut, begin + length - cut, False))
        owners.append(here)

    begins = np.array([one for one, _k, _f in pieces], np.int64)
    lengths = np.array([k for _o, k, _f in pieces], np.int64)
    flips = np.array([f for _o, _k, f in pieces], bool)
    owners = np.array(owners, np.int64)
    alive = lengths > 0
    begins, lengths, flips, owners = (begins[alive], lengths[alive],
                                      flips[alive], owners[alive])
    total = int(lengths.sum())
    offset = np.zeros(len(lengths), np.int64)
    np.cumsum(lengths[:-1], out=offset[1:])
    within = np.arange(total, dtype=np.int64) - np.repeat(offset, lengths)
    low = np.repeat(begins, lengths)
    high = np.repeat(begins + lengths - 1, lengths)
    source_at = np.where(np.repeat(flips, lengths), high - within, low + within)
    out_counts = np.bincount(owners, weights=lengths,
                             minlength=len(out_index)).astype(np.int64)
    out_starts = np.zeros(len(out_counts), np.int64)
    np.cumsum(out_counts[:-1], out=out_starts[1:])
    kept = np.array(sorted(out_index, key=out_index.get), np.int64)
    # `who` is the colour index on whichever side the search ran; the buffer
    # being built says it is on the host, so it takes the host's. Declaring
    # one side and holding an array from the other happened to work -- every
    # reader downstream went through `tolist` -- and was still a lie.
    return StrokeBuffer(np, host.xs[source_at], host.ys[source_at],
                        out_starts, out_counts, widths[kept],
                        host.tint[kept], host.palette)
