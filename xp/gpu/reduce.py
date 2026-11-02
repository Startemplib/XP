"""One summation order, written down, so both sides can follow it.

The fill takes the next level's width from the mean depth of what is still
bare. That number decides which pixels a stroke of that width may stand on,
so if the host and the card compute it differently the page comes out with
different strokes -- which is the rule this project keeps paying to learn:
**the card may make it faster and may not make it different.**

Floating-point addition is not associative, so "the mean" is not a value, it
is a value *and an order*. numpy sums pairwise, cupy sums in a tree, and the
two are not the same tree. On the field's own numbers they happen to agree
-- forty sizes tried, not one differed -- but that is the data's doing: the
field is the square root of a small integer, so there are only a couple of
thousand distinct values and the roundings cancel. On ordinary numbers the
same two calls differ by two to four ulps. A specification that rests on the
input happening to be well behaved is not a specification.

So the order is fixed here, and both sides implement it:

    BLOCK = 4096
    a block's sum is its elements added in index order, in float64
    the total is the block sums added in index order, in float64
    the mean is that total divided by the count, which is exact
    the width is `round(2 * mean, WIDTH_DECIMALS)`

The rounding at the end is belt and braces: it puts the answer on a grid a
millionth of a pixel wide, so even a reduction that drifted by an ulp could
not move it. A millionth of a pixel is nothing to the geometry -- the
rasterizer rounds the width to a whole pixel -- and it makes the number
reproducible by construction rather than by inspection.
"""

import numpy as np

from xp.gpu import backend as gpu
BLOCK = 4096
WIDTH_DECIMALS = 6

try:
    from numba import njit, prange
    JIT = True
except ImportError:                              # pragma: no cover - fallback
    JIT = False


def _blocks(n):
    return (int(n) + BLOCK - 1) // BLOCK


def _masked_sum_python(values, mask, totals, counts):
    """The specification, in numpy, for an install without numba."""
    n = values.shape[0]
    for j in range(len(totals)):
        lo = j * BLOCK
        hi = min(n, lo + BLOCK)
        here = mask[lo:hi]
        # `np.add.reduce` on a contiguous slice of at most BLOCK elements is
        # the same left-to-right walk the other two do at this size.
        totals[j] = float(values[lo:hi][here].astype(np.float64).sum())
        counts[j] = int(here.sum())


if JIT:
    @njit(cache=True, parallel=True)
    def _masked_sum_jit(values, mask, totals, counts):   # pragma: no cover
        """One thread per block, and inside a block, index order.

        Parallel over blocks and sequential inside them: the blocks are
        summed in index order afterwards, so no thread's schedule can reach
        the answer.
        """
        n = values.shape[0]
        for j in prange(totals.shape[0]):
            lo = j * BLOCK
            hi = lo + BLOCK
            if hi > n:
                hi = n
            total = 0.0
            held = 0
            for i in range(lo, hi):
                if mask[i]:
                    total += values[i]
                    held += 1
            totals[j] = total
            counts[j] = held


def masked_mean(values, mask, card=None):
    """The mean of `values` where `mask`, by the order written above.

    `values` and `mask` may be on either side and must be on the same one.
    What comes back is a Python float and an int, which is all the caller
    wants: nothing of this has to be downloaded but the block sums.
    """
    flat = values.reshape(-1)
    wanted = mask.reshape(-1)
    n = int(flat.shape[0])
    if not n:
        return 0.0, 0
    many = _blocks(n)
    if card is not None:
        from xp.gpu import kernels
        totals, counts = kernels.masked_sum(flat, wanted, many, BLOCK)
        totals = gpu.to_host(totals)
        counts = gpu.to_host(counts)
    else:
        totals = np.zeros(many, np.float64)
        counts = np.zeros(many, np.int64)
        held = np.ascontiguousarray(flat)
        on = np.ascontiguousarray(wanted).reshape(-1)
        (_masked_sum_jit if JIT else _masked_sum_python)(held, on, totals, counts)
    # The block sums added in index order -- `math.fsum` would be exact but
    # it is a *different* order, and what matters here is that both sides
    # walk the same one.
    total = 0.0
    for value in totals.tolist():
        total += value
    held = int(counts.sum())
    return (total / held if held else 0.0), held


def next_width(field, bare, card=None):
    """Twice the mean depth of what is still bare, on the agreed grid.

    The fill's ladder: the gap's own mean depth is the widest stroke that can
    stand across it, so the formula reads the region instead of being told a
    ratio -- and it cannot stall, because a width taken from the gap always
    admits seeds that reach the gap.
    """
    mean, held = masked_mean(field, bare, card)
    if not held:
        return 0.0
    return round(2.0 * mean, WIDTH_DECIMALS)
