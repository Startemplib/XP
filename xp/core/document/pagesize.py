"""The one page everything is normalized onto, and who has to move to get there.

QBASE is that page: 187.8236 x 391.3654 pt, which is the 66 x 138 mm of
`Qbase_66x138.pdf`, the template this whole notebook library is written on.
Nearly every .xopp in `doing/` is this size and every conversion writes it,
so it is the convention rather than one option among several -- a merged
page is Q base whatever the two sides claim to be, and a side that is not
already Q base is brought onto it.

Deciding the target is all that happens here. How a side's ink is actually
placed into it is inkfit.py's job, which measures the ink rather than
trusting the page box: two pages can both be Q base and still carry the
writing at different sizes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

QBASE = (187.8236, 391.3654)   # pt; 66 x 138 mm, the Qbase_66x138.pdf template
SIZE_TOLERANCE = 0.5        # points; scanners and exporters round differently
ASPECT_TOLERANCE = 0.01     # 1%: past this the two pages are different shapes


def is_qbase(width: float, height: float) -> bool:
    return (abs(width - QBASE[0]) <= SIZE_TOLERANCE
            and abs(height - QBASE[1]) <= SIZE_TOLERANCE)


def aspects_agree(left: Tuple[float, float], right: Tuple[float, float]) -> bool:
    if min(left) <= 0 or min(right) <= 0:
        return False
    a, b = left[0] / left[1], right[0] / right[1]
    return abs(a - b) <= ASPECT_TOLERANCE * max(a, b)


def same_size(left: Tuple[float, float], right: Tuple[float, float]) -> bool:
    return (abs(left[0] - right[0]) <= SIZE_TOLERANCE
            and abs(left[1] - right[1]) <= SIZE_TOLERANCE)


@dataclass(frozen=True)
class Target:
    """The geometry the merged page will have, and where it came from."""

    size: Tuple[float, float]
    base: str = 'left'           # the side already in this geometry
    aspects_agree: bool = True   # whether the two pages are the same shape
    reason: str = ''


def choose_target(left: Optional[Tuple[float, float]],
                  right: Optional[Tuple[float, float]]) -> Target:
    """The geometry one pair of pages is merged into: Q base, always.

    Which side is already there decides who has to move:

    1. Either page is Q base -> that side stays put and the other comes to it.
    2. Both are -> nothing moves.
    3. Neither is, and they are different sizes -> both are brought onto Q
       base, the left one by its page box (it is the notebook being merged
       into) and the right one onto the left's ink.
    4. Neither is, but they already agree -> that size is kept. Two pages
       that agree have nothing to reconcile.
    """
    if left is None and right is None:
        return Target(QBASE, 'left', True, 'no pages')
    if right is None:
        return Target(QBASE if is_qbase(*left) else left, 'left', True, 'left only')
    if left is None:
        return Target(QBASE if is_qbase(*right) else right, 'right', True, 'right only')

    agree = aspects_agree(left, right)
    left_is_base, right_is_base = is_qbase(*left), is_qbase(*right)
    if left_is_base and right_is_base:
        return Target(QBASE, 'left', agree, 'both pages are Q base')
    if left_is_base:
        return Target(QBASE, 'left', agree, 'left is Q base')
    if right_is_base:
        return Target(QBASE, 'right', agree, 'right is Q base')
    if same_size(left, right):
        # Two pages that already agree have nothing to reconcile, and moving
        # them both onto Q base would only shrink a document that was never
        # written on it. Normalizing is for pages that differ.
        return Target(left, 'left', agree, 'both pages are the same size already')
    return Target(QBASE, 'neither', agree, 'the pages differ; both are brought onto Q base')
