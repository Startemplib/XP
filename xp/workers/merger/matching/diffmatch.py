"""Naive stroke correspondence between two stroke lists on the "same" page,
used only to decide diff-view coloring (which strokes to mute as "present on
both sides" vs highlight as "only on one side"). This is intentionally loose
-- unlike Clean.py's exact pen-width-fingerprint match, it only has to be good
enough to draw your attention to real differences, not to auto-resolve them.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

from xp.core.document.xopp_io import Stroke

CENTER_TOL = 2.0  # xopp units (~pt); strokes are typically 100-400 units across a page
SIZE_REL_TOL = 0.35


def _center_and_size(stroke: Stroke) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x0, y0, x1, y1 = stroke.bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2), (x1 - x0, y1 - y0)


def match_strokes(left: List[Stroke], right: List[Stroke]) -> Tuple[set, set]:
    """Greedy nearest-centroid matching. Returns (matched_left_idx, matched_right_idx).

    Only candidates within CENTER_TOL of each other can ever match, so
    right-side strokes are bucketed into a CENTER_TOL-sized spatial grid and
    each left stroke only checks its own cell and its 8 neighbors -- O(N+M)
    on typical pages instead of the O(N*M) brute-force comparison this had
    originally, which froze the GUI on real pages with thousands of strokes
    per side (that's an N*M in the tens of millions, in pure Python).
    """
    if not left or not right:
        return set(), set()

    left_info = [_center_and_size(s) for s in left]
    right_info = [_center_and_size(s) for s in right]

    cell = CENTER_TOL
    grid: dict = defaultdict(list)
    for j, (rc, _rsz) in enumerate(right_info):
        grid[(int(rc[0] // cell), int(rc[1] // cell))].append(j)

    candidates = []
    for i, (lc, lsz) in enumerate(left_info):
        cx, cy = int(lc[0] // cell), int(lc[1] // cell)
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for j in grid.get((gx, gy), ()):
                    rc, rsz = right_info[j]
                    dist = ((lc[0] - rc[0]) ** 2 + (lc[1] - rc[1]) ** 2) ** 0.5
                    if dist > CENTER_TOL:
                        continue
                    size_ok = all(
                        abs(a - b) <= SIZE_REL_TOL * max(a, b, 1.0) for a, b in zip(lsz, rsz)
                    )
                    if size_ok:
                        candidates.append((dist, i, j))

    candidates.sort(key=lambda t: t[0])
    matched_left, matched_right = set(), set()
    for _, i, j in candidates:
        if i in matched_left or j in matched_right:
            continue
        matched_left.add(i)
        matched_right.add(j)

    return matched_left, matched_right
