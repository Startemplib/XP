"""Work out which page of one document corresponds to which page of another.

Pairing page 1 with page 1, page 2 with page 2 and so on only works when
the two documents have the same pages in the same order. As soon as one
side has a page the other doesn't -- a page added in the middle, a page
split in two -- every later page is compared against the wrong partner and
the whole diff is nonsense.

So: describe each page by where its ink actually is, score every left page
against every right page, and align the two sequences with the usual
edit-distance dynamic program. That gets the order-preserving part right by
construction (pages don't get reordered, they get inserted or dropped), and
a page that matches nothing well enough is reported as inserted rather than
forced onto a bad partner.

Alignment is a starting point, not gospel -- `Alignment.pairs` is plain
data and is meant to be overridable by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from xp.core.frame import frame_for_page, frame_stroke
from xp.core.document.xopp_io import Page, XoppDocument

# **Finer than it used to be, because 16 cells a side could not tell two dense
# pages apart.** Measured on the two sample files (2 pages against 61): at 16x16
# the right answer scored 0.758 while a wrong page scored 0.760 -- a coin flip.
# At 32x32 with the mean removed the right answer is 0.61 and the next is 0.11.
GRID = 32            # page is described by a GRID x GRID ink-coverage map
MATCH_THRESHOLD = 0.30  # below this, a pair isn't considered the same page
GAP_PENALTY = -0.25    # cost of calling a page inserted rather than matched
SHIFT = 1             # cells of slack, so a page written a little off still fits


def reference_box(page: Page) -> Tuple[float, float, float, float]:
    """The box a signature is measured in: the writing area, not the paper.

    Measuring against the page box assumes both sides put the writing in the
    same place on the same paper, and a PDF exported with wide margins does
    not -- its ink, and the frame around it, sit shrunk in the middle, and
    against the full-size original it scores near nothing.

    The frame is the answer to exactly that question, which is why the
    template draws it. Here an *assumed* frame counts as well as an observed
    one: a Q base page with no frame in it really was written on the
    template, so the template's frame is where its writing area is. Only a
    page that is neither framed nor Q base falls back to its own page box.

    The ink's own bounding box is deliberately not used. It would normalize
    a page carrying one mark into a page full of marks, and every sparse
    page would look like every other.
    """
    size = (max(page.width, 1e-6), max(page.height, 1e-6))
    found = frame_for_page(page, size)
    return found.box if found is not None else (0.0, 0.0, size[0], size[1])


def page_ink(page: Page) -> list:
    """The page's strokes, without the frame.

    The frame is on the page but it is not writing, and it is on *every*
    page -- leaving it in makes every page look a little like every other.

    **All of it, not just the piece `frame_stroke` happens to return.** A traced
    border is hundreds of short pieces (the user's converted file has 3,028 of them
    across two pages), so dropping the one found outline left almost all of it
    in the signature.
    """
    from xp.core.frame import without_the_print_frame

    strokes = [s for s in page.strokes() if s.points]
    return without_the_print_frame(
        strokes, (max(page.width, 1e-6), max(page.height, 1e-6)))


def _map_ink(ink, box: Tuple[float, float, float, float], grid: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    w = max(x1 - x0, 1e-6)
    h = max(y1 - y0, 1e-6)
    # **One count for the whole page, not one per stroke.** This was a loop of
    # `np.add.at(sig, (gy, gx), 1.0)` once per stroke, and `np.add.at` is
    # numpy's slow path -- paid 3,479 times on the user's densest page. The points
    # are stacked once and counted once.
    #
    # The answer is **bitwise identical**, not merely close: every point adds
    # exactly 1.0, so each cell holds a whole number, and whole numbers below
    # 2**53 are exact in float64 however they are summed.
    # `test_the_signature_is_the_same_only_faster` asserts that against the
    # old implementation on the sample notebook.
    if not ink:
        return np.zeros((grid, grid), dtype=np.float64)
    pts = np.concatenate([np.asarray(stroke.points, dtype=np.float64)
                          for stroke in ink])
    gx = np.clip(((pts[:, 0] - x0) / w * grid).astype(int), 0, grid - 1)
    gy = np.clip(((pts[:, 1] - y0) / h * grid).astype(int), 0, grid - 1)
    sig = (np.bincount(gy * grid + gx, minlength=grid * grid)
           .astype(np.float64).reshape(grid, grid))

    if sig.max() > 0:
        # Occupancy, not stroke count: a cell either has ink or it doesn't,
        # softened so a densely-inked cell still outweighs a barely-touched one.
        sig = np.sqrt(sig)
        # **The mean goes too.** Raw coverage is dominated by how much ink a
        # page has overall: a densely hatched page and a lightly written one
        # are both "everywhere", and their cosine is high whatever the writing
        # is. What identifies a page is where the ink is *relative to the rest
        # of that page*, which is what removing the mean leaves.
        sig = sig - sig.mean()
        norm = np.linalg.norm(sig)
        if norm > 0:
            sig = sig / norm
    return sig


def page_signature(page: Page, grid: int = GRID) -> np.ndarray:
    """Where the ink is within this page's writing area."""
    return _map_ink(page_ink(page), reference_box(page), grid)


def _normalised(sig: np.ndarray) -> np.ndarray:
    """A signature with its mean removed and unit length, or zeros if blank."""
    if sig.size == 0:
        return sig
    centred = sig - sig.mean()
    norm = np.linalg.norm(centred)
    return centred / norm if norm > 0 else np.zeros_like(sig)


def signature_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """How alike two pages look, in 0..1.

    **A blank page matches nothing, including another blank page.** Two empty
    pages carry no evidence, and calling them a match is the one answer that is
    certainly wrong -- an unrendered page, a failed render and a page nobody
    wrote on all look the same from here, and the map must say so rather than
    pair them up. `is_blank` is what the caller asks instead.

    A little slack in position: a page written two cells over is the same page,
    not a different one, so the best of the small shifts is taken.
    """
    norm_a = _normalised(a)
    norm_b = _normalised(b)
    if not norm_a.any() or not norm_b.any():
        return 0.0
    best = 0.0
    for dy in range(-SHIFT, SHIFT + 1):
        for dx in range(-SHIFT, SHIFT + 1):
            moved = b
            if dx or dy:
                moved = np.roll(np.roll(b, dy, axis=0), dx, axis=1)
                moved = _normalised(moved)
            score = float((norm_a * moved).sum())
            if score > best:
                best = score
    return float(np.clip(best, 0.0, 1.0))


def is_blank(signature: np.ndarray) -> bool:
    """Whether this page carries no ink at all (or could not be read)."""
    return not bool(np.any(_normalised(signature)))


def image_signature(image_bgr, grid: int = GRID) -> np.ndarray:
    """Where the ink is on a *picture* of a page.

    A PDF side has no strokes to measure -- that is the whole point of the
    deferred-conversion protocol -- so `page_signature` sees a blank page and
    every comparison against it scores zero. The page map then "auto matches"
    nothing at all, which is the reliable failure the user reported on a PDF/XOPP
    pair.

    What a page looks like is enough: the paper is most of it, ink is what is
    much darker than the paper, and the frame -- printed or not -- says which
    rectangle the writing lives in, so both sides are measured in the same
    place.
    """
    if image_bgr is None or getattr(image_bgr, 'size', 0) == 0:
        return np.zeros((grid, grid), dtype=np.float64)
    import cv2

    from xp.core.frame import frame_in_image

    gray = cv2.cvtColor(np.ascontiguousarray(image_bgr), cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    paper = float(np.percentile(gray, 90)) if gray.size else 255.0
    ink = gray < max(24.0, paper * 0.62)
    found = frame_in_image(image_bgr)
    x0, y0, x1, y1 = ((found.x0, found.y0, found.x1, found.y1)
                      if found is not None else (0.0, 0.0, float(width), float(height)))
    span_x = max(x1 - x0, 1e-6)
    span_y = max(y1 - y0, 1e-6)
    sig = np.zeros((grid, grid), dtype=np.float64)
    rows, cols = np.nonzero(ink)
    if len(rows) == 0:
        return sig
    gx = np.clip(((cols - x0) / span_x * grid).astype(int), 0, grid - 1)
    gy = np.clip(((rows - y0) / span_y * grid).astype(int), 0, grid - 1)
    np.add.at(sig, (gy, gx), 1.0)
    sig = np.sqrt(sig)
    if sig.max() > 0:
        sig /= np.linalg.norm(sig)
    return sig


def side_signature(side, index: int, grid: int = GRID) -> np.ndarray:
    """A page's signature from whichever description the side really has.

    Strokes when the side is a notebook; a render when it is a picture. The
    same grid either way, so a PDF page and a traced page are comparable.
    """
    strokes = None
    try:
        strokes = side.strokes(index)
    except Exception:
        strokes = None
    if strokes is not None:
        page = side.document.pages[index] if hasattr(side, 'document') else None
        if page is not None:
            return page_signature(page, grid)
    image = side.render(index, 0.5)
    return image_signature(image, grid)


def align_sides(left, right, threshold: float = MATCH_THRESHOLD,
                gap_penalty: float = GAP_PENALTY, progress=None) -> 'Alignment':
    """`align_pages` for two sides, one or both of which may be a picture."""
    total = left.pages + right.pages
    left_sigs = []
    for index in range(left.pages):
        left_sigs.append(side_signature(left, index))
        if progress is not None:
            progress(index + 1, total)
    right_sigs = []
    for index in range(right.pages):
        right_sigs.append(side_signature(right, index))
        if progress is not None:
            progress(left.pages + index + 1, total)
    return _align_signatures(left_sigs, right_sigs, threshold, gap_penalty)


def _hungarian(cost):
    """The assignment of `cost` (square, minimised) that can be proved optimal.

    O(n^3) potentials: rows are matched one at a time and the previous choice is
    repaired by a shortest augmenting path. n is the longer side's page count
    (61 for the user's pair), so this costs a fraction of the signatures it decides
    about.
    """
    n = len(cost)
    INF = float('inf')
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    answer = [-1] * n
    for j in range(1, n + 1):
        if p[j]:
            answer[p[j] - 1] = j - 1
    return answer


def _optimal_assignment(matrix, threshold):
    """Which left page faces which right page, weighing the whole matrix.

    A greedy pass cannot see this: for

            right 1  right 2
    left 1     0.90     0.80
    left 2     0.85     0.00

    it takes 0.90 and leaves left 2 with nothing, while 0.80 + 0.85 = 1.65 is
    better. Pair-wise swaps do not rescue it either, because the swap loop only
    looks at pages that were matched, and left 2 never was.

    So the assignment problem is solved properly. **"Unmatched" is a choice
    like any other**: the matrix is padded to a square with dummy rows and
    columns worth 0, and a pair below the threshold is worth -1, which can never
    be preferred to a dummy. What comes out is the maximum total score; nothing
    is forced, and nothing that could be paired is left out.
    """
    import numpy as np

    rows, columns = matrix.shape
    size = max(rows, columns, 1)
    weight = np.zeros((size, size), dtype=np.float64)
    if rows and columns:
        weight[:rows, :columns] = np.where(matrix >= threshold, matrix, -1.0)
    assignment = _hungarian((-weight).tolist())
    pairs = []
    for left in range(rows):
        right = assignment[left] if left < len(assignment) else -1
        if 0 <= right < columns and weight[left, right] > 0.0:
            pairs.append((left, right, float(matrix[left, right])))
    paired_left = {left for left, _r, _s in pairs}
    paired_right = {right for _l, right, _s in pairs}
    answer = list(pairs)
    answer += [(left, None, 0.0) for left in range(rows)
               if left not in paired_left]
    answer += [(None, right, 0.0) for right in range(columns)
               if right not in paired_right]
    answer.sort(key=lambda pair: (pair[0] if pair[0] is not None else rows,
                                  pair[1] if pair[1] is not None else columns))
    return answer


def _best_assignment(pairs, m, threshold):
    """The pairing for the whole document, decided over the whole matrix.

    `pairs` is kept for the caller's signature and **is not read**: the
    order-preserving pass behind it cannot express a crossing pair, and on the user's
    own files it matched left 2 to right 10 at **0.392** while left 2 to right 1
    scored **0.822**, only because right 1 sits before left 1's partner. Filling
    in the pages it left over was not enough -- the page was not left over, it
    was matched badly.

    So the answer comes from `_optimal_assignment`, which may leave a page
    unmatched and still finds the best total.
    """
    return _optimal_assignment(m, threshold)


def similarity_matrix(left: XoppDocument, right: XoppDocument) -> np.ndarray:
    left_sigs = [page_signature(p) for p in left.pages]
    right_sigs = [page_signature(p) for p in right.pages]
    m = np.zeros((len(left_sigs), len(right_sigs)), dtype=np.float64)
    for i, a in enumerate(left_sigs):
        for j, b in enumerate(right_sigs):
            m[i, j] = signature_similarity(a, b)
    return m


@dataclass
class Alignment:
    # (left_index, right_index, score); either index may be None, meaning
    # that page exists on one side only.
    pairs: List[Tuple[Optional[int], Optional[int], float]]
    matrix: np.ndarray

    @property
    def matched(self) -> List[Tuple[int, int, float]]:
        return [(l, r, s) for l, r, s in self.pairs if l is not None and r is not None]

    @property
    def left_only(self) -> List[int]:
        return [l for l, r, _ in self.pairs if r is None and l is not None]

    @property
    def right_only(self) -> List[int]:
        return [r for l, r, _ in self.pairs if l is None and r is not None]

    def right_for_left(self, left_index: int) -> Optional[int]:
        for l, r, _ in self.pairs:
            if l == left_index:
                return r
        return None

    def summary(self) -> str:
        lines = [f"{len(self.matched)} pages matched, "
                 f"{len(self.left_only)} only on the left, "
                 f"{len(self.right_only)} only on the right"]
        for l, r, s in self.pairs:
            left_s = "--" if l is None else str(l + 1)
            right_s = "--" if r is None else str(r + 1)
            note = "" if l is not None and r is not None else "   (unmatched)"
            score = f"{s:.2f}" if l is not None and r is not None else "    "
            lines.append(f"  left {left_s:>3}  <->  right {right_s:>3}   {score}{note}")
        return "\n".join(lines)


def _signatures(pages, progress=None, start: int = 0, total: int = 1) -> list:
    """Signatures for a document's pages, reporting as it goes.

    **This is where the time is.** On the sample files (two pages of 43,677 and
    22,460 strokes against 61 pages) the 63 signatures cost 6.6 s and the
    comparison that follows costs 0.01 s. A progress bar that only moved during
    the comparison would be a lie; this is the loop it counts.
    """
    signatures = []
    for index, page in enumerate(pages):
        signatures.append(page_signature(page))
        if progress is not None:
            progress(start + index + 1, total)
    return signatures


def align_pages(left: XoppDocument, right: XoppDocument,
                 threshold: float = MATCH_THRESHOLD,
                 gap_penalty: float = GAP_PENALTY,
                 progress=None) -> Alignment:
    """Order-preserving alignment of left's pages to right's.

    Standard Needleman-Wunsch: a pair may only be matched if it scores at
    least `threshold`, otherwise the two pages are treated as unrelated and
    each becomes an insertion on its own side.
    """
    total = len(left.pages) + len(right.pages)
    left_sigs = _signatures(left.pages, progress, 0, total)
    right_sigs = _signatures(right.pages, progress, len(left.pages), total)
    return _align_signatures(left_sigs, right_sigs, threshold, gap_penalty)


def _align_signatures(left_sigs, right_sigs, threshold, gap_penalty) -> 'Alignment':
    """The alignment itself, from signatures, whoever produced them."""
    m = np.zeros((len(left_sigs), len(right_sigs)), dtype=np.float64)
    for i, a in enumerate(left_sigs):
        for j, b in enumerate(right_sigs):
            m[i, j] = signature_similarity(a, b)
    n_left, n_right = m.shape

    score = np.zeros((n_left + 1, n_right + 1), dtype=np.float64)
    # 0 = diagonal (match), 1 = up (left-only), 2 = left (right-only)
    back = np.zeros((n_left + 1, n_right + 1), dtype=np.int8)

    for i in range(1, n_left + 1):
        score[i, 0] = score[i - 1, 0] + gap_penalty
        back[i, 0] = 1
    for j in range(1, n_right + 1):
        score[0, j] = score[0, j - 1] + gap_penalty
        back[0, j] = 2

    for i in range(1, n_left + 1):
        for j in range(1, n_right + 1):
            s = m[i - 1, j - 1]
            diag = score[i - 1, j - 1] + (s if s >= threshold else -np.inf)
            up = score[i - 1, j] + gap_penalty
            left_move = score[i, j - 1] + gap_penalty
            best = max(diag, up, left_move)
            score[i, j] = best
            back[i, j] = 0 if best == diag else (1 if best == up else 2)

    pairs: List[Tuple[Optional[int], Optional[int], float]] = []
    i, j = n_left, n_right
    while i > 0 or j > 0:
        move = back[i, j] if (i > 0 and j > 0) else (1 if i > 0 else 2)
        if move == 0:
            pairs.append((i - 1, j - 1, float(m[i - 1, j - 1])))
            i, j = i - 1, j - 1
        elif move == 1:
            pairs.append((i - 1, None, 0.0))
            i -= 1
        else:
            pairs.append((None, j - 1, 0.0))
            j -= 1

    pairs.reverse()
    return Alignment(pairs=_best_assignment(pairs, m, threshold), matrix=m)
