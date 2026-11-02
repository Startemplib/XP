"""The merge itself: two source documents, one set of decisions, one output.

Nothing here edits either source document. A merge is a set of decisions
about strokes (see compose.py), and the merged document is built from those
decisions when it is exported -- by the same function that builds the Final
Preview, so the preview cannot disagree with the file.

That is a change from how this worked: decisions used to be applied straight
into the left document's XML as you brushed, which made "left" mean "the
document being edited" and "right" mean "ink copied on top". Left and right
are now two equal sides of one composition, told apart by identity rather
than by which one is underneath.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np

from xp.workers.merger.brush import BrushMask
from xp.workers.merger.compose import (KEEPS, LEFT, RIGHT, SIDES, PageDecisions, Ref, TAKE_BOTH, TAKE_LEFT,
                      TAKE_NONE, TAKE_RIGHT)
from xp.core.inkfit import Fit, fit as fit_ink
from xp.core.frame import REFERENCE_FRAME, align as align_frame, frame_for_page
from xp.core.document.pagesize import choose_target, is_qbase
from xp.render.drawing import DEFAULT_DISPLAY_SCALE
from xp.workers.merger.classification import DEFAULT_DPI as DEFAULT_CLASSIFY_DPI
from xp.core.document.xopp_io import (Stroke, XoppDocument, slice_stroke_span,
                                      transform_stroke_element)

# Below this much of the two inks coinciding, the fit is not believed and the
# page boxes are used instead: a page that shares no content with its partner
# has nothing to be fitted to, and a bad fit is worse than none.
MIN_FIT_SCORE = 0.25
# The range the classification's DPI may be set to, in pixels per inch. The
# floor is the page at one pixel per unit (72), below which a pen stroke is
# thinner than the grid it is classified on; the ceiling is four times the
# display scale (1152), above which the classification costs more than the
# detail it finds and the page is the same page.
MIN_CLASSIFY_DPI = 72.0
MAX_CLASSIFY_DPI = 8000.0

# **How large a whole page may be rasterised at all.** One A4-ish page at 8000 dpi is 20,846 x 43,440 pixels -- 2.7 GB
# as RGB -- so the whole-page paths (classification, the fit's material, the paper) stop at this many pixels and say so
# once. The window path is not affected: a window is the size of the screen whatever the DPI says.
MAX_WHOLE_PAGE_PIXELS = 40_000_000

# **And a window is not unlimited either.** A window is the screen, but a caller can hand a large rectangle to the same
# path, and at 8000 dpi a 700 x 1800 pixel window already covers only a fifth of a page. This cap keeps one window at
# 60 Mpx of RGB -- 180 MB -- and says so once.
MAX_WINDOW_PIXELS = 16_000_000

#: Set when a whole-page render had to be coarser than the DPI asked for, so it can be said once, not every frame.
_capped_whole_page = set()
# How much of the two inks must coincide before an ink fit is believed
# instead of the printed border. Measured: a correct fit on real
# pages scores above 0.5; unrelated ink scores below 0.2.
INK_FIT_ENOUGH = 0.35

ALL_PAGES = -1      # the key a hand-set fit uses when it applies to every page
MARGIN_SLACK = 2.0  # pt a stroke may hang off the page before it counts as margin
# How many content points one side contributes to the Manual Fit. The fit scores
# on a grid 320 across, so more than this cannot be told apart anyway; a dense
# page of handwriting reaches it, and a picture side is sampled down to it.
_FIT_MOST_POINTS = 60000

__all__ = ['MergeSession', 'append_merge', 'LEFT', 'RIGHT', 'Ref',
           'TAKE_LEFT', 'TAKE_RIGHT', 'TAKE_BOTH', 'TAKE_NONE']


class _Action(list):
    """One undoable gesture: the stroke records it already was, plus whether it moved the map.

    The list of moved decisions and the region geometry are the same gesture seen twice, and undo
    has to put both back or they drift apart. A flag on the gesture is enough: `region` says the
    gesture also appended a polygon, so `undo` and `redo` know to pop or push the map with it.
    """

    region = False



def _resample_mask(mask, old_fit, new_fit, scale: int):
    """The same decisions, on the ink they were made about, after a fit change.

    `mask` is in merged page **pixels** at `scale` pixels per page unit. A pixel
    becomes a point in the side's own units through one fit and comes back
    through the other, which is what makes the decision follow the content
    instead of staying on the screen position.
    """
    import numpy as np

    height, width = mask.shape
    # Pixel centre -> page units -> side units, once per axis: the transform is
    # affine and axis-aligned, so this does not need a full coordinate grid.
    columns = (np.arange(width) + 0.5) / float(scale)
    rows = (np.arange(height) + 0.5) / float(scale)
    new_x, new_y = new_fit.sx or new_fit.scale, new_fit.sy or new_fit.scale
    old_x, old_y = old_fit.sx or old_fit.scale, old_fit.sy or old_fit.scale
    side_x = (columns - new_fit.dx) / (new_x or 1.0)
    side_y = (rows - new_fit.dy) / (new_y or 1.0)
    old_columns = np.rint(side_x * (old_x or 1.0) + old_fit.dx) * scale - 0.5
    old_rows = np.rint(side_y * (old_y or 1.0) + old_fit.dy) * scale - 0.5
    old_columns = np.clip(old_columns.astype(np.int64), 0, width - 1)
    old_rows = np.clip(old_rows.astype(np.int64), 0, height - 1)
    moved = np.zeros_like(mask)
    moved[:, :] = mask[np.ix_(old_rows, old_columns)]
    return moved


def _pen_width(element) -> float:
    """The stroke's nominal width, whatever else the attribute holds.

    Xournal++ writes pressure as `width="nominal w0 w1 ..."`, one width per
    segment, so `float(width)` on such a stroke raises -- measured, on the user's own
    notebook's strokes, in the middle of an export. The nominal is the first
    number, and it is what a cut needs.
    """
    numbers = (element.get('width') or '').split()
    try:
        return float(numbers[0])
    except (IndexError, ValueError):
        return 1.0


# How close two ends have to be before they are the same point. The cut is
# computed once per side from the same mask, so a pair that really continues
# lands on identical coordinates; this is the slack for the odd rounding in a
# transformed stroke, not a tolerance for a gap.
LINK_REACH = 0.5
# How far the line may turn at the joint and still be the same line. A cut
# leaves the direction untouched, so a genuine joint turns by nothing; a corner
# in two different strokes turns by most of a right angle.
LINK_TURN = 35.0


def _stroke_points(element) -> List[Tuple[float, float]]:
    numbers = (element.text or '').split()
    try:
        values = [float(number) for number in numbers]
    except ValueError:
        return []
    return list(zip(values[0::2], values[1::2]))


def _the_same_pen(first, second) -> bool:
    """Same colour and same pen, which is what "the same line" means here.

    **The colour is part of this test and only part of this test.** During the
    merge the colour never decides anything; at export it is restored, so two
    pieces in different colours are two different lines however well their ends
    meet, and joining them would silently repaint one of them.
    """
    if (first.get('color') or '').lower() != (second.get('color') or '').lower():
        return False
    if (first.get('fill') or '0') != (second.get('fill') or '0'):
        return False
    one, other = _pen_width(first), _pen_width(second)
    if abs(one - other) > 0.05 * max(one, other, 1e-6):
        return False
    # A pressure stroke carries one width per segment. Linking two of those
    # would have to merge the lists, and the nominal alone does not describe
    # the pen -- so they are left as they are rather than guessed at.
    return len((first.get('width') or '').split()) <= 1 and len(
        (second.get('width') or '').split()) <= 1


def _continues(before, after) -> bool:
    """Does `after` carry on from `before` -- same end, same direction?"""
    import math

    head, tail = _stroke_points(before), _stroke_points(after)
    if len(head) < 2 or len(tail) < 2:
        return False
    gap = math.hypot(tail[0][0] - head[-1][0], tail[0][1] - head[-1][1])
    if gap > LINK_REACH:
        return False
    # The direction of the last drawn segment and of the first one after it.
    for at in range(len(head) - 2, -1, -1):
        if math.hypot(head[-1][0] - head[at][0], head[-1][1] - head[at][1]) > 1e-9:
            leaving = math.atan2(head[-1][1] - head[at][1], head[-1][0] - head[at][0])
            break
    else:
        return False
    for at in range(1, len(tail)):
        if math.hypot(tail[at][0] - tail[0][0], tail[at][1] - tail[0][1]) > 1e-9:
            arriving = math.atan2(tail[at][1] - tail[0][1], tail[at][0] - tail[0][0])
            break
    else:
        return False
    turn = abs(math.degrees((leaving - arriving + math.pi) % (2 * math.pi) - math.pi))
    return turn <= LINK_TURN


def _link_the_pieces(elements: List) -> List:
    """Join pieces that are one line into one stroke, and change nothing else.

    Heal, in the user's definition: the file should hold fewer `<stroke>` objects where
    two pieces are plainly the same line, not more ink. The output of this
    function therefore draws **exactly** the same pixels as its input -- the
    joint point is written once instead of twice, on the same coordinates -- and
    the only thing that shrinks is the stroke count.

    Order is the order it was given, so the layers of a page do not reshuffle:
    a chain keeps the place of the piece that started it.
    """
    import copy as _copy
    import math

    if len(elements) < 2:
        return list(elements)

    # **Once per element, not once per comparison.** `_the_same_pen` splits and lower-cases
    # the colour, fill and width attributes; called 3,377,829 times for 2,600 pieces, it was
    # the export's whole cost.
    def pen_of(element):
        return ((element.get('color') or '').lower(),
                (element.get('fill') or '0'),
                _pen_width(element),
                len((element.get('width') or '').split()) <= 1)

    points_of = [_stroke_points(element) for element in elements]
    pens = [pen_of(element) for element in elements]

    def the_same_pen(one, other):
        if pens[one][:2] != pens[other][:2]:
            return False
        width_one, width_other = pens[one][2], pens[other][2]
        if abs(width_one - width_other) > 0.05 * max(width_one, width_other, 1e-6):
            return False
        return pens[one][3] and pens[other][3]

    def leaving(head):
        if len(head) < 2:
            return None
        for at in range(len(head) - 2, -1, -1):
            if math.hypot(head[-1][0] - head[at][0],
                          head[-1][1] - head[at][1]) > 1e-9:
                return math.atan2(head[-1][1] - head[at][1],
                                  head[-1][0] - head[at][0])
        return None

    def arriving(tail):
        if len(tail) < 2:
            return None
        for at in range(1, len(tail)):
            if math.hypot(tail[at][0] - tail[0][0],
                          tail[at][1] - tail[0][1]) > 1e-9:
                return math.atan2(tail[at][1] - tail[0][1],
                                  tail[at][0] - tail[0][0])
        return None

    countinues = [None] * len(elements)

    def the_same_line(one, other):
        """`_continues(elements[one], elements[other])`, over points parsed once."""
        head, tail = points_of[one], points_of[other]
        if len(head) < 2 or len(tail) < 2:
            return False
        gap = math.hypot(tail[0][0] - head[-1][0], tail[0][1] - head[-1][1])
        if gap > LINK_REACH:
            return False
        out = leaving(head)
        into = arriving(tail)
        if out is None or into is None:
            return False
        turn = abs(math.degrees((out - into + math.pi) % (2 * math.pi) - math.pi))
        return turn <= LINK_TURN

    # **A bucket by first point**, one cell wide in LINK_REACH: a candidate further than that
    # from the chain's last point cannot pass `_continues`, so looking in the neighbouring
    # cells finds exactly the pieces the list scan could have matched -- in the same order.
    cell = LINK_REACH if LINK_REACH > 1e-9 else 1.0
    buckets = {}
    for at, held in enumerate(points_of):
        if len(held) < 2:
            continue
        key = (int(math.floor(held[0][0] / cell)),
               int(math.floor(held[0][1] / cell)))
        buckets.setdefault(key, []).append(at)

    def candidates_near(tail_point):
        x = int(math.floor(tail_point[0] / cell))
        y = int(math.floor(tail_point[1] / cell))
        found = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                found.extend(buckets.get((x + dx, y + dy), ()))
        found.sort()
        return found

    used = [False] * len(elements)
    chains = []            # lists of element indices, in order
    for start in range(len(elements)):
        if used[start]:
            continue
        used[start] = True
        chain = [start]
        grew = True
        while grew:
            grew = False
            tail = points_of[chain[-1]]
            if len(tail) < 2:
                break
            for at in candidates_near(tail[-1]):
                if used[at]:
                    continue
                if the_same_pen(chain[-1], at) and the_same_line(chain[-1], at):
                    chain.append(at)
                    used[at] = True
                    grew = True
                    break
        chains.append([elements[at] for at in chain])

    out = []
    for chain in chains:
        if len(chain) == 1:
            out.append(chain[0])
            continue
        joined = _copy.deepcopy(chain[0])
        points = []
        for piece in chain:
            piece_points = _stroke_points(piece)
            if points and piece_points and abs(points[-1][0] - piece_points[0][0]) < 1e-9 \
                    and abs(points[-1][1] - piece_points[0][1]) < 1e-9:
                piece_points = piece_points[1:]
            points.extend(piece_points)
        joined.text = ' '.join('%.4f %.4f' % (x, y) for x, y in points)
        out.append(joined)
    return out


def _owner_runs(element, owner, scale: float) -> List[Tuple[float, float, int]]:
    """`(low, high, owner)` along this stroke, in point-index fractions.

    One walk of the polyline, one mask lookup per step: a run ends where the
    **owner** of the page changes, which is where the other side's area begins.
    Positions are in the unit `slice_stroke_span` takes, so a run can be cut with it
    directly -- and because both sides are cut from the same plane, they are cut at
    the same boundary: no gap and no doubled ink by construction, which is what the
    piece path gave and what `_visible_spans`' pen inset could not, since its inset
    exists for a *converted* stroke whose cap would otherwise double.
    """
    coords = (element.text or '').split()
    if len(coords) < 4:
        return []
    tall, wide = owner.shape
    points = [(float(coords[at]) * scale, float(coords[at + 1]) * scale)
              for at in range(0, len(coords) - 1, 2)]

    def owner_at(x, y) -> int:
        px, py = int(round(x)), int(round(y))
        if not (0 <= px < wide and 0 <= py < tall):
            return 0
        return int(owner[py, px])

    walks = []
    for index, ((x0, y0), (x1, y1)) in enumerate(zip(points, points[1:])):
        steps = max(1, int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5))
        for k in range(steps):
            t = k / steps
            walks.append((index + t, owner_at(x0 + (x1 - x0) * t,
                                              y0 + (y1 - y0) * t)))
    walks.append((float(len(points) - 1), owner_at(*points[-1])))

    runs = []
    start, held = None, None
    for position, value in walks:
        if value != held:
            if held is not None and start is not None:
                runs.append((start, position, held))
            start, held = position, value
    if held is not None and start is not None:
        runs.append((start, float(len(points) - 1), held))
    return runs


def _visible_spans(element, taken, scale: float, width: float = 1.0) -> List[Tuple[float, float]]:
    """The parts of this stroke that lie inside the area the brush took.

    `_covered_by` answers "is all of it inside", which is the right question
    for a *decision* and the wrong one for the **file**: a converted line that
    the brush took half of was dropped whole, so the export lost the half the user had
    just selected. The user's rule for the converter's output is the same as for a
    native stroke -- "if only part of a line is finally selected, convert only
    the visible part" -- so the answer is spans, and `slice_stroke_span` cuts
    the element at them, inside a segment if that is where the edge falls.

    or the outside by its width. Positions are **point-index fractions**, which
    is the unit `slice_stroke_span` takes: `2.5` is half way along the segment
    from point 2 to point 3. The walk is one mask pixel at a time, so where a
    stroke enters the area is known to within a pixel of the mask the brush
    painted.

    **Its own width has to be inside too.** The notebook side is cut where its
    centre line leaves the mask, so its pen still reaches half a width outward;
    clipping the converted pen the same way leaves the two caps side by side at
    the seam instead of on top of each other. Measured on the user's pair, that was the
    difference between one doubled stroke and none.
    """
    coords = (element.text or '').split()
    if len(coords) < 4:
        return []
    tall, wide = taken.shape
    radius = max(0.0, float(width) * scale * 0.5)
    offsets = ((0.0, 0.0), (radius, 0.0), (-radius, 0.0),
               (0.0, radius), (0.0, -radius))
    points = [(float(coords[at]) * scale, float(coords[at + 1]) * scale)
              for at in range(0, len(coords) - 1, 2)]

    def inside(x, y) -> bool:
        for dx, dy in offsets:
            px, py = int(round(x + dx)), int(round(y + dy))
            if not (0 <= px < wide and 0 <= py < tall) or not taken[py, px]:
                return False
        return True

    # **A flag per sample, with its position in point-index space.** The
    # vertices come first and each segment adds its interior, so the list is in
    # stroke order and the runs come out of one pass.
    walks = []
    for index, ((x0, y0), (x1, y1)) in enumerate(zip(points, points[1:])):
        span = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        steps = max(1, int(span))
        for k in range(steps):
            t = k / steps
            walks.append((index + t, inside(x0 + (x1 - x0) * t,
                                            y0 + (y1 - y0) * t)))
    walks.append((float(len(points) - 1), inside(*points[-1])))

    spans = []
    start = None
    for position, here in walks:
        if here and start is None:
            start = position
        elif not here and start is not None:
            spans.append((start, position))
            start = None
    if start is not None:
        spans.append((start, walks[-1][0]))
    # A span of no length draws nothing.
    return [(low, high) for low, high in spans if high - low > 1e-9]


def append_merge(left: XoppDocument, right: XoppDocument) -> XoppDocument:
    """Right's pages after left's, unchanged: the old Merge_xopp.py's job,
    kept because plain concatenation never needed reviewing.

    Mutates and returns `left` rather than copying it first: these documents
    can hold tens of thousands of stroke elements, and this mode exists to be
    cheap.
    """
    for page in right.root.findall('page'):
        left.root.append(page)
    return left


def _onto(size, target) -> Fit:
    """A page's own box scaled onto the target, centred: no ink involved."""
    scale = min(target[0] / size[0], target[1] / size[1])
    return Fit(scale, (target[0] - size[0] * scale) / 2, (target[1] - size[1] * scale) / 2,
               0.0, 'page box onto Q base')


def _placed_stroke(stroke: Stroke, fit: Fit) -> Stroke:
    return Stroke(element=stroke.element,
                  points=[fit.apply(x, y) for x, y in stroke.points],
                  color=stroke.color,
                  widths=[w * fit.pen_scale for w in stroke.widths],
                  fill=stroke.fill)


@dataclass(frozen=True)
class PageGeometry:
    """One merged page's size, and how each side's ink is placed into it."""

    size: Tuple[float, float]
    fits: Dict[str, Fit]
    aspects_agree: bool = True
    reason: str = ''

    @property
    def compatible(self) -> bool:
        """Whether the two pages could be brought together at all.

        Different-shaped pages are not a refusal by themselves: what matters
        is whether the ink could be fitted. When it could not, the merge
        still happens -- on the page boxes -- and the window says so.
        """
        return self.aspects_agree or any(f.score >= MIN_FIT_SCORE for f in self.fits.values())


@dataclass
class MergedPage:
    """One page of the merged document, and where its two sides come from.

    The merge is a list of these rather than a walk over the left document's
    pages. That is what lets a split be a real page: splitting replaces one
    entry that has both sides with two that have one each, the page count
    goes up by one the moment you ask for it, and the two can be reordered
    like any other pages. `key` is what decisions are filed under, so they
    survive pages being inserted, moved or re-paired around them.
    """

    key: int
    left: Optional[int] = None      # index into the left document's pages
    right: Optional[int] = None     # index into the right document's pages

    @property
    def origin(self) -> str:
        if self.left is not None and self.right is not None:
            return 'both'
        return 'left only' if self.left is not None else 'right only'


def _stand_in(side) -> XoppDocument:
    """A document of the right shape with no strokes in it, for a PDF side.

    A PDF has ink, not strokes, and turning the one into the other costs
    about twenty minutes for the user's notebook. The merge window used to pay that
    before it would open. It does not have to: everything in here needs a
    page to have a *size* and a list of strokes, and a page of a PDF can
    answer the first honestly and the second with nothing -- because there
    are no strokes yet, and whether there ever need to be is what the brush
    is about to decide.

    So the PDF side is this, plus the `Side` that can draw it; the drawing
    is what the user brushes over, and `export` converts the pages the brush
    actually took something from.
    """
    document = XoppDocument.new_empty(page_count=side.pages)
    for index, page in enumerate(document.pages):
        width, height = side.page_box(index)
        page.element.set('width', f'{width:.4f}')
        page.element.set('height', f'{height:.4f}')
    return document


def _source(given):
    """(document, side) for whatever was handed in.

    A path or a `Side` on either side; an `XoppDocument` still works and
    comes back with no `Side`, so every existing caller is unchanged.
    """
    from xp.workers.merger import sides as _sides

    if isinstance(given, XoppDocument):
        return given, None
    side = given if isinstance(given, _sides.Side) else _sides.Side(given)
    if side.kind == _sides.XOPP:
        return side.document, side
    return _stand_in(side), side


# Written as codes rather than escapes so that a tool that rewrites this
# file cannot quietly turn one into a real line break, which is how the
# first draft of `save_page_by_page` came out unparseable.
_BREAK = bytes([10])
_DECLARATION = ("<?xml version='1.0' encoding='UTF-8'?>").encode() + _BREAK
_CLOSING = b'</xournal>' + _BREAK


_ANONYMOUS_KEYS = None          # one WeakKeyDictionary, made on first use


def _document_key(document, side=None) -> str:
    """What a document is, for the life of a session -- **the same string every time**.

    Its path when one is known -- the same notebook opened twice is the same
    notebook -- and otherwise a counter, **never an address**: `id()` is the
    first thing a released object's successor gets, which is how one notebook's
    book could be handed to another. The path lives on the `Side` that opened the
    document (`XoppDocument` itself does not keep one).

    **The counter used to advance on every call**, so a session built from
    documents rather than paths (`MergeSession(left_doc, right_doc)`, which is
    what the tests and any in-memory caller do) got a *new* key each time
    `layers_for_page` looked one up. Every cache-miss then rebuilt the piece
    books and re-seeded them from the boolean `kept` map, which cannot say
    "Both": a piece decided Both came back as Left on one side and Right on the
    other the moment a Fit changed the geometry. Measured on
    `test_fit_controls`' own fixture -- book 2 before the fit, book 4 after, the
    states `[3, 3, 3]` -> `[2, 2, 2]`. The key is held per document now, weakly
    so an abandoned document is not kept alive by its own name.
    """
    from pathlib import Path as _Path

    for candidate in (getattr(side, 'path', None),
                      getattr(document, 'path', None)):
        if candidate:
            return 'path:' + str(_Path(candidate).resolve())

    global _ANONYMOUS_KEYS
    if _ANONYMOUS_KEYS is None:
        import weakref

        _ANONYMOUS_KEYS = weakref.WeakKeyDictionary()
    held = _ANONYMOUS_KEYS.get(document)
    if held is not None:
        return held
    counter = getattr(_document_key, '_count', 0) + 1
    _document_key._count = counter
    name = 'anonymous:%d' % counter
    try:
        _ANONYMOUS_KEYS[document] = name
    except TypeError:               # not weak-referenceable: say so once and go on
        return name
    return name


class MergeSession:
    """The decisions, the sources they are about, and the document they make.

    Sources are read-only here. `export()` is the only thing that builds a
    document, and both saving and the Final Preview go through it.

    A side may be an `XoppDocument`, a path, or a `sides.Side`. A PDF side
    arrives with no strokes and is **not** converted to open: see `_source`.
    """

    def __init__(self, left, right,
                 display_scale: float = DEFAULT_DISPLAY_SCALE,
                 alignment=None,
                 classify_dpi: float = DEFAULT_CLASSIFY_DPI):
        self.left, self.left_side = _source(left)
        self.right, self.right_side = _source(right)
        self.scale = display_scale
        # **The resolution the page's three classes are decided at**, in pixels
        # per inch. Independent of the zoom by construction: a view samples this
        # answer (`views.classification_of`) instead of asking the question of
        # two rasters made at the zoom of the moment, which is what used to move
        # black/red/blue from one zoom to the next. `set_classify_dpi` is the
        # only thing that may change it.
        # **One resolution for the whole base pipeline.** `classify_dpi` is the name this was built under and the
        # library and tests still speak it; `render_dpi` is what it is -- the resolution the source is turned into a
        # raster at, which the ink, the classes, the layers and the picture on screen are all derived from.
        self.render_dpi = float(classify_dpi)
        self._classification_cache: Dict[object, object] = {}
        # Which right page each left page corresponds to. Without one, page
        # i is paired with page i, which is only correct when both documents
        # have the same pages in the same order (see pagealign.py).
        self.alignment = alignment

        self._decisions: Dict[int, PageDecisions] = {}
        self._next_key = 0
        # **A source page keeps its key even while it is not navigable.**
        # Decisions are filed under a page's key, and since only real
        # pairs become pages, unpairing a page takes it out of `_pages`
        # -- so a key remembered only from there would be forgotten the
        # moment the user unpaired, and a fresh one handed out when the user paired
        # it again. Everything the user had decided on that page would then be
        # filed under a key nothing asks for. This remembers it.
        self._keys_by_source = {}
        self._pages: List[MergedPage] = []
        # Which two page keys were once one page, so a split can be undone
        # and so each half knows it is a half.
        self._split_pairs: Dict[int, int] = {}
        self._build_pages()
        self._stroke_cache: Dict[int, tuple] = {}
        self._flat_cache: Dict[int, tuple] = {}
        # The page's three ink layers and the composition state of every
        # piece: see `layers.py`. Built when a page is first brushed or
        # drawn, and then left alone -- a gesture changes pieces, not this.
        self._layer_cache: Dict[object, tuple] = {}
        # **The layer books are user state, not a cache.** They hold the cuts
        # and the per-piece decisions -- what the user brushed -- and clearing them
        # on a Fit or a re-pair threw those away: the pieces were rebuilt from
        # the stroke-level decisions, and a stroke the user had erased the middle of
        # came back whole. Keyed by the page's key, which survives reordering.
        self._layer_state: Dict[object, object] = {}
        # **One book per side's page, not per merged page.** Which right page a
        # left page faces is a question the user answers with the map; the cuts on
        # that left page are not about the pairing at all. Keyed by the side,
        # the document it came from and the page's own index, so re-pairing,
        # moving and splitting keep them.
        self._side_books: Dict[tuple, object] = {}
        self._geometry_cache: Dict[int, 'PageGeometry'] = {}
        self._raw_cache: Dict[int, tuple] = {}
        # Strokes made from what the brush took off a picture side, by page.
        # Empty until `convert_areas` is asked for: opening, looking and
        # brushing all leave it so.
        self._converted: Dict[int, list] = {}
        self._picture_frames: Dict[tuple, object] = {}
        # One frame per *file*, measured once from the pages that carry the
        # template's border, and the spread between those pages. The geometry
        # is a property of the document, not of the page: a PDF whose page 17
        # has no border is still the same PDF.
        self._file_frames: Dict[str, tuple] = {}
        # Where the template's frame sits on each merged page, in page units.
        # A page is asked for it on every redraw (the dashed guide and the
        # band the picture's printed border is painted out over), and finding
        # it means rendering a picture page, so the answer is kept.
        self._frame_boxes: Dict[int, tuple] = {}
        self._profile = None
        # Fits set by hand, by (page, side); ALL_PAGES applies to every page.
        # A measured fit is a guess about a page, and some pages need telling.
        self._fit_overrides: Dict[tuple, Fit] = {}
        # Per-page undo stacks (plain lists -- unbounded by design, so undo
        # depth is limited only by memory). Each entry is one gesture's worth
        # of decision changes, grouped by begin_action()/end_action(), so
        # Ctrl+Z undoes "that stroke", not one motion event's worth of it.
        self._undo_stacks: Dict[int, List[list]] = {}
        self._redo_stacks: Dict[int, List[list]] = {}
        self._current_action: Optional[list] = None

    # ---------------- pages ----------------

    def _build_pages(self, keep_decisions: bool = False) -> None:
        """The output page list, from the pairing.

        Rebuilt when the pairing changes. Decisions are filed under a page's
        key, and a page that still pairs the same two source pages keeps its
        key -- so re-pairing page 30 leaves everything decided on pages 1-29
        exactly where it was.
        """
        # Keyed by which source page it draws from, not by the pair: decisions
        # are about strokes, and a left page re-paired with a different right
        # page is still the same left page with the same ink decided about.
        previous = {}
        if keep_decisions:
            for page in self._pages:
                previous[(LEFT, page.left) if page.left is not None else (RIGHT, page.right)] = page.key
        pages: List[MergedPage] = []
        if self.alignment is None:
            # **Every left page is a page**, even when the right side runs out:
            # left 1 with right 1, left 2 with right 2, and a left page with no
            # partner beyond that. This used to stop at `min(len(left),
            # len(right))`, so three left pages against one right one produced a
            # single-page merge and the other two documents' pages vanished --
            # not the "in order" pairing the user asked for, and not what an explicit
            # alignment already did.
            pairs = [(index, index if index < len(self.right.pages) else None)
                     for index in range(len(self.left.pages))]
        else:
            pairs = [(i, self.alignment.right_for_left(i)) for i in range(len(self.left.pages))]
        for left_index, right_index in pairs:
            if right_index is not None and right_index >= len(self.right.pages):
                right_index = None
            # **Only a real pair is a page here**, 2026-10-30: *"the Merger
            # main view must display only valid Left-Right page pairs ...
            # unpaired pages remain in Page Mapping but are excluded from
            # Merger page navigation."* A page with no partner has nothing to
            # compare and nothing to merge, so it takes no place in the
            # navigation; `L1<->None, L2<->None, L3<->R1` is one merge page,
            # not three.
            #
            # Nothing is lost by it: `PageMapApp` is given the two documents,
            # not this list, so every source page is still there to be
            # re-paired. And the key is still `(LEFT, left_index)`, so a
            # decision stays with the source page it was made on however the
            # pairing moves -- which is what keeps a brush stroke from
            # landing on someone else's page after a re-map.
            #
            # This reverses the rule written here before ("every left page is
            # a page, even when the right side runs out"), which came from
            # the user reporting the opposite: three left pages against one right
            # gave a one-page merge and the rest seemed to vanish. They did
            # not vanish then and do not now -- they are in the mapping -- and
            # the user has since asked for the navigation to hold pairs only.
            if right_index is None or left_index is None:
                continue
            key = previous.get((LEFT, left_index))
            if key is None:
                key = self._keys_by_source.get((LEFT, left_index))
            if key is None:
                key, self._next_key = self._next_key, self._next_key + 1
            self._keys_by_source[(LEFT, left_index)] = key
            pages.append(MergedPage(key, left_index, right_index))
        self._pages = pages
        # **A re-pairing ends any split it touches; it never makes one.** This
        # block used to re-apply every split it found, on whatever page now
        # carried that left index -- and because `split_page` re-points the
        # halves at the *new* partner, mapping left 1 to right 2 left three
        # pages behind: a left-only half, a right-only half still pointing at
        # the old partner, and the pair it was asked for. That is the "page
        # mapping is still affected by two-page splitting" the user reported: the
        # mapping was applied and then immediately taken apart.
        #
        # A split is a statement about one pair ("this sheet is two pages").
        # Re-pairing one of its two pages makes that statement about a sheet
        # that no longer exists, so the page comes back whole. Splitting is a
        # user action (`Ctrl+4`), and only a user action.

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def page_at(self, page_index: int) -> MergedPage:
        return self._pages[max(0, min(page_index, len(self._pages) - 1))]

    def page_origin(self, page_index: int) -> str:
        """Whether this output page has both sides, or is one of a split."""
        return self.page_at(page_index).origin

    def set_alignment(self, alignment) -> None:
        """Re-pair the pages part-way through a merge.

        Decisions already made stay made: they are about strokes, not about
        the pairing. What changes is which right page each left page is shown
        against -- so the page list is rebuilt, keeping the key of every page
        that still pairs the same two, and everything cached about where the
        ink sits has to go.
        """
        self.alignment = alignment
        self._build_pages(keep_decisions=True)
        self._stroke_cache.clear()
        self._flat_cache.clear()
        self._layer_cache.clear()
        self._geometry_cache.clear()
        self._raw_cache.clear()
        self._frame_boxes.clear()

    def right_index_for(self, page_index: int) -> Optional[int]:
        """The right-hand page this output page draws from, if any."""
        return self.page_at(page_index).right

    def left_index_for(self, page_index: int) -> Optional[int]:
        return self.page_at(page_index).left

    def _page_pair(self, page_index: int):
        page = self.page_at(page_index)
        left_pages, right_pages = self.left.pages, self.right.pages
        left_page = (left_pages[page.left]
                     if page.left is not None and page.left < len(left_pages) else None)
        right_page = (right_pages[page.right]
                      if page.right is not None and page.right < len(right_pages) else None)
        return left_page, right_page

    def geometry(self, page_index: int) -> 'PageGeometry':
        """What size this merged page is, and how each side gets into it.

        The target size comes from the page boxes (pagesize.py); how the
        other side's ink is brought into it comes from the ink itself
        (inkfit.py), because a page box is only a claim. A PDF from
        elsewhere can be any size in any shape with its writing anywhere on
        it, and scaling by the ratio of the two boxes then lines up nothing.
        """
        if page_index not in self._geometry_cache:
            self._geometry_cache[page_index] = self._fit_page(page_index)
        return self._geometry_cache[page_index]

    def _fit_page(self, page_index: int) -> 'PageGeometry':
        """Work out this page's geometry, by the ink wherever there is ink.

        Two pages the same size are not two pages that line up: a notebook
        page and a PDF traced from a scan of it can both be the notebook
        format and still carry the writing at different sizes, because the
        scan was cropped or enlarged before it ever became a PDF. So the ink
        decides, and the page boxes are only the starting guess -- skipping
        the fit whenever the boxes matched is what left one side's writing
        visibly bigger than the other's.
        """
        left_page, right_page = self._page_pair(page_index)
        left_size = None if left_page is None else (left_page.width, left_page.height)
        right_size = None if right_page is None else (right_page.width, right_page.height)
        target = choose_target(left_size, right_size)
        fits = {LEFT: Fit(), RIGHT: Fit()}
        if left_page is None or right_page is None:
            return self._with_overrides(page_index,
                PageGeometry(target.size, fits, target.aspects_agree, target.reason))

        left_raw, right_raw = self._raw_strokes(page_index)

        # The frame first, wherever both sides have one: it is drawn on the
        # page by the template, so it says where the writing goes -- which a
        # page box does not, and which the ink can only be asked to guess.
        framed = self._frame_fits(page_index, target, left_raw, right_raw)
        if framed is not None:
            return self._with_overrides(page_index, framed)

        if target.base == 'neither':
            # Neither page is Q base, so the left one is brought onto it by
            # its page box first -- it is the notebook being merged into --
            # and the right one is then fitted to where that ink landed.
            fits[LEFT] = _onto(left_size, target.size)
            left_raw = [_placed_stroke(stroke, fits[LEFT]) for stroke in left_raw]
            moving = RIGHT
        else:
            moving = RIGHT if target.base == LEFT else LEFT
        moving_size = right_size if moving == RIGHT else left_size
        same_box = (abs(moving_size[0] - target.size[0]) <= 0.01
                    and abs(moving_size[1] - target.size[1]) <= 0.01)
        # What the page boxes alone would say: the ratio that fits the whole
        # page into the target without cropping it.
        box_scale = 1.0 if same_box else min(target.size[0] / moving_size[0],
                                             target.size[1] / moving_size[1])
        source = right_raw if moving == RIGHT else left_raw
        against = left_raw if moving == RIGHT else right_raw
        fitted = fit_ink(source, against, target.size, initial=box_scale)
        if fitted.score < MIN_FIT_SCORE:
            # Nothing in common to fit to. Same-sized pages are left alone;
            # different ones fall back to their boxes, centred, so the page
            # at least lands where it belongs.
            fitted = (Fit(reason='same page size, and the inks did not match') if same_box
                      else Fit(box_scale,
                               (target.size[0] - moving_size[0] * box_scale) / 2,
                               (target.size[1] - moving_size[1] * box_scale) / 2,
                               fitted.score, 'page boxes (the inks did not match)'))
        fits[moving] = fitted
        reason = '%s; %s side: %s' % (target.reason, moving, fitted.reason)
        return self._with_overrides(page_index,
            PageGeometry(target.size, fits, target.aspects_agree, reason))

    FRAME_SCAN_SCALE = 2.0      # enough for a 2-3pt line; measured stable 1x-4x

    def _frame_in_picture(self, page_index: int, side: str):
        """The template's frame, found in a picture side's rendered page.

        **A PDF side has no strokes, by design** -- `Side.strokes` returns
        None for one, because turning its ink into strokes is exactly what
        this merge postpones. `frame_for_page` looks in a page's strokes, then
        at a `.xopp`'s pdf background element, then assumes the template if
        the page box is Q base. A PDF side offers none of the three: no
        strokes, no background element, and a page box that is not Q base.

        So it answered None, `_frame_fits` needs both sides, and the frame
        path was skipped for **every merge with a PDF in it**. Those fell
        through to fitting the ink, and when that did not score, to scaling
        page boxes and centring -- a guess that ignores the frame drawn on the
        page. Measured on the user's TEST pair: the box guess put the PDF at scale
        0.1834 and 70.4pt down the page; the frame puts it at 0.2854 and
        0.4pt. That is the offset between the red and the blue borders.

        The frame is right there in the picture. This looks for it.

        Cached per page and side: `geometry()` is asked on every redraw, and
        rendering a page to measure it each time would be paid at pointer
        speed.
        """
        key = (page_index, side)
        if key in self._picture_frames:
            return self._picture_frames[key]
        found = None
        try:
            from xp.core.frame import Frame, frame_in_image

            held = self.left_side if side == LEFT else self.right_side
            source = self.page_at(page_index)
            index = source.left if side == LEFT else source.right
            if held is not None and index is not None:
                scale = self.FRAME_SCAN_SCALE
                seen = frame_in_image(held.render(index, scale))
                if seen is not None:
                    found = Frame(seen.x0 / scale, seen.y0 / scale,
                                  seen.x1 / scale, seen.y1 / scale,
                                  source='seen in the %s page' % side)
        except Exception:
            found = None            # a frame we cannot find is not an error
        self._picture_frames[key] = found
        return found

    def _frame_of(self, page_index, side, page, raw):
        """One side's frame: from its ink, from its background, or from the
        picture it is -- whichever the side actually has.

        The document and its folder go in because the background is where the
        frame usually lives, and the filename recorded in a notebook is
        relative to wherever that notebook used to be. See `background_pdf`.
        """
        document = self.left if side == LEFT else self.right
        held = self.left_side if side == LEFT else self.right_side
        beside = None
        path = getattr(held, 'path', None)
        if path is not None:
            from pathlib import Path as _Path
            beside = _Path(path).parent
        found = frame_for_page(page, (page.width, page.height), raw,
                               document=document, beside=beside)
        if found is not None and found.observed:
            return found
        if side in self.picture_sides():
            drawn = self._frame_in_picture(page_index, side)
            if drawn is not None:
                return drawn
        return found

    def _file_frame(self, side: str):
        """`(frame, pages_used, spread)` for one side of the merge, once.

        A side whose frame cannot be measured is left on the identity, which
        is the honest answer for a document that has no border anywhere. It is
        also what a **bug** in the measuring looks like from here, and this
        caught one silently for a round: `Side.file_frame` raised on its first
        call, every merge whose geometry comes from the PDF behind the page
        opened uncalibrated, and the only trace was the words "nothing to
        measure" in a reason string nobody reads. So the failure is printed
        now. It is still not raised -- a document too big to walk is not an
        error -- but it no longer passes for an answer.
        """
        if side in self._file_frames:
            return self._file_frames[side]
        held = self.left_side if side == LEFT else self.right_side
        frame, used, spread = None, 0, 0.0
        try:
            if held is not None and hasattr(held, 'file_frame'):
                frame, used, spread = held.file_frame()
            elif held is None:
                # No `Side`: this session was built from documents (the tests,
                # and anything that loads a pair directly). The same rule
                # applies to them -- measure the pages that carry a border,
                # take the centreline, one frame for the file.
                document = self.left if side == LEFT else self.right
                frame, used, spread = self._document_frame(document, side=side)
        except Exception as exc:
            frame, used, spread = None, 0, 0.0
            print('[merge] %s side: the file frame could not be measured, so '
                  'that side is left where it is - %s: %s'
                  % (side, type(exc).__name__, exc), flush=True)
        self._file_frames[side] = (frame, used, spread)
        return self._file_frames[side]

    def _document_frame(self, document, limit: int = 6, side=None):
        """A file frame from a document that was handed in directly.

        `side` is the **name** of a side, 'left' or 'right' -- the one caller
        has only that. It used to be used here as though it were a `Side`
        object, so `side.use(index)` raised `'str' object has no attribute
        'use'`, the caller swallowed it, and the merge opened saying *"the
        file frame could not be measured, so that side is left where it is"*
        on every pair with an XOPP in it. The frame was never measured and
        the side was never cleaned; nothing said so but that one line.
        """
        from xp.core.frame import (centreline_frame, file_frame, frame_in_strokes)

        held = None
        if side is not None:
            held = self.left_side if side == LEFT else self.right_side
            if not hasattr(held, 'use'):
                held = None

        frames = []
        for index, page in enumerate(list(document.pages)[:limit]):
            # **A border still in the strokes is a border this would measure as the frame.** Cleaning is per
            # page now, so a page nobody has looked at has to be asked for here first.
            if held is not None:
                held.use(index)
            strokes = page.strokes()
            outline = frame_in_strokes(strokes, (page.width, page.height))
            if outline is None:
                continue
            frames.append(centreline_frame(strokes, outline) or outline)
        return file_frame(frames)

    def _qbase_fit_for(self, side) -> Optional['Fit']:
        """A side's initial geometry from the Q base calibration, or None.

        Only for a side that can render itself (a PDF); an XOPP is placed by its own
        frame. A non-uniform or refused calibration returns None and the caller says
        "nothing to measure", which is what the user's section 6 asks: the same component, and
        no transform invented when it refuses.
        """
        held = self.left_side if side == LEFT else self.right_side
        if held is None or not hasattr(held, 'render'):
            return None
        try:
            from xp.core import qbasefit

            rendered = held.render(0, 1.0)          # 1 px per point
            array = np.asarray(rendered)
            if array.ndim == 3:
                array = array[:, :, :3].min(axis=2)
            policy = qbasefit.transform_for(qbasefit.fit_qbase(
                array, page=(held.page_width, held.page_height)
                if hasattr(held, 'page_width') else qbasefit.Q_BASE))
        except Exception:
            return None
        if not policy.applies or abs(policy.sx - policy.sy) > 0.02:
            return None
        return Fit(policy.sx, -policy.dx * policy.sx, -policy.dy * policy.sy,
                   1.0, 'calibrated from the printed border (qbasefit)',
                   sx=policy.sx, sy=policy.sy)

    def _frame_fits(self, page_index, target, left_raw, right_raw):
        """Both sides placed by their **files'** frames, or None if neither has one.

        This used to be measured page by page, and a page with no border fell
        back to guessing. The user's requirement: a PDF that has a border on a few
        pages is still one document with one geometry -- so the frame is
        measured once per file (the median of the pages that have one) and that
        single transform is applied to every page, with or without a border.

        Two frames that really disagree about the *aspect* are stretched per
        axis (`align_frame` says so in `reason`); under the slack they are
        treated as one, because a traced frame is never exactly square.
        """
        from xp.core.document.pagesize import is_qbase

        left_frame, left_used, left_spread = self._file_frame(LEFT)
        right_frame, right_used, right_spread = self._file_frame(RIGHT)
        # **The shared calibration counts as "measured", so this must not bail out
        # first.** The early return here skipped the `qbasefit` branch entirely whenever
        # neither file had a blue frame -- which is exactly the case the branch exists
        # for (measured: a black-bordered PDF came out with "the inks did not match").
        calibrated = {}
        for side, frame in ((LEFT, left_frame), (RIGHT, right_frame)):
            calibrated[side] = None if frame is not None else self._qbase_fit_for(side)
        if left_frame is None and right_frame is None \
                and not any(calibrated.values()):
            return None
        reference = (REFERENCE_FRAME if is_qbase(*target.size)
                     else (left_frame or right_frame))
        fits = {}
        notes = []
        for side, frame, used, spread in ((LEFT, left_frame, left_used, left_spread),
                                          (RIGHT, right_frame, right_used, right_spread)):
            if frame is None:
                # **The shared Q base calibration, when the file's own frame is
                # missing.** Same component as the converter (`xp/core/qbasefit.py`),
                # same three steps, applied to the side's page rendered at one pixel
                # per point. The transform it returns is in "reference = (page -
                # offset) * scale", which converts to this Fit as measured on a known
                # frame: `page' = page * scale + d`.
                measured = calibrated.get(side)
                if measured is not None:
                    fits[side] = measured
                    notes.append('%s: %s' % (side, measured.reason))
                    continue
                fits[side] = Fit()
                notes.append('%s: nothing to measure, left as it is' % side)
                continue
            fits[side] = align_frame(frame, reference)
            notes.append('%s: %s' % (side, frame.source))
            if spread > 1.0:
                # Said, not averaged away: a file whose pages really were laid
                # out differently cannot be calibrated by one transform, and
                # pretending otherwise would put the writing in the wrong place
                # on every page but the middle one.
                notes.append('%s pages differ by %.1f pt, one transform cannot '
                             'fit them all' % (side, spread))
        return PageGeometry(target.size, fits, target.aspects_agree,
                            'file calibration (%s)' % '; '.join(notes))

    def frame_box(self, page_index: int) -> Tuple[float, float, float, float]:
        """Where the file's frame lands on this merged page, in page units.

        The picture side's printed border is painted out with the paper colour
        over this rectangle, so the blue never reaches the screen. It is the
        *file's* frame -- the same calibration the fit uses -- not a page's.
        """
        key = self.page_at(page_index).key
        held = self._frame_boxes.get(key)
        if held is not None:
            return held
        geometry = self.geometry(page_index)
        for side in (LEFT, RIGHT):
            frame, _used, _spread = self._file_frame(side)
            fit = geometry.fits.get(side)
            if frame is None or fit is None:
                continue
            x0, y0 = fit.apply(frame.x0, frame.y0)
            x1, y1 = fit.apply(frame.x1, frame.y1)
            box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            page_w, page_h = geometry.size
            if box[2] - box[0] < 1.0 or box[3] - box[1] < 1.0:
                continue
            if box[2] <= 0 or box[3] <= 0 or box[0] >= page_w or box[1] >= page_h:
                continue
            self._frame_boxes[key] = box
            return box
        from xp.core.document.pagesize import is_qbase
        from xp.core.frame import REFERENCE

        page_w, page_h = geometry.size
        box = REFERENCE if is_qbase(page_w, page_h) else (0.0, 0.0, page_w, page_h)
        self._frame_boxes[key] = box
        return box

    def _content_points(self, page_index: int, side: str, placed: bool = True):
        """One side's **actual content** as points, for the Manual Fit.

        The user's rule: "Manual Fit only uses the actual content of the two pages on
        screen -- it compares the geometry of the two images, whatever their
        colours, and completely ignores the Q base blue border."

        A side with strokes already *is* its content, and its points are exact,
        so they are used as they are. A side without strokes -- a PDF, or a
        notebook page whose ink is a picture -- has no points to hand, and the
        honest reading of "the two images" is that the picture is rendered and
        its ink sampled: the same `imagediff.ink_of` question the comparison
        asks, so the fit measures exactly the content the comparison sees.

        `placed` says whether the points come back in the merged page's
        coordinates (the other side, already positioned by its own fit) or in
        the side's own coordinates (the moving side, whose fit is the answer).

        Returns an (N, 2) float array, or None when the side has no content.
        """
        from xp.core.imaging import imagediff
        from xp.workers.merger import views
        from xp.workers.merger.matching.common import sample_along

        held = self.left_side if side == LEFT else self.right_side
        page_now = self.page_at(page_index)
        side_index = page_now.left if side == LEFT else page_now.right
        picture = views.picture_side(self, side, page_index)
        if picture is None:
            left_raw, right_raw = self._raw_strokes(page_index)
            strokes = left_raw if side == LEFT else right_raw
            if not strokes:
                return None
            fit = self.geometry(page_index).fits[side] if placed else Fit()
            blocks = []
            for stroke in strokes:
                points = np.asarray(stroke.points, dtype=np.float64)
                if not len(points):
                    continue
                if not fit.is_identity:
                    points = np.column_stack((points[:, 0] * fit.scale_x + fit.dx,
                                              points[:, 1] * fit.scale_y + fit.dy))
                # **Dense along the stroke, not just at its vertices.** The
                # other side of this fit may be sampled ink -- thousands of
                # pixels along a line -- and a two-point stroke against that is
                # two dots against a line: measured, the correct fit scored
                # 0.19 and was refused. Resampling both sides as ink is what
                # makes the two comparable, and it is the same idea as
                # `common.point_coverage`, which already resamples for exactly
                # this reason.
                blocks.append(sample_along(points, 0.4))
            if not blocks:
                return None
            answer = np.concatenate(blocks)
            if len(answer) > _FIT_MOST_POINTS:
                answer = answer[::int(np.ceil(len(answer) / _FIT_MOST_POINTS))]
            return answer
        if side_index is None:
            return None
        # ---- a picture side: render it and take the ink ----
        # **The pipeline's own resolution, not a second one.** The fit compares the picture at the DPI the page is
        # rasterised at -- and a whole page cannot be 8000 dpi, so that is the capped scale, which is the same one the
        # classification reads.
        scale = self.whole_page_scale(page_index)
        if placed:
            image = np.asarray(views._rendered(self, picture, page_index, scale)
                               .convert('RGB'), np.uint8)
        else:
            image = np.asarray(picture.render(side_index, scale), np.uint8)
            if image.ndim == 3 and image.shape[2] >= 3:
                image = np.ascontiguousarray(image[:, :, ::-1])
            # **The blue border is not content.** It is painted out of the
            # unplaced render the same way `_rendered` paints it out of the
            # placed one: on the frame's rectangle, and blue. Otherwise the
            # border would be the largest thing in the fit and would win it.
            frame, _pages, _spread = picture.file_frame()
            if frame is not None:
                paper = tuple(int(v) for v in imagediff.paper_of(image))
                views._erase_the_printed_frame(image, frame, scale,
                                               (0.0, 0.0), paper)
        paper = imagediff.paper_of(image)
        ink = np.asarray(imagediff.ink_of(image, paper)).astype(bool)
        rows, columns = np.nonzero(ink)
        if not len(rows):
            return None
        # A page of writing is a lot of ink; the fit scores on a 320-pixel grid
        # and cannot tell more than a few tens of thousands of points apart.
        most = 60000
        if len(rows) > most:
            keep = np.linspace(0, len(rows) - 1, most).astype(np.int64)
            rows, columns = rows[keep], columns[keep]
        return np.column_stack((columns / scale, rows / scale))

    def _fit_render_scale(self, page_index: int) -> float:
        """Kept for callers that ask; the fit itself no longer has a resolution of its own.

        It used to answer a size-based number between 0.5 and 2 pixels per unit. With one DPI driving the whole base
        pipeline, the fit samples at `render_scale()` like everything else -- a second resolution for the fit meant it
        measured a picture that was neither the one on screen nor the one the classes came from.
        """
        return self.render_scale()

    def refit(self, page_index: Optional[int] = None, report=None) -> bool:
        """Re-measure from **the page the user is looking at**, and use it everywhere.

        The user's instruction: "Fit should use the page currently on screen as the
        calibration sample, but the result must apply to the whole document --
        do not turn the other pages into independent calibration jobs."

        **Nothing is thrown away until there is something to put in its place.**
        This used to clear `_fit_overrides` first and return when the sample had
        nothing to measure, which left cached pages on the old geometry and
        uncached ones on the file's -- two different transforms in one
        document, with a log line saying the calibration had been kept. Now the
        candidate is computed first; only a real measurement replaces anything.

        Returns whether a new fit was stored.
        """
        from xp.core.inkfit import fit_axes

        if page_index is None:
            self._fit_overrides.clear()
            self._file_frames.clear()
            self._forget_placement(None)
            return True

        page_index = max(0, min(int(page_index), self.page_count - 1))

        def say(label: str) -> None:
            """Say what is happening, for a window with a progress panel. Silent by default."""
            if report is not None:
                report(label)

        say('measuring the two pages\' actual content')
        geometry = self.geometry(page_index)
        moving = self.moving_side(page_index)
        other = LEFT if moving == RIGHT else RIGHT
        fit = None
        # **The two images decide, and the printed border is never asked.**
        # The user's order, word for word: "Manual Fit compares the geometry of the two
        # actual images; ignore the Q base blue border completely." It used to
        # read the moving side's raw *strokes*, which meant a PDF side had
        # nothing to measure at all -- a merge with a PDF in it could not be
        # fitted by hand on any page. Both sides now answer with their actual
        # content: stroke points where there are strokes, sampled ink where the
        # content is a picture (`_content_points`).
        source = self._content_points(page_index, moving, placed=False)
        against = self._content_points(page_index, other, placed=True)
        if source is not None and against is not None and len(source) >= 2 \
                and len(against) >= 2:
            # Each axis separately: a scan cropped to another aspect is not the
            # same document stretched uniformly, and one scale cannot fit it.
            say('matching the two pages\' content, ignoring the Q base border')
            fitted = fit_axes(source, against, geometry.size,
                              initial=geometry.fits[moving].scale or 1.0)
            if fitted.score >= INK_FIT_ENOUGH:
                fit = fitted
        # **The printed border is not a fallback.** The user's section 7: "even if the
        # blue border is completely wrong, distorted or missing, Manual Fit must
        # work from the actual content alone". Falling back to the frame is
        # exactly where a wrong border does the most damage -- it hands back an
        # alignment that looks measured and is not.
        if fit is None:
            # Nothing measurable on this page: say so, and leave the file's own
            # calibration and every cached page exactly as they were.
            print('[merge] fit page %d: not enough shared content to measure (the '
                  'printed border is not used for a manual fit), the file '
                  'calibration stands' % (page_index + 1), flush=True)
            return False
        say('normalising to the Q base page size')
        # One fit for the whole side: every page of this document uses it.
        say('applying the fit to every page of this side')
        self._fit_overrides.clear()
        self._fit_overrides[(ALL_PAGES, moving)] = fit
        self._file_frames.clear()
        self._forget_placement(None)
        print('[merge] fit from page %d: %s side x%.4f y%.4f at (%+.2f, %+.2f) - %s'
              % (page_index + 1, moving, fit.scale_x, fit.scale_y, fit.dx,
                 fit.dy, fit.reason), flush=True)
        return True

    def _with_overrides(self, page_index: int, geometry: 'PageGeometry') -> 'PageGeometry':
        """Whatever was set by hand wins over what was measured."""
        overrides = {side: self._fit_overrides.get((page_index, side),
                                                   self._fit_overrides.get((ALL_PAGES, side)))
                     for side in SIDES}
        if not any(overrides.values()):
            return geometry
        fits = dict(geometry.fits)
        for side, fit in overrides.items():
            if fit is not None:
                fits[side] = fit
        return PageGeometry(geometry.size, fits, geometry.aspects_agree,
                            geometry.reason + '; adjusted by hand')

    def moving_side(self, page_index: int) -> str:
        """The side whose ink is being placed onto the other's page.

        The notebook format is the target where either side has it, so this
        is normally the right side -- but not always, and the controls that
        adjust the fit have to act on whichever one is actually moving.
        """
        fits = self.geometry(page_index).fits
        if not fits[RIGHT].is_identity:
            return RIGHT
        if not fits[LEFT].is_identity:
            return LEFT
        left_page, right_page = self._page_pair(page_index)
        if left_page is not None and right_page is not None:
            target = choose_target((left_page.width, left_page.height),
                                   (right_page.width, right_page.height))
            return RIGHT if target.base == LEFT else LEFT
        return RIGHT

    def fit_for(self, page_index: int, side: str = RIGHT) -> Fit:
        """How that side's ink is currently placed on this page."""
        return self.geometry(page_index).fits[side]

    def set_fit(self, page_index: Optional[int], side: str, fit: Fit) -> None:
        """Place a side's ink by hand. `page_index` None sets every page.

        Kept apart from what the fit measured: clearing the override brings
        the measured one straight back, so a hand adjustment is never a
        one-way door.
        """
        pages = (list(range(self.page_count)) if page_index is None
                 else [page_index])
        # The fit each page had **before** this one, so the masks can be carried
        # across. Read first: after the override is stored the old fit is gone.
        before = {}
        for at in pages:
            try:
                before[at] = self.fit_for(at, side)
            except Exception:
                before[at] = None
        self._fit_overrides[(ALL_PAGES if page_index is None else page_index, side)] = fit
        for at in pages:
            was = before.get(at)
            if was is None:
                continue
            self._move_the_decisions(at, side, was, fit)
        self._forget_placement(page_index)

    def adjust_fit(self, page_index: Optional[int], side: str, ratio: float = 1.0,
                   dx: float = 0.0, dy: float = 0.0) -> Fit:
        """Nudge the current fit: multiply the ratio, shift by dx/dy.

        The ratio is taken about the middle of the page, so turning it up
        does not also walk the ink off the edge.
        """
        current = self.fit_for(0 if page_index is None else page_index, side)
        width, height = self.page_size(0 if page_index is None else page_index)
        centre = (width / 2, height / 2)
        scale = current.scale * ratio
        # Keep the page's centre where it is while the ratio changes.
        offset_x = centre[0] - (centre[0] - current.dx) * ratio + dx
        offset_y = centre[1] - (centre[1] - current.dy) * ratio + dy
        adjusted = Fit(scale, offset_x, offset_y, current.score, 'set by hand')
        self.set_fit(page_index, side, adjusted)
        return adjusted

    def clear_fit(self, page_index: Optional[int] = None, side: Optional[str] = None) -> None:
        """Drop hand adjustments and go back to what the ink measured."""
        for key in list(self._fit_overrides):
            page, stored_side = key
            if (page_index is None or page in (page_index, ALL_PAGES)) and side in (None, stored_side):
                del self._fit_overrides[key]
        self._forget_placement(page_index)

    def _move_the_decisions(self, page_index: int, side: str, was, now) -> int:
        """Carry this side's drawn decisions across a fit change.

        Returns how many masks moved, which is what the test counts. The masks in
        `taken`/`cleared` are the picture side's whole decision record, so leaving
        them on the screen position is the same defect as losing them; the
        difference is only that the ink moved and they did not.
        """
        if was is None or now is None:
            return 0
        was_x, now_x = was.sx or was.scale, now.sx or now.scale
        was_y, now_y = was.sy or was.scale, now.sy or now.scale
        if (abs(was_x - now_x) < 1e-12 and abs(was_y - now_y) < 1e-12
                and abs(was.dx - now.dx) < 1e-12
                and abs(was.dy - now.dy) < 1e-12):
            return 0
        page = self.decisions(page_index)
        moved = 0
        for which in ('taken', 'cleared'):
            areas = page._areas(which)
            held = areas.get(side)
            if held is None:
                continue
            areas[side] = _resample_mask(held, was, now, self.scale)
            moved += 1
        if moved:
            page.stamp += 1
        return moved

    def _forget_placement(self, page_index: Optional[int]) -> None:
        """Everything derived from where the ink sits has to be built again."""
        if page_index is None:
            # The page list itself changed: everything keyed by page index is
            # about a page that may no longer be the same page.
            self._geometry_cache.clear()
            self._stroke_cache.clear()
            self._flat_cache.clear()
            # `_layer_cache` only holds placed copies; the books themselves
            # live in `_layer_state` and are what the decisions are made of.
            self._layer_cache.clear()
            self._raw_cache.clear()
            # A split gives the right half a new key and leaves the left half
            # with the old one, so a frame box cached for that key is now about
            # a page that has only one side: the guide would sit where the
            # two-sided page put it.
            self._frame_boxes.clear()
            self._picture_frames.clear()
            return
        key = self.page_at(page_index).key
        for cache in (self._geometry_cache, self._stroke_cache,
                      self._flat_cache, self._raw_cache):
            cache.pop(page_index, None)
        self._layer_cache.pop(key, None)

    def _raw_strokes(self, page_index: int) -> tuple:
        """Both sides' strokes in their own coordinates, before any fitting.

        **The printed Q base frame is already gone.** It is removed from the
        notebook when the side is opened (`sides.Side.drop_the_print_frame`),
        not here: decisions are filed by stroke index, so a filter applied after
        the load would shift those indices against the XML the export walks, and
        a decision would land on the wrong stroke.
        """
        if page_index not in self._raw_cache:
            left_page, right_page = self._page_pair(page_index)
            # **The source page, not the merge page.** `page_index` counts the merged pages; the sides are
            # indexed by the page each one contributes, and a merged page can carry source page 7 of the right
            # file. Cleaning the merge number here was the same mistake as the fit that measured the wrong page.
            page = self.page_at(page_index)
            if self.left_side is not None and page.left is not None:
                self.left_side.use(page.left)
            if self.right_side is not None and page.right is not None:
                self.right_side.use(page.right)
            self._raw_cache[page_index] = (left_page.strokes() if left_page else [],
                                           right_page.strokes() if right_page else [])
        return self._raw_cache[page_index]

    def unhandled(self, page_index: Optional[int] = None) -> list:
        """Pages that hold something the merge does not handle.

        The merge reads, compares and brushes the **first layer's strokes**.
        Text, images and higher layers on the left are carried through **and
        moved by the same fit as the ink**, so they cannot drift away from it --
        but they are not compared with the other side and not brushable, and a
        right side's copy is not brought over at all. This is what the window
        says out loud, so nobody discovers it in the exported file.
        """
        targets = (range(self.page_count) if page_index is None
                   else [max(0, min(page_index, self.page_count - 1))])
        found = []
        for index in targets:
            page = self.page_at(index)
            for side, side_index, held in (
                    (LEFT, page.left, self.left_side),
                    (RIGHT, page.right, self.right_side)):
                if held is None or side_index is None:
                    continue
                counts = held.beyond_the_first_layer(side_index)
                if counts:
                    found.append((index, side, counts))
        return found

    def page_size(self, page_index: int) -> Tuple[float, float]:
        return self.geometry(page_index).size

    @property
    def classify_dpi(self) -> float:
        """The old name for `render_dpi`, kept because callers and tests say it."""
        return self.render_dpi

    @classify_dpi.setter
    def classify_dpi(self, value) -> None:
        self.render_dpi = float(value)

    def render_scale(self) -> float:
        """Pixels per page unit the **source** is rasterised at. The one resolution in the pipeline."""
        from xp.workers.merger.classification import pixels_per_unit

        return pixels_per_unit(self.render_dpi)

    def window_scale(self, units_w: float, units_h: float) -> float:
        """The render scale for a **window** of `units_w` x `units_h` page units.

        The DPI, unless that rectangle would need more pixels than `MAX_WINDOW_PIXELS`. The line is printed once per
        scale, because a caller in a loop must not fill the user's console.
        """
        # **A ceiling, not the number itself.** Returning the DPI here would silently undo "the window follows the
        # zoom": `min(effective, this)` with this equal to the DPI caps a sharper zoom back down. What this answers is
        # the largest scale the budget allows for that rectangle, which is at or above the DPI when there is room.
        wanted = self.render_scale()
        area = max(1e-6, float(units_w) * float(units_h))
        if area * wanted * wanted <= MAX_WINDOW_PIXELS:
            return (MAX_WINDOW_PIXELS / area) ** 0.5
        allowed = (MAX_WINDOW_PIXELS / area) ** 0.5
        key = ('window', round(allowed, 3))
        if key not in _capped_whole_page:
            _capped_whole_page.add(key)
            print('[merge] a window that large at %g dpi would be %.1f megapixels, so it is rendered at '
                  '%.2f pixels per unit (%.0f dpi)' % (self.render_dpi, area * wanted * wanted / 1e6,
                                                       allowed, allowed * 72.0), flush=True)
        return max(1.0, allowed)

    def whole_page_scale(self, page_index: int = 0) -> float:
        """The render scale for a **whole page**, reduced when the DPI would need more pixels than there are.

        The user's number, 8000 dpi, is a real request and is honoured for a window: a window is the size of the screen. A
        whole page at that resolution is 2.7 GB of RGB, so the classification, the fit's material and the paper are
        capped here and the reason is printed once rather than silently obeyed into a swap storm.
        """
        wanted = self.render_scale()
        try:
            width, height = self.page_size(page_index)
        except Exception:
            return wanted
        pixels = (max(1.0, width) * wanted) * (max(1.0, height) * wanted)
        if pixels <= MAX_WHOLE_PAGE_PIXELS:
            return wanted
        allowed = (MAX_WHOLE_PAGE_PIXELS / (max(1.0, width) * max(1.0, height))) ** 0.5
        key = (round(allowed, 3), self.page_count)
        if key not in _capped_whole_page:
            _capped_whole_page.add(key)
            print('[merge] %g dpi on a whole page would be %.1f megapixels, so the whole-page work runs at '
                  '%.2f pixels per unit (%.0f dpi); a window is still drawn at the dpi you set'
                  % (self.render_dpi, pixels / 1e6, allowed, allowed * 72.0), flush=True)
        return max(0.5, allowed)

    def classify_scale(self) -> float:
        """Pixels per page unit the classification is decided at -- the same number, by construction."""
        return self.render_scale()

    def set_render_dpi(self, dpi: float, report=None) -> int:
        """Change the resolution the **source is rasterised at**, and drop everything derived from it.

        The user's instruction: "a DPI change must re-render the source at the new resolution, extract the ink again,
        classify again, rebuild the layers and refresh the picture -- not just recompute the classification plane."
        So the kept page renders, the paper colours, the classification planes and the placed layers all go; nothing
        is rebuilt here, because the page on screen is rebuilt when it is drawn and the others when they are asked
        for. An invariant of the user's: **the decisions, the stored fit, the pairing and the zoom are not touched.**

        Returns how many cached things were dropped.
        """
        from xp.workers.merger.classification import pixels_per_unit

        wanted = max(MIN_CLASSIFY_DPI, min(MAX_CLASSIFY_DPI, float(dpi)))
        if wanted == self.render_dpi:
            return 0

        def say(label: str) -> None:
            if report is not None:
                report(label)

        say('discarding what was rendered at %g dpi' % self.render_dpi)
        dropped = 0
        for name in ('_rendered_pages', '_papers', '_classification_cache', '_layer_cache',
                     '_picture_frames'):
            held = getattr(self, name, None)
            if held:
                dropped += len(held)
                held.clear()
        say('rendering the source at %g dpi' % wanted)
        self.render_dpi = wanted
        say('the classes will be decided at %.2f pixels per unit' % pixels_per_unit(wanted))
        return dropped

    def set_classify_dpi(self, dpi: float, report=None) -> int:
        """The old name for `set_render_dpi`, kept because callers and tests say it."""
        return self.set_render_dpi(dpi, report=report)

    def decisions(self, page_index: int) -> PageDecisions:
        return self._decisions.setdefault(self.page_at(page_index).key,
                                          PageDecisions(kept={}))

    def regions(self, page_index: int):
        """This page's **region geometry**: the four-state map, in page units.

        Seeded from the initial comparison -- the classification is a measurement of two rasters, so
        it is a plane and it is the map's base -- and then edited by the brush, which adds polygons
        on top. `state_at_page` asks the polygons first and the base second, so the base is never
        rewritten by a decision.

        Held per page and dropped with the same epoch the classification uses, so re-pairing or a
        re-fit rebuilds it rather than leaving a map for a page that no longer exists.
        """
        from xp.workers.merger import regions as _regions

        held = getattr(self, '_region_maps', None)
        if held is None:
            held = self._region_maps = {}
        width, height = self.page_size(page_index)
        # **Keyed by the page and the scale, and by nothing that moves when the user brushes.** The first
        # version keyed it on `_layer_epoch` and `id(strokes)`, which change *during* a gesture -- so
        # the map was rebuilt from the comparison the moment the next redraw asked for it, and the
        # polygon the gesture had just written was gone. `reset_page` drops it explicitly instead.
        epoch = (self.page_at(page_index).key, round(float(self.scale), 6))
        keep = held.get(page_index)
        if keep is not None and keep[0] == epoch:
            return keep[1]
        from xp.workers.merger import views as _views
        from xp.workers.merger import classification as _classify

        page_px = _views.page_pixels(self, page_index, self.scale)
        base = _views.classification_of(self, page_index).in_view(page_px, self.scale)
        # **Translated, not renamed.** The classification answers PAPER=0, COMMON=1, LEFT=2, RIGHT=3
        # and this map answers COMMON=0, LEFT=1, RIGHT=2, NEITHER=3: two different numbering schemes
        # over the same four ideas, and the numbers collide (its LEFT is 2, this map's RIGHT is 2).
        # Passing one through as the other would have made every left-only stroke answer "right".
        # PAPER is Common here: the map has four states and no fifth for blank -- where neither side
        # has ink the page is paper, and "both sides' content" already says that.
        plane = np.zeros(base.shape, np.uint8)
        plane[base == _classify.LEFT] = _regions.LEFT
        # **Right-only is RIGHT.** The user's four states mean what they say -- left alone is LEFT, right
        # alone is RIGHT, shared is COMMON -- and the earlier seed, which turned the right's own ink
        # into NEITHER because the old merger "takes nothing from the right by default", was the old
        # rule wearing the new vocabulary. That is the rule the user told me not to keep: a difference the
        # right has is a right-only region, and it takes a decision to remove it, not a default.
        plane[base == _classify.RIGHT] = _regions.RIGHT

        classification = _views.classification_of(self, page_index)

        def sample(shape, scale, window):
            """The comparison at *this view's* scale, in the map's numbering.

            Not a resize of `plane`: a zoom has to re-ask the classification, or every view above the
            measurement's scale is an upscale of it -- which is what made the render's own "truth"
            come back as a stretch.
            """
            answer = classification.in_view(shape, scale, window)
            out = np.zeros(answer.shape, np.uint8)
            out[answer == _classify.LEFT] = _regions.LEFT
            out[answer == _classify.RIGHT] = _regions.RIGHT
            return out

        held[page_index] = (epoch, _regions.RegionMap(width, height, base=plane,
                                                      scale=float(self.scale),
                                                      base_sampler=sample))
        return held[page_index][1]


    # ---------------- strokes ----------------

    def strokes_for_page(self, page_index: int) -> tuple:
        """(left_strokes, right_strokes) in the merged page's coordinates.

        Parsing a stroke means float()-ing its whole coordinate text, and a
        dense page has thousands of them, so this is cached: redoing it on
        every mouse-motion event of a brush drag was a real source of the GUI
        feeling frozen. Where the two pages differ in size, the side being
        scaled is scaled here, once, so everything downstream -- the brush,
        the diff, the view -- works in one coordinate system.
        """
        if page_index not in self._stroke_cache:
            fits = self.geometry(page_index).fits
            left_raw, right_raw = self._raw_strokes(page_index)
            self._stroke_cache[page_index] = (self._placed(left_raw, fits[LEFT]),
                                              self._placed(right_raw, fits[RIGHT]))
        return self._stroke_cache[page_index]

    @staticmethod
    def _placed(strokes: List[Stroke], fit: Fit) -> List[Stroke]:
        """The same strokes, in the merged page's coordinates."""
        if fit.is_identity:
            return strokes
        return [Stroke(element=stroke.element,
                       points=[fit.apply(x, y) for x, y in stroke.points],
                       color=stroke.color,
                       widths=[w * fit.pen_scale for w in stroke.widths],
                       fill=stroke.fill)
                for stroke in strokes]

    def stroke_for(self, page_index: int, ref: Ref) -> Optional[Stroke]:
        left_strokes, right_strokes = self.strokes_for_page(page_index)
        strokes = left_strokes if ref.side == LEFT else right_strokes
        return strokes[ref.index] if 0 <= ref.index < len(strokes) else None

    def refs_for_page(self, page_index: int) -> List[Ref]:
        left_strokes, right_strokes = self.strokes_for_page(page_index)
        return ([Ref(LEFT, i) for i in range(len(left_strokes))]
                + [Ref(RIGHT, j) for j in range(len(right_strokes))])

    @staticmethod
    def _flatten(strokes: List[Stroke], scale: float) -> tuple:
        """Every stroke's points as one (N,2) pixel array plus per-stroke
        start offsets, which is what BrushMask.fully_covers_batch wants."""
        starts = np.zeros(len(strokes), dtype=np.int64)
        total = 0
        for i, stroke in enumerate(strokes):
            starts[i] = total
            total += max(1, len(stroke.points))
        flat = np.zeros((max(total, 1), 2), dtype=np.float64)
        position = 0
        for stroke in strokes:
            points = stroke.points or [(0.0, 0.0)]
            block = np.asarray(points, dtype=np.float64) * scale
            flat[position:position + len(block)] = block
            position += len(block)
        return flat, starts

    def _flat_for_page(self, page_index: int) -> tuple:
        if page_index not in self._flat_cache:
            left_strokes, right_strokes = self.strokes_for_page(page_index)
            self._flat_cache[page_index] = (self._flatten(left_strokes, self.scale),
                                            self._flatten(right_strokes, self.scale))
        return self._flat_cache[page_index]

    # ---------------- decisions ----------------

    def base_side(self) -> str:
        """The side an undecided stroke is kept from: the notebook.

        Normally the left one -- it is the document being merged into. But a
        **picture side has no strokes at all**, so when the left side is a PDF
        "keep the left side" keeps nothing, and the notebook on the right is
        dropped in its entirety. That is what a merge of a PDF against a
        notebook used to export: an empty document, on every page that had
        not been brushed.

        So the base is the side that can actually contribute strokes. With two
        notebooks it is the left one, as before; with two pictures there is
        nothing to keep by default either way.

        **A notebook with an embedded picture on it is still a notebook.** It
        appears in `picture_sides()` now -- its page is compared as an image --
        but it has strokes, so "the left side is a picture" no longer implies
        "the left side has nothing to keep". The question here is only whether
        the left side has a document to keep at all, so it asks that instead:
        no left document, or no left strokes on this page, means the right one
        is the base.
        """
        from xp.workers.merger import sides as _sides

        if self.left is None:
            return RIGHT
        if self.left_side is not None and self.left_side.kind == _sides.PDF:
            return RIGHT
        return LEFT

    def keeps(self, page_index: int, ref: Ref) -> bool:
        """Whether any of this stroke survives the merge as things stand.

        On a layered page this is a **summary** of the pieces: the layers are
        where a decision lives now, and "does this stroke survive" is a
        question about a whole stroke that may well have two answers. True
        when any piece of it is composed. Callers that need the real answer
        per piece ask `composed_pieces`.

        Undecided ink from the base side stays and the other side's does not
        -- the base is your notebook (see `base_side`). On a page that *is*
        the right side, though (one half of a split), its ink is the page:
        there is no left side for it to be an addition to.
        """
        if self.uses_layers(page_index):
            held = self.layers_for_page(page_index)[ref.side]
            pieces = held.pieces_of(ref.index)
            if len(pieces):
                # `kept`, not `shown`: this question is about the file.
                return any(bool(held.kept[at]) for at in pieces)
        decisions = self.decisions(page_index)
        if decisions.is_decided(ref):
            return decisions.kept[ref]
        if ref.side == RIGHT and self.page_at(page_index).left is None:
            return True
        return PageDecisions.default_keeps(ref, self.base_side())

    def is_decided(self, page_index: int, ref: Ref) -> bool:
        """Whether the brush has said anything about this stroke yet.

        Keeping the left side changes nothing about the page, so this is the
        only trace that decision leaves -- which is why it is recorded.
        """
        if self.uses_layers(page_index):
            from xp.workers.merger import layers as _layers

            held = self.layers_for_page(page_index)[ref.side]
            pieces = held.pieces_of(ref.index)
            if len(pieces):
                return any(int(held.state[at]) != _layers.DEFAULT
                           for at in pieces)
        return self.decisions(page_index).is_decided(ref)

    def begin_gesture(self, page_index: int) -> None:
        """The mouse went down: one snapshot for the whole drag.

        `brush_segment` used to snapshot both sides on every motion event -- on
        the user's page that is megabytes copied per event, and a fresh "before" each
        time, which is how an undo chain could leave ink decided.
        """
        if self.uses_layers(page_index):
            self.layers_for_page(page_index).begin()

    def end_gesture(self, page_index: int) -> None:
        """The mouse came up: the next drag snapshots again."""
        if self.uses_layers(page_index):
            self.layers_for_page(page_index).finish()

    def begin_action(self, page_index: Optional[int] = None) -> None:
        self._current_action = _Action()

    def end_action(self, page_index: int) -> None:
        action, self._current_action = self._current_action, None
        if action:
            key = self.page_at(page_index).key
            self._undo_stacks.setdefault(key, []).append(action)
            self._redo_stacks.pop(key, None)

    def _record(self, entries) -> None:
        if self._current_action is not None:
            self._current_action.extend(entries)

    def _index_of_key(self, key):
        """Where a page is *now*. A key outlives a page move; an index does not."""
        for index, page in enumerate(self._pages):
            if page.key == key:
                return index
        return None

    def can_undo(self, page_index: int) -> bool:
        return bool(self._undo_stacks.get(self.page_at(page_index).key))

    def can_redo(self, page_index: int) -> bool:
        return bool(self._redo_stacks.get(self.page_at(page_index).key))

    def undo(self, page_index: int) -> List[Ref]:
        """Revert the most recent gesture on this page; returns what changed."""
        key = self.page_at(page_index).key
        stack = self._undo_stacks.get(key)
        if not stack:
            return []
        action = stack.pop()
        page = self.decisions(page_index)
        undone = []
        applied = False
        for entry in reversed(action):
            if entry[0] == 'pieces':
                applied = self._move_pieces(entry[1], forward=False) or applied
                continue
            undone.extend(self._move(page, entry, forward=False))
            applied = True
        if not applied:
            # **The press is not spent.** Nothing in this record belongs to the
            # page's book any more (it was rebuilt since), so putting the action
            # on the redo stack would mean "undone" for something that never
            # happened -- and Ctrl+Z would appear to do nothing, twice.
            stack.append(action)
            return []
        # **The map goes back with the decisions.** One gesture wrote both; undoing one and not the
        # other is how a region map drifts from what the user sees.
        if getattr(action, 'region', False):
            self.regions(page_index).undo()
        self._redo_stacks.setdefault(key, []).append(action)
        return undone

    def redo(self, page_index: int) -> List[Ref]:
        key = self.page_at(page_index).key
        stack = self._redo_stacks.get(key)
        if not stack:
            return []
        action = stack.pop()
        page = self.decisions(page_index)
        redone = []
        applied = False
        for entry in action:
            if entry[0] == 'pieces':
                applied = self._move_pieces(entry[1], forward=True) or applied
                continue
            redone.extend(self._move(page, entry, forward=True))
            applied = True
        if not applied:
            stack.append(action)
            return []
        if getattr(action, 'region', False):
            self.regions(page_index).redo()
        self._undo_stacks.setdefault(key, []).append(action)
        return redone

    def _move_pieces(self, payload, forward: bool) -> None:
        """Undo or redo one gesture's worth of piece states.

        The splits are left where they are: a split is geometry, and putting
        a piece's state back does not need the stroke joined again -- two
        neighbouring pieces in the same state draw as one.
        """
        key, journal = payload
        page_index = self._index_of_key(key)
        if page_index is None:
            return False
        box = self.layers_for_page(page_index).set_states(journal, forward)
        if box is None:
            return False
        self.decisions(page_index).stamp += 1
        return True

    @staticmethod
    def _move(page: PageDecisions, entry, forward: bool) -> List[Ref]:
        """One recorded change, applied in either direction.

        Every entry holds both the value before and the value after, so undo
        and redo are the same operation with a different target -- there is
        no second code path that can disagree with the first.
        """
        kind, payload = entry
        if kind == 'kept':
            ref, before, after = payload
            page.set(ref, after if forward else before)
            return [ref]
        if kind == 'untaken':
            # The mirror: the pixels a gesture *removed* from the other record.
            side, flat, shape, which = payload
            if forward:
                page.give_back(side, flat, which)
            else:
                page.take_back(side, flat, shape, which)
            return []
        if kind in ('taken', 'cleared'):
            # Area taken from, or refused on, a side whose content is not
            # strokes. The entry holds only the pixels this gesture added --
            # a few thousand indices, not another page of bools -- which is
            # enough to go either way.
            side, flat, shape = payload
            if forward:
                page.take_back(side, flat, shape, kind)
            else:
                page.give_back(side, flat, kind)
            return []
        before, after = payload
        page.split = after if forward else before
        return []

    def _set_kept(self, page_index: int, decision: str, refs) -> List[Ref]:
        """Drive every ref to what `decision` means for its side.

        The value written depends only on the decision, never on what was
        there before, which is what makes Left, Right, Both and Neither four
        states you switch between rather than four things you pile up.
        """
        page = self.decisions(page_index)
        keeps = KEEPS[decision]
        changed, entries = [], []
        for ref in refs:
            before = page.kept.get(ref)
            after = keeps[ref.side]
            if page.set(ref, after):
                entries.append(('kept', (ref, before, after)))
                changed.append(ref)
        self._record(entries)
        return changed

    # ---------------- the brush ----------------

    def picture_sides(self) -> List[str]:
        """The sides that carry something the stroke path cannot see.

        Two kinds, and the second one used to be missing:

          * a **PDF** side -- ink with no strokes at all;
          * a notebook side with an **embedded picture on any page** -- it has
            strokes, but part of the page is an `<image>`, and the stroke path
            would show that part as a hole. The user's specification: during editing a
            picture in a XOPP is simply part of that side's image, so the page
            is compared and brushed as an image, and only the selected part is
            converted at export.
        """
        from xp.workers.merger import embedded
        from xp.workers.merger import sides as _sides

        out = []
        for name, held, document in ((LEFT, self.left_side, self.left),
                                     (RIGHT, self.right_side, self.right)):
            if held is None:
                continue
            if held.kind == _sides.PDF:
                out.append(name)
                continue
            if document is None:
                continue
            if any(embedded.images_of(page) for page in document.pages):
                out.append(name)
        return out

    def page_picture_sides(self, page_index: int) -> List[str]:
        """The sides of **this page** that have to be compared as images.

        A page-level question, because a notebook may have a picture on one
        page and none on the next: the page with the picture goes through the
        pixel path and the page without it stays on the stroke path, so the
        brush's exact pieces are not given up for pages that do not need them.
        """
        from xp.workers.merger import embedded
        from xp.workers.merger import sides as _sides

        left_page, right_page = self._page_pair(page_index)
        out = []
        for side, held, own in ((LEFT, self.left_side, left_page),
                                (RIGHT, self.right_side, right_page)):
            if held is None or own is None:
                continue
            if held.kind == _sides.PDF:
                out.append(side)
                continue
            if embedded.images_of(own):
                out.append(side)
        return out

    # ---------------- the layers, and the brush that moves them -------

    def pictures_left(self) -> int:
        """How many embedded pictures are still pictures, both sides."""
        from xp.workers.merger import embedded

        return (embedded.pictures_left(self.left)
                + (embedded.pictures_left(self.right)
                   if self.right is not None else 0))

    def convert_pictures(self, progress=None) -> int:
        """Turn every embedded picture into strokes, both sides, and refresh.

        The user's order (2026-10-26): "the images embedded in the XOPP must be
        converted to native strokes in the background ... on both sides", and the
        final export must be editable strokes rather than a PNG patch.

        The conversion is in place on the session's own copy of each document, so
        everything downstream -- the layers, the brush, the export -- sees
        strokes. Every cache derived from the pages is dropped afterwards,
        because the ink on a page has just changed.

        `progress(side, page_index, done, total)` is called per picture, so a
        caller can say which page and which side it is working on.
        """
        from xp.workers.merger import embedded

        if self.right is not None:
            sides = ((LEFT, self.left), (RIGHT, self.right))
        else:
            sides = ((LEFT, self.left),)
        converted = 0
        for side, document in sides:
            for page_index, page in enumerate(document.pages):
                def say(done, total, page_index=page_index, side=side):
                    if progress is not None:
                        progress(side, page_index, done, total)

                converted += embedded.convert_page(page, progress=say)
        if converted:
            self._stroke_cache.clear()
            self._raw_cache.clear()
            self._layer_cache.clear()
            self._geometry_cache.clear()
        return converted

    def uses_layers(self, page_index: int) -> bool:
        """Whether this page is composed from layers of pieces.

        Every stroke-only page is. A page with a picture side -- a PDF, or a
        notebook page carrying an `<image>` -- goes through the area protocol
        instead, because ink that is not strokes has no pieces to state; it
        becomes strokes at export, by the deferred conversion.
        """
        return not self.page_picture_sides(page_index)

    def layers_for_page(self, page_index: int):
        """This page's three ink layers, built once and then left alone.

        Keyed on the strokes themselves as well as the index: re-pairing the
        pages changes what is on a page without anything being decided.
        """
        import numpy as np

        from xp.workers.merger import layers as _layers
        from xp.workers.merger import views as _views

        strokes = self.strokes_for_page(page_index)
        key = self.page_at(page_index).key
        have = self._layer_cache.get(key)
        if have is not None and have[0] is strokes:
            return have[1]
        left_arrays, right_arrays = _views._page_arrays(self, page_index)
        matched_left, matched_right = _views.matched_sides(self, page_index)
        flags = []
        for arrays, matched in ((left_arrays, matched_left),
                                (right_arrays, matched_right)):
            many = len(arrays[2])
            mask = np.zeros(many, bool)
            if many and len(matched):
                hit = np.fromiter((int(i) for i in matched), np.int64,
                                  len(matched))
                mask[hit[(hit >= 0) & (hit < many)]] = True
            flags.append(mask)
        # The base is asked of the **page**, not only of the session: one
        # half of a split has no left side at all, and on that page the
        # right side's ink is the page rather than an addition to something.
        # Answering with the session's base dropped the whole half.
        base = self.base_side()
        if self.page_at(page_index).left is None:
            base = RIGHT
        page_now = self.page_at(page_index)
        books = {}
        for side, arrays, matched, page_at_side, document, held_side in (
                (LEFT, left_arrays, flags[0], page_now.left, self.left,
                 self.left_side),
                (RIGHT, right_arrays, flags[1], page_now.right, self.right,
                 self.right_side)):
            if page_at_side is None:
                books[side] = _layers.SideLayers(side, arrays, matched, base)
                continue
            # **The document's own path, not its address.** `id(document)` is
            # reused by the next document after a rebuild, and a key that
            # survives its object can hand one notebook's book to another whose
            # ink happens to have the same shape.
            ident = (side, _document_key(document, held_side), page_at_side)
            book = self._side_books.get(ident)
            if book is not None and book.reattach(arrays, matched):
                books[side] = book
                continue
            book = _layers.SideLayers(side, arrays, matched, base)
            # Start from the whole-stroke decisions this page already carries.
            for ref, keep in self.decisions(page_index).kept.items():
                if ref.side != side:
                    continue
                state = (_layers.PICK_LEFT if ref.side == LEFT
                         else _layers.PICK_RIGHT) if keep else (
                    _layers.PICK_RIGHT if ref.side == LEFT
                    else _layers.PICK_LEFT)
                for at in book.pieces_of(ref.index):
                    book.state[at] = state
                    book._recolour(at)
            self._side_books[ident] = book
            books[side] = book
        made = _layers.PageLayers.from_sides(books[LEFT], books[RIGHT], base)
        self._layer_cache[key] = (strokes, made)
        return made

    def _seed_layers(self, page_index: int, held) -> None:
        """Start the pieces from the decisions this page already carries.

        The layers are built when a page is first drawn or brushed, and they
        can be built **again** -- re-pairing the documents changes what is
        on a page, so everything derived from where the ink sits is dropped
        and made afresh. Decisions already made are not derived from that
        and must survive it; without this they did not, and re-pairing
        quietly returned the page to undecided.

        A stroke-level decision is a bool per side, and the four brush
        decisions come back out of it exactly: the side that keeps its ink
        is the side that won, and a piece of the other side is hidden there.
        Take Both reads as both sides keeping (both green) and Take Neither
        as neither (both hidden), which is what those two look like.
        """
        from xp.workers.merger import layers as _layers

        decided = self.decisions(page_index).kept
        if not decided:
            return
        for ref, keep in decided.items():
            side = held[ref.side]
            if keep:
                state = (_layers.PICK_LEFT if ref.side == LEFT
                         else _layers.PICK_RIGHT)
            else:
                state = (_layers.PICK_RIGHT if ref.side == LEFT
                         else _layers.PICK_LEFT)
            for at in side.pieces_of(ref.index):
                side.state[at] = state
                side._recolour(at)

    def brush_segment(self, page_index: int, x0: float, y0: float,
                      x1: float, y1: float, radius: float, decision: str):
        """Set the composition state where the brush is, **now**.

        One segment of the drag, applied the moment it arrives: no waiting
        for the mouse to come up, no asking whether a whole stroke has been
        covered, and nothing accumulated between gestures. The pieces the
        segment touches take `decision` and the ones it does not are left
        exactly as they were, so the last brush over a place is what that
        place shows.

        Coordinates and `radius` arrive in the page's own pixels -- the space
        the brush has always worked in -- and the box that comes back is in
        the same pixels, ready for the window to repair just that rectangle.
        Returns None when nothing changed, which is what brushing the same
        place a second time does.
        """
        if not self.uses_layers(page_index):
            return None
        held = self.layers_for_page(page_index)
        scale = self.scale
        held.begin()
        box = held.paint(x0 / scale, y0 / scale, x1 / scale, y1 / scale,
                         radius / scale, decision)
        journal = held.end()
        # **The same gesture, as an area.** One capsule along the segment, drawn the
        # way the window's own mask is drawn, so the record the export cuts against
        # is the record the brush made and not a second reading of the drag.
        from xp.workers.merger.brush import BrushMask

        page_width, page_height = self.page_size(page_index)
        capsule = BrushMask(int(round(page_width * scale)),
                            int(round(page_height * scale)),
                            radius=max(1, int(round(radius))))
        capsule.stamp(x0, y0, x1, y1)
        self._capture_area(page_index, capsule, decision)
        if journal:
            # One entry per gesture-segment, on the same undo stack as every
            # other kind of change, so one Ctrl+Z still undoes one gesture.
            # **The page's identity, not its position.** This held the index,
            # so after a page move an undo or a redo landed on whichever page
            # now stood at that position -- the user lost ink on a page the user had not
            # touched. Resolved when it is applied, not when it is recorded.
            self._record([('pieces', (self.page_at(page_index).key, journal))])
        if box is None:
            return None
        # The drawing caches key on this, so they must learn that the page
        # has changed -- and only when it really has.
        self.decisions(page_index).stamp += 1
        return tuple(float(v) * scale for v in box)

    def reset_layers(self, page_index: int) -> None:
        """Every piece back to undecided."""
        if not self.uses_layers(page_index):
            return
        self.layers_for_page(page_index).reset()
        self.decisions(page_index).stamp += 1

    def composed_pieces(self, page_index: int, side: str):
        """(stroke index, offset, count) for every piece that survives."""
        return self.layers_for_page(page_index)[side].kept_pieces()

    def brush_capture(self, page_index: int, mask: BrushMask, decision: str) -> List[Ref]:
        """Record `decision` for every stroke the brush has fully covered.

        Full coverage, not touching: you can be as sloppy as you like as long
        as you cover what you mean. One decision drives both sides at once --
        brushing Left over ink both documents have keeps the left stroke and
        drops the right one, rather than leaving it hidden behind.

        On a side that has no strokes the same gesture records the **area**
        instead, because there is nothing to write the decision onto yet.
        Nothing about the brush changes: it is the same mask, the same drag
        and the same four colours. What changes is only that on that side the
        mask is kept rather than thrown away at the end of the gesture, and
        `export` is what finally turns it into strokes -- through the
        converter, once, on the pages it actually touches.
        """
        (left_flat, left_starts), (right_flat, right_starts) = self._flat_for_page(page_index)
        left_covered = mask.fully_covers_batch(left_flat, left_starts)
        right_covered = mask.fully_covers_batch(right_flat, right_starts)
        refs = ([Ref(LEFT, int(i)) for i in np.flatnonzero(left_covered)]
                + [Ref(RIGHT, int(j)) for j in np.flatnonzero(right_covered)])
        # The layers go first, and the order is not a detail. They are
        # built the first time anything asks for them, and `_seed_layers`
        # starts them from the decisions the page already carries -- so if
        # the decision were written first, the layers would be born already
        # holding it, nothing would look like a change, and the gesture
        # would record nothing for undo to put back. Measured: one brush,
        # one undo, and the ink stayed decided.
        self._state_whole_strokes(page_index, refs, decision)
        changed = self._set_kept(page_index, decision, refs)
        self._capture_area(page_index, mask, decision)
        self._capture_region(page_index, mask, decision)
        return changed

    def capture_region(self, page_index: int, mask, decision: str) -> None:
        """Public: write this gesture's outline into the page's region map.

        Needed by the **layered** path, which applies each drag segment as it arrives and never calls
        `brush_capture` -- so on a stroke page this is the only place the gesture becomes geometry.
        `brush_capture` does it itself for the callers that go through it, and the window calls this
        one only when it did not.
        """
        self._capture_region(page_index, mask, decision)

    def _capture_region(self, page_index: int, mask, decision: str) -> None:
        """The gesture as **geometry**: the mask's own outline, in page units.

        The mask is the truth of what the user painted -- a freehand band, a loop the user closed, two separate
        patches from one drag -- and its outline is the polygon that says the same thing: `findContours`
        follows the mask exactly, so this is not an approximation of the shape, and the region map
        stops depending on the resolution the mask happened to be drawn at.

        Written here, in the session, and not in the window: every path that captures a brush goes
        through this method, and a decision that only reached the map when it came from a Tk event
        would leave the map empty for any other caller.
        """
        from xp.workers.merger import regions as _regions

        state = self._STATE_OF.get(decision)
        if state is None:
            return
        scale = float(self.scale or 1.0)
        polygons = mask.outline(scale) if hasattr(mask, 'outline') else []
        held = self.regions(page_index)
        for polygon in polygons:
            if len(polygon) >= 3:
                held.add(polygon, state)
                if isinstance(self._current_action, _Action):
                    self._current_action.region = True

    def _state_whole_strokes(self, page_index: int, refs, decision: str) -> None:
        """Carry a whole-stroke decision into the layers.

        The layers are where a decision lives now, so anything that still
        decides by whole strokes has to say so there too, or the picture and
        the file would not know what it did. That is this one gesture kind --
        the loop fill, and a page with a picture side -- and the callers that
        came before the layers existed, including the suite's own.

        The brush does not come through here: it states the pieces it
        touched, which is the whole point of it.
        """
        if not refs or not self.uses_layers(page_index):
            return
        from xp.workers.merger import layers as _layers

        held = self.layers_for_page(page_index)
        state = _layers.STATE_OF[decision]
        held.begin()
        for ref in refs:
            side = held[ref.side]
            for at in side.pieces_of(ref.index):
                if int(side.state[at]) != state:
                    side._note(at, int(side.state[at]), state)
                    side.state[at] = state
                    side._recolour(at)
        journal = held.end()
        if journal:
            self._record([('pieces', (page_index, journal))])

    def _capture_area(self, page_index: int, mask: BrushMask, decision: str) -> None:
        """The same gesture, recorded as area -- for both sides, not only the
        one with no strokes.

        Which side gives way is the whole question a decision answers, so it
        is asked of both sides and the answer is kept for both: "take right"
        over an area means the right's content is used there and the left's
        own content is replaced, and a record of the right alone cannot say
        the second half. That gap is why "take neither" left a PDF on screen.

        Recorded only when the merge has a picture side at all. A merge of two
        notebooks writes nothing here, which is the behaviour it has always
        had -- the stroke decisions are the whole record for them.
        """
        # **Every page records the area now.** The area is the record the export
        # cuts against; a merge of two notebooks used to write nothing here, and the
        # piece states were then the only record -- they are still written (the view
        # draws them), but the file is built from the areas.
        painted = mask.as_overlay_array() > 0
        if not painted.any():
            return
        page = self.decisions(page_index)
        keeps = KEEPS[decision]
        entries = []
        for side in (LEFT, RIGHT):
            which = 'taken' if keeps[side] else 'cleared'
            fresh = page.take_area(side, painted, which)
            if len(fresh):
                entries.append((which, (side, fresh, painted.shape)))
            # **The opposite record goes.** The two masks are otherwise additive,
            # and `owner()` applies the refused ones last -- so a refusal recorded
            # before a decision outlived it: measured, Neither then Both left the
            # place empty. The user's rule is the last brush over a place decides it.
            opposite = 'cleared' if which == 'taken' else 'taken'
            dropped = page.remove_area(side, painted, opposite)
            if len(dropped):
                entries.append(('untaken', (side, dropped, painted.shape,
                                            opposite)))
        self._record(entries)

    def _decide_whole_area(self, page_index: int, decision: str) -> None:
        """A page action decides the areas too, and not only the strokes.

        The brush keeps the mask it painted on a side whose content is not
        strokes; the four page buttons are the same decision with the whole
        page as the mask. Without this, "take right" on a page whose right
        side is a PDF wrote a document with the PDF's ink left out and showed
        the user a blank window, because no area had ever been recorded for it.

        **And on a stroke page too, since `6047e46`.** This used to return
        early unless a side was a picture, which was right while a stroke
        page's view was painted from its pieces. Every page is edited as two
        images now, so the view reads the areas -- and a whole-page button
        wrote `kept` flags and piece states and no area at all, so the
        decision was invisible on screen. Measured on two stroke pages: the
        brush covering everything with Take Both gave 1,226 green pixels and
        no red; the button gave 0 green and 613 red, with the piece states
        identical (`[3, 3]`, coloured black and green) in both. The pieces
        were right and nothing was reading them.
        """
        from xp.workers.merger import views
        width, height = views.page_pixels(self, page_index)
        whole = np.ones((height, width), bool)
        page = self.decisions(page_index)
        keeps = KEEPS[decision]
        entries = []
        for side in (LEFT, RIGHT):
            which = 'taken' if keeps[side] else 'cleared'
            fresh = page.take_area(side, whole, which)
            if len(fresh):
                entries.append((which, (side, fresh, whole.shape)))
            # **And drop the opposite record, as a gesture does.** The two
            # records are additive and `owner()` applies the refused pixels
            # last, so without this Take Left followed by Take Right left both
            # sides in `taken` *and* in `cleared` and the whole page read as
            # "neither" -- measured, the Final Preview came back blank white.
            # The brush has had this since `PageDecisions.remove_area`; the
            # four page buttons are the same decision with the page as the
            # mask, so they need the same clause. The last decision over a
            # place decides it.
            opposite = 'cleared' if which == 'taken' else 'taken'
            dropped = page.remove_area(side, whole, opposite)
            if len(dropped):
                entries.append(('untaken', (side, dropped, whole.shape,
                                            opposite)))
        self._record(entries)

    def apply_whole_page(self, page_index: int, decision: str) -> List[Ref]:
        # The layers before the decisions, for the reason `brush_capture`
        # gives: they are seeded from the decisions when they are first
        # built, so writing the decisions first would leave the gesture with
        # nothing recorded and undo with nothing to do.
        self._decide_whole_layers(page_index, decision)
        refs = self._set_kept(page_index, decision, self.refs_for_page(page_index))
        self._decide_whole_area(page_index, decision)
        self._apply_whole_page_region(page_index, decision)
        return refs

    #: The decision letter as a region state. Two vocabularies over the same four ideas, translated
    #: in one place so neither leaks into the other.
    _STATE_OF = {TAKE_LEFT: 1, TAKE_RIGHT: 2, TAKE_BOTH: 0, TAKE_NONE: 3}

    def _apply_whole_page_region(self, page_index: int, decision: str) -> None:
        """The same whole-page action, written into the region map as one covering polygon."""
        state = self._STATE_OF.get(decision)
        if state is None:
            return
        width, height = self.page_size(page_index)
        self.regions(page_index).add(
            [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)], state)
        if isinstance(self._current_action, _Action):
            self._current_action.region = True

    def add_region_gesture(self, page_index: int, polygon, decision: str) -> None:
        """One brush drag, as a polygon in page units.

        Called while the gesture's action is still open, so `undo` takes the polygon and the stroke
        records back together: they are one gesture, and two records that undo separately are two
        truths.
        """
        state = self._STATE_OF.get(decision)
        if state is None or not polygon:
            return
        self.regions(page_index).add(polygon, state)
        if isinstance(self._current_action, _Action):
            self._current_action.region = True

    def _decide_whole_layers(self, page_index: int, decision: str) -> None:
        """Every piece on the page, one decision -- recorded for undo.

        A whole-page action is the one case that really does mean the whole
        page, so it is also the one case that may walk it. The brush never
        comes through here.
        """
        if not self.uses_layers(page_index):
            return
        from xp.workers.merger import layers as _layers

        held = self.layers_for_page(page_index)
        state = _layers.STATE_OF[decision]
        held.begin()
        for side in held.sides.values():
            for at in range(len(side)):
                if int(side.state[at]) != state:
                    side._note(at, int(side.state[at]), state)
                    side.state[at] = state
                    side._recolour(at)
        journal = held.end()
        if journal:
            self._record([('pieces', (page_index, journal))])
            self.decisions(page_index).stamp += 1

    def apply_page(self, page_index: int, decision: str) -> List[Ref]:
        """Make the whole page one decision, in a single undoable step."""
        self.begin_action(page_index)
        try:
            return self.apply_whole_page(page_index, decision)
        finally:
            self.end_action(page_index)

    def reset_page(self, page_index: int) -> int:
        """Put this page back exactly as it was loaded, split and all.

        Everything decided goes, and if the page was split apart the two
        halves are joined again -- a reset that left the page count changed
        would not be a reset. Returns where the page ended up, since joining
        moves it.
        """
        while self.can_undo(page_index):
            self.undo(page_index)
        self._redo_stacks.pop(self.page_at(page_index).key, None)
        # The map goes back to the comparison it was seeded from: every polygon the user drew is gone, the
        # base plane is untouched, so "reset" means the page as the merger opened it.
        self.regions(page_index).clear()
        if self.is_split(page_index):
            page_index = self.join_page(page_index)
            self._decisions.pop(self.page_at(page_index).key, None)
        self.clear_fit(page_index)
        self.reset_layers(page_index)
        return page_index

    # ---------------- splitting a page in two ----------------

    def can_split(self, page_index: int) -> bool:
        """Only a page that has both sides can be split into two."""
        page = self.page_at(page_index)
        return page.left is not None and page.right is not None

    def is_split(self, page_index: int) -> bool:
        """Whether this page is one half of a pair that was split apart."""
        page = self.page_at(page_index)
        if page.left is not None and page.right is not None:
            return False
        return self._partner_index(page_index) is not None

    def _partner_index(self, page_index: int) -> Optional[int]:
        """Where the other half of this split is, wherever it has been moved.

        Halves know each other by key, not by sitting next to each other:
        which of the two comes first is a decision about the notebook, and
        making it must not cost you the ability to put them back together.
        """
        page = self.page_at(page_index)
        if page.left is not None and page.right is not None:
            return None
        partner_key = self._split_pairs.get(page.key)
        if partner_key is None:
            return None
        for index, other in enumerate(self._pages):
            if other.key == partner_key:
                return index
        return None

    def split_page(self, page_index: int, right_first: bool = False) -> int:
        """Keep both versions of this page, as two pages, from now on.

        The page stops being a comparison and becomes two pages of the
        merged document: the left one as it stands with its decisions, and
        the right one as it was written. They are ordinary pages after that
        -- they can be brushed, reordered and reset like any other -- and the
        merge is one page longer straight away, rather than at save time.

        Returns the index of the page that still carries the left side.
        """
        if not self.can_split(page_index):
            return page_index
        page = self._pages[page_index]
        left_half = MergedPage(page.key, page.left, None)
        key, self._next_key = self._next_key, self._next_key + 1
        right_half = MergedPage(key, None, page.right)
        order = [right_half, left_half] if right_first else [left_half, right_half]
        self._pages[page_index:page_index + 1] = order
        self._split_pairs[left_half.key] = right_half.key
        self._split_pairs[right_half.key] = left_half.key
        self._forget_placement(None)
        return page_index + (1 if right_first else 0)

    def join_page(self, page_index: int) -> int:
        """Put a split back together: one page comparing the two again."""
        partner = self._partner_index(page_index)
        if partner is None:
            return page_index
        first, second = sorted((page_index, partner))
        halves = (self._pages[first], self._pages[second])
        left_half = next(h for h in halves if h.left is not None)
        right_half = next(h for h in halves if h.right is not None)
        joined = MergedPage(left_half.key, left_half.left, right_half.right)
        # The joined page takes the earlier of the two places, whichever half
        # was there: putting them back together is not a reordering.
        del self._pages[second]
        self._pages[first] = joined
        self._split_pairs.pop(left_half.key, None)
        self._split_pairs.pop(right_half.key, None)
        self._forget_placement(None)
        return first

    def toggle_split(self, page_index: int) -> bool:
        """Split this page in two, or put a split back together."""
        if self.can_split(page_index):
            self.split_page(page_index)
            return True
        self.join_page(page_index)
        return False

    def move_page(self, page_index: int, delta: int) -> int:
        """Move a page up or down the merged document; returns where it went.

        Which of a split's two halves comes first is a decision about the
        notebook, not something to be stuck with.
        """
        target = page_index + delta
        if not (0 <= page_index < len(self._pages) and 0 <= target < len(self._pages)):
            return page_index
        page = self._pages.pop(page_index)
        self._pages.insert(target, page)
        self._forget_placement(None)
        return target

    def _on_page(self, page_index: int, stroke) -> bool:
        """Whether a stroke is on the page at all once it has been placed.

        Aligning by the frame moves a page's margin off the sheet: a PDF
        exported with margin notes has writing outside the frame, and the
        frame is what defines where the page is. Margin is margin -- it is
        not carried into the merged file.
        """
        if not stroke.points:
            return False
        width, height = self.page_size(page_index)
        xs = [x for x, _ in stroke.points]
        ys = [y for _, y in stroke.points]
        return not (max(xs) < -MARGIN_SLACK or min(xs) > width + MARGIN_SLACK
                    or max(ys) < -MARGIN_SLACK or min(ys) > height + MARGIN_SLACK)

    def composed_refs(self, page_index: int) -> List[Ref]:
        """Every stroke that survives, in drawing order.

        The order is left's ink then right's, each in its own document order.
        It is a drawing convention and nothing depends on it: what survives
        was decided above, per stroke, before anything is drawn.
        """
        return [ref for ref in self.refs_for_page(page_index)
                if self.keeps(page_index, ref) and self._on_page(page_index, self.stroke_for(page_index, ref))]

    def composed_strokes(self, page_index: int) -> List[Tuple[Ref, Stroke]]:
        left_strokes, right_strokes = self.strokes_for_page(page_index)
        composed = []
        for ref in self.composed_refs(page_index):
            strokes = left_strokes if ref.side == LEFT else right_strokes
            if ref.index < len(strokes):
                composed.append((ref, strokes[ref.index]))
        return composed

    def export(self, progress=None) -> XoppDocument:
        """Build the merged document from the decisions.

        Both the Final Preview and saving call this, so what the preview
        shows is what the file will contain. Neither source is touched: the
        pages are copies, and each carries its own background with it.
        """
        merged = XoppDocument.new_empty(page_count=0)
        for element in list(merged.root):
            merged.root.remove(element)
        # The left document's own header (title, preview thumbnail) rather
        # than a fresh one: this is that notebook, continued.
        merged.root.attrib.update(self.left.root.attrib)
        for element in list(self.left.root):
            if element.tag != 'page':
                merged.root.append(copy.deepcopy(element))

        # **The file keeps every left page; the navigation keeps the pairs.**
        # Since 2026-10-30 an unpaired page is not a merge page -- it has
        # nothing to compare -- but it is still the user's page, and a save that
        # dropped it would be losing work, not filtering a view. The user's own rule
        # for the filter says so: "filtering unpaired pages must not modify
        # the source documents". So the file is walked in the left document's
        # order: a paired page is written as the merge, an unpaired one is
        # written as itself.
        #
        # Walked in the **merge's** order, not the left document's. The first
        # version of this walked `range(len(self.left.pages))` and looked each
        # one up, which lost two things: a page the merge holds with no left
        # side at all -- the right half of a split -- was never reached and so
        # never written, and a page the user had moved went back to where it started.
        # Both are losing work, and the split one was caught by
        # `test_the_brush_survives_the_geometry`.
        #
        # So the merge pages are written in their own order, and an unpaired
        # left page is slotted in among them by its index: it keeps its place
        # in the notebook without the merge having to hold a page for it.
        unpaired = [index for index in range(len(self.left.pages))
                    if index not in {page.left for page in self._pages}]
        order = []
        waiting = list(unpaired)
        for page_index, page in enumerate(self._pages):
            while waiting and (page.left is None or waiting[0] < page.left):
                order.append((waiting.pop(0), None))
            order.append((page.left, page_index))
        for left_index in waiting:              # anything after the last pair
            order.append((left_index, None))
        total = len(order)
        for at, (left_index, page_index) in enumerate(order):
            # Real per-page progress, because building the pages is where the
            # time goes on a dense document and a caller that cannot say how
            # far it has got can only spin a cursor at the user.
            if progress is not None:
                progress(at, total)
            if page_index is None:
                # **An unpaired page is still a page, and it may never have been looked at.** Its element is
                # copied as it stands, so it has to be cleaned first or the saved file would depend on how far
                # the user had scrolled: a page the user never visited would go out with its printed border in it.
                if self.left_side is not None:
                    self.left_side.use(left_index)
                merged.root.append(copy.deepcopy(self.left.pages[left_index].element))
                continue
            merged.root.append(self._merged_page(page_index))
        return merged

    def convert_everything_taken(self, profile=None, progress=None) -> int:
        """Convert every page the brush took picture content from.

        This is the deferred conversion, and it is the only place in the
        merge that runs the converter. A document whose picture side was
        never brushed converts nothing at all.
        """
        total = 0
        for page_index in range(self.page_count):
            total += self.convert_areas(page_index, profile, progress)
        return total

    def _runs_as_elements(self, page_index: int, side: str, elements, fit):
        """This side's strokes, cut where the **owner** of the page changes.

        The export path for a stroke page whose decisions were drawn. The plane
        `PageDecisions.owner()` answers says who is on top at each pixel, so a
        stroke is walked once and cut where that answer changes. Both sides read the
        same plane and are therefore cut at the same boundary -- no gap, no doubled
        ink -- with no pen inset, which a *converted* stroke needs because its cap
        would otherwise double the notebook's.
        """
        from xp.core.document.xopp_io import slice_stroke_span

        from xp.workers.merger import views

        page = self.decisions(page_index)
        width, height = views.page_pixels(self, page_index)
        # **The region map, not the decision masks.** `owner()` was the export's answer and the view
        # has stopped reading it; two answers about what is kept is the state the user called out, and this
        # is the last reader. `owner_plane` is the same four states in the vocabulary `_owner_runs`
        # speaks, so the cutting below is unchanged.
        plane = self.regions(page_index).owner_plane((int(height), int(width)),
                                                     float(self.scale))
        strokes = self.strokes_for_page(page_index)[0 if side == LEFT else 1]
        # 0 undecided, 1 left won, 2 right won, 3 neither, 4 the user confirmed both.
        # The base side writes undecided ink and all of a "both" region; the other
        # side writes what it won and, inside a "both" region, only the ink that is
        # **not** common -- common ink is written once, from the base.
        base = side == self.base_side()
        matched = set()
        if not base:
            matched = set(views.matched_sides(self, page_index)[1]
                          if side == RIGHT else
                          views.matched_sides(self, page_index)[0])
        wanted = (0, 1, 4) if base else (2, 4)
        out = []
        for index, element in enumerate(elements):
            if index < len(strokes) and not self._on_page(page_index,
                                                          strokes[index]):
                continue
            moved = (element if fit.is_identity
                     else transform_stroke_element(element, fit.scale_x, fit.dx,
                                                   fit.dy, scale_y=fit.scale_y))
            runs = _owner_runs(moved, plane, self.scale)
            if not runs:
                continue
            # **Common ink is written once, from the base side** -- and this has to
            # be asked before the whole-run fast path below, which used to append
            # the stroke without it: Take Both then wrote both copies of the same
            # line (measured: 16,024 strokes where 8,515 was the ceiling).
            if (not base and index in matched
                    and not any(owner == 2 for _low, _high, owner in runs)):
                continue
            points = len((moved.text or '').split()) // 2
            if (len(runs) == 1 and runs[0][2] in wanted
                    and runs[0][0] <= 0.0 and runs[0][1] >= points - 1):
                out.append(moved)
                continue
            for low, high, owner in runs:
                if owner not in wanted or high - low <= 1e-9:
                    continue
                if not base and owner == 4 and index in matched:
                    continue          # the base side writes the common ink
                cut = slice_stroke_span(moved, low, high)
                if len((cut.text or '').split()) >= 4:
                    out.append(cut)
        return out

    def _kept_by_mask(self, page_index: int, side: str, elements, fit):
        """This side's strokes, cut to where the page still shows it.

        The same rule the pictures are cut by (`_visible_area`), applied to
        native strokes: spans, not whole strokes, so a stroke that crosses the
        brush's edge keeps the part the user can see. `_visible_spans` is the one
        implementation of that cut in this file.

        `None` from `_visible_area` means "everywhere" -- the base side of a
        page nobody decided -- and then the strokes are copied through exactly
        as they are, which is what an untouched merge writes.
        """
        from xp.core.document.xopp_io import slice_stroke_span

        visible = self._visible_area(page_index, side)
        strokes = self.strokes_for_page(page_index)[0 if side == LEFT else 1]
        out = []
        for index, element in enumerate(elements):
            if index < len(strokes) and not self._on_page(page_index,
                                                          strokes[index]):
                continue
            moved = (element if fit.is_identity
                     else transform_stroke_element(element, fit.scale_x, fit.dx,
                                                   fit.dy, scale_y=fit.scale_y))
            if visible is None:
                out.append(moved)
                continue
            points = len((moved.text or '').split()) // 2
            if points < 2:
                continue
            spans = _visible_spans(moved, visible, self.scale,
                                   _pen_width(moved))
            for low, high in spans:
                if low <= 0.0 and high >= points - 1:
                    out.append(moved)
                    break
                cut = slice_stroke_span(moved, low, high)
                if len((cut.text or '').split()) >= 4:
                    out.append(cut)
        return out

    def final_strokes(self, page_index: int) -> List:
        """The strokes the file will hold for this page, placed, and nothing else.

        Split out of `_merged_page` so the **Final Preview** draws the same
        selection the export writes. It did not, and the difference was
        measurable: with a band taken from the PDF side of the user's pair, the preview
        held 8,097 ink pixels fewer than the export -- the preview drew the
        stroke decisions (`composed_strokes`, whole strokes) while the export
        cuts by the mask and adds what the brush took. What the user reviews must be
        what the user gets, so the selection lives in one function and both callers
        use it.
        """
        left_page, right_page = self._page_pair(page_index)
        if left_page is None:
            fallback = self._right_page_copy(page_index)
            return fallback.findall('.//stroke')
        geometry = self.geometry(page_index)
        kept = []
        page = self.decisions(page_index)
        drawn = bool(page.taken) or bool(page.cleared)
        # **The areas win when the page has them.** A page whose decisions were
        # drawn is cut by them at export, which is where the specification puts
        # stroke-level work; a page nobody has brushed still writes its strokes
        # through the pieces, exactly as before.
        by_piece = self.uses_layers(page_index) and not drawn
        left_fit = geometry.fits[LEFT]
        elements = left_page.layer.findall('stroke')
        if by_piece:
            kept.extend(self._pieces_as_elements(page_index, LEFT, elements,
                                                 left_fit))
        elif self.uses_layers(page_index):
            kept.extend(self._runs_as_elements(page_index, LEFT, elements,
                                               left_fit))
        else:
            kept.extend(self._kept_by_mask(page_index, LEFT, elements, left_fit))
        if right_page is not None:
            right_fit = geometry.fits[RIGHT]
            others = right_page.layer.findall('stroke')
            if by_piece:
                kept.extend(self._pieces_as_elements(page_index, RIGHT, others,
                                                     right_fit))
            elif self.uses_layers(page_index):
                kept.extend(self._runs_as_elements(page_index, RIGHT, others,
                                                   right_fit))
            else:
                kept.extend(self._kept_by_mask(page_index, RIGHT, others,
                                               right_fit))
        return kept

    def final_page_strokes(self, page_index: int) -> List:
        """Everything the exported page will hold as a stroke, in order, healed.

        One place, because there were two: `_merged_page` appended the native
        pieces and then the converted ones, and `views.final_view` did the same
        thing with its own two calls. Heal has to see **both** -- the user's example is
        a converted piece and a native piece that are one line -- so the list is
        built here and the linking happens once, for the file and the preview
        alike.
        """
        elements = list(self.final_strokes(page_index))
        elements.extend(self._converted_for(page_index))
        return _link_the_pieces(elements)

    def _merged_page(self, page_index: int) -> ET.Element:
        left_page, right_page = self._page_pair(page_index)
        if left_page is None:
            return self._right_page_copy(page_index)

        element = copy.deepcopy(left_page.element)
        geometry = self.geometry(page_index)
        width, height = geometry.size
        element.set('width', f'{width:.4f}')
        element.set('height', f'{height:.4f}')
        layer = element.find('layer')
        if layer is None:
            layer = ET.SubElement(element, 'layer')

        kept = self.final_page_strokes(page_index)
        for stroke in layer.findall('stroke'):
            layer.remove(stroke)
        for stroke in kept:
            layer.append(stroke)
        # **The left page's other objects move with its ink.** Text, images and
        # any layer above the first were carried through exactly as they stood,
        # which is right only while the left fit is the identity: a caption and
        # a picture stayed where the paper used to be while the writing under
        # them moved. They are transformed here, by the same numbers; they are
        # still not compared or brushed -- see `unhandled`.
        left_fit = geometry.fits[LEFT]
        if not left_fit.is_identity:
            from xp.core.document.xopp_io import transform_page_objects

            self.moved_objects = getattr(self, 'moved_objects', 0) + \
                transform_page_objects(element, left_fit.scale_x, left_fit.dx,
                                       left_fit.scale_y, left_fit.dy)
        # **Pictures become strokes here, at the last moment, and only here.**
        # During editing they were just part of that side's image; the export
        # is the only place that has to resolve them, and it resolves only what
        # the page still shows.
        self._convert_the_pictures(page_index, element)
        # The strokes of a page compared as an image were already cut by their
        # own mask in `_kept_by_mask`, so there is nothing left to take out.
        # A layered page has no areas at all -- its decisions live on pieces.
        return element

    def _pieces_as_elements(self, page_index: int, side: str, elements, fit):
        """The surviving **pieces** of this side, as real Xournal++ strokes.

        A decision is now about a piece of a stroke, so the file has to say
        the same thing the screen does: a stroke the user brushed half of is
        written out as the half that survives, and a stroke nobody touched is
        written through whole. Without this the picture and the file would
        disagree -- the fault the old model hid by never deciding anything
        about part of a stroke in the first place.

        Still native strokes, sliced and moved; nothing is rasterised.
        """
        from xp.core.document.xopp_io import slice_stroke_span

        held = self.layers_for_page(page_index)[side]
        wanted = {}
        for origin, first, last in held.kept_pieces():
            wanted.setdefault(origin, []).append((first, last))
        strokes = (self.strokes_for_page(page_index)[0 if side == LEFT else 1])
        out = []
        for index, element in enumerate(elements):
            parts = wanted.get(index)
            if not parts:
                continue
            if index < len(strokes) and not self._on_page(
                    page_index, strokes[index]):
                continue
            points = len((element.text or '').split()) // 2
            for first, last in sorted(parts):
                whole = first <= 0 and last >= points - 1
                piece = (copy.deepcopy(element) if whole
                         else slice_stroke_span(element, first, last))
                if len((piece.text or '').split()) < 4:
                    continue          # fewer than two points draws nothing
                out.append(piece if fit.is_identity
                           else transform_stroke_element(piece, fit.scale_x, fit.dx, fit.dy, scale_y=fit.scale_y))
        return out

    def _held(self, side: str):
        """The `sides.Side` for a side, or None."""
        return self.left_side if side == LEFT else self.right_side

    def _visible_area(self, page_index: int, side: str):
        """Where this side's content is **written to the file**, as a mask.

        None means "everywhere", and that is only true for the base side on a
        page nobody has decided: undecided ink is kept from the base and
        dropped from the other side (`keeps`, `default_keeps`). The mask has to
        say the same thing the strokes do, or a picture on the incoming side
        would be written into a page where none of that side's strokes are.

        `taken` is where this side won, `cleared` where it was refused. Taken
        adds, cleared subtracts, and what was never decided falls back to the
        side's default.
        """
        page = self.decisions(page_index)
        taken = page.area_taken(side)
        cleared = page.area_cleared(side)
        base = self.base_side()
        if taken is None and cleared is None:
            if side == base:
                return None
            from xp.workers.merger import views

            width, height = views.page_pixels(self, page_index)
            return np.zeros((height, width), bool)
        shape = taken.shape if taken is not None else cleared.shape
        shown = np.ones(shape, bool) if side == base else np.zeros(shape, bool)
        if taken is not None:
            shown |= taken
        if cleared is not None:
            shown &= ~cleared
        return shown

    def _convert_the_pictures(self, page_index: int, element) -> int:
        """Turn this page's embedded pictures into strokes, **in the visible part only**.

        The user's specification, on pictures in a XOPP: during the merge they are
        simply part of that side's image; at export "only the part that is
        finally kept is converted", and the result is native editable strokes,
        never a PNG patch. So each `<image>` is traced with the converter's own
        routine (`embedded.trace`, cached by content), placed by the side's fit,
        cut to the area the page still shows, and the `<image>` element is
        removed from the output whether it was converted or refused.

        A side whose pages are a PDF is not handled here: it has no `<image>`
        elements, and `convert_areas` renders and converts it.
        """
        from xp.workers.merger import embedded
        from xp.workers.merger.sides import XOPP

        layer = element.find('layer')
        if layer is None:
            return 0
        made = []
        for side in (LEFT, RIGHT):
            if side not in self.page_picture_sides(page_index):
                continue
            held = self._held(side)
            if held is None or held.kind != XOPP:
                continue
            page_now = self.page_at(page_index)
            own_index = page_now.left if side == LEFT else page_now.right
            document = self.left if side == LEFT else self.right
            if own_index is None or document is None:
                continue
            own_page = document.pages[own_index]
            visible = self._visible_area(page_index, side)
            fit = self.geometry(page_index).fits[side]
            for _layer, picture, box in embedded.images_of(own_page):
                for held_stroke in embedded.trace(picture, box):
                    piece = embedded._element_from(held_stroke)
                    moved = (piece if fit.is_identity
                             else transform_stroke_element(
                                 piece, fit.scale_x, fit.dx, fit.dy,
                                 scale_y=fit.scale_y))
                    if visible is None:
                        made.append(moved)
                        continue
                    spans = _visible_spans(moved, visible, self.scale,
                                           _pen_width(moved))
                    points = len((moved.text or '').split()) // 2
                    for low, high in spans:
                        if low <= 0.0 and high >= points - 1:
                            made.append(moved)
                        else:
                            cut = slice_stroke_span(moved, low, high)
                            if len((cut.text or '').split()) >= 4:
                                made.append(cut)
        if made:
            for piece in made:
                layer.append(piece)
        # **Nothing is remembered here.** These strokes are in the page the
        # caller is about to receive, which is the only place they need to be;
        # keeping them in `_converted` as well made every second export hold them
        # again -- measured, 3,661 strokes became 3,843 and then 4,025 on
        # repeated calls, so the saved file did not match what was previewed.
        # **The picture itself does not go into the file, converted or not.**
        # Its ink is the strokes above where it was kept; leaving the `<image>`
        # would write the content twice, and the second copy is the un-editable
        # patch the user refuses. Where it was refused, removing it is the refusal --
        # a page decided Neither must not carry the picture through.
        for parent in list(element.iter()):
            for picture in list(parent):
                if picture.tag == 'image':
                    parent.remove(picture)
        return len(made)

    def convert_areas(self, page_index: int, profile=None, progress=None) -> int:
        """Turn what the brush took from a picture side into strokes.

        **The existing converter, on the whole page, once.** Not a second
        algorithm and not a partial-page one: `convert_page_image` is what
        `convert_pdf` calls for every page it converts, so what comes out of
        here is what the converter produces, exactly. The brush's mask then
        picks from it with `fully_covers_stroke`, which is the same test the
        brush uses on a side that already has strokes.

        Measured on the sample page, this is also the cheaper way round: the
        page costs 2.53s at 150 dpi and 2.93s at 300, and every further area
        on it then costs 0.03s, because the page is converted once and kept.
        A per-area converter was 0.95s for the first and 0.38s for each
        after -- dearer from the second area on, and it lost 32.5% of the
        ink, because `recover_interiors` answers a question about the whole
        page and a crop cannot see it.

        Returns how many strokes it added. Called by `export`, and callable
        on its own so a window can show its cost and its stages.
        """
        from xp.workers.converter.profiles import Profile
        from xp.workers.merger import sides as _sides

        page = self.decisions(page_index)
        # **Only a PDF side is converted by rendering the page.** A notebook
        # side with an embedded picture is *not*: rendering it would trace the
        # notebook's own strokes as well, so the export would hold every stroke
        # twice -- once native, once traced. Its pictures are converted one
        # element at a time by `_convert_the_pictures`.
        wanted = [side for side in self.picture_sides()
                  if self._held(side) is not None
                  and self._held(side).kind == _sides.PDF
                  and page.area_taken(side) is not None
                  and page.area_taken(side).any()]
        if not wanted:
            self._converted.pop(page_index, None)
            return 0
        profile = profile or self._profile or Profile.named('hand')
        made = []
        for side in wanted:
            made.extend(self._convert_side_area(page_index, side, profile,
                                                progress))
        self._converted[page_index] = made
        return len(made)

    def _convert_side_area(self, page_index: int, side: str, profile,
                           progress=None) -> List:
        from xp.workers.converter.convert import convert_page_image

        held = self.left_side if side == LEFT else self.right_side
        source = self.page_at(page_index)
        source_index = source.left if side == LEFT else source.right
        if source_index is None:
            return []
        scale = profile.dpi / 72.0
        if progress is not None:
            progress(page_index, side)
        raw = held.render(source_index, scale)
        strokes = convert_page_image(raw, profile, scale, 0.0, 0.0)

        # Into the merged page's coordinates, the same way every other
        # stroke from this side gets there.
        fit = self.geometry(page_index).fits[side]
        taken = self.decisions(page_index).area_taken(side)
        kept = []
        for element in strokes:
            moved = (element if fit.is_identity
                     else transform_stroke_element(element, fit.scale_x, fit.dx, fit.dy, scale_y=fit.scale_y))
            # **The visible part, not the whole line and not nothing.** A line
            # the brush took half of used to be dropped entirely, so the export
            # had a hole exactly where the user had selected. Each visible run becomes
            # its own stroke; a fully visible line is copied whole.
            spans = _visible_spans(moved, taken, self.scale,
                                   _pen_width(moved))
            if not spans:
                continue
            points = len((moved.text or '').split()) // 2
            if len(spans) == 1 and spans[0][0] <= 0.0 and spans[0][1] >= points - 1:
                kept.append(copy.deepcopy(moved))
                continue
            for low, high in spans:
                piece = slice_stroke_span(moved, low, high)
                if len((piece.text or '').split()) >= 4:
                    kept.append(piece)
        return kept

    def _converted_for(self, page_index: int) -> List:
        return self._converted.get(page_index, [])

    def _right_page_copy(self, page_index: int) -> ET.Element:
        """A page that is the right side: its own page, its own decisions.

        One half of a split is an ordinary page of the merged document -- it
        can be brushed like any other -- so what it holds is what was decided
        about it, placed into the merged geometry.
        """
        _, right_page = self._page_pair(page_index)
        element = copy.deepcopy(right_page.element)
        geometry = self.geometry(page_index)
        width, height = geometry.size
        element.set('width', f'{width:.4f}')
        element.set('height', f'{height:.4f}')
        fit = geometry.fits[RIGHT]
        # **Pieces, exactly as an ordinary page has them.** This used to test
        # whole strokes against `composed_refs`: a stroke the user had brushed half
        # of was exported whole if any part of it survived, so the split page's
        # file said something the screen did not. The same exporter the
        # ordinary pages use answers this one.
        layers = element.findall('layer')
        if not layers:
            return element
        # **Elements, not `Stroke` objects**: `_pieces_as_elements` slices the
        # real XML, which is what keeps the file a notebook rather than a
        # picture of one.
        elements = layers[0].findall('stroke')
        kept = self._pieces_as_elements(page_index, RIGHT, elements, fit)
        for layer in layers:
            for stroke in layer.findall('stroke'):
                layer.remove(stroke)
        for stroke in kept:
            layers[0].append(stroke)
        return element

    def is_lazy(self) -> bool:
        """Whether either side is too big to hold whole."""
        from xp.core.document.bigxopp import BigDocument

        return isinstance(self.left, BigDocument) or isinstance(self.right,
                                                                BigDocument)

    def save(self, path, compresslevel: int = 1, progress=None) -> None:
        """Write the merge. A document too big to hold goes page by page."""
        if self.is_lazy():
            self.save_page_by_page(path, compresslevel=compresslevel,
                                   progress=progress)
            return
        self.export(progress=progress).save(path, compresslevel=compresslevel)

    def save_page_by_page(self, path, compresslevel: int = 1,
                          progress=None) -> None:
        """Write the merge out without ever holding it whole.

        `export` builds the merged document in memory, and for a document of
        the user's that is 17.6 million elements -- the same wall `bigxopp` exists
        to get around, reached from the other side. Here a page is built,
        written and let go, so what is held at once is one page.

        The pages come out of exactly the same `_merged_page` the ordinary
        export uses, so a merge saved this way and a merge saved the other
        way are the same merge. Only the holding differs.
        """
        import gzip
        from pathlib import Path

        header = copy.deepcopy(self.left.root)
        for element in list(header):
            if element.tag == 'page':
                header.remove(element)
        head = ET.tostring(header, encoding='unicode')
        cut = head.rfind('</xournal>')
        if cut >= 0:
            head = head[:cut]
        elif head.endswith(' />'):
            head = head[:-3] + '>'

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # The same rule as the ordinary save: a temporary file beside the
        # target, flushed, then atomically replaced. A big notebook takes
        # minutes to write, which is exactly when a kill would otherwise leave
        # a truncated file where the previous save used to be.
        import os

        temporary = target.with_name(target.name + '.partial-%d' % os.getpid())
        from xp.core.document.atomic import fsync_directory, sweep_partials

        sweep_partials(target)
        try:
            with gzip.open(temporary, 'wb', compresslevel=compresslevel) as out:
                out.write(_DECLARATION)
                out.write(head.encode('utf-8'))
                for page_index in range(self.page_count):
                    if progress is not None:
                        progress(page_index + 1, self.page_count)
                    element = self._merged_page(page_index)
                    out.write(ET.tostring(element, encoding='utf-8'))
                    out.write(_BREAK)
                    element.clear()
                    self._forget_page(page_index)
                out.write(_CLOSING)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, target)
            fsync_directory(target.parent)
        except BaseException:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise

    def _forget_page(self, page_index: int) -> None:
        """Let go of everything cached about one page.

        Saving walks every page once and never comes back, so holding what
        each one cost would rebuild the whole document in memory by another
        route.
        """
        self._stroke_cache.pop(page_index, None)
        self._flat_cache.pop(page_index, None)
        self._layer_cache.pop(self.page_at(page_index).key, None)
        self._raw_cache.pop(page_index, None)

    def commit(self) -> XoppDocument:
        """The merged document, built from the decisions as they stand."""
        return self.export()
