"""Every way a merge can be looked at, and the line between them.

Two families of view, which must never be confused with each other:

- the **editing view**, which answers "where do these two documents differ
  and what have I decided?" in four diagnostic colours that exist only on
  screen: red for left-only ink, blue for right-only, grey for ink both
  sides have, black for anything decided and kept;
- the **original-colour views**, which answer "what is actually there?" --
  the raw left page, the raw right page, the two overlaid, and the merged
  result. These draw every stroke in the colour it was written in.

Each view is built from the immutable sources and, where it needs them, the
current decisions. None of them keeps state, and none of them writes any:
switching between views cannot leak a mask, a tint or a decision from the
last one, because there is nothing to leak -- every call starts from a blank
image and the source data.

`final` is the merged result, and it is drawn from exactly what `export()`
writes, so Final Preview and the exported file cannot disagree.
"""
from __future__ import annotations

import threading

#: Guards the session's render caches. Held for a dict operation, never across a render.
_CACHE_LOCK = threading.RLock()

from typing import Optional, Tuple

import numpy as np

from types import SimpleNamespace
from PIL import Image, ImageDraw

from xp.workers.merger.compose import LEFT, RIGHT
from xp.render.drawing import _drawn_width, _hex_to_rgb
from xp.workers.merger.matching.diffmatch import match_strokes
from xp.render.drawing import BACKGROUND_COLOR, draw_strokes

# The editing palette. On screen only: no source colour is ever changed.
#
# Three colours and three answers, which is what the view is for: black for
# content both sides have, red for content only the left has, blue for content
# only the right. Red is the left and blue is the right because that is the
# convention this project has always used -- `render.py` said "blood red: only
# in left / existing xopp" and "sky blue: only in right / PDF-derived" before
# these constants moved here, and it did not change on the way.
#
# Black carries two meanings that are the same meaning: ink both sides have,
# and ink that has been decided and kept. Both are content that belongs in the
# result, which is why they are one colour rather than two.
LEFT_ONLY = (155, 17, 30)      # blood red: only the left side has this
RIGHT_ONLY = (62, 166, 227)    # sky blue: only the right side has this
OVERLAP = (0, 0, 0)            # black: both sides have this, or it was kept
DECIDED = (0, 0, 0)            # the same black: decided, and kept
# Kept, and unique to one side: the user confirmed it. A selection colour -- it
# lives in this view and never reaches the saved notebook, which is written from
# the strokes and the decisions, not from what was drawn.
#
# Green is the colour of *simplification, organisation, general connection* --
# the same meaning it carries in the five-colour scheme this program is built
# on. This is a reference to that idea rather than a shade chosen to look nice.
CONFIRMED = (24, 154, 70)

# The order the colours are painted in, and it is not arbitrary. Common content
# goes down first and the unique colours go on top of it, so a difference is
# never hidden under the ink both sides agree about. Painting in the other order
# is how a stroke that is only on one side ends up invisible under a black one
# that is on both.
#
# CONFIRMED is in this list and must stay in it. Leaving it out is invisible in
# the obvious test: the live window repaints the strokes it just decided about
# itself, so green still appears in the gesture that made it and then vanishes
# on the next rebuild. `tests/test_view_repair.py` compares the view against a
# fresh render after every action and is the test that catches it.
PAINT_ORDER = (OVERLAP, LEFT_ONLY, RIGHT_ONLY, CONFIRMED)

EDIT = 'edit'
SOURCE_LEFT = 'left'
SOURCE_RIGHT = 'right'
OVERLAY = 'overlay'
FINAL = 'final'
PREVIEW_MODES = (SOURCE_LEFT, SOURCE_RIGHT, OVERLAY, FINAL)

MODE_LABELS = {
    EDIT: 'Editing',
    SOURCE_LEFT: 'Preview: left page, original colours',
    SOURCE_RIGHT: 'Preview: right page, original colours',
    OVERLAY: 'Preview: left and right, original colours',
    FINAL: 'Final preview: what will be exported',
}


def page_pixels(session, page_index: int, scale: Optional[float] = None) -> Tuple[int, int]:
    width, height = session.page_size(page_index)
    scale = session.scale if scale is None else scale
    return max(1, round(width * scale)), max(1, round(height * scale))


def _blank(session, page_index: int, scale: float, window=None) -> Image.Image:
    if window is None:
        return Image.new('RGB', page_pixels(session, page_index, scale),
                         BACKGROUND_COLOR)
    return Image.new('RGB', (max(1, int(window[2])), max(1, int(window[3]))),
                     BACKGROUND_COLOR)


def boxes_of(strokes):
    """Every stroke's box and half-width, as arrays, worked out once.

    `Stroke.bbox` walks all of a stroke's points on every read and
    `max(widths)` walks its widths, and the window cull asks both of every
    stroke on every frame. On a 140,203-stroke page that is 0.176s and
    0.058s of *deciding* before anything is drawn -- and the sample notebook
    has a page with 1,694,566, where it would be two seconds a frame.

    So it is done once per page and the cull becomes a numpy comparison.
    This is the same fix as `sides.Side._index`, which is where it was
    measured: 60x, and the difference between dragging a view and waiting
    for one.
    """
    count = len(strokes)
    boxes = np.empty((count, 4), np.float64)
    half = np.empty(count, np.float64)
    for at, stroke in enumerate(strokes):
        boxes[at] = stroke.bbox
        half[at] = (max(stroke.widths) if stroke.widths else 1.0) / 2.0
    return boxes, half


def _page_boxes(session, page_index: int):
    """The two sides' boxes for this page, held on the session.

    Thrown away wherever the strokes themselves are -- `strokes_for_page`
    owns that, and this follows it by being keyed on the tuple it returns.
    """
    held = getattr(session, '_box_cache', None)
    if held is None:
        held = session._box_cache = {}
    strokes = session.strokes_for_page(page_index)
    have = held.get(page_index)
    if have is not None and have[0] is strokes:
        return have[1]
    made = (boxes_of(strokes[0]), boxes_of(strokes[1]))
    held[page_index] = (strokes, made)
    return made


def arrays_of(strokes):
    """A side's strokes as flat arrays, in page units, worked out once.

    The user's own instruction, from `strokes.py`: *all strokes stored as
    contiguous arrays, not repeatedly built as objects, flattened, uploaded
    and downloaded.* The conversion pipeline was taught that; drawing was
    not, and it cost the same way. Measured on the user's 140,203-stroke page, one
    frame of the merge view spent 0.271s building a `_Band` per stroke and
    0.187s flattening them back, against 0.199s drawing -- three quarters of
    it turning objects into arrays and back, every frame, while the user zoomed.

    Widths stay in page units so one set of arrays serves every zoom.
    """
    count = len(strokes)
    counts = np.fromiter((len(st.points) for st in strokes), np.int64, count)
    starts = np.zeros(count, np.int64)
    if count:
        np.cumsum(counts[:-1], out=starts[1:])
    total = int(counts.sum())
    xs = np.empty(total, np.float64)
    ys = np.empty(total, np.float64)
    widths = np.empty(count, np.float64)
    own = np.zeros((count, 3), np.uint8)
    for at, stroke in enumerate(strokes):
        begin = starts[at]
        points = stroke.points
        for k, (x, y) in enumerate(points):
            xs[begin + k] = x
            ys[begin + k] = y
        widths[at] = max(0.25, _drawn_width(stroke))
        own[at] = _hex_to_rgb(stroke.color)
    return xs, ys, starts, counts, widths, own


def _page_arrays(session, page_index: int):
    """Both sides' flat arrays for this page, held on the session."""
    held = getattr(session, '_array_cache', None)
    if held is None:
        held = session._array_cache = {}
    strokes = session.strokes_for_page(page_index)
    have = held.get(page_index)
    if have is not None and have[0] is strokes:
        return have[1]
    made = (arrays_of(strokes[0]), arrays_of(strokes[1]))
    held[page_index] = (strokes, made)
    return made


def _cull(arrays, scale, window):
    """Which strokes touch `window`, as a boolean over the side."""
    xs, ys, starts, counts, widths, _own = arrays
    many = len(starts)
    if window is None:
        return np.ones(many, np.bool_)
    if not many:
        return np.zeros(0, np.bool_)
    boxes = _boxes_from(arrays)
    left, top, width, height = window
    right, bottom = left + width, top + height
    pad = widths * scale * 0.5 + 1.0
    return ~((boxes[:, 2] * scale + pad < left)
             | (boxes[:, 0] * scale - pad > right)
             | (boxes[:, 3] * scale + pad < top)
             | (boxes[:, 1] * scale - pad > bottom))


def _boxes_from(arrays):
    """Every stroke's box, from the flat arrays, with no Python loop.

    `np.minimum.reduceat` over the point array gives all of them in one
    call; the alternative is `Stroke.bbox` per stroke, which walks every
    point again and was 0.176s a frame on the user's page.
    """
    xs, ys, starts, counts, _w, _o = arrays
    held = getattr(arrays, '_boxes', None)
    if held is not None:
        return held
    many = len(starts)
    boxes = np.empty((many, 4), np.float64)
    if many:
        at = starts[counts > 0]
        boxes[:, 0] = np.minimum.reduceat(xs, at)
        boxes[:, 1] = np.minimum.reduceat(ys, at)
        boxes[:, 2] = np.maximum.reduceat(xs, at)
        boxes[:, 3] = np.maximum.reduceat(ys, at)
    return boxes


def _in_window(strokes, scale, window, tints=None, boxes=None):
    """The strokes that touch `window`, and their tints, in the same order.

    The point of the window is not the smaller image -- it is this. On a
    twenty-thousand-stroke page at 4x only 1,870 strokes are in the window,
    and handing the rasterizer the other eighteen thousand to clip is most
    of the cost of a redraw.

    A stroke's own width is added to its box, or a thick stroke whose spine
    is just outside would have its edge clipped away at the window's rim.

    `boxes` is `boxes_of(strokes)` from somewhere that holds it. Without it
    they are worked out here, which is correct and costs a pass.
    """
    if window is None:
        return list(strokes), (None if tints is None else list(tints))
    left, top, width, height = window
    right, bottom = left + width, top + height
    if not strokes:
        return [], (None if tints is None else [])
    span, half = boxes if boxes is not None else boxes_of(strokes)
    pad = half * scale + 1.0
    keep = ~((span[:, 2] * scale + pad < left)
             | (span[:, 0] * scale - pad > right)
             | (span[:, 3] * scale + pad < top)
             | (span[:, 1] * scale - pad > bottom))
    wanted = np.nonzero(keep)[0]
    kept = [strokes[i] for i in wanted]
    if tints is None:
        return kept, None
    return kept, [tints[i] for i in wanted]


def _in_window_slow(strokes, scale, window, tints=None):
    """The loop the vectorised cull replaces, kept for the test that says
    the two agree. Not used by anything that draws."""
    if window is None:
        return list(strokes), (None if tints is None else list(tints))
    left, top, width, height = window
    right, bottom = left + width, top + height
    kept, keep_tints = [], []
    for index, stroke in enumerate(strokes):
        x0, y0, x1, y1 = stroke.bbox
        pad = (max(stroke.widths) if stroke.widths else 1.0) * scale / 2.0 + 1.0
        if (x1 * scale + pad < left or x0 * scale - pad > right
                or y1 * scale + pad < top or y0 * scale - pad > bottom):
            continue
        kept.append(stroke)
        if tints is not None:
            keep_tints.append(tints[index])
    return kept, (None if tints is None else keep_tints)


def _offset(window):
    return (0.0, 0.0) if window is None else (float(window[0]), float(window[1]))


def matched_sides(session, page_index: int):
    """Which strokes on each side have a counterpart on the other.

    This is what "grey" means in the editing view, and it costs a pass over
    the page, so callers that redraw often (the merge window, on every
    mouse-motion event of a drag) work it out once per page and hand it back
    in rather than paying for it again.
    """
    left_strokes, right_strokes = session.strokes_for_page(page_index)
    # **By where the ink is, not by matching stroke centres.** A traced side's
    # "strokes" are contour pieces whose centres and sizes have nothing to do
    # with the pen strokes on the other side, so the old matcher called almost
    # nothing common and a correctly aligned page showed red and blue where it
    # should have shown black. On the sample pair this finds 100% common when the
    # pages are right and 21-31% when they are not.
    from xp.workers.merger.matching.common import common_strokes

    width, height = session.page_size(page_index)
    # **Once per page, and the decisions cannot change the answer.** Which strokes
    # are common is a question about the two sides' **ink** -- where the strokes sit
    # -- and not about what the user has decided; `edit_colors` is what turns this answer
    # into colours, and the brush changes that, not this. The key used to include
    # the decision stamp, so every gesture of a drag recomputed the coverage pass
    # over the whole page: measured on the user's 42,000-stroke page that was most of the
    # brush's input-to-picture cost.
    cache = getattr(session, '_matched_cache', None)
    if cache is None:
        cache = session._matched_cache = {}
    key = (page_index, id(left_strokes), id(right_strokes),
           getattr(session, '_layer_epoch', 0))
    hit = cache.get(key)
    if hit is not None:
        return hit
    answer = common_strokes(left_strokes, right_strokes, width, height)
    cache.clear()            # one page's answer is all a window is showing
    cache[key] = answer
    return answer


def edit_colors(session, page_index: int, matched=None):
    """The colour for every stroke in the editing view, by (side, index).

    Pulled out of the drawing so that a repaint of one small area colours
    its strokes exactly as a repaint of the whole page would.

    Held between frames, against the page's decision stamp. It is a pass
    over every stroke on the page and a dict with one entry each -- 0.264s
    and 140,203 entries on the user's TEST page, every frame, and the user has a page
    with 1,694,566. Nothing about it changes when the user zooms or pans, which
    is what the user does most; it changes when the brush decides something, and
    `PageDecisions.stamp` is what says so.
    """
    stamp = session.decisions(page_index).stamp
    strokes = session.strokes_for_page(page_index)
    held = getattr(session, '_colour_cache', None)
    if held is None:
        held = session._colour_cache = {}
    have = held.get(page_index)
    # Keyed on the strokes themselves as well, because re-pairing the pages
    # changes what is on this page without deciding anything.
    if (have is not None and have[0] == stamp and have[1] is strokes
            and have[2] is matched):
        return have[3]

    left_strokes, right_strokes = strokes
    matched_left, matched_right = matched if matched is not None else matched_sides(session, page_index)
    colors = {}
    for index in range(len(left_strokes)):
        colors[(LEFT, index)] = _color_for(session, page_index, LEFT, index, index in matched_left)
    for index in range(len(right_strokes)):
        colors[(RIGHT, index)] = _color_for(session, page_index, RIGHT, index, index in matched_right)
    held[page_index] = (stamp, strokes, matched, colors)
    return colors


def recolor(colours, session, page_index, refs, matched=None):
    """Bring a held colour map up to date for the strokes a decision changed.

    `edit_colors` walks every stroke on the page -- 0.264s and 140,203 entries
    on the user's TEST page, and the user has a page with 1,694,566 -- and a gesture
    decides a handful of strokes, so walking all of them for each gesture is
    exactly the cost the incremental repaint exists to avoid. The structural
    promise is held by `tests/test_brush_is_quick.py`: four gestures must
    rebuild the map zero times.

    `_color_for` is the same function the full walk uses, so an updated map
    and a rebuilt one cannot disagree. That is not a hope, it is asserted --
    the same file checks the incremental map against a full rebuild after a
    second gesture over the first, after undo, after redo, and after a
    whole-page decision.
    """
    matched_left, matched_right = (matched if matched is not None
                                   else matched_sides(session, page_index))
    for ref in refs:
        inside = ref.index in (matched_left if ref.side == LEFT else matched_right)
        colours[(ref.side, ref.index)] = _color_for(
            session, page_index, ref.side, ref.index, inside)
    return colours


def _color_for(session, page_index, side, index, matched):
    """One stroke's editing colour.

    Undecided ink is drawn on both sides whichever way it would go if you
    did nothing -- the right side's ink is not kept by default, but you
    cannot decide about what you cannot see, so it shows as a candidate in
    its own colour. Once decided, a stroke is either black (kept) or gone.
    """
    from xp.workers.merger.compose import Ref

    ref = Ref(side, index)
    if session.is_decided(page_index, ref):
        if not session.keeps(page_index, ref):
            return None
        # Kept. Black only when both documents had it -- that is shared
        # content, and confirming it again means nothing. A unique stroke the user
        # confirmed shows green, so the two states cannot be confused.
        return DECIDED if matched else CONFIRMED
    if matched:
        return OVERLAP
    return LEFT_ONLY if side == LEFT else RIGHT_ONLY


def edit_table(session, page_index: int, matched=None):
    """The editing colours as an (N, 3) array and a visible mask, per side.

    The dict form is what `edit_colors` gives and what everything that asks
    about one stroke wants; the painter wants an array. Turning one into the
    other is a pass over every stroke, so it is held against the same
    decision stamp -- zooming does not change a decision.
    """
    stamp = session.decisions(page_index).stamp
    strokes = session.strokes_for_page(page_index)
    held = getattr(session, '_table_cache', None)
    if held is None:
        held = session._table_cache = {}
    have = held.get(page_index)
    if (have is not None and have[0] == stamp and have[1] is strokes
            and have[2] is matched):
        return have[3]

    colors = edit_colors(session, page_index, matched)
    made = []
    for side, side_strokes in ((LEFT, strokes[0]), (RIGHT, strokes[1])):
        count = len(side_strokes)
        table = np.zeros((count, 3), np.uint8)
        visible = np.zeros(count, np.bool_)
        for index in range(count):
            colour = colors.get((side, index))
            if colour is None:
                continue
            table[index] = colour
            visible[index] = True
        made.append((table, visible))
    held[page_index] = (stamp, strokes, matched, tuple(made))
    return tuple(made)


def _class_mask(table, visible, colour):
    """Which of these strokes are drawn in exactly this colour."""
    return visible & np.all(table == np.asarray(colour, np.uint8), axis=1)


def _paper_rgb():
    """The notebook's own paper, as the rest of the project measures ink."""
    if BACKGROUND_COLOR == 'white':
        return (255, 255, 255)
    return tuple(BACKGROUND_COLOR)


def _resident_page(session, held, page_index: int):
    """A whole-page render of this side already held, or None.

    The largest one, because the reduction that follows reads it at a tenth of its size either way, and a
    bigger one loses less.
    """
    held_pages = getattr(session, '_rendered_pages', None) or {}
    side = side_of(session, held)
    best = None
    for (cached_side, cached_page, _scale, masked), picture in held_pages.items():
        if cached_side != side or cached_page != page_index:
            continue
        if best is None or picture.width > best[1].width:
            best = (masked, picture)
    return None if best is None else best[1]


def _remember_the_paper(session, held, page_index: int, picture) -> None:
    """Fill the paper colour from a whole-page picture, if it is not known yet.

    The same reduction `_paper_of` uses, so the value does not depend on which route reached it; and it is
    filled here, where the picture exists, so the call inside `_rendered` finds it instead of rendering a
    second time. Measured on the user's pair: that second render is 172 ms against 1.3 ms for the reduction.
    """
    cache = getattr(session, '_papers', None)
    if cache is None:
        cache = session._papers = {}
    key = (side_of(session, held), page_index)
    if key in cache or picture is None:
        return
    from xp.core.imaging import imagediff

    small = picture.resize((max(1, picture.width // 10), max(1, picture.height // 10)), Image.BOX)
    cache[key] = imagediff.paper_of(np.asarray(small))


def _paper_of(session, held, page_index):
    """The picture side's paper colour, worked out once per page.

    Not assumed to be white. A scan's paper is cream and a photograph's is
    grey, and `imagediff.paper_of` exists because assuming white made the
    whole page ink. It is asked of a small whole-page render rather than of
    the window being drawn, because a window zoomed into a stroke is mostly
    ink and its most common colour is the ink, not the paper.
    """
    cache = getattr(session, '_papers', None)
    if cache is None:
        cache = session._papers = {}
    key = (side_of(session, held), page_index)
    if key not in cache:
        from xp.core.imaging import imagediff

        # **The page that is already resident answers it.** The tenth-scale render is the one that builds
        # PyMuPDF's display list -- measured, 172 ms against 72 ms for the full page -- and the full page is
        # in hand by the time this is asked (`_picture_edit_view` renders it first). Its own tenth-scale
        # reduction gives the same colour on the user's page and on gradients, cream and a vignette; the ink masks
        # that follow are identical. The slow route stays for a caller with nothing resident.
        resident = _resident_page(session, held, page_index)
        if resident is not None:
            small = resident.resize((max(1, resident.width // 10),
                                     max(1, resident.height // 10)), Image.BOX)
        else:
            # The fallback asks for the same resolution the pipeline renders at, a tenth of it -- the view's own
            # scale is not what the paper is a property of.
            small = _rendered(session, held, page_index, _render_scale_of(session) * 0.1,
                              mask_frame=False)
        cache[key] = imagediff.paper_of(np.asarray(small))
    return cache[key]


def _stroke_ink(session, page_index, at, scale, window, tables):
    """One stroke side's ink as a mask: what is there, not what was decided.

    Every stroke, including ones a decision has dropped, because this is the
    side's *content* and it is being compared with the other side's content.
    Whether it is shown is a separate question and `_shown_in_window` answers
    it -- a decision about a stroke side in a merge that has a picture side is
    recorded as an area like every other decision, so dropping a stroke here
    as well would count it twice and lose the common ink it shares with the
    picture.

    Painted onto a colour nothing is ever drawn in, so "ink" is "this pixel is
    not the canvas" -- which is true of an anti-aliased edge as well, and an
    edge is where a difference shows.
    """
    from xp.render import raster
    arrays = _page_arrays(session, page_index)[at]
    # The mask is "this pixel is not the canvas colour", so one colour for every stroke
    # gives exactly the same answer as the per-stroke table -- and the table would be a
    # matching pass, which is the thing this branch no longer pays for.
    table = np.zeros((len(arrays[2]), 3), np.uint8)
    height, width = _blank_size(session, page_index, scale, window)
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[:, :] = canvas_ink
    if not len(arrays[2]):
        return np.zeros((height, width), bool)
    # **The window's strokes come from the cell index.** `_cull` tests the box of every
    # stroke on the page, so a brush segment paid for the page (measured: P50 81.0 ms at
    # 40,000 strokes against the piece painter's 50.3). The index answers the same question
    # for the cost of the rectangle -- "the answer costs what is there rather than what the
    # page holds" -- and this is the same index the piece painter has always used.
    if window is None:
        wanted = _cull(arrays, scale, window)
    else:
        held = session.layers_for_page(page_index)[LEFT if at == 0 else RIGHT]
        left, top, width, height = window
        pieces = held.pieces_crossing(left / scale, top / scale,
                                      (left + width) / scale,
                                      (top + height) / scale)
        wanted = np.zeros(len(arrays[2]), bool)
        if len(pieces):
            origins = np.asarray(held.origin[:held._many], dtype=np.int64)
            chosen = np.asarray(pieces, dtype=np.int64)
            chosen = chosen[(chosen >= 0) & (chosen < len(origins))]
            wanted[origins[chosen]] = True
    if not wanted.any():
        return np.zeros((height, width), bool)
    # **The card first, the processor as the definition.** `views._draw` has preferred the card for this
    # same call for a long time, and `tests/test_gpu_render.py` holds it to being bit-identical to
    # `paint_flat` -- measured on the user's page: 158.1 ms against 5.4 ms, 0 of 3,525,945 values different.
    # `_stroke_ink` was the last place still painting strokes on the processor, which is why the card
    # looked idle. `XP_CPU_PAINT=1` still forces the processor; any refusal falls back to it.
    painted = False
    card = _the_card()
    if card is not None:
        try:
            card.hold(_page_key(session, page_index, arrays), arrays[0], arrays[1], arrays[2],
                      arrays[3], arrays[4], table)
            canvas = np.asarray(card.paint(canvas, scale, _offset(window), wanted, colours=table))
            painted = True
        except Exception:
            # The fallback below is the definition of what the drawing looks like.
            painted = False
    if not painted:
        raster.paint_flat(canvas, arrays[0], arrays[1], arrays[2], arrays[3],
                          arrays[4], table, wanted=wanted, scale=scale,
                          offset=_offset(window))
    return np.any(canvas != canvas_ink, axis=2)


canvas_ink = np.array([0, 255, 0], np.uint8)   # never a diagnostic colour


def _blank_size(session, page_index, scale, window):
    if window is None:
        width, height = page_pixels(session, page_index, scale)
        return height, width
    _left, _top, width, height = window
    return max(1, int(round(height))), max(1, int(round(width)))


def _asked_in_window(session, page_index, side, which, scale, window, shape):
    """One of a picture side's area masks, cropped to the window being drawn.

    The brush records in the merged page's own pixels at the session's scale,
    which is the space the brush works in; a view is drawn at the zoom and
    offset of the window. Nearest, because a pixel was taken or it was not,
    and half a taken pixel is not a thing.
    """
    mask = session.decisions(page_index).taken.get(side) if which == 'taken' \
        else session.decisions(page_index).cleared.get(side)
    if mask is None:
        return None
    height, width = shape
    if window is None and mask.shape == shape:
        return mask
    ratio = (scale / session.scale) or 1.0
    if window is None:
        piece = mask
    else:
        left, top, _w, _h = window
        x0, y0 = int(np.floor(left / ratio)), int(np.floor(top / ratio))
        x1, y1 = int(np.ceil((left + width) / ratio)), int(np.ceil((top + height) / ratio))
        x0, y0 = max(0, x0), max(0, y0)
        piece = mask[y0:y1, x0:x1]
    if piece.shape != shape:
        held = Image.fromarray(piece.astype(np.uint8) * 255)
        piece = np.asarray(held.resize((width, height), Image.NEAREST)) > 0
    return piece


def _owner_in_window(session, page_index, scale, window, shape):
    """The layer plane, cropped to the window being drawn.

    `PageDecisions.owner()` works in the page's own pixels -- the space the
    brush records in, which is the session's scale -- and a view is drawn at
    the zoom and offset of the window. Handing `owner()` the window's shape
    instead was the bug: `out[won] = value` then indexed a full-page plane with
    a window-sized mask, so the moment a picture side had a decision on it, any
    zoomed redraw raised `IndexError` and the page came back blank.

    Crop and resize exactly as `_asked_in_window` does -- nearest, because a
    pixel was taken or it was not, and half a taken pixel is not a thing. The
    values (0..4) survive nearest untouched, so the cheap scaling here is only
    about which plane pixels land in which view pixels.
    """
    from xp.workers.merger.compose import PageDecisions

    decisions = session.decisions(page_index)
    planes = [held for store in (decisions.taken, decisions.cleared)
              for held in store.values() if held is not None]
    if not planes:
        return np.zeros(shape, np.uint8)
    if window is None:
        full = decisions.owner(planes[0].shape)
        if full.shape == tuple(shape):
            return full
        # **The view's scale is not always the session's.** A preview drawn at another scale asked for the whole
        # page and got the session's plane; nearest keeps the value of a place, which is all a decision says.
        height, width = shape
        rows = (np.arange(height) * (full.shape[0] / float(height))).astype(np.intp)
        cols = (np.arange(width) * (full.shape[1] / float(width))).astype(np.intp)
        return full[np.ix_(rows, cols)]
    # **Composed from the window's own pixels.** `owner()` is a pure function of the two
    # masks, so cropping first costs the window and not the page: with the full plane it
    # was a pass over the page on every repaint and every brush segment.
    ratio = (scale / session.scale) or 1.0
    height, width = shape
    left, top, _w, _h = window
    x0 = max(0, int(np.floor(left / ratio)))
    y0 = max(0, int(np.floor(top / ratio)))
    x1 = int(np.ceil((left + width) / ratio))
    y1 = int(np.ceil((top + height) / ratio))
    cropped = PageDecisions(
        kept={},
        taken={side: held[y0:y1, x0:x1]
               for side, held in decisions.taken.items() if held is not None},
        cleared={side: held[y0:y1, x0:x1]
                 for side, held in decisions.cleared.items() if held is not None})
    piece_shape = next(iter(cropped.taken.values())).shape if cropped.taken else \
        next(iter(cropped.cleared.values())).shape
    full = cropped.owner(piece_shape)
    if full.shape != (height, width):
        held = Image.fromarray(full)
        full = np.asarray(held.resize((width, height), Image.NEAREST))
    if window is None and full.shape == tuple(shape):
        return full
    piece = full
    if False:  # the window is already composed above
        left, top, _w, _h = window
        x0, y0 = int(np.floor(left / ratio)), int(np.floor(top / ratio))
        x1, y1 = int(np.ceil((left + width) / ratio)), int(np.ceil((top + height) / ratio))
        piece = full[max(0, y0):y1, max(0, x0):x1]
    if piece.shape != (height, width):
        held = Image.fromarray(np.ascontiguousarray(piece, np.uint8))
        piece = np.asarray(held.resize((width, height), Image.NEAREST), np.uint8)
    return piece


def _shown_in_window(session, page_index, side, scale, window, shape):
    """Where this side's content is shown: taken, or never refused.

    Nothing recorded means everything shown, which is the case for every
    merge of two notebooks and for every page a PDF side has not been
    brushed on. Once the user has decided about an area, the side the user decided
    against gives way there.
    """
    taken = _asked_in_window(session, page_index, side, 'taken', scale, window, shape)
    cleared = _asked_in_window(session, page_index, side, 'cleared', scale, window, shape)
    if taken is None and cleared is None:
        return np.ones(shape, bool)
    # **A side is hidden only where it was refused.** This started from an empty mask
    # and OR'd `taken` into it, so a side that had been taken somewhere and never
    # refused anywhere was shown *only* where it was taken: one TAKE_RIGHT gesture left
    # the right side's ink outside the brushed band invisible (measured: 4,041 shown
    # pixels, the capsule, against a whole page of ink).
    shown = np.ones(shape, bool) if cleared is None else ~cleared
    if taken is not None:
        shown |= taken
    return shown


def _ink_on_the_card(rgb, paper):
    """`imagediff.ink_of`, asked with whichever array library is in use.

    The same threshold channel by channel and the same `INK_DELTA`, so a
    picture side's ink is the same question asked of the same pixels wherever
    it is asked. `ink_of` sums `cv2.absdiff` of three `uint8` channels into
    `uint16`; this does the same arithmetic, and `tests/test_gpu_render.py`
    holds the two to giving the same answer on a real rendered page.

    It is worth moving because this is the one per-pixel pass over the PDF's
    whole rendered window, and the window is where the PDF layer is composited.
    """
    from xp.gpu import backend as gpu
    from xp.core.imaging import imagediff

    xp = gpu.xp
    image = xp.asarray(rgb)
    total = xp.zeros(image.shape[:2], xp.uint16)
    for channel, value in enumerate(paper):
        total += xp.abs(image[:, :, channel].astype(xp.int16)
                        - int(value)).astype(xp.uint16)
    return total > imagediff.INK_DELTA


def _ink_distance(rgb, paper):
    """How far every pixel is from the paper, before anything thresholds it.

    `_ink_on_the_card` computes exactly this and then throws it away with
    `> INK_DELTA`. **That one comparison is where the page stops being as good
    as its source.** Measured on the user's pair at the same scale and window: XP's
    own raw right view carries 188 grey levels and the raw left view 119,
    while the editing view built from the same pixels carries **4**. The PDF
    arrives antialiased and the composition binarises it.

    The user's rule for it, 2026-11-01: *"the classification should decide which
    layer a region belongs to; it should not force the whole layer's visible
    content to degrade into a coarse binary image."* So the distance is kept
    as well as the threshold: the threshold still decides class membership --
    nothing about black/red/blue moves -- and the distance says how much of
    the pixel the ink actually covers.
    """
    from xp.gpu import backend as gpu

    xp = gpu.xp
    image = xp.asarray(rgb)
    total = xp.zeros(image.shape[:2], xp.uint16)
    for channel, value in enumerate(paper):
        total += xp.abs(image[:, :, channel].astype(xp.int16)
                        - int(value)).astype(xp.uint16)
    return total


def _ink_full_of(session, held, page_index, distance):
    """What a *fully* inked pixel measures on this page, worked out once.

    Coverage is the distance from paper over the distance a solid pixel has,
    and that denominator is a property of the page's ink -- not of the window.
    Taking it per window would make an edge's grey depend on what else is on
    screen, so a pan would change the colour of a stroke the user is not touching.

    The 95th percentile of the inked pixels rather than the maximum: one
    stray dark speck would otherwise set the scale for the whole page and
    wash every real edge out.
    """
    from xp.gpu import backend as gpu

    cache = getattr(session, '_ink_fulls', None)
    if cache is None:
        cache = session._ink_fulls = {}
    key = (side_of(session, held), page_index)
    if key in cache:
        return cache[key]
    from xp.core.imaging import imagediff

    try:
        flat = np.asarray(gpu.to_host(distance)).reshape(-1)
        inked = flat[flat > imagediff.INK_DELTA]
        full = float(np.percentile(inked, 95)) if inked.size else 0.0
    except Exception:
        full = 0.0
    # Below the threshold there is no scale to speak of; fall back to it so
    # the division is never by something smaller than "not paper".
    full = max(full, float(imagediff.INK_DELTA) * 2.0)
    cache[key] = full
    return full


def _compose_three(shape, left_ink, right_ink, shown_left, shown_right,
                   owner=None):
    """The three colours, composited from two ink layers and two visibility masks.

    This is the step the objective calls compositing: two layers, what each is
    allowed to show, and the palette, per pixel. It runs on the card when there
    is one -- `gpu.xp` is whichever array library is in use, so this is one
    path rather than two that can drift -- and `gpu.to_host` is what comes back
    to become the window's image.

    The rule, unchanged from the numpy version it replaces: common ink stays as
    long as either side is still using it; a side's own ink is drawn only where
    that side has not been decided against.

    **Both means exactly both.** A pixel is Common when this pixel of the left
    image has content *and* this pixel of the right image has content -- nothing
    else. It used to grow each side's ink by two pixels first, so content that
    merely sat near the other side's came out black: the comparison said
    "Common" about two places that never overlapped. A tolerance cannot tell a
    misaligned page from ink that really is two pixels apart, so it answers the
    wrong question loudly. If the pages are off, the fit is what moves them;
    the classifier is not allowed to paint the error black.
    """
    from xp.gpu import backend as gpu
    xp = gpu.xp
    canvas = xp.empty(tuple(shape) + (3,), xp.uint8)
    canvas[:, :] = xp.asarray(_paper_rgb(), xp.uint8)
    both = left_ink & right_ink
    left_own = left_ink & ~right_ink
    right_own = right_ink & ~left_ink
    canvas[both & (shown_left | shown_right)] = xp.asarray(OVERLAP, xp.uint8)
    canvas[left_own & shown_left] = xp.asarray(LEFT_ONLY, xp.uint8)
    canvas[right_own & shown_right] = xp.asarray(RIGHT_ONLY, xp.uint8)
    # The layer relation: a side that won a pixel confirmed the ink only it has,
    # so that ink is green rather than red or blue. Common ink stays black
    # whatever the owner says -- both sides had it, so there was nothing to
    # decide there. 3 ("neither") needs no case: `shown` is already false.
    if owner is not None:
        # `owner` comes from PageDecisions, which lives on the host; the ink
        # layers are on the card. Comparing across the two is a TypeError, so
        # it goes to whichever library is in use, the same way the palette does.
        owner = xp.asarray(owner)
        won_left = ((owner == 1) | (owner == 4)) & left_own
        won_right = ((owner == 2) | (owner == 4)) & right_own
        canvas[won_left | won_right] = xp.asarray(CONFIRMED, xp.uint8)
    return np.asarray(gpu.to_host(canvas), np.uint8)


def ink_layer(session, page_index, side, scale, window, coverage=False):
    """One side's content as an ink mask, at any scale. Decided by nothing.

    This is the question "where does this side have ink", asked of whichever
    kind of side it is: painted strokes, or a rendered picture thresholded
    against its own paper. It is the raw material of the classification and of
    the residual rule in `_picture_edit_view`, so it lives on its own instead
    of inside one caller's loop.

    Always on the host. The threshold may run on the card, but both callers
    want a mask they can compare and sample, and a bool plane of the page is
    small next to the picture it came from.
    """
    from xp.gpu import backend as gpu

    held = picture_side(session, side, page_index)
    if held is None:
        mask = _stroke_ink(session, page_index, 0 if side == LEFT else 1,
                           scale, window, None)
        # **A stroke side has no source raster to preserve, and no coverage
        # to hand back.** `paint_flat` gives none, so the honest answer is
        # None and the window's own supersampling is what softens these edges
        # (`MergeApp._supersample`). Returning `mask.astype(float32)` instead
        # cost a full-window float copy per side per frame for an array whose
        # every value was 0 or 1 -- measured, `test_brush_is_local` went from
        # passing to a P95 of 153 ms against a 100 ms target, on a page with
        # no picture side at all. A picture side is the opposite case: the
        # coverage is already in its pixels and all that was needed was to
        # stop throwing it away.
        return (mask, None) if coverage else mask
    drawn = _rendered(session, held, page_index, scale, window)
    # The picture is here; the paper is read off it before anything asks the slow way.
    _remember_the_paper(session, held, page_index, drawn)
    from xp.core.imaging import imagediff

    distance = _ink_distance(np.asarray(drawn.convert('RGB'), np.uint8),
                             _paper_of(session, held, page_index))
    mask = np.asarray(gpu.to_host(distance > imagediff.INK_DELTA), bool)
    if coverage:
        full = _ink_full_of(session, held, page_index, distance)
        soft = np.asarray(gpu.to_host(distance), np.float32) / max(full, 1.0)
        return mask, np.clip(soft, 0.0, 1.0)
    return mask


def classification_of(session, page_index):
    """This page's three classes, decided once, at the session's DPI.

    Held per page against the ink it was decided from, so zooming and panning
    read it and never rebuild it. The DPI is a property of the session -- see
    `MergeSession.classify_dpi` -- and `MergeSession.set_classify_dpi` is what
    changes it; every DPI change clears this cache through the same key.
    """
    from xp.workers.merger import classification as _classification

    # **The whole-page scale, like every other whole-page path.** Asking the raw DPI here is what made 8000 dpi a
    # 908-megapixel plane (43,485 x 20,869, 67 seconds on the user's pair) while the rest of the pipeline was capped at 40.
    scale_of = getattr(session, 'whole_page_scale', None)
    if scale_of is not None:
        scale = float(scale_of(page_index))
    else:
        scale = _classification.pixels_per_unit(
            getattr(session, 'classify_dpi', _classification.DEFAULT_DPI))
    strokes = session.strokes_for_page(page_index)
    cache = getattr(session, '_classification_cache', None)
    if cache is None:
        cache = session._classification_cache = {}
    key = (page_index, round(scale, 6), id(strokes),
           getattr(session, '_layer_epoch', 0))
    hit = cache.get(key)
    if hit is not None:
        return hit

    # **Distinct pictures, not a guessed one.** `ink_layer` renders a picture
    # side at this scale; the parts of a page that nobody has looked at yet are
    # exactly what this first pass is for.
    left = ink_layer(session, page_index, LEFT, scale, None)
    right = ink_layer(session, page_index, RIGHT, scale, None)
    made = _classification.build(left, right, scale, session.page_size(page_index))
    cache.clear()            # one page's answer is all a window is showing
    cache[key] = made
    return made


def _compose_classes(shape, classes, left_ink, right_ink, shown_left, shown_right,
                     owner=None, left_cover=None, right_cover=None):
    """The three colours from one label per pixel, plus the ink at this zoom.

    `classes` is the page's classification sampled onto this view (values as in
    `classification.py`). The inks are here for one reason: a view can be drawn
    finer than the classification was decided, and ink that exists only at that
    finer resolution sits inside a PAPER cell. Those pixels -- and only those --
    take the class their own side's ink implies, which is the same rule, asked
    of the detail the fixed DPI could not hold. They are never asked again at a
    different zoom, because the classification is not re-derived here.
    """
    from xp.gpu import backend as gpu
    xp = gpu.xp
    classes = xp.asarray(classes)
    canvas = xp.empty(tuple(shape) + (3,), xp.uint8)
    canvas[:, :] = xp.asarray(_paper_rgb(), xp.uint8)

    # **The ink at this view's scale decides the shape; the class decides only the colour.**
    #
    # A class plane cannot be finer than the resolution it was measured at, so painting every cell it
    # labels means painting whole blocks once the view is drawn coarser than the measurement: at 2x a
    # stroke two pixels wide is covered by a label cell four pixels wide, and the drawn piece comes
    # out as a NEAREST stretch of the 1x page -- measured, exactly that, which is what made pointing
    # the editing view at the region map look like a map bug for three rounds. It was this line.
    #
    # The ink is drawn at the view's own scale, so it has the edges; the class only says which of the
    # four colours a covered pixel takes. The user's rule, verbatim: *"区域决定来源，素材决定细节"*.
    ink = left_ink | right_ink
    common = (classes == 1) & ink
    left_only = (classes == 2) & ink
    right_only = (classes == 3) & ink
    # Ink that sits in a PAPER cell is ink the comparison did not resolve -- drawn at a finer scale
    # than it was classified -- so it takes the colour its own side implies. Nothing else may.
    residual = (classes == 0) & ink
    common = common | (residual & left_ink & right_ink)
    left_only = left_only | (residual & left_ink & ~right_ink)
    right_only = right_only | (residual & right_ink & ~left_ink)

    # **Only when a side really has coverage to give.** Two stroke sides have
    # none -- their edges come from the window's supersampling -- so a page
    # without a picture in it takes the plain path and pays nothing for a
    # blend that would be between 0 and 1 and nothing else.
    if left_cover is None and right_cover is None:
        canvas[common & (shown_left | shown_right)] = xp.asarray(OVERLAP, xp.uint8)
        canvas[left_only & shown_left] = xp.asarray(LEFT_ONLY, xp.uint8)
        canvas[right_only & shown_right] = xp.asarray(RIGHT_ONLY, xp.uint8)
        # The plain path has no coverage to blend, so its green is a stamp -- and only there.
        if owner is not None:
            owner = xp.asarray(owner)
            won_left = ((owner == 1) | (owner == 4)) & left_only
            won_right = ((owner == 2) | (owner == 4)) & right_only
            canvas[won_left | won_right] = xp.asarray(CONFIRMED, xp.uint8)
    else:
        # **Mixed by how much of the pixel the ink covers, not stamped on.**
        # Assigning a flat colour wherever a boolean is true is what turned a
        # source carrying 188 grey levels into a view carrying 4. The class
        # still decides *which* colour -- the booleans above are untouched and
        # nothing about black/red/blue moves -- and the coverage decides how
        # much of it lands, which is what an antialiased edge is.
        # **One blend, not three.** The first version mixed each class in its
        # own pass -- three float copies of the window and three `where`s over
        # three channels -- and `test_brush_is_local` caught it at once: P95
        # input-to-picture 129.6 ms against a 100 ms target, which is a
        # regression and not noise. The colour and the alpha are gathered into
        # one plane each and the window is blended once.
        tint = xp.empty(tuple(shape) + (3,), xp.uint8)
        tint[:, :] = xp.asarray(_paper_rgb(), xp.uint8)
        # **uint8 alpha and integer arithmetic**, which is what `gpu.wash`
        # already uses for the brush -- `(over * a + base * (255 - a) + 127)
        # // 255`. `xp` here is cupy on the user's machine, and the float version of
        # this allocated three float32 copies of the window on the card: on a
        # full-screen window that is about 165 MB of device memory for a
        # blend whose inputs are bytes.
        alpha = xp.zeros(tuple(shape), xp.float32)

        # A side with no coverage of its own is solid where it has ink: that
        # is what the plain path meant, written as an alpha so the two kinds
        # of side can be mixed on one page.
        solid_l = left_cover is None
        solid_r = right_cover is None
        lc = xp.asarray(left_ink, xp.float32) if solid_l else left_cover
        rc = xp.asarray(right_ink, xp.float32) if solid_r else right_cover

        here = common & (shown_left | shown_right)
        tint[here] = xp.asarray(OVERLAP, xp.uint8)
        alpha[here] = xp.maximum(lc, rc)[here]
        here = left_only & shown_left
        tint[here] = xp.asarray(LEFT_ONLY, xp.uint8)
        alpha[here] = lc[here]
        here = right_only & shown_right
        tint[here] = xp.asarray(RIGHT_ONLY, xp.uint8)
        alpha[here] = rc[here]

        # **Green is blended like every other colour, not stamped on afterwards.** It used to be
        # assigned into the finished canvas below -- `canvas[won] = CONFIRMED` -- which put a solid
        # green pixel wherever a side had won, however little of that pixel its ink covered: an
        # antialiased edge turned back into a staircase, on the one colour the user uses to say "this side's
        # ink is the one I keep". The class still decides *which* colour and the coverage still
        # decides how much of it lands, exactly as for black, red and blue.
        if owner is not None:
            owner = xp.asarray(owner)
            won = (((owner == 1) | (owner == 4)) & left_only & shown_left
                   | ((owner == 2) | (owner == 4)) & right_only & shown_right)
            if bool(xp.any(won)):
                tint[won] = xp.asarray(CONFIRMED, xp.uint8)
                alpha[won] = xp.maximum(xp.maximum(lc, rc), alpha)[won]

        a8 = (xp.clip(alpha, 0.0, 1.0) * 255.0 + 0.5).astype(xp.uint8)[..., None]
        canvas = ((canvas.astype(xp.uint16) * (255 - a8).astype(xp.uint16)
                   + tint.astype(xp.uint16) * a8.astype(xp.uint16)
                   + 127) // 255).astype(xp.uint8)
    return np.asarray(gpu.to_host(canvas), np.uint8)


def _picture_edit_view(session, page_index, scale, window, tables):
    """The same three colours, asked of the ink instead of the strokes.

    A PDF has ink and no strokes, so "which stroke matches which" cannot
    answer whether the two sides agree: there is nothing on one side to match
    with, every stroke on the other comes back unique, and the page is red
    from edge to edge -- which is what it was.

    Between a stroke side and a picture side the question is asked of the ink,
    because ink is the only thing the two have in common. Both sides' content is
    compared once per page at the session's DPI (`classification.py`), and then
    a place is black where both have ink there, red where only the left, blue
    where only the right. This is the same rule the stroke view follows,
    evaluated per pixel rather than per stroke, and it is the only rule either
    of them uses.

    **The comparison is not made here.** It is made once, at the load, and this
    function samples that answer onto the window it is drawing. It used to
    derive the classes from two rasters made at the current zoom, which put the
    question in the draw call: measured on the user's pair, only 47.2 % of the page's
    ink points kept their class between 0.5x and 4x, and the same line came out
    red at one zoom and black at another. Now a zoom changes how wide a boundary
    is drawn and nothing else.

    A decision does not change what the ink *is* -- it decides which side's
    version of it is used. So the three classes are found first, from both
    sides' whole content, and then a class is shown only if the side it
    belongs to has not been decided against. That is what makes "take right"
    show the right's page and drop the left's own ink without also erasing
    everything the two agree about.

    Both the PDF layer's ink and the composition are asked of `gpu`, so when
    there is a card they happen on it: the layer is uploaded, thresholded and
    composited there, and only the finished window comes back.
    """
    from xp.gpu import backend as gpu
    height, width = _blank_size(session, page_index, scale, window)
    shape = (height, width)
    # **The window's inks, which are this zoom's detail.** They are not what
    # decides a class any more: the classes come from the page's classification,
    # decided once at `session.classify_dpi` and sampled here. The inks are what
    # a class is drawn *on*, so a page at 4x keeps its sharp edges while its
    # black/red/blue boundaries stay where the classification put them.
    inks = {}
    covers = {}
    for side in (LEFT, RIGHT):
        mask, cover = ink_layer(session, page_index, side, scale, window,
                                coverage=True)
        # **An ink that came back whole where a window was asked for.** Intermittent, and older than
        # the whole-plane cache: `test_brush_is_local` fails with `operands could not be broadcast
        # together with shapes (48, 309) (1889, 2686)` -- one side's mask at the page's own size while
        # everything else in the call is the window. The cause was not found in the time this round
        # had; what is done here is the arithmetic that cannot be wrong for that shape: a page-sized
        # mask cut at the window's corner is the window's mask. Said out loud rather than cropped
        # silently, so the next reader knows it is a repair and not the design.
        if window is not None and tuple(mask.shape) != shape:
            page_w, page_h = page_pixels(session, page_index, scale)
            if tuple(mask.shape) == (page_h, page_w):
                left, top = int(window[0]), int(window[1])
                print('[merge] an ink came back whole for a %dx%d window; cropped (%s)'
                      % (shape[1], shape[0], 'left' if side == LEFT else 'right'), flush=True)
                mask = mask[top:top + shape[0], left:left + shape[1]]
                if cover is not None:       # a stroke side hands back none
                    cover = cover[top:top + shape[0], left:left + shape[1]]
            elif mask.shape[0] >= shape[0] and mask.shape[1] >= shape[1]:
                mask = mask[:shape[0], :shape[1]]
                if cover is not None:
                    cover = cover[:shape[0], :shape[1]]
        inks[side] = gpu.to_device(np.asarray(mask, bool))
        covers[side] = (None if cover is None
                        else gpu.to_device(np.asarray(cover, np.float32)))
    # **The region map is the authority.** Two bitmaps give the content; the map says which of the
    # four states each place takes; the ink at this view's scale gives the shape (see the comment on
    # `ink` inside `_compose_classes`). Nothing below reads `owner()` or `classification` -- those
    # were the old answer, and two answers is the state the user called out.
    held = session.regions(page_index)
    states = held.labels(shape, scale, window)
    classes = np.zeros(states.shape, np.uint8)
    classes[states == 1] = 2          # LEFT    -> left-only
    classes[states == 2] = 3          # RIGHT   -> right-only
    classes[states == 0] = 1          # COMMON  -> the ink both sides have
    # NEITHER (3) stays 0: no ink is drawn there at all.

    # **Green is a decision, read off the map.** A place is decided where its state is not what the
    # comparison alone would say -- and the comparison *is* the map's own base, asked at this view's
    # scale. Nothing stores "decided" separately, so the cue cannot drift from the geometry.
    decided = np.zeros(states.shape, bool)
    if held.base_sampler is not None:
        decided = states != np.asarray(held.base_sampler(shape, scale, window), np.uint8)
    keep = states != 3
    shown = {LEFT: gpu.to_device(keep), RIGHT: gpu.to_device(keep)}
    owner = np.zeros(states.shape, np.uint8)
    owner[decided & (states == 1)] = 1
    owner[decided & (states == 2)] = 2
    owner[decided & (states == 0)] = 4
    return Image.fromarray(
        _compose_classes(shape, classes, inks[LEFT], inks[RIGHT],
                         shown[LEFT], shown[RIGHT],
                         gpu.to_device(owner),
                         left_cover=covers[LEFT], right_cover=covers[RIGHT]),
        'RGB')

def _local_coverage(session, page_index: int):
    """Per side, a boolean per **point**: is the other side's ink under it?

    **This is what makes black local.** The colour of a piece used to come from
    `common_strokes`, which answers "is this stroke common": a stroke crossing
    the other side's writing turned black whole, ends included, and two strokes
    that merely crossed were both called shared. The user's order (2026-10-26): only
    what really coincides is black.

    Cached against the decision stamp and the stroke lists, like the colour map,
    so zooming and panning never recompute it -- only a decision does.
    """
    from xp.workers.merger.matching.common import ink_mask, point_coverage

    # **Keyed on the ink, not on the decisions.** A brush segment bumps the
    # decision stamp, but it does not move a single point: the masks describe
    # where the ink is, and the answer is the same before and after the gesture.
    # Caching on the stamp recomputed both masks on every motion event -- P95 at
    # 4x went from under 50 ms to 378 ms.
    strokes = session.strokes_for_page(page_index)
    held = getattr(session, '_coverage_cache', None)
    if held is None:
        held = session._coverage_cache = {}
    have = held.get(page_index)
    if have is not None and have[0] is strokes:
        return have[1]

    left_strokes, right_strokes = strokes
    width, height = session.page_size(page_index)
    left_mask = ink_mask(left_strokes, width, height)
    right_mask = ink_mask(right_strokes, width, height)
    answer = {'size': (width, height),
              LEFT: right_mask, RIGHT: left_mask}
    held[page_index] = (strokes, answer)
    return answer


def _runs(common_flags):
    """Point ranges where the colour is constant, over **segments**.

    A polyline of `n` points has `n - 1` segments, and a segment counts as
    common when either end of it is on the other side's ink -- so the black is
    never narrower than the overlap. Ranges of one point would draw nothing, and
    that is how the first version lost the ends of every stroke it split.
    """
    if len(common_flags) < 2:
        return []
    segments = [bool(common_flags[i]) or bool(common_flags[i + 1])
                for i in range(len(common_flags) - 1)]
    runs = []
    start = 0
    for index in range(1, len(segments)):
        if segments[index] != segments[start]:
            runs.append((start, index, segments[start]))
            start = index
    runs.append((start, len(segments), segments[start]))
    return runs


def _resampled(xs, ys, step=0.75, most=240):
    """The piece's polyline, walked in short steps.

    **A two-point line is one segment**: asking "is this stroke common" about it
    answers for the whole line, which is how two crossing strokes turned into
    one black cross. Walking the segment and asking at each step is what makes
    the answer local. A step of three-quarters of a page unit with a cap of 240
    samples keeps a very long traced chain from being re-sampled for ever.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if len(xs) < 2:
        return xs, ys
    lengths = np.hypot(np.diff(xs), np.diff(ys))
    total = float(lengths.sum())
    if total <= step:
        return xs, ys
    count = int(min(most, max(2, round(total / step))))
    along = np.linspace(0.0, total, count)
    walked = np.concatenate(([0.0], np.cumsum(lengths)))
    return np.interp(along, walked, xs), np.interp(along, walked, ys)


def _undecided_side_runs(session, page_index, side, one, pieces, coverage):
    """The paint order for undecided, exclusive pieces: black where they coincide.

    Returns `[(colour, [(xs, ys, width), ...]), ...]` for the pieces given, and
    the set of pieces handled here (the caller draws the rest the usual way).
    """
    from xp.workers.merger.matching.common import point_coverage

    own_only = LEFT_ONLY if side == LEFT else RIGHT_ONLY
    width, height = coverage['size']
    other = coverage[side]
    handled = set()
    out = []
    colours = None
    try:
        colours = session.layers_for_page(page_index)[side].arrays()[5]
    except Exception:
        colours = None
    for piece in pieces:
        start = int(one.starts[piece])
        count = int(one.counts[piece])
        if count < 2:
            continue
        xs, ys = _resampled(one.xs[start:start + count],
                            one.ys[start:start + count])
        flags = point_coverage(np.column_stack((xs, ys)), other, width, height)
        if not flags.any():
            # **Nothing of this piece is on the other side, so it cannot be black.**
            # `common_strokes` decides a colour per *stroke*, and a stroke that touches the other
            # side anywhere was painted black everywhere it went -- the black that grew along a
            # stroke in the user's picture. Local means local: a piece with no coverage is its own side's
            # ink, and it is drawn in that side's colour.
            if colours is not None and piece < len(colours) and tuple(
                    int(v) for v in colours[piece]) == OVERLAP:
                handled.add(piece)
                out.append((piece, own_only,
                            (xs, ys, float(one.widths[piece]))))
            continue
        if flags.all():
            continue                      # `common_strokes` already made it black
        handled.add(piece)
        stroke_width = float(one.widths[piece])
        for first, last, common in _runs(flags):
            # `first`/`last` are point indexes: the run covers points
            # first..last inclusive, which is what `last + 1` slices. The
            # colour comes from the run's own segment flag -- asking `flags[first]`
            # put every boundary run on the wrong side, because the first point
            # of a common run can itself sit just off the ink.
            if last <= first:
                continue
            colour = OVERLAP if common else own_only
            out.append((piece, colour,
                        (xs[first:last + 1], ys[first:last + 1], stroke_width)))
    return out, handled


def _paint_runs(image, runs, scale, offset):
    """Draw the black/red runs of undecided pieces, through **the same painter**.

    Through `raster.paint_flat`, not `draw_strokes`: the full-page view and the
    incremental patch must be the same picture pixel for pixel, and the patch
    paints with `paint_flat`. Drawing the runs with the other rasterizer left 58
    pixels different at the runs' edges -- `test_brush_is_local` caught it.
    """
    from xp.render import raster

    if not runs:
        return
    # **Black under the differences, always.** The two callers collect their
    # runs in different orders (the full view per colour pass, the patch per
    # damaged piece), and a blue run painted over a black one in one path and
    # under it in the other left 58 pixels different. Sorting by the paint order
    # makes the two paths the same picture.
    order = {tuple(colour): at for at, colour in enumerate(PAINT_ORDER)}
    runs = sorted(runs, key=lambda run: order.get(tuple(run[0]), len(PAINT_ORDER)))
    xs, ys, starts, counts, widths, colours = [], [], [], [], [], []
    at = 0
    for colour, (run_xs, run_ys, width) in runs:
        starts.append(at)
        counts.append(len(run_xs))
        widths.append(width)
        colours.append(tuple(colour))
        xs.extend(run_xs.tolist())
        ys.extend(run_ys.tolist())
        at += len(run_xs)
    pixels = np.array(image, np.uint8)
    raster.paint_flat(pixels, np.asarray(xs, np.float64), np.asarray(ys, np.float64),
                      np.asarray(starts, np.int64), np.asarray(counts, np.int64),
                      np.asarray(widths, np.float64),
                      np.asarray(colours, np.uint8),
                      wanted=np.ones(len(starts), bool), scale=scale,
                      offset=offset)
    image.paste(Image.fromarray(pixels, 'RGB'))


def _local_runs(session, page_index: int):
    """`(handled, runs)`: the pieces whose colour is local, and the runs to paint.

    **Computed once for the page, not per painter.** The full view and the
    incremental patch must be the same picture pixel for pixel; when each of
    them decided for itself which pieces were split, they disagreed by 59 pixels
    at a run's edge -- `test_brush_is_local` caught it. Cached against the ink
    (like the coverage), so a gesture does not recompute it.
    """
    held = getattr(session, '_local_runs_cache', None)
    if held is None:
        held = session._local_runs_cache = {}
    strokes = session.strokes_for_page(page_index)
    have = held.get(page_index)
    if have is not None and have[0] is strokes:
        return have[1]

    coverage = _local_coverage(session, page_index)
    arrays = session.layers_for_page(page_index)
    handled = {}
    runs = []
    for side in (LEFT, RIGHT):
        one = arrays[side]
        pieces = np.arange(one._many)
        side_runs, side_handled = _undecided_side_runs(
            session, page_index, side, one, pieces, coverage)
        handled[side] = side_handled
        runs.extend((side, piece, colour, geometry)
                    for piece, colour, geometry in side_runs)
    # **The boxes, as arrays.** `layer_view` filters the runs to its window on
    # every redraw, and a redraw is a brush motion event: a Python loop over
    # every run on the page cost 20-30 ms of the 67 ms a 4x repair took.
    boxes = np.zeros((len(runs), 4), dtype=np.float64)
    pieces = np.full(len(runs), -1, dtype=np.int64)
    sides = np.zeros(len(runs), dtype=np.int8)          # 0 = left, 1 = right
    for at, (side, piece, _colour, (xs, ys, _width)) in enumerate(runs):
        if len(xs):
            boxes[at] = (xs.min(), xs.max(), ys.min(), ys.max())
        else:
            boxes[at] = (0.0, -1.0, 0.0, -1.0)
        sides[at] = 0 if side == LEFT else 1
        pieces[at] = -1 if piece is None else int(piece)
    answer = (handled, runs,
              {'boxes': boxes, 'pieces': pieces,
               'left': sides == 0, 'right': sides == 1})
    held[page_index] = (strokes, answer)
    return answer


def layer_view(session, page_index: int, scale: float, window=None,
               by_side=None) -> Image.Image:
    """The editing view drawn from the page's **layers**, piece by piece.

    The three base layers -- common, left-only, right-only -- are the ink
    itself and do not move when the user brushes; what a gesture changes is the
    composition state of the pieces it touched, and `layers.colour_of` turns
    (layer, state) into the colour. See `layers.py` for why selection is a
    layer operation and not a coverage threshold.

    Nothing here is new drawing code. A piece is a `(start, count)` slice of
    the very same flat point arrays `arrays_of` built, so this is
    `edit_view`'s own three-pass loop over the same `_draw`, with one colour
    per piece instead of one per stroke. Splitting a stroke copies no points.
    """
    from xp.workers.merger import layers as _layers

    image = _blank(session, page_index, scale, window)
    held = session.layers_for_page(page_index)
    visible = {side: held[side].visible() for side in (LEFT, RIGHT)}
    # **`arrays()` once per side, not once per colour pass.** It is a view over
    # the pieces, rebuilt when a gesture has moved them, and asking for it eight
    # times a frame is eight chances to rebuild it.
    packed = {side: held[side].arrays() for side in (LEFT, RIGHT)}
    _handled, all_runs, run_index = _local_runs(session, page_index)
    # **A run belongs to a piece, and a piece can be decided or hidden.** Drawing
    # every run regardless put black back over ink that `Neither` had just
    # removed -- `test_the_colour_is_local` and `test_view_repair` both caught
    # it. The runs are filtered by the piece's current visibility, which is the
    # same condition the per-colour passes use.
    local_runs = []
    if all_runs:
        # **Only the runs whose piece is still shown, and that this window can
        # see**, in numpy with no loop over the page's runs: this is on every
        # motion event, and a Python pass over ~20k runs was most of the cost of
        # a 4x repair.
        boxes = run_index['boxes']
        pieces = run_index['pieces']
        keep = np.zeros(len(all_runs), bool)
        for side, which in ((LEFT, 'left'), (RIGHT, 'right')):
            here = run_index[which]
            index = pieces[here]
            mask = visible[side]
            ok = (index >= 0) & (index < len(mask))
            good = np.zeros(len(index), bool)
            if ok.any():
                good[ok] = mask[index[ok]]
            keep[here] = good
        if window is not None:
            box = (window[0] / scale, window[1] / scale,
                   (window[0] + window[2]) / scale,
                   (window[1] + window[3]) / scale)
            keep &= ~((boxes[:, 1] < box[0]) | (boxes[:, 0] > box[2])
                      | (boxes[:, 3] < box[1]) | (boxes[:, 2] > box[3]))
        local_runs = [(all_runs[at][2], all_runs[at][3])
                      for at in np.flatnonzero(keep)]
    for colour in PAINT_ORDER:
        for side in (LEFT, RIGHT):
            one = held[side]
            arrays = packed[side]
            if not len(arrays[2]):
                continue
            side_visible = visible[side]
            if by_side is not None:
                # **The caller already knows which pieces are in the window**
                # (the page's cell index answered it), so the cull over the
                # whole page is skipped. The drawing below is the same code:
                # one painter, two ways to find the pieces it draws.
                picked = by_side.get(side, {}).get(tuple(colour)) or []
                wanted = np.zeros(len(arrays[5]), bool)
                if len(picked):
                    wanted[np.asarray(picked, np.int64)] = True
                # **The same conditions as the full cull**: the index answers
                # "which pieces are in the rectangle", not "which may be drawn".
                # Without the visibility test the patch put decided ink back --
                # `test_view_repair` measured 90,459 pixels of drift.
                wanted &= side_visible
            else:
                same = (arrays[5] == np.array(colour, np.uint8)).all(axis=1)
                wanted = _cull(arrays, scale, window) & side_visible & same
            if not wanted.any():
                continue
            # One pass per colour, each building on the last, so the common
            # ink stays under the differences rather than replacing them --
            # the same reason `edit_view` paints in this order.
            _draw(session, page_index, image, arrays, arrays[5], wanted,
                  scale, window)
    _paint_runs(image, local_runs, scale, _offset(window))
    return image


def layer_patch(session, page_index: int, scale: float, window):
    """One damaged rectangle of a layered page, drawn from its pieces alone.

    **The page's cell index answers which pieces are in the rectangle; the
    drawing is `layer_view`'s.** Two painters disagreed by a few pixels at a
    run's edge once the colour became local, and `test_brush_is_local` compares
    the repaired rectangle with a fresh render pixel for pixel. This is one
    painter with two ways to find the pieces it draws: a full cull for the whole
    page, the cell index for a brush-sized rectangle.
    """
    held = session.layers_for_page(page_index)
    left, top, width, height = window
    by_side = {side: held[side].shown_crossing(
                   left / scale, top / scale,
                   (left + width) / scale, (top + height) / scale)
               for side in (LEFT, RIGHT)}
    # **And the pieces that own a run in this window.** The cell index answers
    # "whose box crosses the rectangle"; a run belongs to a piece whose box may
    # not, and a run drawn without its piece leaves the piece's own colour
    # missing at its edges -- 96 pixels of drift at a stroke's left edge.
    _handled, all_runs, _run_index = _local_runs(session, page_index)
    box = (left / scale, top / scale, (left + width) / scale,
           (top + height) / scale)
    for side, piece, _colour, (xs, ys, _w) in all_runs:
        if piece is None or len(xs) == 0:
            continue
        if (xs.max() < box[0] or xs.min() > box[2]
                or ys.max() < box[1] or ys.min() > box[3]):
            continue
        arrays = held[side].arrays()
        if piece >= len(arrays[5]):
            continue
        colour = tuple(int(v) for v in arrays[5][piece])
        bucket = by_side[side].setdefault(colour, [])
        if piece not in bucket:
            bucket.append(piece)
    # **Drawn a pixel wider, then cropped.** A run that crosses the edge of the
    # window is rasterised with the edge cutting it, and the pixel that straddles
    # the edge then differs from the full render by one -- measured, and enlarging
    # the rectangle made it worse (four), which is what pointed at the edge itself
    # rather than at an omitted neighbour.
    left, top, width, height = window
    padded_left, padded_top = max(0, left - 1), max(0, top - 1)
    padded = (padded_left, padded_top, width + 2, height + 2)
    drawn = layer_view(session, page_index, scale, padded, by_side=by_side)
    return drawn.crop((left - padded_left, top - padded_top,
                       left - padded_left + width, top - padded_top + height))


def edit_view(session, page_index: int, scale: Optional[float] = None,
              matched=None, window=None) -> Image.Image:
    """The diagnostic view you brush in.

    Both sides are painted from the page's own arrays, in three passes: the
    content both sides have first, then the content only the left has, then
    the content only the right. One pass per colour rather than one pass per
    side, because which side a stroke is on is not what decides whether it can
    be seen -- a red stroke that lands under a black one is a difference the user
    cannot see, and the differences are the whole point of the view.

    When either side is a picture the question is asked of the ink instead,
    and `_picture_edit_view` gives the same three colours by the same rule.
    """
    scale = session.scale if scale is None else scale
    # **One painter for every page.** The composition renders each side's ink, turns the
    # decisions into two visibility masks and composites the three colours per pixel with
    # the owner plane -- and the plane is composed for the window now, so a brush segment
    # costs its own rectangle rather than the page.
    return _picture_edit_view(session, page_index, scale, window, None)
    # **No table until a branch asks for one.** It costs a matching pass over the page
    # (`edit_table` -> `edit_colors`) and only the plain three-colour painter below reads
    # it: the piece painter takes none, and the composition paints each side as an ink
    # mask and thresholds it, so the colours never enter its answer. Measured:
    # `test_brush_is_quick` counted three whole-page builds in four gestures.
    if session.page_picture_sides(page_index):
        return _picture_edit_view(session, page_index, scale, window, None)
    if getattr(session, 'uses_layers', lambda _i: False)(page_index):
        return layer_view(session, page_index, scale, window)
    tables = edit_table(session, page_index, matched)

    image = _blank(session, page_index, scale, window)
    sides_arrays = _page_arrays(session, page_index)
    for colour in PAINT_ORDER:
        for at in (0, 1):
            arrays = sides_arrays[at]
            table, visible = tables[at]
            if not len(arrays[2]):
                continue
            wanted = _cull(arrays, scale, window) & _class_mask(table, visible, colour)
            if not wanted.any():
                continue
            # One pass per colour, each building on what the last one left:
            # `_draw` reads the image it is given, so the common ink stays
            # under the differences instead of being replaced by them.
            _draw(session, page_index, image, arrays, table, wanted, scale, window)
    return image


def _in_window_indices(strokes, scale, window, boxes):
    """Which of these strokes touch `window`, as indices into `strokes`."""
    if window is None:
        return strokes, range(len(strokes))
    if not strokes:
        return [], ()
    left, top, width, height = window
    right, bottom = left + width, top + height
    span, half = boxes
    pad = half * scale + 1.0
    keep = ~((span[:, 2] * scale + pad < left)
             | (span[:, 0] * scale - pad > right)
             | (span[:, 3] * scale + pad < top)
             | (span[:, 1] * scale - pad > bottom))
    return strokes, np.nonzero(keep)[0]


def picture_side(session, side: str, page_index: Optional[int] = None):
    """The `sides.Side` to render when this side has to be compared as an image.

    Two cases, and the second one is new:

      * a **PDF** side -- ink and no strokes, so `draw_strokes` has nothing to
        draw and the page is rendered;
      * a notebook page carrying an **`<image>`** -- it has strokes, but part of
        the page is a picture and the stroke path would show a hole there. The user's
        specification: during editing the picture is simply part of that side's
        image.

    `page_index` is what tells the second case apart, because a notebook may
    have a picture on one page and none on the next; without it, only a whole
    picture *side* is answered, which is the old behaviour.
    """
    from xp.workers.merger import sides as _sides

    held = getattr(session, 'left_side' if side == LEFT else 'right_side', None)
    if held is None:
        return None
    if held.kind == _sides.PDF:
        return held
    if page_index is not None:
        try:
            if side in session.page_picture_sides(page_index):
                return held
        except Exception:
            return None
    return None


def side_of(session, held) -> str:
    return LEFT if getattr(session, 'left_side', None) is held else RIGHT


def _other_side_image(session, side: str, page_index: int, scale: float):
    """The other side's page, to put this one on.

    The side with strokes is the one that stays put: it is the notebook the
    merge is building on, and its page box is the merged page's. Drawn once
    per page, by `_placement_for`, and never again.
    """
    other = RIGHT if side == LEFT else LEFT
    if picture_side(session, other, page_index) is not None:
        return None                 # two pictures: neither is the ruler
    left_strokes, right_strokes = session.strokes_for_page(page_index)
    strokes = left_strokes if other == LEFT else right_strokes
    if not strokes:
        return None
    page = _blank(session, page_index, scale)
    draw_strokes(page, strokes, scale)
    return np.asarray(page.convert('RGB'))[:, :, ::-1].copy()


def _placement_for(session, held, page_index: int):
    """Where this picture side sits on the merged page, worked out once.

    Cached, and cached for a reason that was measured: it is found by
    rendering **both** sides and comparing their frames, so doing it per
    redraw meant drawing the other side's whole page every time the user moved.

    The answer is kept at the session's own scale and scaled from there,
    because the shape of the placement does not depend on how far the user is
    zoomed in -- `factor` is a ratio of page boxes and the offsets are
    lengths. What changes with the zoom is only how many pixels a length is.
    """
    from xp.workers.converter import placement
    cache = getattr(session, '_placements', None)
    if cache is None:
        cache = session._placements = {}
    side = side_of(session, held)
    key = (side, page_index)
    if key in cache:
        return cache[key]

    ref = session.scale
    # **A page that is already the Q base page needs no placement at all.**
    # The picture side's own frame was compared with the other side's *ink*, and
    # the ink fit can gain overlap by shrinking: a PDF whose printed frame is
    # exactly the Q base reference and whose page box is the merged page came
    # back at factor 0.51 with offsets of 28.8 and 119.1 -- drawn at half size,
    # in the wrong place, so nothing coincided and nothing was black. When the
    # file says it is this page, it is this page. The user's words: "a PDF with a
    # stretched border cannot be calibrated" -- this is one of the reasons.
    from xp.core.frame import REFERENCE

    frame, pages, _spread = held.file_frame()
    want = page_pixels(session, page_index, ref)
    if frame is not None and pages == 1:
        box_w, box_h = held.page_box(page_index)
        near_frame = (abs(frame.x0 - REFERENCE[0]) < 2.0
                      and abs(frame.y0 - REFERENCE[1]) < 2.0
                      and abs(frame.x1 - REFERENCE[2]) < 2.0
                      and abs(frame.y1 - REFERENCE[3]) < 2.0)
        near_page = (abs(box_w * ref - want[0]) < 2.0
                     and abs(box_h * ref - want[1]) < 2.0)
        if near_frame and near_page:
            cache[key] = (1.0, 1.0, 0.0, 0.0)
            return cache[key]

    # **The same numbers the export uses.** The export places a picture side by
    # `session.geometry(page).fits[side]`; this function used to work out its own
    # placement from the ink, and on a page whose border is the wrong size and
    # shape the two disagreed -- the fit recovered x0.896 / y1.317 (the truth is
    # 0.894 / 1.318) while the view placed the page at 0.53, so the writing on
    # screen did not coincide even though the file was right. One calibration,
    # read twice, cannot disagree with itself.
    fitted = session.geometry(page_index).fits.get(side)
    if fitted is not None:
        sx = fitted.sx if fitted.sx is not None else fitted.scale
        sy = fitted.sy if fitted.sy is not None else fitted.scale
        answer = (sx, sy, fitted.dx, fitted.dy)
        cache[key] = answer
        return answer

    drawn = held.render(page_index, ref)
    other = _other_side_image(session, side, page_index, ref)
    if other is None:
        want = page_pixels(session, page_index, ref)
        factor = min(want[0] / drawn.shape[1], want[1] / drawn.shape[0])
        at_x = (want[0] - drawn.shape[1] * factor) / 2.0
        at_y = (want[1] - drawn.shape[0] * factor) / 2.0
    else:
        factor, at_x, at_y = placement.place(other, drawn)
    answer = (factor, factor, at_x / ref, at_y / ref)   # offsets in page units
    cache[key] = answer
    return answer


# How far, in page units, a printed hairline can sit from the template's own
# border and still be that border. The same band the stroke side uses
# (`convert.BORDER_BAND`), because it is the same pen and the same rasterising.
FRAME_BAND = 3.2


# How far, in page units, a printed hairline can sit from the template's own
# border and still be that border. The same band the stroke side uses
# (`convert.BORDER_BAND`), because it is the same pen and the same rasterising.
FRAME_BAND = 3.2


def _erase_the_printed_frame(piece, frame, px_per_unit, origin, paper):
    """Paint out the template's blue border in a rendered page, and nothing else.

    **The picture side is ink, and its frame is ink too.** The stroke side drops
    the printed Q base border when the page is read (`sides.drop_the_print_frame`);
    a PDF has no strokes to drop, so the border came through in the render and
    the merge drew it -- a solid blue rectangle the user explicitly asked never to see.
    It is masked here by the same two conditions the stroke side uses:
    **on the frame's rectangle** and **blue**. Real blue handwriting is not on
    the frame and is kept, which is the other half of the user's order.

    `px_per_unit` is one number for an ordinary page and `(x, y)` for a page the
    fit stretched, because on that one the border's rows and columns are not the
    same number of pixels apart.
    """
    import numpy as np

    if frame is None:
        return piece
    x0, y0, x1, y1 = frame.x0, frame.y0, frame.x1, frame.y1
    if isinstance(px_per_unit, (tuple, list)):
        px_x, px_y = float(px_per_unit[0]), float(px_per_unit[1])
    else:
        px_x = px_y = float(px_per_unit)
    left = origin[0]
    top = origin[1]
    height, width = piece.shape[0], piece.shape[1]
    columns = left + np.arange(width) / px_x
    rows = top + np.arange(height) / px_y
    near_x = (np.abs(columns[:, None] - np.array([x0, x1])[None, :])
              .min(axis=1) <= FRAME_BAND)
    near_y = (np.abs(rows[:, None] - np.array([y0, y1])[None, :])
              .min(axis=1) <= FRAME_BAND)
    on_the_frame = near_y[:, None] | near_x[None, :]
    if not on_the_frame.any():
        return piece
    # **BGR**, the way PyMuPDF draws and the way it is flipped back just before
    # it is pasted -- so the blue channel is index 0 here, not 2.
    blue = piece[:, :, 0].astype(np.int16)
    green = piece[:, :, 1].astype(np.int16)
    red = piece[:, :, 2].astype(np.int16)
    blueish = (blue > 150) & (blue - red > 25) & (blue - green > 8)
    mask = on_the_frame & blueish
    if not mask.any():
        return piece
    piece[mask] = paper
    return piece


def _rendered_stretched(session, held, page_index, scale, window, sx, sy,
                        at_x, at_y, mask_frame=True):
    """A picture side drawn with one scale in x and another in y.

    The whole page is rendered once (a stretched page is rare, and a windowed
    path for it would have to resample a rectangle and then paste it back), then
    resized to the size the fit asks for. The printed border is masked after the
    resize, where the border's own coordinates are the page's.
    """
    from PIL import Image

    page_w, page_h = page_pixels(session, page_index, scale)
    if window is None:
        left, top, width, height = 0, 0, page_w, page_h
    else:
        left, top, width, height = (int(round(v)) for v in window)
        width, height = max(1, int(width)), max(1, int(height))
    canvas = Image.new('RGB', (width, height), BACKGROUND_COLOR)
    box_w, box_h = held.page_box(page_index)
    # `render` hands back BGR rows; PIL wants RGB.
    drawn = np.asarray(held.render(page_index, scale * max(sx, sy)))
    picture = Image.fromarray(np.ascontiguousarray(drawn[:, :, ::-1]))
    want = (max(1, int(round(box_w * sx * scale))),
            max(1, int(round(box_h * sy * scale))))
    picture = picture.resize(want, Image.LANCZOS)
    rgb = np.asarray(picture, np.uint8)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    if mask_frame:
        frame, _pages, _spread = held.file_frame()
        if frame is not None:
            paper = _paper_of(session, held, page_index)
            bgr = _erase_the_printed_frame(
                bgr, frame, (sx * scale, sy * scale), (0.0, 0.0),
                tuple(int(v) for v in reversed(tuple(paper)[:3])))
    canvas.paste(Image.fromarray(bgr[:, :, ::-1]),
                 (int(round(at_x - left)), int(round(at_y - top))))
    return canvas


def _render_scale_of(session) -> float:
    """The resolution the source is rasterised at -- the DPI's, not the zoom's."""
    getter = getattr(session, 'render_scale', None)
    if getter is not None:
        return float(getter())
    return float(getattr(session, 'scale', 4.0))


def _rendered(session, held, page_index: int, scale: float, window=None,
              mask_frame: bool = True):
    """One page of a picture side, or one window of it, on the merged page.

    **The whole page is kept.** A PDF page costs PyMuPDF about 150-250 ms to rasterise, and every
    redraw of the same page -- a decision, a repair, coming back to it -- paid that again. The
    finished rgb is held per (side, page, scale) for a few pages; a window is not held, because there
    is one per zoom and they are small.
    """
    if window is None:
        held_page = _the_rendered_page(session, held, page_index, scale, mask_frame)
        if held_page is not None:
            return held_page.copy()
    return _render_it(session, held, page_index, scale, window, mask_frame)


#: How many whole pages of a picture side are kept. Four of the user's pages are about 14 MB.
_RENDERED_PAGES = 4


def _resample(have, want):
    """The filter for a resize, **by direction**.

    `BOX` is right when the picture is going down: it averages exactly the cells the smaller grid
    covers, which is what "a coarser view keeps the ink of every cell it holds" means. It is wrong
    when it is going **up**: BOX repeats a cell into a square block, so a 1.4 pt line rendered at
    29 pixels per unit and shown at 60 comes back as a staircase of square pixels with no grey
    between black and paper -- the user's screenshot, at "DPI: 800 / 4292 dpi on screen". Enlarging has to
    interpolate (BICUBIC) so the edge keeps its coverage; it cannot invent detail, but it must not
    turn a smooth edge into steps.
    """
    from PIL import Image

    grew = want[0] >= have[0] or want[1] >= have[1]
    return Image.BICUBIC if grew else Image.BOX


def _the_rendered_page(session, held, page_index: int, scale: float,
                       mask_frame: bool):
    """The kept render of this page, or None and it is built and kept."""
    cache = getattr(session, '_rendered_pages', None)
    if cache is None:
        cache = session._rendered_pages = {}
    # **Keyed by the resolution the page is rasterised at and nothing else.** The view's own scale is applied to the
    # result, not to the cache: keeping the scale in the key added a whole-page render per zoom step, and measured on the user's
    # pair that was 619 / 792 / 1593 MB live at 2x, 4x and 8x with a 2,957 MB peak.
    key = (side_of(session, held), page_index, round(_render_scale_of(session), 4),
           bool(mask_frame))
    with _CACHE_LOCK:
        kept = cache.get(key)
    if kept is not None:
        width = max(1, round(session.page_size(page_index)[0] * scale))
        height = max(1, round(session.page_size(page_index)[1] * scale))
        if kept.size == (width, height):
            return kept
        # A view at another zoom is the same page reduced, not a second page.
        return kept.resize((width, height), _resample(kept.size, (width, height)))
    picture = _render_it(session, held, page_index, scale, None, mask_frame)
    if picture is None:
        return None
    with _CACHE_LOCK:
        cache[key] = picture
        while len(cache) > _RENDERED_PAGES:
            cache.pop(next(iter(cache)))
    return picture


def _render_it(session, held, page_index: int, scale: float, window=None,
               mask_frame: bool = True):
    """One page of a picture side, or one window of it, on the merged page.

    THE WINDOW IS THE POINT, and the first version of this ignored it: it
    drew the whole page at the zoom and cropped, so at 8x it rasterized
    12800x17920 -- 229 megapixels -- for the patch on screen. Measured, the
    stroke side costs 0.011s for a window at 8x and this cost 3.5s.

    The PDF is rendered at `scale * factor` and placed; no resampling step,
    because rendering it at the size it is wanted is the same arithmetic and
    one fewer pass over the pixels. For a window, only the part of the PDF
    that falls inside it is rasterized -- PyMuPDF is given the clip.
    """
    sx, sy, at_x, at_y = _placement_for(session, held, page_index)
    at_x, at_y = at_x * scale, at_y * scale
    if abs(sx - sy) > 1e-3:
        # **Two different scales cannot be one render.** The page is drawn with
        # one, then resized to the other -- which is what a PDF with a stretched
        # border needs, and what the export already does with coordinates.
        return _rendered_stretched(session, held, page_index, scale, window,
                                   sx, sy, at_x, at_y, mask_frame)
    page_w, page_h = page_pixels(session, page_index, scale)
    if window is None:
        left, top, width, height = 0, 0, page_w, page_h
    else:
        left, top, width, height = (int(round(v)) for v in window)
        width, height = max(1, width), max(1, height)

    factor = sx
    # The same rectangle in the PDF's own pixels at the scale it will be
    # rendered at.
    # **The source is rasterised at the render DPI, whatever the zoom is.** The window is the same clip as before,
    # expressed in the source's own pixels at that resolution; the piece comes back larger and is reduced to the
    # size the view asked for.
    render_scale = _render_scale_of(session)
    # **The finer of the two.** The DPI is the pipeline's floor; the zoom is what the screen is asking for. Rendering
    # at only the DPI magnified a fixed picture at 8x and stayed blurred -- the user's words: "it keeps blur, it didn't
    # improve". A window is small, so following the zoom costs a window and not a page.
    effective = max(render_scale, float(scale or render_scale))
    # A window is small by construction, but it is a rectangle like any other: cap what it may rasterise.
    capped = getattr(session, 'window_scale', None)
    if capped is not None:
        width_units = (width / float(scale or 1.0)) if scale else float(width)
        height_units = (height / float(scale or 1.0)) if scale else float(height)
        effective = min(effective, float(capped(width_units, height_units)))
    pixels_per_unit = effective * factor
    zoom_to_render = (effective / scale) if scale else 1.0
    source = ((left - at_x) * zoom_to_render, (top - at_y) * zoom_to_render,
              width * zoom_to_render, height * zoom_to_render)
    canvas = Image.new('RGB', (width, height), BACKGROUND_COLOR)
    drawn = held.render_box(page_index, pixels_per_unit, source)
    if drawn is not None:
        piece, put_x, put_y = drawn
        # The printed border goes, before it is ever put on the page.
        frame, _pages, _spread = held.file_frame()
        if mask_frame and pixels_per_unit > 0 and frame is not None:
            # `_paper_of` renders this same page at a tenth of the scale to find
            # the paper, so it asks for that render without the mask -- else the
            # two calls ask each other for ever.
            paper_rgb = _paper_of(session, held, page_index)
            paper_bgr = tuple(int(v) for v in reversed(tuple(paper_rgb)[:3]))
            piece = _erase_the_printed_frame(
                piece, frame, pixels_per_unit,
                (source[0] / pixels_per_unit, source[1] / pixels_per_unit),
                paper_bgr)
        drawn_image = Image.fromarray(piece[:, :, ::-1])
        if drawn_image.size != (width, height):
            # **One resampling on the way to the screen, and its direction decides the filter.** When the budget
            # capped the raster below the zoom, this is an *enlargement*: BOX would repeat each cell as a square
            # block, which is the staircase in the user's screenshot. `_resample` says why.
            drawn_image = drawn_image.resize((width, height),
                                             _resample(drawn_image.size, (width, height)))
        canvas.paste(drawn_image, (put_x, put_y))
        # **Only when the caller wants the frame masked.** _paper_of renders this same page at a
        # tenth of the scale to find the paper, and it asks for that render with mask_frame=False;
        # painting the frame out unconditionally sent that call back to _paper_of, which rendered
        # again -- measured, 331 PDF renders and 10 seconds to load one page of the user's merge.
        if mask_frame:
            _paint_out_the_frame(canvas, session, held, page_index, scale, left, top)
    return canvas


# How far, in page units, the printed border's band is painted out. The border
# is a hairline and the geometry puts it on the template's frame, so a band
# around that rectangle covers it with room for a pixel of resampling either
# way without eating writing a few points inside it.
FRAME_BAND = 3.2


def _paint_out_the_frame(canvas, session, held, page_index: int, scale: float,
                         left: float, top: float) -> None:
    """The printed border is paper: paint it out of the picture.

    The user's instruction, 2026-10-24: "the real boundary ... should be discarded
    after the alignment, I should never see it". The converter already keeps it
    out of the strokes; the merge view draws the picture side as a picture, and
    the border is in those pixels. What replaces it is the dashed guide the
    window draws at the same rectangle -- so the alignment is still visible to
    the eye, and the printed blue line is not.
    """
    try:
        box = session.frame_box(page_index)
        paper = _paper_of(session, held, page_index)
    except Exception:
        return
    if paper is None:
        return
    band = max(2.0, FRAME_BAND * scale)
    x0 = box[0] * scale - left
    y0 = box[1] * scale - top
    x1 = box[2] * scale - left
    y1 = box[3] * scale - top
    # **Only the border's own pixels.** This used to paste the paper colour
    # over the whole band around the reference rectangle, so a black mark that
    # happened to lie there vanished from the picture: the rule is "the printed
    # blue border is not shown", not "the band is paper". The mask is the same
    # test the converter uses to decide a piece is the frame (`blue > 60`,
    # `blue - red > 40`, `blue - green > 20`), applied to the pixels.
    import numpy as np

    pixels = np.asarray(canvas.convert('RGB'), dtype=np.int16)
    height, width = pixels.shape[:2]
    # `paper` is a **colour**, not a picture: `_paper_of` returns the tuple
    # `imagediff.paper_of` found, and `Image.paste` takes it as a fill. The
    # first version of this mask asked it for `convert('RGB')` and a tuple has
    # no such method (`test_gpu_render` caught it).
    paper_rgb = np.asarray(paper, dtype=np.int16).reshape(-1)[:3]
    regions = ((x0 - band, y0 - band, x1 + band, y0 + band),
               (x0 - band, y1 - band, x1 + band, y1 + band),
               (x0 - band, y0, x0 + band, y1),
               (x1 - band, y0, x1 + band, y1))
    changed = False
    for region in regions:
        left_at = max(0, int(round(region[0])))
        top_at = max(0, int(round(region[1])))
        right_at = min(width, int(round(region[2])))
        bottom_at = min(height, int(round(region[3])))
        if right_at <= left_at or bottom_at <= top_at:
            continue
        part = pixels[top_at:bottom_at, left_at:right_at]
        red, green, blue = part[:, :, 0], part[:, :, 1], part[:, :, 2]
        border = (blue > 60) & (blue - red > 40) & (blue - green > 20)
        if border.any():
            part[border] = paper_rgb
            changed = True
    if changed:
        canvas.paste(Image.fromarray(pixels.astype(np.uint8), 'RGB'), (0, 0))


def source_view(session, page_index: int, side: str, scale: Optional[float] = None,
                window=None) -> Image.Image:
    """One side's page exactly as it was written: every stroke, original
    colours, no decisions applied and nothing hidden.

    A side with no strokes -- a PDF, which has not been converted and will
    not be unless the brush asks for it -- is **rendered** instead.
    """
    scale = session.scale if scale is None else scale
    held = picture_side(session, side, page_index)
    if held is not None:
        return _rendered(session, held, page_index, scale, window)
    arrays = _page_arrays(session, page_index)[0 if side == LEFT else 1]
    return _paint_arrays(session, page_index, scale, window, arrays,
                         arrays[5], _cull(arrays, scale, window))


_CARD = None


def _the_card():
    """The card's painter, or None to use the processor.

    Built on first use and **kept**: it holds a page's strokes, so building one
    per frame would send them up again every frame and the card would be
    slower than the processor for exactly the reason it was made a class.

    `XP_CPU_PAINT=1` forces the serial painter. That is what the comparison in
    `tests/test_gpu_render.py` drives, and it is what to reach for if a driver
    ever disagrees with it.
    """
    global _CARD
    if _CARD is None:
        try:
            import os

            from xp.gpu import painter as gpurender
            if os.environ.get('XP_CPU_PAINT'):
                _CARD = False
            else:
                _CARD = (gpurender.Painter() if gpurender.Painter.available()
                         else False)
        except Exception:
            _CARD = False
    return _CARD or None


def _page_key(session, page_index, arrays):
    """What a resident page is. The same key means nothing needs sending.

    The page and the decisions that built it -- and **which side's arrays**,
    because the editing view draws both sides of one page through here. Keying
    on the page alone sent the left side's strokes up, then matched the key
    when the right side arrived, so the right side's "wanted" mask was applied
    to the left side's strokes: the view came out with no red in it at all,
    which is how `tests/test_previews` caught it in one run.

    The colour table is deliberately *not* in the key. The editing palette and
    the original colours are the same strokes drawn twice, and the table is
    handed to `paint` each time; including it here sent 8.5 MB up again for
    every pass of a view that draws three.

    **And neither are the decisions.** They were, and that meant every brush segment sent the whole page up again -- the
    geometry does not change with a decision, only which strokes are drawn, and that is the `wanted` mask `paint` takes
    per call. Measured on a 40,000-stroke page, the two ink layers cost 34.9 ms a frame with the stamp in the key.
    """
    return (id(session), page_index, id(arrays))


def _draw(session, page_index, image, arrays, colours, wanted, scale, window):
    """Paint a page's arrays into `image` -- on the card, or on the processor.

    The card is preferred and the processor is the answer whenever there is no
    card, or it refuses, or it is told to stand down. `raster.paint_flat` is
    not a slower substitute for this: it is the definition of what the drawing
    looks like, and `tests/test_gpu_render.py` holds the card to being
    bit-identical to it -- on the sample page, at four zooms, on all three
    channels.
    """
    pixels = np.array(image, np.uint8)
    card = _the_card()
    if card is not None:
        try:
            card.hold(_page_key(session, page_index, arrays), arrays[0], arrays[1],
                      arrays[2], arrays[3], arrays[4], colours)
            painted = card.paint(pixels, scale, _offset(window), wanted,
                                 colours=colours)
            image.paste(Image.fromarray(painted, 'RGB'))
            return image
        except Exception:
            # Any refusal at all falls back, and the fallback is the
            # definition. A window that draws the wrong page because a driver
            # said something unexpected is worse than a slow one.
            pass
    from xp.render import raster
    raster.paint_flat(pixels, arrays[0], arrays[1], arrays[2], arrays[3],
                      arrays[4], colours, wanted=wanted, scale=scale,
                      offset=_offset(window))
    image.paste(Image.fromarray(pixels, 'RGB'))
    return image


def _paint_arrays(session, page_index, scale, window, arrays, colours, wanted):
    """The one place a page's arrays become pixels."""
    image = _blank(session, page_index, scale, window)
    if not len(arrays[2]) or not wanted.any():
        return image
    return _draw(session, page_index, image, arrays, colours, wanted, scale, window)


def overlay_view(session, page_index: int, scale: Optional[float] = None,
                 window=None) -> Image.Image:
    """Both raw pages on top of each other, both in their own colours."""
    scale = session.scale if scale is None else scale
    image = _blank(session, page_index, scale, window)
    left_strokes, right_strokes = session.strokes_for_page(page_index)
    at = _offset(window)
    draw_strokes(image, _in_window(left_strokes, scale, window)[0], scale,
                 offset=at)
    return draw_strokes(image, _in_window(right_strokes, scale, window)[0],
                        scale, offset=at)


def _picture_ink(session, side: str, page_index: int, scale: float, window):
    """The picture side's visible ink as an image, or None.

    The Final Preview has to show what the export will write, and the export
    writes **traced strokes** for a picture side where the brush took it. The
    preview cannot run the converter (that is the work the merge exists to
    defer), but it can show the picture's own pixels in exactly those places --
    the traced strokes are a re-drawing of that ink, median 0.17 pt away.

    A picture side that is a notebook is drawn from its `<image>` objects only,
    because its strokes are already drawn from `final_strokes`.
    """
    held = picture_side(session, side, page_index)
    if held is None:
        return None
    if held.kind == 'pdf':
        return _rendered(session, held, page_index, scale, window)
    from xp.render.drawing import draw_pictures

    page_now = session.page_at(page_index)
    own_index = page_now.left if side == LEFT else page_now.right
    document = session.left if side == LEFT else session.right
    if own_index is None or document is None:
        return None
    if window is None:
        width, height = page_pixels(session, page_index, scale)
        left, top = 0, 0
        size = (int(width), int(height))
    else:
        left, top, width, height = (int(round(v)) for v in window)
        size = (max(1, width), max(1, height))
    canvas = Image.new('RGB', size, BACKGROUND_COLOR)
    fit = session.geometry(page_index).fits[side]
    # The placement is the same one `_rendered` uses; drawn here with the
    # picture's own scale so the visible-area mask (which is in the page's
    # pixels) lines up.
    draw_pictures(canvas, document.pages[own_index], scale * fit.scale_x,
                  offset=((left - fit.dx * scale) / (scale * fit.scale_x),
                          (top - fit.dy * scale) / (scale * fit.scale_y)))
    return canvas


def final_view(session, page_index: int, scale: Optional[float] = None,
               window=None) -> Image.Image:
    """The merged page, in original colours: what export() will write.

    **The same selection the export uses** (`merge.final_strokes`), plus what
    the brush took from a picture side. It used to draw `composed_strokes` --
    the whole-stroke decisions -- and the difference was measurable on the user's pair:
    with a band taken from the PDF side, the preview held 8,097 ink pixels fewer
    than the file. A preview that disagrees with the file is worse than no
    preview, and this was hidden until the two were compared by ink.
    """
    import numpy as np

    from xp.core.document.xopp_io import Stroke

    scale = session.scale if scale is None else scale
    image = _blank(session, page_index, scale, window)
    # **One list, healed.** `final_page_strokes` is the native pieces and the
    # converted ones in one place, run through Heal's linking pass -- so the
    # preview draws exactly the strokes the file will hold, including which
    # pieces were joined into one.
    elements = list(session.final_page_strokes(page_index))
    strokes = [Stroke(element=element,
                      points=_points_of(element),
                      color=element.get('color') or '#000000ff',
                      widths=[float(v) for v in (element.get('width') or '1').split()])
              for element in elements]
    strokes = [stroke for stroke in strokes if len(stroke.points) >= 2]
    draw_strokes(image, _in_window(strokes, scale, window)[0], scale,
                 offset=_offset(window))

    # **What the brush took from a picture side**, where the file will hold it.
    # A side whose conversion has already run needs nothing here: its strokes
    # are in `_converted_for` above.
    converted = bool(session._converted_for(page_index))
    if converted:
        return image
    for side in session.page_picture_sides(page_index):
        drawn = _picture_ink(session, side, page_index, scale, window)
        if drawn is None:
            continue
        visible = session._visible_area(page_index, side)
        if visible is None:
            image.paste(drawn, (0, 0))
            continue
        height, width = visible.shape
        left, top = (0, 0) if window is None else (int(round(window[0])),
                                                   int(round(window[1])))
        piece = visible[int(top):int(top) + drawn.height,
                        int(left):int(left) + drawn.width]
        if piece.size == 0:
            continue
        mask = Image.fromarray((piece * 255).astype(np.uint8), 'L')
        image.paste(drawn, (0, 0), mask)
    return image


def _points_of(element):
    numbers = [float(value) for value in (element.text or '').split()]
    return list(zip(numbers[0::2], numbers[1::2]))


def render(session, page_index: int, mode: str, scale: Optional[float] = None,
           matched=None, window=None) -> Image.Image:
    """Whichever view `mode` names, built from scratch.

    `window` is (left, top, width, height) in the pixels of `scale`, and the
    answer is that rectangle of the page rather than all of it. Left out,
    every view behaves exactly as it did.
    """
    if mode == SOURCE_LEFT:
        return source_view(session, page_index, LEFT, scale, window)
    if mode == SOURCE_RIGHT:
        return source_view(session, page_index, RIGHT, scale, window)
    if mode == OVERLAY:
        return overlay_view(session, page_index, scale, window)
    if mode == FINAL:
        return final_view(session, page_index, scale, window)
    return edit_view(session, page_index, scale, matched, window)
