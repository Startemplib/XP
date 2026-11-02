"""The blue frame, which is what everything is normalized against.

The Q base template (`Qbase_66x138.pdf`, 66 x 138 mm) carries one drawing: a
blue rectangle inset 7pt from the edge of the page. Every notebook page is
written inside it, and everything outside it is margin.

That frame, not the page box and not the ink, is the reference. Two pages
line up when their frames line up -- which is the only thing that survives a
page being exported with wide margins, cropped, or enlarged before it was
ever traced. A page box says nothing about where the writing went; the ink
is a guess about the same question; the frame is the answer, drawn on the
page by the template itself.

A frame reaches a document one of three ways, and all three are read here:

- as the page's PDF **background**, which is the template itself;
- as blue **ink**, when a PDF carrying the frame has been traced into
  strokes -- the tracer has no idea it was a frame, and draws it;
- in a **rendered page**, while converting, where it is a blue rectangle in
  the image.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from xp.core.document.pagesize import QBASE

# The template's own frame, in Q base points: 7pt in from every edge.
REFERENCE = (7.0, 7.0, QBASE[0] - 7.0, QBASE[1] - 7.0)
# The blue it is drawn in, BGR. The template's is RGB(30,144,255); traced or
# re-exported copies drift, so the test is generous about how blue is blue.
FRAME_BGR = (255, 144, 30)
BLUE_TOLERANCE = 90
MIN_FRAME_SHARE = 0.35      # of the page, in both directions: smaller is not a frame
RECT_TOLERANCE = 0.06       # how far from a rectangle a frame may be, as a share of its size


@dataclass(frozen=True)
class Frame:
    """Where the frame sits, and how it was found."""

    x0: float
    y0: float
    x1: float
    y1: float
    source: str = ''

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def box(self) -> Tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1

    @property
    def observed(self) -> bool:
        """Whether this frame was actually seen, rather than assumed.

        A Q base page with no frame in it is assumed to have the template's,
        because that is what it was written on -- but an assumption on both
        sides tells you nothing about how they line up, and two assumptions
        must not be allowed to stand in for looking at the ink.
        """
        return not self.source.startswith('assumed')


REFERENCE_FRAME = Frame(*REFERENCE, source='the Q base template')


def _is_blue(color: str) -> bool:
    """Whether an #rrggbbaa is the frame's blue, generously."""
    try:
        value = color.lstrip('#')
        r, g, b = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    except (ValueError, IndexError):
        return False
    return (abs(b - FRAME_BGR[0]) <= BLUE_TOLERANCE
            and abs(g - FRAME_BGR[1]) <= BLUE_TOLERANCE
            and abs(r - FRAME_BGR[2]) <= BLUE_TOLERANCE)


def _rectangular(points: np.ndarray) -> bool:
    """Whether these points trace the outline of a rectangle.

    Checked by area rather than by counting corners: a traced frame has been
    through a threshold and a contour walk, so it has far more than four
    points and none of them exactly on a corner.
    """
    if len(points) < 4:
        return False
    x0, y0 = points[:, 0].min(), points[:, 1].min()
    x1, y1 = points[:, 0].max(), points[:, 1].max()
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        return False
    # Every point should sit near one of the four edges.
    near_edge = (np.minimum(np.abs(points[:, 0] - x0), np.abs(points[:, 0] - x1))
                 <= RECT_TOLERANCE * width)
    near_edge |= (np.minimum(np.abs(points[:, 1] - y0), np.abs(points[:, 1] - y1))
                  <= RECT_TOLERANCE * height)
    return bool(near_edge.mean() > 0.9)


# How far, in page units, a piece may sit from the template's printed frame and
# still be that frame. The border is a hairline, and a traced one comes back as
# a run of short pieces a fraction of a point wide, so the band allows for the
# pen and for the rounding.
PRINT_BAND = 2.5


def is_print_frame_piece(stroke, frame=None, band: float = PRINT_BAND) -> bool:
    """Whether this piece is part of the template's printed border.

    Two conditions together: every point of it lies on the frame's rectangle
    (within `band`), **and** it is the frame's blue. Geometry alone would take a
    black pen run along the edge; colour alone would take the blue handwriting,
    which is ink.

    This is the same rule the converter applies when it traces a page -- there
    it is a tolerance on a picture, here it is a tolerance on strokes that were
    already traced by an older build, and the user's `TEST_converted-Arra1.xopp` holds
    **3,028** such pieces that a converter without this rule left behind.
    """
    # **The colour first, because it costs nothing.** The two conditions are
    # an `and` of two pure tests, so their order cannot change the answer --
    # but the geometry builds a numpy array per stroke, and this is asked of
    # every stroke on the page (3,479 of them on the user's densest). On a page of
    # handwriting almost nothing is the frame's blue, so asking that first
    # skips the array for almost all of them. This sits inside `page_ink`,
    # which is 80% of what a page signature costs.
    if not _is_blue(str(getattr(stroke, 'color', ''))):
        return False
    try:
        x0, y0, x1, y1 = (frame.box if frame is not None else REFERENCE)
        points = np.asarray(stroke.points, dtype=float)
    except Exception:
        return False
    if len(points) == 0:
        return False
    on_edge = ((np.abs(points[:, 0] - x0) <= band)
               | (np.abs(points[:, 0] - x1) <= band)
               | (np.abs(points[:, 1] - y0) <= band)
               | (np.abs(points[:, 1] - y1) <= band))
    return bool(on_edge.all())


def centreline_frame(strokes: Sequence, frame=None) -> Optional['Frame']:
    """The printed border as four lines, measured on their **centres**.

    A traced border is a picture of a line with width: its bounding box is the
    *outside* of the ink, which is half a pen wider than the line the template
    draws on every side. Aligning the outside of the ink to the template puts
    the writing half a pen off, consistently -- on the sample converted file the
    rectangle spans (7.01, 7.00) to (181.06, 383.75) while the writing it
    belongs to sits within half a point of where the template says it should.

    So each edge is the median of the coordinates of the pieces lying on it,
    which is the centre of the line however wide the pen was and however many
    pieces the trace broke it into. Returns None when nothing is on the frame.
    """
    x0, y0, x1, y1 = (frame.box if frame is not None else REFERENCE)
    edges = {'top': [], 'bottom': [], 'left': [], 'right': []}
    for stroke in strokes:
        if not is_print_frame_piece(stroke, frame):
            continue
        points = np.asarray(stroke.points, dtype=float)
        if not len(points):
            continue
        # **Per point, not per stroke.** A border traced in pieces is one edge
        # per piece, but a border written as a single rectangle outline runs
        # along all four -- assigning the whole stroke to its nearest edge left
        # three edges empty and the frame unmeasurable.
        for x, y in points:
            # A point within the band of *two* edges is a corner, and it is
            # evidence for both: a border drawn as one rectangle outline has
            # nothing but corners, and taking only the nearest edge left two
            # sides with no points and the frame unmeasurable.
            distances = {'top': abs(y - y0), 'bottom': abs(y - y1),
                         'left': abs(x - x0), 'right': abs(x - x1)}
            for edge, distance in distances.items():
                if distance <= PRINT_BAND:
                    edges[edge].append(y if edge in ('top', 'bottom') else x)
    if not all(edges[name] for name in ('top', 'bottom', 'left', 'right')):
        return None
    return Frame(
        float(np.median(edges['left'])),
        float(np.median(edges['top'])),
        float(np.median(edges['right'])),
        float(np.median(edges['bottom'])),
        source='the printed border, centreline')


def centreline_in_image(image_bgr) -> Optional['Frame']:
    """The printed frame in a rendered page, measured on the line's centres.

    `frame_in_image` returns the bounding box of the blue ink, which is the
    *outside* of a line that has width. The frame the template draws is the
    centre of that line, so each edge here is the median of the mask's
    coordinates in the band where that edge lives.
    """
    if image_bgr is None or getattr(image_bgr, 'size', 0) == 0:
        return None
    import cv2

    lower = np.array([max(0, c - BLUE_TOLERANCE) for c in FRAME_BGR], np.int16)
    upper = np.array([min(255, c + BLUE_TOLERANCE) for c in FRAME_BGR], np.int16)
    mask = cv2.inRange(image_bgr, lower.astype(np.uint8), upper.astype(np.uint8))
    rows, cols = np.nonzero(mask)
    if len(rows) < 20:
        return None
    height, width = mask.shape
    near_top = rows <= rows.min() + max(2, int(height * 0.02))
    near_bottom = rows >= rows.max() - max(2, int(height * 0.02))
    near_left = cols <= cols.min() + max(2, int(width * 0.02))
    near_right = cols >= cols.max() - max(2, int(width * 0.02))
    if not (near_top.any() and near_bottom.any()
            and near_left.any() and near_right.any()):
        return None
    # **A blue diagonal is not a border.** The four tests above only ask whether
    # some blue sits at each extreme of the blue set, and a straight line from
    # one corner to the other satisfies all four: a page with no printed border
    # but blue handwriting had that handwriting's bounding box taken as the
    # file's frame, and the merge calibrated the page to it (x1.49 / y2.85 on a
    # page whose writing was one stroke). A border is *four lines*, so each edge
    # band has to be blue along most of its length, not at one point.
    span_x = int(cols.max() - cols.min()) + 1
    span_y = int(rows.max() - rows.min()) + 1
    def covered(values, span):
        return span <= 2 or (np.unique(values).size / float(span)) >= 0.5
    if not (covered(cols[near_top], span_x) and covered(cols[near_bottom], span_x)
            and covered(rows[near_left], span_y)
            and covered(rows[near_right], span_y)):
        return None
    return Frame(float(np.median(cols[near_left])),
                 float(np.median(rows[near_top])),
                 float(np.median(cols[near_right])),
                 float(np.median(rows[near_bottom])),
                 source='the printed border, centreline in a render')


def file_frame(frames) -> Tuple[Optional['Frame'], int, float]:
    """One frame for a whole document, from the pages that have a real one.

    **The geometry is a property of the file, not of a page.** A PDF can have
    pages that carry no border -- added content, a crop, a different template --
    while still following the same coordinates. Measuring every page on its own
    and giving up on the ones with no border is what made those pages fall back
    to guessing; one reliable border is enough to calibrate the file.

    Returns `(frame, pages_used, spread)` where `spread` is the largest
    disagreement between a page's frame and the file's, in page units -- a file
    whose pages really were laid out differently shows up there rather than
    being averaged into a transform that fits none of them.
    """
    observed = [f for f in frames if f is not None and f.observed]
    if not observed:
        return None, 0, 0.0
    boxes = np.array([[f.x0, f.y0, f.x1, f.y1] for f in observed], dtype=float)
    middle = np.median(boxes, axis=0)
    spread = float(np.abs(boxes - middle).max()) if len(boxes) > 1 else 0.0
    return (Frame(float(middle[0]), float(middle[1]), float(middle[2]),
                  float(middle[3]),
                  source='%d page%s of this file' % (len(observed),
                                                     '' if len(observed) == 1 else 's')),
            len(observed), spread)


def without_the_print_frame(strokes: Sequence, page_size: Tuple[float, float],
                            frame=None) -> list:
    """The page's strokes with the printed frame taken out.

    Only for a page that is Q base (or that names a frame of its own): a page
    written elsewhere has no template border to be mistaken for ink.
    """
    from xp.core.document.pagesize import is_qbase

    if frame is None and not is_qbase(*page_size):
        return list(strokes)
    return [s for s in strokes
            if not is_print_frame_piece(s, frame)]


def frame_stroke(strokes: Sequence, page_size: Tuple[float, float]):
    """The stroke that *is* the frame, when a traced page carries one.

    Whoever wants the box calls `frame_in_strokes`; whoever wants to leave the
    frame out of the ink -- a page signature, say -- wants the stroke itself.
    Both are the same search, so it is written once, here.
    """
    best = None
    for stroke in strokes:
        # **The colour before the array**, for the same reason as in
        # `is_print_frame_piece`: all four conditions are pure, so the order
        # cannot change which stroke wins, and this loop is walked once per
        # page by `reference_box` -- 2.52 s across the user's 68 pages, almost all of
        # it spent building an array for strokes that were never the frame's
        # blue to begin with.
        if not _is_blue(stroke.color):
            continue
        points = np.asarray(stroke.points, dtype=np.float64)
        if len(points) < 4:
            continue
        width = float(points[:, 0].max() - points[:, 0].min())
        height = float(points[:, 1].max() - points[:, 1].min())
        if width < page_size[0] * MIN_FRAME_SHARE or height < page_size[1] * MIN_FRAME_SHARE:
            continue
        if not _rectangular(points):
            continue
        if best is None or width * height > _stroke_area(best):
            best = stroke
    return best


def _stroke_area(stroke) -> float:
    points = np.asarray(stroke.points, dtype=np.float64)
    return float((points[:, 0].max() - points[:, 0].min())
                 * (points[:, 1].max() - points[:, 1].min()))


# How close to the edge of their own bounding box the blue strokes have to
# stay before they are a frame rather than blue writing, as a share of the
# page. A traced frame is a thin border; handwriting fills the middle.
BORDER_BAND = 0.06


def frame_in_scattered_strokes(strokes: Sequence,
                               page_size: Tuple[float, float]) -> Optional[Frame]:
    """The frame when tracing broke it into pieces.

    `frame_stroke` looks for one stroke that is the whole frame, which is what
    a frame drawn by hand or copied from the template looks like. **A traced
    page does not have one.** Converting the Q base template turns its border
    into hundreds of short strokes, and every one of them fails the test for
    being a frame on its own: measured on the user's two converted documents, 2,648
    blue strokes on page 1 and **not one** of them spans 35% of the page in
    both directions.

    So the frame was never found in either, both sides fell through to
    "assume the template", two assumptions were refused, and the ink was asked
    instead -- which placed one document 2.4pt down the page from the other.
    That is the gap between the red and the blue borders in the user's photograph.

    Taken together, though, those strokes are exactly the frame. This takes
    the blue ink as a whole and asks whether it is shaped like a border:
    wide and tall enough to be the frame, and *hollow* -- the blue staying
    near the edge of its own box rather than filling it, which is what tells
    a frame from a page of blue writing.
    """
    boxes = []
    for stroke in strokes:
        if not _is_blue(stroke.color):
            continue
        points = np.asarray(stroke.points, dtype=np.float64)
        if len(points) == 0:
            continue
        boxes.append((points[:, 0].min(), points[:, 1].min(),
                      points[:, 0].max(), points[:, 1].max()))
    if len(boxes) < 4:
        return None
    boxes = np.asarray(boxes, dtype=np.float64)
    x0, y0 = float(boxes[:, 0].min()), float(boxes[:, 1].min())
    x1, y1 = float(boxes[:, 2].max()), float(boxes[:, 3].max())
    if (x1 - x0) < page_size[0] * MIN_FRAME_SHARE or (y1 - y0) < page_size[1] * MIN_FRAME_SHARE:
        return None

    # Is there a line along each of the four edges?
    #
    # Not "is all the blue near the edge" -- that was the first test and it
    # is wrong, because the page has blue *writing* on it too: measured on
    # the user's converted document, 2,648 blue strokes with only 66.9% of them near
    # the border, flat at every band width. The writing in the middle is not
    # evidence against the frame; it is simply not part of it.
    #
    # A frame is four lines. Each edge is asked whether blue ink runs along
    # most of its length, which extra blue in the middle cannot fake and a
    # page of blue handwriting does not satisfy.
    band_x = max(page_size[0], x1 - x0) * BORDER_BAND
    band_y = max(page_size[1], y1 - y0) * BORDER_BAND

    def runs_along(selected, axis_min, axis_max, span):
        """How much of one edge the selected strokes cover, 0..1."""
        if not selected.any() or span <= 0:
            return 0.0
        lows = boxes[selected, axis_min]
        highs = boxes[selected, axis_max]
        order = np.argsort(lows)
        covered, reach = 0.0, None
        for low, high in zip(lows[order], highs[order]):
            if reach is None or low > reach:
                if reach is not None:
                    covered += reach - start
                start, reach = low, high
            else:
                reach = max(reach, high)
        if reach is not None:
            covered += reach - start
        return float(covered / span)

    edges = {
        'left': runs_along(boxes[:, 0] <= x0 + band_x, 1, 3, y1 - y0),
        'right': runs_along(boxes[:, 2] >= x1 - band_x, 1, 3, y1 - y0),
        'top': runs_along(boxes[:, 1] <= y0 + band_y, 0, 2, x1 - x0),
        'bottom': runs_along(boxes[:, 3] >= y1 - band_y, 0, 2, x1 - x0),
    }
    if min(edges.values()) < 0.8:
        return None          # not four sides: blue ink, but not a frame
    return Frame(x0, y0, x1, y1, 'blue ink, traced in pieces')


def frame_in_strokes(strokes: Sequence, page_size: Tuple[float, float]) -> Optional[Frame]:
    """The frame as ink, from a document traced off a page that had one."""
    stroke = frame_stroke(strokes, page_size)
    if stroke is None:
        return frame_in_scattered_strokes(strokes, page_size)
    points = np.asarray(stroke.points, dtype=np.float64)
    return Frame(float(points[:, 0].min()), float(points[:, 1].min()),
                 float(points[:, 0].max()), float(points[:, 1].max()), 'blue ink')


def frame_in_pdf(path, page_number: int = 0) -> Optional[Frame]:
    """The frame as a page's PDF background -- the template itself."""
    try:
        import pymupdf

        with pymupdf.open(str(path)) as document:
            if page_number >= len(document):
                return None
            page = document[page_number]
            best = None
            for drawing in page.get_drawings():
                rect = drawing['rect']
                if (rect.width < page.rect.width * MIN_FRAME_SHARE
                        or rect.height < page.rect.height * MIN_FRAME_SHARE):
                    continue
                color = drawing.get('color')
                if color is None:
                    continue
                r, g, b = [int(round(c * 255)) for c in color[:3]]
                if not _is_blue('#%02x%02x%02x' % (r, g, b)):
                    continue
                found = Frame(rect.x0, rect.y0, rect.x1, rect.y1, 'the page background')
                if best is None or found.width * found.height > best.width * best.height:
                    best = found
            return best
    except Exception:
        return None            # a background that cannot be read is not a frame


def frame_in_image(image_bgr: np.ndarray) -> Optional[Frame]:
    """The frame in a rendered page, while converting it.

    In pixels of that image. Found by colour rather than by shape: at this
    point the page is a photograph of ink, and the one thing that marks the
    frame out from the writing is that it is the template's blue.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    lower = np.array([max(0, c - BLUE_TOLERANCE) for c in FRAME_BGR], np.int16)
    upper = np.array([min(255, c + BLUE_TOLERANCE) for c in FRAME_BGR], np.int16)
    mask = cv2.inRange(image_bgr, lower.astype(np.uint8), upper.astype(np.uint8))
    if not mask.any():
        return None
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * MIN_FRAME_SHARE or h < height * MIN_FRAME_SHARE:
            continue
        found = Frame(float(x), float(y), float(x + w), float(y + h), 'blue in the render')
        if best is None or found.width * found.height > best.width * best.height:
            best = found
    return best


def background_pdf(page, document=None, beside=None):
    """The PDF behind a page, as `(path, page number)`, found on this machine.

    Two things stop the filename in the file from being usable, and both are
    normal Xournal++ documents rather than anything unusual:

      * **it is relative to wherever the notebook used to be.** Notebooks
        carry `../../../../../somewhere/.../Qbase_66x138.pdf`, written when
        the document lived somewhere else. `Path(filename).is_file()` is
        False, and the template is sitting right beside the notebook.
      * **only the first page names it.** Later pages are
        `<background type="pdf" pageno="7"/>` and inherit the filename from
        the document, which is Xournal++'s own convention -- so asking a page
        for its filename answers None for every page but the first.

    Before this, both were simply misses: the frame lookup fell through to
    "assume the template", the merge then had two assumed frames, decided
    they said nothing, and handed the job to the ink -- which scaled a
    notebook against its own autosave by 1.200x.
    """
    background = page.element.find('background')
    if background is None or background.get('type') != 'pdf':
        return None
    pageno = max(0, int(background.get('pageno', '1') or 1) - 1)

    filename = background.get('filename')
    if not filename and document is not None:
        # Inherited: the first page of the document that names one.
        for other in document.pages:
            other_background = other.element.find('background')
            if other_background is not None and other_background.get('filename'):
                filename = other_background.get('filename')
                break
    if not filename:
        return None

    candidates = [Path(filename)]
    if beside is not None:
        beside = Path(beside)
        candidates.append(beside / filename)          # relative to the notebook
        candidates.append(beside / Path(filename).name)   # or just beside it
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate, pageno
        except OSError:
            continue
    return None


def frame_for_page(page, page_size: Tuple[float, float], strokes=None,
                   document=None, beside=None) -> Optional[Frame]:
    """A .xopp page's frame: its ink if it was traced, else its background.

    A Q base page with neither is still a Q base page -- the template is what
    it was written on -- so the template's own frame is assumed rather than
    giving up on it. `document` and `beside` are what let the background
    actually be found; without them this falls back to assuming far more
    often than it needs to.
    """
    from xp.core.document.pagesize import is_qbase

    strokes = page.strokes() if strokes is None else strokes
    found = frame_in_strokes(strokes, page_size)
    if found is not None:
        return found
    behind = background_pdf(page, document, beside)
    if behind is not None:
        found = frame_in_pdf(str(behind[0]), behind[1])
        if found is not None:
            return found
    if is_qbase(*page_size):
        return Frame(*REFERENCE, source='assumed: the page is Q base')
    return None


# How far the two axes may disagree before the fit is asked to stretch them
# separately. Under this it is measurement noise -- a traced frame is never
# exactly square, and the tracer's error is well under a percent -- so a
# uniform fit is the safer answer there. A PDF that was printed, cropped or
# exported at a different aspect disagrees by several percent, and a single
# scale would leave the writing stretched one way.
ASPECT_SLACK = 0.03


def align(source: Frame, target: Frame = REFERENCE_FRAME):
    """The scale and shift that put one frame onto another.

    Uniform unless the two frames really do disagree about the aspect: a frame
    that comes back a little out of square (a traced one always does) must not
    be allowed to stretch the writing inside it, but a PDF that was printed or
    cropped at a different aspect genuinely needs the two axes scaled
    separately, and a single number would leave it stretched one way.

    The shift is computed from the centre of the frame, so `dx`/`dy` are the
    translation of the box's middle -- which is what a crop moves.
    """
    from xp.core.inkfit import Fit

    if source.width <= 0 or source.height <= 0:
        return Fit()
    sx = target.width / source.width
    sy = target.height / source.height
    if abs(sx - sy) <= ASPECT_SLACK * max(sx, sy):
        scale = (sx + sy) / 2
        return Fit(scale, target.x0 - source.x0 * scale,
                   target.y0 - source.y0 * scale,
                   1.0, 'frame aligned (%s)' % source.source)
    sx_shift = (target.x0 + target.width / 2) - (source.x0 + source.width / 2) * sx
    sy_shift = (target.y0 + target.height / 2) - (source.y0 + source.height / 2) * sy
    return Fit((sx + sy) / 2, sx_shift, sy_shift, 1.0,
               'frame aligned, two axes (x%.4f / y%.4f) (%s)'
               % (sx, sy, source.source), sx=sx, sy=sy)
