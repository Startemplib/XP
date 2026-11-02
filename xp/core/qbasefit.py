"""Q base geometry from a PDF page: the three steps, with lines instead of a box.

The user's mission (Miss_3): **Binarization -> Background Boundary Detection -> Qbase
Rectangle Fitting**, "do not assume X/Y are the same", recovering scale X, scale Y,
translation and **rotation**.

The first version took the bounding box of "not paper" as the rectangle. The
distorted samples showed what that costs: a border partly off the page, or a
rotated one, gives a box that is not the border -- 9 % on a translation, 19 % on a
seven-degree rotation. So each of the four edges is now fitted as a **line**:

* an edge's profile is the first non-paper pixel per column (top/bottom) or per row
  (left/right), over the middle of the side so the corners cannot bend it;
* the line gives the edge's angle and its straightness; an edge whose profile covers
  too little of its side is **incomplete**, not a rectangle;
* the page is de-rotated by the top edge's angle, the rectangle is measured there,
  and everything is rotated back -- so scale, offset and angle all come out, and a
  shear shows as the top and left angles not being ninety degrees apart.

Only a PDF page is asked: an XOPP written by XP has no printed border in it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from xp.core.frame import REFERENCE

Q_BASE = (187.8236, 391.3654)
WHITE = 200            # everything darker than this is not paper
MIN_COVERAGE = 0.6     # an edge seen over less of its side than this is not an edge
MAX_RESIDUAL = 3.0     # pixels; a bent "edge" is not a line
SHEAR_REFUSED_DEGREES = 3.0   # beyond this the transform cannot express the page


@dataclass
class Edge:
    slope: float           # in the image's own axes
    intercept: float
    coverage: float        # share of the side the profile spans
    residual: float        # RMS distance from the fitted line, in pixels

    @property
    def is_valid(self) -> bool:
        return self.coverage >= MIN_COVERAGE and self.residual <= MAX_RESIDUAL


@dataclass
class QbaseFit:
    sx: float
    sy: float
    dx: float
    dy: float
    angle_deg: float
    shear_deg: float
    box: Tuple[float, float, float, float]         # in points, unrotated
    corners: List[Tuple[float, float]] = field(default_factory=list)
    source: str = 'border'
    incomplete: bool = False
    from_page: int = 0            # which page this was measured on
    pages_measured: int = 1       # how many pages were looked at to find it

    def is_the_page(self) -> bool:
        return (abs(self.sx - 1.0) < 1e-3 and abs(self.sy - 1.0) < 1e-3
                and abs(self.angle_deg) < 1e-3)

    def place(self, ref_x: float, ref_y: float) -> Tuple[float, float]:
        """Where a Q base point lands on this page, rotation included."""
        x = ref_x / self.sx + self.dx
        y = ref_y / self.sy + self.dy
        if abs(self.angle_deg) < 1e-9:
            return x, y
        # **Counter-clockwise in the page's frame (y up).** In the raster's axes,
        # where y points down, that is the matrix with the signs swapped; using the
        # other one turned every rotated sample's corners the wrong way (measured:
        # 115-265 pt of error with the angle itself correct to 0.2 degrees).
        a = math.radians(self.angle_deg)
        cx = 0.5 * (self.box[0] + self.box[2])
        cy = 0.5 * (self.box[1] + self.box[3])
        return (cx + (x - cx) * math.cos(a) + (y - cy) * math.sin(a),
                cy - (x - cx) * math.sin(a) + (y - cy) * math.cos(a))
    # **The four corners as they were found on the page**, in page points, in the order
    # top-left, top-right, bottom-right, bottom-left. `corners` above is where they are
    # *placed*; this is where they *are*, and it is what a warp needs.
    detected: Optional[List[Tuple[float, float]]] = None



def binarize(grey: np.ndarray, threshold: int = WHITE) -> np.ndarray:
    """Step 1: paper (0) against everything else (1)."""
    array = np.asarray(grey)
    if array.ndim == 3:
        array = array.min(axis=2)
    return array < threshold


def background_boundary(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Step 2: where the outer band of paper stops."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _profile(mask: np.ndarray, box, side: str):
    """First non-paper coordinate per column (top/bottom) or per row (left/right)."""
    x0, y0, x1, y1 = box
    width, height = x1 - x0 + 1, y1 - y0 + 1
    # **The middle 40 %.** A rotated rectangle's axis-aligned bounding box puts a
    # corner into the outer fifth of every side, so a profile taken that wide is
    # bent at both ends and no line fits it -- the seven-degree sample came back as
    # `incomplete` with the angle at zero because of exactly this.
    margin = max(1, int(0.30 * (width if side in ('top', 'bottom') else height)))
    if side in ('top', 'bottom'):
        columns = list(range(x0 + margin, x1 - margin + 1))
        found = []
        for column in columns:
            strip = mask[y0:y1 + 1, column]
            rows = np.nonzero(strip)[0]
            if not len(rows):
                continue
            row = rows[0] if side == 'top' else rows[-1]
            found.append((float(column), float(row + y0)))
        return found, len(columns)
    rows_range = list(range(y0 + margin, y1 - margin + 1))
    found = []
    for row in rows_range:
        strip = mask[row, x0:x1 + 1]
        columns = np.nonzero(strip)[0]
        if not len(columns):
            continue
        column = columns[0] if side == 'left' else columns[-1]
        found.append((float(row), float(column + x0)))
    return found, len(rows_range)


def _fit_edge(points, count: int) -> Edge:
    if len(points) < 3 or count <= 0:
        return Edge(0.0, 0.0, 0.0, 1e9)
    t = np.array([p[0] for p in points], dtype=float)
    v = np.array([p[1] for p in points], dtype=float)
    try:
        slope, intercept = np.polyfit(t, v, 1)
        # One robust pass: a corner or a speck that survived the middle-of-the-side
        # cut must not drag the line. Points beyond twice the median residual are
        # dropped and the line is refitted once.
        residual = v - (slope * t + intercept)
        median = float(np.median(np.abs(residual))) or 1e-9
        keep = np.abs(residual) <= 2.0 * median + 0.5
        if keep.sum() >= 3 and not keep.all():
            slope, intercept = np.polyfit(t[keep], v[keep], 1)
    except Exception:
        return Edge(0.0, 0.0, 0.0, 1e9)
    residual = float(np.sqrt(np.mean((v - (slope * t + intercept)) ** 2)))
    return Edge(float(slope), float(intercept), len(points) / float(count), residual)


def _refine(mask: np.ndarray, box, side: str, rough: Edge) -> Edge:
    """Refit that edge on the border's own pixels, by **orthogonal** regression.

    The per-column profile above is a good first guess and a biased second one: it
    takes the first non-paper pixel of every column, which on a rotated page walks
    along the border's antialiased edge and adds a slope error. The intersections of
    two such lines amplify that error -- measured as a rotated sample needing 1.25 in
    x and coming out 1.55.

    So the second pass collects the pixels **within a few pixels of the rough line**
    over the middle 80 % of the side and fits a line through them by principal
    direction, which has no preferred axis and therefore no such bias. Heavy, it is
    not: a few thousand pixels, one eigendecomposition.
    """
    x0, y0, x1, y1 = box
    width, height = x1 - x0 + 1, y1 - y0 + 1
    margin = max(1, int(0.10 * (width if side in ('top', 'bottom') else height)))
    band = 3.0
    if side in ('top', 'bottom'):
        columns = np.arange(x0 + margin, x1 - margin + 1)
        rows_expected = rough.slope * columns + rough.intercept
        offsets = np.arange(-band, band + 1)
        rows = np.clip(rows_expected[:, None] + offsets[None, :], y0, y1).astype(int)
        cols = np.repeat(columns[:, None], offsets.size, axis=1)
        keep = mask[rows, cols]
        if keep.sum() < 10:
            return rough
        t = cols[keep].astype(float)
        v = rows[keep].astype(float)
    else:
        rows_range = np.arange(y0 + margin, y1 - margin + 1)
        cols_expected = rough.slope * rows_range + rough.intercept
        offsets = np.arange(-band, band + 1)
        cols = np.clip(cols_expected[:, None] + offsets[None, :], x0, x1).astype(int)
        rr = np.repeat(rows_range[:, None], offsets.size, axis=1)
        keep = mask[rr, cols]
        if keep.sum() < 10:
            return rough
        t = rr[keep].astype(float)
        v = cols[keep].astype(float)
    centred_t = t - t.mean()
    centred_v = v - v.mean()
    cov = np.cov(np.vstack([centred_t, centred_v]))
    values, vectors = np.linalg.eigh(cov)
    direction = vectors[:, int(np.argmax(values))]
    if abs(direction[0]) < 1e-9:
        return rough
    slope = float(direction[1] / direction[0])
    intercept = float(v.mean() - slope * t.mean())
    residual = float(np.sqrt(np.mean((v - (slope * t + intercept)) ** 2)))
    return Edge(slope, intercept, rough.coverage, residual)


def edges_of(mask: np.ndarray, box) -> dict:
    """The four edges: a profile first, then a fit on the border's own pixels."""
    sides = {}
    for side in ('top', 'bottom', 'left', 'right'):
        points, count = _profile(mask, box, side)
        rough = _fit_edge(points, count)
        sides[side] = _refine(mask, box, side, rough)
    return sides


def is_a_border(mask: np.ndarray, box) -> bool:
    """Is this rectangle a printed outline, or a block of content?

    **Two conditions, both measured on the four sample pages.** The
    rectangle must have a **straight line on each of its four sides**, and its
    **middle must be empty** -- a printed border is a thin outline around writing, not
    a filled block.

        page                  four straight sides?   ink in the middle 30 %  verdict
        a scanned notebook    yes (residual ~1 px)   5.1 %                  border
        Q base template       yes (residual ~0.4)    0.0 %                  border
        template on 400x800   yes (residual 0.0)     0.0 %                  border
        a page of text        no (sides 32 % seen)   11.4 %                 content
        a filled rectangle    yes                    100 %                  content

    Neither condition alone is enough: edges alone call a filled block a border, and a
    middle-density alone calls a page of handwriting "hollow" once it is rendered at a
    different scale -- the mistake that stretched the reference output by 1.556.
    """
    sides = edges_of(mask, box)
    if not all(sides[name].is_valid for name in ('top', 'bottom', 'left', 'right')):
        return False
    x0, y0, x1, y1 = box
    whole = mask[y0:y1 + 1, x0:x1 + 1]
    if not whole.size:
        return False
    inset_x = max(3, int(0.15 * whole.shape[1]))
    inset_y = max(3, int(0.15 * whole.shape[0]))
    inside = whole[inset_y:whole.shape[0] - inset_y,
                   inset_x:whole.shape[1] - inset_x]
    return bool(inside.size) and float(inside.mean()) <= 0.15


def _cross(top: Edge, left: Edge):
    """Where a horizontal-ish line and a vertical-ish line meet, in pixels."""
    denominator = 1.0 - left.slope * top.slope
    if abs(denominator) < 1e-9:
        return None
    x = (left.slope * top.intercept + left.intercept) / denominator
    return (x, top.slope * x + top.intercept)


def _distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fit_qbase(grey: np.ndarray, page: Tuple[float, float] = Q_BASE,
              reference: Tuple[float, float, float, float] = REFERENCE,
              threshold: int = WHITE) -> Optional[QbaseFit]:
    """Step 3: the four edges, onto the Q base writing area.

    **The corners are the intersections of the fitted lines.** Squaring a rotated
    bounding box up was measured at 29 pt and 42 pt of error with the two signs of
    the de-rotation; a line intersection is one point whatever the angle
    convention, so the sign cannot be lost. The edge lengths then give the two
    scales, and the corner mean gives the translation.
    """
    mask = binarize(grey, threshold)
    box = background_boundary(mask)
    if box is None:
        return None
    height, width = mask.shape
    edges = edges_of(mask, box)
    top, bottom = edges['top'], edges['bottom']
    left, right = edges['left'], edges['right']
    to_points_x = page[0] / float(width)
    to_points_y = page[1] / float(height)
    ref_x = reference[2] - reference[0]
    ref_y = reference[3] - reference[1]

    corners_px = None
    if top.is_valid and bottom.is_valid and left.is_valid and right.is_valid:
        top_left = _cross(top, left)
        top_right = _cross(top, right)
        bottom_left = _cross(bottom, left)
        bottom_right = _cross(bottom, right)
        if None not in (top_left, top_right, bottom_left, bottom_right):
            corners_px = [top_left, top_right, bottom_right, bottom_left]

    if corners_px is not None:
        # In points, and the page's own axes (y down, as the raster has it).
        corners = [(x * to_points_x, y * to_points_y) for x, y in corners_px]
        draw = 0.5 * (_distance(corners[0], corners[1])
                      + _distance(corners[3], corners[2]))
        rise = 0.5 * (_distance(corners[0], corners[3])
                      + _distance(corners[1], corners[2]))
        # The top edge's direction, in the raster's axes; a PDF rotation of +t
        # appears here as a negative slope (y points down), hence the sign.
        angle = -math.atan2(corners[1][1] - corners[0][1],
                            corners[1][0] - corners[0][0])
        centre = (sum(c[0] for c in corners) / 4.0, sum(c[1] for c in corners) / 4.0)
        sx = ref_x / max(draw, 1e-9)
        sy = ref_y / max(rise, 1e-9)
        # A shear shows as the two edges' angles not being square: measured as the
        # angle between the left edge and the top edge, less ninety degrees.
        # **Shear is the angle between the two edges, less ninety degrees.** Measured
        # against the top edge's own direction, so a pure rotation gives zero: the
        # first version compared the left edge's absolute angle with a right angle,
        # which reported -6 degrees on a page that was merely rotated by 3.
        # **Both angles in the raster's own frame.** `angle` is a page-frame rotation
        # (negated), so comparing it with a raster-space `atan2` reported -6 degrees of
        # shear on a page that was merely rotated by 3.
        top_angle = math.atan2(corners[1][1] - corners[0][1],
                               corners[1][0] - corners[0][0])
        left_angle = math.atan2(corners[3][1] - corners[0][1],
                                corners[3][0] - corners[0][0])
        shear = math.degrees(abs(left_angle - top_angle) - math.pi / 2.0)
        incomplete = False
    else:
        # An edge was missing or bent: fall back to the axis-aligned box and say so.
        ys, xs = np.nonzero(mask)
        corners = [(xs.min() * to_points_x, ys.min() * to_points_y),
                   (xs.max() * to_points_x, ys.min() * to_points_y),
                   (xs.max() * to_points_x, ys.max() * to_points_y),
                   (xs.min() * to_points_x, ys.max() * to_points_y)]
        draw = corners[1][0] - corners[0][0]
        rise = corners[3][1] - corners[0][1]
        angle = 0.0
        shear = 0.0
        sx = ref_x / max(draw, 1e-9)
        sy = ref_y / max(rise, 1e-9)
        centre = (0.5 * (corners[0][0] + corners[2][0]),
                  0.5 * (corners[0][1] + corners[2][1]))
        incomplete = True

    centre_ref = (0.5 * (reference[0] + reference[2]),
                  0.5 * (reference[1] + reference[3]))
    dx = centre[0] - centre_ref[0] / sx
    dy = centre[1] - centre_ref[1] / sy
    placed = []
    half_w = 0.5 * ref_x / sx
    half_h = 0.5 * ref_y / sy
    for ux, uy in ((-half_w, -half_h), (half_w, -half_h),
                   (half_w, half_h), (-half_w, half_h)):
        x = centre[0] + ux * math.cos(angle) + uy * math.sin(angle)
        y = centre[1] - ux * math.sin(angle) + uy * math.cos(angle)
        placed.append((x, y))
    # **Cut by the paper's edge is incomplete, whether or not the profiles notice.**
    # A clipped border's bounding box simply shrinks, so its four edges look fine and
    # the fit is confidently wrong; the tell is that the mask reaches the page edge.
    touches = (box[0] <= 1 or box[1] <= 1 or box[2] >= width - 2
               or box[3] >= height - 2)
    hollow = is_a_border(mask, box)
    if touches or not hollow:
        incomplete = True
    # **A sheared page is refused, not fitted.** The transform has no shear term, so
    # returning one for a parallelogram is a wrong answer that looks measured; the
    # samples showed 685 pt of corner error on a 20 % shear. Detected, named, refused.
    if abs(shear) > SHEAR_REFUSED_DEGREES:
        incomplete = True
    return QbaseFit(sx, sy, dx, dy, math.degrees(angle), shear,
                    (centre[0] - half_w, centre[1] - half_h,
                     centre[0] + half_w, centre[1] + half_h),
                    placed, 'border' if hollow else 'content', incomplete,
                    detected=corners)


def fit_document(pages: Sequence[np.ndarray],
                 page: Tuple[float, float] = Q_BASE,
                 reference: Tuple[float, float, float, float] = REFERENCE,
                 threshold: int = WHITE) -> Optional[QbaseFit]:
    """**One calibration, the whole document.** (The user's section 4.)

    Pages of one PDF share a convention, so the transform is measured on the first
    page whose border is complete and applied to all of them -- pages without a
    border (a plain scan, a text page) are not left to fit themselves, and no page
    is measured twice. `from_page` and `pages_measured` say which page answered and
    how many were looked at before it did, so the report can state it.
    """
    fallback = None
    content = None
    looked = 0
    for index, grey in enumerate(pages):
        if grey is None:
            continue
        looked += 1
        fit = fit_qbase(grey, page, reference, threshold)
        if fit is None:
            continue
        fit.from_page = index
        fit.pages_measured = looked
        # **A page whose "rectangle" is its own content is not a calibration.** A
        # text page has a bounding box too; fitting the document to it resized
        # everything to the text block (measured: sx 2.06, sy 5.71 on a page with six
        # lines). Only a rectangle inset from the paper counts as a border -- but the
        # content fit is kept as the *answer* so the refusal can say why.
        if fit.source != 'border':
            if content is None:
                content = fit
            continue
        if not fit.incomplete:
            return fit
        if fallback is None:
            fallback = fit
    return fallback if fallback is not None else content


@dataclass
class Policy:
    """What to do with a page's transform, and why."""

    sx: float = 1.0
    sy: float = 1.0
    dx: float = 0.0
    dy: float = 0.0
    angle_deg: float = 0.0
    applies: bool = False
    reason: str = ''

    def place(self, x: float, y: float) -> Tuple[float, float]:
        return (x / self.sx + self.dx, y / self.sy + self.dy)


def transform_for(fit: Optional[QbaseFit]) -> Policy:
    """**When the calibration may be applied, and when it must be left alone.**

    Measured on the two sample files, and this is the guard the integration needs:

    * `TEST.pdf` is a **1024 x 1366 pt** page whose rectangle is the handwriting plus
      a narrow border -- `source` is `content`, not `border`. Stretching it to fill
      the Q base writing area would move the reference from (27.5, 17.5)-(159.5,
      364.5) to somewhere else; the conversion already matches that reference within
      1 pt precisely *because* nothing is stretched.
    * `Qbase_66x138.pdf` is the sheet itself: 187.82 x 391.37 pt, a hollow border,
      `sx = sy = 1`. Applying its transform changes nothing, which is the correct
      answer for a page already on the sheet.

    So the transform applies only to a **complete fit from a real border**, and
    everything else is left at the page's own coordinates, with the reason attached
    for the status line.
    """
    if fit is None:
        return Policy(reason='no border found')
    if fit.source != 'border':
        return Policy(reason='the rectangle is content, not a border')
    if fit.incomplete:
        return Policy(reason='the border is cut by the paper edge')
    return Policy(fit.sx, fit.sy, fit.dx, fit.dy, fit.angle_deg, True,
                  'calibrated from the printed border')


def preflight(path, pages: int = 3, zoom: float = 1.5):
    """What the converter should do with a PDF, decided before any tracing.

    Renders up to `pages` pages, calibrates once with `fit_document`, and answers with
    the `Policy` -- whether to apply it, and why -- plus the page size in points.
    One function, so the converter and the merge can ask the same question and get
    the same answer.

    Returns `(policy, (width_pt, height_pt), from_page)`; `from_page` is -1 when no
    page answered.
    """
    # **The modern name.** \itz\ is PyMuPDF's old alias: it is not a package of
    # its own and equirements.txt\ names \pymupdf\, so an import of \itz\ here
    # was the one third-party name the requirements did not cover
    # (\	est_without_the_extras\ caught it). Imported as \itz\ so the rest of this
    # function reads as it did.
    import pymupdf as fitz

    document = fitz.open(str(path))
    try:
        size = (document[0].rect.width, document[0].rect.height)
        greys = []
        for index in range(min(pages, document.page_count)):
            pixmap = document[index].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            array = np.frombuffer(pixmap.samples, dtype=np.uint8)
            array = array.reshape(pixmap.height, pixmap.width, pixmap.n)
            greys.append(array[:, :, :3].min(axis=2))
    finally:
        document.close()
    # **The page's own size, not the Q base size.** `fit_qbase` divides pixels into
    # points with it; given 187.82 x 391.37 for a 1024 x 1366 pt page, the scales come
    # out 5.45 times wrong (measured 1.5537 against a true 0.2850).
    fit = fit_document(greys, page=size)
    policy = transform_for(fit)
    return policy, size, (-1 if fit is None else fit.from_page)


def warp_into_qbase(image, fit, dpi: float):
    """The page **rewritten into the Q base convention**, and the geometry that follows.

    The user's design: "after the auto alignment ... all images are already in the Q base convention,
    with correct size, x/y ratio and the corner region identified". Doing it on the image rather
    than on the coordinates is what removes the class of defect we kept finding -- a guide drawn
    in one coordinate system beside a picture cropped in another.

    Returns `(image, (scale, dx, dy))` where the geometry is now trivial: the warped page **is**
    the sheet, so `scale` is pixels per page unit and there is no offset. Returns the image
    untouched, and the caller's own geometry, when the fit cannot be used.
    """
    import cv2

    if fit is None or fit.detected is None:
        return None, None
    policy = transform_for(fit)
    # **No uniformity rule here.** A homography expresses unequal x and y exactly, so a page
    # squeezed 0.64 in x and 0.76 in y is warpable -- it is the *geometry* path that needs one
    # scale and therefore refuses it. The first version of this function carried the rule over
    # from there and refused the very sample that shows the warp working.
    if not policy.applies:
        return None, None
    px_per_unit = dpi / 72.0
    source = np.array([[x * px_per_unit, y * px_per_unit] for x, y in fit.detected],
                      dtype=np.float32)
    target = np.array([[REFERENCE[0] * px_per_unit, REFERENCE[1] * px_per_unit],
                       [REFERENCE[2] * px_per_unit, REFERENCE[1] * px_per_unit],
                       [REFERENCE[2] * px_per_unit, REFERENCE[3] * px_per_unit],
                       [REFERENCE[0] * px_per_unit, REFERENCE[3] * px_per_unit]],
                      dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, target)
    width = max(8, int(round(Q_BASE[0] * px_per_unit)))
    height = max(8, int(round(Q_BASE[1] * px_per_unit)))
    warped = cv2.warpPerspective(image, matrix, (width, height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255) if image.ndim == 3 else 255)
    return warped, (px_per_unit, 0.0, 0.0)
