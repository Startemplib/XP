"""Rasterize xopp pages to PIL images: one stroke drawn the way Xournal++
draws it, and nothing that knows what a merge is.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from xp.render import raster
from xp.core.document.xopp_io import Page, Stroke

DEFAULT_DISPLAY_SCALE = 4.0  # pixels per xopp unit, for on-screen review only

# The conversion window's difference overlay borrows these two; the merge
# window's editing palette lives in views.py, where the editing view is.
LEFT_ONLY_COLOR = (155, 17, 30)     # blood red: in the source, not in the tracing
RIGHT_ONLY_COLOR = (62, 166, 227)   # sky blue: in the tracing, not in the source
BACKGROUND_COLOR = "white"


class _Band:
    """What `raster` needs of a stroke: its points and its width.

    A `Stroke` carries a nominal width, a list of per-point widths, a colour
    as hex and a fill; none of that belongs in a rasterizer, so this hands
    over the two numbers it reads and nothing else.
    """

    __slots__ = ('points', 'width')

    def __init__(self, points, width):
        self.points, self.width = points, width


def to_pixels(points, scale: float) -> List[Tuple[float, float]]:
    return [(x * scale, y * scale) for x, y in points]


def _hex_to_rgb(hexcolor: str) -> Tuple[int, int, int]:
    h = hexcolor.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _drawn_width(stroke: Stroke) -> float:
    """The width one number has to stand for, when only one is drawn.

    A pressure-varying stroke's `width` is a nominal width followed by the
    width at each point, and the nominal one is not what you see: on real
    notes it is nine times the median of the points that follow it. Drawing a
    whole stroke at it made ordinary handwriting come out as a broad blot.
    """
    if not stroke.widths:
        return 1.0
    if len(stroke.widths) > 1:
        return float(np.median(stroke.widths[1:]))
    return float(stroke.widths[0])


def _draw_varying(draw: ImageDraw.ImageDraw, pts, widths, scale: float, color) -> None:
    """A stroke whose width changes along it, segment by segment.

    This is what a pen with pressure leaves, and what Xournal++ draws. Each
    segment gets its own width, with a round cap at the join so the stroke
    reads as one line rather than a row of tapered pieces.
    """
    for index in range(len(pts) - 1):
        width = widths[min(index, len(widths) - 1)]
        pixels = max(1, round(float(width) * scale))
        draw.line([pts[index], pts[index + 1]], fill=color, width=pixels)
        radius = pixels / 2
        # Both ends, not only the join: cairo caps the whole stroke, so the
        # first segment's start and the last one's end need a disc too.
        ends = [pts[index + 1]]
        if index == 0:
            ends.append(pts[0])
        for x, y in ends:
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)


def _round_joins(draw: ImageDraw.ImageDraw, points, width_px: int, fill) -> None:
    """Cairo's round cap and round join, which are the same disc.

    A stroke of constant width along a polyline is that polyline grown by a
    disc of half the width -- caps at the two ends and joins at every turn
    are all that one statement. PIL draws the quadrilaterals and a wedge at
    the joints; the wedge is right for a gentle bend and open for a sharp
    one, so the disc is drawn at every point. It costs one ellipse per point,
    against the two per stroke it replaces, and a page has fewer points than
    it has stroke ends once the strokes are joined.
    """
    radius = width_px / 2
    for x, y in points:
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill)


def draw_stroke(draw: ImageDraw.ImageDraw, stroke: Stroke, scale: float,
                 color_override: Optional[Tuple[int, int, int]] = None,
                 offset: Tuple[float, float] = (0.0, 0.0)) -> None:
    """`offset` shifts the stroke into the coordinates of a cropped region,
    for redrawing part of a page without touching the rest of it."""
    ox, oy = offset
    pts = [(x * scale - ox, y * scale - oy) for x, y in stroke.points]
    color = color_override if color_override is not None else _hex_to_rgb(stroke.color)
    width_px = max(1, round(_drawn_width(stroke) * scale))
    if len(stroke.widths) > 1 and stroke.fill is None and len(pts) >= 2:
        _draw_varying(draw, pts, stroke.widths[1:], scale, color)
        return
    if stroke.fill is not None and len(pts) >= 3:
        # A filled stroke is a shape, not a line: Xournal++ fills the path's
        # interior and still draws its edge, so this has to do both or the
        # rasterization the tracer is judged against won't be what Xournal++
        # will actually show.
        draw.polygon(pts, fill=color)
        draw.line(list(pts) + [pts[0]], fill=color, width=width_px, joint="curve")
    elif len(pts) == 1 or (len(pts) == 2 and pts[0] == pts[1]):
        x, y = pts[0]
        r = width_px / 2
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    else:
        draw.line(pts, fill=color, width=width_px, joint="curve")
        # The same discs `segment_ink` draws, for the same reason: cairo
        # finishes a stroke with a disc at each end and joins it with a disc
        # at each turn, and PIL's `line` lays down the bare quadrilateral
        # with a wedge at the joints that does not close a sharp one. Having
        # it in one of the two and not the other is worse than having it in
        # neither -- the fill's ruler and the picture a person looks at would
        # disagree, which is what tests/test_trace_fidelity.py exists to
        # catch.
        _round_joins(draw, pts, width_px, color)


# A <stroke>'s coordinate text has a length limit, so a long path is written
# as several strokes sharing a point rather than thinned (convert.
# segments_to_strokes). Where the pieces meet, PIL caps each one separately,
# which is not the same picture as one unbroken line -- so anything that needs
# to know what will be drawn has to cut the same way. Both callers use these.
STROKE_CHARS_PER_POINT = 24   # a conservative bound for "X.XXXXXX Y.YYYYYY "


def path_limit(max_coord_str_len: int) -> int:
    """How many points one <stroke> may carry."""
    return max(2, max(1, int(max_coord_str_len) - 80) // STROKE_CHARS_PER_POINT)


def cut_path(points, limit: int):
    """A path as the pieces it will be written as, each sharing its last point
    with the next one's first, so they join up."""
    start = 0
    while start < len(points) - 1 or (start == 0 and len(points) == 1):
        piece = points[start:start + limit]
        yield piece
        if len(piece) < limit:
            return
        start += limit - 1


def segment_ink(segments, size, default_width: float = 1.0,
                max_coord_str_len: int = 30000):
    """Where a tracing's segments actually land, as a boolean mask.

    The ruler. Every stage measures through here: the fill loop asks it what
    it has covered, stage 16 asks it what is bare and what is outside, and
    the panel's two percentages come from the page it draws.

    What it draws is the sentence in `raster`: the points within half the
    width of the polyline, which is what cairo draws and so what Xournal++
    shows. It used to be one `ImageDraw` call per stroke, and that was wrong
    three ways at once -- 101,773 Python calls where the arithmetic is
    trivial, a band fatter than the geometry (161,139 px outside the ink on
    the user's page against the band's 2,295), and a band of even width that moves
    a row depending on which direction the segment was drawn in.

    `max_coord_str_len` is still taken and no longer used for the drawing. A
    path written as several `<stroke>` elements sharing a point draws the
    same ink as one unbroken path, because a shared point is a disc either
    way -- so the ruler no longer has to imitate the cutting. The cutting
    still happens on the way out, where the length limit is real.

    A *filled* segment is a shape rather than a line and still goes through
    PIL: `polygon` has no equivalent here, and nothing this pipeline produces
    uses it -- only the older `outline` tracing does.
    """
    width, height = int(size[0]), int(size[1])
    plain = [seg for seg in segments if getattr(seg, 'fill', None) is None]
    shapes = [seg for seg in segments if getattr(seg, 'fill', None) is not None]
    out = raster.ink(plain, (width, height), default_width)
    if not shapes:
        return out
    image = Image.new("1", (max(1, width), max(1, height)), 0)
    draw = ImageDraw.Draw(image)
    limit = path_limit(max_coord_str_len)
    for segment in shapes:
        whole = np.round(np.asarray(segment.points, dtype=np.float64), 6).tolist()
        if len(whole) < 3:
            continue
        pixels = max(1, round(float(segment.width if segment.width else default_width)))
        stride = max(1, -(-len(whole) // limit))
        points = whole[::stride]
        draw.polygon(points, fill=1)
        draw.line(points + [points[0]], fill=1, width=pixels, joint="curve")
    return out | np.array(image, dtype=bool)


def _simple(stroke: Stroke) -> bool:
    """A stroke of one width and no fill -- which is all this pipeline makes."""
    return stroke.fill is None and len(stroke.widths) <= 1


def draw_strokes(image: Image.Image, strokes, scale: float, colours=None,
                 offset: Tuple[float, float] = (0.0, 0.0)) -> Image.Image:
    """Draw strokes into an RGB image, through the one ruler.

    Every picture a person looks at comes through here, and so does every
    picture something measures, because they have to be the same picture:
    `tests/test_previews` compares the Final Preview against the export, and
    the conversion panel measures its two ink percentages off the page it
    renders. Two rasterizers means two pages, and the user would be deciding what
    to keep by looking at one of them and exporting the other.

    `colours` overrides each stroke's own colour, one per stroke, for the
    views that paint a decision rather than the ink. A stroke whose geometry
    this cannot express -- a fill, or a width that varies along it -- sends
    the whole call back to the per-stroke path, so those pages stay exactly
    as they were rather than half one thing and half another.
    """
    strokes = list(strokes)
    if not strokes:
        return image
    if image.mode != "RGB" or not all(_simple(stroke) for stroke in strokes):
        draw = ImageDraw.Draw(image)
        for index, stroke in enumerate(strokes):
            draw_stroke(draw, stroke, scale,
                        None if colours is None else colours[index], offset)
        return image
    ox, oy = offset
    bands = [_Band([(x * scale - ox, y * scale - oy) for x, y in stroke.points],
                   max(1.0, _drawn_width(stroke) * scale))
             for stroke in strokes]
    tints = [(_hex_to_rgb(stroke.color) if colours is None or colours[index] is None
              else colours[index])
             for index, stroke in enumerate(strokes)]
    pixels = np.array(image, np.uint8)
    raster.paint(pixels, bands, tints, 1.0)
    image.paste(Image.fromarray(pixels, "RGB"))
    return image


def _image_objects(page):
    """Every `<image>` on the page, with its box, in every layer."""
    found = []
    for layer in page.layers() if hasattr(page, 'layers') else [page.layer]:
        for element in layer.findall('image'):
            try:
                box = tuple(float(element.get(name, '0'))
                            for name in ('left', 'top', 'right', 'bottom'))
            except (TypeError, ValueError):
                continue
            if box[2] <= box[0] or box[3] <= box[1] or not (element.text or '').strip():
                continue
            found.append((element, box))
    return found


def draw_pictures(image: Image.Image, page, scale: float,
                  offset: Tuple[float, float] = (0.0, 0.0)) -> Image.Image:
    """Paste the page's `<image>` objects into an RGB image.

    **A page is its pictures as well as its strokes.** `render_page` drew only
    strokes, so a notebook page carrying an embedded picture came back as the
    writing with a hole where the picture was -- which is why the merge used to
    trace every picture into strokes *before* it could be looked at. The user's
    specification says a picture in an XOPP is simply part of that side's image
    during editing, and only the selected part is converted at export; this is
    the "simply part of the image" half.

    Drawn under the strokes, because that is the order Xournal++ shows for the
    common case (a picture placed on the page, then written over) and because
    the comparison and the export both only need the *union* of the ink.
    """
    if not hasattr(page, 'layers') and not hasattr(page, 'layer'):
        return image
    import base64
    import io

    for element, box in _image_objects(page):
        try:
            png = base64.b64decode((element.text or '').strip())
            with Image.open(io.BytesIO(png)) as opened:
                picture = opened.convert('RGB')
        except Exception:
            continue                       # a picture that will not open is not ink
        left = (box[0] - offset[0]) * scale
        top = (box[1] - offset[1]) * scale
        want = (max(1, int(round((box[2] - box[0]) * scale))),
                max(1, int(round((box[3] - box[1]) * scale))))
        if picture.size != want:
            picture = picture.resize(want, Image.LANCZOS)
        image.paste(picture, (int(round(left)), int(round(top))))
    return image


def render_page(page: Page, scale: float = DEFAULT_DISPLAY_SCALE,
                smooth: int = 1) -> Image.Image:
    """A page, drawn the way Xournal++ draws it.

    With `smooth` above 1 it is drawn that many times over and scaled back
    down, since PIL has no antialiasing of its own. Off by default, and asked
    for by the one caller that shows a page to a person (convertgui's
    render_traced): a measurement has to see solid ink, not a grey suggestion
    of it, and two callers each smoothing on their own would draw the page
    six times over to show it once.
    """
    width = max(1, round(page.width * scale))
    height = max(1, round(page.height * scale))
    smooth = max(1, int(smooth))
    image = Image.new("RGB", (width * smooth, height * smooth), BACKGROUND_COLOR)
    draw_pictures(image, page, scale * smooth)
    draw_strokes(image, page.strokes(), scale * smooth)
    if smooth > 1:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def render_page_view(page: Page, scale: float, box,
                     smooth: int = 1) -> Image.Image:
    """One rectangle of a page, drawn at its own resolution.

    `render_page` sizes the canvas to the whole page, so asking it for a
    close-up at 6000 dpi allocates the whole page -- 510 megapixels for a Q
    base sheet -- to show a 700-pixel pane. This draws only `box` (page units,
    x0, y0, x1, y1) and keeps the vectors at `scale` pixels per unit, so a
    zoomed pane is rendered from the strokes themselves instead of magnifying
    a stored thumbnail.

    The two offsets are not the same quantity and are passed to their own
    function: `draw_pictures` subtracts page units, `draw_strokes` subtracts
    pixels (the same distinction that put an embedded picture 4,000 pixels
    off the merge canvas). Both are given the crop's own corner.
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    width = max(1, int(round((x1 - x0) * scale)))
    height = max(1, int(round((y1 - y0) * scale)))
    smooth = max(1, int(smooth))
    image = Image.new("RGB", (width * smooth, height * smooth), BACKGROUND_COLOR)
    draw_pictures(image, page, scale * smooth, offset=(x0, y0))
    # A page of handwriting has ~100,000 strokes and a zoomed pane sees a
    # few hundred of them. The rasterizer clips, but it still walks every
    # band; taking the ones whose box cannot reach the window out here is
    # what keeps a wheel step from redrawing the whole page.
    margin = 2.0
    visible = []
    for stroke in page.strokes():
        points = stroke.points
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        if (max(xs) + margin < x0 or min(xs) - margin > x1
                or max(ys) + margin < y0 or min(ys) - margin > y1):
            continue
        visible.append(stroke)
    draw_strokes(image, visible, scale * smooth,
                 offset=(x0 * scale * smooth, y0 * scale * smooth))
    if smooth > 1:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image
