"""A page as a small picture, with no Xournal++ and no XP.

The arranger only needs cards that let a person recognise a page, so this
draws exactly what the format says: for every `<layer>`, in order, every
`<stroke>`'s polyline in its own colour at its own width, plus a disc at the
two ends for the round cap Xournal++ draws. A filled stroke is a polygon.

It is deliberately the simple reading of the format, not a replica of
Xournal++'s renderer: at 136 px a page is a shape, not a specimen, and the
one thing that must be exact -- the page's bytes -- never goes through this.
"""

from __future__ import annotations

import base64
import io
import xml.etree.ElementTree as ET
from typing import List, Tuple

from PIL import Image, ImageDraw

BACKGROUND = 'white'


def _colour(text: str) -> Tuple[int, int, int]:
    value = (text or '#000000FF').lstrip('#')
    try:
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
    except (ValueError, IndexError):
        return (0, 0, 0)


def _drawn_width(stroke: ET.Element) -> float:
    """One number for the width.

    Xournal++ writes `width` as "global width, then the pressure at each
    point"; a pressure-varying stroke drawn as one number wants the median of
    the per-point values, because the nominal one is often much larger than
    what the pen actually put down.
    """
    values = [float(v) for v in (stroke.get('width') or '1').split()]
    if len(values) > 1:
        pressures = sorted(values[1:])
        return pressures[len(pressures) // 2]
    return values[0]


def _points(stroke: ET.Element) -> List[Tuple[float, float]]:
    numbers = [float(v) for v in (stroke.text or '').split()]
    return list(zip(numbers[0::2], numbers[1::2]))


def render_page(page: ET.Element, scale: float) -> Image.Image:
    """One `<page>` element as an RGB image, `scale` pixels per page unit."""
    width = max(1, round(float(page.get('width', 595.275591)) * scale))
    height = max(1, round(float(page.get('height', 841.889764)) * scale))
    image = Image.new('RGB', (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    for layer in page.findall('layer'):
        for child in layer:
            if child.tag == 'stroke':
                _draw_stroke(draw, child, scale)
            elif child.tag == 'image':
                _draw_image(image, child, scale)
    return image


def _draw_image(canvas: Image.Image, element: ET.Element, scale: float) -> None:
    """Paste an inline base64 image at its page box, scaled for the card."""
    try:
        left = float(element.get('left'))
        top = float(element.get('top'))
        right = float(element.get('right'))
        bottom = float(element.get('bottom'))
        payload = base64.b64decode(''.join((element.text or '').split()))
        picture = Image.open(io.BytesIO(payload)).convert('RGBA')
    except Exception:
        return
    x0, y0 = round(left * scale), round(top * scale)
    width = round((right - left) * scale)
    height = round((bottom - top) * scale)
    if width <= 0 or height <= 0:
        return
    picture = picture.resize((width, height), Image.Resampling.LANCZOS)
    canvas.paste(picture, (x0, y0), picture)


def _draw_stroke(draw: ImageDraw.ImageDraw, stroke: ET.Element,
                 scale: float) -> None:
    points = _points(stroke)
    if not points:
        return
    draw_polyline(draw, points, scale, _colour(stroke.get('color')),
                  _drawn_width(stroke), stroke.get('fill') is not None)


def draw_polyline(draw: ImageDraw.ImageDraw, points, scale: float, colour,
                  width_units: float, fill: bool = False, origin=(0.0, 0.0)) -> None:
    """One stroke: its polyline, round joins, round caps, and its fill.

    Shared by the cards and the big preview, so the two cannot disagree about
    what a stroke looks like. `origin` is the page-unit corner of the image
    being drawn into, which is how the preview draws only its window.
    """
    ox, oy = origin
    if not points:
        return
    placed = [((x - ox) * scale, (y - oy) * scale) for x, y in points]
    width = max(1, round(width_units * scale))
    if fill and len(placed) >= 3:
        draw.polygon(placed, fill=colour)
        draw.line(placed + [placed[0]], fill=colour, width=width, joint='curve')
        return
    if len(placed) == 1 or (len(placed) == 2 and placed[0] == placed[1]):
        x, y = placed[0]
        r = width / 2.0
        draw.ellipse([x - r, y - r, x + r, y + r], fill=colour)
        return
    draw.line(placed, fill=colour, width=width, joint='curve')
    # The two ends are what `joint='curve'` does not cover: a round cap is a
    # disc of half the width, which is the same thing the interior turns get.
    r = width / 2.0
    for x, y in (placed[0], placed[-1]):
        draw.ellipse([x - r, y - r, x + r, y + r], fill=colour)
