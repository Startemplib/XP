"""One page's content, parsed once and drawn at any scale.

The arranger's cards only need shapes; the big preview needs the page. This
is the difference between the two: `PageArt` keeps every element of a page --
strokes, text, images and the background -- so the preview can draw it at
whatever zoom and pan are on screen, from the page's own data rather than by
enlarging a thumbnail.

Coordinates are page units (1/72"), which is what the file stores. `render`
takes a scale in pixels per unit and a box in page units, and returns only
that region -- so a zoomed preview costs the size of the window, not the size
of the page at 12x.
"""

from __future__ import annotations

import base64
import io
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageColor, ImageDraw, ImageFont

from .thumb import draw_polyline

DEFAULT_COLOUR = (0, 0, 0)


def _colour(text: Optional[str], default=DEFAULT_COLOUR):
    if not text:
        return default
    try:
        return ImageColor.getrgb(text)[:3]
    except (ValueError, AttributeError):
        return default


def _width_of(stroke: ET.Element) -> float:
    values = [float(v) for v in (stroke.get('width') or '1').split()]
    if len(values) > 1:
        pressures = sorted(values[1:])
        return pressures[len(pressures) // 2]
    return values[0]


def _points_of(element: ET.Element):
    numbers = [float(v) for v in (element.text or '').split()]
    return list(zip(numbers[0::2], numbers[1::2]))


def _bbox(points):
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return min(xs), min(ys), max(xs), max(ys)


class PageArt:
    """A page as drawable things, in layer order."""

    def __init__(self, element: ET.Element, base_dir=None, resolve_pdf=None):
        self.width = float(element.get('width', 595.275591))
        self.height = float(element.get('height', 841.889764))
        self.base_dir = Path(base_dir) if base_dir else None
        # `resolve_pdf(original, folder) -> path or None` lets the window ask
        # about a background that moved, without this module knowing about Tk.
        self.resolve_pdf = resolve_pdf
        self.background = self._background(element)
        self.items: List[tuple] = []
        for layer in element.findall('layer'):
            for child in layer:
                if child.tag == 'stroke':
                    item = self._stroke(child)
                elif child.tag == 'text':
                    item = self._text(child)
                elif child.tag == 'image':
                    item = self._image(child)
                else:
                    item = None
                if item is not None:
                    self.items.append(item)
        self.stroke_count = sum(1 for kind, _ in self.items if kind == 'stroke')
        self._fonts = {}

    # ---------------- parsing ----------------

    def _background(self, page: ET.Element):
        element = page.find('background')
        if element is None:
            return ('solid', (255, 255, 255))
        kind = (element.get('type') or 'solid').lower()
        if kind == 'solid':
            name = element.get('color') or 'white'
            try:
                rgb = ImageColor.getrgb(name)[:3]
            except (ValueError, AttributeError):
                rgb = (255, 255, 255)
            return ('solid', rgb)
        filename = element.get('filename') or ''
        if not filename:
            return ('solid', (255, 255, 255))
        domain = (element.get('domain') or 'absolute').lower()
        if domain == 'attach':
            # The PDF lives beside the notebook as `<file>.xopp.<name>`; a
            # gzip `.xopp` cannot carry an attachment, and this program does
            # not write one, so there is nothing to open.
            return ('missing', filename)
        path = Path(filename)
        if not path.is_absolute() and self.base_dir is not None:
            path = self.base_dir / filename
        page_number = int(element.get('pageno') or 1)
        if path.is_file():
            return ('pdf', str(path), page_number)
        if self.resolve_pdf is not None:
            found = self.resolve_pdf(path, self.base_dir)
            if found is not None and Path(found).is_file():
                return ('pdf', str(found), page_number)
        return ('missing', str(path))

    def _stroke(self, stroke: ET.Element):
        points = _points_of(stroke)
        if not points:
            return None
        box = _bbox(points)
        colour = _colour(stroke.get('color'))
        fill = stroke.get('fill') is not None
        return ('stroke', (box, points, colour, _width_of(stroke), fill))

    def _text(self, element: ET.Element):
        content = element.text or ''
        if not content.strip():
            return None
        x = float(element.get('x') or 0.0)
        y = float(element.get('y') or 0.0)
        size = float(element.get('size') or 12.0)
        colour = _colour(element.get('color'))
        font = element.get('font') or 'Sans'
        return ('text', ((x, y, x, y + size), (x, y), font, size, colour, content))

    def _image(self, element: ET.Element):
        try:
            left = float(element.get('left'))
            top = float(element.get('top'))
            right = float(element.get('right'))
            bottom = float(element.get('bottom'))
            payload = base64.b64decode(''.join((element.text or '').split()))
            image = Image.open(io.BytesIO(payload)).convert('RGBA')
        except Exception:
            return None
        return ('image', ((left, top, right, bottom), image))

    # ---------------- drawing ----------------

    def _font(self, size_px: int):
        held = self._fonts.get(size_px)
        if held is not None:
            return held
        size_px = max(1, int(round(size_px)))
        for name in ('times.ttf', 'Times New Roman.ttf', 'arial.ttf',
                     'segoeui.ttf'):
            try:
                held = ImageFont.truetype(name, size_px)
                break
            except OSError:
                continue
        else:
            try:
                held = ImageFont.load_default(size_px)
            except TypeError:              # older Pillow
                held = ImageFont.load_default()
        self._fonts[size_px] = held
        return held

    def render(self, scale: float, box=None) -> Image.Image:
        """The page (or `box` = (x0, y0, x1, y1) in page units) at `scale`."""
        x0, y0, x1, y1 = box if box else (0.0, 0.0, self.width, self.height)
        width = max(1, round((x1 - x0) * scale))
        height = max(1, round((y1 - y0) * scale))
        image = Image.new('RGB', (width, height), (255, 255, 255))
        self._paint_background(image, scale, (x0, y0, x1, y1))
        draw = ImageDraw.Draw(image, 'RGBA')
        for kind, payload in self.items:
            if kind == 'stroke':
                stroke_box, points, colour, stroke_width, fill = payload
                if not _overlaps(stroke_box, (x0, y0, x1, y1), stroke_width):
                    continue
                draw_polyline(draw, points, scale, colour, stroke_width, fill,
                              origin=(x0, y0))
            elif kind == 'text':
                text_box, (x, y), font_name, size, colour, content = payload
                if not _overlaps(text_box, (x0, y0, x1, y1), size):
                    continue
                draw.text(((x - x0) * scale, (y - y0) * scale), content,
                          font=self._font(size * scale), fill=colour + (255,),
                          anchor='ls')
            elif kind == 'image':
                image_box, picture = payload
                if not _overlaps(image_box, (x0, y0, x1, y1), 0):
                    continue
                left, top, right, bottom = image_box
                target = (round((left - x0) * scale), round((top - y0) * scale),
                          round((right - x0) * scale), round((bottom - y0) * scale))
                if target[2] > target[0] and target[3] > target[1]:
                    picture = picture.resize((target[2] - target[0],
                                              target[3] - target[1]),
                                             Image.Resampling.LANCZOS)
                    image.paste(picture, (target[0], target[1]), picture)
        return image

    def _paint_background(self, image, scale, box) -> None:
        x0, y0, x1, y1 = box
        kind = self.background[0]
        if kind == 'pdf':
            _pdf_background(image, scale, box, self.background[1],
                            self.background[2])
            return
        if kind == 'solid':
            image.paste(Image.new('RGB', image.size, self.background[1]))
            return
        # missing: white paper is the honest thing to draw, and the header of
        # the preview says the file was not found.

    def background_missing(self) -> Optional[str]:
        return self.background[1] if self.background[0] == 'missing' else None


def _overlaps(box, view, pad: float) -> bool:
    bx0, by0, bx1, by1 = box
    vx0, vy0, vx1, vy1 = view
    return not (bx1 + pad < vx0 or bx0 - pad > vx1
                or by1 + pad < vy0 or by0 - pad > vy1)


def _pdf_background(image, scale, box, path, page_number) -> None:
    """The PDF behind the page, clipped to `box`, at `scale`.

    PyMuPDF is optional here: without it (or with the PDF missing) the paper
    stays white and the preview's header says which file it wanted.
    """
    try:
        import pymupdf
    except ImportError:
        return
    try:
        with pymupdf.open(path) as pdf:
            if not (1 <= page_number <= pdf.page_count):
                return
            x0, y0, x1, y1 = box
            page = pdf[page_number - 1]
            clip = pymupdf.Rect(x0, y0, x1, y1)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                                  colorspace=pymupdf.csRGB, clip=clip)
            patch = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
            image.paste(patch, (0, 0))
    except Exception:
        return
