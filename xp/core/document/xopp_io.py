"""Load/save .xopp files (gzip + XML) with the strokes exposed for rendering,
hit-testing and editing, while keeping every other element (background, text,
images, other layers) round-tripped untouched via the underlying ElementTree.
"""

from __future__ import annotations

import copy
import math
import gzip
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

DEFAULT_PAGE_WIDTH = 187.8236
DEFAULT_PAGE_HEIGHT = 391.3654
# Xournal++ refuses to open a file that contains a stroke with fewer than two
# points ("LoadHandler::parseXml: Wrong count of points"), and refusing means
# the whole document fails to load, not just that one stroke.
MIN_STROKE_POINTS = 2


@dataclass
class Stroke:
    """A view over a <stroke> element. `element` is the source of truth."""

    element: ET.Element
    points: List[Tuple[float, float]]
    color: str
    widths: List[float]
    fill: Optional[int] = None      # 0-255: the path's interior is filled

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)


def _parse_stroke(elem: ET.Element) -> Stroke:
    coords = list(map(float, (elem.text or "").split()))
    points = list(zip(coords[0::2], coords[1::2]))
    widths = [float(w) for w in elem.get("width", "1").split()]
    color = elem.get("color", "#000000FF")
    fill = elem.get("fill")
    return Stroke(element=elem, points=points, color=color, widths=widths,
                  fill=None if fill is None else int(fill))


# How many decimals a coordinate is written with. A page is 187.82 units
# across and renders to 11,378 pixels, so one unit is sixty pixels: four
# decimals is a hundred-and-sixtieth of a pixel, and the two after that were
# bytes and nothing else. On the user's page, 393,464 points, dropping them takes
# the file from 3.22 MB to 2.44 MB.
#
# Three decimals would take it to 2.22 MB and is where this stopped being
# free: a sixteenth of a pixel is enough to push a coordinate across a pixel
# boundary, and `tests/test_trace_fidelity` found one point of 393,464 where
# it did -- the ruler and the renderer then disagree about one pixel, which
# is the thing this project has paid for twice already. The bytes are not
# worth it.
COORD_DECIMALS = 4


def make_stroke_element(points, color, width, tool="pen", cap_style="round",
                        fill=None) -> ET.Element:
    """One <stroke> element.

    `fill` (0-255) fills the path's interior, which is how Xournal++ draws a
    filled shape; None leaves it a plain line.

    A stroke is never written with fewer than MIN_STROKE_POINTS points: a
    dot becomes two points in the same place, which Xournal++ draws as a
    round dot and, unlike a one-point stroke, will actually open.
    """
    points = list(points)
    if not points:
        raise ValueError("a stroke needs at least one point")
    while len(points) < MIN_STROKE_POINTS:
        points.append(points[-1])
    attributes = {"tool": tool, "color": color, "width": str(width), "capStyle": cap_style}
    if fill is not None:
        attributes["fill"] = str(int(fill))
    elem = ET.Element("stroke", attributes)
    elem.text = " ".join(f"{x:.{COORD_DECIMALS}f} {y:.{COORD_DECIMALS}f}"
                         for x, y in points)
    elem.tail = "\n"
    return elem


def transform_stroke_element(elem: ET.Element, scale: float,
                             dx: float = 0.0, dy: float = 0.0,
                             scale_y: Optional[float] = None) -> ET.Element:
    """A copy of `elem` moved into another page's coordinates.

    One scale and a shift by default: a factor per axis fits a page better and
    distorts the handwriting on it, which is why this was uniform. `scale_y`
    exists for the one case where the *document* really is stretched -- a PDF
    printed or cropped at a different aspect -- and the frame says so: then a
    single number leaves the writing stretched the wrong way, and the geometry
    that came off the paper is the truth. Widths follow the mean of the two, so
    a pen stays the same pen.
    """
    copied = copy.deepcopy(elem)
    values = [float(value) for value in (copied.text or "").split()]
    sx = float(scale)
    sy = sx if scale_y is None else float(scale_y)
    placed = []
    for index in range(0, len(values) - 1, 2):
        placed.append(f"{values[index] * sx + dx:.6f}")
        placed.append(f"{values[index + 1] * sy + dy:.6f}")
    copied.text = " ".join(placed)
    widths = copied.get("width")
    if widths:
        # **The same rule the display uses.** The in-memory stroke's width is
        # multiplied by `Fit.pen_scale`, the mean of the two axes; the XML used
        # the X scale alone, so a page stretched 2:4 drew a width of 6 and
        # exported a width of 4. One rule, in one place: the mean.
        pen = (sx + sy) / 2
        copied.set("width", " ".join(f"{float(w) * pen:.6f}" for w in widths.split()))
    return copied


def transform_page_objects(element: ET.Element, scale_x: float, dx: float,
                           scale_y: float, dy: float,
                           skip_first_layer: bool = True) -> int:
    """Move everything on a page that is **not** a stroke: text, images, layers.

    The merge's model is strokes -- the brush, the comparison and the export all
    work on them -- and the first layer's strokes are placed by the fit. The
    rest of the page was carried through as it stood, which is fine while the
    fit is the identity and visibly wrong when it is not: a caption and a
    picture would stay where the paper used to be while the writing under them
    moved.

    So they are moved by the same numbers: text by its `x`/`y`, images by their
    four corners, and the strokes of any layer above the first like the strokes
    they are. It returns how many objects it moved, which is what the window's
    "not merged" line is allowed to say.

    **Not a transform of their own.** None of this makes text or an image
    comparable or brushable; it only keeps them beside the ink they belong to.
    """
    if abs(scale_x - 1.0) < 1e-9 and abs(scale_y - 1.0) < 1e-9 \
            and abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0
    moved = 0
    layers = element.findall('layer')
    for index, layer in enumerate(layers):
        if not (skip_first_layer and index == 0):
            for stroke in layer.findall('stroke'):
                placed = transform_stroke_element(stroke, scale_x, dx, dy,
                                                  scale_y=scale_y)
                stroke.text = placed.text
                if placed.get('width'):
                    stroke.set('width', placed.get('width'))
                moved += 1
        for text_element in layer.findall('text'):
            try:
                x = float(text_element.get('x', '0'))
                y = float(text_element.get('y', '0'))
            except (TypeError, ValueError):
                continue
            text_element.set('x', '%.4f' % (x * scale_x + dx))
            text_element.set('y', '%.4f' % (y * scale_y + dy))
            # Xournal text has a scalar font size; use the same mean scale as pens.
            if text_element.get('size') is not None:
                text_element.set('size', '%.4f' %
                                 (float(text_element.get('size')) * (scale_x + scale_y) / 2))
            moved += 1
        for image in layer.findall('image'):
            # Xournal++ writes the corners; older files may carry only a size.
            corners = {}
            for name in ('left', 'right', 'top', 'bottom'):
                value = image.get(name)
                if value is not None:
                    try:
                        corners[name] = float(value)
                    except ValueError:
                        corners = {}
                        break
            if {'left', 'top'} <= set(corners):
                for name, axis in (('left', 'x'), ('right', 'x'),
                                   ('top', 'y'), ('bottom', 'y')):
                    if name in corners:
                        value = (corners[name] * scale_x + dx
                                 if axis == 'x' else corners[name] * scale_y + dy)
                        image.set(name, '%.4f' % value)
                moved += 1
                continue
            width = image.get('width')
            height = image.get('height')
            if width is None and height is None:
                continue
            try:
                if width is not None:
                    image.set('width', '%.4f' % (float(width) * scale_x))
                if height is not None:
                    image.set('height', '%.4f' % (float(height) * scale_y))
            except ValueError:
                continue
            moved += 1
    return moved


def slice_stroke_span(elem: ET.Element, first: float, last: float) -> ET.Element:
    """A copy of `elem` holding the span from point `first` to point `last`.

    `first` and `last` are **fractions in point-index space**: `2.5` is half way
    along the segment from point 2 to point 3. A fractional end is a real point
    on the polyline and gets its own coordinate -- that is what lets the brush
    cut a stroke **inside a segment**, which a whole-point slice cannot do: a
    two-point straight line brushed in the middle will otherwise be either
    untouched or erased whole, and both are wrong.

    The arithmetic is linear interpolation between the two neighbouring points,
    which is the polyline the stroke already is, so the cut lands exactly where
    the brush edge did and the rest of the stroke is untouched. The result is
    still an ordinary stroke of points.

    Xournal++ writes pressure as `width="nominal w0 w1 ..."`, one `w` per
    segment, so the span keeps the nominal width and the segment widths that
    belong to the segments it covers.
    """
    copied = copy.deepcopy(elem)
    values = [float(v) for v in (copied.text or "").split()]
    points = len(values) // 2
    if points == 0:
        return copied
    if points == 1:
        copied.text = " ".join(f"{value:.6f}" for value in values)
        return copied

    def at(t: float):
        t = max(0.0, min(t, points - 1))
        index = min(int(t), points - 2)
        fraction = t - index
        x0, y0 = values[index * 2], values[index * 2 + 1]
        x1, y1 = values[index * 2 + 2], values[index * 2 + 3]
        return x0 + (x1 - x0) * fraction, y0 + (y1 - y0) * fraction

    first = max(0.0, min(float(first), points - 1))
    last = max(first, min(float(last), points - 1))
    low = int(first)
    high = int(last)
    if abs(last - round(last)) < 1e-9:
        high = int(round(last))

    stops = [first]
    for index in range(low + 1, high + (0 if abs(last - high) < 1e-9 else 1)):
        if index > first + 1e-9:
            stops.append(float(index))
    stops.append(last)
    # strictly increasing, no duplicate ends
    kept_t = []
    for t in stops:
        if not kept_t or t > kept_t[-1] + 1e-9:
            kept_t.append(t)
    if len(kept_t) == 1:
        kept_t.append(kept_t[0])          # a span of no length is one point

    placed = []
    for t in kept_t:
        x, y = at(t)
        placed.append(f"{x:.6f}")
        placed.append(f"{y:.6f}")
    copied.text = " ".join(placed)

    widths = (copied.get("width") or "").split()
    if len(widths) > 1:
        begin = max(0, int(low))
        end = max(begin, min(int(round(last)) if abs(last - round(last)) < 1e-9
                             else int(last) + 1, points - 1))
        kept = widths[1 + begin:1 + end]
        copied.set("width", " ".join([widths[0]] + kept))
    return copied


def slice_stroke_element(elem: ET.Element, start: int, count: int) -> ET.Element:
    """A copy of `elem` holding only points `start` .. `start + count`.

    The merge needs this because a decision is now about a *piece* of a
    stroke and not the whole of it: brushing a span of a long stroke must
    write that span out and leave the rest of it alone, and the result has
    to stay an editable Xournal++ stroke rather than a picture of one.

    A whole-point range is a span with integral ends, so this is the same code.
    """
    points = len((elem.text or "").split()) // 2
    first = max(0, min(int(start), points - 1)) if points else 0
    last = max(first, min(int(start + count) - 1, points - 1)) if points else 0
    return slice_stroke_span(elem, first, last)


class Page:
    def __init__(self, element: ET.Element):
        self.element = element

    @property
    def width(self) -> float:
        return float(self.element.get("width", DEFAULT_PAGE_WIDTH))

    @property
    def height(self) -> float:
        return float(self.element.get("height", DEFAULT_PAGE_HEIGHT))

    @property
    def layer(self) -> ET.Element:
        layer = self.element.find("layer")
        if layer is None:
            layer = ET.SubElement(self.element, "layer")
        return layer

    def strokes(self) -> List[Stroke]:
        return [_parse_stroke(e) for e in self.layer.findall("stroke")]

    def remove_stroke(self, stroke: Stroke) -> None:
        self.layer.remove(stroke.element)

    def beyond_the_first_layer(self) -> Dict[str, int]:
        """What this page holds that the merge does **not** handle.

        Reports content outside the first-layer stroke comparison. The merger
        carries left-side objects through and transforms them with the page fit;
        this inventory does not itself decide what can be merged.
        """
        counts = {'other_layers': 0, 'text': 0, 'image': 0, 'other_strokes': 0}
        layers = self.layers()
        counts['other_layers'] = max(0, len(layers) - 1)
        for index, layer in enumerate(layers):
            counts['text'] += len(layer.findall('text'))
            counts['image'] += len(layer.findall('image'))
            if index > 0:
                counts['other_strokes'] += len(layer.findall('stroke'))
        return counts

    def layers(self) -> List[ET.Element]:
        """Every layer of this page, bottom first, which is paint order."""
        return list(self.element.findall("layer"))

    def add_layer(self, name: str = None) -> ET.Element:
        """A new layer on top of the ones already there.

        Xournal++ paints layers in document order, so appending puts this one
        above everything -- which is what a merge layer has to be, or the
        thing being merged in goes underneath the page it is being added to.

        The name is kept as an attribute even though Xournal++ 1.x does not
        read one back: it is how a merge says which layer it owns, so it can
        be removed again without guessing.
        """
        layer = ET.SubElement(self.element, "layer")
        if name:
            layer.set("name", name)
        return layer

    def add_image(self, png: bytes, box, layer: ET.Element = None) -> ET.Element:
        """A PNG placed at `box` = (left, top, right, bottom) in page units.

        Xournal++'s image object is the raw PNG in base64 with its corners as
        attributes, and it is what a visual merge puts down: the target's own
        pixels over the region that changed, rather than a re-tracing of
        them. Nothing else in this file writes one -- images round-tripped
        untouched before this, because nothing here ever made one.
        """
        import base64

        left, top, right, bottom = (float(v) for v in box)
        element = ET.SubElement(layer if layer is not None else self.layer,
                                "image",
                                {"left": f"{left:.4f}", "top": f"{top:.4f}",
                                 "right": f"{right:.4f}", "bottom": f"{bottom:.4f}"})
        element.text = base64.b64encode(png).decode("ascii")
        return element

    def add_stroke(self, stroke: Stroke) -> ET.Element:
        """Append a copy of `stroke`'s element (safe to call with a stroke
        that belongs to a different document's tree). Returns the copy that
        was actually inserted, so a caller can remove that exact element
        again later (e.g. to undo the add) without needing another lookup."""
        copied = copy.deepcopy(stroke.element)
        self.layer.append(copied)
        return copied


def set_pdf_background(page: ET.Element, filename: str, pageno: int = 1,
                       domain: str = 'absolute') -> ET.Element:
    """Point this page's background at a PDF page.

    Measured from real notebooks (2026-10-28), not guessed:

        <background type="pdf" domain="absolute" filename="..../Qbase_66x138.pdf" pageno="1"/>

    with the absolute path of the template.

    **`absolute` is the only usable domain**, which is why it is the default here. An earlier
    round wrote that `relative` was what was asked for, and it was -- but Xournal++ 1.2.5 refuses
    it: *"XML parser error: unknown domain type: relative"*, and the program's own strings list
    `absolute` and `attach` and nothing else. The comment survived the discovery and said the
    opposite of the code for a day; it is corrected rather than left to mislead the next reader.

    Since the path has to be absolute, the only thing left to get right is *which* absolute path,
    and `place_the_background` answers that: the folder the user keeps the templates in.

    Xournal++ does **not** bind a PDF merely because the names are similar -- the `<background>`
    element is the binding, which is why this has to be written.
    """
    for old in page.findall('background'):
        page.remove(old)
    element = ET.Element('background', {'type': 'pdf', 'filename': filename,
                                        'pageno': str(pageno), 'domain': domain})
    page.insert(0, element)
    return element


def _new_page_element(width: float, height: float) -> ET.Element:
    page = ET.Element("page", {"width": f"{width:.4f}", "height": f"{height:.4f}"})
    ET.SubElement(page, "background", {"type": "solid", "color": "white", "style": "plain"})
    ET.SubElement(page, "layer")
    page.text = "\n"
    return page


class XoppDocument:
    def __init__(self, tree: ET.ElementTree):
        self.tree = tree
        self.root = tree.getroot()

    @property
    def pages(self) -> List[Page]:
        return [Page(p) for p in self.root.findall("page")]

    def ensure_page_count(self, count: int, width: float = DEFAULT_PAGE_WIDTH,
                           height: float = DEFAULT_PAGE_HEIGHT) -> None:
        while len(self.root.findall("page")) < count:
            self.root.append(_new_page_element(width, height))

    @classmethod
    def load(cls, path) -> "XoppDocument":
        """Parse without ever handing expat one enormous string.

        `ET.fromstring(text)` needs the whole document as a single str, and its
        length does not fit in the C int expat takes: a real 472 MB converted
        notebook raised `OverflowError: size does not fit in an int` and could
        not be opened at all. Feeding the same parser in chunks is the same
        parse, without the length ever being represented as one number.
        """
        parser = ET.XMLParser()
        with gzip.open(path, "rt", encoding="utf-8") as f:
            while True:
                chunk = f.read(4 * 1024 * 1024)
                if not chunk:
                    break
                parser.feed(chunk)
        root = parser.close()
        return cls(ET.ElementTree(root))

    @classmethod
    def new_empty(cls, page_count: int = 1, width: float = DEFAULT_PAGE_WIDTH,
                  height: float = DEFAULT_PAGE_HEIGHT) -> "XoppDocument":
        root = ET.Element("xournal", {"creator": "XP", "fileversion": "4"})
        title = ET.SubElement(root, "title")
        title.text = "Xournal++ document - see https://xournalpp.github.io/"
        for _ in range(page_count):
            root.append(_new_page_element(width, height))
        return cls(ET.ElementTree(root))

    def repair(self) -> int:
        """Make the tree safe to write, and say how many strokes it fixed.

        The one rule that costs a whole document: Xournal++ will not open a
        file with a stroke of fewer than two points. Strokes come from the
        tracer, from other documents and from files written by other tools,
        so the check belongs here, at the single point everything is written
        through, rather than in each producer.
        """
        fixed = 0
        for stroke in self.root.iter("stroke"):
            text = stroke.text or ""
            coordinates = text.split()
            if len(coordinates) % 2:
                raise ValueError('stroke has an odd number of coordinates')
            if any(not math.isfinite(float(value)) for value in coordinates):
                raise ValueError('stroke contains non-finite coordinates')
            if len(coordinates) >= 2 * MIN_STROKE_POINTS:
                continue
            if len(coordinates) < 2:
                stroke.text = ""              # nothing to place; dropped below
            else:
                x, y = coordinates[0], coordinates[1]
                stroke.text = " ".join([x, y] * MIN_STROKE_POINTS)
            fixed += 1
        for page in self.root.findall("page"):
            for layer in page.findall("layer"):
                for stroke in [s for s in layer.findall("stroke") if not (s.text or "").split()]:
                    layer.remove(stroke)
        return fixed

    def save(self, path, compresslevel: int = 1) -> None:
        # gzip's own module default (9) was still running after several
        # minutes on a real merged Astromlogy doc (18 pages, ~180k strokes,
        # 335MB of XML) during testing; level 6 took 17s, level 1 takes ~5s
        # for only ~15% larger output. Xournal++ re-saves at its own
        # preferred level the next time you save from inside the app anyway,
        # so there's nothing lost by writing fast here.
        self.repair()
        xml_bytes = ET.tostring(self.root, encoding="utf-8", xml_declaration=True)
        from xp.core.document.atomic import atomic_gzip
        with atomic_gzip(path, compresslevel) as stream:
            stream.write(xml_bytes)
