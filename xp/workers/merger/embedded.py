"""Embedded pictures become strokes, so a merge can compare and keep them.

The user's order (2026-10-26): "The images embedded in the XOPP must be converted to
native strokes in the background, with a Loading/Progress that says which page,
which side, that conversion is running, and that it is done -- on both sides. The
final export is always an editable XOPP, never a PNG patch."

Why it has to happen at all: the merge reads, draws, matches and brushes the
strokes of a page. An `image` element is not a stroke, so a picture on a page was
invisible in the merge and was carried through the export as pixels -- which is
exactly the PNG patch the user does not want.

The conversion is the converter's own: `convert_page_image` on the picture's
pixels, placed at the picture's own corners. The result is cached on disk, keyed
by a digest of the image data and its corners, so a second open is instant and a
cancel costs only what was not yet traced.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

# Two pixels per page unit. The page itself is rendered at one, but a traced
# picture is where the fine strokes of a diagram come from, so it is worth the
# second pixel; `scale` below halves the coordinates back into page units.
PIXELS_PER_UNIT = 2.0


def cache_dir() -> Path:
    """Where traced pictures live: one file per picture, under local app data."""
    root = os.environ.get('LOCALAPPDATA') or str(Path.home() / 'AppData' / 'Local')
    where = Path(root) / 'XP' / 'image-strokes'
    where.mkdir(parents=True, exist_ok=True)
    return where


def images_of(page) -> List[Tuple[object, object, Tuple[float, float, float, float]]]:
    """Every picture on a page, in every layer: `(layer, element, box)`."""
    found = []
    for layer in page.layers():
        for element in layer.findall('image'):
            try:
                box = tuple(float(element.get(name, '0'))
                            for name in ('left', 'top', 'right', 'bottom'))
            except (TypeError, ValueError):
                continue
            right, bottom = box[2], box[3]
            if right <= box[0] or bottom <= box[1]:
                continue
            found.append((layer, element, box))
    return found


def digest_of(element) -> str:
    """A name for this picture: its pixels and where they sit."""
    text = (element.text or '').strip()
    box = '|'.join(element.get(name, '') for name in
                   ('left', 'top', 'right', 'bottom'))
    return hashlib.sha256((box + ':' + text).encode('utf-8')).hexdigest()[:32]


def _stroke_xml(digest: str) -> Optional[List[dict]]:
    path = cache_dir() / (digest + '.json')
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _store(digest: str, strokes: List[dict]) -> None:
    path = cache_dir() / (digest + '.json')
    try:
        writing = path.with_suffix('.partial')
        writing.write_text(json.dumps(strokes), encoding='utf-8')
        os.replace(writing, path)
    except OSError:
        pass                    # a cache that cannot be written is only slower


def _png_to_bgr(png: bytes, size: Tuple[int, int]):
    """The picture as the converter wants it: BGR, at `size` pixels."""
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(png)) as opened:
        picture = opened.convert('RGB').resize(size, Image.LANCZOS)
    rgb = np.asarray(picture, dtype=np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def trace(element, box, profile=None) -> List[dict]:
    """One picture as stroke elements, in page units, cached by content."""
    from xp.workers.converter.convert import convert_page_image
    from xp.workers.converter.profiles import Profile

    digest = digest_of(element)
    held = _stroke_xml(digest)
    if held is not None:
        return held

    png = base64.b64decode((element.text or '').strip())
    width = max(2, int(round((box[2] - box[0]) * PIXELS_PER_UNIT)))
    height = max(2, int(round((box[3] - box[1]) * PIXELS_PER_UNIT)))
    image = _png_to_bgr(png, (width, height))
    if profile is None:
        profile = Profile('hand', scale=PIXELS_PER_UNIT,
                          dx=box[0], dy=box[1])
    # **`scale` is pixels per page unit**, not the other way round: the
    # converter divides the traced pixels by it and then adds the offset
    # (`segments_to_strokes`). Passing 1 / 2 put the ink at twice the box --
    # traced strokes came out at x 342 on a 188-unit page.
    strokes = convert_page_image(image, profile, PIXELS_PER_UNIT,
                                 box[0], box[1])
    written = []
    for stroke in strokes:
        # `convert_page_image` returns <stroke> elements; keep them as the text
        # of their attributes, which is what the cache stores and what an
        # ElementTree rebuilds without carrying the converter's tree around.
        written.append({'attrib': dict(stroke.attrib), 'text': stroke.text or ''})
    _store(digest, written)
    return written


def _element_from(held: dict):
    import xml.etree.ElementTree as ET

    element = ET.Element('stroke', dict(held.get('attrib') or {}))
    element.text = held.get('text') or ''
    return element


def convert_page(page, progress=None) -> int:
    """Turn every picture on this page into strokes, **in place**.

    In place, because everything downstream -- the layers, the brush, the export
    -- reads the page's own elements. Returns how many pictures were converted.
    The strokes go into the layer the picture was on, and the picture is removed
    only once its strokes are there, so a failure leaves the page as it was.
    """
    pictures = images_of(page)
    done = 0
    for index, (layer, element, box) in enumerate(pictures):
        if progress is not None:
            progress(index, len(pictures))
        held = trace(element, box)
        element_from = _element_from
        placed = [element_from(one) for one in held]
        for stroke in placed:
            layer.append(stroke)
        layer.remove(element)
        done += 1
    if progress is not None and pictures:
        progress(len(pictures), len(pictures))
    return done


def convert_document(document, progress=None) -> int:
    """Every picture in every page, which is what opening a merge needs."""
    total = 0
    for index, page in enumerate(document.pages):
        total += convert_page(page, progress=(
            None if progress is None
            else (lambda done, count, page=index: progress(page, done, count))))
    return total


def pictures_left(document) -> int:
    """How many pictures are still pictures -- the thing the status line says."""
    return sum(len(images_of(page)) for page in document.pages)
