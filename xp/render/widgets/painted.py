"""The guide, painted **into** the picture instead of beside it.

The user's order: "that fucking dashed line should be drawn on the live view image". Drawing it
as canvas items beside the picture meant two mappings had to agree -- and they did not: the
guide's left edge landed 391 page units outside the page at 1x, 192 at 2x and 98 at 3.8x,
so it moved against the page as the user zoomed.

Painted into the image there is nothing for it to disagree with: it is made out of the same
crop and the same resize as the picture itself, so it cannot drift. The functions here are
pure PIL and take the numbers explicitly, so the property is testable without a window:

    guide_rectangle(box, view, source_size, crop_box, displayed) -> pixel box
    paint_the_guide(image, ...) -> a new image with dashes on it
"""
from __future__ import annotations

from typing import Tuple

GREY = (90, 90, 90)
DASH = 6
GAP = 5
WIDTH = 2


def guide_rectangle(box, view, source_size, crop_box, displayed) -> Tuple[int, int, int, int]:
    """The dashed rectangle's place **in the displayed picture**, in pixels.

    `box` and `view` are in normalised page units (0..1), `source_size` is the size of the
    picture the crop came from, `crop_box` is that crop in its pixels, and `displayed` is
    the size after the resize that fills the pane.
    """
    sw, sh = source_size
    dw, dh = displayed
    cx0, cy0, cx1, cy1 = crop_box
    span_x = max(view[2] - view[0], 1e-9)
    span_y = max(view[3] - view[1], 1e-9)
    scale_x = dw / max(cx1 - cx0, 1)
    scale_y = dh / max(cy1 - cy0, 1)

    def to_pixels(u, v):
        # normalised page -> the picture's own pixels -> the crop's -> the pane's
        return ((u * sw - cx0) * scale_x, (v * sh - cy0) * scale_y)

    x0, y0 = to_pixels(box[0], box[1])
    x1, y1 = to_pixels(box[2], box[3])
    return (int(round(min(x0, x1))), int(round(min(y0, y1))),
            int(round(max(x0, x1))), int(round(max(y0, y1))))


def paint_the_guide(image, box, view, source_size, crop_box, colour=GREY,
                    dash=DASH, gap=GAP, width=WIDTH):
    """Return a copy of `image` with the guide's four dashed sides on it."""
    from PIL import ImageDraw

    x0, y0, x1, y1 = guide_rectangle(box, view, source_size, crop_box, image.size)
    drawing = ImageDraw.Draw(image)
    step = max(2, dash + gap)

    def dashes(ax, ay, bx, by):
        length = max(abs(bx - ax), abs(by - ay))
        if length <= 0:
            return
        ux, uy = (bx - ax) / length, (by - ay) / length
        start = 0.0
        while start < length:
            end = min(start + dash, length)
            drawing.line([(ax + ux * start, ay + uy * start),
                          (ax + ux * end, ay + uy * end)],
                         fill=colour, width=width)
            start += step

    dashes(x0, y0, x1, y0)
    dashes(x1, y0, x1, y1)
    dashes(x1, y1, x0, y1)
    dashes(x0, y1, x0, y0)
    return image
