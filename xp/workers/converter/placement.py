"""Where one page's pixels land on another's, measured rather than assumed.

Two documents of the same writing do not share a page box. The user's own TEST pair
is 187.82 x 391.37 page units on one side and 1024 x 1366 on the other --
aspect 0.4799 against 0.7496 -- so resizing one onto the other stretches the
writing and every page comes back changed.

NEITHER SIDE IS TOLD WHERE IT OUGHT TO BE. Both are asked where they are.

    the frame    when both sides carry the template's printed frame, it
                 decides: a printed frame is a constant and does not move
                 when the user adds three lines of writing
    the ink      otherwise, the ink's own box, trimmed as `inkfit` trims it
                 so one far speck cannot set the scale

WHY NOT `convert.page_geometry`, which is the converter's own answer to
"where does this page's ink go": because it is the right question for making
a *new* document and the wrong one here. A base `.xopp` already has whatever
geometry it was made with and nothing in the file says which. Measured,
`page_geometry` puts the user's PDF's frame on `REFERENCE_FRAME` at 173.82 page
units wide, and the base's own frame is 112.3 -- that base was fitted by
page size, not by the frame. Two different rules, and every page differs.

A FRAME THAT TOUCHES AN EDGE IS A CUT ONE and is not trusted. Half a frame
fitted onto a whole one scales everything inside it; a fixture whose page ran
off the bottom gave exactly that, and a 26% error.
"""
from typing import Optional, Tuple

import numpy as np

from xp.core.imaging import imagediff
# Share of the ink ignored at each end when boxing it. `inkfit.TRIM` is the
# same number for the same reason.
TRIM = 0.02


def whole_frame(image):
    """The template's frame, but only if all four sides of it are there."""
    from xp.core.frame import frame_in_image

    found = frame_in_image(image)
    if found is None:
        return None
    height, width = image.shape[:2]
    if (found.x0 <= 0 or found.y0 <= 0
            or found.x1 >= width - 1 or found.y1 >= height - 1):
        return None
    return found


def _ink_box(image) -> Optional[Tuple[float, float, float, float]]:
    mask = imagediff.ink_of(image)
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return None
    low, high = TRIM, 1.0 - TRIM
    x0, x1 = np.quantile(xs, [low, high])
    y0, y1 = np.quantile(ys, [low, high])
    return float(x0), float(y0), float(x1), float(y1)


def _from_boxes(here, there):
    """Scale and offset putting the box `there` onto the box `here`."""
    wide = there[2] - there[0], here[2] - here[0]
    tall = there[3] - there[1], here[3] - here[1]
    if wide[0] <= 1 or tall[0] <= 1:
        return None
    # Both sides, averaged: a frame that comes back a little out of square
    # must not be allowed to stretch the writing inside it.
    factor = (wide[1] / wide[0] + tall[1] / tall[0]) / 2.0
    return factor, here[0] - there[0] * factor, here[1] - there[1] * factor


def place(onto, image, fallback=None):
    """How `image` lands on `onto`: (factor, at_x, at_y), in `onto`'s pixels.

    Scale `image` by `factor` and put its top-left at `(at_x, at_y)` and the
    two are on each other. `fallback` is used when neither the frames nor
    the ink give an answer; without one the page boxes are used, which is at
    least not a guess dressed up as one.
    """
    here, there = whole_frame(onto), whole_frame(image)
    if here is not None and there is not None:
        answer = _from_boxes((here.x0, here.y0, here.x1, here.y1),
                             (there.x0, there.y0, there.x1, there.y1))
        if answer is not None:
            return answer
    here, there = _ink_box(onto), _ink_box(image)
    if here is not None and there is not None:
        answer = _from_boxes(here, there)
        if answer is not None:
            return answer
    if fallback is not None:
        return fallback
    tall, wide = onto.shape[:2]
    return ((wide / image.shape[1] + tall / image.shape[0]) / 2.0, 0.0, 0.0)


def onto(image, size, factor, at_x, at_y, paper=None) -> np.ndarray:
    """`image` scaled and placed on a page of `size` = (width, height).

    Ink that runs off the page keeps what fits: the page being landed on is
    the one that has to stay true.
    """
    import cv2
    from PIL import ImageColor

    from xp.render.drawing import BACKGROUND_COLOR

    want_w, want_h = int(size[0]), int(size[1])
    new_w = max(1, int(round(image.shape[1] * factor)))
    new_h = max(1, int(round(image.shape[0] * factor)))
    resized = cv2.resize(
        image, (new_w, new_h),
        interpolation=cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR)
    page = np.zeros((want_h, want_w, 3), np.uint8)
    page[:, :] = (ImageColor.getrgb(BACKGROUND_COLOR)[::-1] if paper is None
                  else tuple(paper))
    at_x, at_y = int(round(at_x)), int(round(at_y))
    src_x, src_y = max(0, -at_x), max(0, -at_y)
    put_x, put_y = max(0, at_x), max(0, at_y)
    wide = min(new_w - src_x, want_w - put_x)
    tall = min(new_h - src_y, want_h - put_y)
    if wide > 0 and tall > 0:
        page[put_y:put_y + tall, put_x:put_x + wide] = \
            resized[src_y:src_y + tall, src_x:src_x + wide]
    return page
