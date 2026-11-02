"""The Q base reference, drawn once and used by every XP window.

The user wants the same thing in the converter and the merger: **one silver-grey
dashed line where the template's writing area is**, never the printed blue
border itself, and never so long or so solid that it can be mistaken for the
paper's own edge (that one is a separate, solid, one-pixel silver line).

The three callers differ only in *where* the rectangle is in their own pixels --
the merger maps it through the page's calibration and the zoom, the converter
maps it through the visible page rectangle and the pane it is shown in -- so the
geometry lives here and the mapping stays with whoever knows it.
"""
from __future__ import annotations

from typing import Iterable, Tuple

GUIDE = (150, 150, 150)
GUIDE_DASH = (6, 4)


def guide_segments(box: Tuple[float, float, float, float],
                   dash: Tuple[int, int] = GUIDE_DASH) -> list:
    """The dashes of a rectangle, as line segments, in the box's own units."""
    x0, y0, x1, y1 = box
    on, off = dash
    segments = []
    for horizontal in (True, False):
        if horizontal:
            start, end, near, far = x0, x1, y0, y1
        else:
            start, end, near, far = y0, y1, x0, x1
        at = start
        while at < end:
            stop = min(end, at + on)
            if horizontal:
                segments.append((at, near, stop, near))
                segments.append((at, far, stop, far))
            else:
                segments.append((near, at, near, stop))
                segments.append((far, at, far, stop))
            at = stop + off
    return segments


def draw_guide(canvas, segments: Iterable[Tuple[float, float, float, float]],
               tag: str = 'qbase-guide', colour: str = '#5a5a5a') -> list:
    """Put the dashes on a Tk canvas; returns the items it made.

    **Dark grey, two pixels.** `#969696` at one pixel is invisible on the white
    paper the converter's panes are -- the user asked for the printed border to be
    *replaced* by this line, and with the border gone and the line unreadable the
    page looked blank. The user's word for it is "dark grey dashed", so this is the dark
    grey part, and the width is what makes it survive a downscaled screenshot.
    """
    items = []
    for x0, y0, x1, y1 in segments:
        items.append(canvas.create_line(x0, y0, x1, y1, fill=colour, width=2,
                                        dash=GUIDE_DASH, tags=(tag, 'gui-aid')))
    return items
