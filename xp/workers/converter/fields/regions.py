"""Phase 2 as arrays: which colour each ink pixel is, and the user's edits to that.

The old order drew strokes first and then asked each one what colour it ought
to be, which meant cutting a finished stroke wherever the colour changed
along it. Every artefact of that -- the minimum run length, the hysteresis,
the cluster tiling printing itself onto the writing, the black stroke in the
middle of blue handwriting -- came from asking the question in that order.

So the ink is cut by colour while it is still an area, and strokes are made
last. The map

    M = { (R_i, c_i) }

is a **label image and a colour table**, not a collection of objects. That is
the user's design read literally: the outlines off the perfect image are the
geometry pattern, the clustering gives the colour, and "split the area
according to the colour" is what a label image is. It is total and
single-valued in the strongest sense available -- an array has exactly one
value per pixel -- and its cost does not grow with the number of pieces.

<~ This module used to hold `Region` (a crop and a boolean mask per piece of
   ink), `Geometry` (a component inventory with an area and a hole count per
   piece), `geometry_of`, `_carry_holes`, `colour_map` and `ink_of`. All of it
   was mine rather than the user's, and two things settled it: obeying the
   clustering's own colours makes 121,064 pieces on the user's page, and building
   the objects for them cost 27 seconds before a stroke was drawn; and the user
   pointed out that an inner contour already *is* a hole, so counting holes
   per piece was answering a question nobody had asked. What is left is the
   two things phase 3 actually reads. ~>
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, Tuple

import cv2
import numpy as np

CONNECTIVITY = 8           # ink is 8-connected, so a diagonal pen stroke is
                           # one piece and not a dotted line of them. The dual
                           # choice (4 for the paper) is what
                           # `convert.enclosed_regions` relies on.


def colour_index(field, mask=None) -> np.ndarray:
    """Which colour each ink pixel is, as an index into the field's palette.

    With nothing gathering the clusters into pens, a cluster *is* a colour --
    so this is `field.labels`, which the clustering already produced, and the
    only work left is to say that nothing off the ink has a colour.

    <~ it used to recover the index from the per-pixel *colour*: pack the
       pixel's BGR into a 24-bit integer, sort the palette, searchsorted.
       That existed because the merge threw the cluster-to-pen mapping away
       and the only honest question left was "what colour is this". It cost
       `field.bgr.astype(np.uint32)` -- 173 million pixels by three channels
       upcast, **2.07 GB** -- and eleven seconds of the user's page, to recompute
       something the clustering had already written down. The merge is gone;
       so is the need. ~>
    """
    labels = np.array(field.labels, np.int32, copy=True)
    if mask is not None:
        labels[~mask] = -1
    else:
        labels[labels >= len(field.colours_bgr)] = -1
    return labels


def paint(labels, colours_bgr, paper=255) -> np.ndarray:
    """The map as a picture: every pixel in the colour the map gives it.

    One array operation. It used to be a loop over the pieces, which is a
    loop that gets longer the more faithful the colour is.

    The page's own colours and not random ones: a random palette shows where
    the boundaries are and hides whether they are in the right place, and the
    second is what stage 13 is for.
    """
    view = np.full(labels.shape + (3,), paper, np.uint8)
    inside = labels >= 0
    palette = np.asarray(colours_bgr, np.uint8).reshape(-1, 3)
    if inside.any() and len(palette):
        view[inside] = palette[np.clip(labels[inside], 0, len(palette) - 1)]
    return view


def pieces_at(labels, index) -> Tuple[int, np.ndarray]:
    """The connected pieces of one colour, for a caller that needs them.

    Nothing in the conversion does: phase 3 obeys the colour point by point
    while it grows. This is here for stage 13, which has to turn a click into
    a piece, and it is per colour on purpose -- labelling one colour's pixels
    is cheap and labelling all 6,534 colours' is not.
    """
    same = (labels == index).view(np.uint8)
    count, pieces = cv2.connectedComponents(same, connectivity=CONNECTIVITY)
    return count, pieces


@dataclass
class Correction:
    """Stage 13, as data: what the user changed about the map.

    Held apart from the map so a re-run of phase 2 does not throw the edits
    away -- which is the only reason the correction is a stage of its own
    rather than a button that redraws.

    Keyed by *where* a mark is and how big its box is, never by an index or an
    object: the index depends on how many colours the page turned out to have
    and on the order the pieces came back in, and both move when an earlier
    setting changes. A decision that forgets itself when a slider moves is not
    a decision.
    """
    dropped: set = _field(default_factory=set)                   # keys to leave out
    recoloured: Dict[tuple, int] = _field(default_factory=dict)  # key -> colour

    @staticmethod
    def key(top, left, height, width) -> tuple:
        return (int(top), int(left), int(height), int(width))

    def apply(self, labels):
        """The user's edits, as array writes. Returns `(labels, dropped, recoloured)`.

        Dropping clears those pixels from the ink -- the one place outside
        phase 1 where ink leaves, and it leaves because the user said so.
        Recolouring writes a different label there.
        """
        if self.empty:
            return labels, 0, 0
        labels = labels.copy()
        for top, left, height, width in self.dropped:
            labels[top:top + height, left:left + width] = -1
        for (top, left, height, width), index in self.recoloured.items():
            window = labels[top:top + height, left:left + width]
            window[window >= 0] = index
        return labels, len(self.dropped), len(self.recoloured)

    @property
    def empty(self) -> bool:
        return not self.dropped and not self.recoloured
