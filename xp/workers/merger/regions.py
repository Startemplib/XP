"""Where each part of the page comes from: one state per place, in page units.

The merge has exactly two kinds of data, and this is the second one.

  * **Bitmap material** -- one complete page image per side, rendered at the DPI the user set. Content
    only; it never says which side wins.
  * **Region geometry** (this module) -- the page divided into regions, each carrying one of four
    mutually exclusive states, in page coordinates.

    COMMON = 0   both sides' content here; shared ink is drawn once
    LEFT   = 1   only the left side's content here
    RIGHT  = 2   only the right side's content here
    NEITHER= 3   neither side's content here

**A region is a polygon in page units, and the last one drawn wins.** That single rule gives the
three properties the specification asks for, without an interval tree or a planar subdivision:

  * *covering*: the map starts as COMMON over the whole page, so every point has a state;
  * *non-overlapping*: a point is answered by the last polygon that contains it, which is one state;
  * *the brush overrides*: a new gesture appends its polygon, so it wins over what was there.

Nothing here knows about zoom, DPI or pixels. `labels()` rasterises the map for a view -- that bitmap
is a cache, dropped and rebuilt whenever it is asked for, and never the authority. The export reads
the same polygons, which is the point: the decision exists without an image.

Coordinates are **page units**, the same space `xopp_io` writes and `views` measures in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

COMMON, LEFT, RIGHT, NEITHER = 0, 1, 2, 3
STATES = (COMMON, LEFT, RIGHT, NEITHER)
NAMES = {COMMON: 'common', LEFT: 'left', RIGHT: 'right', NEITHER: 'neither'}

Point = Tuple[float, float]
Polygon = Sequence[Point]


@dataclass
class Region:
    """One polygon and the state it imposes inside it."""

    points: Polygon
    state: int
    #: Which gesture drew it. Undo works in gestures, not in polygons: one drag is one gesture.
    gesture: int = 0

    def __post_init__(self):
        if self.state not in STATES:
            raise ValueError('a region state is one of %r, not %r' % (STATES, self.state))
        if len(self.points) < 3:
            raise ValueError('a region needs at least three points, got %d' % len(self.points))


@dataclass
class RegionMap:
    """The page's four-state region map, in page units.

    The page starts COMMON. Every added region wins over what it covers, and undo removes whole
    gestures, which is how the brush behaves: one drag, one undo step.
    """

    page_width: float
    page_height: float
    regions: List[Region] = field(default_factory=list)
    #: Gestures undone, kept so redo can put them back exactly.
    _undone: List[List[Region]] = field(default_factory=list)
    _next_gesture: int = 1
    stamp: int = 0
    #: **The initial comparison, as a plane.** The first answer to "which side has ink here" is a
    #: measurement of two rasters, and a measurement is a raster: turning it into polygons would be
    #: re-describing it as geometry nobody drew. It is the *base* the polygons sit on -- `state_at`
    #: answers with the last polygon that contains the point, and falls back to this plane when none
    #: does -- and `scale` says what pixel it was measured at, so page units stay page units.
    base: Optional[object] = None
    scale: float = 1.0
    #: **How to ask for the base at another scale.** The classification can be sampled at any zoom
    #: (classification.in_view); resizing the plane it was measured at would turn every zoom into an
    #: upscale of a 4 px/unit picture -- which is what a NEAREST resize of ase is. When this is
    #: given it wins, and ase is only the fallback.
    base_sampler: Optional[object] = None

    # ---- building ------------------------------------------------------

    def _region_covering(self, x: float, y: float) -> Optional[Region]:
        """The last region that contains the point, or None -- the whole stacking rule."""
        found = None
        for region in self.regions:
            if _inside(region.points, x, y):
                found = region
        return found

    def state_at(self, x: float, y: float) -> int:
        """The state one drawn region imposes, or COMMON where none does."""
        region = self._region_covering(x, y)
        return COMMON if region is None else region.state

    def state_at_page(self, x: float, y: float) -> int:
        """The answer for a page point: the last drawn region, else the initial comparison."""
        region = self._region_covering(x, y)
        if region is not None:
            return region.state
        base = self.base
        if base is None:
            return COMMON
        row = int(y * self.scale)
        column = int(x * self.scale)
        if 0 <= row < base.shape[0] and 0 <= column < base.shape[1]:
            return int(base[row, column])
        return COMMON

    def add(self, polygon: Polygon, state: int) -> Region:
        """One gesture's polygon. It wins over everything already there."""
        region = Region(tuple((float(x), float(y)) for x, y in polygon), state,
                        self._next_gesture)
        self._next_gesture += 1
        self.regions.append(region)
        self._undone.clear()          # a new decision is a new branch
        self.stamp += 1
        return region

    def undo(self) -> bool:
        """Remove the **whole last gesture**, so redo can restore it as one step."""
        if not self.regions:
            return False
        last = self.regions[-1].gesture
        taken = []
        while self.regions and self.regions[-1].gesture == last:
            taken.append(self.regions.pop())
        self._undone.append(taken)
        self.stamp += 1
        return True

    def redo(self) -> bool:
        if not self._undone:
            return False
        for region in reversed(self._undone.pop()):
            self.regions.append(region)
        self.stamp += 1
        return True

    def clear(self) -> None:
        self.regions.clear()
        self._undone.clear()
        self.stamp += 1

    def __len__(self) -> int:
        return len(self.regions)

    # ---- asking --------------------------------------------------------

    def labels(self, shape: Tuple[int, int], scale: float,
               window: Optional[Tuple[float, float, float, float]] = None):
        """The map as one uint8 plane for a view: `shape` is (height, width), `window` is in the
        pixels of `scale` and defaults to the whole page.

        **A cache, not the truth.** Anything that needs the decision itself asks `state_at_page` or
        walks `regions`; this exists so a view can be drawn cheaply. It starts from the initial
        comparison (`base`) where there is one, then draws the polygons over it in order.
        """
        import numpy as np
        from PIL import Image, ImageDraw

        height, width = int(shape[0]), int(shape[1])
        left, top = (0.0, 0.0)
        if window is not None:
            left, top = float(window[0]), float(window[1])
        if self.base_sampler is not None:
            # **Asked at this view's scale, not resized from the measurement's.** The classification
            # can answer at any zoom; a NEAREST resize of the plane it was measured at turns every
            # zoom into an upscale of a picture taken at one scale, which is exactly the failure the
            # switch kept producing.
            plane = Image.fromarray(np.asarray(self.base_sampler(shape, scale, window), np.uint8))
        elif self.base is None:
            plane = Image.new('L', (max(1, width), max(1, height)), COMMON)
        else:
            # The base was measured at `self.scale`; this view is at `scale`. Nearest, because a
            # state is a label and half a label is not a thing.
            source = Image.fromarray(np.asarray(self.base, np.uint8))
            if source.size != (max(1, width), max(1, height)):
                source = source.resize((max(1, width), max(1, height)), Image.NEAREST)
            plane = source
        draw = ImageDraw.Draw(plane)
        for region in self.regions:
            points = [((x * scale) - left, (y * scale) - top) for x, y in region.points]
            draw.polygon(points, fill=region.state)
        return np.asarray(plane, np.uint8)

    def coverage(self, shape, scale, window=None):
        """How much of the page each state owns, in the pixels of one view."""
        import numpy as np

        plane = self.labels(shape, scale, window)
        counts = np.bincount(plane.ravel(), minlength=4)
        return {NAMES[state]: int(counts[state]) for state in STATES}

    def owner_plane(self, shape, scale, window=None):
        """The map in the vocabulary the stroke cutter already speaks.

        `_owner_runs` walks a stroke and cuts it where the answer changes, and its vocabulary is the
        old one: **0** undecided/base, **1** the left won, **2** the right won, **3** neither. That is
        the same information as the four states, read from the other end -- COMMON is 0 because shared
        ink is written once, from the base side -- so this is a translation, not a second opinion.

        The export reads this, which is what makes the region map the authority for the document and
        not only for the screen.
        """
        import numpy as np

        states = self.labels(shape, scale, window)
        out = np.zeros(states.shape, np.uint8)
        out[states == LEFT] = 1
        out[states == RIGHT] = 2
        out[states == NEITHER] = 3
        return out

    # ---- geometry ------------------------------------------------------

    def polygons_for(self, state: int) -> List[Polygon]:
        """The polygons that currently impose `state`, last-drawn last. For the export."""
        return [region.points for region in self.regions if region.state == state]


def _inside(points: Polygon, x: float, y: float) -> bool:
    """Even-odd point-in-polygon: the same rule `ImageDraw.polygon` fills by."""
    inside = False
    count = len(points)
    for index in range(count):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % count]
        if (y0 > y) != (y1 > y):
            if x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
                inside = not inside
    return inside


def polygon_from_boxes(boxes: Iterable[Tuple[float, float, float, float]]) -> List[Polygon]:
    """One polygon per box. A brush gesture is a swept band; a box is the simplest form of it."""
    return [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)] for x0, y0, x1, y1 in boxes]


def buffer_polyline(points: Iterable[Point], radius: float) -> Polygon:
    """The band a brush of `radius` sweeps along a path, as one polygon.

    A brush gesture is not a rectangle: it is a line with a width, and the thing it decides is the
    band around it. This offsets the path to both sides and closes the ends -- mitre joins are not
    modelled, so a sharp corner is rounded off by the join rather than cut square, which for a
    brush is closer to what the pen did than a mitre would be.

    The path is in **page units** and so is the radius, so the polygon means the same thing at every
    zoom and every DPI. Degenerate input answers an empty polygon, which `Region` refuses: a click
    with no drag decides nothing, and saying so here keeps that rule in one place.
    """
    import math

    path = [(float(x), float(y)) for x, y in points]
    if not path:
        return []
    size = float(radius)
    if len(path) == 1 or size <= 0:
        x, y = path[0]
        return [(x - size, y - size), (x + size, y - size), (x + size, y + size), (x - size, y + size)]

    left, right = [], []
    last = len(path) - 1
    for index, (x, y) in enumerate(path):
        if index == 0:
            dx, dy = path[1][0] - x, path[1][1] - y
        elif index == last:
            dx, dy = x - path[last - 1][0], y - path[last - 1][1]
        else:
            dx, dy = path[index + 1][0] - path[index - 1][0], path[index + 1][1] - path[index - 1][1]
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length * size, dx / length * size
        left.append((x + nx, y + ny))
        right.append((x - nx, y - ny))
    return left + list(reversed(right))

