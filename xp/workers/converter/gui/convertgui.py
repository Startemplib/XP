"""Editable automatic settings and stage previews, without a terminal or arpy."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import queue
import shutil
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageTk

from xp.workers.converter.fields import orthofill
from xp.workers.converter.fields import regions
from xp.render import drawing as render
from xp.workers.converter.gui import timeline
from xp.render.widgets import timingbar
from xp.render import appearance
from xp.render.appearance import decorate
from xp.workers.converter.optimization.autotune import TuneCancelled, tune_page
from xp.workers.converter.convert import (apply_morph, convert_page_image, denoise_and_sharpen,
                      the_page_in_qbase,
                      page_ink,
                      encode_regions, region_map, validate_output,
                      page_geometry, perfect_image, render_pdf_page, segments_to_strokes,
                      shape_segments, traced_segments, max_dpi_for, usable_dpi)
from xp.core.imaging.inkdiff import (GRID_MAX_DIM, Reference, class_reach, color_error, contrast, coverage,
                      grid_size, ink_classes, mark_area, segment_classes, source_ink, traced_ink)
from xp.workers.converter.probe import probe_page
from xp.workers.converter.profiles import Profile, applies as profiles_applies
from xp.render.drawing import LEFT_ONLY_COLOR, RIGHT_ONLY_COLOR, render_page
from xp.render.widgets.hotkeys import HotkeyButton
from xp.render.widgets.progress import Progress
from xp.core.document.xopp_io import XoppDocument

# Every step, in the order the page goes through them. Two stages, and the
# second in two halves: stage one decides what is ink, stage two A turns ink
# into shape, stage two B gives the shape its colour. A stage a setting can
# change has to be a stage you can look at, or the setting is being tuned
# blind.
# The user's seventeen, in three phases. The numbering is the user's and the order is the
# architecture: everything that decides anything happens while the ink is
# still regions, and strokes are made last. A stroke is the one
# representation a person cannot correct and an algorithm cannot reconsider
# -- changing its colour means cutting it -- so nothing is turned into one
# until every decision has been taken and shown.
STAGES = (
    # PHASE 1 -- image purification: which pixels are ink. The only phase
    # allowed to throw ink away, and every discard has a picture.
    'Original',             # 1  render
    'Probe',                # 2  what the page says about itself
    'Denoise',              # 3
    'Sharpen',              # 4
    'Threshold',            # 5
    'Morphology',           # 6
    'Holes to ink',         # 7  interior recovery
    'Specks',               # 8  speck removal, with every candidate's area
    'Perfect image',        # 9
    # PHASE 2 -- region geometry and colour: which region is which colour.
    'Geometry',             # 10 components and their holes
    'Colour cloud',         # 11 the 5-D clustering, in three pictures:
    'Colour clusters',      # 11 what it was handed, how it grouped it,
    'Colours',              # 11 and what colour each one is
    'Region colour map',    # 12 M = {(R_i, c_i)}
    'Region correction',    # 13 the user's edits to M
    # PHASE 3 -- stroke generation: what strokes draw those regions.
    'Medial axis',          # 14
    'Orthogonal filling',   # 15
    'Output validation',    # 16 the one the user would not let go of
    'Traced result',        # 17 Xournal++ export
)

# Marks drawn over a stage so a decision is visible rather than implied.
ADDED_COLOUR = (40, 190, 90)       # ink given back to the mask
DROPPED_COLOUR = (230, 60, 60)     # specks about to be thrown away
LOST_COLOUR = (230, 60, 60)
INVENTED_COLOUR = (70, 140, 230)
# Stage images are captured at this resolution and cropped/rescaled per view
# at draw time, so zooming in reveals real pixel detail instead of just
# magnifying an already-shrunk thumbnail.
PREVIEW_MAX_DIM = GRID_MAX_DIM  # the diff grid, so masks line up with every stage image
# One grey for both lines of the sheet: solid for the writing area, dashed
# where the printed border was found. The user's word for it: "grey".
GUIDE_LINE = ('#5a5a5a', '#5a5a5a')
# How many dashes one pane's guide may be cut into. A guard against the
# 190,416-item paint a wrong rectangle once produced, not a look: at a 20 px
# pitch a full-height pane needs about 145, so 400 leaves room without
# letting a mistake reach the canvas. The canvas holds this many per pane.
GUIDE_DASH_CAP = 400
# **The key table the Hotkeys button shows.** The user's design, 2026-10-28: one
# row per action, the key on the right, grouped -- "a compact reference
# table, not a long document". Each row carries the Tk sequence it really
# is, and `test_the_hotkey_button_tells_the_truth` asks Tk whether that
# sequence is bound, so a row nobody binds fails rather than misleads.
#
# **One way to do each thing.** The user struck out Ctrl+Up/Down, Ctrl+Left/Right,
# Ctrl+Alt+WASD and Ctrl+Enter: "do not offer several hotkeys for the same
# function". They are gone from the bindings as well as from this table --
# hiding them in the menu while leaving them bound is the thing the user said not
# to do.
HOTKEYS = (
    ('Navigation', (
        ('Previous stage', 'Alt+W', '<Alt-w>'),
        ('Next stage', 'Alt+S', '<Alt-s>'),
        ('Previous parameter', 'Alt+A', '<Alt-a>'),
        ('Next parameter', 'Alt+D', '<Alt-d>'),
    )),
    ('Pages', (
        ('Previous page', 'PageUp', '<Prior>'),
        ('Next page', 'PageDown', '<Next>'),
    )),
    ('Saving', (
        ('Save', 'Ctrl+S', '<Control-s>'),
        ('Save as...', 'Ctrl+Shift+S', '<Control-S>'),
    )),
    ('In a parameter field', (
        ('Select all', 'Ctrl+A', '<Control-a>'),
        ('Undo', 'Ctrl+Z', '<Control-z>'),
        ('Redo', 'Ctrl+Y', '<Control-y>'),
    )),
    ('The window', (
        ('Quit XP', 'Alt+Q', '<Alt-q>'),
    )),
)
# Between the settings column and its scrollbar. The canvas is sized to the
# width the settings ask for, so without this the widest entry ends on the
# same pixel the bar begins (measured: both at 297) and the two read as one
# control overlapping another.
SETTINGS_BAR_GAP = 8
TRACED_PREVIEW_MAX_SCALE = 24   # px per page unit: past this the render costs more than it shows
TRACED_PREVIEW_SUPERSAMPLE = 6  # the stage pictures are drawn at the page's own
                                # resolution and area-averaged down -- about
                                # seven samples a pixel on an 800-DPI page --
                                # so a traced pane drawn at two read as coarser
                                # ink beside them when the ink is the same
MIN_ZOOM = 1.0
# **Deeper than the old 16, on the user's order (2026-11-02).** At 6,000 dpi a
# 64x window is about 245 x 510 source pixels -- still native detail, not a
# stretched thumbnail. At 800 dpi the same 64x is 12.5 dpi of source; the
# number is a view limit, not a promise that the source can fill it.
MAX_ZOOM = 64.0
# `PagePreview`'s frame rate, and the same three-piece structure: the event
# only records the pointer, this loop reads it, and the adjust moves the view.
FOLLOW_MS = 30
CLICK_SLOP = 3   # a press/release that moves less than this is a click, not a drag
# Same colors and meaning as the merge view: red = only in the source (ink
# the tracer lost), blue = only in the output (ink it invented).
MISSING_COLOR = LEFT_ONLY_COLOR
EXTRA_COLOR = RIGHT_ONLY_COLOR
HALO_ALPHA = 0.28   # soft ring around each difference so a lone lost dot shows at Fit
HALO_PX = 4
HALO_MAX_SPOTS = 25  # past this many, rings merge into a carpet that hides the page
AUTO = 'Auto-tune'
LIVE = 'Live stages'
LOST_LABEL = 'PDF ink lost'
EXTRA_LABEL = 'invented'
# Every setting that can change what comes out, with the stage it belongs to.
# Nothing that moves the result may be invisible: a parameter you cannot see
# is one that is tuned blind, and a parameter on the panel that does nothing
# is worse -- `color_break` sat here doing nothing for a day.
#
# (name, label, kind, low, high)
SECTIONS = (
    ('PHASE 1 -- what is ink', (
        # **The upper bound is what a small page can hold, not what the user's page can.** It used to be
    # 1200, so a page whose ceiling was 997 answered "use a value from 72 to 1200" -- a range the
    # page could not do. The per-page ceiling is refused separately, with its own number; this is
    # only the field's sanity bound (a Q base page can reach 6153 dpi under the pixel budget).
    ('dpi', 'Resolution (DPI)', int, 72, 6153),
        ('adaptive_block_size', 'Threshold window (odd)', int, 3, 999),
        ('adaptive_c', 'Threshold offset', float, -100, 100),
        ('denoise', 'Denoise', float, 0, 10),
        ('sharpen', 'Sharpen', float, 0, 10),
        ('morph_open', 'Morphology open', float, 0, 10),
        ('morph_close', 'Morphology close', float, 0, 10),
        ('recover_interiors', 'Holes to ink', bool, 0, 1),
        ('min_area', 'Remove specks below', float, 0, 100000),
    )),
    ('PHASE 2 -- region and colour', (
        ('colour_5d', 'Cluster the colours', bool, 0, 1),
        ('colour_bandwidth', 'Two pens apart by (0 = auto)', float, 0, 200),
        ('space_bandwidth', 'Look this far (pens)', float, 0.2, 40),
        ('color_break', 'Break on color change', float, 0, 10),
    )),
    ('PHASE 3 -- strokes', (
        ('rib_spacing', 'Ribs every (rib widths)', float, 0.1, 4),
        ('merge_gap', 'Merge ends within (pens)', float, 0, 8),
        ('merge_colour', 'Merge colours within (Delta-E)', float, 0, 40),
        ('boundary_width', 'Boundary width (pens)', float, 0.02, 4),
        ('fill_spacing', 'Ring spacing (pens)', float, 0.1, 4),
        ('fill_overlap', 'Ring overlap', float, 1, 3),
        ('boundary_smooth', 'Field smoothing (px)', float, 0, 5),
        ('boundary_shift', 'Smoothing bound (px)', float, 0, 3),
        ('simplify_epsilon', 'Simplify (px)', float, 0, 3),
        ('fill_target', 'Fill until missed <', float, 0, 1),
        ('extra_target', 'Allow drawn outside <', float, 0, 1),
        ('invented_point_share', 'Keep stroke if on ink >', float, 0, 1),
    )),
    ('OUTPUT', (
        ('line_width', 'Output line width', float, .001, 10),
    )),
)
FIELDS = tuple(field for _title, group in SECTIONS for field in group)

# Settings the 5-D colour pass replaces. Shown, because the switch above can
# turn it back on, and greyed out while it is off.
LEGACY_COLOUR = ('color_break',)


# The earliest stage that reads each setting, as an index into STAGES. A
# setting whose stage comes after the one the user is aiming at cannot change what
# the user is looking at, so it is N/A exactly as a setting from another tracing
# mode is. Anything not named here is read by the whole pipeline.
READ_AT = {
    'dpi': 0,                     # Original
    'denoise': 2,                 # Denoise
    'sharpen': 3,                 # Sharpen
    'adaptive_block_size': 4,     # Threshold
    'adaptive_c': 4,
    'morph_open': 5,              # Morphology
    'morph_close': 5,
    'recover_interiors': 6,       # Holes to ink
    'min_area': 7,                # Specks
    'colour_5d': 10,              # Colour cloud
    'colour_bandwidth': 10,
    'space_bandwidth': 10,
    'rib_spacing': 16,            # Orthogonal filling
    'merge_gap': 16,              # the last step, inside stage 15
    'merge_colour': 16,
    'color_break': 16,
    'boundary_width': 16,
    'fill_spacing': 16,
    'fill_overlap': 16,
    'boundary_smooth': 16,
    'boundary_shift': 16,
    'simplify_epsilon': 16,
    'fill_target': 16,
    'extra_target': 16,
    'invented_point_share': 16,
    'line_width': 17,             # Output validation
}


def reads_before(name, aim):
    """Can a setting read at `name`'s stage still matter, aiming at `aim`?"""
    if not aim or aim not in STAGES:
        return True
    at = READ_AT.get(name)
    return True if at is None else at <= STAGES.index(aim)


def field_applies(profile, name, aim=None):
    """Can this setting change the result as things are set right now?

    Two reasons it cannot: it belongs to a tracing mode that is not the one
    selected (profiles.MODE_ONLY), or it belongs to the colour path that is
    switched off. Either way the panel greys it and auto-tune does not spend
    an evaluation on it.
    """
    if not profiles_applies(profile.trace, name):
        return False
    if not reads_before(name, aim):
        return False
    if name in LEGACY_COLOUR:
        return not profile.colour_5d
    if name in ('colour_bandwidth', 'space_bandwidth'):
        return bool(profile.colour_5d)
    return True


REFUSED_COLOUR = '#e05252'


class FieldError(ValueError):
    """A value the pipeline will not take, with the field it came from.

    Before this, the refusal was a bare `ValueError` whose text named the label, and the only
    place it went was the status line at the bottom. The user's order: "you should pop out shit when I
    input sth unacceptable for all params". A `FieldError` carries `name`, so the panel can mark
    **that** field and, on Enter, say it out loud.
    """

    def __init__(self, name, label, message):
        super().__init__(message)
        self.name = name
        self.label = label


def as_he_typed_it(value) -> str:
    """A settings box shows `7`, not `7.0`.

    Most of these fields are declared `float`, so a value that came back from
    a JSON file as `7.0` used to be printed with its `.0`. Nobody noticed
    while the shipped profiles happened to store whole numbers as JSON ints;
    saving the user's default writes real floats, and the whole panel would have
    filled up with trailing zeros the first time the user kept one.
    """
    if isinstance(value, bool) or not isinstance(value, float):
        return str(value)
    return str(int(value)) if value.is_integer() else str(value)


def profile_from_fields(base, values):
    profile = copy.deepcopy(base)
    for name, label, kind, low, high in FIELDS:
        raw = values[name]
        if kind is bool:
            setattr(profile, name, str(raw).strip().lower() in ('1', 'true', 'yes', 'on'))
            continue
        try:
            value = kind(raw)
        except (ValueError, TypeError):
            raise FieldError(name, label, f'{label}: enter a valid number.') from None
        if not math.isfinite(value) or not low <= value <= high:
            raise FieldError(name, label,
                             f'{label}: use a value from {low} to {high}.')
        setattr(profile, name, value)
    if profile.adaptive_block_size % 2 == 0:
        raise FieldError('adaptive_block_size', 'Threshold window',
                         'Threshold window must be an odd number (3, 5, 7, ...).')
    # These are the values the user has just reviewed, not hidden per-page overrides.
    profile.auto_threshold = False
    return profile


def suggest_profile(base, probe):
    profile = copy.deepcopy(base)
    profile.adaptive_block_size = probe.suggested_block_size
    profile.adaptive_c = round(probe.suggested_c, 2)
    profile.min_area = round(probe.suggested_min_area, 2)
    profile.auto_threshold = False
    return profile


def a_free_name(path):
    """`name.xopp` if it does not exist, else `name-1.xopp`, `name-2.xopp`, ...

    A conversion is never written over an existing notebook. The user asked for the numbering out loud:
    "it should use hyphen number to avoid a replace but just add up".
    """
    target = Path(path)
    if not target.exists():
        return str(target)
    stem, suffix = target.stem, target.suffix
    number = 1
    while True:
        candidate = target.with_name('%s-%d%s' % (stem, number, suffix))
        if not candidate.exists():
            return str(candidate)
        number += 1


def _thumbnail(array, bgr=False):
    height, width = array.shape[:2]
    ratio = min(1, PREVIEW_MAX_DIM / max(width, height))
    if ratio < 1:
        array = cv2.resize(array, (max(1, round(width * ratio)), max(1, round(height * ratio))), interpolation=cv2.INTER_AREA)
    if bgr:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(array).convert('RGB')


class _CropSegment:
    """A segment with its points already moved into a crop's coordinates."""
    __slots__ = ('points', 'width', 'fill')

    def __init__(self, points, width, fill):
        self.points = points
        self.width = width
        self.fill = fill


class StageDetail:
    """The full-resolution array behind one stage, for the pane that is zoomed.

    `_thumbnail` keeps the live view small enough to hold every stage at once,
    and that is right at Fit. It is not right when the user zooms: a 2,200-pixel
    preview of a 6,000-dpi page is 405 dpi, so at 8x there are only 275 source
    pixels behind a 500-pixel pane and the text is a magnified thumbnail --
    while the raster the pipeline actually used has 15,652 x 32,614 of them.

    This object holds that source array and cuts the requested window out of
    it at native resolution. It never invents pixels: if the crop is smaller
    than the pane it is enlarged with LANCZOS (the same as any other display
    magnification), but a crop that is at least as large as the pane is only
    ever reduced.
    """

    def __init__(self, name, payload, size, profile):
        self.name = name
        self.payload = payload
        self.size = (int(size[0]), int(size[1]))
        self.profile = profile

    @property
    def dpi(self):
        """The raster's own resolution, not the page's arithmetic."""
        return float(self.profile.dpi)

    @property
    def thumbnail_dpi(self):
        """What the live thumbnail is worth, at the same page size."""
        longest = max(1, self.size[0], self.size[1])
        return self.dpi * min(1.0, PREVIEW_MAX_DIM / float(longest))

    def _crop_box(self, view):
        width, height = self.size
        x0 = min(width - 1, max(0, round(view[0] * width)))
        y0 = min(height - 1, max(0, round(view[1] * height)))
        x1 = max(x0 + 1, min(width, round(view[2] * width)))
        y1 = max(y0 + 1, min(height, round(view[3] * height)))
        return x0, y0, x1, y1

    def _segments_crop(self, x0, y0, x1, y1):
        """A vector stage's strokes, shifted into the crop's own pixels.

        The points in the payload are page pixels, so the picture is simply
        drawn at the crop's size with `(x0, y0)` subtracted -- the same
        ruler, the same widths, no resampling. Segments that cannot touch
        the window are left out; a page of text has thousands and shifting
        all of them for a 700-pixel pane is the only cost here.
        """
        width, height = x1 - x0, y1 - y0
        shifted = []
        for segment in self.payload.get('segments') or ():
            points = np.asarray(segment.points, dtype=float)
            if not len(points):
                continue
            if (points[:, 0].max() < x0 or points[:, 0].min() > x1
                    or points[:, 1].max() < y0 or points[:, 1].min() > y1):
                continue
            shifted.append(_CropSegment(points - np.array([x0, y0]),
                                        getattr(segment, 'width', None),
                                        getattr(segment, 'fill', None)))
        if not shifted:
            return Image.new('RGB', (max(1, width), max(1, height)), 'white')
        drawn = render.segment_ink(shifted, (width, height),
                                   self.profile.line_width,
                                   self.profile.max_coord_str_len)
        picture = np.full((height, width, 3), 255, np.uint8)
        picture[drawn] = (30, 30, 30)
        for key, colour in (('residue', LOST_COLOUR), ('lost', LOST_COLOUR),
                            ('invented', INVENTED_COLOUR)):
            marks = self.payload.get(key)
            if marks is not None:
                picture[np.asarray(marks)[y0:y1, x0:x1]] = colour
        wrong = self.payload.get('wrong')
        if wrong:
            bad = []
            for segment in wrong:
                points = np.asarray(segment.points, dtype=float)
                if not len(points):
                    continue
                if (points[:, 0].max() < x0 or points[:, 0].min() > x1
                        or points[:, 1].max() < y0 or points[:, 1].min() > y1):
                    continue
                bad.append(_CropSegment(points - np.array([x0, y0]),
                                        getattr(segment, 'width', None),
                                        getattr(segment, 'fill', None)))
            if bad:
                over = render.segment_ink(bad, (width, height),
                                          self.profile.line_width,
                                          self.profile.max_coord_str_len)
                picture[over] = LOST_COLOUR
        return Image.fromarray(picture)

    def crop(self, view):
        """The requested window as an RGB image, or None when this stage has
        no pixels of its own (the segment stages; their picture is vector)."""
        x0, y0, x1, y1 = self._crop_box(view)
        if 'colour' in self.payload:
            array = np.asarray(self.payload['colour'])
            if array.ndim == 2:
                array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
            piece = array[y0:y1, x0:x1, :3]
            return Image.fromarray(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))
        if 'segments' in self.payload:
            return self._segments_crop(x0, y0, x1, y1)
        mask = self.payload.get('mask')
        if mask is None:
            return None
        piece = mask[y0:y1, x0:x1]
        picture = np.full(piece.shape + (3,), 255, np.uint8)
        picture[piece] = (30, 30, 30)
        for key, colour in (('added', ADDED_COLOUR), ('dropped', DROPPED_COLOUR)):
            marks = self.payload.get(key)
            if marks is not None:
                picture[np.asarray(marks)[y0:y1, x0:x1]] = colour
        return Image.fromarray(picture)


@dataclass
class Preview:
    profile: Profile
    images: dict
    probe: object
    strokes: int
    missing: np.ndarray = None       # source ink the output lacks (source grid)
    extra: np.ndarray = None         # output ink the source lacks (source grid)
    traced_frame: tuple = (0.0, 0.0, 1.0, 1.0)  # source page inside the traced page, normalized
    missing_share: float = 0.0       # of all source ink
    extra_share: float = 0.0         # of all output ink
    marks: int = 0                   # ink blobs at least half a pen dot on the page
    marks_lost: int = 0
    marks_extra: int = 0
    color_wrong: float = 0.0        # share of traced ink in a color that isn't under it
    tune: object = None              # TuneResult when the settings were auto-tuned
    pieces: int = 0                  # regions phase 2 handed over: the pieces of
                                     # ink the drawing was asked to make, each
                                     # one piece of one colour. Under the old
                                     # order this was "geometry pieces before
                                     # colour sliced them"; colour slices
                                     # nothing now, and it is the map that
                                     # says how many pieces there are
    page: object = None              # the finished output page, kept so the
                                     # Traced pane can be drawn from its vectors
                                     # at the zoom the user is looking through instead
                                     # of from a stored 24 px/unit picture


class Reached(Exception):
    """The stage the user is aiming at has just been produced.

    Not an error: it is how the pipeline stops without every stage after it
    having to know there is a target. The pictures are already with the user --
    `on_image` hands each one over as it is made -- so there is nothing to
    carry out of here.
    """


def _announced(name, announce, picture):
    """Hand a picture over the moment it exists, and keep it."""
    announce(name, picture)
    return picture


def _stage_picture(payload, size, profile):
    """One milestone of the pipeline, as something you can look at.

    A stage carries either pixels or geometry. Pixels come back as they are,
    with whatever decision the stage made drawn over them in colour -- the
    ink it gave back, the specks it is about to throw away, what it still
    misses. Geometry is drawn the way the renderer draws it, in the page's
    own pixels, so a stage of the shape can be compared with the stage
    before it rather than with an idea of it.
    """
    width, height = size
    if 'colour' in payload:
        return _thumbnail(payload['colour'], bgr=True)
    if 'segments' in payload:
        drawn = render.segment_ink(payload['segments'], (width, height),
                                   profile.line_width, profile.max_coord_str_len)
        view = np.full((height, width, 3), 255, np.uint8)
        view[drawn] = (30, 30, 30)
        for key, colour in (('residue', LOST_COLOUR), ('lost', LOST_COLOUR),
                            ('invented', INVENTED_COLOUR)):
            marks = payload.get(key)
            if marks is not None and marks.any():
                view[marks] = colour
        # Strokes in the wrong colour are drawn over, so the view says which
        # ones rather than only how many pixels.
        bad = payload.get('wrong')
        if bad:
            over = render.segment_ink(bad, (width, height), profile.line_width,
                                      profile.max_coord_str_len)
            view[over] = LOST_COLOUR
        return _thumbnail(view, bgr=True)
    mask = payload['mask']
    view = np.full(mask.shape + (3,), 255, np.uint8)
    view[mask] = (30, 30, 30)
    for key, colour in (('added', ADDED_COLOUR), ('dropped', DROPPED_COLOUR)):
        marks = payload.get(key)
        if marks is not None and marks.any():
            view[marks] = colour
    return _thumbnail(view, bgr=True)


def traced_preview_scale(profile, raw_size, geometry):
    """Render scale for the 'Traced result' image, in pixels per page unit.

    At least as fine as the pane beside it. The source pane is an 800-DPI
    photograph scaled down, so every edge on it is soft and every detail is
    there; the traced pane is drawn from the vectors, and drawn too coarsely
    it looks like the tracing lost detail that is in fact still in it. It was
    at 64% of the source pane's resolution on a real page, which at 16x is
    exactly what a lost detail looks like.
    """
    raw_w, raw_h = raw_size
    # What one page unit is worth on the source pane, after its own fit.
    source_ratio = min(1.0, PREVIEW_MAX_DIM / max(raw_w, raw_h))
    source = geometry[0] * source_ratio
    fit = min(6, PREVIEW_MAX_DIM / max(profile.page_width, profile.page_height))
    content = grid_size(raw_w, raw_h)[0] * geometry[0] / raw_w
    return min(TRACED_PREVIEW_MAX_SCALE, max(fit, content, source))


def render_traced(page, scale):
    """The traced page, twice over: as measured, and as shown.

    PIL has no antialiasing, so a hairline drawn straight into the pane comes
    out as a staircase beside a source pane whose every edge is soft -- which
    reads as detail the tracing lost, when the detail is there. So it is drawn
    large and scaled down for the eye, while the measurement reads the
    full-size image, where a thin line is still solid ink rather than a grey
    suggestion of one. Returns (measured, shown).
    """
    drawn = render_page(page, scale=scale * TRACED_PREVIEW_SUPERSAMPLE)
    shown = drawn.resize((max(1, drawn.width // TRACED_PREVIEW_SUPERSAMPLE),
                          max(1, drawn.height // TRACED_PREVIEW_SUPERSAMPLE)),
                         Image.Resampling.LANCZOS)
    return drawn, shown


# What each stage's answer depends on. Change a setting in one group and only
# that stage and the ones after it are redone -- which is the difference
# between "I moved the speck size" costing a moment and costing the whole
# tracing. The lists are the honest dependency: a setting named in one of
# them is read by that stage, and a setting named in none of them cannot
# change the page.
ONE_KEYS = ('dpi', 'denoise', 'sharpen', 'adaptive_block_size', 'adaptive_c',
            'morph_open', 'morph_close', 'correct_against_page',
            'recover_interiors', 'min_area')
# Phase 2's own settings. Region-First put the expensive stage -- the 5-D
# clustering, eleven seconds of the page -- *before* the drawing instead of
# after it, so moving a rib spacing no longer invalidates the colour. Under
# the old order the colour pass was downstream of the shape and every drawing
# setting re-ran it.
MAP_KEYS = ('colour_5d', 'colour_bandwidth', 'space_bandwidth')
STROKE_KEYS = ('trace', 'rib_spacing', 'merge_gap', 'merge_colour',
               'boundary_width', 'fill_spacing', 'fill_overlap',
               'boundary_smooth', 'boundary_shift', 'simplify_epsilon', 'fill_target',
               'remove_invented', 'extra_target', 'invented_point_share',
               'color_break', 'mode', 'line_width', 'max_coord_str_len')

_stage_cache = {}
_stage_cache_lock = threading.Lock()


def _settings_key(profile, names):
    return tuple(repr(getattr(profile, name, None)) for name in names)


def _cached(key, build):
    """One stage's answer, computed once per set of settings it reads.

    Deliberately small: the last answer for each of the three keys. A page is
    edited by moving one number back and forth, not by visiting a hundred
    states, and holding a hundred full-page masks would cost more memory than
    the recomputation costs time.
    """
    with _stage_cache_lock:
        hit = _stage_cache.get(key[0])
        if hit is not None and hit[0] == key:
            # A hit computed nothing, so it announced nothing: the caller has
            # to be told, or a run aimed at a stage inside this group would
            # sail past its target and compute the rest of the page. That was
            # the aim being ignored exactly when it was cheapest to obey.
            return hit[1] + (True,)
    value = build()
    with _stage_cache_lock:
        _stage_cache[key[0]] = (key, value)
    return value + (False,)


_page_cache = {}
_page_cache_lock = threading.Lock()


def _render(path, page_index, profile, seen=None):
    """The page render, its probe and its ink (the diff reference) -- none of
    which any threshold setting changes. Kept for the most recent page so an
    edit of the settings doesn't pay to redo them on an 800-DPI page (~2s on
    A4), and so auto-tune can reuse the reference.

    `seen(raw)` is called as soon as the page is rasterized, before it is
    measured. Rendering an 800-DPI page costs a second and a half and
    probing it another four, and there is no reason to look at a blank
    window for the second of those: the picture exists already.
    """
    key = (str(Path(path).resolve()), os.path.getmtime(path), page_index, profile.dpi)
    with _page_cache_lock:
        if key in _page_cache:
            entry = _page_cache[key]
            if seen is not None:
                seen(entry[0])
            return entry
    with pymupdf.open(path) as pdf:
        raw = render_pdf_page(pdf[page_index], profile.dpi)
    # **The warp happens before anything looks at the page**, which is where it has to be.
    # The user's design: after the auto alignment every image *is* the Q base sheet. Warping after
    # `_render` left this page and everything downstream in two different worlds -- `Original`,
    # the probe and the diff reference were the source page (aspect 1.334 for the user's file) while
    # every stage after the threshold was the sheet (2.084), so the two panes drew different
    # paper, the dashed guide was right on one and wrong on the other, and `PDF ink lost` was
    # comparing 1649x2200 against 522x1087 -- a percentage of two different pages.
    raw, geometry = the_page_in_qbase(raw, profile)
    raw.setflags(write=False)
    if seen is not None:
        seen(raw)
    probe = probe_page(raw)
    grid = grid_size(raw.shape[1], raw.shape[0])
    reference = Reference(source_ink(raw, probe.background_bgr, grid),
                          mark_area(max(1.0, probe.stroke_thickness_px), grid[0] / raw.shape[1]),
                          contrast(raw, probe.background_bgr, grid))
    entry = (raw, probe, reference, geometry)
    with _page_cache_lock:
        _page_cache.clear()
        _page_cache[key] = entry
    return entry


def preview_page(path, page_index, profile, suggest=False, tune=False, progress=None,
                 cancel=None, on_image=None, correction=None, upto=None,
                 clock=None, on_detail=None):
    """One page through every stage. `tune` searches this page's settings
    first (autotune.py); `suggest` just takes the probe's formula guess.

    `on_image(name, picture)` is called as each stage finishes, not at the
    end. Denoising a page takes a hundredth of a second and the filling takes
    fifteen, so a viewer that waits for the whole pipeline before it shows
    anything is not a live view of the stages -- it is the finished job,
    replayed. The stage one pictures are up in about a second.

    `upto` is the stage the user is aiming at. The run stops as soon as that
    stage's picture exists, by raising `Reached`, and the caller gets the
    pictures through `on_image` rather than a `Preview`: a partial run has no
    strokes to count and no tracing to compare, so a Preview of one would be
    a record of things that were never measured. Aiming at `None` or at the
    last stage runs the whole pipeline as before.

    `cancel`, when it is set, stops the run the same way -- at the next stage
    boundary, with `TuneCancelled`. That is what makes choosing a stage able
    to *stop* what is running rather than merely out-date it: the generation
    counter already threw the result away, but the thread went on computing
    it to the end.

    `clock` is a `Timeline` the panel is holding and drawing while this runs.
    The marks are taken where the pipeline already calls back -- at a stage's
    payload, which is the one place it is between things -- so nothing here
    is timed by wrapping it.
    """
    if clock is None:
        clock = timeline.Timeline(STAGES)
    # `begin`, not `reset`: what earlier runs measured is what those stages
    # cost on this page, and stepping through the stages runs the pipeline
    # once per stage with everything before it coming out of the cache. The
    # clock restarting here is what keeps the time the user spent looking at the
    # last stage out of the next one.
    clock.begin()
    made = {}
    # Every picture reaches the user once. A step and the group it belongs to can
    # both hold the same picture, and `first_look` sends the page itself
    # before either of them exists.
    sent = set()

    def first_look(raw_page):
        # The page itself, before anything has measured or thresholded it.
        # Kept, so the stage that owns it does not draw it a second time.
        made['Original'] = picture = _thumbnail(raw_page, bgr=True)
        sent.add('Original')
        if on_image is not None:
            on_image('Original', picture)
        if on_detail is not None:
            # The pane's small copy is for Fit; this is the array the pipeline
            # really rendered, so a zoom can cut the window out at its own
            # resolution instead of enlarging the 2,200-pixel thumbnail.
            on_detail('Original', {'colour': raw_page},
                      (raw_page.shape[1], raw_page.shape[0]), profile)

    raw, probe, reference, geometry = _render(path, page_index, profile, first_look)
    # `first_look` cannot raise: it runs inside `_render`, before the probe
    # exists, and the panel reads the probe. So aiming at the render is
    # checked here, one line later, with the probe in hand. `_render` caches
    # per page and dpi, so this costs nothing the second time.
    if cancel is not None and cancel.is_set():
        raise TuneCancelled()
    if upto == 'Original':
        # `first_look` only fires when the page had to be rasterized. On a
        # second visit the render comes from the cache, nobody was told, and
        # aiming at the page itself used to stop with no picture to show --
        # which is what going back to Original looked like once a full run
        # had cleared the live pictures.
        if 'Original' not in made and on_image is not None:
            sent.add('Original')
            on_image('Original', _thumbnail(raw, bgr=True))
        raise Reached()
    tuned = None
    if tune:
        tuned = tune_page(raw, profile, probe, progress, cancel, reference,
                          geometry=geometry)
        profile = tuned.profile
    elif suggest:
        profile = suggest_profile(profile, probe)
    else:
        profile = copy.deepcopy(profile)
    raw_h, raw_w = raw.shape[:2]
    page_key = (str(Path(path).resolve()), os.path.getmtime(path), page_index)

    def detail_colour(name, array):
        """The full array for a stage that draws a grey picture directly."""
        if on_detail is not None and upto == name:
            on_detail(name, {'colour': array}, (raw_w, raw_h), profile)

    def announce(name, picture):
        if on_image is not None:
            on_image(name, picture)
        # The two ways a run ends early, both at a stage boundary, which is
        # the only place the pipeline is between things.
        if cancel is not None and cancel.is_set():
            raise TuneCancelled()
        if upto and name == upto:
            raise Reached()

    def collect(announce_now=True):
        """A stage's pictures, built as it runs.

        `announce_now` is off for the steps of phase one, which announce
        after the step has finished instead -- so that a step that came out
        of the cache and a step that just ran deliver their pictures by the
        same path.
        """
        shots, notes = {}, {}

        def record(name, payload):
            # The mark first: what this stage cost is the work that produced
            # the payload, and the picture below is the panel's cost, not the
            # pipeline's. `hold`/`release` keeps the two apart so the bar
            # shows the conversion and not the watching of it.
            clock.mark(name)
            notes[name] = payload
            drawing = clock.hold()
            shots[name] = picture = _stage_picture(payload, (raw_w, raw_h), profile)
            if on_detail is not None and upto is not None and name == upto:
                # One stage's full arrays, not every stage's: this is the pane
                # the user is looking at, and a 6,000-dpi mask is half a gigabyte.
                on_detail(name, payload, (raw_w, raw_h), profile)
            clock.release(drawing)
            if announce_now:
                announce(name, picture)
        return shots, notes, record

    images, milestones = {}, {}

    def deliver(shots, notes):
        """Hand over a step's pictures, once each, whoever computed them."""
        images.update(shots)
        milestones.update(notes)
        for name, picture in shots.items():
            if name in sent:
                continue
            sent.add(name)
            announce(name, picture)

    read_so_far = []

    def step(tag, reads, make):
        """One step of phase one, inherited when nothing it reads has moved.

        Phase one used to be a single cached group, and a run aimed *inside*
        it never finished the group, so the group was never cached: stepping
        from Probe to Denoise recomputed the probe's ink, and stepping on
        recomputed everything before it again. The user put it plainly -- if the user did
        not change a parameter, moving to the next stage must not do them
        again.

        So every step is cached on its own ancestry: the page, and every
        setting that it or any step before it reads. Moving the threshold
        offset costs the threshold and what follows, and nothing before it;
        moving nothing at all costs nothing at all.
        """
        read_so_far.extend(reads)
        key = ('one:' + tag, page_key, _settings_key(profile, tuple(read_so_far)))
        value, shots, notes, hit = _cached(key, make)
        # A step that came out of the cache computed nothing, and the bar
        # says so rather than showing a sliver: the user asked for inheritance by
        # name, and seeing that it happened is how the user knows it did.
        for name in shots:
            clock.mark(name, inherited=hit)
        deliver(shots, notes)
        return value

    def look():
        shots, notes, _record = collect(False)
        # `first_look` only fires when the page had to be rasterized, so on a
        # second visit this is where the picture comes from.
        shots['Original'] = made.get('Original') or _thumbnail(raw, bgr=True)
        if on_detail is not None:
            on_detail('Original', {'colour': raw},
                      (raw.shape[1], raw.shape[0]), profile)
        return None, shots, notes

    def look_at_the_ink():
        """Stage 2. What the probe measures from is the page's own ink: no
        window, no offset, no setting of any kind, just how far each pixel
        sits from the paper. Showing it is what lets the user see how far the user's
        threshold window is from what the page says (7 against 69 on the user's own
        page)."""
        shots, notes, record = collect(False)
        record('Probe', {'mask': page_ink(raw, probe.background_bgr),
                         'paper': probe.background_bgr,
                         'pen': probe.stroke_thickness_px,
                         'window': probe.suggested_block_size})
        return None, shots, notes

    def soften():
        shots, notes, _record = collect(False)
        grey = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        denoised = denoise_and_sharpen(grey, profile.denoise, 0)
        shots['Denoise'] = _thumbnail(denoised)
        detail_colour('Denoise', denoised)
        return grey, shots, notes

    def sharpen_it(grey):
        def make():
            shots, notes, _record = collect(False)
            sharp = denoise_and_sharpen(grey, profile.denoise, profile.sharpen)
            shots['Sharpen'] = _thumbnail(sharp)
            detail_colour('Sharpen', sharp)
            return sharp, shots, notes
        return make

    def threshold_it(sharp):
        def make():
            shots, notes, _record = collect(False)
            binary = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV,
                                           profile.adaptive_block_size,
                                           profile.adaptive_c)
            shots['Threshold'] = _thumbnail(255 - binary)
            detail_colour('Threshold', 255 - binary)
            return binary, shots, notes
        return make

    def morph_it(binary):
        def make():
            shots, notes, _record = collect(False)
            morphed = apply_morph(binary, profile.morph_open, profile.morph_close)
            shots['Morphology'] = _thumbnail(255 - morphed)
            detail_colour('Morphology', 255 - morphed)
            return morphed, shots, notes
        return make

    def perfect_it(morphed):
        def make():
            shots, notes, record = collect(False)
            clean, _allowed, _dropped = perfect_image(morphed > 0, raw, profile,
                                                      probe.background_bgr, record)
            return clean, shots, notes
        return make

    step('original', ('dpi',), look)
    step('probe', (), look_at_the_ink)
    grey = step('denoise', ('denoise',), soften)
    sharp = step('sharpen', ('sharpen',), sharpen_it(grey))
    binary = step('threshold', ('adaptive_block_size', 'adaptive_c'),
                  threshold_it(sharp))
    morphed = step('morphology', ('morph_open', 'morph_close'), morph_it(binary))
    clean = step('perfect', ('recover_interiors', 'min_area',
                             'correct_against_page'), perfect_it(morphed))

    one_key = ('one', page_key, _settings_key(profile, ONE_KEYS))

    def arrive(shots, hit):
        """A group of stages has finished -- or was already in hand.

        The user's rule: the stage the user chose is what the run is for. A cached group
        computed nothing and so announced nothing, which left two ways for an
        aimed run to miss its target: the picture never reached the user, and the
        run carried on into the stages after it. Both are closed here, at the
        one place a group of stages ends. Phase one does not use this any
        more -- each of its steps delivers its own -- but phases two and three
        are each one thing, and this is where they end.
        """
        deliver(shots, {})
        if cancel is not None and cancel.is_set():
            raise TuneCancelled()
        if upto and upto in shots:
            raise Reached()

    def build_map():
        shots, notes, record = collect()
        labels, colours, field = region_map(clean, raw, profile,
                                            probe.background_bgr,
                                            probe.stroke_thickness_px,
                                            correction=correction, stages=record)
        return (labels, colours, field), shots, notes

    map_key = ('map', one_key, _settings_key(profile, MAP_KEYS))
    (labels, colours, field), map_shots, map_notes, map_hit = _cached(map_key, build_map)
    milestones.update(map_notes)
    arrive(map_shots, map_hit)

    def build_strokes():
        shots, notes, record = collect()
        if not colours:
            return [], shots, notes
        drawn = labels >= 0
        made = encode_regions(labels, colours, drawn, raw, profile,
                              probe.background_bgr, probe.stroke_thickness_px,
                              record, cancel=cancel)
        validate_output(made, drawn, profile, labels, field, record)
        return made, shots, notes

    stroke_key = ('strokes', map_key, _settings_key(profile, STROKE_KEYS))
    segments, stroke_shots, stroke_notes, stroke_hit = _cached(stroke_key, build_strokes)
    milestones.update(stroke_notes)
    images.update(stroke_shots)
    if stroke_hit and on_image is not None:
        for name, picture in stroke_shots.items():
            if name not in sent:
                sent.add(name)
                on_image(name, picture)

    # **The same border the export drops.** Measured here from the page the preview already
    # rendered, so the picture the user is looking at cannot disagree with the notebook next door.
    from xp.workers.converter.convert import the_border_of

    strokes = segments_to_strokes(segments, profile, *geometry,
                                  border=the_border_of(raw, profile))
    output = XoppDocument.new_empty(1, profile.page_width, profile.page_height)
    output.pages[0].layer.extend(strokes)
    # Drawn at twice the scale it is shown at and scaled back down: PIL has
    # no antialiasing, and a hairline drawn straight into the pane comes out
    # as a staircase beside a source pane whose every edge is soft. The
    # measurement below reads the full-size image, where a thin line is still
    # solid ink rather than a grey suggestion of one.
    traced_scale = traced_preview_scale(profile, (raw_w, raw_h), geometry)
    drawn, images['Traced result'] = render_traced(output.pages[0], traced_scale)
    # The last stage, and the last mark. The export is the pipeline; what
    # follows it is the panel measuring the two panes against each other, so
    # that is held apart like any other preview.
    clock.mark('Traced result')
    measuring = clock.hold()

    scale, dx, dy = geometry
    frame = (dx / profile.page_width, dy / profile.page_height,
             raw_w / scale / profile.page_width, raw_h / scale / profile.page_height)
    grid = grid_size(raw_w, raw_h)
    diff = reference.compare(traced_ink(drawn, frame, grid))
    if profile.colour_5d:
        # The palette measure cannot read a measured colour: it matches by
        # palette *name*, and the 5-D pass gives a stroke the colour the page
        # had. It was reporting on nothing. The number on the panel is now
        # the one the Colour error stage draws.
        wrong = milestones.get('Colour error', {}).get('share', 0.0)
    else:
        labels, names = ink_classes(raw, reference.source, profile, grid)
        wrong = color_error(segment_classes(segments, names, (raw_w, raw_h), grid),
                            class_reach(labels, reference.kernel))
    clock.release(measuring)
    clock.finish()
    return Preview(profile, images, probe, len(strokes), diff.missing, diff.extra, frame,
                   diff.lost_share, diff.extra_share, diff.marks, diff.marks_lost, diff.marks_extra,
                   wrong, tuned,
                   pieces=milestones.get('Region colour map', {})
                   .get('colours', len(segments)),
                   page=output.pages[0])


class ConversionCancelled(Exception):
    pass


def _page_name(number):
    """What a page is called on the document bar. One place, because the bar
    and the timeline have to agree or every page comes out pending."""
    return 'Page %d' % number


def convert_to_file(source, target, profile, cancel=None, progress=None,
                    clock=None, say=None):
    """Write atomically, so cancelling/failing cannot truncate an existing result.

    `clock` is a `Timeline` whose order is the document's pages, so the bar
    at the bottom of the panel shows the conversion the same way it shows a
    page: a piece per page, as long as that page took.

    `say(text)` is called at each real step, because a page is not one thing.
    The user's correction, 2026-10-29: *"why render and suddenly the page is
    converted -- you mistakenly included the conversion of the entire page
    into the render."* The user was right: the panel said "rendering page 1" and
    kept saying it for the whole page, while rendering is the first two
    seconds of about twelve. The three steps are the render, the fit onto the
    Q base sheet, and the tracing, and the file is written after them all.
    """
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError('Choose a new .xopp output file; the input PDF must be preserved.')
    if target.suffix.lower() != '.xopp':
        raise ValueError('The output filename must end in .xopp.')
    profile = copy.deepcopy(profile)
    profile.auto_threshold = False
    cancel = cancel or threading.Event()
    with pymupdf.open(source) as pdf:
        if clock is not None:
            clock.reset([_page_name(n + 1) for n in range(pdf.page_count)])
            clock.begin()
        output = XoppDocument.new_empty(pdf.page_count, profile.page_width, profile.page_height)
        pages = output.pages
        for index in range(pdf.page_count):
            if cancel.is_set():
                raise ConversionCancelled()
            page = pdf[index]
            # The page's own ceiling, not a fixed one: past it MuPDF hands
            # back a white pixmap without raising, and the export would be an
            # empty notebook that looked like a finished conversion.
            page_dpi, _too_high = usable_dpi(page, profile.dpi)
            page_profile = (replace(profile, dpi=page_dpi)
                            if page_dpi != profile.dpi else profile)
            # In slices, so the window can paint while MuPDF works: the
            # pixels are identical and the main thread's worst freeze
            # drops from 1.15 s to 0.46 s. See `render_pdf_page`.
            if say:
                say('Page %d of %d: rendering at %d dpi'
                    % (index + 1, pdf.page_count, page_dpi))
            raw = render_pdf_page(page, page_dpi, in_slices=True)
            # **The same warp the preview uses**, or the saved notebook and the picture the user
            # approved would be two different conversions -- which the window's own test caught
            # the moment the preview started warping and this path did not.
            if say:
                say('Page %d of %d: fitting it to the sheet'
                    % (index + 1, pdf.page_count))
            raw, geometry = the_page_in_qbase(raw, page_profile)
            if say:
                say('Page %d of %d: tracing the ink -- the long part'
                    % (index + 1, pdf.page_count))
            strokes = convert_page_image(raw, page_profile, *geometry)
            if cancel.is_set():
                raise ConversionCancelled()
            pages[index].layer.extend(strokes)
            if progress:
                progress(index + 1, pdf.page_count)
            if clock is not None:
                clock.mark(_page_name(index + 1))
    if clock is not None:
        clock.finish()
    if say:
        say('Writing the notebook')
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.xp-', suffix='.xopp', dir=target.parent)
    os.close(fd)
    # **The background reference goes in before the save**, and the template is copied beside
    # the notebook by the same helper the CLI uses, so the recorded name is relative and resolves
    # wherever the folder goes.
    from xp.core.document.xopp_io import set_pdf_background
    from xp.workers.converter.convert import place_the_background

    background = place_the_background(target)
    if background is not None:
        for page in output.pages:
            set_pdf_background(page.element, background, pageno=1)
    try:
        output.save(temporary)
        if cancel.is_set():
            raise ConversionCancelled()
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def _put_in_place(staged: Path, target: Path) -> None:
    """Move a finished conversion onto its target **without a window of loss**.

    `shutil.move` is a rename only while both paths are on one volume. The
    staging file is made in the system TEMP, so converting a PDF onto another
    drive -- a USB stick, a second disk, exactly the case this project is meant
    to keep working on -- falls back to copy-then-delete: the target is
    truncated and rewritten in place, and an interruption in the middle leaves
    a half notebook where a good one was. Measured on the code before this:
    `mkstemp(prefix='xp-staged-')` with no `dir=`, then `shutil.move`.

    So a cross-volume move is done in two steps: copy to a temporary file **in
    the target's own directory**, flush it, and `os.replace` it. The rename is
    then on one volume whatever the staging file's volume was.
    """
    import os

    from xp.core.document.atomic import fsync_directory, sweep_partials

    sweep_partials(target)
    if Path(staged).anchor == Path(target).anchor:
        os.replace(staged, target)
        fsync_directory(target.parent)
        return
    _copy_into_place(staged, target)


def _copy_into_place(staged: Path, target: Path) -> None:
    """The cross-volume half: copy beside the target, then rename over it."""
    import os

    from xp.core.document.atomic import fsync_directory

    handle, temporary = tempfile.mkstemp(prefix=target.name + '.partial-',
                                         dir=str(target.parent))
    os.close(handle)
    try:
        shutil.copyfile(staged, temporary)
        # A read-only handle cannot be flushed on Windows (`os.fsync` on it
        # raises EBADF), and the point is to get the bytes on the disk before
        # the rename, so the handle is opened writable.
        with open(temporary, 'rb+') as check:
            os.fsync(check.fileno())
        os.replace(temporary, target)
        fsync_directory(target.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    try:
        os.unlink(staged)
    except OSError:
        pass


class ConversionWindow:
    """The converter, with its stages, its settings and its autotune.

    `settle` turns the same window into the merge's conversion step instead
    of a document conversion. Everything the user can do is identical -- the
    sixteen stages, the sliders, Autotune, the previews; the one difference
    is what the button at the bottom does. With `settle` it hands back the
    settings the user arrived at, and the merge converts the pages the brush took
    from and finishes the save. Without it, it writes a `.xopp` as it always
    has.

    `pages` limits which pages the user can step through, for the same reason: the
    merge only has a question about the pages the brush touched.
    """

    def __init__(self, path, master=None, mode='live', profile=None,
                 settle=None, pages=None, on_finish=None):
        self.path = Path(path).resolve()
        self.settle = settle
        # Told once, when this window's task is over: saved, or closed by the user.
        # Not while it is working, and not when the merge opened it as a step
        # inside its own task -- see `_report_done`.
        self.on_finish = on_finish
        self._settled = False
        self.only_pages = sorted(set(int(n) for n in pages)) if pages else None
        with pymupdf.open(self.path) as pdf:
            self.page_count = pdf.page_count
            # **How high this document's Resolution field may go.** Not the
            # 1200 that used to be written here: past a page's own pixel cap
            # MuPDF hands back a white pixmap without raising, and the user got two
            # blank panes and "why i cnats ee shit after i ajdustthe dpi".
            # The smallest ceiling across the pages, so every page of this
            # document can be rendered at whatever the field allows -- a
            # number that is right for the document the user opened rather than for
            # some other one.
            self.dpi_ceiling = min(
                [max_dpi_for(page.rect.width, page.rect.height) for page in pdf]
                or [1200])
        if not self.page_count:
            raise ValueError('This PDF has no pages.')
        self.root = tk.Toplevel(master) if master is not None else tk.Tk()
        from xp.runtime.shutdown import arm
        arm(self.root)
        from xp.runtime.shutdown import fill_the_screen
        fill_the_screen(self.root)
        # The code version on the title, so "did my change reach this window?" is
        # answerable without restarting hopefully.
        from xp.runtime.version import code_version

        self.root.title(f'XP / Convert / {self.path.name}  ·  {code_version()}')
        # No size from the screen here. `_fit_to_content` asks the widgets
        # what they need once they exist, which is the only answer that can be
        # right on a display and a font this file has never seen.
        decorate(self.root)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.profile = copy.deepcopy(profile) if profile else Profile.named('hand')
        self.preview = None
        self._live = {}
        # The full arrays behind the stage the user is looking at, and the original
        # render, so a zoomed pane can show the raster the pipeline used
        # instead of a 2,200-pixel thumbnail. At most two entries.
        self._details = {}
        self._closed = False
        self._working = False
        self._running = None
        self._converting = False
        # A conversion that finished before the user said where it goes. The user may
        # only have been looking -- stepping through the stages, trying
        # settings -- and then decided against the whole thing, so the name
        # is asked for after there is something to name, not before. What is
        # written in the meantime lands here, and it stays here if the user refuses
        # the dialog: pressing Convert again offers it rather than converting
        # the document a second time.
        self._staged: Path | None = None
        # **What produced the staged file.** Re-using it after the settings
        # changed handed the user the old output and recorded the new profile: the
        # result was stale and the Learning Material could not be reproduced
        # from the parameters written beside it. The snapshot is what decides
        # whether the staged file may be offered again.
        self._staged_profile = None
        # Ctrl+Shift+S sets this for one save; Ctrl+S leaves it alone and
        # the default name is used without a dialog.
        self._ask_where = False
        # The Learning Material tick in Save As. On by default, as the user asked,
        # and remembered for the next Save As; it does not apply to Ctrl+S,
        # which is the quick save and asks nothing.
        self.learning_material = tk.BooleanVar(self.root, True)
        self._keep_learning_material = False
        # Close once the file is on disk. Tests that look at the window
        # afterwards turn it off; nothing in the program does.
        self.exit_after_save = True
        self._pending = None
        self._generation = 0
        self._debounce = None
        self._cancel = threading.Event()
        self._results = queue.Queue()
        self._photos = []
        self.values = {name: tk.StringVar(self.root, as_he_typed_it(getattr(self.profile, name))) for name, *_ in FIELDS}
        # Opened on the first page the caller cares about: the merge only
        # has a question about the pages the brush took something from.
        self.page = tk.IntVar(self.root,
                              self.only_pages[0] if self.only_pages else 1)
        self.stage = tk.StringVar(self.root, 'Original')
        self.mode = tk.StringVar(self.root, LIVE if mode == 'live' else AUTO)
        self.palette = tk.StringVar(self.root, self.profile.mode)
        self.trace = tk.StringVar(self.root, self.profile.trace)
        # One clock for the panel, handed to every run, built on the stage
        # list so every stage is on the bar whether or not it has run. The
        # bar reads it while the worker thread writes it, hence the lock.
        self.clock = timeline.Timeline(STAGES)
        # The document's own clock, a piece per page. The bar draws whichever
        # of the two is working, because a converting panel should be showing
        # the conversion and not the last page the user happened to preview.
        self.pages_clock = timeline.Timeline(noun='pages')
        self._shown = self.clock       # which of the two the bar is drawing
        self.status = tk.StringVar(self.root, 'Auto-tuning the settings on page 1...' if mode != 'live'
                                   else 'Measuring the page and preparing suggested settings...')
        self._tuning = None  # cancel event of the auto-tune in progress
        # Shared view state for the two preview panes: a zoom factor and a
        # pan centre in normalized (0..1) page coordinates. Both panes are
        # cropped from this same (zoom, pan) on every redraw, so "old" and
        # "new" always show the same patch of the page at the same scale --
        # there's one view, not two independently-scrolled ones to keep lined
        # up by hand.
        self.zoom = 1.0
        self.pan = (0.5, 0.5)
        self._pane_rects = []  # filled by _draw(): (x, y, disp_w, disp_h) per pane
        self._pan_drag = None
        self._drag_end = None
        # The pointer, recorded by <Motion> and read by the follow loop. Same
        # shape as `PagePreview`: the callback moves nothing; the loop does.
        self._pointer = {'x': None, 'y': None}
        self._follow_job = None
        # On by default (the user's last sentence), and a click on the canvas toggles
        # it. A drag is not a click: `_on_pan_end` measures the movement.
        self._follow_enabled = True
        self.zoom_label_var = tk.StringVar(self.root, '')
        self.show_diff = tk.BooleanVar(self.root, True)
        self.lost_text = tk.StringVar(self.root, LOST_LABEL)
        self.extra_text = tk.StringVar(self.root, EXTRA_LABEL)
        self._build()
        self._install_field_keys()
        self._fit_to_content()
        self._poll_id = self.root.after(60, self._poll)
        # **The view follows the pointer, the way the arranger's page does.**
        # Started after the canvas exists; the first ticks do nothing until
        # there is a picture and a pane to measure.
        self._follow_job = self.root.after(FOLLOW_MS, self._follow_loop)
        # Auto mode tunes on open; live mode starts from the probe's quick
        # guess so the stages are up at once, with Auto-tune a click away.
        self.root.after(100, lambda: self.request_preview(suggest=mode == 'live', tune=mode != 'live'))

    def _fit_to_content(self):
        """Open at the size the widgets ask for; the screen only caps it.

        The window used to be sized from the screen -- `min(740, screen height
        - 140)` -- and how tall a settings row is comes from the font, which
        comes from the display. Measured on this screen the column asks for
        1,342 pixels, so with a 740-pixel window the last six settings, the
        whole OUTPUT section and three of the buttons were below the bottom
        edge, with no scrollbar to reach them: not clipped, unreachable.

        Asking the widgets is the only version of this that can be right on a
        display and a font this file has never seen, and it is why the fix is
        not a bigger number -- a bigger number is wrong on the next screen,
        which is the shape the bug had already taken.

        The height floor is deliberately below the window: the settings column
        scrolls now, so a shorter window is a smaller view rather than a
        cut-off form. The width floor is not, because the pane and the viewer
        are side by side and no amount of scrolling puts them back.
        """
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        want_w = min(max(760, self.root.winfo_reqwidth()), screen_w - 48)
        want_h = min(max(520, self.root.winfo_reqheight()), screen_h - 140)
        # **No geometry call: the window opens maximised and stays there.** It
        # used to be maximised by `fill_the_screen` and then *shrunk* to the
        # content's size a moment later, which is the "some windows shrink when
        # they open" the user reported -- and on a narrow window the settings column
        # and the two panes are the first thing to be cut. The minimum keeps the
        # panes usable if the user resizes; it never moves the window.
        self.root.minsize(min(want_w, screen_w - 48), min(520, want_h))

    def _build(self):
        header = ttk.Frame(self.root, style='Lab.TFrame', padding=14)
        header.pack(fill='x')
        # **The key map goes top right, where the user looked for it.** It was in
        # the footer beside the buttons and the user never saw it: "I didn't see
        # that hotkey button, which should be hidden in the right up part of
        # the GUI". Packed before the labels so it keeps the corner however
        # long the file name is.
        self.hotkey_button = HotkeyButton(header, HOTKEYS, title='Converter')
        self.hotkey_button.pack(side='right', anchor='ne')
        ttk.Label(header, text='PDF  /  EDITABLE INK', style='Title.Lab.TLabel').pack(anchor='w')
        ttk.Label(header, text=self.path.name, style='Muted.Lab.TLabel').pack(anchor='w')
        footer = ttk.Frame(self.root, style='Lab.TFrame', padding=10)
        footer.pack(side='bottom', fill='x')
        # The user's line, cut at every stage, each piece as long as that stage
        # took. Above the buttons and across the whole width, because the
        # thing it has to show is proportion and a short bar cannot.
        self._take_the_keyboard()
        self.refusal_notice = ttk.Label(
            footer, text=' ', style='Lab.TLabel', foreground=REFUSED_COLOUR,
            wraplength=520, justify='left')
        self.refusal_notice.pack(side='top', fill='x', pady=(0, 4))
        self.timing = timingbar.TimingBar(footer, self._working_clock)
        self.timing.pack(side='top', fill='x', pady=(0, 6))
        self.convert_button = ttk.Button(
            footer,
            text=('Use these settings' if self.settle is not None
                  else 'Convert document...'),
            command=self.convert, state='disabled')
        self.convert_button.pack(side='right', padx=4)
        self.cancel_button = ttk.Button(footer, text='Cancel conversion', command=self._cancel.set, state='disabled')
        self.cancel_button.pack(side='right', padx=4)
        # **One row, wrapped at the window's width.** It was `wraplength=560`
        # beside the buttons, so on a maximised window it broke into two lines
        # and the second one ran under them. The wrap now follows the window, and
        # the row's height stays what the zoom maths was measured against -- an
        # extra row here shortens the canvas and the pane's own geometry with it.
        self.status_label = ttk.Label(footer, textvariable=self.status,
                                      style='Muted.Lab.TLabel', wraplength=900)
        self.status_label.pack(side='left', fill='x', expand=True)
        # The panes' own keys, on the same row: one row high, so the canvas keeps
        # the height the zoom maths was measured against.
        ttk.Label(footer, style='Muted.Lab.TLabel',
                  text='Click: follow on/off   Wheel: zoom   Drag: pan   '
                       'Double-click: fit   Both panes show the same place').pack(side='right', padx=(12, 4))
        self.root.bind('<Configure>', self._wrap_the_status, add='+')
        body = ttk.Frame(self.root, style='Lab.TFrame', padding=(14, 0, 14, 8))
        body.pack(fill='both', expand=True)
        # The settings column is about forty rows, and how tall a row is comes
        # from the font, which comes from the display. Measured here it asks
        # for 1,342 pixels; the window it was put in was 740, chosen from the
        # screen, so the last six settings, the whole OUTPUT section and three
        # of the buttons sat below the bottom edge with nothing to scroll them
        # with. A list taller than the screen has to scroll -- the alternative
        # is the one that was being done, adding to a fixed height, and it
        # cannot work on a display shorter than the list.
        shell = ttk.Frame(body, style='Lab.TFrame')
        shell.pack(side='left', fill='y', padx=(0, 14))
        pane = tk.Canvas(shell, highlightthickness=0, borderwidth=0,
                         bg=appearance.ABYSS)
        bar = ttk.Scrollbar(shell, orient='vertical', command=pane.yview)
        pane.configure(yscrollcommand=bar.set)
        # **A gap between the fields and the bar.** The canvas is set to
        # exactly the width the settings ask for, so the widest entry ends at
        # the same pixel the scrollbar starts at -- measured: entry right edge
        # 297, scrollbar x 297. Nothing overlapped in the layout's own terms,
        # but on screen a field and a bar that touch read as one thing sitting
        # on the other, which is what the user saw. The gap is a few pixels of the
        # panel's own padding unit, not a number picked to look right at one
        # font size.
        bar.pack(side='right', fill='y', padx=(SETTINGS_BAR_GAP, 0))
        pane.pack(side='left', fill='both', expand=True)
        settings = ttk.Frame(pane, style='Lab.TFrame')
        inner = pane.create_window((0, 0), window=settings, anchor='nw')

        def _settings_grew(_event=None):
            """Scroll to what is in the pane, and be as wide as it needs.

            Width as well as height, because the column's natural width comes
            from the captions and the font: the window asks for what the
            settings ask for, and this is what carries that request out of the
            scrolling canvas, which would otherwise report only its own.
            """
            pane.configure(scrollregion=pane.bbox('all'),
                           width=settings.winfo_reqwidth())

        settings.bind('<Configure>', _settings_grew)

        def _wheel(event):
            pane.yview_scroll(-1 if event.delta > 0 else 1, 'units')

        def _wheel_everywhere(widget):
            """The wheel over anything in the pane scrolls the pane.

            Bound on each widget rather than on the window, because a wheel
            over the viewer must keep doing what it does -- and a binding on
            the toplevel would take it.
            """
            widget.bind('<MouseWheel>', _wheel)
            for child in widget.winfo_children():
                _wheel_everywhere(child)

        self.controls = []
        ttk.Label(settings, text='WORKFLOW', style='Muted.Lab.TLabel').grid(row=0, column=0, sticky='w', columnspan=2)
        workflow = ttk.Combobox(settings, textvariable=self.mode, values=(AUTO, LIVE), state='readonly', width=26)
        workflow.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(3, 12))
        workflow.bind('<<ComboboxSelected>>', lambda _e: self._mode_changed())
        self.controls.append(workflow)
        ttk.Label(settings, text='Ink palette', style='Lab.TLabel').grid(row=2, column=0, sticky='w')
        palette = ttk.Combobox(settings, textvariable=self.palette, values=('hand', 'elec'), state='readonly', width=9)
        palette.grid(row=2, column=1)
        palette.bind('<<ComboboxSelected>>', lambda _e: self.request_preview())
        self.controls.append(palette)
        ttk.Label(settings, text='Stroke shape', style='Lab.TLabel').grid(row=3, column=0, sticky='w', pady=(6, 0))
        shape = ttk.Combobox(settings, textvariable=self.trace,
                             values=('onion-fill', 'centerline', 'outline'),
                             state='readonly', width=14)
        shape.grid(row=3, column=1, pady=(6, 0))
        # **The list of stages belongs to this choice**, so it is refreshed
        # before the preview starts -- otherwise `Medial axis` appears only after
        # the next picture lands, one choice late.
        shape.bind('<<ComboboxSelected>>',
                   lambda _e: (self._refresh_stage_list(), self.request_preview()))
        self.controls.append(shape)
        # Grouped by the stage each setting acts in, in pipeline order, so
        # the panel reads the same way the stage list does.
        self.widgets = {}
        self._focus_by_stage = {}      # the parameter the user was last on, per stage
        self._focused_parameter = None
        row = 4
        for title, group in SECTIONS:
            ttk.Label(settings, text=title, style='Muted.Lab.TLabel').grid(
                row=row, column=0, columnspan=2, sticky='w', pady=(10, 2))
            row += 1
            for name, label, kind, _low, _high in group:
                caption = ttk.Label(settings, text=label, style='Lab.TLabel')
                caption.grid(row=row, column=0, sticky='w', pady=2)
                if kind is bool:
                    widget = ttk.Checkbutton(settings, variable=self.values[name],
                                             onvalue='True', offvalue='False',
                                             command=self._edit)
                else:
                    widget = ttk.Entry(settings, textvariable=self.values[name], width=10)
                    widget.bind('<KeyRelease>', self._edit)
                widget.grid(row=row, column=1, padx=(10, 0), pady=2, sticky='e')
                self.widgets[name] = (caption, widget)
                self.controls.append(widget)
                row += 1
        row += 1
        # 'Auto-tune this page' used to sit first here. The user's instruction,
        # 2026-10-28: *"change this into save the parameters as the default,
        # so sometimes I can cook it step by step, as the auto-tune is a
        # redundant button, not necessary."* The search itself is untouched --
        # `python -m xp convert --auto-tune` still runs it -- but the button is
        # gone and the slot now does the thing the user wants from this panel: keep
        # what the user has arrived at, so the next window opens on it.
        for offset, (label, command) in enumerate((('Save these settings as the default', self.save_as_default), ('Reset to profile defaults', self.reset_defaults), ('Save settings as JSON...', self.save_profile), ('Load settings...', self.load_profile))):
            button = ttk.Button(settings, text=label, command=command)
            button.grid(row=row + offset, column=0, columnspan=2, sticky='ew', pady=3)
            self.controls.append(button)
        ttk.Label(settings, text='Save these settings as the default and every window after this one opens on them, so a page can be tuned a step at a time across sittings.\nReset goes back to that saved default; Load settings... opens the ones XP ships with.\nRed and blue show what still differs.\nEdits apply to every output page.', style='Muted.Lab.TLabel', wraplength=250).grid(row=row + 4, column=0, columnspan=2, sticky='w', pady=12)
        _wheel_everywhere(settings)
        viewer = ttk.Frame(body, style='Lab.TFrame')
        viewer.pack(fill='both', expand=True)
        nav = ttk.Frame(viewer, style='Lab.TFrame')
        nav.pack(fill='x', pady=(0, 8))
        ttk.Label(nav, text='Page', style='Lab.TLabel').pack(side='left')
        first = self.only_pages[0] if self.only_pages else 1
        last = self.only_pages[-1] if self.only_pages else self.page_count
        page = ttk.Spinbox(nav, from_=first, to=last, textvariable=self.page, width=5, command=self.request_preview)
        page.pack(side='left', padx=5)
        page.bind('<Return>', lambda _e: self.request_preview())
        self.controls.append(page)
        ttk.Label(nav, text=f'/ {self.page_count}', style='Muted.Lab.TLabel').pack(side='left')
        stage = ttk.Combobox(nav, textvariable=self.stage, values=STAGES, state='readonly', width=17)
        self.stage_box = stage
        stage.pack(side='right')
        stage.bind('<<ComboboxSelected>>', self._stage_chosen)
        self.stage_button = ttk.Button(nav, text='Next stage >', command=self.next_stage)
        self.stage_button.pack(side='right', padx=6)
        # In `self.controls` so they grey out with everything else while a
        # document is converting. They were not, and `request_preview`
        # returns on its first line while `_converting` -- so the list opened,
        # a stage could be picked, and nothing happened and nothing said why.
        # The question was "why can't I choose in this list?", and the honest
        # answer was that the user could, and it did nothing.
        self.controls.append(stage)
        self.controls.append(self.stage_button)
        ttk.Button(nav, text='Reset view', command=self._reset_view).pack(side='right', padx=6)
        ttk.Label(nav, textvariable=self.zoom_label_var, style='Muted.Lab.TLabel', width=5).pack(side='right')
        diffbar = ttk.Frame(viewer, style='Lab.TFrame')
        diffbar.pack(fill='x', pady=(0, 6))
        ttk.Checkbutton(diffbar, text='Show differences', variable=self.show_diff,
                        command=self._draw, style='Lab.TCheckbutton').pack(side='left')
        for color, variable in ((MISSING_COLOR, self.lost_text), (EXTRA_COLOR, self.extra_text)):
            tk.Label(diffbar, bg='#%02x%02x%02x' % color, width=2, relief='flat').pack(side='left', padx=(14, 5))
            ttk.Label(diffbar, textvariable=variable, style='Lab.TLabel').pack(side='left')
        # **What the pane is really showing.** The field says 6,000 DPI; a
        # thumbnail of a 6,000-DPI page is 405 DPI, and calling both "the
        # preview" is the lie the user asked about. This says the resolution of the
        # picture on screen, updated as the window is drawn.
        self.detail_text = tk.StringVar(self.root, '')
        ttk.Label(diffbar, textvariable=self.detail_text,
                  style='Muted.Lab.TLabel').pack(side='right')
        # **The toggle the user asked for.** A left click on the picture starts or
        # stops the follow; this label is the state, not another control.
        self.follow_text = tk.StringVar(self.root, '')
        ttk.Label(diffbar, textvariable=self.follow_text,
                  style='Lab.TLabel').pack(side='right', padx=(0, 14))
        self._show_follow_state()
        self.canvas = tk.Canvas(viewer, bg='#1a2632', highlightthickness=0)
        # **Something to watch while the user waits.** The user's instruction,
        # 2026-10-29: *"I want, when I open the GUI, even before the
        # original page is loading, I can trigger Ctrl+S ... and just
        # show me a loading reminder pop out, and I shall wait with
        # something I can see."* A line of status text at the bottom of
        # a window that has stopped repainting is not something the user can
        # see; this is a panel in the middle of the picture, with a bar
        # that pulses until the page count is known and then counts.
        self.loading = Progress(self.canvas)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda _e: self._draw())
        self.canvas.bind('<MouseWheel>', self._on_wheel)
        self.canvas.bind('<Motion>', self._record_the_mouse)
        self.canvas.bind('<Leave>', self._forget_the_mouse)
        self.canvas.bind('<ButtonPress-1>', self._on_pan_start)
        self.canvas.bind('<B1-Motion>', self._on_pan_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_pan_end)
        self.canvas.bind('<Double-Button-1>', self._reset_view)
        # Keys. The converter had none: every command here was a button or a
        # field. Ctrl+Up/Down walk the stages through the same path the
        # dropdown uses; PageUp/PageDown walk the pages being previewed.
        # The keys the user specified. Alt, not Ctrl, so Ctrl+A/C/V/X keep meaning
        # select-all, copy, paste and cut inside the boxes they land in.
        # **Both cases, because CapsLock changes the keysym.** `<Alt-w>` does
        # not match when the key arrives as `W` -- which it does under
        # CapsLock, and with Shift. The arranger learned this and binds both;
        # the converter bound only the lower case, so Alt+WASD was dead for
        # anyone whose CapsLock was on. Measured by asking Tk: only
        # `<Alt-Key-w>` was there, never `<Alt-Key-W>`.
        for key, what, delta in (('w', 'stage', -1), ('s', 'stage', 1),
                                 ('a', 'parameter', -1), ('d', 'parameter', 1)):
            for case in (key, key.upper()):
                self.root.bind(
                    '<Alt-KeyPress-%s>' % case,
                    lambda _e, what=what, delta=delta: self._alt_command(what, delta))
                self.root.bind(
                    '<Alt-%s>' % case,
                    lambda _e, what=what, delta=delta: self._alt_command(what, delta))
        # **The same four on Ctrl+Alt, because Ctrl is what the user's log shows arriving.** Alt alone is a
        # system key on Windows and never reached Tk; a Ctrl chord does come through, and both
        # modifiers together are not a menu activation.
        # **`bind_all` for these four, and only these.** A Ctrl chord is not consumed by any widget
        # class, and `bind_all` is the one binding that still fires when the focus is inside a
        # parameter field or the page spinbox -- which is where it is after the user types a number, and
        # where plain W/S cannot be used because the letter belongs to the field.
        # **Ctrl+arrows, which the user's log proves arrive.** `Ctrl+Down`, `Ctrl+Up` and `Ctrl+Right`
        # all reached the window in the key log; `Alt` never did and `Ctrl+Alt` landed as a bare
        # `Control_L`. So the parameter stepping gets the arrow keys too.
        # **The keys that actually arrive.** The user's key log has no Alt in it at all -- not `Alt-s`, not
        # even a bare `Alt_L` -- while `Control`, `S` and `Shift` come through. On the user's machine the
        # Alt chords never reach the window (Windows turns them into a menu activation before Tk
        # sees a key). So the same four actions are also on plain WASD, guarded so they type
        # normally whenever a field has the focus.
        self.root.bind('<KeyPress>', self._a_plain_key)
        self.root.bind('<Next>', lambda _e: self.step_page(1))
        self.root.bind('<Prior>', lambda _e: self.step_page(-1))
        # **Which of the two it is comes from the Shift bit, not from the
        # letter's case.** The user's report, 2026-10-29: *"why Ctrl+S doesn't work?
        # only Ctrl+Shift+S work?"* -- with **Caps Lock on**, which it was.
        # Measured on the user's machine, through real key presses:
        #
        #   Caps on, Ctrl+S         keysym 'S', state 0x6 (Control|Caps)
        #                           `<Control-s>` wants 's'        -> no match
        #                           `<Control-Shift-S>` wants Shift -> no match
        #                           so nothing fired at all
        #   Caps on, Ctrl+Shift+S   Shift and Caps cancel: keysym 's'
        #                           -> matched `<Control-s>` and did a plain
        #                           save, which is not what that key means
        #
        # So it was worse than one dead key: the other one was quietly doing
        # the wrong save. Binding both letter cases and reading `state` is
        # what the arranger already does (`_on_any_key` lowercases its keysym
        # for exactly this reason); the converter never got the same cure.
        for case in ('s', 'S'):
            self.root.bind('<Control-%s>' % case, self._save_key)
            self.root.bind('<Control-KeyPress-%s>' % case, self._save_key)
        self._mode_changed()

    def _mode_changed(self):
        # Never *enable* while a conversion is running: `_set_controls` has
        # just greyed everything, and a widget that turns itself back on is
        # how the dropdown came to be bright with nothing behind it.
        self._mode_changed_state()
        # Live opens on the render, which is the first picture there is;
        # the other modes want the finished tracing.
        self.stage.set('Original' if self.mode.get() == LIVE else 'Traced result')
        self._draw()

    def _take_the_keyboard(self) -> None:
        """**The window has to actually have the keyboard.**

        The user's report: "none of them work, the Ctrl+S or Alt+WASD". The bindings were there -- measured,
        Alt+S fires from the canvas, from a field and from the toplevel -- but a key goes to
        whichever *application* is in front, and a Toplevel opened from the dashboard does not
        always take the focus from it (or from whatever else was in front, Xournal++ in the user's case).
        So: lift it, take the focus, put the focus in the canvas, and pulse topmost for a moment so
        it comes up in front of the window that opened it.
        """
        # **No topmost pulse, and the keys stay on the toplevel.** The user said the keys used to work
        # and stopped: the difference was `bind_all` plus a 300 ms `-topmost` flash, both of which I
        # added while chasing this. `bind_all` sits at the very end of Tk's bindtags, so anything
        # that calls `break` earlier wins -- and the flash took the focus away and back. The keys
        # are bound exactly as the version that worked had them; what is kept is only the lift and
        # the focus, which cannot remove a binding.
        try:
            self.root.deiconify()
            self.root.lift()
            # **The window, not only Tk's focus.** The question: "why the focus is automatically in
            # the terminal not that GUI". `focus_force` moves the focus inside Tk; it does not make
            # this the Windows foreground window, so the launcher's console kept the keyboard and
            # every key went there. `SetForegroundWindow`, with the thread attach Windows requires,
            # is what actually moves it -- and it is retried once at 400 ms, because the first call
            # can land before the window is mapped.
            from xp.runtime.foreground import take_the_foreground

            take_the_foreground(self.root)
            self.root.after(400, lambda: take_the_foreground(self.root))
            self.root.focus_force()
            self.canvas.focus_set()
        except Exception:
            pass

    def _working_clock(self):
        """Which of the two the bar should be drawing.

        The document's, while a document is being converted or as soon as one
        has been; the page's otherwise. A panel in the middle of a conversion
        showing the stages of the page the user last previewed would be showing the user
        the wrong thing at the only moment the user is waiting.
        """
        if self._converting:
            self._shown = self.pages_clock
            return self.pages_clock
        # And for as long as the finished conversion is the last thing that
        # happened. The first line of this used to be `or
        # self.pages_clock.measured()`, which is true for ever once a
        # document has been converted -- so the bar never went back to the
        # stages of a page, which is what the panel is mostly for.
        if self._shown is self.pages_clock and not self.clock.running():
            return self.pages_clock
        self._shown = self.clock
        return self.clock

    def _picture(self, name):
        """The newest picture for a stage: the one still arriving if there is
        one, otherwise the finished page's."""
        live = getattr(self, '_live', {})
        if name in live:
            return live[name]
        if self.preview is not None:
            return self.preview.images.get(name)
        return None

    def available_stages(self):
        """Every stage, always -- the user's choice is not limited by what has arrived.

        This used to list only the stages the page had already been through,
        which reads as a tidy menu and is the wrong way round: the user aims at a
        stage *before* it exists, the run follows the aim, and a stage missing
        from the list is a stage the user cannot ask for at all. A stage that has
        not been computed yet is shown as pending and computed behind it.
        """
        # **Only the stages this shape actually has.** The question was "why so
        # many stages -- should it follow my settings". `Medial axis` and
        # `Orthogonal filling` are two different ways to make a stroke, chosen by
        # `Stroke shape`, so the one that is not in use is removed; `Original` and
        # `Traced result` are always there, because every shape starts and ends
        # with them.
        stages = list(STAGES)
        # The combo's value first: it is what the user is looking at, and the profile
        # is only updated when a preview is asked for -- so reading the profile
        # here made the list lag one choice behind.
        shown = getattr(self, 'trace', None)
        if shown is not None and hasattr(shown, 'get'):
            trace = str(shown.get())
        else:
            trace = str(getattr(self.profile, 'trace', 'ortho-fill'))
        if trace == 'ortho-fill':
            stages = [name for name in stages if name != 'Medial axis']
        else:
            stages = [name for name in stages if name != 'Orthogonal filling']
        return stages

    def _stage_chosen(self, _event=None):
        """The user's choice re-aims the machine, it does not merely change the view.

        Whatever is running is stopped, and a run aimed at the stage the user chose
        is started -- every time, and not only when the picture is missing.
        It used to return early when the picture was already in hand, which
        looked like a saving and was the bug the user reported: the run the user had just
        overruled went on computing the stages after the one the user was looking
        at, for as long as the whole page took.

        Re-aiming at a stage already computed is not a recomputation. The
        three stage groups are cached on the settings they read, so the run
        walks to the user's stage through the cache and stops there; what it costs
        is the redraw. The redraw comes first so the picture the user has is up
        before the run even starts.
        """
        # **The mark moves first, before any work is asked for.** The user's order:
        # the stage being viewed is marked in dark blue on the bar, and changing
        # stage moves it at once without restarting the computation. Drawing it
        # here is what makes "at once" true even while a run is in flight.
        try:
            self.timing.mark_viewing(self.stage.get())
        except Exception:
            pass            # a bar that cannot draw must not break a stage change
        self._draw()
        if self.mode.get() != LIVE:
            # Outside the live view a run is aimed at the finished tracing, so
            # the user's stage is usually produced on the way there. If it is not --
            # and nothing is running to produce it -- ask for it rather than
            # leaving a pending pane with no work behind it.
            if (self._picture(self.stage.get()) is None
                    and not self._working and not self._pending):
                self.request_preview(upto=self.stage.get())
            return
        self.request_preview(upto=self.stage.get())

    def next_stage(self):
        """The button is another way of choosing a stage, so it chooses one.

        It used to set the name and redraw, and nothing else: no stop, no
        re-aim, no computation. From a run aimed at `Original` -- which is
        where a live session starts -- pressing it moved the label to `Probe`
        and left the user looking at an empty pane with the status line still
        reporting the stage before. The dropdown had been fixed and this had
        not, which is the same bug twice: *every* way of choosing a stage
        has to go through the one path, or there is a way of choosing that
        the machine does not follow.
        """
        stages = self.available_stages()
        if not stages:
            return
        current = self.stage.get()
        index = stages.index(current) if current in stages else -1
        self.stage.set(stages[min(index + 1, len(stages) - 1)])
        self._stage_chosen()

    # ---- editing keys in the parameter fields ---------------------------

    FIELD_TYPES = None

    def _install_field_keys(self) -> None:
        """`Ctrl+A/C/V/Z` work in **every** parameter field.

        The user's order (2026-10-26): "Alt+A/D get into the parameter fields; once in,
        Ctrl+A/C/V/Z must work normally." Tk gives an Entry select-all and
        clipboard keys on Windows, but **no undo at all**, and a field that
        cannot be undone is a field the user has to retype. So: the handlers below act
        on whatever widget has the focus, whichever of the panel's field types
        it is, and they are bound once on the window so a field added later is
        covered without anyone remembering to bind it.
        """
        self.root.bind_all('<Control-a>', self._field_select_all, add='+')
        self.root.bind_all('<Control-z>', self._field_undo, add='+')
        self.root.bind_all('<Control-y>', self._field_redo, add='+')
        self.root.bind_all('<KeyRelease>', self._field_remember, add='+')
        # **Enter commits the field and re-runs.** The question, word for word: "why I
        # change the parameter and it did not rerun?" Nothing was bound to Return on a
        # parameter field at all -- only the page spinbox had one -- so a typed value sat
        # in the box while the picture kept showing the previous settings. The binding is
        # on the window, like the others, so a field added later is covered too.
        self.root.bind_all('<Return>', self._field_enter, add='+')

    def _mark_refused(self, name, message):
        """Put the refusal where the user is looking: on the field, not only in the status line.

        The user's order: "you should pop out shit when I input sth unacceptable for all params". The
        caption of the offending field turns red and carries the range, and the widget gets a
        red border; the next accepted read clears it.
        """
        self._refused = name
        self._refused_message = message
        notice = getattr(self, 'refusal_notice', None)
        if notice is not None:
            try:
                notice.configure(text=' ')
            except Exception:
                pass
        widget = self.widgets.get(name)
        if widget is not None:
            caption, control = widget
            try:
                caption.configure(foreground=REFUSED_COLOUR)
            except Exception:
                pass
            try:
                control.configure(style='Refused.TEntry')
            except Exception:
                # The style may not exist in a bare theme; the caption is the signal that
                # always works, so a missing style is not allowed to break the refusal.
                pass
        self.status.set(message)

    def _clear_refused(self):
        name = getattr(self, '_refused', None)
        if not name:
            return
        widget = self.widgets.get(name)
        if widget is not None:
            caption, control = widget
            try:
                caption.configure(foreground=MUTED)
            except Exception:
                pass
            try:
                control.configure(style='Field.TEntry')
            except Exception:
                pass
        self._refused = None
        self._refused_message = None

    def _refuse_loudly(self, exc):
        """A refused value, said on the field and in a red notice -- **not in a dialog**.

        The user asked for the refusal to pop out, and a modal `messagebox` does that and takes the
        keyboard with it: while it is up every key goes to the dialog, and a second refusal stacks
        another one behind it. That is one more way for "none of the keys work" to happen right after
        the user types something the program does not like. The notice is a red line in the panel, always
        visible, and it never owns the keyboard.
        """
        name = getattr(exc, 'name', None)
        message = str(exc)
        if name:
            self._mark_refused(name, message)
        notice = getattr(self, 'refusal_notice', None)
        if notice is not None:
            try:
                notice.configure(text=message)
            except Exception:
                pass

    def _field_enter(self, event=None):
        """Enter in a parameter field: apply what is typed and run again."""""
        # **The widget that received the key, not the focused one.** `focus_get()`
        # answers None whenever the window itself has not taken focus (which is the
        # normal state in a harness, and can happen on screen too), and the first
        # version of this refused to do anything in exactly that case.
        widget = getattr(event, 'widget', None)
        if widget is None:
            widget = self._a_field()
        if widget is None:
            return None
        # The value is already in the widget; `request_preview` reads the fields, and a
        # changed setting clears the bar (see `request_preview`).
        self._loud = True
        try:
            self.request_preview()
        finally:
            self._loud = False
        return 'break'

    def _a_field(self):
        """The focused widget, if it is one of the text fields."""
        import tkinter.ttk as ttk

        try:
            widget = self.root.focus_get()
        except Exception:
            return None
        if isinstance(widget, (ttk.Combobox, ttk.Entry)):
            return widget
        try:
            import tkinter as tk

            if isinstance(widget, tk.Entry):
                return widget
        except Exception:
            pass
        return None

    @staticmethod
    def _field_value(widget):
        try:
            return str(widget.get())
        except Exception:
            return None

    @staticmethod
    def _field_put(widget, value) -> None:
        import tkinter.ttk as ttk

        if isinstance(widget, ttk.Combobox):
            widget.set(value)
            return
        widget.delete(0, 'end')
        widget.insert(0, value)

    def _field_remember(self, _event=None) -> None:
        widget = self._a_field()
        if widget is None:
            return
        value = self._field_value(widget)
        if value is None:
            return
        history = getattr(widget, '_xp_history', None)
        if history is None:
            history = widget._xp_history = [value]
            widget._xp_at = 1
            return
        if widget._xp_at < len(history) and history[widget._xp_at - 1] == value:
            return
        if history and history[widget._xp_at - 1] == value:
            return
        del history[widget._xp_at:]
        history.append(value)
        widget._xp_at = len(history)

    def _field_undo(self, _event=None):
        widget = self._a_field()
        history = getattr(widget, '_xp_history', None) if widget is not None else None
        if not history or widget._xp_at <= 1:
            return None
        widget._xp_at -= 1
        self._field_put(widget, history[widget._xp_at - 1])
        return 'break'

    def _field_redo(self, _event=None):
        widget = self._a_field()
        history = getattr(widget, '_xp_history', None) if widget is not None else None
        if not history or widget._xp_at >= len(history):
            return None
        widget._xp_at += 1
        self._field_put(widget, history[widget._xp_at - 1])
        return 'break'

    def _field_select_all(self, _event=None):
        widget = self._a_field()
        if widget is None:
            return None
        try:
            if hasattr(widget, 'select_range'):
                widget.select_range(0, 'end')
                widget.icursor('end')
                return 'break'
            # A combobox's own entry: select what it holds.
            widget.selection_range(0, 'end')
            return 'break'
        except Exception:
            return None

    # ---- keyboard focus in the settings panel ---------------------------

    def parameter_ring(self):
        """The editable parameters of the settings panel, in panel order.

        `self.widgets` is `{name: (caption, widget)}` filled while the panel
        is built, so its order is the order the user sees. Skipped here: anything
        destroyed, unmapped or disabled -- a focus ring that stops on a greyed
        box is a ring that wastes a keystroke.
        """
        ring = []
        for name, pair in self.widgets.items():
            caption, widget = pair
            try:
                if not widget.winfo_exists() or not widget.winfo_ismapped():
                    continue
                if 'disabled' in str(widget.cget('state')):
                    continue
            except Exception:
                continue
            ring.append((name, caption, widget))
        return ring

    def focus_parameter(self, delta: int) -> str:
        """Move the keyboard focus along the parameters. Changes no value.

        `Alt+A` / `Alt+D`. The rule the user set and the reason it is worth setting:
        **a navigation key must not edit.** These move the real Tk focus and
        nothing else -- they do not step a number, cycle a preset, or start a
        conversion. What happens to the value after that is whatever the user types.

        From the preview (or anywhere outside the panel) the first press
        enters the panel, landing on the parameter the user was last on *for this
        stage*, or the first one if the user has not been in it yet. Inside the
        panel it walks, and wraps.
        """
        ring = self.parameter_ring()
        if not ring:
            return 'break'
        names = [name for name, _caption, _widget in ring]
        try:
            focused = self.root.focus_get()
        except Exception:
            focused = None
        at = next((index for index, (_n, _c, widget) in enumerate(ring)
                   if widget is focused), None)
        if at is None:
            remembered = self._focus_by_stage.get(self.stage.get())
            at = names.index(remembered) if remembered in names else (
                0 if delta > 0 else len(ring) - 1)
        else:
            at = (at + delta) % len(ring)
        name, caption, widget = ring[at]
        self._highlight_parameter(name)
        try:
            widget.focus_set()
            # Ready to be typed over, which is what the user asked the key for.
            # Selecting is not editing: the value is unchanged until the user types.
            widget.selection_range(0, 'end')
        except Exception:
            pass
        self._focus_by_stage[self.stage.get()] = name
        return 'break'

    # A text widget takes the letter; anywhere else it is a command.
    PLAIN_KEYS = {'w': ('stage', -1), 's': ('stage', 1),
                  'a': ('parameter', -1), 'd': ('parameter', 1)}
    # Windows' Alt bit, and X11's Mod1, because Tk reports them differently
    # and a window should not care which one it is running under.
    ALT_HELD = 0x20000 | 0x0008
    # Control, so a plain-key command never fires for a Ctrl chord. Lower-
    # casing the keysym made `Ctrl+S` match the plain `s` and step the stage
    # as well as convert -- measured with a real key press, `Original ->
    # Probe` on a Ctrl+S that should only have saved.
    CTRL_HELD = 0x0004

    def _alt_command(self, what, delta):
        """Alt+W/S/A/D, wherever the focus is.

        **With Alt held it is a command even inside a field.** Plain W/S/A/D
        have to stand aside while the user is typing a number -- the letter belongs
        to the field -- but Alt+W never means a letter, so the guard that
        makes the plain keys safe must not also disarm the Alt ones. It
        returns `break` so the field does not also receive it.
        """
        if what == 'stage':
            self.step_stage(delta)
        else:
            self.focus_parameter(delta)
        return 'break'

    def _a_plain_key(self, event):
        """The safety net for Alt+W/S/A/D, and nothing else.

        **W/S/A/D with nothing held are no longer commands.** They were added
        while Alt appeared not to arrive at all; the real cause was that only
        the lower-case `<Alt-w>` was bound while Windows sends `W`, and that
        is fixed. Keeping both would be two hotkeys for one action, which is
        the thing the user asked to be rid of -- and a bare letter that steers the
        window is a letter that cannot be typed anywhere outside a field.

        What is left is the net: if Alt is held and the explicit binding did
        not match the keysym this keyboard produced, the action still runs.
        """
        state = getattr(event, 'state', 0)
        if state & self.CTRL_HELD or not state & self.ALT_HELD:
            # Not an Alt chord: a letter is a letter, and a Ctrl chord
            # belongs to the Ctrl bindings.
            return None
        action = self.PLAIN_KEYS.get((event.keysym or '').lower())
        if action is not None:
            return self._alt_command(*action)
        return None

    def _highlight_parameter(self, name) -> None:
        """Show which parameter has the focus, and un-show the last one."""
        from xp.render.appearance import PAPER
        previous = getattr(self, '_focused_parameter', None)
        if previous and previous in self.widgets and previous != name:
            try:
                self.widgets[previous][0].configure(style='Lab.TLabel')
            except Exception:
                pass
        self._focused_parameter = name
        try:
            self.widgets[name][0].configure(foreground=PAPER)
        except Exception:
            pass

    def step_stage(self, delta: int) -> None:
        """The next or previous stage, through the one path every chooser uses.

        `Ctrl+Down` / `Ctrl+Up`. The converter had no keys at all -- every
        command was a button or a field -- so walking the stages meant leaving
        the picture to click a dropdown, which is the wrong place to look while
        you are judging a trace.
        """
        stages = self.available_stages()
        if not stages:
            return
        current = self.stage.get()
        index = stages.index(current) if current in stages else 0
        target = max(0, min(len(stages) - 1, index + delta))
        if stages[target] == current:
            return                  # already at the end; nothing to re-aim
        self.stage.set(stages[target])
        self._stage_chosen()

    def step_page(self, delta: int) -> None:
        """`PageDown` / `PageUp`: the next or previous page being previewed.

        Steps through the pages the user asked for and not the document's numbering,
        so a run limited to pages 3, 7 and 9 walks those three and never lands
        on one that is not being converted.
        """
        pages = (list(self.only_pages) if self.only_pages
                 else list(range(1, self.page_count + 1)))
        try:
            current = int(self.page.get())
        except (TypeError, ValueError):
            current = pages[0]
        index = pages.index(current) if current in pages else 0
        target = pages[max(0, min(len(pages) - 1, index + delta))]
        if target == current:
            return
        self.page.set(target)
        # **Say so at once when the page has to be rendered.** The user's report: a
        # PageUp/PageDown sometimes had no sign that anything was happening.
        # The status line is only rewritten when the first stage picture
        # arrives, which on an 800-dpi page is seconds away, so until then
        # the window still said whatever it said about the page the user had left.
        #
        # And only when there is really something to wait for: `_render`
        # keeps the last page, so stepping back onto it costs nothing and an
        # invented "loading..." for an instant answer is the other half of
        # the same lie. The cache key is the one `_render` builds.
        ready = self._page_is_ready(target)
        self.request_preview()
        # **After, not before.** `request_preview` writes its own line --
        # "Recomputing with the settings you just changed" -- which is not
        # what happened: the user turned a page. Saying it afterwards replaces a
        # sentence that was both wrong and the reason the turn looked silent.
        self.status.set('Page %d' % target if ready
                        else 'Page %d: rendering...' % target)

    def _page_is_ready(self, page: int) -> bool:
        """Is this page already rendered and measured?"""
        try:
            profile = self._read_profile()
        except Exception:
            return False
        try:
            key = (str(Path(self.path).resolve()),
                   os.path.getmtime(self.path), page - 1, profile.dpi)
        except Exception:
            return False
        with _page_cache_lock:
            return key in _page_cache

    def _refresh_stage_list(self):
        """**Only write the list when it changed.** Assigning `values` on a ttk Combobox moves the
        focus through it, and this is called every time a stage picture arrives -- so during a live
        run the focus was pulled out of the canvas once a second, one full round in and out. Measured
        in the user's key log: 28 of 34 lines were focus events, one round a second through this combobox.
        A focus that moves by itself loses whatever the user was pressing -- Alt chords cannot be completed
        across two focus states -- and it also interrupts typing in a parameter field.
        """
        stages = self.available_stages()
        if hasattr(self, 'stage_box'):
            if tuple(self.stage_box['values']) != tuple(stages):
                self.stage_box['values'] = stages
                # Counted so a test can prove the list is not rewritten when nothing changed --
                # `configure` cannot be spied on from Python, because ttk's `__setitem__` does not
                # go through it.
                self._stage_list_writes = getattr(self, '_stage_list_writes', 0) + 1
        # <~ this used to move the user's selection to the last stage whenever the
        #    one the user had chosen was not among those that arrived. That is the
        #    program overruling the user, which is the one thing the comment above
        #    it promised not to do. The user's choice stands; if its picture is not
        #    there yet the pane says so and the run is aimed at producing it. ~>

    def _read_profile(self):
        base = copy.deepcopy(self.profile)
        base.trace = self.trace.get()
        if self.palette.get() != base.mode:
            other = Profile.named(self.palette.get())
            base.mode, base.palette_bgr, base.hsv_ranges = other.mode, other.palette_bgr, other.hsv_ranges
        return profile_from_fields(base, {name: value.get() for name, value in self.values.items()})

    def _fill(self, profile):
        self.profile = profile
        self.palette.set(profile.mode)
        self.trace.set(profile.trace)
        for name, variable in self.values.items():
            variable.set(as_he_typed_it(getattr(profile, name)))
        self._grey_out(profile)

    def _grey_out(self, profile=None):
        """A setting that cannot change the result right now is shown as
        unavailable rather than left looking live.

        Three ways it can be dead: it belongs to a tracing mode that is not
        selected, to the colour path that is switched off, or to a stage
        *after* the one the user is aiming at -- which cannot change what the user is
        looking at, so aiming at stage 5 greys out everything from stage 6
        on. Leaving a dead setting editable invites an evening of turning a
        knob that is not connected to anything -- `color_break` was exactly
        that for a day.
        """
        if profile is None:
            try:
                profile = self._read_profile()
            except (ValueError, tk.TclError):
                return
        aim = self.stage.get() if self.mode.get() == LIVE else None
        for name, (caption, widget) in getattr(self, 'widgets', {}).items():
            # **Every applicable field stays editable.** It used to be disabled
            # when its stage came *after* the stage the user was looking at, and that
            # is exactly what the user asked to stop: "Alt+A/D must reach all the
            # parameter boxes", and a value typed there plus Enter is a run to
            # the last stage. A field that cannot change the stage on screen is
            # still marked -- muted caption -- so the hint survives without the
            # locked knob.
            usable = field_applies(profile, name, None)
            widget.configure(state='normal' if usable else 'disabled')
            matters = usable and reads_before(name, aim)
            caption.configure(style='Lab.TLabel' if matters else 'Muted.Lab.TLabel')

    def _edit(self, _event=None):
        # Invalidate in-flight results immediately, before the debounce fires.
        self._generation += 1
        self._pending = None
        if self._debounce:
            self.root.after_cancel(self._debounce)
        self.convert_button.configure(state='disabled')
        self._debounce = self.root.after(450, self.request_preview)

    def request_preview(self, suggest=False, tune=False, upto=None):
        if self._closed:
            return
        if self._converting:
            # Out loud. A refusal the user cannot see is indistinguishable from a
            # thing that is broken, and that is what it looked like.
            self.status.set('Converting the document -- the stage view comes '
                            'back when it finishes, or when you cancel it.')
            return
        if self._tuning is not None:
            self._tuning.set()  # a newer request (an edit, another page) supersedes the search
        if getattr(self, '_running', None) is not None:
            # Stop the run in flight rather than leaving it to finish into a
            # result the generation counter will throw away. The user's words: shut
            # down the old calculation and aim at the stage I choose.
            self._running.set()
        if self._debounce:
            self.root.after_cancel(self._debounce)
            self._debounce = None
        self._generation += 1
        self.convert_button.configure(state='disabled')
        try:
            profile = self._read_profile()
            page = self.page.get() - 1
            if not 0 <= page < self.page_count:
                raise ValueError(f'Choose a page from 1 to {self.page_count}.')
            # **A DPI this page cannot hold is corrected here, out loud.**
            # It cannot be left to the render: `page_geometry` is handed the
            # same DPI, so a quiet clamp downstream would put the ink in the
            # wrong place. The field is set to what is being used, so what the user
            # sees and what ran are the same number.
            # **Refused with the real ceiling, not silently clamped.** The user typed a value, the
            # panel rewrote it to 997 and the status line said something else; the user's words: "actually
            # it's 997 why lie". The ceiling for this page is computed, shown, and the run does
            # not start with a number the user did not choose.
            ceiling = getattr(self, 'dpi_ceiling', None)
            if ceiling and profile.dpi > ceiling:
                # The numbers that exist: the ceiling, and what the user asked for. Naming a page
                # width here meant `self.page_width`, which the window does not have -- the
                # raise raised AttributeError instead of a refusal, which is how a "lie" in a
                # message became silence.
                raise FieldError(
                    'dpi', 'Resolution (DPI)',
                    'Resolution (DPI): the largest page of this document can take at most '
                    '%d dpi. %d was asked for.' % (ceiling, profile.dpi))
            self._dpi_note = None
        except (ValueError, tk.TclError) as exc:
            self._pending = None
            if isinstance(exc, FieldError):
                # **Out loud on Enter, quiet while typing.** The debounce fires on every
                # keystroke, so a popup there would be a popup per character; a red caption is
                # enough while the user is still typing.
                if getattr(self, '_loud', False):
                    self._refuse_loudly(exc)
                else:
                    self._mark_refused(exc.name, str(exc))
            else:
                self.status.set(str(exc))
            return
        # A different page is a different set of timings. Within one page the
        # measurements are kept across runs, so stepping through the stages
        # builds up the whole bar instead of showing one piece at a time.
        # **Changed settings clear the bar too.** Within one page the
        # measurements are kept across runs so stepping through the stages builds
        # the bar up -- but when the parameters change, every measurement in the
        # bar belongs to a run the user has just replaced. The user's words (2026-10-27): "if I
        # change the parameter ... the time bar and the load should be reloaded,
        # and everything should be reloaded with the parameter I selected."
        self._clear_refused()
        settings = repr(sorted((name, repr(value))
                                for name, value in vars(profile).items()))
        fresh = False
        if (getattr(self, '_clocked_page', None) != page
                or getattr(self, '_clocked_settings', settings) != settings):
            self._clocked_page = page
            self._clocked_settings = settings
            self.clock.reset()
            # **A parameter change is a fresh start, not a patch.** The user's words: "it
            # should recompute just exactly like when I first enter this GUI, all
            # reset". The bar was cleared and nothing else was: the old picture stayed
            # in the panes until the new one arrived (ten seconds of a stage that
            # reports nothing), and there was no sign anywhere that the change had been
            # taken. So the pictures go, the panes are redrawn blank with what is being
            # waited for, and the status says it out loud.
            self.preview = None
            self._live = {}
            self._details = {}
            self._pictures_cleared_for = (page, settings)
            fresh = True
            try:
                self._draw()
            except Exception:
                pass                # a window on its way out must not fail here
        if tune:
            # A search is never aimed. It is choosing settings for the whole
            # pipeline and its answer is the finished tracing, so stopping at
            # the stage the user happens to be looking at would be stopping it
            # before it has said anything.
            upto = None
        elif upto is None and self.mode.get() == LIVE:
            # In the live view the stage the user is on *is* the target; the other
            # modes want the finished tracing, which is the last stage.
            upto = self.stage.get()
        if upto == STAGES[-1]:
            upto = None                      # the last stage is the whole run
        request = (self._generation, page, profile, suggest, tune, upto)
        if self._working:
            self._pending = request
        else:
            self._start_preview(request)
        # **After** the start, because `_start_preview` writes its own line ("Aiming at
        # ..."): setting this before was overwritten immediately, and the change looked
        # untaken.
        if fresh:
            self.status.set('Recomputing with the settings you just changed...')

    def _with_dpi_note(self, text):
        """`text`, with the clamped-resolution sentence in front of it.

        The note has to survive the run it caused: it is set when the fields
        are read, and the status line is rewritten several times after that,
        so saying it once at the moment of clamping would show it for a
        fraction of a second and then lose it. It stays until the user puts a
        resolution in the field that the page can actually take.
        """
        note = getattr(self, '_dpi_note', None)
        return '%s %s' % (note, text) if note else text

    def _start_preview(self, request):
        self._working = True
        generation, page, profile, suggest, tune, upto = request
        # One event for both jobs it has to do: stop a search, and stop a run
        # that has been re-aimed. Auto-tune keeps its own name for it because
        # that is the one the cancel button reaches for.
        cancel = threading.Event()
        self._running = cancel
        self._tuning = cancel if tune else None
        self.status.set(f'Auto-tuning page {page + 1}...' if tune
                        else (f'Aiming at {upto}...' if upto else 'Preparing preview...'))
        # **The opening gap.** The user's report, 2026-10-29: *"I didn't actually see
        # it. The loading of the conversion is okay, just add a pre-loading
        # stage as there is a gap."* Measured: the window's constructor
        # returns in 0.59 s and the first preview then freezes the main thread
        # for **1.84 s** with nothing on screen but an empty picture area --
        # the status line is set, and the freeze starts before Tk draws it.
        # So the key the user presses in that second lands on a window that looks
        # dead, which is the gap the user is pointing at.
        #
        # The panel the user already has is put up for the opening too, and painted
        # before the work starts. It comes down on the first picture; a later
        # run has something on screen already and does not need it.
        if not self._converting and self.preview is None and not self._live:
            try:
                self.loading.show('Opening the document',
                                  detail='measuring page %d at %d dpi -- the '
                                         'first look is the slow one'
                                         % (page + 1, profile.dpi))
                self.root.update_idletasks()
            except Exception:
                pass

        def progress(done, total):
            self._results.put(('tuning', generation, (page, done, total)))

        def stage_ready(name, picture):
            self._results.put(('stage', generation, (name, picture)))

        def stage_detail(name, payload, size, run_profile):
            # One object per arrival, carrying references to the arrays the
            # worker already has. It is not copied; it is kept alive so the
            # pane can cut a native-resolution window out of it on demand.
            self._results.put(('detail', generation,
                               (name, StageDetail(name, payload, size, run_profile))))

        def worker():
            try:
                result = preview_page(self.path, page, profile, suggest, tune,
                                      progress if tune else None, cancel,
                                      on_image=stage_ready, upto=upto,
                                      clock=self.clock, on_detail=stage_detail)
                self._results.put(('preview', generation, result))
            except Reached:
                # The stage the user was aiming at exists and the user already has its
                # picture; there is no Preview to carry back, because a
                # partial run has no strokes to count and no tracing to
                # compare against the page.
                self._results.put(('aimed', generation, upto))
            except (TuneCancelled, orthofill.Cancelled):
                # Either boundary: a stage that noticed, or the fill loop
                # between two levels. The user stopped it; that is not an error.
                self._results.put(('superseded', generation, None))
            except Exception as exc:
                self._results.put(('error', generation, str(exc)))
            finally:
                # However this run ended -- finished, aimed at a stage,
                # cancelled or broken -- the bar stops growing a piece on the
                # end. There is no other way for a timeline to tell "the next
                # stage has not reported yet" from "there is no next stage".
                self.clock.finish()
        threading.Thread(target=worker, daemon=True).start()

    def _poll(self):
        if self._closed:
            return
        try:
            while True:
                kind, generation, result = self._results.get_nowait()
                if kind == 'step':
                    # What it is doing *now*, not what it started with. The
                    # panel used to say "rendering page 1" for the whole of a
                    # page, and rendering is the first two seconds of twelve.
                    try:
                        self.loading.detail.configure(text=result)
                    except Exception:
                        pass
                    if (time.monotonic()
                            - getattr(self, '_answered_at', 0.0)) > 1.5:
                        self.status.set(self._with_dpi_note(result))
                    continue
                if kind == 'progress':
                    self.status.set(f'Converting page {result[0]} / {result[1]}...')
                    try:
                        self.loading.step(result[0], result[1],
                                          'page %d of %d' % (result[0], result[1]))
                    except Exception:
                        pass
                    continue
                if kind == 'detail':
                    # The full-resolution arrays behind the stage the user is on.
                    # Kept only for this run: a 6,000-dpi raster is gigabytes.
                    if generation == self._generation:
                        name, detail = result
                        self._details[name] = detail
                        # Never hold a stage from a page the user has left, and never
                        # more than the original plus the one the user is looking at.
                        for held in list(self._details):
                            if held != 'Original' and held != self.stage.get():
                                self._details.pop(held, None)
                        if name == 'Original' or name == self.stage.get():
                            self._draw()
                    continue
                if kind == 'stage':
                    # A stage arrives the moment it is computed. Stage one is
                    # up in about a second while the fill loop is still
                    # running, which is what makes this a live view of the
                    # pipeline rather than a replay of a finished job.
                    if generation == self._generation:
                        name, picture = result
                        self._live[name] = picture
                        self._refresh_stage_list()
                        # The user's choice of stage is the user's. A picture arriving
                        # redraws the pane only when it is the one the user is
                        # looking at; the status line says where the page has
                        # got to so the wait is not a blank one.
                        if not self._converting:
                            # The first picture: the opening panel has done
                            # its job and the view can be seen behind it.
                            try:
                                self.loading.hide()
                            except Exception:
                                pass
                        if self.stage.get() == name:
                            self._draw()
                        self.status.set(self._with_dpi_note(
                            f'Page {self.page.get()}: {name}...'))
                    continue
                if kind == 'aimed':
                    # The worker has finished either way, so the slot is
                    # released whether or not this is the generation the user is
                    # waiting for. Releasing it only for the current one left
                    # `_working` set for good if the newer request had failed
                    # its own validation, and every later request then parked
                    # in `_pending` behind a run that had already ended.
                    self._working = False
                    self._running = None
                    if generation == self._generation:
                        self.status.set(self._with_dpi_note(
                            f'Page {self.page.get()}: {result}. '
                            'Stopped there -- that is the stage you are on.'))
                        # Converting does not need a preview: it runs the
                        # whole pipeline itself. Leaving the button disabled
                        # until the user had aimed at the last stage would make an
                        # aimed run feel like a broken one.
                        self.convert_button.configure(state='normal')
                        self._grey_out()
                        self._draw()
                    if self._pending is not None:
                        request, self._pending = self._pending, None
                        self._start_preview(request)
                    continue
                if kind == 'superseded':
                    # A run the user stopped, by choosing a stage or editing a
                    # setting. Not a failure: it did what it was told.
                    self._working = False
                    self._running = None
                    self._tuning = None
                    if self._pending is not None:
                        request, self._pending = self._pending, None
                        self._start_preview(request)
                    continue
                if kind == 'tuning':
                    if generation == self._generation:
                        page, done, total = result
                        self.status.set(f'Auto-tuning page {page + 1}: {done} / {total} threshold settings tried...')
                    continue
                if kind in ('converted', 'cancelled', 'conversion_error'):
                    self._converting = False
                    # However it ended -- saved, cancelled, or failed. A
                    # loading panel left on the screen is worse than none.
                    try:
                        self.loading.hide()
                    except Exception:
                        pass
                    self._set_controls(True)
                    self.cancel_button.configure(state='disabled')
                    self.convert_button.configure(state='normal')
                    if kind == 'converted':
                        path = Path(result)
                        if self._staged is not None and path == Path(self._staged):
                            # A staging file: the conversion is finished and
                            # the question has not been asked yet.
                            self._offer_to_save(path)
                        else:
                            self.status.set(f'Saved {path}')
                            self._finish_after_saving(path)
                    elif kind == 'cancelled':
                        self._discard_staged()
                        self.status.set('Cancelled. No output file was changed.')
                    else:
                        self._discard_staged()
                        self.status.set(f'Conversion failed: {result}')
                        messagebox.showerror('Could not convert PDF', result, parent=self.root)
                    continue
                self._working = False
                self._tuning = None
                if generation == self._generation:
                    if kind == 'preview':
                        self.preview = result
                        self._live = {}
                        for held in list(self._details):
                            if held != 'Original' and held != self.stage.get():
                                self._details.pop(held, None)
                        self._refresh_stage_list()
                        self._fill(result.profile)
                        self.lost_text.set(f'{LOST_LABEL}: {result.missing_share:.1%} '
                                           f'({result.marks_lost:,} whole marks)')
                        self.extra_text.set(f'{EXTRA_LABEL}: {result.extra_share:.1%} '
                                            f'({result.marks_extra:,} specks)')
                        self._draw()
                        if result.tune is not None:
                            self.status.set(f'Page {self.page.get()}: {result.tune.summary()} Ready to convert.')
                        else:
                            self.status.set(f'{result.strokes:,} strokes / page {self.page.get()} preview. '
                                            f'Color off on {result.color_wrong:.1%} of the ink. Ready to convert.')
                        self.convert_button.configure(state='normal')
                    else:
                        self.status.set(f'Preview failed: {result}')
                if self._pending:
                    request, self._pending = self._pending, None
                    self._start_preview(request)
        except queue.Empty:
            pass
        self._poll_id = self.root.after(60, self._poll)

    def _wrap_the_status(self, event=None) -> None:
        """Wrap the status at the window's own width, not at a fixed 560."""
        try:
            width = self.root.winfo_width()
        except Exception:
            return
        wanted = max(320, width - 120)
        if getattr(self.status_label, '_xp_wrap', None) != wanted:
            self.status_label._xp_wrap = wanted
            self.status_label.configure(wraplength=wanted)

    def _box_size(self):
        """Room each pane's image gets on the canvas."""
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        return max(1, width // 2 - 20), max(1, height - 62)

    def _half_extents(self):
        """Half the visible window, per axis, in normalized page units. At Fit
        that's the whole page; zoomed in, the window takes the pane's shape
        rather than the page's, so a tall page fills the pane instead of
        showing as a narrow strip down its middle."""
        zoom = max(MIN_ZOOM, self.zoom)
        # Guarded on the picture, which is what the next line needs. Guarding
        # on `_pane_rects` looks equivalent and is not: `_draw` empties it
        # before it asks for the view, so this fell back to a square window
        # on every draw and every pane came out short. Guarding on
        # `self.preview` -- which is what it used to say -- had the same
        # fault waiting for the first run with no Preview to produce.
        source = self._picture('Original')
        if source is None:
            return 0.5 / zoom, 0.5 / zoom
        iw, ih = source.size
        box_w, box_h = self._box_size()
        fit = min(box_w / iw, box_h / ih)
        return (min(0.5, box_w / (fit * zoom * iw) / 2),
                min(0.5, box_h / (fit * zoom * ih) / 2))

    def _view_rect(self):
        """Normalized (0..1) page window both panes crop from -- one window
        for both, so zooming or panning either pane moves the other with it."""
        hw, hh = self._half_extents()
        cx, cy = self.pan
        cx = min(1 - hw, max(hw, cx))
        cy = min(1 - hh, max(hh, cy))
        return cx - hw, cy - hh, cx + hw, cy + hh

    def _clamp_pan(self):
        # Keep the centre where the window still fits on the page, so a drag
        # past the edge doesn't leave a dead zone to drag back through.
        hw, hh = self._half_extents()
        x, y = self.pan
        self.pan = (min(1 - hw, max(hw, x)), min(1 - hh, max(hh, y)))

    @staticmethod
    def _crop_box(size, rect):
        iw, ih = size
        x0, y0, x1, y1 = rect
        px0 = min(iw - 1, max(0, round(x0 * iw)))
        py0 = min(ih - 1, max(0, round(y0 * ih)))
        px1 = max(px0 + 1, min(iw, round(x1 * iw)))
        py1 = max(py0 + 1, min(ih, round(y1 * ih)))
        return px0, py0, px1, py1

    def _pane_rect(self, name, view):
        """The shared view in this pane's own image. The output page may be
        letterboxed around the source content (fit_geometry), so for the
        traced render the source-page window is mapped through that frame --
        otherwise the two panes would drift apart on any PDF whose page size
        differs from the profile's."""
        if name != 'Traced result':
            return view
        ox, oy, sx, sy = (self.preview.traced_frame if self.preview is not None
                          else (0.0, 0.0, 1.0, 1.0))
        x0, y0, x1, y1 = view
        return ox + x0 * sx, oy + y0 * sy, ox + x1 * sx, oy + y1 * sy

    def _paint_difference(self, image, mask, view, color):
        """Recolor `mask` (source grid) onto a displayed pane, merge-view
        style: the differing ink itself in solid color."""
        if mask is None:
            return image
        mx0, my0, mx1, my1 = self._crop_box((mask.shape[1], mask.shape[0]), view)
        crop = np.ascontiguousarray(mask[my0:my1, mx0:mx1])
        if not crop.any():
            return image
        size = image.size
        pixels = np.array(image)
        if crop.shape[1] > size[0]:
            # Shown smaller than the mask grid (Fit or a low zoom): a lone lost
            # dot could be a single pixel here, so ring it to make it findable.
            # Only while there are few: many rings merge and hide the page.
            core = coverage(crop, size)
            spots = cv2.connectedComponents(core.view(np.uint8), connectivity=8)[0] - 1
            if spots <= HALO_MAX_SPOTS:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * HALO_PX + 1,) * 2)
                halo = (cv2.dilate(core.view(np.uint8), kernel) > 0) & ~core
                tint = np.array(color, dtype=np.float32)
                pixels[halo] = (pixels[halo] * (1 - HALO_ALPHA) + tint * HALO_ALPHA).astype(np.uint8)
        else:
            core = cv2.resize(crop.view(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        pixels[core] = color
        return Image.fromarray(pixels)

    def _draw(self):
        # Not `if not self.preview`: a page being traced has pictures before
        # it has a result, and those are the whole point of a live view.
        if self._closed or (self.preview is None and not getattr(self, '_live', None)):
            return
        self.canvas.delete('all')
        self._photos = []
        self._pane_rects = []
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 50 or height < 50:
            return
        view = self._view_rect()
        box_w, box_h = self._box_size()
        show_diff = self.show_diff.get() and self.preview is not None
        waiting = None
        shown_dpi = []
        for index, name in enumerate(('Original', self.stage.get())):
            pending = False
            image = None
            source_size = None
            box = None
            current_dpi = None
            run_profile = self.preview.profile if self.preview is not None else self.profile
            raster_dpi = float(getattr(run_profile, 'dpi', self.profile.dpi))

            # **A vector page is drawn where it is looked at.** The stored
            # 'Traced result' picture is capped at 24 px per page unit; at a
            # 6,000-dpi setting and 8x zoom that is a thumbnail again. The
            # finished page is in hand, so the visible rectangle is rendered
            # from its strokes at this zoom's own scale.
            page = getattr(self.preview, 'page', None) if self.preview is not None else None
            if name == 'Traced result' and page is not None:
                page_w, page_h = float(page.width), float(page.height)
                out_view = self._pane_rect(name, view)
                span_w = max(1e-6, (out_view[2] - out_view[0]) * page_w)
                span_h = max(1e-6, (out_view[3] - out_view[1]) * page_h)
                scale = min(box_w / span_w, box_h / span_h)
                page_box = (out_view[0] * page_w, out_view[1] * page_h,
                            out_view[2] * page_w, out_view[3] * page_h)
                image = render.render_page_view(page, scale, page_box, smooth=2)
                source_size = (max(1, round(page_w * scale)),
                               max(1, round(page_h * scale)))
                box = (max(0, round(page_box[0] * scale)),
                       max(0, round(page_box[1] * scale)),
                       min(source_size[0], round(page_box[2] * scale)),
                       min(source_size[1], round(page_box[3] * scale)))
                current_dpi = 'vector'

            detail = self._details.get(name)
            if detail is not None:
                raster_dpi = detail.dpi
            if image is None and detail is not None:
                # Native detail is worth taking when the window is a
                # magnification of the thumbnail; at Fit the thumbnail is
                # already finer than the screen and cheaper to draw.
                crop_w = (view[2] - view[0]) * detail.size[0]
                crop_h = (view[3] - view[1]) * detail.size[1]
                # A window of at most 32 megapixels is cheap enough to cut
                # natively; past that the thumbnail is already finer than the
                # screen at that zoom and converting the whole page to RGB
                # would cost more than it shows. The cap is on the crop, not
                # on the page: 6,000 dpi at 4x is 32 MP, at 16x it is 2 MP.
                if crop_w * crop_h <= 32_000_000 and crop_w <= 12_000 and crop_h <= 12_000:
                    try:
                        image = detail.crop(view)
                    except Exception:
                        image = None
                    if image is not None:
                        source_size = detail.size
                        box = detail._crop_box(view)
                        current_dpi = '%d dpi' % detail.dpi

            if image is None:
                source = self._picture(name)
                pending = source is None
                if pending:
                    # The stage the user is aiming at has not been produced yet. Draw
                    # the pane blank rather than drop it: a pane that vanishes
                    # makes the layout jump and leaves the user looking at one
                    # picture wondering which it is. The caption says what is
                    # being waited for.
                    other = self._picture('Original')
                    if other is None:
                        continue
                    source = Image.new('RGB', other.size, (248, 248, 248))
                    waiting = name
                box = self._crop_box(source.size, self._pane_rect(name, view))
                image = source.crop(box)
                source_size = source.size
                current_dpi = ('%d dpi (thumbnail)' % detail.thumbnail_dpi
                               if detail is not None else None)

            if current_dpi is not None:
                shown_dpi.append(current_dpi)
            # **The PDF is shown as the PDF is.** The user's correction (2026-10-27):
            # "why you change that pdf, i want you to brush out the blue boundary
            # for xopp not for pdf". Erasing the printed border here was the
            # wrong target: this pane is the file the user gave XP, and what must have
            # no blue border in it is the **XOPP** XP writes -- which the merge
            # view and the exporter already handle. The dashed guide is drawn
            # where the border was recognised, not instead of the border itself.
            # resize, not thumbnail(): thumbnail() only shrinks, so a zoomed
            # crop smaller than the pane would stay small instead of filling it.
            crop_w, crop_h = image.size
            fit = min(box_w / crop_w, box_h / crop_h)
            image = image.resize((max(1, round(crop_w * fit)), max(1, round(crop_h * fit))),
                                 Image.Resampling.LANCZOS)
            if show_diff:
                if index == 0:
                    image = self._paint_difference(image, self.preview.missing, view, MISSING_COLOR)
                else:
                    image = self._paint_difference(image, self.preview.extra, view, EXTRA_COLOR)
            # **The guide is part of the picture now.** The user's order: "that dashed line
            # should be drawn on the live view image". As canvas items beside the image it
            # needed a second mapping to agree with the first, and it did not -- measured,
            # its left edge landed 391 page units outside the page at 1x, 192 at 2x, 98 at
            # 3.8x. Painted here it is made of the same crop and resize as the picture.
            # **The view that was actually used to crop this pane**, not the source page's.
            # For `Traced result` the picture is the *output* page, cropped through
            # `traced_frame` (letterbox), so mapping the guide with the source view put the
            # dashes a letterbox away from the writing area -- the user's picture: the dashes did not
            # sit on the content while the content itself was measured to be within 0.5 pt.
            image = self._paint_the_guide(image, self._pane_rect(name, view),
                                          source_size, box)
            photo = ImageTk.PhotoImage(image, master=self.root)
            self._photos.append(photo)
            x = width * (index + .5) / 2
            self.canvas.create_text(x, 15, text='Source PDF' if index == 0 else name, fill='#b8d0e0', font=appearance.font(11))
            self.canvas.create_image(x, 34, image=photo, anchor='n')
            disp_w, disp_h = image.size
            if pending:
                # Said in the pane, not only in the caption above it: a blank
                # white rectangle looks the same whether the stage is coming
                # or the view has failed.
                self.canvas.create_text(
                    x, 34 + disp_h / 2, text='Loading %s...' % name,
                    fill='#6f8799', font=appearance.font(12))
            self._pane_rects.append((x - disp_w / 2, 34, disp_w, disp_h))
        # **Nothing drawn over the panes.** A dark strip was here to keep this
        # line readable, and it is the "shade on the right side" the user saw when the
        # window opened: a band of background colour across the bottom of both
        # pages. The hint is in the footer row now (see `_build`), where it cannot
        # touch the picture at all.
        # **No dead "Fit" word.** It was a label that read like a control, sat
        # where the other buttons are, and did nothing when clicked -- the user's
        # words: "the useless Fit in the converter, delete it". At the default
        # zoom there is nothing to say, so it says nothing; a real zoom still
        # gets its number.
        self.zoom_label_var.set(f'{self.zoom:.1f}x' if self.zoom > 1.0 else '')
        if shown_dpi:
            self.detail_text.set('Raster %d dpi · view %s'
                                 % (raster_dpi, ' / '.join(shown_dpi)))
        else:
            self.detail_text.set('Raster %d dpi · view thumbnail' % raster_dpi)

    def _paint_the_guide(self, image, view, source_size, crop_box):
        """The dashed grey guide, drawn **on** the pane's picture (see `painted.py`)."""
        from xp.core.document.pagesize import QBASE
        from xp.core.frame import REFERENCE
        from xp.render.widgets import painted

        page = (REFERENCE[0] / QBASE[0], REFERENCE[1] / QBASE[1],
                REFERENCE[2] / QBASE[0], REFERENCE[3] / QBASE[1])
        found = self._the_border_in_the_page()
        box = page if found is None else found
        return painted.paint_the_guide(image, box, view, source_size, crop_box)

    def _draw_qbase_guide(self, rect, view):
        """The Q base writing area inside a pane: one grey dashed line.

        The user's instruction for both GUIs: the printed blue border is never shown,
        and the reference it stands for is a **dashed** line that follows the
        geometry. A converted page has no border in it at all, so this is drawn
        from the page standard, mapped through the part of the page the pane is
        showing -- which is why it moves and scales with zoom and pan.
        """
        from xp.core.document.pagesize import QBASE
        from xp.core.frame import REFERENCE
        from xp.render.widgets.guide import draw_guide, guide_segments

        left, top, drawn_w, drawn_h = rect
        if drawn_w <= 1 or drawn_h <= 1:
            return
        span_x = max(view[2] - view[0], 1e-6)
        span_y = max(view[3] - view[1], 1e-6)

        # **Two coordinate systems meet here, and that was the bug.** The view is
        # in **normalised page units** (0..1 across the page), while `REFERENCE`
        # is in **points** (7..180 across a 187.82-point page). Feeding points to
        # a view that spans 1.0 put the guide 40x off the canvas -- and made
        # `guide_segments` cut 190,000 dashes, once per paint. A span of about a
        # page or less means normalised; anything wider is points.
        normalised = span_x <= 1.5 and span_y <= 1.5
        if normalised:
            page = (REFERENCE[0] / QBASE[0], REFERENCE[1] / QBASE[1],
                    REFERENCE[2] / QBASE[0], REFERENCE[3] / QBASE[1])
        else:
            page = REFERENCE

        def to_pane(page_x, page_y):
            return (left + (page_x - view[0]) / span_x * drawn_w,
                    top + (page_y - view[1]) / span_y * drawn_h)

        def on_canvas(box_in_page):
            x0, y0 = to_pane(box_in_page[0], box_in_page[1])
            x1, y1 = to_pane(box_in_page[2], box_in_page[3])
            return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

        # **The page's own writing area: a solid grey line.** The user's correction
        # (2026-10-27): "not green i said grey". This is the Sheet the user composes on
        # -- it exists whether or not the PDF has a border.
        sheet = on_canvas(page)
        self.canvas.create_rectangle(*sheet, outline=GUIDE_LINE[0],
                                     width=1, dash=(), tags=('qbase-sheet',))

        # **The printed border, where it really is: dashed grey.** The user said the
        # dashed line was drawn "separately, without identifying where the blue
        # arrow is" -- so the border is *found* in the page, and the dashes go on
        # that rectangle. Nominal only when nothing is found.
        # **Where the border ends up, not where it started.** The pipeline maps
        # the page onto the sheet (`convert.fit_geometry`): the ratio the border
        # needs is measured, then applied, so after the conversion the printed
        # border sits on the sheet's rectangle -- measured on the reference,
        # whose ink lands within 1 pt of ours. Drawing the dashes at the *found*
        # position would put them inside the sheet by the very ratio that has
        # already been applied, which is what the user saw as "not aligned".
        # **Where the border actually is.** This line read
        # `page if found is not None else page` -- both branches the same
        # rectangle -- so the dashes were laid on the sheet whatever the
        # detector found, and the printed border was left in the picture in
        # blue. That is what the user photographed at 18:30: "clearly you idndt
        # recoigniz ethis bklue boodundray didnt tuenr it into dashed grey
        # line". `_the_border_in_the_page` answers in normalised page units,
        # so it needs scaling to points when the view is in points.
        found = self._the_border_in_the_page()
        if found is None:
            box = on_canvas(page)         # no border found: the sheet is all there is
        elif normalised:
            box = on_canvas(found)
        else:
            box = on_canvas((found[0] * QBASE[0], found[1] * QBASE[1],
                             found[2] * QBASE[0], found[3] * QBASE[1]))
        # A cap as a guard, not as the answer: a wrong rectangle must never be
        # able to put a hundred thousand items on the canvas again. It is
        # **per pane**, and the panes are painted one after the other, so the
        # canvas carries this many times the number of panes.
        segments = guide_segments(box)
        if len(segments) > GUIDE_DASH_CAP:
            step = max(1, len(segments) // GUIDE_DASH_CAP)
            segments = segments[::step]
        draw_guide(self.canvas, segments)

    def the_ratio_to_the_sheet(self):
        """How the found border has to be scaled to sit on the Q base sheet.

        Returns `(sx, sy)` -- the x and y factors that take the page the user gave XP
        onto the sheet XP composes on -- or None when no border was found (a
        borderless page is already the sheet, nothing to stretch).

        The user's words: "just find out that the boundary compared with Q base... add
        some ratio in the x dimension or y dimension, and just extend, crop
        whatever you want". This is the ratio; applying it to the drawn page is
        the next step, and this method exists so that step can be measured.
        """
        from xp.core.document.pagesize import QBASE
        from xp.core.frame import REFERENCE

        found = self._the_border_in_the_page()
        if found is None:
            return None
        sheet = (REFERENCE[0] / QBASE[0], REFERENCE[1] / QBASE[1],
                 REFERENCE[2] / QBASE[0], REFERENCE[3] / QBASE[1])
        span_x = max(found[2] - found[0], 1e-9)
        span_y = max(found[3] - found[1], 1e-9)
        return ((sheet[2] - sheet[0]) / span_x, (sheet[3] - sheet[1]) / span_y)

    def _the_border_in_the_page(self):
        """The printed blue border, in normalised page coordinates, or None.

        Found once per page image and cached. `centreline_in_image` is the same
        detector the merge view calibrates itself with -- four blue edges, each
        measured on the line's centre -- so the dashed line lands where the
        border **is**, not where the template usually puts it.
        """
        picture = self._picture('Original')
        if picture is None:
            return None
        cache = getattr(self, '_borders', None)
        if cache is None:
            cache = self._borders = {}
        key = id(picture)
        if key in cache:
            return cache[key]
        found = None
        try:
            import numpy as np

            from xp.core.frame import centreline_in_image

            array = np.asarray(picture.convert('RGB'))[:, :, ::-1]   # BGR
            frame = centreline_in_image(array)
            if frame is not None:
                height, width = array.shape[:2]
                found = (frame.x0 / float(width), frame.y0 / float(height),
                         frame.x1 / float(width), frame.y1 / float(height))
        except Exception:
            found = None
        cache[key] = found
        return found

    def _pane_at(self, x, y):
        for index, (left, top, w, h) in enumerate(self._pane_rects):
            if left <= x <= left + w and top <= y <= top + h:
                return index
        return None

    def _display_rect(self, index, hw, hh):
        """Where pane `index`'s image will sit for a window of half-size
        (hw, hh) -- the same fit-and-centre _draw() does."""
        iw, ih = self._picture('Original').size
        box_w, box_h = self._box_size()
        cw, ch = 2 * hw * iw, 2 * hh * ih
        scale = min(box_w / cw, box_h / ch)
        w, h = cw * scale, ch * scale
        return self.canvas.winfo_width() * (index + .5) / 2 - w / 2, 34, w, h

    def _record_the_mouse(self, event):
        """`mouse_callback`: store the pointer, and do no view work here."""
        self._pointer['x'] = event.x
        self._pointer['y'] = event.y

    def _forget_the_mouse(self, _event=None):
        """Pointer off the canvas: hold the last view rather than jumping."""
        self._pointer['x'] = None
        self._pointer['y'] = None

    def _show_follow_state(self):
        if not hasattr(self, 'follow_text'):
            return
        self.follow_text.set('Follow: on  (click to stop)' if self._follow_enabled
                             else 'Follow: off  (click to start)')

    def _follow_loop(self):
        """`display_image`'s loop: adjust from the record, every frame."""
        if self._closed:
            self._follow_job = None
            return
        try:
            self._adjust_to_mouse_center()
        except Exception:
            # A view that cannot follow holds still; it does not take the
            # window down with it.
            pass
        try:
            self._follow_job = self.root.after(FOLLOW_MS, self._follow_loop)
        except Exception:
            self._follow_job = None

    def _adjust_to_mouse_center(self):
        """`adjust_to_mouse_center`, over this window's shared pan.

        The arranger's formula, in normalized page units: the pointer's
        fraction of the pane is the page fraction that goes to the middle of
        that pane. `self.pan` is already that middle, so the adjust is the
        fraction itself, clamped by the same half-extents `_view_rect` uses.
        A still pointer is a still view; a drag owns the view and is not
        fought.
        """
        if not self._follow_enabled or self._pan_drag is not None:
            return
        px, py = self._pointer['x'], self._pointer['y']
        if px is None or py is None or not self._pane_rects:
            return
        index = self._pane_at(px, py)
        if index is None:
            return
        left, top, width, height = self._pane_rects[index]
        if width <= 1 or height <= 1:
            return
        relative_x = min(1.0, max(0.0, (px - left) / float(width)))
        relative_y = min(1.0, max(0.0, (py - top) / float(height)))
        hw, hh = self._half_extents()
        wanted = (min(1 - hw, max(hw, relative_x)),
                  min(1 - hh, max(hh, relative_y)))
        if wanted != self.pan:
            self.pan = wanted
            self._draw()

    def _on_wheel(self, event):
        # `_pane_rects`, not `self.preview`: zooming is about there being
        # pictures on the canvas, not about a tracing having finished. A run
        # aimed at Threshold has no Preview, and looking closely at that
        # stage to see whether the window is wide enough is the whole point
        # of a live view -- the old guard would have refused it.
        if not self._pane_rects:
            return
        factor = 1.25 if event.delta > 0 else 1 / 1.25
        new_zoom = min(MAX_ZOOM, max(MIN_ZOOM, self.zoom * factor))
        if new_zoom == self.zoom:
            return
        index = self._pane_at(event.x, event.y)
        x0, y0, x1, y1 = self._view_rect()
        if index is None:
            px, py = (x0 + x1) / 2, (y0 + y1) / 2
        else:
            left, top, w, h = self._pane_rects[index]
            px = x0 + (event.x - left) / w * (x1 - x0)
            py = y0 + (event.y - top) / h * (y1 - y0)
        self.zoom = new_zoom
        hw, hh = self._half_extents()
        if index is None:
            self.pan = (px, py)
        else:
            # Zoom about the cursor: put the page point that was under it
            # back under it, measured against where the image will sit at the
            # new zoom (going from Fit to zoomed, the image changes shape).
            # **This anchor is momentary by the user's choice (2026-11-02, "A")**:
            # the follow loop runs 30 ms later and re-centres on the pointer's
            # fraction, exactly as the arranger's page view does. Suppressing
            # that frame is the other design, and the user did not choose it.
            left, top, w, h = self._display_rect(index, hw, hh)
            fx, fy = (event.x - left) / w, (event.y - top) / h
            self.pan = (px + (0.5 - fx) * 2 * hw, py + (0.5 - fy) * 2 * hh)
        self._clamp_pan()
        self._draw()

    def _on_pan_start(self, event):
        if not self._pane_rects:
            self._pan_drag = None
            self._drag_end = None
            return
        self._pan_drag = (event.x, event.y, self.pan, self._view_rect())
        self._drag_end = (event.x, event.y)

    def _on_pan_drag(self, event):
        if not self._pan_drag:
            return
        self._drag_end = (event.x, event.y)
        start_x, start_y, (start_cx, start_cy), (x0, y0, x1, y1) = self._pan_drag
        left, top, disp_w, disp_h = self._pane_rects[0]
        dx = (event.x - start_x) / disp_w * (x1 - x0) if disp_w else 0
        dy = (event.y - start_y) / disp_h * (y1 - y0) if disp_h else 0
        self.pan = (start_cx - dx, start_cy - dy)
        self._clamp_pan()
        self._draw()

    def _on_pan_end(self, event=None):
        """A click toggles the mouse follow; a drag only ends the drag.

        The left button is both gestures: press starts a pan, so the release
        has to ask whether the pointer actually moved. `CLICK_SLOP` is in
        pixels -- a press that wandered one or two pixels because a hand is
        not a machine is still a click.
        """
        drag = self._pan_drag
        self._pan_drag = None
        if drag is None:
            self._drag_end = None
            return
        if event is not None:
            end = (event.x, event.y)
        elif self._drag_end is not None:
            end = self._drag_end
        else:
            end = (drag[0], drag[1])
        moved = abs(end[0] - drag[0]) + abs(end[1] - drag[1])
        self._drag_end = None
        if moved <= CLICK_SLOP:
            self._follow_enabled = not self._follow_enabled
            self._show_follow_state()

    def _reset_view(self, _event=None):
        self.zoom = MIN_ZOOM
        self.pan = (0.5, 0.5)
        self._draw()

    def reset_defaults(self):
        """Back to the default -- **the user's**, if the user has saved one.

        I built this the other way round first, reasoning that `reset` had to
        reach the factory numbers or there was no exit from a default the user had
        stopped liking. The user reported it as a bug the same evening: *"like I
        edit the denoise to 0.1, save as default and reset to default, why it
        turn back to 0?"* -- and the user is right. Once the user can save a default,
        **the default is the one the user saved**; a reset that jumps over it to the
        shipped guess is throwing away the thing the button above it just
        kept.

        The factory numbers are still reachable: `<palette>_default.json`
        sits beside the user's file and `Load settings...` opens it, and deleting
        `<palette>_mine.json` goes back to them for good.
        """
        self._fill(Profile.named(self.palette.get()))
        self.request_preview()

    def save_as_default(self):
        """Keep these settings; every window after this one opens on them.

        The user's instruction, 2026-10-28: *"change this into save the parameters
        as the default, so sometimes I can cook it step by step."* Tuning a
        scan is not one sitting -- the user moves a threshold, looks at it, comes
        back later -- and until now every new window threw that away and
        started from the shipped guess again.

        It is written beside the shipped profiles so it travels with the
        project, and it is a plain JSON file the user can read or delete.
        """
        try:
            profile = self._read_profile()
        except ValueError as exc:
            messagebox.showerror('Check settings', str(exc), parent=self.root)
            return
        where = Profile.mine_path(self.palette.get())
        try:
            profile.save(where)
        except OSError as exc:
            messagebox.showerror('Could not save the default', str(exc),
                                 parent=self.root)
            return
        self.profile = profile
        self.status.set('Saved as the default for %s -- every window opens on '
                        'these now (%s)' % (self.palette.get(), where.name))

    def save_profile(self):
        try:
            profile = self._read_profile()
        except ValueError as exc:
            messagebox.showerror('Check settings', str(exc), parent=self.root)
            return
        filename = filedialog.asksaveasfilename(parent=self.root, title='Save conversion settings', defaultextension='.json', filetypes=[('Conversion settings', '*.json')])
        if filename:
            try:
                profile.save(filename)
                self.status.set(f'Settings saved to {filename}')
            except OSError as exc:
                messagebox.showerror('Could not save settings', str(exc), parent=self.root)

    def load_profile(self):
        filename = filedialog.askopenfilename(parent=self.root, filetypes=[('Conversion settings', '*.json')])
        if filename:
            try:
                loaded = Profile.load(filename)
                validated = profile_from_fields(loaded, {name: str(getattr(loaded, name)) for name, *_ in FIELDS})
                if loaded.mode not in ('hand', 'elec'):
                    raise ValueError('Profile mode must be hand or elec.')
                self._fill(validated)
                self.request_preview()
            except Exception as exc:
                messagebox.showerror('Could not load settings', str(exc), parent=self.root)

    def _set_controls(self, enabled):
        for widget in self.controls:
            widget.configure(state=('readonly' if isinstance(widget, ttk.Combobox)
                                    else 'normal') if enabled else 'disabled')
        if enabled:
            # The Next-stage button has a second reason to be off -- it is
            # meaningless outside the live view -- so it is asked rather than
            # simply switched on with the rest.
            self._mode_changed_state()

    def _mode_changed_state(self):
        live = self.mode.get() == LIVE and not self._converting
        self.stage_button.configure(state='normal' if live else 'disabled')

    def default_output(self):
        """Where Ctrl+S writes: `<document>_converted.xopp`, beside it.

        The name the dialog has always offered. Ctrl+S uses it without
        asking, which is what the arranger's Ctrl+S does (`quick_save`) and
        what the merge's does; Ctrl+Shift+S is the one that asks.
        """
        return str(self.path.parent / (self.path.stem + '_converted.xopp'))

    SHIFT_HELD = 0x0001

    def _save_key(self, event=None):
        """Ctrl+S saves, Ctrl+Shift+S asks where -- told apart by `state`.

        The letter's case cannot be used for this. Caps Lock inverts it, so
        `S` means *no* Shift when Caps is on and `s` means Shift is down.
        `state` says what the user's fingers are actually holding.
        """
        shift = bool(getattr(event, 'state', 0) & self.SHIFT_HELD)
        if shift:
            self.convert_as()
        else:
            self.convert()
        return 'break'

    def convert_as(self):
        """Ctrl+Shift+S: convert, then ask where, prefilled."""
        self._ask_where = True
        self.convert()

    def _stop_the_preview(self) -> None:
        """Abandon a preview in flight, the way `request_preview` does.

        The same idiom a newer request uses: signal the run, drop the pending
        debounce, move the generation on so whatever the worker posts next is
        recognised as stale and thrown away.
        """
        if self._tuning is not None:
            self._tuning.set()
        running = getattr(self, '_running', None)
        if running is not None:
            running.set()
        if self._debounce:
            try:
                self.root.after_cancel(self._debounce)
            except Exception:
                pass
            self._debounce = None
        self._pending = None
        self._generation += 1

    def convert(self, target=None):
        """Ctrl+S: convert the document and write it, now.

        **A preview in flight is not a reason to wait.** The user's report,
        2026-10-28: *"why can't I just Ctrl+S or Ctrl+Shift+S when I just
        enter the GUI? I mean sometimes I'm confident with my default
        settings, so I want to convert and save directly."* Opening the
        window starts a preview at once, so the user's first keystroke always landed
        on a busy window.

        What happened then was worse than a refusal: it set
        `_convert_when_idle`, said *"Converting as soon as this preview
        stops..."*, and in live mode **nothing ever consumed that flag**.
        Live stage pictures `continue` out of the pump long before the check,
        and so does the `aimed` result that ends the run. The promise was
        made, the status line was overwritten by the next stage message, and
        no file was written. Measured: press it while the preview runs, wait
        40 s, no output.

        The flag is deleted rather than repaired. A conversion does not need
        the preview -- everything below reads the settings off the panel and
        hands them to `convert_to_file`, which runs the pipeline on the
        document itself -- and it runs on a thread of its own. So the preview
        the user is not waiting for is stopped and the conversion starts. Only a
        conversion already running makes the user wait, because there is one
        output file.
        """
        if self._converting:
            # **Never silent.** A key that reaches here and returns having
            # said nothing is a key that looks broken.
            self.status.set('Already converting -- the file comes when it finishes.')
            # **And it stays long enough to be read.** The step messages
            # rewrite this line every couple of seconds, so without this the user's
            # answer was replaced before it was drawn -- a second press would
            # have looked exactly like a dead key, which is the thing this
            # branch exists to avoid.
            self._answered_at = time.monotonic()
            return
        if self._working or self._pending or self._debounce:
            self._stop_the_preview()
        try:
            profile = self._read_profile()
        except ValueError as exc:
            messagebox.showerror('Check settings', str(exc), parent=self.root)
            return
        if self.settle is not None:
            # The merge's conversion step: it already knows which pages it
            # wants and where the result is going, so there is nothing to
            # ask. What it needs is the settings the user arrived at here.
            settle, self.settle = self.settle, None
            # Not this window's task: it is a step inside the merge's, and
            # reporting it would let the launcher close while the merge was
            # still writing the file.
            self._settled = True
            self.close()
            settle(profile)
            return
        if target is None:
            if (self._staged is not None and Path(self._staged).exists()
                    and self._staged_profile == profile):
                # The user converted once and refused the dialog, so the finished
                # conversion is still here **and it was made with exactly these
                # settings**. Offer it again rather than converting the
                # document a second time.
                self._profile_used = self._staged_profile
                self._offer_to_save(self._staged)
                return
            if self._staged is not None:
                # The settings changed since that conversion. Its output no
                # longer answers the question being asked, so it is thrown away
                # rather than saved under the new parameters.
                self._discard_staged()
            # Convert first, ask after. It used to ask here, before it had
            # computed anything -- so if the user did not like the result, the name
            # had already been chosen for the user. Nothing is chosen now: the
            # conversion goes to a staging file and the name is asked for
            # once there is something to name.
            handle, staged = tempfile.mkstemp(prefix='xp-staged-', suffix='.xopp')
            os.close(handle)
            self._staged = Path(staged)
            self._staged_profile = profile
            target = staged
        self._converting = True
        # The settings this run really used, kept for the learning material:
        # the JSON must describe the conversion that happened, not whatever
        # the sliders say by the time the file is named.
        self._profile_used = profile
        self._cancel.clear()
        self._set_controls(False)
        self.convert_button.configure(state='disabled')
        self.cancel_button.configure(state='normal')
        # **The cue is painted before the work starts, not after.** Rendering
        # page 1 at this profile's dpi takes about two seconds on the user's own
        # page, and MuPDF holds the GIL for most of it -- measured: a 2.1 s
        # render on a worker thread leaves the main thread a single 1.06 s
        # gap, so putting it on a thread does not save the window. The user's
        # report: *"why always stuck when I enter and just Ctrl+S ... frankly
        # I don't see why it will stick for 2-3 s."*
        #
        # Nothing here can make that render quick. What is fixed is the part
        # that was actually wrong: `status.set` only marks the variable, and
        # the freeze began before Tk ever drew it, so the user got a dead window
        # with the previous message still on it. Painting first costs a
        # millisecond and turns a hang into a wait with a reason on it.
        self.status.set(self._with_dpi_note(
            'Converting document... page 1 renders at %d dpi first, which '
            'takes a moment.' % profile.dpi))
        try:
            # The panel first and the paint after it, so what gets drawn in
            # that last millisecond is the thing the user is meant to watch.
            # The first frame already names the cost: page 1 has to be
            # rendered before anything else can happen, and that is the part
            # the user feels. The steps replace this line as they come.
            self.loading.show('Converting the document',
                              detail='opening it -- page 1 renders at %d dpi '
                                     'first' % profile.dpi)
        except Exception:
            pass
        try:
            self.root.update_idletasks()
        except Exception:
            pass
        def worker():
            try:
                result = convert_to_file(
                    self.path, target, profile, self._cancel,
                    lambda n, total: self._results.put(('progress', 0, (n, total))),
                    clock=self.pages_clock,
                    say=lambda text: self._results.put(('step', 0, text)))
                self._results.put(('converted', 0, str(result)))
            except ConversionCancelled:
                self._results.put(('cancelled', 0, None))
            except Exception as exc:
                self._results.put(('conversion_error', 0, str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _discard_staged(self) -> None:
        """Throw away a conversion nobody is going to be offered."""
        if self._staged is None:
            return
        try:
            Path(self._staged).unlink()
        except OSError:
            pass
        self._staged = None
        self._staged_profile = None

    def _offer_to_save(self, staged) -> None:
        """Put the finished conversion where it goes, and finish.

        Reached only when the user did not name a target on the command line.
        **Ctrl+S does not ask**: it writes `<document>_converted.xopp` beside
        the document, the name this dialog always offered, and then the
        window closes -- the conversion is done, and leaving it up means the
        next thing the user does is close it. Ctrl+Shift+S (`convert_as`) is the
        one that asks, prefilled with that same name.

        The order is still the point: the document has already been traced,
        so what the user is asked -- when the user is asked -- is a question about a real
        result rather than about a hope, and refusing is not a loss. The
        staged file stays and the Convert button offers it again instead of
        tracing the document a second time.
        """
        ask, self._ask_where = self._ask_where, False
        if ask:
            target, keep = self.ask_save_target(self.default_output())
            if not target:
                self.status.set('Converted, not saved -- press Convert again '
                                'to put it somewhere.')
                return
            self._keep_learning_material = keep
        else:
            target = self.default_output()
        # **Nothing is ever replaced.** The user's order: "why the save didn't add hyphen and number but
        # replace the already existing one, it should use hyphen number to avoid a replace but
        # just add up". `-1`, `-2`, ... until the name is free.
        target = a_free_name(target)
        try:
            _put_in_place(Path(staged), Path(target))
        except OSError as exc:
            messagebox.showerror('Could not save', str(exc), parent=self.root)
            return
        self._staged = None
        self._staged_profile = None
        self.status.set(f'Saved {target}')
        if getattr(self, '_keep_learning_material', False):
            # After the notebook is on disk, and never before it: the archive
            # is a copy of a result, so a copy that fails is a line in the
            # status bar and not a lost conversion.
            self._archive_then_finish(target)
            return
        self._finish_after_saving(target)

    def ask_save_target(self, suggested):
        """Save As: the name, the folder, and whether to keep the sample.

        One window, one question, the same prefill as before. It replaces the
        system file dialog because that dialog cannot hold a control, and the
        checkbox the user asked for is part of the same question -- a name, a place
        and "keep this one". Everything else about the step is unchanged:
        Ctrl+S still writes the default name without asking, and a cancelled
        dialog still costs nothing.

        Returns `(path or None, keep_as_learning_material)`.
        """
        suggested = Path(suggested)
        answer = {'target': None,
                  'keep': bool(self.learning_material.get())}
        dialog = tk.Toplevel(self.root, name='save-as')
        # Held on the window as well as in the widget tree: it is one window,
        # it is modal, and anyone looking for it (a test included) should not
        # have to guess at its path.
        self._save_dialog = dialog
        dialog.title('Save converted notebook')
        dialog.transient(self.root)
        dialog.resizable(False, False)
        body = ttk.Frame(dialog, padding=14)
        body.pack(fill='both', expand=True)

        folder = tk.StringVar(dialog, str(suggested.parent))
        name = tk.StringVar(dialog, suggested.name)
        keep = tk.BooleanVar(dialog, answer['keep'])
        # Kept on the dialog so a test (or a person reading it later) can see
        # what it is holding without walking the widget tree.
        dialog.folder, dialog.name, dialog.keep = folder, name, keep

        ttk.Label(body, text='Folder').grid(row=0, column=0, sticky='w')
        ttk.Entry(body, textvariable=folder, width=54).grid(
            row=0, column=1, sticky='we', padx=(8, 0))

        def browse():
            chosen = filedialog.askdirectory(parent=dialog,
                                             initialdir=folder.get())
            if chosen:
                folder.set(chosen)

        ttk.Button(body, text='Browse...', command=browse).grid(
            row=0, column=2, padx=(6, 0))
        ttk.Label(body, text='Name').grid(row=1, column=0, sticky='w',
                                          pady=(8, 0))
        ttk.Entry(body, textvariable=name, width=54).grid(
            row=1, column=1, sticky='we', padx=(8, 0), pady=(8, 0))
        ttk.Checkbutton(body, text='Save as Learning Material',
                        variable=keep).grid(row=2, column=0, columnspan=3,
                                            sticky='w', pady=(12, 0))
        ttk.Label(body, foreground='#5a6b7b',
                  text='Keeps this PDF, these settings and the notebook, '
                       'together, in one sample folder.').grid(
            row=3, column=0, columnspan=3, sticky='w')

        problem = tk.StringVar(dialog, '')
        ttk.Label(body, textvariable=problem, foreground='#b03a2e').grid(
            row=4, column=0, columnspan=3, sticky='w', pady=(6, 0))

        def accept():
            chosen = Path(folder.get()) / name.get().strip()
            if not name.get().strip():
                problem.set('Give the notebook a name.')
                return
            if chosen.suffix.lower() != '.xopp':
                chosen = chosen.with_suffix('.xopp')
            if not chosen.parent.is_dir():
                problem.set('That folder is not there: %s' % chosen.parent)
                return
            if chosen.resolve() == self.path.resolve():
                problem.set('That would overwrite the document being converted.')
                return
            answer['target'] = str(chosen)
            answer['keep'] = bool(keep.get())
            dialog.destroy()

        def cancel():
            dialog.destroy()

        buttons = ttk.Frame(body)
        buttons.grid(row=5, column=0, columnspan=3, sticky='e', pady=(14, 0))
        ttk.Button(buttons, text='Cancel', command=cancel).pack(side='right')
        ttk.Button(buttons, text='Save', command=accept).pack(
            side='right', padx=(0, 6))
        dialog.bind('<Return>', lambda _e: accept())
        dialog.bind('<Escape>', lambda _e: cancel())
        # Reachable from the dialog itself, so a test can answer it the way a
        # person does -- and so there is one place `Save` means.
        dialog.accept, dialog.cancel = accept, cancel
        dialog.grab_set()
        dialog.wait_window(dialog)
        self.learning_material.set(answer['keep'])
        return answer['target'], answer['keep']

    def _archive_then_finish(self, target) -> None:
        """Copy the sample on a worker, then close -- the window stays live.

        Two files of a few megabytes, so this is short; it is on a thread
        because "short" is a measurement and not a promise, and the window must
        not stop redrawing for a copy. Whatever happens, the notebook that was
        just written is not touched: a failed archive is a line in the status
        bar.
        """
        from xp.workers.converter import archive

        self.status.set('Learning material: copying the pdf, the settings and '
                        'the notebook...')
        # Read from Tk *here*: `self.page` is a variable on the widget, and
        # reading it on the worker is what "main thread is not in main loop"
        # means -- the copy itself touches no widget at all.
        page = self.page.get() - 1
        outcome = {}

        def work():
            try:
                outcome['path'] = archive.archive_sample(
                    self.path, target, getattr(self, '_profile_used', None),
                    page=page)
            except BaseException as exc:
                outcome['error'] = exc

        worker = threading.Thread(target=work, daemon=True,
                                  name='xp-learning-material')
        worker.start()

        def wait():
            if worker.is_alive():
                self.root.after(50, wait)
                return
            if 'error' in outcome:
                # Said twice on purpose: the window may be closing, and the
                # console is where a background failure is read afterwards.
                print('learning material: %s' % (outcome['error'],), flush=True)
                self.status.set('Saved %s -- learning material was not kept: %s'
                                % (target, outcome['error']))
            else:
                self.status.set('Saved %s -- learning material in %s'
                                % (target, outcome['path']))
                self._last_learning_material = outcome['path']
            self._finish_after_saving(target)

        self.root.after(50, wait)

    def _finish_after_saving(self, target) -> None:
        """Report the save, and close: a finished conversion is a finished window.

        `exit_after_save` is off only in tests that look at the window
        afterwards. The merge's `settle` path never reaches this method at all
        -- it hands the profile back before anything is written -- so there is
        nothing here to switch off for it.
        """
        if self.exit_after_save:
            self.close()           # reports to the launcher on its way out
            return
        self._report_done(True)

    def _report_done(self, clean: bool) -> None:
        """Tell the launcher this window's task is over. Once only.

        Three cases are deliberately not a report:

        * **the merge opened this window** as its own conversion step
          (`settle`). That is a step inside somebody else's task, and reporting
          it would let the launcher close while the merge was still writing.
        * **cancelling a conversion on purpose.** The user asked it to stop; it
          stopped; nothing is broken. The window stays so the file can be
          looked at, and closing it later reports.
        * **a recoverable warning**, such as a page the tuner could not
          improve. Those say so in the window and are not failures.
        """
        if self._settled:
            return
        done, self.on_finish = self.on_finish, None
        if done is not None:
            done(clean)

    def close(self):
        self._closed = True
        # A conversion the user never answered the dialog for is a file in the
        # system temp directory that nothing will ever look at again. Closing
        # the window is the user deciding, so it goes. A saved conversion has
        # already cleared `_staged`, and in the merge's `settle` path this is
        # called before there is anything staged at all.
        self._discard_staged()
        self._cancel.set()
        if self._tuning is not None:
            self._tuning.set()
        if self._follow_job is not None:
            try:
                self.root.after_cancel(self._follow_job)
            except Exception:
                pass
            self._follow_job = None
        self.root.after_cancel(self._poll_id)
        if self._debounce:
            self.root.after_cancel(self._debounce)
        # Closing the window is finishing with it, so the launcher hears about
        # it here -- reported before the destroy, so the launcher's own work is
        # not racing a window that is already gone.
        self._report_done(True)
        self.root.destroy()

    def run(self):
        self.root.mainloop()
