"""Pick a page's conversion settings by trying them against the page itself.

Every candidate is traced and measured with inkdiff against the page render,
and the choice is made in order of what matters:

1. fewest whole marks lost or invented -- a vanished full stop or a stray
   speck of noise is a visible defect, whatever its share of the ink;
2. then as close to the page as the best candidate, within DIFF_SLACK
   (half a percent) -- closer than that is invisible. Lost ink counts double
   against invented ink (LOST_WEIGHT): ink the tracing never got is writing
   you no longer have, while a speck it invented is one you can see and rub
   out;
3. then the fewest strokes -- among equally faithful tracings, the one without
   thousands of anti-aliasing fragments, which is what a lower-difference
   setting usually buys (on real notes: 4,369 strokes vs 363 for 0.25%).

**Everything is searched, not just the threshold.** Denoise, sharpen, the
opening and the closing used to be left wherever they were set and applied
to every candidate, so a page whose grain needs cleaning before it is
thresholded could not be tuned at all -- and they are not independent of the
threshold: a denoise that is wrong on its own is right beside a looser
offset, and an opening undoes what a lower offset bought. On the user's
phone-scanned page the search now picks denoise 10 with a closing of 3,
which no amount of searching the threshold alone could have found, and the
result is 13.4% of the ink lost where the threshold-only search leaves 15.0%.

That means a space too big to sweep, so the axes are split by what they cost
(see `anneal.py` for the walk and why it is annealing with a momentum term
rather than a gradient):

* **Expensive** -- denoise, sharpen, the morphology and the threshold window.
  Each costs a blur of the page, so these are *walked*: a global search with
  a budget in seconds.
* **Cheap** -- the threshold offset and the speck size. At fixed values of
  the five above, every offset is one comparison against an already-computed
  blur and every speck size is free, so these are *swept* exactly inside each
  evaluation of the walk.

Speed comes from three exact shortcuts. Adaptive thresholding is a Gaussian
blur plus a comparison, and the blur doesn't depend on the offset: blur once
per window, then every offset is one cheap comparison (bit-for-bit what
cv2.adaptiveThreshold gives). The speck size only drops whole pieces, so one
pass scores every speck size at once: rasterize them smallest first so each
grid pixel records the largest piece through it, spread that by the diff
tolerance, and a source pixel is still covered at speck size A exactly when
that value is >= A. And a state the walk has already measured is a
dictionary lookup.

**What a "piece" is depends on the mode, and it used to not.** `outline`
tracing draws contours and drops them by contour area, so it is scored on
contours (`_Page.score`). Every other mode draws the *area* the mask covers
and drops whole connected components of it, so it is scored on the region
(`_Page.score_region`) -- which is the mask itself, measured exactly. Scoring
the outline for a centreline tracing measured a tracing nobody asked for:
the search reported 16.3% of the ink lost on a page whose chosen mask loses
10.2%, and it is why the numbers recorded in HANDOFF.md could not be
reproduced from the settings written beside them.

**Then the bridge.** Getting the region right and deciding how to draw it are
two different questions, and the second one cannot be answered on a proxy:
whether a boundary comes back the weight it was, whether a hairline survives
being rendered, whether the curves inside a thick stroke leave a gap down
the middle -- none of it exists until the strokes are rasterized. So once the
mask is settled, `tune_bridge` searches the drawing settings
(profiles.Profile: boundary_width, fill_spacing, ...) against the tracing as
it will actually be drawn, on the densest patch of the page because each
candidate costs a trace and a render.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import threading
import time

import cv2
import numpy as np

from xp.workers.converter.optimization import anneal
from xp.workers.converter.fields import centerline
from xp.workers.converter.fields import color5d
from xp.workers.converter import profiles
from xp.workers.converter.fields import regions
# The function and not the module: `geometry` is a parameter name in
# `tune_bridge` and `_rendered_ink` (the page's scale and offset), so the
# module would be shadowed exactly where this is wanted.
from xp.core.geometry import pick as pick_components
from xp.workers.converter.convert import (Segment, apply_morph, colour_error, denoise_and_sharpen,
                      encode_regions, page_ink, perfect_image, recover_interiors,
                      region_map, segments_to_strokes, shape_segments, traced_segments)
from xp.core.imaging.inkdiff import (INK_DELTA, Reference, class_reach, color_error, contrast, coverage,
                      grid_size, ink_classes, mark_area, segment_classes, source_ink)
from xp.render.drawing import render_page
from xp.core.document.xopp_io import XoppDocument

WINDOW_FACTORS = (1.5, 2.5, 4, 6, 9)                      # threshold window, in pen widths
OFFSETS = (1, 2, 3, 4, 6, 8, 11, 15, 20, 27, 35)          # grey levels; the threshold floors C
COARSE_OFFSETS = (1, 3, 6, 11, 20, 35)  # tried for every window; the rest only beside the best
SPECK_FACTORS = (0, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2)  # in pen widths squared
MAX_WINDOW = 201
LOST_WEIGHT = 2.0   # lost ink counts double against ink that was invented
# In the weighted units `Trial.difference` is in, so still half a percent of
# lost ink -- the point of the slack is that a difference that small is
# invisible, and weighting lost ink more heavily must not quietly halve it.
DIFF_SLACK = 0.005 * LOST_WEIGHT
# A setting that makes this many contours per mark on the page is tracing the
# paper's grain, and every looser offset only adds more (at a 7px window on a
# phone scan: 400k contours, 37s just to find them). Offsets are tried strict
# to loose, so each window stops there.
NOISE_CONTOURS_PER_MARK = 50
COLOR_BREAKS = (0, 2, 4, 6, 8, 10)  # how readily a contour is cut where its ink color changes
COLOR_SLACK = 0.01                  # under a point of color gain isn't worth more strokes

# The rungs of the preprocessing axes the global search moves along: the same
# 0-10 sliders purify.py used, thinned to the values that differ visibly. The
# morphology stops at 6 because past that the kernel is wider than a pen on
# any of the user's pages, and an opening that wide takes the writing with it.
DENOISE_RUNGS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
SHARPEN_RUNGS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
MORPH_RUNGS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0)
# One mark lost or invented outweighs the whole range `Trial.difference` can
# take (LOST_WEIGHT + 1), so the scalar the walk minimizes is ordered the way
# choose() decides. The stroke term is small enough to settle a tie inside
# DIFF_SLACK and never large enough to outweigh a real difference.
MARK_ENERGY = 10.0
STROKE_ENERGY = 1e-4
# How long each searched phase may take, and how many states it may try in
# that time. Whichever runs out first stops it; the phase before them is the
# exhaustive sweep that auto-tune always did, so a budget of nearly nothing
# degrades to the old behaviour rather than to no answer.
#
# The bridge gets more states than the mask does because its space has six
# axes rather than five and its evaluations are cheaper (a patch, not the
# page). At sixty it was still improving on its last evaluation, and the
# answer depended on whether the clock stopped it one state early -- which
# is how one run kept a simplification of zero and 413,000 points while the
# next found 0.15 and 146,000.
BUDGET_SECONDS = 30.0
STATE_BUDGET = 60
BRIDGE_BUDGET = 100
# The bridge settings are measured on the densest patch of the page rather
# than the whole of it. They are all in pen widths and all local -- how wide
# to stroke a boundary, how far apart to space the curves inside it -- so a
# patch with plenty of writing in it answers the same question as the page,
# for a fifteenth of the work, which is the difference between a search and
# five evaluations. 768 rather than 1024 because the cost is quadratic in it
# and what the budget buys is evaluations: the smaller patch still holds
# hundreds of marks on any of the user's pages.
BRIDGE_PATCH = 768
# Which modes have a bridge worth searching. Centreline tracing has none of
# these settings -- its strokes are the skeleton at the width the ink had,
# and there is nothing to choose.
# Which modes have a bridge to search at all: the modes that own at least one
# axis this search actually varies. profiles.MODE_ONLY is the one table that
# says which setting belongs to which mode, so this is read from it rather
# than written twice -- but it has to be read through `bridge_space`, not over
# the whole table. Reading the whole table made every mode named anywhere in
# it a bridge mode, which is how `ortho-fill` came to spend thirty seconds
# searching boundary widths and ring spacings that it never reads: exactly
# the fault of searching an axis with no effect, which does not merely waste
# time but reports a "best" value for a number that did nothing.
#
# What was never done is the other half: nobody gave the bridge the numbers
# `ortho-fill` *does* read, so the default mode had no fourth phase at all
# and the settings that decide its stroke count were never searched. They
# are below, and `line_width` is deliberately not among them -- every
# ortho-fill stroke carries its own measured width, so the profile's value
# is the default for a stroke whose width is None, which this mode never
# makes. Moving it from 0.04 to 0.12 changes nothing about the output, which
# is exactly the sort of axis this table exists to keep out.
#
# `rib_spacing` is NOT here, and that is the second axis this table has had
# to keep out. It is a live setting -- it changes the ink, which is more than
# `line_width` does -- but **the densest patch cannot judge it**. Measured
# both ways on the user's page:
#
#   on the whole page   rib 1.0 -> 1,811 px outside; rib 1.3 -> 37,305
#   on the patch        rib 1.0 -> 1.317% outside;   rib 0.6 -> 0.919%
#
# The page says a wider spacing is eight times worse and the patch says a
# narrower one is better, because a 768-pixel crop's outside ink is mostly
# strokes cut off at the crop's own edge. An axis whose only sample
# disagrees with the page is an axis this phase cannot yet decide, and
# searching it anyway would report a chosen value that makes the page worse.
# `merge_gap` and `merge_colour` have no such problem: across the whole gap
# ladder the page's ink does not move at all (bare 0 throughout, outside
# 1,796 to 1,826), so what the patch sees of them is all there is to see.
BRIDGE_AXES = {
    'ortho-fill': ('merge_gap', 'merge_colour'),
    'onion-fill': ('boundary_width', 'fill_spacing', 'fill_overlap',
                   'boundary_smooth', 'boundary_shift', 'simplify_epsilon'),
}
BRIDGE_MODES = tuple(sorted(BRIDGE_AXES))
# Heavier than STROKE_ENERGY, and on points rather than strokes: here the
# size of the tracing is not just a tie-break. Xournal++ strokes every
# segment of every polyline separately with cairo, measured at 1.26 fps
# against 45 for a filled boundary (issue #4662), so what it pays is a point
# at a time whatever the strokes are cut into.
#
# Set from the trade as measured on the sample page, not from taste. On the
# phone scan at the mask the search picks for it, with the curve spacing held
# at half a pen:
#
#     simplify at   ink lost    points
#         0.0 px     12.05%     413,543
#         0.3 px     12.62%      98,598
#         0.8 px     12.87%      75,113
#
# Simplifying at 0.3px is a quarter of the points for half a point of ink, so
# the term must be worth 0.0113 over a drop of 1.43 in log points -- above
# 0.008 per e-fold. Going on to 0.8px would need above 0.018, and *widening
# the curve spacing* to save the same points would need 0.074. At 0.012 the
# search therefore takes the simplification, which is nearly free, and
# refuses to buy points back by spacing the curves further apart, which is
# not. Nothing at this weight can outrun a mark (MARK_ENERGY is 10).
#
# Why the term is here at all: 413,000 points on one page is 41 million
# across the user's notebook, and Xournal++ strokes every segment of every polyline
# separately.
BRIDGE_POINT_ENERGY = 0.012
# And a stroke, because the bridge could not say "fewer strokes" and so said
# the opposite. On `ortho-fill` every candidate is equally faithful -- no ink
# lost, none invented, no marks off, at any setting of these axes -- so the
# only live term was points, joining costs about a point per fold, and the
# search turned the join off: 201,629 strokes where the default gives 22,460.
# The user's order of priority has five entries and the fourth is fewer strokes.
#
# Derived, not chosen. A stroke and a point are both bytes in the file, and
# the marginal cost of each was measured by serialising a document with and
# without one: 76.06 bytes a stroke, 18.00 a point. So a stroke is 4.23
# points' worth of file, and the weight is the point weight times that ratio.
# Nothing here was turned until the answer came out right -- the ratio is a
# measurement of the format, and this is what it implies.
BRIDGE_STROKE_BYTES = 76.06
BRIDGE_POINT_BYTES = 18.00
BRIDGE_STROKE_ENERGY = BRIDGE_POINT_ENERGY * BRIDGE_STROKE_BYTES / BRIDGE_POINT_BYTES
# OpenCV already spreads each call over every core; extra threads only pay
# off on big pages (1.4-1.8x at 800 DPI) and cost 2x on small ones.
WORKERS = 3   # each holds a few full-page buffers; 800 DPI A4 is ~60 MB apiece
THREADED_ABOVE_PX = 4_000_000


class TuneCancelled(Exception):
    pass


@dataclass(frozen=True)
class Trial:
    block: int
    offset: int
    min_area: float
    lost: float
    extra: float
    marks_lost: int
    marks_extra: int
    strokes: int
    # The preprocessing this was measured under. Last, and with defaults, so
    # a trial can still be written positionally; zero is what this stage used
    # to be fixed at, and still is for the first phase of the search.
    denoise: float = 0.0
    sharpen: float = 0.0
    morph_open: float = 0.0
    morph_close: float = 0.0

    @property
    def marks_off(self):
        return self.marks_lost + self.marks_extra

    @property
    def before(self):
        """The preprocessing, under the profile's own field names."""
        return {'denoise': self.denoise, 'sharpen': self.sharpen,
                'morph_open': self.morph_open, 'morph_close': self.morph_close}

    @property
    def difference(self):
        """How far this tracing is from the page, lost ink weighted heavier.

        The two are not equally bad: ink that never made it into the tracing
        is writing you no longer have, while ink it invented is a speck you
        can see and rub out. Weighting them the same let the search trade a
        lost mark for a couple of fewer specks.
        """
        return LOST_WEIGHT * self.lost + self.extra


@dataclass
class TuneResult:
    profile: object
    chosen: Trial
    marks: int
    seconds: float
    trials: list = field(repr=False, default_factory=list)
    color_break: float = 0.0    # chosen strength for cutting a contour where its color changes
    color_wrong: float = 0.0    # share of traced ink left in a color that isn't under it
    strokes: int = 0             # after the color split (chosen.strokes is before it)
    walk: object = None          # the anneal.Result for the page's own settings
    bridge: object = None        # the anneal.Result for region-to-strokes, or None

    def summary(self):
        t = self.chosen
        if t.marks_lost:
            marks = f'{t.marks_lost:,} of {self.marks:,} marks still lost (red)'
        else:
            marks = f'All {self.marks:,} marks kept'
        if t.marks_extra:
            marks += f', {t.marks_extra:,} specks invented (blue)'
        before = ''
        if t.denoise or t.sharpen or t.morph_open or t.morph_close:
            before = (f' Cleaned first: denoise {t.denoise:g}, sharpen {t.sharpen:g},'
                      f' open {t.morph_open:g}, close {t.morph_close:g}.')
        bridge = ''
        if self.bridge is not None:
            # Whatever that mode's bridge actually searched. Naming two axes
            # here by hand was a `KeyError` waiting for the second mode to
            # get a fourth phase, and it did.
            chosen = ', '.join('%s %g' % (name, value) for name, value
                               in self.bridge.settings.items())
            bridge = (' Strokes drawn at %s (%d tried).'
                      % (chosen, self.bridge.evaluations))
        color = ''
        if self.color_break:
            color = f' Colors split at {self.color_break:g}, {self.color_wrong:.1%} still off.'
        return (f'Auto-tuned: tried {len(self.trials):,} settings in {self.seconds:.1f}s. '
                f'{marks}; {t.lost:.1%} of the ink lost, {t.extra:.1%} invented; '
                f'{self.strokes or t.strokes:,} traced.{before}{bridge}{color}')


def choose(trials):
    fewest_marks = min(t.marks_off for t in trials)
    pool = [t for t in trials if t.marks_off == fewest_marks]
    closest = min(t.difference for t in pool)
    pool = [t for t in pool if t.difference <= closest + DIFF_SLACK]
    return min(pool, key=lambda t: (t.strokes, t.difference, -t.min_area))


def _odd(value):
    value = max(3, int(round(value)))
    return value if value % 2 else value + 1


class _Page:
    """The page under one set of preprocessing, and the reference to score it.

    The preprocessing -- denoise, sharpen and the morphology -- is part of
    what the search now chooses, so it is a constructor argument rather than
    something read off the profile once. It defaults to the profile's own
    values, which is what the first phase of the search uses and what every
    other caller wants.
    """

    def __init__(self, raw, profile, probe, reference=None, before=None):
        before = dict(before or {})
        self.denoise = float(before.get('denoise', profile.denoise))
        self.sharpen = float(before.get('sharpen', profile.sharpen))
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        self.gray = denoise_and_sharpen(gray, self.denoise, self.sharpen)
        self.gray_f = self.gray.astype(np.float32)
        self.morph = (float(before.get('morph_open', profile.morph_open)),
                      float(before.get('morph_close', profile.morph_close)))
        self.before = {'denoise': self.denoise, 'sharpen': self.sharpen,
                       'morph_open': self.morph[0], 'morph_close': self.morph[1]}
        self.height, self.width = self.gray.shape
        self.grid = grid_size(self.width, self.height)
        gw, gh = self.grid
        if reference is None:
            reference = Reference(source_ink(raw, probe.background_bgr, self.grid),
                                  mark_area(max(1.0, probe.stroke_thickness_px), gw / self.width),
                                  contrast(raw, probe.background_bgr, self.grid))
        self.reference = reference
        ref = self.reference
        self.page_ink = None
        self.page_ink_allowed = None
        if getattr(profile, 'correct_against_page', False) and getattr(profile, 'recover_interiors', False):
            self.page_ink = page_ink(raw, probe.background_bgr)
            self.page_ink_allowed = cv2.dilate(self.page_ink.view(np.uint8),
                                               np.ones((3, 3), np.uint8)) > 0
        self.source_idx = np.flatnonzero(ref.source)
        self.source_labels = ref.labels.ravel()[self.source_idx]
        self.near = ref.near.ravel()

    def mean(self, block):
        # cv2.adaptiveThreshold's Gaussian mean: blurred in float, rounded to 8 bits.
        blurred = cv2.GaussianBlur(self.gray_f, (block, block), 0,
                                   borderType=cv2.BORDER_REPLICATE | cv2.BORDER_ISOLATED)
        return cv2.convertScaleAbs(blurred)  # saturate_cast rounding, as convertTo does; values are >= 0

    def binary(self, below_mean, offset):
        """One candidate's mask -- the same one the pipeline would trace.

        Including the interiors the page says are ink: the search has to
        score the mask that is really going to be drawn, or it optimises a
        thing that never exists. Stage one is shared by every mode for
        exactly this reason (convert.perfect_image), and the page's own ink
        is measured once because no setting can move it.
        """
        # Ink where gray <= mean - floor(C), i.e. (mean - gray) >= offset.
        binary = cv2.threshold(below_mean, offset - 1, 255, cv2.THRESH_BINARY)[1]
        binary = apply_morph(binary, *self.morph)
        if self.page_ink is not None:
            recovered, _given = recover_interiors(binary > 0, self.page_ink)
            binary = (recovered & self.page_ink_allowed).view(np.uint8) * 255
        return binary

    def contours(self, below_mean, offset):
        contours, _ = cv2.findContours(self.binary(below_mean, offset),
                                       cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        return [c for c in contours if len(c) >= 3]

    def trials(self, below_mean, block, offset, speck_sizes, region):
        """This (window, offset) scored at every speck size, and how many
        separate pieces it traced -- which is what the noise limit reads: a
        setting making tens of thousands of them is tracing the paper."""
        binary = self.binary(below_mean, offset)
        if region:
            scored = self.score_region(binary, block, offset, speck_sizes)
            return scored, max((t.strokes for t in scored), default=0)
        contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contours = [c for c in contours if len(c) >= 3]
        return self.score(contours, block, offset, speck_sizes, binary), len(contours)

    def _contour_areas(self, contours, binary):
        """The area the speck rule weighs each contour by.

        The *component's* area, not the contour's own, because that is what
        the pipeline drops a mark by (convert.perfect_image filters on
        CC_STAT_AREA, for every mode alike since 2026-10-08). The two differ
        wherever a mark encloses anything: a loop of handwriting has an outer
        contour whose area counts the paper inside it and a hole contour of
        its own, and the search kept or dropped each of them on its own
        terms. On a drawn test page that is invisible; on real handwriting, which
        is full of loops, the search's estimate was out by 13%.

        A contour point sits on the ink of its own component, outer or hole
        alike, so one point is enough to find it.
        """
        if binary is None:
            return np.array([cv2.contourArea(c) for c in contours], dtype=np.float32)
        _count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
        picked = np.array([c.reshape(-1, 2)[0] for c in contours], dtype=np.int64)
        rows = np.clip(picked[:, 1], 0, labels.shape[0] - 1)
        columns = np.clip(picked[:, 0], 0, labels.shape[1] - 1)
        return areas[labels[rows, columns]]

    def score(self, contours, block, offset, speck_sizes, binary=None):
        ref = self.reference
        gw, gh = self.grid
        if not contours:
            lost = 1.0 if ref.total else 0.0
            return [Trial(block, offset, a, lost, 0.0, ref.marks, 0, 0, **self.before)
                    for a in speck_sizes]
        areas = self._contour_areas(contours, binary)
        order = np.argsort(areas, kind='stable')
        areas = areas[order]
        lengths = np.array([len(contours[i]) for i in order])
        points = np.concatenate([contours[i] for i in order]).reshape(-1, 2).astype(np.int64)
        ids = np.repeat(np.arange(len(order), dtype=np.int32), lengths)
        flat = (points[:, 1] * gh // self.height) * gw + points[:, 0] * gw // self.width
        # Smallest first, so each pixel ends up owned by the largest contour
        # through it: the pixel stays traced for as long as that one is kept.
        owner = np.zeros(gw * gh, np.int32)
        owner[flat] = ids + 1
        drawn = np.flatnonzero(owner)
        own = owner[drawn] - 1
        area_at = np.full(gw * gh, -1, np.float32)
        area_at[drawn] = areas[own]
        reach = cv2.dilate(area_at.reshape(gh, gw), ref.kernel).ravel()

        source_reach = reach[self.source_idx]
        sorted_source_reach = np.sort(source_reach)
        mark_reach = np.full(ref.label_count, -1, np.float32)
        np.maximum.at(mark_reach, self.source_labels, source_reach)
        mark_reach = np.sort(mark_reach[ref.mark_ids])
        near = self.near[drawn]
        drawn_area = np.sort(areas[own])
        far_area = np.sort(areas[own[~near]])
        total = np.bincount(own, minlength=len(areas))
        grounded = np.bincount(own[near], minlength=len(areas))
        invented = np.sort(areas[(total > 0) & (grounded == 0)])

        a = np.asarray(speck_sizes, np.float32)
        lost_px = np.searchsorted(sorted_source_reach, a, 'left')
        traced_px = len(drawn_area) - np.searchsorted(drawn_area, a, 'left')
        extra_px = len(far_area) - np.searchsorted(far_area, a, 'left')
        marks_lost = np.searchsorted(mark_reach, a, 'left')
        marks_extra = len(invented) - np.searchsorted(invented, a, 'left')
        strokes = len(areas) - np.searchsorted(areas, a, 'left')
        return [Trial(block, offset, float(speck_sizes[i]),
                      float(lost_px[i]) / ref.total if ref.total else 0.0,
                      float(extra_px[i]) / traced_px[i] if traced_px[i] else 0.0,
                      int(marks_lost[i]), int(marks_extra[i]), int(strokes[i]),
                      **self.before)
                for i in range(len(speck_sizes))]

    def score_region(self, binary, block, offset, speck_sizes):
        """The same measurement, of the region rather than of its outline.

        For every mode but `outline`, what gets drawn is the *area* the mask
        covers -- a centreline down the middle of it, or the boundary and the
        curves inside it -- and `min_area` drops whole connected components
        of the mask, not contours (see convert._solid_segments and
        _center_segments, both of which filter on CC_STAT_AREA). Scoring the
        outline instead measured a tracing nobody asked for: on one of the user's
        pages the search reported 16.3% of the ink lost where the mask it had
        chosen loses 10.2%, and the recorded numbers in HANDOFF.md could not
        be reproduced from the settings beside them because they had been
        measured against a different thing entirely.

        The shortcut is the same one and just as exact: each grid pixel
        records the area of the largest component through it, so a pixel is
        still covered at speck size A exactly when that value is at least A,
        and one pass scores every speck size at once.
        """
        ref = self.reference
        gw, gh = self.grid
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count <= 1:
            lost = 1.0 if ref.total else 0.0
            return [Trial(block, offset, a, lost, 0.0, ref.marks, 0, 0, **self.before)
                    for a in speck_sizes]
        areas = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
        where = np.flatnonzero(binary.ravel())
        own = labels.ravel()[where]
        own_area = areas[own]
        # Smallest first, so each grid pixel ends up holding the area of the
        # largest component through it.
        order = np.argsort(own_area, kind='stable')
        pixels, sorted_area = where[order], own_area[order]
        flat = ((pixels // self.width) * gh // self.height) * gw \
            + (pixels % self.width) * gw // self.width
        area_at = np.full(gw * gh, -1, np.float32)
        area_at[flat] = sorted_area
        reach = cv2.dilate(area_at.reshape(gh, gw), ref.kernel).ravel()

        drawn = np.flatnonzero(area_at >= 0)
        drawn_area = np.sort(area_at[drawn])
        far_area = np.sort(area_at[drawn[~self.near[drawn]]])
        sorted_source_reach = np.sort(reach[self.source_idx])
        mark_reach = np.full(ref.label_count, -1, np.float32)
        np.maximum.at(mark_reach, self.source_labels, reach[self.source_idx])
        mark_reach = np.sort(mark_reach[ref.mark_ids])
        # A component is invented when no pixel of it lands near the page's
        # own ink -- the same test as for a contour, one level up.
        grid_labels = np.zeros(gw * gh, np.int32)
        grid_labels[flat] = own[order] + 1
        marked = np.flatnonzero(grid_labels)
        held = grid_labels[marked] - 1
        touching = self.near[marked]
        seen = np.bincount(held, minlength=count)
        grounded = np.bincount(held[touching], minlength=count)
        invented = np.sort(areas[(seen > 0) & (grounded == 0)])

        a = np.asarray(speck_sizes, np.float32)
        kept = np.sort(areas[1:])
        lost_px = np.searchsorted(sorted_source_reach, a, 'left')
        traced_px = len(drawn_area) - np.searchsorted(drawn_area, a, 'left')
        extra_px = len(far_area) - np.searchsorted(far_area, a, 'left')
        marks_lost = np.searchsorted(mark_reach, a, 'left')
        marks_extra = len(invented) - np.searchsorted(invented, a, 'left')
        strokes = len(kept) - np.searchsorted(kept, a, 'left')
        return [Trial(block, offset, float(speck_sizes[i]),
                      float(lost_px[i]) / ref.total if ref.total else 0.0,
                      float(extra_px[i]) / traced_px[i] if traced_px[i] else 0.0,
                      int(marks_lost[i]), int(marks_extra[i]), int(strokes[i]),
                      **self.before)
                for i in range(len(speck_sizes))]


def search_space(profile, probe):
    """Threshold windows, offsets and speck sizes to try, plus the current
    (window, offset). The current settings are always scored, so tuning can't
    pick something that measures worse than what was already there."""
    pen = max(1.0, probe.stroke_thickness_px)
    blocks = sorted({min(MAX_WINDOW, _odd(f * pen)) for f in WINDOW_FACTORS})
    specks = sorted({round(f * pen * pen, 2) for f in SPECK_FACTORS} | {float(profile.min_area)})
    current = None
    if profile.adaptive_c >= 1:
        current = (profile.adaptive_block_size, math.floor(profile.adaptive_c))
    return blocks, sorted(COARSE_OFFSETS, reverse=True), specks, current


COLOUR_EVALUATIONS = 60    # a candidate is arithmetic unless it moves the
                           # spatial bandwidth, so this is seconds, not minutes
COLOUR_BANDWIDTHS = (6.0, 10.0, 14.0, 20.0, 28.0, 40.0)   # Delta-E-ish, in direction space
SPACE_BANDWIDTHS = (1.0, 1.75, 2.5, 4.0, 6.0)             # pen widths
# What a stroke is worth against a share of the ink drawn in the wrong
# colour, so that cutting a page into pieces to chase a tenth of a per cent
# is not free. Same shape as BRIDGE_POINT_ENERGY and set the same way: a
# whole per cent of the ink in the wrong pen is worth about a thousand cuts.
COLOUR_WRONG_ENERGY = 100000.0
COLOUR_PIECE_ENERGY = 1.0


def colour_space():
    """The axis of the colour pass. One, now.

    Region-First took the other three away rather than tuned them. The two
    slicing dials no longer exist, because a stroke is born inside one colour
    and is never cut. There is no bandwidth here either, because there is no
    merge left to tune: the clustering's answer about the colour *is* the
    answer, which is what the user asked for twice.

    What is left is how far the clustering looks for a pixel's neighbours --
    the one setting that changes the colour *estimate* rather than what is
    done with it.
    """
    # colour_bandwidth is not here: how many pens a page has is decided by
    # the page (color5d.auto_bandwidth), and searching a number that the
    # page already answers is searching for permission to overrule it.
    return anneal.Space([
        anneal.Axis('space_bandwidth', SPACE_BANDWIDTHS),
    ])


# <~ _crop_field stood here: the page's colour map narrowed to the densest
#    patch, so a candidate could be scored where the writing is. The colour
#    search no longer draws anything -- it scores the Region-Colour Map, and
#    partition_error is already per ink pixel over the whole page -- so there
#    is no patch to crop to. ~>


def tune_colour(raw, profile, probe, mask, budget=None, cancel=None,
                seed=0, progress=None):
    """Search the colour settings against what the page actually has.

    Scored on the Region-Colour Map rather than on strokes, because the map
    is what this setting moves: `color5d.partition_error` is the share of ink
    sitting in a colour the page does not have there, asked per ink pixel.
    Against it stands a price per region, since a finer cut will always lower
    the error and a page in ten thousand pieces is not an improvement.

    That the search no longer draws anything is the Region-First dividend.
    Before, every candidate sliced seven thousand strokes and rasterized the
    patch once per pen, which was most of what made this phase unaffordable.

    Only `space_bandwidth` reaches the clustering, so the grouping is cached
    per value of it and a repeat candidate costs arithmetic over a few
    thousand cluster centres (`color5d.field_from`), not a page of k-means.
    """
    if not profile.colour_5d:
        return None
    pen = centerline.pen_width(mask)
    groupings = {}

    def energy(settings):
        if cancel is not None and cancel.is_set():
            raise anneal.Cancelled()
        candidate = copy.deepcopy(profile)
        for name, value in settings.items():
            setattr(candidate, name, float(value))
        space = round(max(1.0, float(candidate.space_bandwidth) * pen), 3)
        grouping = groupings.get(space)
        if grouping is None:
            grouping = groupings[space] = color5d.group(mask, raw, space,
                                                        probe.background_bgr)
        # No merge to judge any more: every ink pixel takes its own
        # cluster's colour, so the only thing `space_bandwidth` moves is how
        # good that colour is as an estimate. `partition_error` with the
        # identity mapping asks exactly that -- how much ink ends up in a
        # colour the page does not have there -- and it is a measurement, not
        # a choice, which is why it survived the pen layer that used it.
        identity = np.arange(len(grouping.centres_lab))
        share = color5d.partition_error(grouping, identity, grouping.centres_lab)
        # The price per piece, without labelling anything: with no merge
        # left, the pieces the drawing has to make are the colours, and
        # labelling 6,534 colours' components per candidate is the cost the
        # page-wide design exists to avoid.
        pieces = len(grouping.centres_lab)
        return share * COLOUR_WRONG_ENERGY + pieces * COLOUR_PIECE_ENERGY

    space = colour_space()
    start = space.nearest(space_bandwidth=profile.space_bandwidth)
    return anneal.search(space, energy, start=start,
                         evaluations=COLOUR_EVALUATIONS, seconds=budget, seed=seed,
                         cancel=cancel, progress=progress)


def tune_color(page, raw, profile, mask, probe):
    """How readily to cut a contour where its ink color changes.

    Two pen strokes that cross are one contour, so one color for both --
    the minority ends up drawn in the wrong one. Splitting doesn't change
    which pixels are traced, so the ink measures above can't see it at all:
    it's judged on its own, by how much traced ink is left in a color that
    isn't on the page under it, and settled by the fewest strokes once
    that's within COLOR_SLACK of the best. Returns (color_break, color
    error, strokes), or None when the page has only one ink color.
    """
    labels, names = ink_classes(raw, page.reference.source, profile, page.grid)
    reach = class_reach(labels, page.reference.kernel)
    if len(reach) < 2:
        return None
    results = []
    for strength in COLOR_BREAKS:
        candidate = copy.deepcopy(profile)
        candidate.color_break = strength
        # This phase searches the palette splitter, which is the thing
        # color_break turns; the 5-D pass replaces it rather than tuning it.
        candidate.colour_5d = False
        segments = traced_segments(mask, raw, candidate, probe.background_bgr,
                                   probe.stroke_thickness_px)
        masks = segment_classes(segments, names, (page.width, page.height), page.grid)
        results.append((strength, color_error(masks, reach), len(segments)))
    closest = min(error for _, error, _ in results)
    pool = [r for r in results if r[1] <= closest + COLOR_SLACK]
    return min(pool, key=lambda r: (r[2], r[1]))


def energy_of(trial):
    """One number for the annealing to make small, ordered as choose() is.

    `choose` decides lexicographically -- marks first, then how close the
    tracing is, then how many strokes -- and that stays the rule: it is what
    finally picks the winner out of everything the search measured. But a
    walk needs a scalar to compare two states with, so the same order is
    written as one: a mark is worth more than the whole range `difference`
    can take (LOST_WEIGHT + 1 = 3), and the stroke term is small enough to
    settle a tie inside DIFF_SLACK without ever outweighing a real
    difference.

    Keeping the two apart matters. The search decides *where to look* by this
    number; `choose` decides *what to keep* by the rule that was already
    tested and argued for. So a change to the walk cannot quietly change
    which settings are considered best.
    """
    return (MARK_ENERGY * trial.marks_off + trial.difference
            + STROKE_ENERGY * math.log1p(max(0, trial.strokes)))


def _sweep(page, blocks, offsets, specks, noise_limit, region, check, tick,
           current=None, workers=1):
    """Every (window, offset) at this page's preprocessing, all speck sizes at
    once. This was the whole of auto-tune, and is still its first phase."""
    def run(block):
        below_mean = cv2.subtract(page.mean(block), page.gray)
        out = []
        if current and current[0] == block:
            check()
            out.extend(page.trials(below_mean, block, current[1], specks, region)[0])
            tick()
        if block not in blocks:
            return out
        for i, offset in enumerate(offsets):
            check()
            scored, pieces = page.trials(below_mean, block, offset, specks, region)
            if pieces > noise_limit:
                # Offsets are tried strict to loose, so every looser one only
                # traces more grain: stop this window here.
                tick(len(offsets) - i)
                break
            out.extend(scored)
            tick()
        return out

    windows = sorted(set(blocks) | ({current[0]} if current else set()), reverse=True)
    if workers > 1:
        with ThreadPoolExecutor(workers) as pool:
            return [t for chunk in pool.map(run, windows) for t in chunk]
    return [t for block in windows for t in run(block)]


def before_space(profile, probe, blocks):
    """The axes the global search moves in: the preprocessing, and the window.

    These are the settings that change the page itself -- denoise and sharpen
    change the grey, the morphology changes the binary, the window changes
    what "darker than its surroundings" means -- so each one of them costs a
    blur of the page to try. The offset and the speck size are not here: at a
    fixed state of these five, every offset is one comparison against an
    already-computed blur and every speck size is free (see the module
    docstring), so they are enumerated exactly inside each evaluation rather
    than searched. That split is the whole reason a search over the full
    space is affordable at all: the expensive axes get a walk, the cheap ones
    get swept.

    The rungs are the ones that mean something, and they stop where the
    empirical range stops. Denoise and sharpen are the 0-10 sliders purify.py
    used; the morphology rungs stop at 6 because above that the kernel is
    wider than a pen on any of the user's pages and an opening takes the writing
    with it.
    """
    return anneal.Space([
        anneal.Axis('denoise', DENOISE_RUNGS),
        anneal.Axis('sharpen', SHARPEN_RUNGS),
        anneal.Axis('morph_open', MORPH_RUNGS),
        anneal.Axis('morph_close', MORPH_RUNGS),
        anneal.Axis('adaptive_block_size', tuple(blocks)),
    ])


def densest_patch(mask, size=BRIDGE_PATCH):
    """The window of the page with the most ink in it, as (x, y, w, h).

    A summed-area table, so every candidate window is four lookups and the
    whole page is one pass. Picking the busiest patch rather than the middle
    matters on the user's material: half of a scanned page is often margin, and a
    patch of margin would let any bridge setting look perfect.
    """
    height, width = mask.shape
    size = int(min(size, height, width))
    if size >= min(height, width):
        return 0, 0, width, height
    table = cv2.integral(mask.view(np.uint8))
    # Every window whose top-left is on a stride of an eighth of its size:
    # the point is to find a busy patch, not the busiest pixel-for-pixel.
    stride = max(1, size // 8)
    rows = np.arange(0, height - size + 1, stride)
    columns = np.arange(0, width - size + 1, stride)
    total = (table[np.ix_(rows + size, columns + size)]
             - table[np.ix_(rows, columns + size)]
             - table[np.ix_(rows + size, columns)]
             + table[np.ix_(rows, columns)])
    best = np.unravel_index(int(np.argmax(total)), total.shape)
    return int(columns[best[1]]), int(rows[best[0]]), size, size


# The rungs of each bridge axis. Every one is a value that means something on
# a page of handwriting, and each ladder stops where the empirical range
# stops -- which is what makes searching all of them at once affordable.
BRIDGE_RUNGS = {
    # ortho-fill. In pen widths, and bounded by what it costs rather than
    # by taste. On
    # the real page the time is flat from 0 to 4 -- the chunked search
    # absorbed the growth that used to be quadratic in the gap -- while the
    # strokes fall 201,629 -> 22,460 -> 15,758 -> 14,232 and the ink outside
    # does not move. The benefit saturates about 2; the ladder runs to 3 so
    # the search can see that it has, and no further, because past there it
    # would be buying four hundred strokes with a tenth of a second that
    # nothing in the objective can see.
    'merge_gap': (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
    # Delta-E. It decides nothing on a one-pen page and everything where two
    # pens touch: a near-black blue merges with black at 30 and not at 8.
    #
    # Four rungs, not seven, and the reason is what it is worth: on the user's page
    # moving it from 2 to 4 moved the drawing by 0.6% of its strokes, while
    # `merge_gap` moves it by a factor of fourteen. Every rung here is a
    # candidate that has to be traced and rendered -- about 0.6s -- so seven
    # of them spent a third of phase four on the axis that decides least.
    # These four span what it does: never across colours, one noticeable
    # difference, several, and the value at which a near-black blue joins
    # black.
    'merge_colour': (0.0, 2.0, 8.0, 30.0),
    # onion-fill. A boundary below a fifth of a pen is thinner than the
    # renderer can draw and above one pen is heavier than the ink; a curve
    # spacing above about one pen leaves a gap down the middle of a thick
    # stroke, and below half a pen spends strokes covering ink that is
    # already covered. See profiles.Profile for what each does.
    'boundary_width': (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00),
    'fill_spacing': (0.50, 0.65, 0.75, 0.85, 1.00, 1.15),
    'fill_overlap': (1.00, 1.15, 1.30, 1.50, 1.75),
    'boundary_smooth': (0.0, 0.3, 0.6, 0.9, 1.3),
    'boundary_shift': (0.0, 0.2, 0.35, 0.5, 0.8),
    'simplify_epsilon': (0.0, 0.15, 0.30, 0.50, 0.80),
}


def bridge_space(profile):
    """The axes of the last bridge: how the settled region becomes strokes.

    The axes are the ones **this mode reads**, and no others. Searching an
    axis with no effect does not merely waste the time: it reports a chosen
    value for a number that did nothing, which reads afterwards as a finding.
    That is why the table is keyed by mode and why `line_width` is not in
    `ortho-fill`'s row -- see `BRIDGE_AXES`.
    """
    names = BRIDGE_AXES.get(getattr(profile, 'trace', None), ())
    return anneal.Space([anneal.Axis(name, BRIDGE_RUNGS[name])
                         for name in names])


def _rendered_ink(segments, profile, size, geometry):
    """The tracing as pixels, drawn the way Xournal++ draws it.

    In the page's own pixels at scale 1 when the caller has no geometry to
    hand: the bridge settings are all in pen widths, so a self-consistent
    resolution answers the same question. When the caller does have the
    page's (scale, dx, dy), the strokes are placed and drawn through it
    instead, which is the resolution the output will really be seen at -- a
    hairline that survives at full size and rounds away at the page's own
    scale is a hairline that is lost.
    """
    width, height = size
    if geometry is None:
        scale, dx, dy = 1.0, 0.0, 0.0
        page_width, page_height, draw = float(width), float(height), 1.0
    else:
        scale, dx, dy = geometry
        page_width, page_height = profile.page_width, profile.page_height
        draw = scale                     # back to the page's own pixels
    document = XoppDocument.new_empty(1, page_width, page_height)
    document.pages[0].layer.extend(segments_to_strokes(segments, profile, scale, dx, dy))
    drawn = render_page(document.pages[0], scale=draw)
    ink = np.abs(255 - np.asarray(drawn, np.int16)).sum(axis=2) > INK_DELTA
    if geometry is None:
        return ink[:height, :width]
    # The content sits at (dx, dy) in page units; in these pixels that is
    # (dx * scale, dy * scale), and the page is as wide as the render.
    left, top = int(round(dx * draw)), int(round(dy * draw))
    return ink[top:top + height, left:left + width]


def tune_bridge(raw, profile, probe, mask, reference, geometry=None, budget=None,
                cancel=None, seed=0, progress=None):
    """How to turn the settled region into strokes, measured on the strokes.

    Everything up to here has been about getting the region right, and it is
    measured on the region. This is the other half, and it cannot be measured
    the same way: whether a boundary comes back the weight it was, whether a
    hairline survives being drawn, whether the curves inside a thick stroke
    leave a gap down the middle -- none of that exists until the strokes are
    rasterized. So each candidate is traced, rendered and compared, which
    costs seconds a page, which is why it is done on the densest patch of the
    page (`densest_patch`) rather than the whole of it.

    Returns the anneal.Result, or None when there is nothing to draw.
    """
    x, y, width, height = densest_patch(mask)
    patch_mask = np.ascontiguousarray(mask[y:y + height, x:x + width])
    patch_raw = np.ascontiguousarray(raw[y:y + height, x:x + width])
    if not patch_mask.any():
        return None
    # The speck filter phase one chose, applied here too, because the tracer
    # applies it (convert._solid_segments, _center_segments) and the bridge
    # must not be charged for ink it was never given. Without this, 55 of the
    # 269 marks on the patch counted as lost by every candidate -- they are
    # the components `min_area` drops -- and a constant 550 of mark energy
    # sat on top of an ink term worth 0.06 and swamped it.
    if profile.min_area > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            patch_mask.view(np.uint8), connectivity=8)
        keep = np.zeros(count, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= profile.min_area
        patch_mask = pick_components(keep, labels, patch_mask)
        if not patch_mask.any():
            return None
    # At the page's own resolution, pixel for pixel, and not on the
    # comparison grid the page-level numbers use. That grid is under half
    # size on an 800-DPI page, and the defects this phase exists to prevent
    # are sub-pixel there: a letter whose middle the curves did not quite
    # close reads as solid once it is shrunk by half, and comes back hollow
    # on the screen. The first version of this measured on the grid, scored
    # 11.4% of the ink lost, and drew writing you could see through.
    grid = (width, height)
    # Against the page, with the patch's own contrast so that a mark is a
    # mark: the page is what the result is finally judged by and what the user
    # looks at, so it is what this is measured against too. The mask is an
    # intermediate.
    #
    # `margins=False` because this is a crop. The margin filter drops a blob
    # that touches the edge and covers more than two percent of the image --
    # the scan's border on a page, and the writing on a 768-pixel window cut
    # from the middle of the densest part of one. With it on, the user's page's
    # patch came back with 0 of its 114,782 ink pixels and this returned None
    # three lines later: phase four has not run on a dense page, for any
    # mode, since the crop was introduced.
    patch_reference = Reference(
        source_ink(patch_raw, probe.background_bgr, grid, margins=False),
        mark_area(max(1.0, probe.stroke_thickness_px), grid[0] / width),
        contrast(patch_raw, probe.background_bgr, grid))
    if not patch_reference.total:
        return None
    # ...but only for the marks it could have got. Drawing the mask perfectly
    # still disagrees with the page wherever the mask does, and on the densest
    # patch of the user's phone scan that is about sixty invented marks for every
    # candidate -- the chosen mask is fatter than the raw ink, so the strokes
    # cover paper. At ten energy a mark that is a term worth 600 wandering by
    # one or two as settings move, against an ink term worth 0.24 and a point
    # term worth 0.02: the search was reading the noise in the mask's own
    # disagreement with the page and calling it a preference, and it picked a
    # simplification of zero (413,000 points a page) on that basis. So the
    # floor the mask already sets is subtracted, and this phase is charged
    # only for marks it loses or invents *beyond* it. The ink is not treated
    # this way and should not be: a slightly heavier stroke really can cover
    # ink the mask missed, and that is a gain worth having.
    floor = patch_reference.compare(coverage(patch_mask, grid))
    floor_marks = floor.marks_lost + floor.marks_extra
    space = bridge_space(profile)

    def energy(settings):
        if cancel is not None and cancel.is_set():
            raise anneal.Cancelled()
        candidate = copy.deepcopy(profile)
        for name, value in settings.items():
            setattr(candidate, name, float(value))
        # The bridge is a search over *shape*: what it scores is where the ink
        # lands, and it runs this hundreds of times. Colour is decided once
        # for the whole page afterwards and cannot change any of these
        # settings, so tracing it here would be the search paying for an
        # answer it then throws away -- several seconds of clustering per
        # candidate, which is what made the auto-tune previews time out.
        candidate.colour_5d = False
        segments = traced_segments(patch_mask, patch_raw, candidate,
                                   probe.background_bgr, probe.stroke_thickness_px)
        if not segments:
            return math.inf
        ink = _rendered_ink(segments, candidate, (width, height), None)
        difference = patch_reference.compare(coverage(ink, grid))
        # Points, not strokes: Xournal++ pays cairo per *segment of a
        # polyline*, so a thousand-point path costs a thousand times a
        # two-point one whether it is one stroke or five hundred. Counting
        # strokes let the search turn simplification off entirely and hand
        # back 296,000 points for a page that needs 31,000.
        points = sum(len(segment.points) for segment in segments)
        marks_off = max(0, difference.marks_lost + difference.marks_extra - floor_marks)
        return (MARK_ENERGY * marks_off
                + LOST_WEIGHT * difference.lost_share + difference.extra_share
                + BRIDGE_POINT_ENERGY * math.log1p(points)
                + BRIDGE_STROKE_ENERGY * math.log1p(len(segments)))

    return anneal.search(space, energy,
                         start=space.nearest(**{axis.name: getattr(profile, axis.name)
                                                for axis in space.axes}),
                         evaluations=BRIDGE_BUDGET, seconds=budget, seed=seed,
                         progress=progress, cancel=cancel)


def tune_page(raw, profile, probe, progress=None, cancel=None, reference=None,
              budget=None, geometry=None, seed=0):
    """Search this page's settings; returns a TuneResult whose `profile` is a
    copy of `profile` with everything the search chose.

    Four phases, in the order the pipeline runs and in the order the numbers
    matter:

    1. every window and offset at the preprocessing the profile already has
       -- which is exactly what auto-tune did before, so the answer can never
       be worse than it used to be, and it is the first few seconds rather
       than the last;
    2. a global search over the preprocessing *and* the window together
       (`anneal.py`), because they interact: a denoise that is wrong on its
       own is right beside a looser offset, and an opening undoes what a
       lower offset bought. This is where "all the settings, not just the
       last stage" happens;
    3. the offsets either side of the best window, to land between the coarse
       rungs;
    4. the bridge from the region to the strokes -- boundary width, curve
       spacing, smoothing, simplification -- measured on the tracing as it
       will actually be rendered, not on a proxy (`tune_bridge`). The page
       comes first and the strokes second on purpose: there is no point
       choosing how to draw a region that is the wrong region.

    `budget` is seconds for each of the searched phases (2 and 4); `reference`
    is the page's inkdiff.Reference if already computed, and `geometry` its
    (scale, dx, dy) if the caller has it, so phase 4 can measure at the scale
    the output will really be drawn at.
    """
    start = time.perf_counter()
    budget = BUDGET_SECONDS if budget is None else float(budget)
    page = _Page(raw, profile, probe, reference)
    blocks, offsets, specks, current = search_space(profile, probe)
    region = profile.trace != 'outline'
    noise_limit = NOISE_CONTOURS_PER_MARK * max(page.reference.marks, 100)
    space = before_space(profile, probe, blocks)
    # Phase 1 is one tick per (window, offset); phase 2 one per state it
    # tries; phase 3 at most one per fine offset; phase 4 one per state.
    total = (len(blocks) * len(offsets) + (1 if current else 0)
             + STATE_BUDGET + len(OFFSETS) + BRIDGE_BUDGET + 1)
    done = [0]
    lock = threading.Lock()

    def tick(n=1):
        with lock:
            done[0] = min(total, done[0] + n)
            if progress:
                progress(done[0], total)

    def check():
        if cancel is not None and cancel.is_set():
            raise TuneCancelled()

    workers = WORKERS if page.gray.size > THREADED_ABOVE_PX else 1
    trials = _sweep(page, blocks, offsets, specks, noise_limit, region, check, tick,
                    current, workers)
    if not trials:  # every candidate traced the grain, the current settings included
        raise ValueError('This page is too noisy to tune automatically; adjust the settings by hand.')

    # Phase 2: the same sweep, but the preprocessing and the window are now
    # searched rather than fixed. One page per (denoise, sharpen), because
    # that is what a change of either costs; kept in a tiny cache, since an
    # 800-DPI page's grey is a quarter of a gigabyte in float.
    pages = {}
    cache_room = 2 if page.gray.size <= THREADED_ABOVE_PX else 1

    def page_for(before):
        key = (before['denoise'], before['sharpen'])
        found = pages.get(key)
        if found is None:
            if len(pages) >= cache_room:
                pages.clear()
            found = pages[key] = _Page(raw, profile, probe, page.reference, before)
        found.morph = (before['morph_open'], before['morph_close'])
        found.before = dict(before)
        return found

    def state_energy(settings):
        check()
        before = {k: v for k, v in settings.items() if k != 'adaptive_block_size'}
        block = int(settings['adaptive_block_size'])
        here = page_for(before)
        below_mean = cv2.subtract(here.mean(block), here.gray)
        best = math.inf
        for i, offset in enumerate(offsets):
            scored, pieces = here.trials(below_mean, block, offset, specks, region)
            if pieces > noise_limit:
                break
            trials.extend(scored)
            best = min([best] + [energy_of(t) for t in scored])
        tick()
        return best

    seeded = choose(trials)
    walk = anneal.search(
        space, state_energy,
        start=space.nearest(denoise=profile.denoise, sharpen=profile.sharpen,
                            morph_open=profile.morph_open, morph_close=profile.morph_close,
                            adaptive_block_size=seeded.block),
        evaluations=STATE_BUDGET, seconds=budget, seed=seed, cancel=cancel)

    # Phase 3: the offsets on either side of the best coarse one, at the
    # preprocessing and window that won.
    best = choose(trials)
    here = page_for(best.before)
    tried = {(t.block, t.offset) for t in trials if t.before == best.before}
    beside = [o for o in OFFSETS if (best.block, o) not in tried and
              max([c for c in COARSE_OFFSETS if c < best.offset], default=0) < o <
              min([c for c in COARSE_OFFSETS if c > best.offset], default=math.inf)]
    if beside:
        below_mean = cv2.subtract(here.mean(best.block), here.gray)
        for offset in beside:
            check()
            scored, pieces = here.trials(below_mean, best.block, offset, specks, region)
            if pieces <= noise_limit:
                trials.extend(scored)
            tick()

    chosen = choose(trials)
    tuned = copy.deepcopy(profile)
    tuned.adaptive_block_size = chosen.block
    tuned.adaptive_c = float(chosen.offset)
    tuned.min_area = chosen.min_area
    tuned.denoise = chosen.denoise
    tuned.sharpen = chosen.sharpen
    tuned.morph_open = chosen.morph_open
    tuned.morph_close = chosen.morph_close
    tuned.auto_threshold = False
    page = page_for(chosen.before)

    mask = page.binary(cv2.subtract(page.mean(chosen.block), page.gray), chosen.offset) > 0

    # Phase 4: how to draw the region, now that the region is settled. Judged
    # on the tracing as it will be rendered -- strokes at their own widths,
    # rasterized and compared with the page -- because that is the thing that
    # is wrong or right, and nothing cheaper tells you whether a boundary is
    # a pixel too heavy.
    check()
    bridge = None
    if tuned.trace in BRIDGE_MODES:
        bridge = tune_bridge(raw, tuned, probe, mask, page.reference, geometry,
                             budget=budget, cancel=cancel, seed=seed,
                             progress=lambda done, whole: tick())
        if bridge is not None:
            for name, value in bridge.settings.items():
                setattr(tuned, name, float(value))

    check()
    # Phase 5: colour. Searched against the page's own colour with no palette
    # in it (convert.colour_error), and against a price per piece, because
    # slicing finer always lowers the error and a page in ten thousand pieces
    # is not an improvement. The user asked for all the settings to be in the
    # search; these four are what decide the colour a stroke comes out.
    colour = None
    if tuned.colour_5d:
        clean, _allowed, _dropped = perfect_image(mask, raw, tuned, probe.background_bgr)
        # <~ a full shape pass stood here, purely to hand the colour search
        #    strokes to slice. Scoring the map instead means the search never
        #    needs them -- which takes thirty seconds off tuning the user's page,
        #    and is why the open-a-file test stopped timing out. ~>
        colour = tune_colour(raw, tuned, probe, clean, budget=budget,
                             cancel=cancel, seed=seed,
                             progress=lambda done, whole: tick())
        if colour is not None:
            for name, value in colour.settings.items():
                setattr(tuned, name, float(value))

    check()
    # The palette splitter, for the one mode that still uses it.
    color = (None if tuned.colour_5d or tuned.trace != 'outline'
             else tune_color(page, raw, tuned, mask, probe))
    if color is not None:
        tuned.color_break = float(color[0])
    tick(total)
    return TuneResult(tuned, chosen, page.reference.marks, time.perf_counter() - start, trials,
                      color[0] if color else 0.0, color[1] if color else 0.0,
                      color[2] if color else chosen.strokes,
                      walk, bridge)
