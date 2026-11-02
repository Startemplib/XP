"""PDF -> xopp strokes. Ported from pdf2xopp_Hand.py / pdf2xopp_Elec.py.

The CV pipeline itself (adaptive threshold -> contours -> per-contour dominant
color -> nearest palette color -> coordinate downsampling to fit Xournal++'s
per-stroke XML length limit) is unchanged from the original scripts, since it
already worked and was never what kept getting revised across the archive.
What moved is everything that used to be hardcoded constants at the bottom of
those scripts: DPI, geometry, thresholds and color tables now live in a
`Profile` (see profiles.py) instead of being copy-pasted per course folder.

Rendering PDF pages uses PyMuPDF instead of the original pdf2image, since
pdf2image needs a separate poppler install that isn't set up on this machine;
PyMuPDF gives the same "PDF page -> raster image at N dpi" with no external
binary.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pymupdf

from xp.core import geometry
from xp.gpu import backend as gpu
from xp.render import drawing as render
from xp.workers.converter.fields import (centerline, color5d, orthofill,
                                         regions, strokejoin)
from xp.core.frame import REFERENCE_FRAME, frame_in_image
from xp.workers.converter.probe import INK_DELTA, page_ink
from xp.workers.converter.profiles import Profile, classify_bgr, map_pixel_elec, map_pixel_hand
from xp.core.document.xopp_io import XoppDocument, make_stroke_element

SMOOTH_WINDOW = 3          # path points averaged, to lose the pixel staircase
SMOOTH_PASSES = 4          # smoothing passes on a geometry loop; each one is
                           # bounded by profile.boundary_shift, so the number
                           # of them decides how smooth, never how far
MAX_ONION_LAYERS = 40      # a region deeper than this many curves is a solid
                           # block, and the last curve's own width closes it
MAX_FILL_PASSES = 8        # top-up passes after the curves; each one draws what
                           # the last one is measured to have missed
MAX_RECOVER_PASSES = 4     # giving ink back can enclose further regions, so
                           # recovery runs to a fixed point -- otherwise running
                           # the stage twice is not the same as running it once
MAX_CORRECT_ROUNDS = 3     # add-then-remove rounds: removing a stroke can open
                           # a hole, so filling runs again after every removal
# <~ INK_DELTA and page_ink moved to probe.py, which is the module that
#    asks the page about itself -- and which had two bare copies of the
#    number. Imported here under the same names, so every use reads the
#    same way it always did. ~>
                           # itself*, as in probe.py and inkdiff.py. The page --
                           # never a threshold setting -- is what decides whether
                           # an enclosed region is a hole or ink


# Fraction of each contour's own brightness range kept when sampling its ink
# color (0 = only the single darkest pixel, 1 = every pixel including the
# lightest anti-aliasing blend). See the comment in _contours_to_strokes.
INK_DARK_FRACTION = 0.35

# How far (summed over B, G and R) a pixel must sit from the paper color to
# count as ink rather than background.
BACKGROUND_TOLERANCE = 30


# Smallest piece a color split may leave, in pen widths, at color_break 10
# and at just above 0. Below about half a pen width a "piece" is shorter than
# the pen is wide, which is noise, not a change of color.
COLOR_BREAK_MIN_RUN = (3.25, 0.5)
HOLE_RIM = (40, 190, 90)   # the rim of a hole, in the same green the
                           # panel uses for ink given back: a hole is
                           # paper the ink encloses, which is the fact
                           # stage 7 is about.
DEFAULT_PEN_WIDTH_PX = 8.0  # when the caller has no probe to hand


def enclosed_regions(mask):
    """The regions the mask encloses: paper that cannot reach the page edge.

    A flood from outside, not a labelling of every component of the paper:
    this runs once per candidate inside the search, so the difference is the
    difference between a tuner that answers and one that times out. The page
    is padded with a ring of paper so the flood has somewhere to start even
    when ink touches the edge.

    Whatever the flood never reaches is surrounded by ink -- the counter of
    an `o`, the gap in an `e`, or the inside of a stroke whose middle the
    threshold dropped. Which of those it is, this does not say:
    `recover_interiors` asks the page.
    """
    if gpu.GPU_AVAILABLE:
        try:
            import cupyx.scipy.ndimage as cnd

            # The same question asked the other way round, because a flood is
            # a walk and a walk is not a parallel job: label the paper and
            # keep what does not carry the edge's label. Identical to the
            # flood, pixel for pixel, and 0.201s becomes 0.033s -- and this
            # runs twice a page and once per candidate inside the search.
            padded = gpu.xp.ones((mask.shape[0] + 2, mask.shape[1] + 2),
                                 gpu.xp.bool_)
            padded[1:-1, 1:-1] = ~gpu.xp.asarray(mask)
            cross = gpu.xp.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            marked, _how_many = cnd.label(padded, structure=cross)
            inner = marked[1:-1, 1:-1]
            held = (inner != marked[0, 0]) & (inner != 0)
            answer = gpu.to_host(held)
            del padded, marked, inner, held
            return answer
        except Exception:
            pass
    height, width = mask.shape
    padded = np.ones((height + 2, width + 2), np.uint8)   # the ring is paper
    padded[1:-1, 1:-1] = (~mask).view(np.uint8)
    cv2.floodFill(padded, np.zeros((height + 4, width + 4), np.uint8), (0, 0), 2, flags=4)
    return padded[1:-1, 1:-1] == 1                        # paper the flood never reached


def _enclosed_on_card(mask_d, padded, cross):
    """`enclosed_regions` for a mask that is already on the card.

    The same question asked the same way -- label the paper, keep what does
    not carry the edge's label -- with nothing crossing the bus. `padded` and
    `cross` are the caller's, so a loop allocates them once instead of once a
    pass. The border of `padded` is True from the moment it is made and the
    interior assignment never reaches it, so it stays paper.
    """
    import cupyx.scipy.ndimage as cnd

    padded[1:-1, 1:-1] = ~mask_d
    marked, _how_many = cnd.label(padded, structure=cross)
    inner = marked[1:-1, 1:-1]
    held = (inner != marked[0, 0]) & (inner != 0)
    del marked, inner
    return held


def recover_interiors(mask, ink):
    """Give back the enclosed regions that the page says are ink.

    The user's instruction, in the user's words: "just outline, which only remains the
    topological region ... then compare with [the original], just decide which
    is which, is hole or ink". So: take what the outline encloses, and keep the
    part of it the page itself calls ink. A real counter -- the hole in an `o`
    -- is paper in the page, so it stays a hole and nothing fills it.

    Run to a fixed point: giving ink back can enclose further regions, so one
    pass is an arbitrary stopping place -- and it made the result depend on
    how many times the stage happened to be run (the search ran it, then the
    pipeline ran it again, and they disagreed by eleven strokes).

    This runs before `min_area`, so a ring that becomes solid is measured at
    its true area rather than at the area of its own outline.
    """
    if gpu.GPU_AVAILABLE:
        try:
            return _recover_on_card(mask, ink)
        except Exception:
            pass
    given_back = 0
    for _pass in range(MAX_RECOVER_PASSES):
        enclosed = enclosed_regions(mask)
        if not enclosed.any():
            break
        recovered = enclosed & ink
        if not recovered.any():
            break
        mask = mask | recovered
        given_back += int(recovered.sum())
    return mask, given_back


def _recover_on_card(mask, ink):
    """The same fixed point with the page staying where it is.

    `enclosed_regions` already labels on the card, but it was given a host
    mask and handed back a host answer -- so each pass carried 173 MB up and
    173 MB down, and the three array operations between the passes ran on the
    host over the whole page. The loop is the same loop; what changes is that
    the page crosses the bus twice instead of twice a pass.

        as it was        0.694s
        resident         0.272s
        the same mask, pixel for pixel, and the same 2,620,350 px given back

    The user's words, and this is the plainest case of them on the page:
    "中间数组不得无故下载".
    """
    xp = gpu.xp
    mask_d = xp.asarray(mask)
    ink_d = xp.asarray(ink)
    cross = xp.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    padded = xp.ones((mask.shape[0] + 2, mask.shape[1] + 2), xp.bool_)
    given_back = 0
    try:
        for _pass in range(MAX_RECOVER_PASSES):
            enclosed = _enclosed_on_card(mask_d, padded, cross)
            if not bool(enclosed.any()):
                break
            recovered = enclosed & ink_d
            del enclosed
            if not bool(recovered.any()):
                break
            mask_d = mask_d | recovered
            given_back += int(recovered.sum())
            del recovered
        return gpu.to_host(mask_d), given_back
    finally:
        del mask_d, ink_d, padded, cross
        xp.get_default_memory_pool().free_all_blocks()


SPECK_LIST_REACH = 3.0     # list a component up to this many times min_area:
                           # near the line is where the decision is, and a
                           # page has a quarter of a million components.


@dataclass
class Speck:
    """One mark stage 8 is deciding about, and what it decided."""
    key: tuple
    area: int
    kept: bool

    @property
    def at(self) -> tuple:
        """Where it is on the page, as (x, y)."""
        return self.key[0], self.key[1]


@dataclass
class SpeckChoice:
    """The user's overrides of `min_area`, in both directions.

    A dot may be the user's notation -- a decimal point, a degree sign, the dot of
    an i that the pen barely touched -- or it may be a slip of the pen, and
    no area threshold can tell those apart. So the threshold proposes and
    this disposes, per mark.
    """
    keep: set = field(default_factory=set)
    drop: set = field(default_factory=set)

    @property
    def empty(self) -> bool:
        return not self.keep and not self.drop


def _speck_key(stats):
    """A name for a mark that survives a change to an earlier setting."""
    return (int(stats[cv2.CC_STAT_LEFT]), int(stats[cv2.CC_STAT_TOP]),
            int(stats[cv2.CC_STAT_WIDTH]), int(stats[cv2.CC_STAT_HEIGHT]))


def perfect_image(mask, img_raw, profile, background_bgr=None, stages=None,
                  specks=None):
    """Stage one's handover: which pixels are ink, and what the page allows.

    Everything that can *throw ink away* happens here and nowhere else, so
    that what follows only has to turn ink into strokes. Every tracing mode
    gets the same handover -- the modes differ in how they draw, never in
    what they believe is ink.

    Returns `(mask, allowed, dropped)`:
      mask     the ink, after the enclosed regions the page calls ink are
               given back and after the specks are dropped
      allowed  the page's own ink with a pixel of slack -- the one yardstick
               both directions of the correction answer to. None when the
               paper colour could not be trusted (see the guard below)
      dropped  what `min_area` removed, so the stage can *show* it rather
               than make it vanish: a dot may be the user's notation or a slip of
               the pen, and only the user can tell which

    The order matters. Interiors come back *before* `min_area`, or a ring
    about to become solid is weighed at the area of its own outline and
    thrown away as a speck.
    """
    ink = page_ink(img_raw, background_bgr) if profile.correct_against_page else None
    if ink is not None and profile.recover_interiors:
        before = mask
        mask, _given_back = recover_interiors(mask, ink)
        if stages is not None:
            stages('Holes to ink', {'mask': mask, 'added': mask & ~before})
    elif stages is not None:
        stages('Holes to ink', {'mask': mask, 'added': np.zeros_like(mask)})

    allowed = None
    if ink is not None and mask.any():
        allowed = cv2.dilate(ink.view(np.uint8), np.ones((3, 3), np.uint8)) > 0
        agreed = mask & allowed
        # A disagreement this large is not overshoot, it is a paper colour
        # read wrongly -- on which the page would be erased. Keep the mask,
        # and with it the correction, out of it.
        if agreed.sum() >= 0.5 * mask.sum():
            mask = agreed
        else:
            allowed = None

    # Specks last, so "how big is this mark" is asked of the mark the page
    # agrees to rather than of the threshold's first draft. It also keeps one
    # order for everybody: autotune scores the mask at every speck size at
    # once, and it can only do that if the sizes are measured on the same
    # thing the pipeline will measure them on.
    dropped = np.zeros_like(mask)
    candidates = []
    if profile.min_area > 0 or specks is not None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.view(np.uint8),
                                                                   connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA]
        keep = np.zeros(count, bool)
        keep[1:] = areas[1:] >= profile.min_area
        if specks is not None:
            # The user's say overrides the threshold in both directions. Keyed by
            # where the mark is and how big its box is, not by component
            # index: the index moves the moment an earlier setting changes,
            # and a decision that forgets itself when a slider moves is not
            # a decision.
            for i in range(1, count):
                key = _speck_key(stats[i])
                if key in specks.keep:
                    keep[i] = True
                elif key in specks.drop:
                    keep[i] = False
        # Asked of the ink rather than of the page. `keep[labels]` reads the
        # whole 691 MB label image and writes a 173 MB answer for a question
        # about the 2% of it that is ink: 0.506s against 0.145s, the same
        # array out. (On the card it is 0.006s, and uploading the label image
        # is 0.100s -- so the card does not pay here until the labelling
        # happens there too, which is a different piece of work.)
        kept = geometry.pick(keep, labels, mask)
        dropped = mask & ~kept
        mask = kept
        # Only the marks near the line are worth listing: a page has
        # hundreds of thousands of components and the user is deciding about the
        # ones the threshold is unsure of, not about the words.
        edge = float(profile.min_area) * SPECK_LIST_REACH
        for i in range(1, count):
            area = int(areas[i])
            if area <= max(edge, 1.0):
                candidates.append(Speck(key=_speck_key(stats[i]), area=area,
                                        kept=bool(keep[i])))
        candidates.sort(key=lambda s: s.area)
    if stages is not None:
        stages('Specks', {'mask': mask, 'dropped': dropped,
                          'candidates': candidates,
                          'thrown': int(np.count_nonzero(dropped))})
        stages('Perfect image', {'colour': ink_colours(mask, img_raw)})
    return mask, allowed, dropped


def ink_colours(mask, img_raw):
    """The perfect image as a picture: the mask's own pixels, in the page's
    own colours, white everywhere else.

    The *shape* is entirely the mask's -- only the colour comes from the page.
    A soft edge in this preview means those edge pixels are in the mask, not
    that ink is somehow half-present there.
    """
    view = np.full_like(img_raw, 255)
    view[mask] = img_raw[mask]
    return view


def colour_field(mask, img_raw, profile, background_bgr=None, pen_width_px=None,
                 stages=None):
    """Stage 11: the page's colour, as a 5-D clustering of the ink.

    The clustering runs in (L, a, b, x, y) at the page's own resolution, and
    the clusters are then gathered into pens by a mean shift in direction
    space -- a pen is a ray out of the paper, not a ball in Lab, which is why
    an isotropic merge used to report fifteen pens for a page written in
    three. How many pens a page has is a fact about the page: with
    `colour_bandwidth` at 0 the page is asked, against the user's 0.3% target.

    The clusters themselves stop here. What leaves this stage is one colour
    per ink pixel with the cluster forgotten -- the guard against the
    clustering's own tiling printing itself onto the writing.
    """
    pen = pen_width_px or centerline.pen_width(mask)
    field = color5d.colour_map(mask, img_raw,
                               max(1.0, float(profile.space_bandwidth) * pen),
                               float(profile.colour_bandwidth) or None, background_bgr)
    if stages is not None:
        # One picture per decision rather than one for all three: what the
        # clustering was handed, how it grouped it, and what it called each
        # group. A setting that moves any of them has to be watchable where
        # it moves.
        stages('Colour cloud', {'colour': color5d.cloud_preview(field)})
        stages('Colour clusters', {'colour': color5d.cluster_preview(field)})
        stages('Colours', {'colour': color5d.preview(field), 'field': field})
    return field


def region_map(mask, img_raw, profile, background_bgr=None, pen_width_px=None,
               field=None, correction=None, stages=None):
    """Stages 10 to 13, as arrays rather than as objects.

    Returns `(labels, colours, field)`:

      labels   which colour each ink pixel is, as an index; -1 off the ink
      colours  {index: colour}, what the Region-Colour Map maps to
      field    the ColourField, or None when the colour pass is off

    **Why not one object per piece of ink.** It was 121,064 of them on the user's
    page once the clustering's own colours were obeyed, and building them
    cost 27 seconds before a single stroke was drawn. The user's design never asked
    for them: the outlines are the geometry pattern, the clustering gives the
    colour, and splitting the area by colour is what a label image *is*. A
    label image does not grow with the number of pieces.

    The map is still total and single-valued, and now by construction in the
    strongest sense: an array has exactly one value per pixel.
    """
    contours, hierarchy = cv2.findContours(mask.view(np.uint8), cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if stages is not None:
        # Stage 10. The outlines, and nothing counted: an inner contour *is*
        # a hole, so "does this 'o' still have its hole" is answered by
        # looking at the picture rather than by a number beside it. The
        # component inventory and the per-piece hole count that used to be
        # here were this author's addition, not requested, and were removed.
        inner = int((hierarchy[0][:, 3] >= 0).sum()) if hierarchy is not None else 0
        stages('Geometry', {'colour': _outline_picture(mask.shape, contours, hierarchy),
                            'outlines': len(contours), 'holes': inner})
    if not profile.colour_5d:
        labels = np.where(mask, 0, -1).astype(np.int32)
        return labels, {0: None}, None
    if field is None:
        field = colour_field(mask, img_raw, profile, background_bgr, pen_width_px, stages)
    if not field.colours:
        return np.full(mask.shape, -1, np.int32), {}, field

    labels = regions.colour_index(field, mask)
    colours = {i: tuple(int(v) for v in bgr)
               for i, bgr in enumerate(field.colours_bgr)}
    if stages is not None:
        stages('Region colour map', {'colour': regions.paint(labels, field.colours_bgr),
                                     'labels': labels, 'field': field,
                                     'colours': len(colours)})
    dropped = recoloured = 0
    if correction is not None and not correction.empty:
        labels, dropped, recoloured = correction.apply(labels)
    if stages is not None:
        stages('Region correction', {'colour': regions.paint(labels, field.colours_bgr),
                                     'labels': labels, 'field': field,
                                     'dropped': dropped, 'recoloured': recoloured})
    return labels, colours, field


def _outline_picture(size, contours, hierarchy):
    """Stage 10 as a picture: the outlines, with the holes' rims told apart.

    An outer rim in ink, an inner one in the colour that means "given back",
    because a hole is paper the ink encloses and that is the same fact stage
    7 is about.
    """
    view = np.full(tuple(size) + (3,), 255, np.uint8)
    if not contours:
        return view
    parents = hierarchy[0][:, 3] if hierarchy is not None else np.full(len(contours), -1)
    outer = [c for c, parent in zip(contours, parents) if parent < 0]
    inner = [c for c, parent in zip(contours, parents) if parent >= 0]
    cv2.drawContours(view, outer, -1, (30, 30, 30), 1)
    cv2.drawContours(view, inner, -1, HOLE_RIM, 1)
    return view


def encode_regions(labels, colours, mask, img_raw, profile, background_bgr=None,
                   pen_width_px=None, stages=None, cancel=None):
    """Stages 14 and 15 for the whole page, in one pass.

    `ortho-fill` grows in from the boundary and obeys the colour point by
    point inside the march, so there is nothing to run per piece. The three
    older encoders are kept and still run per colour, because that is the
    only way they have of knowing what colour they are drawing -- there are
    6,534 colours now, so that path is for small pages and comparisons.
    """
    pen = pen_width_px or centerline.pen_width(mask)
    if profile.trace == 'ortho-fill':
        field = geometry.signed_distance(mask)
        if stages is not None:
            stages('Medial axis', {'colour': _field_picture(field * mask),
                                   'deepest': float(field.max())})
        out = orthofill.fill_page(
            mask, labels, colours, Segment, pen=pen, field=field,
            spacing=profile.rib_spacing, line_width=profile.line_width,
            max_coord_str_len=profile.max_coord_str_len, cancel=cancel,
            merge_gap=_merge_px(profile, pen),
            merge_colour=float(profile.merge_colour),
            max_points=render.path_limit(profile.max_coord_str_len))
        if stages is not None:
            stages('Orthogonal filling', {'segments': out})
        return out

    # The three older encoders run **once over the whole ink**, and each
    # stroke then reads its colour out of the label image at its own points.
    #
    # They used to be run once per colour, which was fine while a page had
    # seven pens and is wrong now that it has the clustering's own six and a
    # half thousand: each pass saw a sliver of a letter, and `centerline`
    # measured the pen's width on the sliver -- 1.1 px where the text was
    # drawn at 10.
    #
    # The honest limitation, and the reason `ortho-fill` is the default: a
    # stroke these produce can cross a colour boundary, because they decide
    # their geometry before anything asks about colour. All that can be done
    # then is to give it the colour it mostly sits on. `ortho-fill` does not
    # have the problem at all -- it obeys the boundary while it grows.
    coloured = any(c is not None for c in colours.values())
    quiet = replace(profile, color_break=0.0) if coloured else profile
    out = []
    height, width = mask.shape
    for segment in shape_segments(mask, img_raw, quiet, background_bgr, pen):
        colour = segment.color
        if coloured:
            points = np.asarray(segment.points, np.int64)
            ys = np.clip(points[:, 1], 0, height - 1)
            xs = np.clip(points[:, 0], 0, width - 1)
            seen = labels[ys, xs]
            seen = seen[seen >= 0]
            if len(seen):
                # Whichever colour it mostly stands on. A stroke that
                # straddles two gets one of them; that is the cost of
                # deciding the shape first, which is what this path does.
                index = int(np.bincount(seen).argmax())
                colour = colours.get(index, colour)
        out.append(Segment(segment.points, colour, segment.width, segment.fill))
    out = _merge_nearby(out, geometry.signed_distance(mask), profile, pen)
    if stages is not None:
        stages('Orthogonal filling', {'segments': out})
    return out


def _merge_px(profile, pen):
    """The user's `merge_gap`, in pixels.

    The user sets it in pen widths because that is the scale a person reads a page
    at: "about a pen width apart" is something the user can see, where a number of
    pixels at 800 DPI is not.
    """
    return float(getattr(profile, 'merge_gap', 0.0)) * max(1.0, float(pen))


def _merge_nearby(segments, field, profile, pen):
    """The same last step for the three older encoders, which have no
    residual-dot step of their own to run after it."""
    gap = _merge_px(profile, pen)
    if gap <= 0.0 or len(segments) < 2:
        return segments
    return strokejoin.join(segments, field, Segment, gap,
                           float(getattr(profile, 'merge_colour', 0.0)),
                           render.path_limit(profile.max_coord_str_len),
                           default_width=profile.line_width)


def _field_picture(field):
    """The distance field as something to look at: dark where the ink is deep.

    Greyscale rather than a colour ramp -- the number it carries is a radius,
    and a ramp would invite reading a boundary into a smooth quantity.
    """
    top = float(field.max()) or 1.0
    grey = 255 - np.clip(field / top, 0.0, 1.0) * 225.0
    return np.repeat(grey.astype(np.uint8)[:, :, None], 3, axis=2)


# <~ _shift moved a region's strokes from its crop back onto the page.
#    Nothing is cropped any more: the page is filled in one pass. ~>


def validate_output(segments, mask, profile, labels=None, field=None, stages=None):
    """Stage 16. The one stage the user would not let go of.

    It is the only measurement that catches the whole of phase 3 at once: if
    the axis came out cut, the ribs mis-angled, or a stroke dropped to
    satisfy the format's two-point minimum, ink goes bare and this says so in
    pixels. Colour is judged in the same breath, against the page rather than
    against the colour map -- a metric made out of the thing it judges cannot
    be wrong, and that mistake once slid the search down to a single pen.
    """
    report = orthofill.validate(segments, mask, profile.line_width,
                                profile.max_coord_str_len)
    wrong, share = None, 0.0
    if field is not None and segments:
        share, wrong = colour_error(segments, field, (mask.shape[1], mask.shape[0]),
                                    profile.line_width, profile.max_coord_str_len)
    if stages is not None:
        stages('Output validation', {'segments': segments, 'report': report,
                                     'wrong': wrong, 'share': share,
                                     'passed': report.passed()})
    return report


COLOUR_ERROR_TOLERANCE = 20.0   # in color5d's direction space (radius 100):
                                # about 11 degrees


def colour_error(segments, field, size, line_width=1.0, max_coord_str_len=30000,
                 tolerance=COLOUR_ERROR_TOLERANCE):
    """Where the tracing drew a colour the page does not have there.

    Asked at the strokes' own points, all of them at once. The question is
    whether a stroke runs through ink of the colour it was given, and its
    points are where it runs; rasterizing the page once per pen to ask the
    same thing cost fifteen seconds on the user's page, as much as the clustering
    itself, for a number that is only ever looked at.

    The page's colour is read **at those points only**. Converting the whole
    page to Lab to answer a question about thirty thousand strokes allocates
    two gigabytes at 800 DPI and takes eight seconds; the points are a few
    hundred thousand pixels and take a tenth of one.

    Against the page, never against the colour map: the map is made at
    whatever setting is being judged, so a setting wide enough to call every
    pen on the page one colour would agree with itself perfectly and score
    zero. A metric that cannot be wrong is no metric.

    Returns `(share, wrong)`: the share of stroke points on ink whose colour
    is not the one beneath them, and the segments that are mostly wrong.
    """
    width, height = size
    if field is None or field.source_bgr is None or not segments:
        return 0.0, []
    kept = [s for s in segments if len(np.asarray(s.points))]
    if not kept:
        return 0.0, []

    counts = np.fromiter((len(np.asarray(s.points)) for s in kept), np.int64, len(kept))
    points = np.concatenate([np.asarray(s.points) for s in kept])
    owner = np.repeat(np.arange(len(kept)), counts)
    xs = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, width - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, height - 1)

    on_ink = field.labels[ys, xs] >= 0
    if not on_ink.any():
        return 0.0, []
    xs, ys, owner = xs[on_ink], ys[on_ink], owner[on_ink]

    here = color5d.directions_of_pixels(field.source_bgr[ys, xs], field.paper_lab)
    colours = [s.color for s in kept]
    index = {}
    for colour in colours:
        index.setdefault(colour, len(index))
    table = np.zeros((len(index), 3), np.float32)
    missing = np.zeros(len(index), bool)
    for colour, slot in index.items():
        if colour is None:
            missing[slot] = True
        else:
            table[slot] = color5d.direction_of(colour, field.paper_lab)
    slots = np.fromiter((index[c] for c in colours), np.int64, len(colours))[owner]

    apart = np.linalg.norm(here - table[slots], axis=1)
    bad_point = (apart > tolerance) | missing[slots]
    total = len(bad_point)
    per_stroke = np.bincount(owner, weights=bad_point.astype(np.float64), minlength=len(kept))
    seen = np.bincount(owner, minlength=len(kept))
    mostly = np.flatnonzero(per_stroke > np.maximum(seen, 1) * 0.5)
    return (float(bad_point.sum()) / total if total else 0.0), [kept[i] for i in mostly]


def _bgr_to_hex(bgr) -> str:
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return "#{:02X}{:02X}{:02X}FF".format(r, g, b)


# **What a pixmap may hold, measured on the user's own TEST.pdf (1024 x 1366 pt).**
#
#   996 dpi   267.7 MP   renders, with ink on it
#  1000 dpi   269.8 MP   MuPDF prints "limit error: Overly large image" to
#                        stderr, hands back a pixmap that is *uniformly 255*,
#                        and **does not raise**
#  1200 dpi   388.6 MP   raises FzErrorLimit
#
# So the cap is on the pixel count, and it is exactly 2**28. The white pixmap
# is the dangerous one: the pipeline traced 0 strokes from it and the window
# showed two blank panes, which is what the user photographed on 2026-10-27 --
# "why i cnats ee shit after i ajdustthe dpi and enter tor erun".
#
# It is a count, not a resolution, so this is **not** a DPI constant: how high
# a page may go depends on how big it is, and a small page goes far higher.
# `max_dpi_for` is the one place that arithmetic lives.
MAX_PIXMAP_PIXELS = 1 << 28


class PageTooLargeToRender(ValueError):
    """Asked for more pixels than a pixmap can hold.

    Raised instead of returning the white image MuPDF gives back: a blank
    page that passes for a rendered one is the worst of the three outcomes,
    because it looks like the converter is broken rather than like the
    request was impossible.
    """


# **How many pixels one page may become, in total.** MuPDF refuses a single pixmap above
# `MAX_PIXMAP_PIXELS`, but that is its limit, not the page's: the page is rendered in tiles and
# pasted together, so what actually bounds the resolution is the memory the array needs. 2**29
# pixels is 512 M bytes at one channel and 1.5 G at three -- and it puts the user's 1024 x 1366 pt page
# at about 1,400 dpi instead of the 997 a single pixmap allowed, which is what the user asked for
# ("i want it can reach at least 1117, 997 is not good").
MAX_PAGE_PIXELS = 1 << 29


def max_dpi_for(width_pts: float, height_pts: float) -> int:
    """The highest whole DPI this page size can be rendered at.

    From the pixel cap and nothing else:

        (w/72 * dpi) * (h/72 * dpi) <= MAX_PIXMAP_PIXELS
        dpi <= 72 * sqrt(cap / (w * h))

    Floored, so the answer always fits, and never below 72: a page too big
    even for that would be fifty metres across, and offering no resolution at
    all would be worse than offering one that still renders.
    """
    area = max(float(width_pts) * float(height_pts), 1e-9)
    dpi = int(math.floor(72.0 * math.sqrt(MAX_PAGE_PIXELS / area)))
    return max(72, dpi)


def usable_dpi(page: "pymupdf.Page", requested: int):
    """`(dpi, note)` -- the DPI to actually use for this page, and why.

    Call this **before** rendering and before `page_geometry`, so the render
    and the geometry agree on one number. `note` is None when the request was
    fine, and otherwise a sentence to put in front of the user: the user asked for
    something the page cannot hold, and the honest answer is the highest it
    can take plus the reason, not a blank pane and not silence.
    """
    rect = page.rect
    top = max_dpi_for(rect.width, rect.height)
    if requested <= top:
        return int(requested), None
    return top, ('%d dpi is more than this %.0f x %.0f pt page can be '
                 'rendered at; using %d dpi, the most a pixmap holds'
                 % (requested, rect.width, rect.height, top))


# Each tile is rendered with this much extra on every side and then cropped
# back, because MuPDF antialiases a stroke against the **clip** boundary, not
# only against the page. A stroke crossing a join was therefore antialiased
# twice, once from each side, and the two halves did not agree. Measured on
# the user's TEST.pdf at 800 dpi, tiled against a single-pixmap render of the same
# page -- worst pixel difference, and how many pixels were off by more than 2:
#
#   margin        0 px    16 levels,  338 pixels
#   margin        2 px    13 levels,  292 pixels
#   margin        8 px     7 levels,   70 pixels
#   margin       32 px     3 levels,    5 pixels
#   margin       64 px     0 levels,    0 pixels   <- identical
#
# (Before the crop was anchored on `pixmap.x` the same comparison was 131
# levels across 389 pixels, all within 25 px of the one vertical join.)
#
# 64 px of border on a tile of roughly 8000 px costs about 3% more area and
# measured 1.92 s against 1.79 s for the single render -- the same page, to
# the pixel, for a twentieth of a second.
TILE_MARGIN_PX = 64


def _paste_tiles(page, dpi: int, width: int, height: int) -> np.ndarray:
    """Render a page in tiles and paste them into one array.

    Each tile stays under MuPDF's single-pixmap cap; the whole page may exceed it. Without this
    the highest resolution was whatever one pixmap held -- 997 dpi for the user's 1024 x 1366 pt page --
    and asking for more got a clap on the ceiling rather than the pixels the user asked for.

    **The tile boundaries are the pasted ones.** The first version sized the canvas with
    `round(w * zoom)` and placed tiles at `round(column * span * scale)`, which differ by a pixel
    and made the last column overflow: "could not broadcast input array from shape (5436, 5218, 3)
    into shape (5436, 5217, 3)". Every boundary is now one number used both to clip and to paste.
    """
    scale = dpi / 72.0
    tile_px = int(math.sqrt(MAX_PIXMAP_PIXELS)) // 2          # a quarter of the cap per tile
    columns = max(1, math.ceil(width / tile_px))
    rows = max(1, math.ceil(height / tile_px))
    xs = [round(width * i / columns) for i in range(columns + 1)]
    ys = [round(height * i / rows) for i in range(rows + 1)]
    canvas = None
    for row in range(rows):
        for column in range(columns):
            x0, x1 = xs[column], xs[column + 1]
            y0, y1 = ys[row], ys[row + 1]
            # Rendered wider than it is kept, so the antialiasing that MuPDF
            # does against the clip edge falls in the part that is thrown away.
            gx0 = max(0, x0 - TILE_MARGIN_PX)
            gx1 = min(width, x1 + TILE_MARGIN_PX)
            gy0 = max(0, y0 - TILE_MARGIN_PX)
            gy1 = min(height, y1 + TILE_MARGIN_PX)
            clip = pymupdf.Rect(page.rect.x0 + page.rect.width * gx0 / width,
                                page.rect.y0 + page.rect.height * gy0 / height,
                                page.rect.x0 + page.rect.width * gx1 / width,
                                page.rect.y0 + page.rect.height * gy1 / height)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip)
            tile = np.frombuffer(pixmap.samples, dtype=np.uint8)
            tile = tile.reshape(pixmap.height, pixmap.width, pixmap.n)
            if canvas is None:
                canvas = np.zeros((height, width, tile.shape[2]), dtype=np.uint8)
            # **Where MuPDF put the tile, not where the fractions said.** The clip is in points
            # and the pixmap's own integer rect is where its first pixel really is
            # (`pixmap.x`, `pixmap.y`); pasting at the computed boundary instead left a
            # one-pixel seam -- measured, a difference of 195 levels along the join. The same
            # number is what the margin is cropped against, so the kept piece is the interior
            # of what was drawn whatever rounding the clip went through.
            start_x = x0 - int(getattr(pixmap, 'x', gx0))
            start_y = y0 - int(getattr(pixmap, 'y', gy0))
            if start_x < 0 or start_y < 0:
                # Should not happen -- the clip begins at or before the kept
                # region -- but a negative index would silently wrap and paste
                # the wrong end of the tile.
                start_x, start_y = max(0, start_x), max(0, start_y)
            piece = tile[start_y:start_y + (y1 - y0), start_x:start_x + (x1 - x0)]
            h = min(piece.shape[0], height - y0)
            w = min(piece.shape[1], width - x0)
            if h > 0 and w > 0:
                canvas[y0:y0 + h, x0:x0 + w, :tile.shape[2]] = piece[:h, :w]
    return canvas


def render_pdf_page(page: "pymupdf.Page", dpi: int,
                    in_slices: bool = False) -> np.ndarray:
    """Render one PDF page to a BGR uint8 array at the given DPI.

    Raises `PageTooLargeToRender` rather than handing back the blank page
    MuPDF returns past its pixel cap -- see `MAX_PIXMAP_PIXELS`. A caller that
    wants to carry on anyway clamps with `max_dpi_for` **before** working out
    the geometry: `fit_geometry` is given the same DPI, so clamping quietly in
    here would put the ink in the wrong place.

    `in_slices` renders through `_paste_tiles` even when the page would fit a
    single pixmap. **Nothing about the result changes** -- the two paths are
    pixel for pixel identical, which `test_the_ceiling_is_honest` now checks
    at four page sizes -- what changes is that MuPDF lets go of the GIL
    between tiles. The user's complaint: *"why always stuck when I enter and just
    Ctrl+S"*. Measured on the user's page at 800 dpi, rendering on a worker thread
    while the main thread ticks:

        one pixmap        2.30 s, the main thread frozen for 1.154 s
        in slices         2.15 s, the main thread frozen for 0.457 s

    It costs one extra full-page copy -- the tiles are pasted into a canvas
    and then flipped to BGR -- so it is asked for rather than assumed. A
    document conversion already holds several arrays this size and takes
    twenty seconds; a preview is a different trade and does not ask.
    """
    rect = page.rect
    zoom = dpi / 72.0
    # **Rounded up, because that is what MuPDF does.** The comment that stood
    # here said the opposite -- "truncated, not rounded" -- and cited
    # 187.8236 pt at 2000 dpi as giving a 5217-wide pixmap. Measured, that
    # page gives **5218**. Every case measured rounds up, with no exception:
    #
    #   page pt    dpi    exact        int     MuPDF
    #   187.8236   200    521.732      521     522
    #   187.8236   2000   5217.322     5217    5218
    #   1024.0000  200    2844.444     2844    2845
    #   1024.0000  800    11377.778    11377   11378
    #   391.3654   800    4348.504     4348    4349
    #
    # It mattered in one place and that place is the one that cannot be
    # checked against a plain render: past `MAX_PIXMAP_PIXELS` the page is
    # built by `_paste_tiles`, which is handed this width and height. So the user's
    # 1117 dpi page -- the resolution the user asked for by name -- came out
    # 15886 x 21191 where MuPDF's own render would be 15887 x 21192: one row
    # and one column of the page's edge missing, every time.
    #
    # Below the cap nothing changes: that path returns MuPDF's own pixmap and
    # never consults these two numbers except to compare against the cap.
    width = max(1, math.ceil(rect.width * zoom))
    height = max(1, math.ceil(rect.height * zoom))
    total = width * height
    top = max_dpi_for(rect.width, rect.height)
    if dpi > top:
        raise PageTooLargeToRender(
            '%d dpi needs %.0f MP for this %.0f x %.0f pt page, and one page may '
            'hold %.0f MP, so %d dpi is the most it can take'
            % (dpi, total / 1e6, rect.width, rect.height,
               MAX_PAGE_PIXELS / 1e6, top))
    if total <= MAX_PIXMAP_PIXELS and not in_slices:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return img[:, :, ::-1].copy()  # RGB -> BGR
    # **Past one pixmap: tiles.** The user's order: "i want it can reach at least 1117, 997 is not good"
    # -- 997 was MuPDF's per-pixmap ceiling for the user's page, not the page's.
    tiled = _paste_tiles(page, dpi, width, height)
    return tiled[:, :, ::-1].copy()  # RGB -> BGR


def geometry_for(pdf_width_pts: float, pdf_height_pts: float, dpi: int,
                 target_width: float, target_height: float, policy=None):
    """The page geometry, with the Q base calibration folded in when it applies.

    `fit_geometry` fits the page to the target and centres it. That is right for a
    page already on the sheet, and it is what the reference output was made with. When
    `qbasefit` has a **complete border** to trust, its scale and offset replace the fit
    -- the border lands on the writing area, and the content with it.

    The pipeline carries **one** scale, so a non-uniform calibration (unequal sx and
    sy) cannot be expressed here: that returns `(None, reason)` rather than quietly
    using one axis, which would be the kind of half-alignment this mission exists to
    stop.

    Returns `((scale, dx, dy), reason)`; `scale` is `None` when it cannot be applied.
    """
    scale, dx, dy = fit_geometry(pdf_width_pts, pdf_height_pts, dpi,
                                 target_width, target_height)
    if policy is None or not getattr(policy, 'applies', False):
        return (scale, dx, dy), 'fit to page'
    sx = float(policy.sx)
    sy = float(policy.sy)
    if abs(sx - sy) > 0.02:
        return None, ('non-uniform calibration (%.3f x %.3f) is not supported by '
                      'the pipeline yet' % (sx, sy))
    # The calibration maps page points; `scale` maps rendered pixels, so the pdf's
    # 72 points per inch enters once.
    return (72.0 / (dpi * sx), policy.dx, policy.dy), 'calibrated from the border'


def fit_geometry(pdf_width_pts: float, pdf_height_pts: float, dpi: int,
                  target_width: float, target_height: float):
    """Scale/offset to map a PDF page rendered at `dpi` onto a
    (target_width, target_height) xopp page, fit-to-page and centered.

    This replaces the original scripts' per-document hand-tuned SCALE/DX/DY
    (e.g. `DPI / 15`, `DPI / 13`): those were eyeballed approximations of
    exactly this fit for whatever single PDF was being converted at the
    time. When a source PDF's own page size already equals the target xopp
    page (true for anything built from this archive's own page templates),
    this reduces to the exact `scale = dpi / 72` relationship (since PDF
    points are 1/72 inch) with zero offset.
    """
    px_w = pdf_width_pts / 72.0 * dpi
    px_h = pdf_height_pts / 72.0 * dpi
    scale = max(px_w / target_width, px_h / target_height)
    dx = (target_width - px_w / scale) / 2.0
    dy = (target_height - px_h / scale) / 2.0
    return scale, dx, dy


@dataclass
class Segment:
    """One stroke to draw: points in image pixels, a palette colour name, the
    width the ink had (image pixels; None means the profile's own), and
    whether the path's interior is filled (Xournal++'s `fill`, 0-255)."""
    points: np.ndarray
    color: str
    width: float = None
    fill: int = None


def _center_segments(mask, img_raw, profile, background_bgr, pen_width_px, stages=None):
    """Centrelines, each coloured from the ink it runs through.

    Sampling colour along the middle of a stroke is what makes pale ink keep
    its colour: on the boundary, where contour tracing samples, every pixel
    is a blend with the paper and a light blue hairline reads as grey.
    """
    if not mask.any():
        return []
    # Measured from this mask, not from the caller's page-wide estimate: the
    # reduction the tracer picks depends on it, and being wrong by 2x means
    # thinning at the wrong scale.
    traced = centerline.trace(mask)
    if not traced:
        return []
    if profile.colour_5d:
        # Shape only. The colour pass runs once for the page, afterwards.
        return [Segment(points, None, stroke_width) for points, stroke_width in traced]

    height, width = img_raw.shape[:2]
    lengths = np.fromiter((len(points) for points, _ in traced), np.int64, len(traced))
    points = np.concatenate([p for p, _ in traced])
    owner = np.repeat(np.arange(len(traced)), lengths)
    rows = np.clip(points[:, 1].astype(np.int64), 0, height - 1)
    columns = np.clip(points[:, 0].astype(np.int64), 0, width - 1)
    colors = img_raw[rows, columns]
    if background_bgr is not None and not np.all(np.array(background_bgr) >= 250):
        divisor = np.maximum(np.array(background_bgr, np.float64), 1.0)
        colors = np.clip(colors.astype(np.float64) / divisor * 255.0, 0, 255).astype(np.uint8)
    classes, names = classify_bgr(profile, colors)
    # Each stroke takes the colour most of its own ink is...
    votes = np.zeros((len(traced), len(names)), np.int32)
    np.add.at(votes, (owner, classes), 1)
    chosen = votes.argmax(axis=1)

    # ...unless the ink under it changes colour along the way, in which case
    # it is cut where it changes. Two pens that cross are one shape, so one
    # path, and one vote would give the whole crossing to whichever pen has
    # more of it on the page.
    min_run = _min_color_run(profile, _traced_pen(mask, pen_width_px))
    segments = []
    starts = np.concatenate(([0], np.cumsum(lengths)))
    for i, (points, stroke_width) in enumerate(traced):
        pieces = None
        if min_run is not None:
            pieces = split_by_color(points, classes[starts[i]:starts[i + 1]], names, min_run)
        if pieces is None:
            segments.append(Segment(points, names[chosen[i]], stroke_width))
        else:
            segments.extend(Segment(piece, name, stroke_width) for piece, name in pieces)
    return segments


def _min_color_run(profile, pen_width_px):
    """How many points a piece of one colour must have to be a pen of its own.

    None when splitting is off. The scale is the pen: a piece shorter than
    the pen is wide is noise where two strokes touch, not a change of pen --
    the same rule the outline tracer uses, in points rather than in contour
    length.

    The pen has to be measured on **the mask being traced**, not on the
    probe's own ink mask. The probe counts anything far from the paper
    colour, anti-aliased halo included; the tracer works on a thresholded
    mask that keeps only the dark core. On a phone scan those differ by 8x
    (15.7px against 1.9px), and a rule sized to the wrong one asks a
    crossing to be twenty-five points long before it counts -- which no
    crossing ever is, so nothing was ever cut.
    """
    if profile.color_break <= 0:
        return None
    strength = min(max(profile.color_break, 0.0), 10.0) / 10.0
    widest, narrowest = COLOR_BREAK_MIN_RUN
    pen = pen_width_px or DEFAULT_PEN_WIDTH_PX
    return max(2, int(round(pen * (widest + (narrowest - widest) * strength))))


def _traced_pen(mask, fallback=None):
    """The pen width of the ink actually being traced, in its own pixels."""
    if not mask.any():
        return fallback or DEFAULT_PEN_WIDTH_PX
    return max(1.0, centerline.pen_width(mask))



def _solid_segments(mask, img_raw, profile, background_bgr, pen_width_px, stages=None):
    """Reproduce the ink as area, not as an outline of it.

    Outline tracing draws the boundary of a blob and leaves the inside empty;
    centreline tracing draws one line down the middle at the pen's width,
    which is right for handwriting but cannot represent a filled shape. Both
    of these rebuild the region itself, so that re-rasterising the tracing
    gives back the page it came from:

    - `onion-fill` takes the region's own boundary and then the curves
      parallel to it going inwards, each one a level set of the page's signed
      distance field, so each is a single continuous closed curve and the
      shape's branching looks after itself (`_offset_fill`). The boundary is
      stroked thin and on its own.

      Then it *draws what it has decided so far, looks at it, and fills in
      whatever is still paper* -- repeatedly, until what is left is under
      `profile.fill_target`. That loop is the whole of the fill's fidelity,
      and it replaces trying to work out in advance what the curves would
      cover. Every attempt at that was a model of the renderer rather than
      the renderer, wrong by about a pixel somewhere different each time, and
      a pixel wrongly believed drawn is a pixel nothing ever draws: it left
      8.5% of the ink more than two pixels deep undrawn, which is writing you
      can see through. Measured the same way, the loop leaves 0.17%.
    """
    if not mask.any():
        return []
    allowed = None
    if profile.correct_against_page:
        ink = page_ink(img_raw, background_bgr)
        allowed = cv2.dilate(ink.view(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if (mask & allowed).sum() < 0.5 * mask.sum():
            allowed = None

    pen = pen_width_px or centerline.pen_width(mask)
    segments = []
    # onion-fill: the boundary of the cleaned region first, then fill what is
    # still paper -- and find out which parts those are by *drawing* what has
    # been decided so far and comparing it with the mask, not by modelling it.
    step = max(1, int(round(profile.dpi / 400)))
    # How short a piece of one colour may be before it is noise where two
    # strokes touch rather than a second pen. None when splitting is off.
    min_run = _min_color_run(profile, _traced_pen(mask, pen))
    height, width = mask.shape
    total = float(mask.sum())

    def drawn(pieces):
        return render.segment_ink(pieces, (width, height), profile.line_width,
                                  profile.max_coord_str_len)

    def top_up(segments, covered):
        """Draw what is still paper, until it is under the target.

        Each pass takes the ink the tracing is measured to have missed and
        draws it down its middle at its own width, which covers it and cannot
        spill outside it; whatever that pass still misses, the next one sees.
        It stops when it is close enough or when a pass stops making progress
        -- it cannot loop for ever on a sliver it keeps redrawing in the same
        wrong place.
        """
        target = max(0.0, float(profile.fill_target)) * total
        before = None
        for _ in range(MAX_FILL_PASSES):
            residue = mask & ~covered
            left = float(residue.sum())
            if left <= target or (before is not None and left >= before):
                break
            before = left
            added = _detail_strokes(residue, mask, img_raw, profile, background_bgr,
                                    step, min_run)
            if not added:
                break
            segments.extend(added)
            covered |= drawn(added)
        return covered

    segments.extend(_offset_fill(mask, img_raw, profile, background_bgr, pen, min_run,
                                 stages=stages))
    if stages is not None:
        stages('Onion rings', {'segments': list(segments)})
    covered = top_up(segments, drawn(segments))
    if stages is not None:
        stages('Fill loop', {'segments': list(segments), 'residue': mask & ~covered})
    if allowed is None or not profile.remove_invented:
        return segments

    # The correction runs both ways. Filling alone can only add, so a stroke
    # drawn where the page has no ink stays for ever -- and the user is right that
    # nothing has to guess about it: the tracing has been drawn, the page is
    # there to compare it with, and what sits outside the ink is known, not
    # estimated. So: drop those strokes, draw again, and let the fill see
    # whatever the removal opened up. Bounded, and it stops as soon as a round
    # removes nothing.
    limit = max(0.0, float(profile.extra_target)) * total
    room = geometry.signed_distance(allowed)
    for _ in range(MAX_CORRECT_ROUNDS):
        extra = float((covered & ~allowed).sum())
        if extra <= limit:
            break
        kept = _without_invented(segments, allowed, float(profile.invented_point_share))
        kept = _trim_overshoot(kept, room)
        corrected = drawn(kept)
        # Only keep the round if it really drew less outside. Trimming widths
        # can cost coverage, and a round that trades ink for nothing is worse
        # than no round at all.
        if float((corrected & ~allowed).sum()) >= extra:
            break
        segments = kept
        covered = top_up(segments, corrected)
    if stages is not None:
        stages('Reconcile', {'segments': list(segments),
                             'lost': mask & ~covered, 'invented': covered & ~allowed})
    return segments


def _trim_overshoot(segments, depth, floor=1.0, quantile=20.0):
    """Narrow each stroke to the room the ink actually gives it.

    The ink drawn outside the page's own is mostly not whole invented strokes
    -- it is a rim half a width wide along every real one, where the stroke
    was given the pen's width in a place the ink is narrower than that. No
    removal of strokes can take a rim off; only the width can.

    `depth` is the distance from each pixel to the paper, so a stroke centred
    on a point may be at most twice that wide before it spills. A low
    quantile along the stroke rather than the minimum: one point crossing a
    narrow neck should not thin the whole stroke to nothing, and whatever
    thinning does leave uncovered, the fill pass that follows draws again.

    Every stroke's points are gathered in one pass. Asking numpy for the
    depth under each stroke separately costs more in call overhead than in
    arithmetic -- on the user's page it was sixteen of the tuner's eighty seconds,
    for seven thousand strokes.
    """
    height, width = depth.shape
    kept = [np.asarray(s.points) for s in segments]
    sizes = np.fromiter((len(points) for points in kept), np.int64, len(kept))
    movable = np.fromiter((s.width is not None and s.fill is None for s in segments),
                          bool, len(segments)) & (sizes > 0)
    if not movable.any():
        return list(segments)

    points = np.concatenate([p for p, take in zip(kept, movable) if take])
    owner = np.repeat(np.flatnonzero(movable), sizes[movable])
    xs = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, width - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, height - 1)
    under = depth[ys, xs]

    # The quantile per stroke, from one sort of (owner, depth) together.
    order = np.lexsort((under, owner))
    starts = np.concatenate(([0], np.cumsum(sizes[movable])[:-1]))
    pick = starts + np.minimum((sizes[movable] - 1),
                               np.floor((sizes[movable] - 1) * quantile / 100.0).astype(np.int64))
    room = 2.0 * under[order][pick]

    out = list(segments)
    for index, allowance in zip(np.flatnonzero(movable), room):
        segment = segments[index]
        narrowed = max(floor, min(float(segment.width), float(allowance)))
        if narrowed < float(segment.width):
            out[index] = Segment(segment.points, segment.color, narrowed, segment.fill)
    return out


def _without_invented(segments, allowed, share):
    """The segments that are drawing the page's own ink, by their own points.

    A stroke is judged on where its points sit, not on the pixels it paints:
    there are tens of thousands of these on a page and rendering each one on
    its own to attribute the overshoot would cost more than the tracing. A
    speck invented in the paper has all of its points off the ink and goes; a
    real stroke whose drawn band spills a little at the edge has all of its
    points on the ink and stays, because the edge is the renderer's rounding
    rather than a mark that should not be there.
    """
    height, width = allowed.shape
    kept = []
    for segment in segments:
        points = np.asarray(segment.points)
        if not len(points):
            continue
        xs = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, width - 1)
        ys = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, height - 1)
        if float(allowed[ys, xs].mean()) >= share:
            kept.append(segment)
    return kept


def fill_widths(profile, pen):
    """(boundary width, curve spacing, curve width) for this page, in pixels.

    All three are set in pen widths, because the pen is the length scale of
    the writing and it is measured per page: the same numbers then fit a
    phone scan at 400 DPI and stylus notes at 800. Never below a pixel --
    a band thinner than the grid it is drawn on covers nothing.
    """
    boundary = max(1.0, float(profile.boundary_width) * pen)
    spacing = max(1.0, float(profile.fill_spacing) * pen)
    return boundary, spacing, max(spacing, spacing * float(profile.fill_overlap))


def _area_marks(mask, depth, room):
    """The marks that are areas; everything else is a line.

    A mark has to be at least `room` deep somewhere to be worth filling --
    that is, deep enough for one offset curve to fit inside its boundary.
    Below that, a loop around it is a loop around a line: the two sides of
    the boundary meet in the middle, the two bands have to line up on the
    pixel grid to within a pixel to cover it, and where they do not, a row
    of the mark is simply never drawn. Down its middle at its own width, it
    is one stroke and covered exactly -- which is what centreline tracing
    does, and why it is right for handwriting.

    So this is where the two ideas meet rather than compete: the ink that is
    an area is filled as an area, and the ink that is a line is drawn as a
    line. On a page written with a 2.5px pen almost every mark is a line and
    almost nothing is filled, which is correct; on a page of drawings and
    highlighting most of it is filled.

    `depth` is the unsmoothed distance field -- smoothing moves depths, and
    this is a measurement of the ink, not of a curve.
    """
    count, labels, _stats, _ = cv2.connectedComponentsWithStats(mask.view(np.uint8),
                                                                connectivity=8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    # Over the ink only: np.maximum.at costs per element, and the page is a
    # couple of percent ink.
    where = np.flatnonzero(mask.ravel())
    deepest = np.zeros(count, np.float32)
    np.maximum.at(deepest, labels.ravel()[where], depth.ravel()[where])
    keep = deepest > room
    keep[0] = False
    return geometry.pick(keep, labels, mask)


def _offset_fill(mask, img_raw, profile, background_bgr, pen, min_run=None, stages=None):
    """The region as its boundary and the curves parallel to it, going inwards.

    One signed distance field for the whole page (`geometry.signed_distance`),
    and every curve is a level set of it:

        the boundary        at half its own stroke width inside the true edge
        offset curve k      one spacing deeper than the one before

    Each is a whole closed loop at sub-pixel accuracy, so a ring is one long
    continuous stroke rather than a contour's worth of short pieces, and the
    shape's topology looks after itself: where a stroke branches, the level
    set at that depth is two loops; where it has run out, there is no loop.
    The old version thresholded a per-blob distance transform and traced the
    result with `findContours`, which asked the pixel grid the same question
    once per layer -- so every layer carried the same one-pixel staircase and
    broke wherever the staircase turned. That is what made an onion fill look
    like a heap of fragments instead of rings.

    Two things matter for not fattening the writing. The boundary curve is
    laid `width / 2` inside the edge, so the stroke drawn on it covers
    exactly the outermost band of the ink and nothing outside it -- a curve
    laid on the edge itself puts half its width on the paper, which is what
    made traced writing come back heavier than it was. And the boundary is
    stroked at its own width, narrower than the fill, rather than being the
    first and widest ring of the fill.

    This routine only *proposes* curves. What they actually cover is not its
    business and was never knowable from here: the caller draws them and
    compares the result with the mask, and fills whatever is still paper
    (`_solid_segments`). Every attempt to answer it here instead -- growing
    the curves by half their width, testing the field directly -- was a model
    of the renderer rather than the renderer, and each was wrong by about a
    pixel in a different place. A pixel of ink wrongly believed drawn is a
    pixel nobody ever draws.
    """
    boundary_width, spacing, curve_width = fill_widths(profile, pen)
    first_curve = boundary_width + curve_width / 2.0
    area = _area_marks(mask, geometry.signed_distance(mask), first_curve)
    if not area.any():
        # Nothing on this page is wide enough to be filled as an area: every
        # mark is a line and the detail pass draws all of it. Say so rather
        # than leave the stage blank -- "no rings here" is an answer.
        if stages is not None:
            stages('Outline', {'segments': []})
        return []
    # The field and its level sets only exist near the ink, and the ink is a
    # couple of per cent of the page. Computed over the whole of it, two
    # distance transforms and a marching-squares pass each cost most of
    # their time on blank paper -- fourteen of the shape stage's seconds on
    # the user's page. Cropped to the marks' own box with a margin wide enough for
    # the deepest curve, the answer is the same and the offset puts the
    # loops back where they belong.
    margin = int(np.ceil(first_curve + spacing * MAX_ONION_LAYERS)) + 2
    rows, columns = np.nonzero(area)
    top = max(0, int(rows.min()) - margin)
    bottom = min(area.shape[0], int(rows.max()) + 1 + margin)
    left = max(0, int(columns.min()) - margin)
    right = min(area.shape[1], int(columns.max()) + 1 + margin)
    window = (slice(top, bottom), slice(left, right))
    origin = np.array([left, top], np.float64)

    field = geometry.signed_distance(np.ascontiguousarray(area[window]),
                                     float(profile.boundary_smooth))
    levels = [(boundary_width / 2.0, boundary_width)]
    levels += [(level, curve_width) for level in
               geometry.offset_levels(field, first_curve, spacing, MAX_ONION_LAYERS)]

    shift = float(profile.boundary_shift)
    epsilon = float(profile.simplify_epsilon)
    out = []
    boundary_only = 0
    for level, width in levels:
        for loop in (piece + origin for piece in geometry.iso_loops(field, level)):
            loop = geometry.simplify(geometry.constrain_smooth(loop, shift, SMOOTH_PASSES),
                                     epsilon)
            if len(loop) < 2:
                continue
            out.extend(_colored_pieces(loop, width, img_raw, profile,
                                       background_bgr, min_run))
        if not boundary_only:
            boundary_only = len(out)        # everything so far is the boundary
            if stages is not None:
                stages('Outline', {'segments': list(out)})
    return out


def _detail_strokes(residue, mask, img_raw, profile, background_bgr, step, min_run=None,
                    smallest=1):
    """The ink still showing as paper, drawn down its middle at its own width.

    This is where hairlines, the tails of strokes and the last sliver in the
    middle of a thick one come from. It runs once for the page rather than
    once per blob -- thinning walks the ink, and asking it a thousand times
    for one page's worth of leftovers took five times as long for the same
    strokes.

    `smallest` used to be four pixels, on the reasoning that anything smaller
    was the fill having missed a corner rather than a mark. That reasoning
    depended on the leftovers being an artefact of *estimating* what the
    curves covered; now they are measured against the drawing itself, so a
    leftover pixel is ink that nothing draws. On the user's phone scan that rule
    was throwing away 18,859 specks -- 3.6% of the page's ink, and every
    pixel of the 3.6% the fill could not get below. A speck of one or two
    pixels comes back as a dot, which is what `centerline.trace` makes of a
    component too small to have a skeleton.
    """
    if not residue.any():
        return []
    # Drop specks, by area -- never by morphology. An opening takes anything
    # thin with it, and what is left over is thin by nature: it wiped out the
    # hairlines and three quarters of the ink on a page of stylus notes.
    if smallest > 1:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(residue.view(np.uint8), connectivity=8)
        if count > 1:
            keep = np.zeros(count, bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= smallest
            residue = geometry.pick(keep, labels, residue)
    if not residue.any():
        return []
    pen = max(1.0, centerline.pen_width(residue.astype(bool)))
    paths = []
    for points, traced_width in centerline.trace(residue.astype(bool), pen=pen,
                                                 width_from=mask):
        if not len(points):
            continue
        paths.append((_smooth_path(points), max(float(step), float(traced_width))))
    if not paths:
        return []
    # One classification for the page, not one per path. There are seventeen
    # thousand of these on a real page, and asking the palette seventeen
    # thousand times cost more than the tracing of several blobs.
    lengths = np.fromiter((len(points) for points, _ in paths), np.int64, len(paths))
    classes, names = _point_classes(np.concatenate([p for p, _ in paths]),
                                    img_raw, profile, background_bgr)
    edges = np.concatenate(([0], np.cumsum(lengths)))
    out = []
    for index, (points, width) in enumerate(paths):
        out.extend(_colored_pieces(points, width, img_raw, profile, background_bgr,
                                   min_run, classes[edges[index]:edges[index + 1]], names))
    return out


def _point_classes(points, img_raw, profile, background_bgr):
    """The palette class of the ink under each point of a path."""
    height, width = img_raw.shape[:2]
    rows = np.clip(points[:, 1].astype(np.int64), 0, height - 1)
    columns = np.clip(points[:, 0].astype(np.int64), 0, width - 1)
    colors = img_raw[rows, columns]
    if background_bgr is not None and not np.all(np.array(background_bgr) >= 250):
        divisor = np.maximum(np.array(background_bgr, np.float64), 1.0)
        colors = np.clip(colors.astype(np.float64) / divisor * 255.0, 0, 255).astype(np.uint8)
    return classify_bgr(profile, colors)


def _points_color(points, mask, img_raw, profile, background_bgr):
    """The palette colour under a path, voted over its own points."""
    classes, names = _point_classes(points, img_raw, profile, background_bgr)
    counts = np.bincount(classes, minlength=len(names))
    return names[int(counts.argmax())]


def _colored_pieces(points, width, img_raw, profile, background_bgr, min_run,
                    classes=None, names=None):
    """A path as one or more segments, cut where the ink changes colour.

    A ring or a centreline runs through whatever ink is there, and where two
    pens cross that is two pens. Without this the whole path takes whichever
    colour has more of it, and a blue annotation over black notes comes back
    entirely one or the other.
    """
    if profile.colour_5d:
        # Shape first, colour afterwards: this path decides neither, it just
        # hands the geometry on whole. Cutting it here would cut it by
        # something other than colour, which is exactly what must not happen.
        return [Segment(points, None, width)]
    if classes is None:
        classes, names = _point_classes(points, img_raw, profile, background_bgr)
    if min_run is not None:
        pieces = split_by_color(points, classes, names, min_run)
        if pieces is not None:
            return [Segment(piece, name, width) for piece, name in pieces]
    counts = np.bincount(classes, minlength=len(names))
    return [Segment(points, names[int(counts.argmax())], width)]


def _smooth_path(points, window=SMOOTH_WINDOW):
    """The same staircase removal for an open path, keeping its ends put."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < window * 2:
        return points
    kernel = np.ones(window) / window
    padded = np.concatenate([np.repeat(points[:1], window, axis=0), points,
                             np.repeat(points[-1:], window, axis=0)])
    smoothed = np.stack([np.convolve(padded[:, 0], kernel, mode='same'),
                         np.convolve(padded[:, 1], kernel, mode='same')], axis=1)
    smoothed = smoothed[window:-window]
    smoothed[0], smoothed[-1] = points[0], points[-1]
    return smoothed






def _coalesce(runs):
    """Merge neighbouring runs of the same color, around the loop."""
    out = []
    for run in runs:
        if out and out[-1][2] == run[2]:
            out[-1][1] += run[1]
        else:
            out.append(list(run))
    if len(out) > 1 and out[0][2] == out[-1][2]:
        last = out.pop()
        out[0][0], out[0][1] = last[0], out[0][1] + last[1]
    return out


def _smooth_labels(classes, window):
    """Majority class within `window` samples either side, around the loop.

    Contour samples sit on anti-aliased edges, so the raw class flickers
    between two colors wherever they meet; taking each sample on its own
    left runs a few samples long, which the length rule below then merges
    away -- no split ever survived. Smoothing first is what makes the runs
    real. O(classes present x samples) via a circular running count.
    """
    n = len(classes)
    window = min(window, (n - 1) // 2)
    if window <= 0:
        return classes
    best = None
    best_score = None
    index = np.arange(n)
    for klass in np.unique(classes):
        indicator = (classes == klass).astype(np.int32)
        wrapped = np.concatenate((indicator[-window:], indicator, indicator[:window]))
        running = np.concatenate(([0], np.cumsum(wrapped)))
        score = running[index + 2 * window + 1] - running[index]
        if best is None:
            best, best_score = np.full(n, klass, classes.dtype), score
        else:
            better = score > best_score
            best[better] = klass
            best_score[better] = score[better]
    return best


def split_by_color(points, classes, names, min_run):
    """Cut an open path where the colour of the ink under it changes.

    Where two pens cross, the ink is one shape and the tracer gives it one
    path -- so a blue annotation over black notes came back entirely black,
    or entirely blue. This cuts the path into pieces of one colour each.

    Pieces share a point at every cut, so the line stays continuous, and a
    piece shorter than `min_run` points is folded back into its neighbour:
    a couple of samples where two strokes touch is not a change of pen.
    """
    classes = np.asarray(classes)
    if len(points) < 2 or len(classes) != len(points) or len(np.unique(classes)) < 2:
        return None
    window = max(1, int(round(min_run / 2)))
    smoothed = _smooth_open(classes, window)
    cuts = np.flatnonzero(np.diff(smoothed)) + 1
    if not len(cuts):
        return None
    bounds = [0, *cuts.tolist(), len(points)]
    pieces = []
    for start, end in zip(bounds, bounds[1:]):
        if end - start < min_run and pieces:
            # Too short to be a pen of its own: hand it to the piece before.
            previous_points, previous_name = pieces[-1]
            pieces[-1] = (np.concatenate([previous_points, points[start:end]]), previous_name)
            continue
        piece = points[max(0, start - 1):end]     # share the point at the cut
        if len(piece) >= 1:
            pieces.append((piece, names[int(smoothed[start])]))
    if len(pieces) < 2:
        return None
    return pieces


def _smooth_open(classes, window):
    """Majority class within `window` samples either side, along a path.

    The same reasoning as _smooth_labels, for a path with two ends rather
    than a loop: samples taken where two colours meet flicker between them,
    and a run of two or three samples is not a change of pen.
    """
    n = len(classes)
    window = min(window, max(1, (n - 1) // 2))
    if window <= 0 or n < 3:
        return classes
    best = None
    best_score = None
    index = np.arange(n)
    for klass in np.unique(classes):
        indicator = (classes == klass).astype(np.int32)
        padded = np.concatenate((np.repeat(indicator[:1], window), indicator,
                                 np.repeat(indicator[-1:], window)))
        running = np.concatenate(([0], np.cumsum(padded)))
        score = running[index + 2 * window + 1] - running[index]
        if best is None:
            best, best_score = np.full(n, klass, classes.dtype), score
        else:
            better = score > best_score
            best = np.where(better, klass, best)
            best_score = np.where(better, score, best_score)
    return best


def _color_runs(count, positions, classes, min_run):
    """Split a closed contour of `count` points into runs of one ink color.

    `positions` are where along the contour the color samples sit and
    `classes` their palette class. Boundaries land midway between the last
    sample of one color and the first of the next; runs shorter than
    `min_run` points are merged into their larger neighbour, so a couple of
    stray samples where two strokes touch can't cut a stroke into pieces.
    Returns [[start, length, class index]] tiling the loop.
    """
    order = np.argsort(positions, kind="stable")
    positions, classes = positions[order], classes[order]
    samples = len(positions)
    labels = _smooth_labels(classes, max(1, int(round(min_run * samples / max(count, 1) / 2))))
    change = np.flatnonzero(labels != np.roll(labels, 1))
    if len(change) == 0:
        return [[0, count, int(labels[0])]]

    # Walk the samples starting at a color change, so runs don't straddle
    # the seam, and cut midway through each gap between two colors.
    rolled = np.roll(np.arange(samples), -change[0])
    ordered = labels[rolled]
    firsts = np.concatenate(([0], np.flatnonzero(np.diff(ordered)) + 1))
    lasts = np.concatenate((firsts[1:] - 1, [samples - 1]))
    bounds = []
    for last, nxt in zip(lasts, np.roll(firsts, -1)):
        here, after = positions[rolled[last]], positions[rolled[nxt]]
        bounds.append(int((here + ((after - here) % count) // 2) % count))
    runs = []
    for i, first in enumerate(firsts):
        begin = (bounds[i - 1] + 1) % count
        runs.append([begin, (bounds[i] - begin) % count + 1, int(ordered[first])])

    runs = _coalesce(runs)
    while len(runs) > 1:
        shortest = min(range(len(runs)), key=lambda i: runs[i][1])
        if runs[shortest][1] >= min_run:
            break
        before, after = runs[shortest - 1], runs[(shortest + 1) % len(runs)]
        runs[shortest][2] = before[2] if before[1] >= after[1] else after[2]
        runs = _coalesce(runs)
    return runs


def shape_segments(mask: np.ndarray, img_raw: np.ndarray, profile: Profile,
                   background_bgr: Optional[tuple] = None,
                   pen_width_px: Optional[float] = None, stages=None):
    """Stage two A alone: ink that stage one has already settled, as shape.

    Takes a mask that has been through `perfect_image` and returns monochrome
    geometry. Split out so a caller can keep the three stages apart and redo
    only the one whose settings changed -- the panel does, which is what
    makes editing a threshold a different price from editing a ring spacing.
    """
    if profile.trace == 'centerline':
        return _center_segments(mask, img_raw, profile, background_bgr, pen_width_px, stages)
    if profile.trace == 'onion-fill':
        return _solid_segments(mask, img_raw, profile, background_bgr, pen_width_px, stages)
    return _outline_segments(mask, img_raw, profile, background_bgr, pen_width_px, stages)


def traced_segments(mask: np.ndarray, img_raw: np.ndarray, profile: Profile,
                     background_bgr: Optional[tuple] = None,
                     pen_width_px: Optional[float] = None,
                     stages=None, correction=None, report=None):
    """What the tracer would draw, before the xopp coordinate transform:
    one entry per stroke, in image pixels, each already carrying its colour.

    Region-First, Stroke-Last, in three phases:

        phase 1  which pixels are ink            `perfect_image`
        phase 2  which region is which colour    `region_map`
        phase 3  what strokes draw those regions `encode_regions`,
                                                 `validate_output`

    Strokes are made last on purpose. They are the one representation a
    person cannot correct and an algorithm cannot reconsider -- once ink is a
    stroke, changing its colour means cutting it, and every artefact of the
    old order came from that cut. A region can be recoloured, dropped or
    merged; a stroke can only be redrawn. So all the deciding happens while
    the ink is still regions, and drawing is the last thing that happens.

    `report` may be a one-element list, which receives stage 16's findings.
    """
    mask, _allowed, _dropped = perfect_image(mask, img_raw, profile, background_bgr, stages)
    if not mask.any():
        return []
    labels, colours, field = region_map(mask, img_raw, profile, background_bgr,
                                        pen_width_px, correction=correction,
                                        stages=stages)
    if not colours:
        return []
    drawn = labels >= 0
    out = encode_regions(labels, colours, drawn, img_raw, profile, background_bgr,
                         pen_width_px, stages)
    checked = validate_output(out, drawn, profile, labels, field, stages)
    if report is not None:
        report.append(checked)
    return out


def _outline_segments(mask, img_raw, profile, background_bgr=None, pen_width_px=None,
                      stages=None):
    """Stage two A for `outline`: the boundary of every blob, as the
    original scripts drew it. Two lines per stroke, which at handwriting
    size merge and fill the letters in -- kept because the user still uses it.

    The speck rule is not applied again here: it lives in stage one, as a
    component area. A contour's own area is a different question, and
    answering it here as well gave the page two opinions about what a mark
    is."""
    if not mask.any():
        return []
    contours, _ = cv2.findContours(mask.view(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    H, W = img_raw.shape[:2]

    kept_raw = [c.reshape(-1, 2) for c in contours if len(c) >= 3]
    if not kept_raw:
        return []
    if profile.colour_5d:
        return [Segment(points.astype(np.float64), None) for points in kept_raw]
    num_contours = len(kept_raw)

    lengths = np.fromiter((len(p) for p in kept_raw), dtype=np.int64, count=num_contours)
    raw_gid = np.repeat(np.arange(num_contours, dtype=np.int64), lengths)
    all_raw_pts = np.concatenate(kept_raw, axis=0)

    xs, ys = all_raw_pts[:, 0], all_raw_pts[:, 1]
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    all_pts = all_raw_pts[valid]
    all_gid = raw_gid[valid]
    if len(all_pts) == 0:
        return []

    # Each point's own position within its contour, for the color split.
    counts = np.bincount(all_gid, minlength=num_contours)
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    all_pos = np.arange(len(all_gid), dtype=np.int64) - offsets[all_gid]

    all_colors = img_raw[all_pts[:, 1], all_pts[:, 0]]  # one gather for the whole page
    bg = np.array(background_bgr if background_bgr is not None else (255, 255, 255), dtype=np.int16)
    is_ink = np.abs(all_colors.astype(np.int16) - bg).sum(axis=1) > BACKGROUND_TOLERANCE
    ink_colors = all_colors[is_ink]
    ink_gid = all_gid[is_ink]
    ink_pos = all_pos[is_ink]

    if len(ink_colors) == 0:
        return []

    # Contour points sit on the *edge* of a thresholded blob, so in a scanned
    # or photographed page they're anti-aliased blends between the real ink
    # and white paper. Those blends keep the ink's hue but at much lower
    # brightness-difference, which HSV classification then reads as a
    # saturated color -- e.g. near-black ink with a faint green cast in the
    # scan came out as bright green. So per contour, keep only the pixels in
    # the darkest part of *its own* brightness range (the ones closest to
    # true ink) before picking the dominant color. The cutoff is per-contour,
    # not global, so a genuinely light stroke on a mostly-dark page keeps its
    # own pixels instead of being wiped out; every contour always keeps at
    # least its darkest sample. A near-no-op for crisp digital strokes, which
    # are already uniformly dark along the edge.
    value = ink_colors.max(axis=1).astype(np.float64)  # HSV "V" of each sample
    seg_start = np.concatenate(([0], np.flatnonzero(np.diff(ink_gid)) + 1))
    seg_index = np.cumsum(np.concatenate(([0], (np.diff(ink_gid) != 0).astype(np.int64))))
    seg_min = np.minimum.reduceat(value, seg_start)
    seg_max = np.maximum.reduceat(value, seg_start)
    seg_cut = seg_min + (seg_max - seg_min) * INK_DARK_FRACTION
    keep = value <= seg_cut[seg_index]
    ink_colors = ink_colors[keep]
    ink_gid = ink_gid[keep]
    ink_pos = ink_pos[keep]

    # White-balance the samples against the paper before classifying them.
    # On tinted paper every ink pixel is a blend with that tint, so graphite
    # on green paper samples as greenish-grey and gets classified as green
    # ink. Dividing through by the paper color maps "just a darker shade of
    # the paper" onto neutral grey (-> black), while ink with a hue of its
    # own (red pen on green paper) keeps that hue. A no-op on white paper,
    # where the divisor is 255.
    if background_bgr is not None and not np.all(bg >= 250):
        divisor = np.maximum(bg.astype(np.float64), 1.0)
        ink_colors = np.clip(
            ink_colors.astype(np.float64) / divisor * 255.0, 0, 255
        ).astype(np.uint8)

    keys_host = (
        ink_colors[:, 0].astype(np.int64) << 16
        | ink_colors[:, 1].astype(np.int64) << 8
        | ink_colors[:, 2].astype(np.int64)
    )
    dominant_keys = gpu.to_host(
        gpu.batched_dominant_keys(gpu.to_device(keys_host), gpu.to_device(ink_gid), num_contours)
    )

    # Recover each contour's own (bounds-filtered) point slice. `all_gid` is
    # non-decreasing (concatenation preserved contour order, and boolean
    # masking preserves relative order), so a handful of np.split calls
    # recovers per-contour chunks without revisiting every point in a Python
    # loop. A contour entirely dropped by bounds-filtering just has no chunk
    # -- consistent with it also having no ink pixels, so the loop below
    # skips it via dominant_key < 0.
    boundaries = np.flatnonzero(np.diff(all_gid)) + 1
    gid_chunks = np.split(all_gid, boundaries)
    pts_chunks = np.split(all_pts, boundaries)
    pts_by_gid = {int(g[0]): p for g, p in zip(gid_chunks, pts_chunks)}

    split = profile.color_break > 0
    if split:
        classes, names = classify_bgr(profile, ink_colors)
        sample_bounds = np.searchsorted(ink_gid, np.arange(num_contours + 1))
        pen = pen_width_px if pen_width_px else DEFAULT_PEN_WIDTH_PX
        loose, tight = COLOR_BREAK_MIN_RUN
        strength = min(max(profile.color_break, 0.0), 10.0) / 10.0
        min_run = max(4, int(round(pen * (loose + (tight - loose) * strength))))

    name_cache: dict = {}
    segments = []
    for gid in range(num_contours):
        points = pts_by_gid.get(gid)
        if points is None:
            continue
        key = int(dominant_keys[gid])
        if key < 0:
            continue  # every pixel on this contour was background/white

        runs = None
        if split:
            lo, hi = int(sample_bounds[gid]), int(sample_bounds[gid + 1])
            if hi - lo >= 2 and len(np.unique(classes[lo:hi])) > 1:
                runs = _color_runs(len(points), ink_pos[lo:hi], classes[lo:hi], min_run)
                if len(runs) < 2:
                    runs = None

        if runs is None:
            # One color for the whole contour, from its dominant sample.
            name = name_cache.get(key)
            if name is None:
                # Classifying is done once per *distinct* dominant color, not
                # once per contour -- a page typically has only a handful of
                # real ink colors even with thousands of contours.
                dominant_bgr = (key >> 16 & 0xFF, key >> 8 & 0xFF, key & 0xFF)
                if profile.mode == "hand":
                    name = map_pixel_hand(dominant_bgr, profile.hsv_ranges, fallback="xopp_black")
                else:
                    name = map_pixel_elec(dominant_bgr, profile.palette_bgr)
                name_cache[key] = name
            segments.append(Segment(points, name))
            continue

        count = len(points)
        for run_start, run_length, klass in runs:
            # A point of overlap each side, so consecutive pieces meet rather
            # than leaving a gap where the color changes.
            index = np.arange(run_start - 1, run_start + run_length + 1) % count
            segments.append(Segment(points[index], names[klass]))
    return segments



def segments_to_strokes(segments, profile: Profile, scale: float, dx: float, dy: float,
                        border=None) -> List:
    """Stroke elements for what traced_segments returned, in xopp coordinates.

    **The printed border is dropped here when `border` is given**, because this is the function
    the live preview uses (`convertgui.preview_page` -> `segments_to_strokes`), while the filter
    lived only in `_contours_to_strokes`, which the preview never calls. That is why the blue
    border survived in the Traced result after it had been removed from the exported notebook:
    two functions build strokes, and only one of them knew about the border.
    """
    if border is not None and getattr(profile, 'use_frame', False):
        segments = drop_the_template_border(segments, profile, None, scale, dx, dy,
                                            border=border)
    limit = render.path_limit(profile.max_coord_str_len)

    strokes = []
    hex_cache: dict = {}
    for segment in segments:
        points_px, name = segment.points, segment.color
        if name is None:
            name = (0, 0, 0)
        if isinstance(name, tuple):
            # A colour measured on the page rather than a name from a palette
            # (color5d). There is no table to look it up in: it *is* the
            # answer, so it is written as it was measured.
            color_hex = hex_cache.get(name)
            if color_hex is None:
                color_hex = hex_cache[name] = _bgr_to_hex(name)
        else:
            color_hex = hex_cache.get(name)
            if color_hex is None:
                color_hex = hex_cache[name] = _bgr_to_hex(profile.palette_bgr.get(name, [0, 0, 0]))
        placed = points_px.astype(np.float64) / scale + np.array([dx, dy])
        # A centreline stroke carries the width its own ink had, in xopp units.
        width = segment.width / scale if segment.width else profile.line_width
        # A <stroke> has a length limit, and a long path has to be got under
        # it. Cut it into pieces rather than thin it out: dropping every
        # twenty-fourth point off a long contour is what made traced writing
        # come back visibly faceted. On a real page only a dozen paths are
        # long enough to need this -- and they carry 58% of all the points.
        # A filled path cannot be cut (its outline would open), so that one
        # is thinned as before.
        if segment.fill is not None:
            stride = max(1, -(-len(placed) // limit))
            strokes.append(make_stroke_element([(x, y) for x, y in placed[::stride]],
                                               color_hex, width, fill=segment.fill))
            continue
        # The cut itself lives in render.cut_path, because the fill loop has
        # to make exactly the same one: two pieces sharing a point are not
        # the same picture as one unbroken line, since PIL caps each piece
        # separately. A loop that rasterized the whole path instead believed
        # it had drawn 35 pixels that nothing draws.
        for piece in render.cut_path(placed, limit):
            strokes.append(make_stroke_element([(x, y) for x, y in piece], color_hex, width))

    return strokes


def _contours_to_strokes(mask, img_raw: np.ndarray, profile: Profile,
                          scale: float, dx: float, dy: float,
                          background_bgr: Optional[tuple] = None,
                          pen_width_px: Optional[float] = None,
                          border=None) -> List:
    """Turn one page's contours into <stroke> elements.

    `background_bgr` is the page's actual paper color (from probe.py) and
    `pen_width_px` its measured pen width, which sets the smallest piece a
    color split may leave.
    """
    segments = traced_segments(mask, img_raw, profile, background_bgr, pen_width_px)
    segments = drop_the_template_border(segments, profile, img_raw, scale, dx, dy,
                                        border=border)
    return segments_to_strokes(segments, profile, scale, dx, dy)


def denoise_and_sharpen(gray: np.ndarray, denoise: float, sharpen: float) -> np.ndarray:
    """Ported from purify.py's denoise/sharpen stage: Gaussian blur for
    denoise, then an unsharp mask for sharpen, on the same 0-10 slider scale
    and formula purify.py used. A no-op (returns `gray` unchanged) when both
    are 0, which is the default -- this stage didn't exist in the original
    pipeline, so profiles that don't set these keep identical output.
    """
    if denoise <= 0 and sharpen <= 0:
        return gray
    d = denoise / 10.0
    s = sharpen / 10.0 * 70
    sigma = 0.3 + 2.0 * d
    alpha = 1 + 1.5 * s
    denoised = cv2.GaussianBlur(gray, (5, 5), sigma)
    if sharpen <= 0:
        return denoised
    blur = cv2.GaussianBlur(denoised, (5, 5), sigma)
    return cv2.addWeighted(denoised, 1 + alpha, blur, -alpha, 0)


def apply_morph(binary: np.ndarray, morph_open: float, morph_close: float,
                 gain: float = 0.3) -> np.ndarray:
    """Ported from purify.py's morph stage: open then close on the binary
    mask, kernel size derived from the same 0-10 slider scale and gain
    purify.py used. A no-op when both are 0 (the default)."""
    out = binary
    if morph_open > 0:
        k = int(1 + morph_open * gain)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    if morph_close > 0:
        k = int(1 + morph_close * gain)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return out


def the_border_of(img_raw: np.ndarray, profile: Profile):
    """The printed border on this page, in page points, or None.

    Measured on a copy brought down to about one pixel per point: the border test ("a straight
    line on each of the four sides, and an empty middle") is scale-free, and measuring it on an
    800 dpi page costs seconds and hundreds of megabytes for an answer that a 1 px/pt copy gives.
    The page size is derived from the image itself, so the caller cannot pass the wrong one.
    """
    from xp.core import qbasefit

    dpi = max(1, int(getattr(profile, 'dpi', 72) or 72))
    height, width = img_raw.shape[:2]
    page = (width * 72.0 / dpi, height * 72.0 / dpi)
    grey = img_raw[:, :, :3].min(axis=2) if img_raw.ndim == 3 else img_raw
    try:
        if width > 2 * page[0] or height > 2 * page[1]:
            new_w = max(32, int(round(page[0])))
            new_h = max(32, int(round(page[1])))
            grey = cv2.resize(grey, (new_w, new_h), interpolation=cv2.INTER_AREA)
        fit = qbasefit.fit_qbase(grey, page=page)
    except Exception:
        return None
    policy = qbasefit.transform_for(fit)
    if not policy.applies or abs(policy.sx - policy.sy) > 0.02:
        return None
    return tuple(fit.box)


def convert_page_image(img_raw: np.ndarray, profile: Profile, scale: float, dx: float,
                        dy: float, auto: bool = True, border=None) -> List:
    """Convert one already-rendered page image into stroke elements.

    Split out of convert_pdf so a single page can be converted on its own --
    which is what the page-by-page interactive flow and `tune` both need,
    rather than having to run a whole document to look at one page.

    `auto` probes the page for its real paper color (and, where the profile
    hasn't been tuned away from its defaults, threshold/area settings) rather
    than assuming a white background.
    """
    probe = None
    background_bgr = None
    if auto:
        from xp.workers.converter.probe import probe_page

        probe = probe_page(img_raw)
        background_bgr = probe.background_bgr

    if border is None and getattr(profile, 'use_frame', False):
        # **One detection, and it lives here.** The user's report: the blue border was still in the
        # Traced result and the dashed guide was beside it, not on it. The reason was that the
        # calibrated rectangle had been threaded into `convert_pdf` only -- the GUI has two
        # other paths (`convert_document` and the live preview) and neither passed it, so they
        # fell back to `frame_in_image`, which returns None on the user's file, and every border
        # stroke survived. Computing it here means all three paths get the same rectangle and
        # there is no second place to forget.
        border = the_border_of(img_raw, profile)
    gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    gray = denoise_and_sharpen(gray, profile.denoise, profile.sharpen)

    block_size = profile.adaptive_block_size
    c = profile.adaptive_c
    if probe is not None and profile.auto_threshold:
        block_size, c = probe.suggested_block_size, probe.suggested_c

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c
    )
    binary = apply_morph(binary, profile.morph_open, profile.morph_close)

    return _contours_to_strokes(binary > 0, img_raw, profile, scale, dx, dy, background_bgr,
                                 probe.stroke_thickness_px if probe is not None else None,
                                 border=border)


# How far, in page units, a traced piece may sit from the template's printed
# border and still be that border. The frame is drawn as a hairline; once it has
# been through a threshold and a contour walk it comes back as a run of short
# two-point pieces a fraction of a point wide, so the band has to allow for the
# pen width as well as the rounding.
BORDER_BAND = 2.5


def _looks_like_the_template_border(colour) -> bool:
    """The frame is printed in blue, and blue ink is also just ink.

    Only the *geometry* decides whether a piece may go (see
    `drop_the_template_border`); this keeps a black pen that happens to run
    along the edge of the writing area.
    """
    if isinstance(colour, tuple) and len(colour) >= 3:
        blue, green, red = colour[0], colour[1], colour[2]
        if blue > 60 and blue - red > 40 and blue - green > 20:
            return True
        # **The border's own antialias, which is not saturated.** Measured on a Q base page
        # drawn at 1.5 pt and warped: the fringe of the line traces as separate thin strokes in
        # `#E0E7F6` and friends -- B-R is 22 there, so the blue test above refused them and 29
        # pale pieces survived on the border. A colour that light cannot be pen ink on paper;
        # the geometry test (inside the band) has already established where it is.
        return min(blue, green, red) >= 200
    return str(colour).lower() in ('blue', 'bluepen', 'blue ink')


def drop_the_template_border(segments, profile: Profile, img_raw, scale: float,
                             dx: float, dy: float, border=None):
    """Remove the printed Q base border from what was traced.

    The border is part of the paper: it comes from the template behind the
    writing, so it must not become strokes -- but *blue handwriting* must, and
    deleting everything blue would be the wrong answer to the same problem.
    Two conditions together, therefore: the piece lies on the frame's rectangle
    (within `BORDER_BAND` of it, in page units) **and** it is blue.

    Only applied when a frame is actually printed on the page. A Q base page
    with no frame is written on the template, and its blue notes may be
    anywhere, including near the edge.
    """
    if not segment_filter_enabled(profile):
        return segments
    from xp.core.frame import REFERENCE, frame_in_image

    # **The rectangle the alignment already found, or nothing.** The user's objection, and it is
    # the right one: the first step (the Q base calibration) *has* the border, so asking
    # `frame_in_image` for it again makes the answer depend on the rendering (it needs a
    # dpi-sized blue frame) and on the parameters. Measured: on the user's file the calibration
    # finds the border and `frame_in_image` on the same page returns None, so the border
    # survived into the traced result. `border` is that rectangle in page units, and it is
    # used instead. `frame_in_image` remains only for callers that have no calibration.
    if border is not None:
        x0, y0, x1, y1 = border
    else:
        found = frame_in_image(img_raw)
        if found is None:
            return segments
        x0, y0, x1, y1 = REFERENCE
    x0, y0, x1, y1 = REFERENCE
    kept = []
    for segment in segments:
        points = np.asarray(segment.points, dtype=float)
        if len(points) == 0:
            kept.append(segment)
            continue
        page = points / scale + np.array([dx, dy])
        on_edge = ((np.abs(page[:, 0] - x0) <= BORDER_BAND)
                   | (np.abs(page[:, 0] - x1) <= BORDER_BAND)
                   | (np.abs(page[:, 1] - y0) <= BORDER_BAND)
                   | (np.abs(page[:, 1] - y1) <= BORDER_BAND))
        if on_edge.all() and _looks_like_the_template_border(segment.color):
            continue
        kept.append(segment)
    return kept


def segment_filter_enabled(profile: Profile) -> bool:
    """Whether this profile wants the printed border taken out.

    Same switch as the frame-based geometry (`use_frame`): a profile that has
    been told to ignore the frame must not silently have pieces removed by it.
    """
    return bool(getattr(profile, 'use_frame', False))


def page_geometry_and_border(pdf_page, profile: Profile, img_raw=None):
    """`(scale, dx, dy)` and the border the calibration found, in page units.

    **One detection, two users.** The user's objection to the old shape: the alignment step
    already finds the printed border, so the border removal must not look for it again --
    it would be a second reading of the same thing, in a different coordinate system, at a
    different resolution, and the two can disagree (they did: the calibration found the user's
    border, `frame_in_image` on the same page returned None, and the border stayed in the
    traced result).

    The rectangle returned is `qbasefit`'s box in **page points**, or None when the page
    has no border the calibration accepts.
    """
    border = None
    if profile.scale is not None:
        return (profile.scale, profile.dx or 0.0, profile.dy or 0.0), border
    if profile.use_frame:
        from xp.core import qbasefit

        size = (pdf_page.rect.width, pdf_page.rect.height)
        try:
            pixmap = pdf_page.get_pixmap(matrix=pymupdf.Matrix(1.0, 1.0))
            array = np.frombuffer(pixmap.samples, dtype=np.uint8)
            array = array.reshape(pixmap.height, pixmap.width, pixmap.n)
            small = array[:, :, :3].min(axis=2)
        except Exception:
            small = None
        if small is not None:
            fit = qbasefit.fit_qbase(small, page=size)
            policy = qbasefit.transform_for(fit)
            if policy.applies and abs(policy.sx - policy.sy) <= 0.02:
                scale = (profile.dpi / 72.0) / policy.sx
                border = tuple(fit.box)
                return (scale, -policy.dx * policy.sx, -policy.dy * policy.sy), border
    geometry = fit_geometry(
        pdf_page.rect.width, pdf_page.rect.height, profile.dpi,
        profile.page_width, profile.page_height,
    )
    return geometry, border


def page_geometry(pdf_page, profile: Profile, img_raw=None):
    """Where this page's ink lands in the output page (the border is dropped)."""
    return page_geometry_and_border(pdf_page, profile, img_raw)[0]


def the_page_in_qbase(img_raw: np.ndarray, profile: Profile):
    """`(image, geometry)` with the page already placed on the Q base sheet.

    The user's design in one function: after the auto alignment the image **is** the sheet -- page size,
    x/y ratio and the corner region all settled -- so every stage after this one works in one
    coordinate system and the geometry is `scale = dpi / 72`, no offset.

    When the calibration cannot be applied (no border, a cut border, content mistaken for a
    border) the page is left as it is and the old geometry answers, so nothing regresses.
    """
    from xp.core import qbasefit

    dpi = max(1, int(getattr(profile, 'dpi', 72) or 72))
    height, width = img_raw.shape[:2]
    page = (width * 72.0 / dpi, height * 72.0 / dpi)
    grey = img_raw[:, :, :3].min(axis=2) if img_raw.ndim == 3 else img_raw
    try:
        small = grey
        if width > 2 * page[0] or height > 2 * page[1]:
            small = cv2.resize(grey, (max(32, int(round(page[0]))),
                                      max(32, int(round(page[1])))),
                               interpolation=cv2.INTER_AREA)
        fit = qbasefit.fit_qbase(small, page=page)
        warped, geometry = qbasefit.warp_into_qbase(img_raw, fit, dpi)
    except Exception:
        warped, geometry = None, None
    if warped is None:
        # `fit_geometry`, not `geometry_for`: the latter answers `(geometry | None, reason)` and
        # carries the calibration policy with it, while this function's contract is the plain
        # three numbers. Handing back two made `segments_to_strokes(*geometry)` fail with
        # "missing 1 required positional argument: 'dy'".
        return img_raw, fit_geometry(page[0], page[1], dpi,
                                     profile.page_width, profile.page_height)
    return warped, geometry


# **The template the background points at.** Packaged with the program, because the notebook has
# to carry a PDF reference and the sample material is read-only and not shipped. It is the
# Q base page itself: 187.82 x 391.37 pt with the printed border around the writing area.
QBASE_TEMPLATE = Path(__file__).resolve().parents[2] / 'data' / 'Qbase_66x138.pdf'

# **The one place the background templates live.** A notebook's `<background>` has to be
# `domain="absolute"` -- Xournal++ 1.2.5 answers `unknown domain type: relative` to anything
# else -- so the recorded reference is an absolute path whatever we do, and the only question
# is *which* one.
#
# It was once wherever the conversion happened to be staged, which is the system temp
# directory, so notebooks pointed at `%TEMP%\Qbase_66x138.pdf` and lost their background the
# moment Windows cleaned up. A folder of our own under LOCALAPPDATA is a path that stays true:
# it survives a reboot, a Temp sweep, and moving the project from one disk to another.
#
# `XP_CONVENTION_DIR` overrides it, so the one absolute folder in the program can be pointed
# somewhere else without editing the code.
CONVENTION_DIR = Path(os.environ.get('XP_CONVENTION_DIR')
                      or Path(os.environ.get('LOCALAPPDATA') or Path.home()) / 'XP' / 'Convention')


def the_convention_folder() -> Path:
    """Where the templates live. Read at call time, so a test can move it."""
    return Path(os.environ.get('XP_CONVENTION_DIR') or CONVENTION_DIR)


def place_the_background(target_path, template=None) -> Optional[str]:
    """Keep the template in the convention folder; return its absolute path.

    **Absolute and fixed**, because those are the two things that make a
    background survive.

    *Absolute*: Xournal++ 1.2.5 refuses `domain="relative"` outright -- it
    answers "XML parser error: unknown domain type: relative", and its own
    strings list `absolute` and `attach` and nothing else.

    *Fixed*: this used to copy the template beside the **target**, and a
    Ctrl+S conversion is staged in the system temp directory before the user is
    asked where it goes -- so the target, at the moment the reference was
    written, was `%TEMP%`. The staged file was then moved to its real home
    and the reference was never updated. It surfaces later as Xournal++'s
    "Missing PDF background file ... It was last seen at: %TEMP%".

    So the template is kept in that one folder and that is what goes in the
    file. Returns None when it cannot be placed anywhere, so a notebook is
    never written pointing at a file that is not on disk.
    """
    source = Path(template) if template is not None else QBASE_TEMPLATE
    if not source.exists():
        return None
    try:
        folder = the_convention_folder()
        folder.mkdir(parents=True, exist_ok=True)
        kept = folder / source.name
        if not kept.exists() or kept.read_bytes() != source.read_bytes():
            kept.write_bytes(source.read_bytes())
        return str(kept)
    except OSError:
        pass
    # The convention folder is not writable -- another drive, a locked
    # profile. Beside the notebook is the next most likely place to still be
    # true, and it is the file Xournal++ offers when the recorded one is
    # missing.
    try:
        beside = Path(target_path).parent / source.name
        if not beside.exists() or beside.read_bytes() != source.read_bytes():
            beside.parent.mkdir(parents=True, exist_ok=True)
            beside.write_bytes(source.read_bytes())
        return str(beside)
    except OSError:
        return None


def convert_pdf(pdf_path, profile: Profile, pages: Optional[List[int]] = None,
                auto: bool = True, background=None) -> XoppDocument:
    """Convert a PDF to a fresh XoppDocument using `profile`.

    `pages`, if given, is a 0-based list of page indices to convert; other
    pages in the output are left blank.
    """
    doc = pymupdf.open(pdf_path)
    page_indices = pages if pages is not None else list(range(doc.page_count))

    out = XoppDocument.new_empty(
        page_count=doc.page_count, width=profile.page_width, height=profile.page_height
    )
    out_pages = out.pages

    for i in page_indices:
        print(f"Converting page {i + 1}/{doc.page_count}")
        pdf_page = doc[i]
        # Clamp before the geometry, not inside the render: both have to be
        # told the same DPI or the ink lands in the wrong place.
        dpi, note = usable_dpi(pdf_page, profile.dpi)
        if note:
            print(note)
        page_profile = replace(profile, dpi=dpi) if dpi != profile.dpi else profile
        img_raw = render_pdf_page(pdf_page, dpi, in_slices=True)
        # **The page is warped into the sheet before anything is traced.** The border is then
        # measured on that same warped page, so the strokes and the rectangle cannot disagree.
        img_raw, (scale, dx, dy) = the_page_in_qbase(img_raw, page_profile)
        calibrated = (scale, dx, dy) == ((page_profile.dpi / 72.0), 0.0, 0.0)
        if calibrated and background is not None:
            # **The Q base page as the background of this page.** The border is then drawn by
            # Xournal++ from the template -- it is *not* in the strokes, which was the other half
            # of the user's rule. `pageno` is the template's page, which is page 1 of a one-page sheet.
            from xp.core.document.xopp_io import set_pdf_background

            set_pdf_background(out_pages[i].element, background, pageno=1)
        border = the_border_of(img_raw, page_profile)

        for stroke_elem in convert_page_image(img_raw, page_profile, scale, dx, dy,
                                              auto=auto, border=border):
            out_pages[i].layer.append(stroke_elem)

    return out
