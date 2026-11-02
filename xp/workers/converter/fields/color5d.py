"""Colour, decided once for the page, in five dimensions.

The user's design, in the user's words: "its a 5D data with RGB and the 2D geometry
position, so use the stroke data we have above and add the color data from
perfect image, than just cluster finding in this 5D space".

    every ink pixel  ->  (L, a, b, x, y)

The two spatial dimensions are there to make the colour *estimate* robust,
not to cut anything. An edge pixel is a mixture of ink and paper, so on its
own it says something between two pens; with x and y in the distance it is
pulled towards the large body of pixels it sits in, and stops voting for a
colour nobody wrote. That is the whole reason this is five-dimensional and
not three: the measured residue on the sample page is not crossings at all, it
is the anti-aliased rim of every mark (the pen measures 1.6px there, so
every ink pixel is a rim pixel). Making an edge pixel listen to its
neighbours is the only attack on that which does not need the pen to be
thicker than it is.

**But the spatial dimensions die at `colour_map`.** After each cluster is
given a colour, the clusters themselves are thrown away and only a colour per
pixel remains. A stroke is cut where the *colour* changes, never where a
cluster boundary runs -- the user's words: "it should only according the color to
cut the stroke ... or I will see real onion everywhere". Two neighbouring
clusters that are both black are one colour, so nothing is cut between them.

No palette. The colour of a cluster is the colour the page had there. A
palette is a standard the user has not set yet, and a nearest-palette step before
one exists would quietly throw away the only measurement there is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from xp.gpu import backend as gpu
# k-means stops when no centre moves further than this (in the normalized
# 5-D space, where 1.0 is one bandwidth) or after this many rounds. Both are
# generous: the assignment stops changing long before either bites.
CONVERGED = 0.05
# Rounds of k-means. Measured on the sample page at its own resolution: three
# rounds and five give the same four pens holding the same 98.8% of the ink
# and differ only in a tail of 0.1%, for 10s against 15s. The centres stop
# moving long before the assignment would.
MAX_ROUNDS = 4
# A cluster smaller than this many pixels is noise, not a pen: its pixels are
# handed to the nearest surviving cluster rather than being given a colour of
# their own.
MIN_CLUSTER_PX = 8
# How many steps a mean-shift seed may climb before it is taken as settled.
MODE_STEPS = 24
# Used while clustering, where only the ratio to the spatial bandwidth
# matters; the real one is applied when the clusters are gathered into pens.
DEFAULT_COLOUR_BANDWIDTH = 20.0
# Longest side the clustering runs at. None means the page's own resolution,
# which is the default and what was asked for: full resolution, never a
# downscaled one.
#
# <~ 之前这里默认缩到 2200 再聚类,理由是「颜色是慢变场」。那个理由对一半:
#    慢变的是颜色本身,不是墨的形状 —— 而缩图会把细笔画和相邻两笔之间的缝
#    一起糊掉,于是边缘像素的颜色被提前混进了邻居。要的是原分辨率,
#    这里就按原分辨率算;要快就去把算法做快,不是把输入做糙。 ~>
WORK_MAX_DIM = None


@dataclass
class ColourField:
    """The page's colour, decided per pixel.

    `labels` is the cluster each ink pixel fell in (-1 off the ink), kept for
    the preview only. `bgr` is what every stage after this one uses: one
    colour per ink pixel, with the clusters already forgotten.
    """
    labels: np.ndarray
    bgr: np.ndarray
    centres_bgr: np.ndarray
    sizes: np.ndarray
    colours_bgr: np.ndarray
    source_bgr: np.ndarray = None      # the page's own colour, for the cloud view
    paper_lab: np.ndarray = None       # so a later measure can use the same space
    _busiest: np.ndarray = None        # see busiest_pen: a constant per page

    @property
    def clusters(self) -> int:
        return len(self.centres_bgr)

    @property
    def colours(self) -> int:
        """How many colours the page turned out to have been written in."""
        return len(self.colours_bgr)


def _lab(img_bgr):
    """CIE Lab with L in 0..100, not OpenCV's 8-bit packing of it.

    The 8-bit form stretches L over 0..255, so a distance in it is 2.55x the
    distance in L and the colour bandwidth would mean something different in
    lightness than in hue. In float it is a plain Delta-E, which is what the
    setting claims to be.
    """
    img_bgr = np.asarray(img_bgr)
    if not img_bgr.size:
        # cvtColor asserts on an empty array. A page can reach here with no
        # ink in any cluster -- a nearly blank page, or one whose ink all
        # fell below the cluster floor -- and that is an answer ("no
        # colours"), not an error to crash on.
        return np.zeros(img_bgr.shape[:-1] + (3,), np.float32)
    return cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def _to_bgr(lab):
    if not len(lab):
        return np.zeros((0, 3), np.uint8)
    rgb = cv2.cvtColor(np.asarray(lab, np.float32).reshape(-1, 1, 3), cv2.COLOR_LAB2BGR)
    return np.clip(rgb.reshape(-1, 3) * 255.0, 0, 255).astype(np.uint8)


def _inkness(lab, paper_bgr):
    """How much each pixel counts: how far it is from the paper.

    A pixel on the edge of a mark is a blend of ink and paper, so the colour
    it reports is a colour nobody wrote. It is still evidence -- it says
    which side its mark is on -- but it must not be allowed to drag a
    cluster's colour halfway to grey, which is exactly what an unweighted
    mean does. On a 1.6px pen almost every pixel is an edge pixel, so this
    is not a corner case, it is the common one.

    Normalized by the ink's own 90th percentile, so a faint pen is not
    silenced: what matters is which pixels are the core of *this* page's
    marks, not an absolute darkness.
    """
    if paper_bgr is None:
        paper_lab = np.array([100.0, 0.0, 0.0], np.float32)
    else:
        paper_lab = _lab(np.array(paper_bgr, np.uint8).reshape(1, 1, 3)).reshape(3)
    distance = np.linalg.norm(lab - paper_lab, axis=1)
    if not len(distance):
        return distance
    scale = float(np.percentile(distance, 90))
    if scale <= 1e-6:
        return np.ones(len(distance), np.float64)
    return np.clip(distance / scale, 0.0, 1.0).astype(np.float64) ** 2


def cluster(mask, img_bgr, space_bandwidth, colour_bandwidth, paper_bgr=None):
    """Cluster the ink in (L, a, b, x, y). Bandwidths are in their own units.

    `space_bandwidth` is in pixels and `colour_bandwidth` in Lab units; each
    axis is divided by its own, so what the distance really depends on is the
    *ratio* between them. Lab is used rather than BGR because a Euclidean
    distance in BGR does not mean "looks different".

    The seeds are a grid one bandwidth apart, and each pixel is only ever
    compared with the centres within one bandwidth of it. That locality is
    what makes this O(n) per round instead of O(n*k) -- the same trick SLIC
    uses, and the reason a page with millions of ink pixels and thousands of
    clusters is seconds rather than hours.
    """
    ys, xs = np.nonzero(mask)
    if not len(ys):
        empty = np.full(mask.shape, -1, np.int32)
        return empty, np.zeros((0, 3), np.uint8), np.zeros(0, np.int64)

    height, width = mask.shape
    # Lab for the ink only. `_lab(img_bgr)[ys, xs]` converted the whole page
    # first: at 800 DPI that is 11378 x 15178 x 3 float32, **2.07 GB of
    # temporary**, to answer a question about the 2% of it that is ink. The
    # ink's own pixels are 42 MB. Same numbers out, fifty times less memory
    # in -- and `_inkness` never looked at the page at all.
    lab = _lab(np.ascontiguousarray(img_bgr[ys, xs]).reshape(-1, 1, 3)).reshape(-1, 3)
    weight = _inkness(lab, paper_bgr)
    xy = np.stack([xs, ys], axis=1).astype(np.float32)
    space = max(1.0, float(space_bandwidth))
    colour = max(1e-3, float(colour_bandwidth))
    step = max(1, int(round(space)))

    # Seeds: one per step x step cell that has ink in it, and each centre
    # stays the centre of its own cell for good. That is what lets the
    # assignment be nine vectorized passes -- one per neighbouring cell --
    # instead of a loop over the centres: a pixel's only candidates are the
    # centres of the 3x3 block of cells around it, and with one centre per
    # cell those can be gathered straight out of an array.
    rows, columns = height // step + 1, width // step + 1
    pixel_row, pixel_column = ys // step, xs // step
    cell_id = pixel_row * columns + pixel_column
    order = np.argsort(cell_id, kind='stable')
    firsts = np.concatenate(([0], np.flatnonzero(np.diff(cell_id[order])) + 1))
    seeds = order[firsts]
    centre_at = np.full(rows * columns, -1, np.int64)
    centre_at[cell_id[seeds]] = np.arange(len(seeds))
    centre_at = centre_at.reshape(rows, columns)

    # Everything above is bookkeeping over the ink's coordinates -- a sort and
    # some index arithmetic. Everything below is the same arithmetic on every
    # ink pixel, nine times a round, which is what the GPU is for. Written
    # once against `gpu.xp`, so it is numpy on a machine without a card and
    # cupy on the user's.
    #
    # The compaction the CPU version did (`flatnonzero` the live pixels, work
    # on the short arrays) is gone on purpose: on a GPU an irregular length is
    # a device-to-host synchronisation per pass, and nine syncs a round cost
    # more than the arithmetic they save. Invalid candidates get a distance of
    # infinity instead and lose on their own.
    device = gpu.xp
    lab_d = device.asarray(lab)
    xy_d = device.asarray(xy)
    weight_d = device.asarray(weight)
    row_d = device.asarray(pixel_row)
    column_d = device.asarray(pixel_column)
    centre_at_d = device.asarray(centre_at)
    centres_lab = device.asarray(lab[seeds])
    centres_xy = device.asarray(xy[seeds])
    count = len(seeds)
    total = len(ys)
    assigned = device.zeros(total, device.int64)

    for _round in range(MAX_ROUNDS):
        best = device.full(total, np.inf, device.float32)
        assigned = device.zeros(total, device.int64)
        for drow in (-1, 0, 1):
            for dcolumn in (-1, 0, 1):
                near_row, near_column = row_d + drow, column_d + dcolumn
                inside = ((near_row >= 0) & (near_row < rows)
                          & (near_column >= 0) & (near_column < columns))
                candidate = centre_at_d[device.clip(near_row, 0, rows - 1),
                                        device.clip(near_column, 0, columns - 1)]
                live = inside & (candidate >= 0)
                pick = device.where(live, candidate, 0)
                dl = (lab_d - centres_lab[pick]) / colour
                dxy = (xy_d - centres_xy[pick]) / space
                distance = (dl * dl).sum(axis=1) + (dxy * dxy).sum(axis=1)
                distance = device.where(live, distance, device.float32(np.inf))
                better = distance < best
                best = device.where(better, distance, best)
                assigned = device.where(better, pick, assigned)

        counts = device.bincount(assigned, weights=weight_d, minlength=count)
        alive = counts > 1e-6
        safe = device.where(alive, counts, 1.0)[:, None]
        new_lab = device.stack(
            [device.bincount(assigned, weights=lab_d[:, c].astype(device.float64) * weight_d,
                             minlength=count) for c in range(3)], axis=1)
        new_xy = device.stack(
            [device.bincount(assigned, weights=xy_d[:, c].astype(device.float64) * weight_d,
                             minlength=count) for c in range(2)], axis=1)
        new_lab = (new_lab / safe).astype(device.float32)
        new_xy = (new_xy / safe).astype(device.float32)
        moved = float(device.abs(device.concatenate(
            [(new_lab - centres_lab) / colour,
             (new_xy - centres_xy) / space], axis=1)).max())
        centres_lab = device.where(alive[:, None], new_lab, centres_lab)
        centres_xy = device.where(alive[:, None], new_xy, centres_xy)
        if moved < CONVERGED:
            break

    assigned = gpu.to_host(assigned)
    centres_lab = gpu.to_host(centres_lab)
    labels = np.full(mask.shape, -1, np.int32)
    labels[ys, xs] = assigned.astype(np.int32)
    sizes = np.bincount(assigned, weights=weight,
                        minlength=len(centres_lab)).astype(np.float64)
    return labels, centres_lab, sizes


def direction_of(bgr, paper_lab, scale=100.0):
    """One BGR colour as a direction. The measure and the clustering have to
    ask the same question, so they share this."""
    lab = _lab(np.array(bgr, np.uint8).reshape(1, 1, 3)).reshape(1, 3)
    return _directions(lab, paper_lab, scale)[0]


def directions_of(img_bgr, paper_lab, scale=100.0):
    """A whole page as directions, one per pixel. Allocates the page twice
    over in float -- two gigabytes at 800 DPI -- so ask it only when every
    pixel is really wanted; `directions_of_pixels` is for a handful."""
    lab = _lab(img_bgr).reshape(-1, 3)
    return _directions(lab, paper_lab, scale).reshape(img_bgr.shape)


def directions_of_pixels(bgr, paper_lab, scale=100.0):
    """Directions for a list of colours gathered out of a page."""
    pixels = np.asarray(bgr, np.uint8).reshape(-1, 1, 3)
    return _directions(_lab(pixels).reshape(-1, 3), paper_lab, scale)


def _directions(colours, paper_lab, scale=100.0):
    """Each colour as *which ink it is*, with how much of it divided out.

    The vector from the paper to the colour, normalized and scaled back out.
    Every point on one pen's ray from the paper lands on the same place, so a
    mark's core, its anti-aliased rim and the spot where it crosses itself
    are one colour here -- which they are. A colour sitting on the paper
    itself has no direction and is parked at the origin; it carries almost no
    ink weight, so it decides nothing.
    """
    paper = (np.asarray(paper_lab, np.float32) if paper_lab is not None
             else np.array([100.0, 0.0, 0.0], np.float32))
    offset = np.asarray(colours, np.float32) - paper
    length = np.linalg.norm(offset, axis=1, keepdims=True)
    return np.where(length > 1e-6, offset / np.maximum(length, 1e-6) * scale, 0.0)


# <~ merge_colours stood here: a mean shift in direction space that gathered
#    the clusters into pens, on the grounds that a pen is a ray out of the
#    paper rather than a ball in Lab. The geometry of that was right and it
#    is not why it is gone. It is gone because the user asked twice for the
#    clustering's own answer about the colour, and because the thing it was
#    protecting against turned out to cost 0.0005 of a percentage point:
#    measured on the user's page, the clusters put 0.0248% of the ink in a colour
#    the page does not have there and the pens 0.0253%.
#
#    What it did cost is recorded in HANDOFF.md, since the user decided it knowing
#    the number: the seam between two clusters that touch is Delta-E 29.4 at
#    the median, because the clustering is 5-D and neighbouring clusters
#    straddle the edge of a stroke. ~>


def colour_map(mask, img_bgr, space_bandwidth, colour_bandwidth,
                paper_bgr=None) -> ColourField:
    """One colour per ink pixel. This is where the spatial dimensions die.

    After this, nothing downstream can tell which cluster a pixel came from
    -- only what colour it is. That is the point: a stroke crossing from one
    cluster to its neighbour is cut only if the colour changed, so the
    clustering's own tiling cannot print itself onto the strokes.
    """
    grouping = group(mask, img_bgr, space_bandwidth, paper_bgr)
    return field_from(grouping, colour_bandwidth)


@dataclass
class Grouping:
    """The expensive half of the colour pass, kept so the cheap half can be
    redone at another bandwidth without paying for it again.

    Only `space_bandwidth` reaches the clustering; `colour_bandwidth` decides
    how the clusters are gathered into pens afterwards, which is arithmetic
    over a few thousand colours. That split is what makes searching the
    colour settings affordable at all -- a tuner can cluster once and then
    try a dozen bandwidths for nothing.
    """
    labels: np.ndarray
    centres_lab: np.ndarray
    sizes: np.ndarray
    paper_lab: np.ndarray
    source_bgr: np.ndarray
    mask: np.ndarray
    # Filled the first time the page is asked how many pens it has. The
    # answer depends on the clustering and on nothing else, so a search
    # trying other colour settings against the same grouping must not pay
    # for the ladder again -- it was 12 partitions and a pass over every ink
    # pixel per candidate, which is most of what made the colour phase cost
    # seventy seconds.
    chosen: tuple = None


def group(mask, img_bgr, space_bandwidth, paper_bgr=None) -> Grouping:
    """The 5-D clustering on its own: everything that depends on where the
    ink is, and nothing that depends on how far apart two pens must be."""
    labels, centres_lab, sizes = cluster(mask, img_bgr, space_bandwidth,
                                         DEFAULT_COLOUR_BANDWIDTH, paper_bgr)
    paper_lab = (_lab(np.array(paper_bgr, np.uint8).reshape(1, 1, 3)).reshape(3)
                 if paper_bgr is not None else None)
    return Grouping(labels=labels, centres_lab=centres_lab, sizes=sizes,
                    paper_lab=paper_lab, source_bgr=img_bgr, mask=mask)


# The ladder the automatic partition is looked for on, in direction space.
# Wide enough at the top to call a whole page one pen and fine enough at the
# bottom to separate two shades of one, so a plateau in between is a real
# feature of the page and not an artefact of where the ladder stops.
# A cluster further than this from the pen it was gathered into is counted
# as mis-coloured; and a partition may be chosen over a stricter one while it
# puts no more than AUTO_SLACK of the ink wrong than the best rung does.
AUTO_TOLERANCE = 20.0
# What the partition has to achieve: no more than this share of the ink in a
# colour the page does not have there. The user's number, and the user's words with it --
# "I don't worry the stroke is cutting into too many pieces, I expect the
# error below 0.3%". So the pen count is not traded against anything; it is
# whatever reaching 0.3% takes.
AUTO_TARGET = 0.003
# <~ BANDWIDTH_LADDER stood here: twelve bandwidths I chose, of which the
#    widest meeting the target won. The user objected to the answer and the user was
#    right -- on the user's page it came out as seven pens on 1.3 units of margin
#    between two of my rungs, which is a settled number wearing automatic's
#    clothes. The ladder is gone; `auto_bandwidth` bisects instead. ~>
# The bracket the bisection starts from. Not the same kind of choice as the
# ladder's rungs: the floor is the smallest separation that can mean anything
# in a space of radius 100, and the ceiling is wide enough that the whole
# page is one colour by construction, so the answer is inside the bracket
# whatever the page is.
# How precisely the widest acceptable separation is pinned down, in the same
# units. Eight halvings of the bracket get under this, and a finer answer
# would be a distinction the clustering itself cannot support.


def partition_error(grouping: Grouping, pen_of, pens_lab, tolerance=AUTO_TOLERANCE,
                    pixel_directions=None):
    """The share of ink whose pen is not the colour the page has there.

    Per ink pixel, not per cluster: a cluster holding a thousand pixels and
    one holding three count for what they hold. The page's own direction at
    each ink pixel does not depend on the bandwidth, so a caller sweeping the
    ladder computes it once and hands it in.
    """
    if not len(pens_lab) or not len(grouping.centres_lab):
        return 1.0
    inside = grouping.labels >= 0
    if not inside.any():
        return 0.0
    if pixel_directions is None:
        pixel_directions = ink_directions(grouping)
    theirs = _directions(pens_lab, grouping.paper_lab)[pen_of[grouping.labels[inside]]]
    apart = np.linalg.norm(pixel_directions - theirs, axis=1)
    return float((apart > tolerance).sum()) / float(len(apart))


def ink_directions(grouping: Grouping):
    """The page's own colour at every ink pixel, as a direction. Fixed for a
    page, so the ladder pays for it once."""
    inside = grouping.labels >= 0
    return directions_of_pixels(grouping.source_bgr[inside], grouping.paper_lab)


# <~ auto_bandwidth stood here, and before it a twelve-rung ladder of
#    bandwidths I had chosen. The user was right that the seven pens were mine and
#    not the page's; the answer was to stop merging, not to pick the
#    separation more cleverly. Both are gone. ~>


def field_from(grouping: Grouping, colour_bandwidth=None) -> ColourField:
    """One colour per ink pixel: the colour of the cluster it fell in.

    That is the whole of it now. Nothing gathers the clusters into pens, so
    there is no bandwidth to choose and no number for anybody to set:
    **the clustering's answer about the colour is the answer.**

    `colour_bandwidth` is accepted and ignored, so callers written against
    the old signature still work. It will go when they have been visited.

    What leaves here is still one colour per pixel with the *cluster*
    forgotten -- that part was never about the merge. It is the guard that
    keeps the clustering's own tiling from printing itself onto the strokes:
    nothing downstream can ask which cluster a pixel came from, only what
    colour it is, so two neighbouring clusters of the same colour are one
    region and there is nothing to cut between them.
    """
    colours_bgr = _to_bgr(grouping.centres_lab)
    labels = grouping.labels
    bgr = np.zeros(labels.shape + (3,), np.uint8)
    inside = labels >= 0
    if len(colours_bgr):
        bgr[inside] = colours_bgr[labels[inside]]
    return ColourField(labels=labels, bgr=bgr, centres_bgr=colours_bgr,
                       sizes=grouping.sizes, colours_bgr=colours_bgr,
                       source_bgr=grouping.source_bgr, paper_lab=grouping.paper_lab)


def cloud_preview(field: ColourField, paper_bgr=None) -> np.ndarray:
    """The 5-D point cloud as the page sees it: every ink pixel in its own
    colour, which is the (L, a, b) half of each point sitting at the (x, y)
    half. The data the clustering is handed, before it has decided anything.
    """
    view = np.full(field.bgr.shape, 255, np.uint8)
    inside = field.labels >= 0
    view[inside] = field.source_bgr[inside]
    return view


def cluster_preview(field: ColourField) -> np.ndarray:
    """Each cluster in its own measured colour -- not a cluster number and
    not a random hue, because colour is the only thing this step decides and
    a preview that hides it says nothing about whether it is right. Reading
    it: neighbouring patches of the same shade are one pen about to be
    gathered, a patch that differs is a cluster that will go its own way.
    """
    view = np.full(field.bgr.shape, 255, np.uint8)
    inside = field.labels >= 0
    if len(field.centres_bgr):
        view[inside] = field.centres_bgr[field.labels[inside]]
    return view


def preview(field: ColourField) -> np.ndarray:
    """The colour map as a picture: every cluster painted its own colour.

    Not random colours. Colour is the only thing this stage decides, so a
    preview that hides it tells you nothing about whether it is right.
    """
    view = np.full(field.bgr.shape, 255, np.uint8)
    inside = field.labels >= 0
    view[inside] = field.bgr[inside]
    return view


# <~ busiest_pen stood here: the pen holding most of the page, for a
#    stroke whose own colour could not be sampled. There are no pens,
#    and a cluster's colour can always be read. ~>


