"""What changed between two renders of the same page, decided on pixels.

The user's instruction, and the first step of it:

    "The original XOPP is the structural base. The comparison happens at the
     image level, not the stroke level."

    "Step one: implement the Image Difference Engine on its own, without
     wiring it to the full stroke converter."

So this module renders nothing, converts nothing and edits nothing. It takes
two images of the same page and says where they differ, as rectangles with a
kind. Everything that acts on that answer is somewhere else.

**WHY NOT STROKES.** The user put it plainly and it is the whole reason this file
exists: the same handwriting can be 100 strokes in one document, 60 in
another and a photograph in a third, and all three look identical. A stroke
count, a stroke id, an endpoint or a polyline is therefore not evidence about
content. Two documents that draw the same ink are the same page here, however
differently they are built.

**FOUR ANSWERS, NOT TWO.** A region is `unchanged`, `added`, `removed` or
`modified`, and they are not the same decision:

    added      the target has ink the base has not
    removed    the base has ink the target has not
    modified   both have ink in the same place and it is not the same ink
    unchanged  what is left, which is nearly all of a page

`removed` and `modified` are the ones that matter, because an image patch
cannot perform them -- painting over a stroke does not delete it. This module
only reports them; refusing to pretend they are additions is the point.

**WHAT IS NOT TREATED AS A DIFFERENCE.** Anti-aliasing and a sub-pixel
offset put a fringe of changed pixels along every edge in the page, and a
plain `target != base` returns the whole drawing. Two things stop that, both
already measured elsewhere in this codebase rather than invented here:

  - ink is "far from the paper colour" (`inkdiff.INK_DELTA`), not "a
    different pixel value", so a slightly darker grey is not new ink;
  - a tolerance dilation of `inkdiff.TOLERANCE_PX` before the comparison, so
    ink within two pixels of matching ink is the same ink.

That second one has a limit and it is deliberate: a pen that grows from three
pixels to seven grows two pixels on each side, which is the tolerance, and is
**not** reported. A difference the size of the anti-aliasing cannot be a
change on one page and noise on another. Past it -- three pixels against
eleven -- it is reported. The number is one number, `inkdiff.TOLERANCE_PX`,
used in both directions.

And the page is **registered** before any of it: a scan half a millimetre
down the page is not a page of new writing. One translation is estimated by
phase correlation on the coarse pair and reported, so a caller can show it
rather than have it silently applied.

**WHAT IS DELIBERATELY NOT DONE HERE.** `target AND NOT base` is not used as
the content of a patch. Where new ink crosses old, that subtraction leaves
the new stroke cut into pieces with holes where it overlapped. A patch's
content is the **target's own image** over the whole changed region, which is
the only thing that is certainly right. Which of the two a caller paints is
its business; this module hands over the rectangle and both crops.

**COARSE TO FINE**, because the page is 173 megapixels and the change is
usually a line of writing. A low-resolution pass finds candidate regions and
only those are looked at again at full resolution. The cost of a comparison
is then set by how much changed, not by how big the document is -- which is
the whole point of the exercise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from xp.core.imaging.inkdiff import INK_DELTA, TOLERANCE_PX, tolerance_kernel

# The coarse pass's longest side. Small enough that a 61-page document can be
# swept quickly, large enough that a line of handwriting is several pixels
# tall -- at 320 a 180 mm page gives about 1.8 px per millimetre, and a 2 mm
# pen stroke is 3 or 4 pixels across. Below about 200 a thin pen disappears
# into the paper and a real addition is missed, which is the one failure this
# must not have.
COARSE_MAX_DIM = 320

# Where the registration is settled. One pass over the page at this size is
# the floor of what a comparison costs -- you cannot know a page is unchanged
# without looking at it -- so it is as small as it can be and still be right.
# Measured on the user's page over four shifts: worst error 1.07 px at 320, 0.32 at
# 640, 0.29 at 1280, 0.00 at full size, against a 2 px tolerance; 0.011s at
# 640 against 0.272s at full size.
REFINE_MAX_DIM = 640

# How far a candidate box is grown before it is looked at again at full
# resolution, in coarse pixels. A change found at 320 across is located to
# about three full-resolution pixels per coarse pixel, and the fine pass must
# see the whole of what the coarse pass saw a corner of.
COARSE_MARGIN = 2

# A blob smaller than this in coarse pixels is noise from the rescale, not a
# mark. It is in COARSE pixels and deliberately tiny: a full stop at 320
# across is about one pixel, so anything above one would start losing them.
# Real noise is rejected by the ink test and the tolerance dilation, not by
# area, because an area cut is exactly how a thin stroke gets thrown away.
LEAST_COARSE_BLOB = 1

# Boxes closer than this (in fine pixels) are merged into one. The user's words:
# "merge neighbouring small patches, reduce the number of image objects in
# the XOPP". A line of writing comes out as a row of separate blobs and is
# one change.
JOIN_WITHIN = 24

# How far two inks' directions out of the paper may be apart and still be the
# same pen, as the distance between unit vectors. Two pens 30 degrees apart
# are 0.52 apart by this measure, and the anti-aliased edge of one pen is
# under 0.1 -- the blend slides along the ray and does not turn it. Half way
# between is a wide margin either side of both.
DIRECTION_APART = 0.30


@dataclass
class Region:
    """One rectangle that differs, and what kind of difference it is.

    `box` is (x, y, w, h) in the fine image's pixels. `base` and `target` are
    the two crops, so a caller can paint either without cropping again, and
    can see for itself what the kind was decided from.
    """
    kind: str                    # 'added' | 'removed' | 'modified'
    box: Tuple[int, int, int, int]
    base_ink: int                # ink pixels inside the box, base side
    target_ink: int              # ink pixels inside the box, target side
    base: Optional[np.ndarray] = None
    target: Optional[np.ndarray] = None

    @property
    def area(self) -> int:
        return self.box[2] * self.box[3]

    def needs_a_decision(self) -> bool:
        """True when a patch cannot do it and a person has to say.

        An addition can be merged on its own; a removal or a modification
        cannot, because painting an image over a stroke leaves the stroke
        underneath. The user asked for those to be reported, never silently
        applied.
        """
        return self.kind in ('removed', 'modified')


@dataclass
class Difference:
    """The whole answer for one page."""
    regions: List[Region]
    shift: Tuple[float, float]       # what registration found, in fine pixels
    confidence: float                # phase correlation's own response
    base_ink: int
    target_ink: int
    size: Tuple[int, int]            # (width, height) of the fine images

    @property
    def changed(self) -> bool:
        return bool(self.regions)

    def of_kind(self, kind) -> List[Region]:
        return [r for r in self.regions if r.kind == kind]

    def conflicts(self) -> List[Region]:
        return [r for r in self.regions if r.needs_a_decision()]

    def summary(self) -> str:
        moved = ''
        if abs(self.shift[0]) > 0.5 or abs(self.shift[1]) > 0.5:
            moved = ' (after a %.1f, %.1f px shift)' % self.shift
        if not self.regions:
            # The shift is said even when nothing changed, because "nothing
            # changed" and "nothing changed once the page was lined up" are
            # different findings and the second one is the interesting one.
            return 'no visible change' + moved
        counts = {}
        for region in self.regions:
            counts[region.kind] = counts.get(region.kind, 0) + 1
        said = ', '.join('%d %s' % (counts[k], k)
                         for k in ('added', 'removed', 'modified')
                         if k in counts)
        return said + moved


# ---- the pieces ----------------------------------------------------------


def paper_of(image) -> Tuple[int, int, int]:
    """The page's own colour: the most common one, coarsely quantised.

    Not a constant and not white. A scan's paper is cream, a photograph's is
    grey, and "ink" is a distance from whatever this page's paper is.
    """
    small = image
    if max(image.shape[:2]) > COARSE_MAX_DIM:
        small = cv2.resize(image, coarse_size(image), interpolation=cv2.INTER_AREA)
    quant = (small.reshape(-1, small.shape[-1]) // 16).astype(np.int32)
    keys = quant[:, 0] * 256 + quant[:, 1] * 16 + quant[:, 2]
    top = np.bincount(keys).argmax()
    held = small.reshape(-1, small.shape[-1])[keys == top]
    return tuple(int(round(v)) for v in held.mean(axis=0))


def ink_of(image, paper=None) -> np.ndarray:
    """Which pixels are ink: far enough from this page's paper colour.

    The same question `inkdiff.source_ink` asks, and the same threshold, so
    "ink" means one thing across the project. Not a threshold against white
    and not an edge detector: a difference in brightness that is not a
    difference in ink is not a change.
    """
    if paper is None:
        paper = paper_of(image)
    total = np.zeros(image.shape[:2], np.uint16)
    for channel, value in zip(cv2.split(image), paper):
        total += cv2.absdiff(channel, np.full_like(channel, int(value)))
    return total > INK_DELTA


def recoloured(base, target, base_ink, target_ink, paper) -> np.ndarray:
    """Where both sides have ink and the ink is not the same colour.

    `ink_of` cannot see this: it asks how far a pixel is from the paper, and
    black and blue are both far from it. A page rewritten in another pen is
    the same ink everywhere and a different drawing, which is the user's section 9 --
    a colour change is a *modification*, not a nothing and not an addition.

    Asked only where both already have ink, because a colour difference
    between ink and paper is not a recolouring: it is the addition or the
    removal that the ink comparison has already found.

    The threshold is `INK_DELTA`, the same distance that separates ink from
    paper, so "a visible difference in colour" means one thing in this
    project rather than two.
    """
    both = base_ink & target_ink
    if not both.any():
        return np.zeros_like(both)
    return both & (_direction_apart(base, target, paper) > DIRECTION_APART)


def _direction_apart(base, target, paper):
    """How far apart two inks are as *directions out of the paper*.

    Not as colours. Anti-aliasing blends a pen with the paper, which slides
    its colour along the ray from the paper towards the pen and changes every
    channel together -- so a plain colour distance calls the soft edge of
    every stroke a recolouring, and a page shifted three pixels comes back
    rewritten. Measured: with a straight channel distance, a 3-pixel shift of
    an unchanged page reported changes, and so did a pen that merely got
    thicker.

    A direction does not move when a pen is blended with paper; it moves when
    the pen changes. The idea is already in this project -- `color5d` gathers
    clusters into pens by a mean shift in direction space, "a pen is a ray out
    of the paper, not a ball in Lab" -- and this is the same question asked
    per pixel.

    Returned as the distance between the two unit vectors, so 0 is the same
    pen at any strength and 2 is opposite.
    """
    one = base.astype(np.float32) - np.asarray(paper, np.float32)
    two = target.astype(np.float32) - np.asarray(paper, np.float32)
    one /= np.maximum(1e-3, np.linalg.norm(one, axis=2, keepdims=True))
    two /= np.maximum(1e-3, np.linalg.norm(two, axis=2, keepdims=True))
    return np.linalg.norm(one - two, axis=2)


def _scaled_size(image, longest) -> Tuple[int, int]:
    height, width = image.shape[:2]
    ratio = min(1.0, longest / max(width, height))
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def coarse_size(image) -> Tuple[int, int]:
    return _scaled_size(image, COARSE_MAX_DIM)


def register(base_ink, target_ink) -> Tuple[Tuple[float, float], float]:
    """How far the target has moved, in the pixels of the images given.

    A page scanned half a millimetre lower is not a page of new writing, and
    without this it is: every stroke moves, nothing lines up, and the whole
    page comes back as removed-and-added. One translation, by phase
    correlation, with its own response returned so a caller can see how much
    to believe it.

    A translation and nothing more. Rotation and scale are real and are not
    guessed at here: a wrong rotation silently applied would be worse than a
    reported misalignment, and rendering both sides through the same page box
    is what is supposed to settle scale.
    """
    if not base_ink.any() or not target_ink.any():
        return (0.0, 0.0), 0.0
    try:
        a = base_ink.astype(np.float32)
        b = target_ink.astype(np.float32)
        window = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(a, b, window)
        return (float(dx), float(dy)), float(response)
    except cv2.error:
        return (0.0, 0.0), 0.0


def _shift_image(image, shift):
    """An image moved like `_shifted` moves a mask, padded with its paper."""
    dx, dy = int(round(shift[0])), int(round(shift[1]))
    if not dx and not dy:
        return image
    out = np.empty_like(image)
    out[:] = np.array(paper_of(image), image.dtype)
    height, width = image.shape[:2]
    sx0, sx1 = max(0, -dx), min(width, width - dx)
    sy0, sy1 = max(0, -dy), min(height, height - dy)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    out[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = image[sy0:sy1, sx0:sx1]
    return out


def _shifted(mask, shift):
    """The mask moved by `shift`, to the nearest pixel.

    Nearest, not interpolated: this is a boolean mask and half a pixel of ink
    is not a thing. The remainder of the shift is absorbed by the tolerance
    dilation, which is what it is for.
    """
    dx, dy = int(round(shift[0])), int(round(shift[1]))
    if not dx and not dy:
        return mask
    out = np.zeros_like(mask)
    height, width = mask.shape
    sx0, sx1 = max(0, -dx), min(width, width - dx)
    sy0, sy1 = max(0, -dy), min(height, height - dy)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    out[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = mask[sy0:sy1, sx0:sx1]
    return out


def _beyond(one, other):
    """Ink in `one` that `other` does not have, within tolerance.

    The dilation is the whole of the anti-aliasing answer: an edge pixel that
    moved half a pixel is still within two of where it was, so it is the same
    ink. Without it the fringe of every letter on the page is a difference
    and the engine reports a page of changes on two identical renders.
    """
    near = cv2.dilate(other.view(np.uint8), tolerance_kernel()) > 0
    return one & ~near


def _boxes(mask, margin, limit):
    """The bounding boxes of what is in `mask`, grown and in order."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.view(np.uint8), connectivity=8)
    out = []
    height, width = mask.shape
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < limit:
            continue
        x = max(0, stats[index, cv2.CC_STAT_LEFT] - margin)
        y = max(0, stats[index, cv2.CC_STAT_TOP] - margin)
        right = min(width, stats[index, cv2.CC_STAT_LEFT]
                    + stats[index, cv2.CC_STAT_WIDTH] + margin)
        bottom = min(height, stats[index, cv2.CC_STAT_TOP]
                     + stats[index, cv2.CC_STAT_HEIGHT] + margin)
        out.append((int(x), int(y), int(right - x), int(bottom - y)))
    return sorted(out, key=lambda b: (b[1], b[0]))


def join_boxes(boxes, within=JOIN_WITHIN):
    """Boxes that are nearly touching, as one box.

    A line of handwriting comes out of the connected-components pass as a row
    of separate blobs, and it is one change. The user's words: merge the neighbouring
    small patches, so the XOPP does not fill up with image objects.

    Repeated until nothing more joins, because joining two boxes can bring a
    third within reach of the result.
    """
    held = list(boxes)
    moved = True
    while moved:
        moved = False
        out = []
        for box in held:
            for at, kept in enumerate(out):
                if _near(box, kept, within):
                    out[at] = _union(box, kept)
                    moved = True
                    break
            else:
                out.append(box)
        held = out
    return sorted(held, key=lambda b: (b[1], b[0]))


def _near(one, two, within):
    ax, ay, aw, ah = one
    bx, by, bw, bh = two
    return (ax - within < bx + bw and bx - within < ax + aw
            and ay - within < by + bh and by - within < ay + ah)


def _union(one, two):
    ax, ay, aw, ah = one
    bx, by, bw, bh = two
    x, y = min(ax, bx), min(ay, by)
    return (x, y, max(ax + aw, bx + bw) - x, max(ay + ah, by + bh) - y)


def classify(base_ink, target_ink, box) -> str:
    """What kind of change is in this box.

    Asked of the ink on both sides inside the box, after the tolerance
    dilation has already decided what counts as the same. The three answers
    are not symmetric in what a caller may do with them, which is why they
    are three and not one.
    """
    x, y, w, h = box
    here_base = base_ink[y:y + h, x:x + w]
    here_target = target_ink[y:y + h, x:x + w]
    had, has = bool(here_base.any()), bool(here_target.any())
    gained = int(_beyond(here_target, here_base).sum())
    lost = int(_beyond(here_base, here_target).sum())
    if not had and has:
        return 'added'
    if had and not has:
        return 'removed'
    if gained and lost:
        return 'modified'
    if gained:
        # Both sides have ink here and the target has more: new writing that
        # crosses the old. Still an addition -- nothing was taken away -- and
        # the reason a patch must carry the target's whole image rather than
        # the subtraction, which would be the new stroke with holes in it.
        return 'added'
    if lost:
        return 'removed'
    return 'modified'


# ---- the engine ----------------------------------------------------------


def compare(base, target, paper=None, join=JOIN_WITHIN,
            coarse=COARSE_MAX_DIM, keep_crops=True) -> Difference:
    """Where two renders of the same page differ.

    `base` and `target` are BGR images of the same page at the same size --
    putting them there is the caller's job, because a page box and a page
    order are documents' business and not an image's. If they are not the
    same size the target is resized to the base, which is a fallback and says
    so in the shift it reports.

    Coarse first: the whole comparison runs at `coarse` across, which finds
    the regions. Only those regions are looked at again at full resolution,
    so what this costs follows what changed and not what the page contains.
    """
    if base.shape[:2] != target.shape[:2]:
        target = cv2.resize(target, (base.shape[1], base.shape[0]),
                            interpolation=cv2.INTER_AREA)
    height, width = base.shape[:2]
    if paper is None:
        paper = paper_of(base)

    small = coarse_size(base)
    base_small = cv2.resize(base, small, interpolation=cv2.INTER_AREA)
    target_small = cv2.resize(target, small, interpolation=cv2.INTER_AREA)
    base_coarse = ink_of(base_small, paper)
    target_coarse = ink_of(target_small, paper)

    # Registration twice, and the second time is not a refinement anyone can
    # skip. A shift of four pixels on a 2134-pixel page is six tenths of a
    # coarse pixel: the coarse pass cannot represent it, rounds it away, and
    # the fine pass then compares a page with itself one pixel out. Measured
    # on the sample page shifted by (4, 6): one registration reported seven
    # changed regions **inside** the page on a page where nothing had
    # changed, which is the failure the user's section 9 names.
    #
    # So the coarse estimate finds the neighbourhood and a full-resolution
    # phase correlation settles it. It runs once a page and costs about a
    # tenth of a second, against a whole page of false findings.
    #
    # At 640 across, not at full resolution: measured on that page, the worst
    # error over four shifts is 0.32 px at 640 and 0.00 at full size, and the
    # tolerance is 2 px -- so full resolution buys nothing and costs 0.272s
    # against 0.011s. 320 would be 1.07 px, inside the tolerance but with no
    # margin left, which is how the coarse pass came to miss it in the first
    # place.
    shift_small, confidence = register(base_coarse, target_coarse)
    scale = width / float(small[0])
    rough = (shift_small[0] * scale, shift_small[1] * scale)
    middle = _scaled_size(base, REFINE_MAX_DIM)
    back = width / float(middle[0])
    exact, surer = register(
        ink_of(cv2.resize(base, middle, interpolation=cv2.INTER_AREA), paper),
        ink_of(cv2.resize(target, middle, interpolation=cv2.INTER_AREA), paper))
    if surer >= confidence * 0.5:
        shift, confidence = (exact[0] * back, exact[1] * back), surer
    else:
        shift = rough
    shift_small = (shift[0] / scale, shift[1] / scale)
    moved = _shifted(target_coarse, shift_small)
    moved_small = _shift_image(target_small, shift_small)

    # Three questions, not one. What the target gained, what it lost, and --
    # where both already have ink -- whether it is the same ink. The third is
    # the one `ink_of` cannot answer, and a page rewritten in another pen is
    # all third.
    changed = (_beyond(moved, base_coarse) | _beyond(base_coarse, moved)
               | recoloured(base_small, moved_small, base_coarse, moved, paper))
    if not changed.any():
        return Difference([], shift, confidence,
                          int(base_coarse.sum()), int(target_coarse.sum()),
                          (width, height))

    # Coarse boxes, scaled up, then joined -- joined at full resolution so the
    # distance that decides it is a real distance on the page.
    rough = _boxes(changed, COARSE_MARGIN, LEAST_COARSE_BLOB)
    grown = [(int(x * scale), int(y * scale),
              int(np.ceil(w * scale)), int(np.ceil(h * scale)))
             for x, y, w, h in rough]
    grown = [(max(0, x), max(0, y), min(w, width - x), min(h, height - y))
             for x, y, w, h in grown]
    boxes = join_boxes(grown, join) if join else grown

    # Fine: only inside the boxes, which is the whole point.
    regions = []
    fine_shift = (int(round(shift[0])), int(round(shift[1])))
    for box in boxes:
        x, y, w, h = box
        if w <= 0 or h <= 0:
            continue
        base_crop = base[y:y + h, x:x + w]
        target_crop = _crop_shifted(target, box, fine_shift)
        here_base = ink_of(base_crop, paper)
        here_target = ink_of(target_crop, paper)
        gained = _beyond(here_target, here_base)
        lost = _beyond(here_base, here_target)
        tinted = recoloured(base_crop, target_crop, here_base, here_target,
                            paper)
        if not gained.any() and not lost.any() and not tinted.any():
            continue              # the coarse pass was looking at a fringe
        if not gained.any() and not lost.any():
            kind = 'modified'     # the same ink, a different colour
        else:
            kind = classify(here_base, here_target, (0, 0, w, h))
        regions.append(Region(
            kind=kind, box=box,
            base_ink=int(here_base.sum()), target_ink=int(here_target.sum()),
            base=base_crop.copy() if keep_crops else None,
            target=target_crop.copy() if keep_crops else None))
    return Difference(regions, shift, confidence,
                      int(base_coarse.sum()), int(target_coarse.sum()),
                      (width, height))


def _crop_shifted(image, box, shift):
    """The target's crop, moved by the registration, padded with its paper.

    Padded rather than clipped: a box at the edge of the page that the shift
    moves partly off it still has to come back the size of the box, or the
    two crops cannot be compared at all.
    """
    x, y, w, h = box
    dx, dy = shift
    height, width = image.shape[:2]
    out = np.zeros((h, w) + image.shape[2:], image.dtype)
    out[:] = np.array(paper_of(image), image.dtype) if image.ndim == 3 else 0
    sx0, sy0 = x + dx, y + dy
    tx0, ty0 = max(0, -sx0), max(0, -sy0)
    sx0, sy0 = max(0, sx0), max(0, sy0)
    sx1, sy1 = min(width, sx0 + w - tx0), min(height, sy0 + h - ty0)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    out[ty0:ty0 + (sy1 - sy0), tx0:tx0 + (sx1 - sx0)] = image[sy0:sy1, sx0:sx1]
    return out
