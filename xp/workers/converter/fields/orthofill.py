"""Stage 15: coarse-to-fine filling. The user's design, in the user's order.

The user's words, and they are the whole specification:

  "Just try to use a thick one, long enough, to fill it as one, then add
   thinner ones individually, one by one."
  "Why not 2 and then 1?"  -- compute first, place second.
  "Why do you always exceed? ... just calculate it."
  "Just calculate before, then draw them, there should be no exceed."

So there is one rule and the level is the width:

    level k has width w_k, with w_1 from the region and w_{k+1} = w_k / 2

  step 1  COMPUTE  where a stroke of width w_k may stand at all: the centre
          has to be at least w_k / 2 from the boundary, so the legal set is
          `{d >= w_k/2}` for the region's own distance field d. A rib is then
          grown along its direction only while it stays legal, so its two
          ends land where d = w_k / 2.
  step 2  PLACE    seeds down the medial axis of that legal set, evenly
          spaced by one rib width, each rib drawn at exactly w_k. Where the
          axis is a point or has no tangent, the direction is drawn at
          random -- the user's rule, and the only case that needs one.
  step 3  COMPUTE  the gap: the region minus the level's strokes, as geometry
          and including the round caps the renderer will draw. Not "look at
          what got painted".
  step 4  REPEAT   with w_k / 2, until the gap is thinner than the pixel or
          the width reaches it; what is still bare is placed individually,
          one dot per leftover blob, at the place with the most room.
  step 5  DRAW     once, at the end. The level widths and positions are all
          decided before anything is drawn, which is the point.

**Why nothing exceeds.** If the centre stays in `{d >= w/2 + slack}` then
every point of the stroke is within w/2 of a point at distance >= w/2 from
the boundary, so by the triangle inequality its own distance is >= 0: the
stroke is inside the region. The `slack` is the renderer's share -- a width
is drawn at `round(w)` integer pixels, so it can land half a pixel wider
than asked for -- and it is a known number, not a value to search over.

**Why it terminates.** Each level leaves a band of at most w_k / 2 along the
boundary, and the next level is half as wide, so the residue falls by half
each time: a geometric series. At w = 1 px the legal set is the whole region
(every ink pixel is at least 1 px from the paper), so the last level can
place a stroke anywhere anything is left.

What this replaces: a loop that drew, rasterised, looked for what was bare
and drew again. It chased its own rasteriser's error at the edges -- the
missing round caps, mostly -- and spawned tens of thousands of ribs hugging
every boundary. That is the fault the user named:

  "Can't you just calculate the geometric relationships first, and then
   generate the strokes? ... Why the stupid feedback loop?"

Stage 14 (the axis and the radius) and stage 16 (the report) are here too,
because they are the same idea seen from before and after: what the levels
read, and whether what they drew kept the ink.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from xp.core import geometry
from xp.gpu import backend as gpu, kernels, reduce
from xp.render import drawing as render, raster
from xp.workers.converter.fields import centerline, detour, strokejoin
from xp.core.document.strokes import StrokeBuffer, concat

NEIGHBOURS = np.ones((3, 3), np.uint8)

MARCH_STEP = 0.5           # how finely a rib is sampled while growing, in
                           # pixels. Half a pixel: the boundary is known to
                           # the pixel, so sampling finer asks the mask a
                           # question it cannot answer.
MARCH_BLOCK = 64           # steps marched per pass, so a rib that leaves
                           # after two pixels costs two pixels.
MIN_WIDTH = 1.0            # the narrowest stroke the renderer can express.
                           # It is the stopping rule: below this there is
                           # nothing left to say, only dots.
RASTER_SLACK = 0.5         # the renderer's share of "the stroke's actual
                           # width". A stroke of width w is drawn at
                           # round(w) integer pixels on a coordinate rounded
                           # to six decimals, so what lands can be half a
                           # pixel wider than what was asked for. The user's step 1
                           # says to allow for the actual width; this is the
                           # part of the actual width that belongs to the
                           # drawing rather than to the geometry. It is
                           # derived, not searched for: the user was explicit that
                           # a ladder of these values is the wrong shape of
                           # solution ("no exceed -- just calculate it").
FINISH_ROUNDS = 3          # join and cover, turn and turn about, until the
                           # page is covered. Two is what the user's page takes; the
                           # third is there because a join can move an
                           # even-width band by a row and the round after is
                           # what covers it.
ABSORB_ROUNDS = 16         # how many times the detour is offered. A bound,
                           # not a dial: the user's page converges in six and the
                           # loop stops of its own accord when a pass folds
                           # nothing, so the convergence decides and the
                           # bound only keeps a pathological page finite.
MAX_LEVELS = 16            # each level halves the width, so sixteen levels
                           # is 2^-16 of the first: far past the pixel. A
                           # guard against a pathological region, not a
                           # budget.
SEED = 20261010            # where a direction has to be drawn at random it
                           # is drawn the *same* way every run, so a page
                           # converts to the same file twice and a test can
                           # mean something.


@dataclass
class Cancelled(Exception):
    """The fill was told to stop, between levels.

    Phase 3 is the long part of a page -- thirteen seconds of the thirty-six
    -- and a stage boundary is too coarse a place to stop it: choosing a
    stage while the page was filling used to leave the fill running to the
    end of phase 3 before it noticed. A level is the finest place that is
    still a whole thought, so that is where it looks.
    """


class Axis:
    """Stage 14: the seeds, and the radius that says how wide they may be.

    `skeleton` is Zhang-Suen thinning -- one pixel wide and connected, which
    is what gives a rib a tangent to be normal to. `field` is the distance to
    the paper: the number every level reads, both to decide where a stroke of
    a given width may stand and to know when the gap has closed.
    """
    skeleton: np.ndarray
    field: np.ndarray

    @property
    def radius(self) -> np.ndarray:
        return np.where(self.skeleton, self.field, 0.0)


@dataclass
class Report:
    """Stage 16: what the strokes actually cover, measured against the region.

    The user kept this stage on purpose ("6 步 Output Validation 不能省"), and it
    is the only one that can catch the whole of phase 3 going wrong at once:
    if the axis came out cut, a rib grew the wrong way, or a stroke was
    dropped to satisfy the format's two-point minimum, ink goes bare and this
    says so in pixels.
    """
    ink: int = 0
    bare: int = 0
    invented: int = 0
    strokes: int = 0
    points: int = 0
    widest_hole: float = 0.0

    @property
    def bare_share(self) -> float:
        return 0.0 if not self.ink else self.bare / self.ink

    @property
    def invented_share(self) -> float:
        return 0.0 if not self.ink else self.invented / self.ink

    def passed(self, bare_limit=0.003, hole_limit=None) -> bool:
        """Did the drawing keep the ink? `bare_limit` is the user's 0.3%.

        It answers that and nothing else. It cannot tell whether the thing it
        was handed was the right thing to draw -- a page drawn perfectly in
        the wrong places passes. That is why phases 1 and 2 were measured
        separately.
        """
        if self.bare_share > bare_limit:
            return False
        return hole_limit is None or self.widest_hole <= hole_limit

    def line(self) -> str:
        return ('%d strokes, %d points; %.3f%% of the ink left bare '
                '(widest hole %.1f px), %.2f%% drawn outside it'
                % (self.strokes, self.points, 100.0 * self.bare_share,
                   self.widest_hole, 100.0 * self.invented_share))


def medial_axis(mask, field=None) -> Axis:
    """Stage 14, for a caller that wants the whole region's axis at once."""
    if field is None:
        field = geometry.signed_distance(mask)
    return Axis(skeleton=centerline.thin(mask), field=field)


def first_width(mask, field, pen=None) -> float:
    """Step 1's first number: the region's own width, coarsely.

    Twice the mean distance from the ink to the paper. For a stroke of width
    W the distance runs from W/2 at the middle to nothing at the edge, so the
    mean is about W/4 and twice it is W/2: the first level is half as wide as
    the region, which leaves the outer half of the region for the levels
    below. That is what makes the first level cheap and the levels after it
    few.

    Not a percentile and not the maximum: the maximum is the peak of the
    distance field, and the user was explicit that the peaks are not the place to
    look -- "I don't care about peaks, just average" -- because a rule built
    on the peaks cannot say anything about the space between two of them.
    """
    if not mask.any():
        return 0.0
    typical = 2.0 * float(field[mask].mean())
    if pen:
        typical = min(typical, float(pen))
    return max(typical, MIN_WIDTH)


def _within(bare, radius, card=None):
    """Everything within `radius` of a bare pixel.

    Which legal pixels can reach what is still uncovered -- a stroke reaches
    half its own width sideways, so the places that can serve the gap are the
    legal ones within that of it.

    A distance transform and not a structuring element: cv2's
    `MORPH_ELLIPSE` is an approximation of a disc and the transform is the
    disc, and more to the point the card has a transform and does not have
    cv2's ellipse. Both sides answer identically -- `DIST_MASK_PRECISE` is
    exact, as `cupyx`'s is -- so this cannot make a page trace differently
    depending on the hardware.
    """
    if radius <= 0.0:
        return bare if card is None else card['xp'].asarray(bare)
    if card is not None:
        import cupyx.scipy.ndimage as cnd

        away = cnd.distance_transform_edt(~bare, float64_distances=False)
        return away <= radius
    away = cv2.distanceTransform((~bare).view(np.uint8), cv2.DIST_L2,
                                 cv2.DIST_MASK_PRECISE)
    return away <= radius


def _legal(mask, field, width):
    """Step 1: where the centre of a stroke of this width may stand.

    `{d >= w/2 + slack}`. Everything follows from this set: it is where seeds
    may go, and it is what a growing rib stops at, so a rib's ends land
    exactly where its own edge meets the boundary. Nothing here consults a
    rendering.
    """
    return mask & (field >= width / 2.0 + RASTER_SLACK)


def _reach(legal, x, y, nx, ny, labels=None, want=None, card=None):
    """Step 1, one side: how far each seed gets along (nx, ny) before leaving.

    Marched rather than solved, because the boundary is a set of pixels and
    not an equation. Every seed is marched at once, in blocks, and a rib
    stops at the *first* sample that is not legal -- a further patch of legal
    ink across a gap belongs to another rib, not to this one.

    `labels` and `want` are the colour boundary, and they are what let the
    whole page be filled in one pass instead of once per region: a rib grows
    while it is legal **and still in the colour it started in**. One integer
    comparison, in the sample that already reads the distance. Obeying the
    boundary per point is stricter than giving each region its own pass,
    which only assumes it at the edges.
    """
    height, width = legal.shape
    if card is not None:
        # One thread per seed, in CUDA C++ (kernels.py). Bit-identical to the
        # blocks below -- `lrint` rounds half to even, as `np.rint` does -- and
        # it allocates nothing: the numpy version builds six arrays of
        # (live seeds x 64) per block, which at a hundred thousand seeds is
        # most of what it costs.
        return gpu.to_host(kernels.march(card['legal'], card.get('labels'),
                                         want, x, y, nx, ny, MARCH_STEP,
                                         float(height + width)))
    reach = np.zeros(len(x))
    alive = np.arange(len(x))
    base = 0.0
    while len(alive):
        steps = base + MARCH_STEP * np.arange(1, MARCH_BLOCK + 1)
        px = np.rint(x[alive, None] + nx[alive, None] * steps[None, :])
        py = np.rint(y[alive, None] + ny[alive, None] * steps[None, :])
        ok = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        inside = np.zeros(px.shape, bool)
        at = np.nonzero(ok)
        iy, ix = py[at].astype(np.intp), px[at].astype(np.intp)
        here = legal[iy, ix]
        if labels is not None:
            here = here & (labels[iy, ix] == want[alive][at[0]])
        inside[at] = here
        run = np.logical_and.accumulate(inside, axis=1).sum(axis=1)
        reach[alive] = base + MARCH_STEP * run
        alive = alive[run == MARCH_BLOCK]
        base += MARCH_STEP * MARCH_BLOCK
        if base > height + width:          # a crop cannot be crossed twice
            break
    return reach


def _seeds(legal, field, spacing, rng):
    """Step 2's positions: along the boundary of the legal set, evenly.

    **Boundary First.** The seeds are the rim of `{d >= w/2 + slack}`, which
    is the region's own boundary offset inward by exactly as much as the
    stroke needs to stay inside it. That is the user's correction in one line: the
    centre line is not pinned to the boundary -- a round-capped stroke
    centred there would hang half its width outside -- the boundary is the
    *constraint*, and the seeds sit the offset in from it.

    Nothing here extracts a medial axis. The axis is where the inward fronts
    meet, and they meet on their own: a rib grows while it is legal and stops
    when it is not, and two ribs coming from opposite rims of the same stroke
    are stopped by the same set at the same place. So the shock of the inward
    normal flow is still respected, as a consequence rather than as an input.

    **The inward normal is grad d.** It points away from the nearest boundary
    pixel, which is inward, everywhere -- so a hole's rim needs no special
    case and no contour-orientation bookkeeping; it reads its direction out
    of the same array as the outer rim. Taken by central differences at the
    contour's own pixels rather than by a Sobel over the page, which at 800
    DPI would be two more arrays of 690 MB to read a few hundred thousand
    numbers out of.

    `CHAIN_APPROX_NONE` so the chain is every boundary pixel in order, which
    makes arc length along it just the running sum of the steps, and one seed
    per `spacing` of arc is one `searchsorted`.

    Where a piece of the legal set is too small to have a contour -- a single
    pixel, a two-pixel neck -- there is no tangent and no rim to walk, so the
    pixel itself is the seed and the direction is drawn at random. That is the user's
    rule 3's fallback, and it is why a lone blob needs no rule of its own.
    """
    if not legal.any():
        return (np.zeros(0),) * 4
    height, page_width = legal.shape
    contours, _hierarchy = cv2.findContours(legal.view(np.uint8),
                                            cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_NONE)
    if not contours:
        return (np.zeros(0),) * 4

    # All the contours at once. A level of the user's page has tens of thousands of
    # them and the loop that was here did six numpy calls on each, which is
    # the fault this project keeps finding: the work is small and the
    # arranging of it is not.
    #
    # Arc length is a running sum, so one `cumsum` over every contour laid
    # end to end gives each contour's arc as an offset from its own start --
    # and because every step is non-negative the whole array is sorted, so
    # one `searchsorted` places every seed of every contour.
    counts = np.fromiter((len(one) for one in contours), np.int64, len(contours))
    points = np.concatenate([one.reshape(-1, 2) for one in contours]).astype(np.float64)
    begins = np.zeros(len(counts), np.int64)
    np.cumsum(counts[:-1], out=begins[1:])
    step = np.empty(len(points))
    step[0] = 0.0
    step[1:] = np.hypot(points[1:, 0] - points[:-1, 0],
                        points[1:, 1] - points[:-1, 1])
    step[begins] = 0.0                      # no step into a contour's first point
    arc = np.cumsum(step)
    span = arc[begins + counts - 1] - arc[begins]
    # One seed every `spacing` of arc, per contour, and always its first
    # point -- which is the only seed a contour too short to measure gets.
    # `np.arange(0, span + spacing, spacing)` is what the loop asked for and
    # it yields `ceil((span + spacing) / spacing)` values, not `floor + 1`:
    # one target past the end, which clips to the contour's last point and
    # seeds it. Getting that wrong moved a level's seeds enough to put ten
    # thousand more strokes on the page.
    wanted = np.maximum(1, np.ceil((span + spacing) / spacing).astype(np.int64))
    which = np.repeat(np.arange(len(counts)), wanted)
    along = (np.arange(len(which)) - np.repeat(np.cumsum(wanted) - wanted, wanted))
    target = arc[begins][which] + along * spacing
    take = np.searchsorted(arc, target, side='right') - 1
    np.clip(take, begins[which], (begins + counts - 1)[which], out=take)
    take = np.unique(take)
    x = points[take, 0]
    y = points[take, 1]
    if height < 3 or page_width < 3:
        # A crop one or two pixels across has no room for a central
        # difference, and nothing to point at either: every direction is as
        # good. The user's rule 3's fallback, for the same reason a lone blob needs
        # it -- and the user's page has eleven thousand regions of two pixels, so
        # this is the common case, not the corner.
        angle = rng.uniform(0.0, np.pi, len(x))
        return x, y, np.cos(angle), np.sin(angle)
    ix = np.clip(x.astype(np.intp), 1, page_width - 2)
    iy = np.clip(y.astype(np.intp), 1, height - 2)
    nx = field[iy, ix + 1] - field[iy, ix - 1]
    ny = field[iy + 1, ix] - field[iy - 1, ix]
    length = np.hypot(nx, ny)
    flat = length < 1e-6
    if flat.any():
        # A rim pixel whose field is flat both ways sits in a one-pixel neck:
        # every direction is as good, so take one at random rather than leave
        # a zero vector. Same rule, same reason, as a lone blob.
        angle = rng.uniform(0.0, np.pi, int(flat.sum()))
        nx = nx.copy()
        ny = ny.copy()
        nx[flat] = np.cos(angle)
        ny[flat] = np.sin(angle)
        length = np.hypot(nx, ny)
    return x, y, nx / length, ny / length


def _ribs(legal, seeds, width, palette, labels=None, card=None):
    """Steps 1 and 2 together: grow to where the level's edge meets the
    boundary, and draw at the level's width.

    One width for the whole level -- not per rib, and not the pen's width.
    That is the correction the user made twice over: the rib that fills a gap has
    to be as thin as the gap, and the way to know how thin is to work out
    where a stroke of that width could stand, which is step 1. A rib drawn at
    the pen's width into a space measured at two pixels is the whole reason
    this used to draw outside the ink.
    """
    x, y, nx, ny = seeds
    if not len(x):
        return StrokeBuffer.empty(np, palette)
    want = None
    if labels is not None:
        # The colour a rib belongs to is the colour where it starts. It then
        # grows only inside that colour, so the stroke cannot be a blend of
        # two and there is nothing to decide later.
        #
        # And the label *is* the index into the palette, so there is no
        # lookup: `colours.get(int(v))` used to run once per rib, two hundred
        # thousand Python calls to say what the array already said.
        want = labels[y.astype(np.intp), x.astype(np.intp)].astype(np.int32)
    # No second subtraction of the half width. `_reach` marches inside
    # `legal`, which *is* `{d >= w/2 + slack}`, so the clearance is already in
    # the set it walks: taking w/2 off again stopped every rib a whole width
    # short of where it was allowed to go, and left the middle of every thick
    # stroke bare.
    plus = _reach(legal, x, y, nx, ny, labels, want, card)
    minus = _reach(legal, x, y, -nx, -ny, labels, want, card)
    # One array for every rib's two ends, not one `np.array` per rib: a page
    # is 140,000 ribs and the old loop made 436,849 of those calls, which
    # cost more than the march they describe. Each Segment then holds a view
    # into this one block.
    # A rib that could not grow is its own seed: one point, which
    # `xopp_io.make_stroke_element` stores twice -- never dropped, because a
    # mark of one point is still a mark. Its edge is still inside the region:
    # a stroke of width w about a centre at distance >= w/2 from the boundary
    # cannot be anything else.
    #
    # Built straight into the flat arrays. Every rib used to be an object
    # holding a view into a block built for the purpose, and the block is all
    # that was ever wanted.
    solo = (plus <= 0.0) & (minus <= 0.0)
    counts = np.where(solo, 1, 2).astype(np.int64)
    starts = np.zeros(len(x), np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    total = int(counts.sum())
    xs = np.empty(total, np.float64)
    ys = np.empty(total, np.float64)
    xs[starts] = np.where(solo, x, x - nx * minus)
    ys[starts] = np.where(solo, y, y - ny * minus)
    pair = starts[~solo] + 1
    xs[pair] = (x + nx * plus)[~solo]
    ys[pair] = (y + ny * plus)[~solo]
    if want is None:
        want = np.zeros(len(x), np.int32)
    return StrokeBuffer(np, xs, ys, starts, counts,
                        np.full(len(x), float(width)), want, palette)


def _dots(bare, field, palette, labels=None):
    """Rule 5: cover what the lines left, with dots of a fitting size.

    No distance-peak search, and no one-dot-per-blob either -- that was the
    version before this and it could not work: what the lines leave is a
    shell a pixel or two deep running along every edge, which is *one*
    connected blob per letter, so one dot per blob covered a dot's worth of
    it and reported the rest bare.

    The rule instead: a bare pixel's legal dot is the disc of radius d there,
    which is rule 4 again applied to a stroke of no length. So the only
    question is which bare pixels get one, and the answer is a grid whose
    cell is the dot's own radius -- one dot per cell. Every dot is at least as
    wide as its cell, so the cell it sits in is covered and its neighbours
    overlap. That is one `unique` over a packed key: no ordering, no greed,
    no search, and the same answer every run.
    """
    if not bare.any():
        return StrokeBuffer.empty(np, palette)
    ys, xs = np.nonzero(bare)
    room = field[ys, xs].astype(np.float64)
    # `2d - 1` and not `2d`: the renderer rounds a width up by at most half a
    # pixel on each side, and that is a known number, not a guess.
    thick = np.maximum(2.0 * room - 2.0 * RASTER_SLACK, MIN_WIDTH)
    cell = np.maximum(np.rint(thick / 2.0), 1.0).astype(np.int64)
    key = (((ys // cell).astype(np.int64) * 0x40000 + (xs // cell))
           * 0x400 + cell)
    _seen, first = np.unique(key, return_index=True)
    tint = (labels[ys[first], xs[first]].astype(np.int32) if labels is not None
            else np.zeros(len(first), np.int32))
    return StrokeBuffer(np, xs[first].astype(np.float64),
                        ys[first].astype(np.float64),
                        np.arange(len(first), dtype=np.int64),
                        np.ones(len(first), np.int64),
                        thick[first], tint, palette)


def coverage(segments, mask, line_width=1.0, max_coord_str_len=30000):
    """The strokes as pixels, drawn the way Xournal++ draws them."""
    if not segments:
        return np.zeros_like(mask)
    # segment_ink takes (width, height), not a numpy shape.
    return render.segment_ink(segments, (mask.shape[1], mask.shape[0]),
                              line_width, max_coord_str_len)


def _only_the_useful(buffer, drawn, shape, line_width, card=None):
    """Drop the ribs of this level that draw nothing not already drawn.

    The user's words: avoid repeated and overlapping strokes. A level lays its ribs
    one width apart along the boundary and marches each one both ways, so
    wherever the boundary curves -- which is everywhere in handwriting -- the
    ribs beside each other cover much of the same ink. Some of them end up
    covering nothing at all that is not already covered, and such a stroke is
    still a stroke in the file, in Xournal++'s selection, and in every pass
    that comes after it.

    `kernels.useful` gives every pixel to the lowest-numbered rib that covers
    it and that no earlier level had covered, so a rib owning none can go and
    the drawing is pixel for pixel what it was. It is not a tolerance and not
    a threshold: zero is zero.
    """
    if card is not None:
        held = gpu.to_host(kernels.useful_of(buffer, drawn, shape))
    else:
        held = raster.useful_of(buffer, drawn, (shape[1], shape[0]))
    if not len(held) or held.all():
        return buffer
    return buffer.take(held > 0)


def _cover_on_card(buffer, shape, line_width):
    """The same band, drawn straight into device memory.

    `raster.flatten` still runs on the CPU -- it is one concatenate over the
    stroke list, 0.088s for a page -- and only the flat arrays cross, which
    are the points and not the page.
    """
    return kernels.band_of(buffer, shape)


def _widest(bare):
    """The thickest thing left bare, in pixels.

    On the card when there is one: a full-page distance transform, asked
    twice a fill, 1.42s of cv2 against 0.06s of `cupyx`. Both sides answer
    the same -- `DIST_MASK_PRECISE` is exact, as `cupyx`'s is -- and that is
    the point: this number decides when the level ladder stops, so a page
    traced with a card and without one must stop at the same place. It did
    not, while the fallback was the 5x5 approximation: the same page came out
    with 140,203 strokes or 138,912 depending on whether the card had memory
    free, which is not a difference anything should have.
    """
    if not bool(bare.any()):
        return 0.0
    if gpu.GPU_AVAILABLE:
        try:
            import cupyx.scipy.ndimage as cnd

            on = gpu.xp.asarray(bare)
            far = cnd.distance_transform_edt(on, float64_distances=False)
            answer = 2.0 * float(far.max())
            del far, on
            return answer
        except Exception:
            pass
    return 2.0 * float(cv2.distanceTransform(bare.view(np.uint8), cv2.DIST_L2,
                                             cv2.DIST_MASK_PRECISE).max())


def fill_page(mask, labels, colours, Segment, pen=None, field=None, spacing=1.0,
              line_width=1.0, max_coord_str_len=30000, levels=None,
              cancel=None, merge_gap=0.0, merge_colour=0.0,
              max_points=1246) -> List:
    """Stages 14 and 15 for a whole page, in one pass.

    `labels` says which colour each ink pixel is and `colours` maps an index
    to the colour itself, so a stroke takes its colour from where it stands.
    There is no object per piece of ink: building 121,064 of them took 27
    seconds on the user's page before a stroke was drawn, and nothing in the user's design
    asks for them.

    The colour boundary is obeyed point by point inside the march rather than
    by running each piece separately -- see `_reach`.
    """
    return fill(mask, Segment, pen=pen, field=field, spacing=spacing,
                line_width=line_width, max_coord_str_len=max_coord_str_len,
                levels=MAX_LEVELS if levels is None else levels,
                labels=labels, colours=colours, cancel=cancel,
                merge_gap=merge_gap, merge_colour=merge_colour,
                max_points=max_points)


def fill(mask, Segment, pen=None, field=None, spacing=1.0, colour=None,
         line_width=1.0, max_coord_str_len=30000, levels=MAX_LEVELS,
         labels=None, colours=None, cancel=None, merge_gap=0.0,
         merge_colour=0.0, max_points=1246) -> List:
    """Stage 15, once per region.

    `Segment` is passed in rather than imported so this module stays below
    `convert` and the import does not go in a circle. `spacing` is the seed
    spacing as a fraction of the level's own width: 1.0 lays the ribs of a
    level side by side across the axis.
    """
    if not mask.any():
        return []
    if field is None:
        field = geometry.signed_distance(mask)
    if pen is None:
        pen = centerline.pen_width(mask)
    rng = np.random.default_rng(SEED)

    width = first_width(mask, field, pen)
    # The palette is the colour table itself: a cluster label is an index
    # into it, so a stroke carries the label and nothing has to be looked up.
    if colours is None:
        palette = [colour]
    elif hasattr(colours, 'get'):
        palette = [colours.get(i) for i in range(max(colours) + 1)]
    else:
        palette = list(colours)
    out: List = []
    # The page, once, for the whole fill. `labels` is the expensive one --
    # 173 million int32 -- and it never changes, so it goes up here and not
    # per level. What comes *down* each level is one bool page, because
    # `findContours` walks a boundary in order and that is not a parallel
    # job; everything else stays where it was computed.
    card = None
    if kernels.ready():
        # The page itself, once. `labels` and `field` never change, and the
        # rest of the loop's arrays are made here and stay here: the only
        # thing that comes back each level is the one bool page
        # `findContours` reads, because walking a boundary in order is not a
        # parallel job.
        card = {'xp': gpu.xp,
                'labels': gpu.xp.asarray(labels) if labels is not None else None,
                'mask': gpu.xp.asarray(mask),
                'field': gpu.xp.asarray(field, gpu.xp.float32)}
    gap = mask
    # What has been drawn, kept and added to. Each level used to re-rasterize
    # *every stroke so far* to find out what was still bare, which is
    # quadratic: four levels over a hundred and eighty thousand strokes draws
    # them four times. Drawing is a union, so only the new strokes have to be
    # drawn and OR-ed in.
    drawn = (gpu.xp.zeros(mask.shape, gpu.xp.uint8) if card is not None
             else np.zeros_like(mask))
    hole = gpu.xp.asarray(mask) if card is not None else mask
    for _ in range(max(1, int(levels))):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if width < MIN_WIDTH or not bool(hole.any()):
            break
        if card is not None:
            # The legal set, computed where the page already is. It used to
            # be built on the host out of two 691 MB arrays and then handed
            # back up for the march.
            legal = card['mask'] & (card['field'] >= width / 2.0 + RASTER_SLACK)
            card['legal'] = legal.view(gpu.xp.uint8)
        else:
            legal = _legal(mask, field, width)       # step 1: compute
        # Where a rib of this width may stand *and* can reach what is still
        # bare. Intersecting `legal` with the gap itself was self-defeating:
        # what is left is always shallower than the set a stroke this wide
        # may stand in, so the two barely meet and the rim never got seeded.
        # A stroke reaches half its own width sideways, so the places that
        # can serve the gap are the legal ones within that of it.
        # `ceil`, as the structuring element this replaces had to be: it
        # selects which seeds are worth trying and legality is enforced
        # separately, so being half a pixel generous costs nothing and being
        # half a pixel mean costs a level.
        reach = legal & _within(hole, float(np.ceil(width / 2.0)), card)
        need = gpu.to_host(reach).view(bool) if card is not None else reach
        before = int(hole.sum())
        made = StrokeBuffer.empty(np, palette)
        if need.any():
            step = max(1.0, float(spacing) * width)
            made = _ribs(legal, _seeds(need, field, step, rng),   # step 2: place
                         width, palette, labels, card)
            if len(made):
                # On whichever side the page is: the card may make this
                # faster and may not make it different.
                made = _only_the_useful(
                    made,
                    drawn if card is not None else drawn.view(np.uint8),
                    mask.shape, line_width, card)
                out.append(made)
                if card is not None:
                    drawn |= _cover_on_card(made, mask.shape, line_width)
                    hole = card['mask'] & ~drawn.view(bool)
                else:
                    drawn |= raster.ink_of(made, (mask.shape[1], mask.shape[0]))
                    hole = mask & ~drawn
        # A level has to earn its strokes. Measured on the sample page, the
        # levels went 1,122 ribs for 157,359 bare pixels, then 9,213 for
        # 147,217, then 11,489 for 6,469, then **3,902 ribs for 29 pixels**
        # -- the thin levels spend a stroke each on ground a dot covers for
        # nothing, because a narrow rib is also a short one. So a level
        # stands only if it removed more bare pixels than it drew strokes,
        # and what is left goes to step 4's dots. Not a dial: the criterion
        # is a stroke against a pixel, which is the exchange rate the whole
        # stage is about.
        if len(made) and (before - int(hole.sum())) < len(made):
            out = out[:-1]
            # Taking strokes back is the one thing a union cannot do, so this
            # is the one place the whole drawing is rasterized again. It
            # happens at most once -- the level that fails is the last one --
            # which is why keeping `drawn` as a union the rest of the time is
            # safe and not merely cheaper.
            held = concat(out, np)
            if card is not None:
                drawn = _cover_on_card(held, mask.shape, line_width)
                hole = card['mask'] & ~drawn.view(bool)
            else:
                drawn = raster.ink_of(held, (mask.shape[1], mask.shape[0]))
                hole = mask & ~drawn
            break
        if _widest(hole) < MIN_WIDTH or not bool(hole.any()):   # step 4: check
            break
        # The next width comes from what is still bare, not from halving. The
        # gap's own mean depth is the widest stroke that can stand across it
        # -- the formula reads the region instead of being told a ratio, and
        # it cannot stall, because a width taken from the gap always admits
        # seeds that reach the gap.
        #
        # Taken where the page already is, by the summation order written
        # down in `xp/reduce.py`. A mean is a reduction, and numpy and cupy
        # do not add three and a half million floats in the same order --
        # this number decides the next level's width, so the difference comes
        # out as a page with a thousand more strokes on one machine than on
        # another. The order is the specification; both sides follow it.
        following = reduce.next_width(
            card['field'] if card is not None else field, hole, card)
        width = min(width / 2.0, following) if following >= MIN_WIDTH else following
    on_card = None
    if card is not None:
        drawn = gpu.to_host(drawn).view(bool)
        # The field stays until the finishing passes are done with it. The
        # join and the detour both ask rule 4 about a bridge, which reads
        # this array, and handing it back here meant 691 MB going up again
        # for every one of those calls -- the fill went from 3.9s to 5.6s on
        # that alone. Everything else goes now.
        on_card = card.get('field')
        card.clear()
        del hole
        gpu.xp.get_default_memory_pool().free_all_blocks()
    held = concat(out, np)
    # The width a stroke is stored with, made the width it is drawn with.
    # Every rasterizer here takes `max(1, rint(w))` -- a stroke of 5.76 px is
    # drawn at 6 and always was -- so this changes no pixel of the drawing.
    # What it changes is who may join whom: one `<stroke>` carries one width,
    # and ten widths that round to six were keeping strokes apart over a
    # difference nothing could draw.
    if len(held):
        held = held.with_widths(np.maximum(1.0, np.rint(held.widths)))
    # Join, cover what is left, and offer the cover a join of its own.
    #
    # The join has to come before the dots -- a chain reverses every other
    # rib, a reversed band of even width sits one row over, and covering what
    # moves is exactly what step 4 is for. But it was *only* before them, so
    # the dots, which are strokes like any other, were never offered one:
    # 25,061 of the 25,187 single-point strokes on the user's page were width-1
    # dots, and the only things they could have joined to were each other.
    # Turn and turn about until nothing is bare: on the user's page that is two
    # rounds and 43,957 strokes become 30,800, with the ink unchanged.
    size = (mask.shape[1], mask.shape[0])
    for _ in range(FINISH_ROUNDS):
        if merge_gap > 0.0 and len(held) > 1:
            held = strokejoin.join_buffer(held, on_card if on_card is not None
                                          else field, merge_gap, merge_colour,
                                          max(2, int(max_points)), RASTER_SLACK)
            drawn = raster.ink_of(held, size)
        gap = mask & ~drawn
        if not gap.any():
            break
        more = _dots(gap, field, palette, labels)
        if not len(more):
            break
        held = concat([held, more], np)
        drawn = drawn | raster.ink_of(more, size)
    else:
        # Out of rounds with ink still bare: cover it and stop. Rule 1 is not
        # negotiable and this costs a handful of strokes.
        gap = mask & ~drawn
        if gap.any():
            held = concat([held, _dots(gap, field, palette, labels)], np)
    if merge_gap > 0.0 and len(held) > 1:
        # And last, the user's first instruction: do not keep a separate stroke for
        # something an existing one can reach. The join connects ends to
        # ends, and after it has converged there are five such bridges left
        # on the user's page -- an end sits on the *rim* of the legal set, so a
        # bridge between two of them has to cross the gap outside the ink and
        # rule 4 refuses it. A point in the middle of a stroke is deep in the
        # ink, and from a rim to a depth the bridge stays inside: 98,966
        # legal bridges over 13,405 strokes. Folding them in, out and back,
        # is 30,800 strokes -> 23,944.
        #
        # And then again, to a fixed point. One pass forbids a host from also
        # being a source, so that the answer cannot depend on which fold was
        # decided first; across passes there is no such conflict, and a host
        # of pass one is a source of pass two. That is how a branch is walked
        # down a layer at a time, and it converges on the user's page in six passes:
        # 23,944 -> 22,729 -> 22,546 -> 22,488 -> 22,467 -> 22,460, which is
        # within 236 strokes of the 22,224 that a perfect out-and-back
        # traversal of every branch would give. 0.70s, and not one pixel of
        # the coverage moves.
        field_now = on_card if on_card is not None else field
        for _ in range(ABSORB_ROUNDS):
            fewer = detour.absorb(held, field_now, merge_gap, merge_colour,
                                  RASTER_SLACK, max(2, int(max_points)))
            if len(fewer) == len(held):
                break
            held = fewer
    if on_card is not None:
        # Now it can go. A document is sixty-one pages in a row, and a pool
        # that only ever grows is how page one is quick and page four is not.
        del on_card
        gpu.xp.get_default_memory_pool().free_all_blocks()
    return held.to_segments(Segment)            # step 5: drawn once


def validate(segments, mask, line_width=1.0, max_coord_str_len=30000) -> Report:
    """Stage 16. The user asked for this one by name and would not have it dropped."""
    report = Report(ink=int(mask.sum()), strokes=len(segments),
                    points=sum(len(np.asarray(s.points)) for s in segments))
    drawn = coverage(segments, mask, line_width, max_coord_str_len)
    bare = mask & ~drawn
    report.bare = int(bare.sum())
    report.invented = int((drawn & ~mask).sum())
    if report.bare:
        report.widest_hole = _widest(bare)
    # <~ it used to count how many regions held a bare pixel. There are
    #    no region objects to count: the map is a label image, and what
    #    a bare pixel is worth is already said in pixels and in the
    #    widest hole. ~>
    return report
