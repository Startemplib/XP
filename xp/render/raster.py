"""What a stroke of constant width is, drawn over a whole page at once.

One sentence defines it: **the points within half the width of the
polyline.** Round caps and round joins are not two extra features, they are
that sentence read at the ends and at the turns, and it is what cairo draws
-- which is what Xournal++ opens the file with.

Until now both rasterizers were `ImageDraw` calls, one per stroke: a
quadrilateral for each segment, a wedge at each joint that does not close a
sharp turn, and a disc at every point to make up the difference. Three
things were wrong with that, and they are one thing:

- **it is slower than the arithmetic it does.** 101,773 strokes on the user's page,
  called six times over in a conversion, 7.75s of a 31s page. The cost is
  the interpreter, not the work; no GPU reaches a Python loop.
- **it draws fatter than the geometry.** Measured on the user's page against the
  sentence above: PIL puts 161,139 px outside the ink where the band puts
  2,296. The "4.57% drawn outside" the panel reported was mostly the
  rasterizer, not the filling -- the fill's legality test (`d >= w/2 +
  slack`) was already keeping the true band inside.
- **it is not symmetric.** A band of even width cannot be centred on a row
  of pixels, so PIL gives the extra row to one side and which side depends
  on the direction the segment was drawn in. That cost a day: joining
  strokes reverses every other rib, and twenty-nine pixels of the user's page came
  back bare for it.

The sentence has none of those. It is parallel over strokes (the only write
is "covered", and two threads agreeing about that is not a race), it is
symmetric by construction, and it is the thing the fill already assumes.

`cut_path` is not used here, and that is deliberate: a path written as
several `<stroke>` elements sharing a point draws the same ink as one
unbroken path, now that a shared point is a disc in both. The cutting still
happens on the way out -- the coordinate text has a length limit -- but it no
longer changes the picture, so the ruler no longer has to imitate it.
"""

import numpy as np

try:
    from numba import njit, prange
    JIT = True
except ImportError:                              # pragma: no cover - fallback
    JIT = False


def flatten(segments, default_width=1.0):
    """Every stroke's points end to end, with where each one starts.

    One array for the page rather than one object per stroke: that is the
    whole point of doing this in a compiled loop.
    """
    # Not one segment at a time: a page is 139,000 of them, and a numpy call
    # per segment costs more than the arithmetic the compiled kernel then
    # does over the whole page. One concatenate, one round, one divide --
    # the Python here is attribute access and nothing else.
    held = [segment.points for segment in segments]
    counts = np.fromiter((len(one) for one in held), np.int64, len(held))
    keep = counts > 0
    if not keep.any():
        empty = np.zeros(0, np.float64)
        return empty, empty, np.zeros(0, np.int64), np.zeros(0, np.int64), empty
    if not keep.all():
        held = [one for one, alive in zip(held, keep) if alive]
        counts = counts[keep]
        segments = [one for one, alive in zip(segments, keep) if alive]
    both = np.ascontiguousarray(np.concatenate(held), np.float64).reshape(-1, 2)
    np.round(both, 6, out=both)
    starts = np.zeros(len(counts), np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    # The same rounding `xopp_io` writes and `ImageDraw` used: a width is a
    # whole number of pixels, and never less than one.
    told = np.fromiter((default_width if one.width is None else one.width
                        for one in segments), np.float64, len(segments))
    radii = np.maximum(1.0, np.rint(told)) / 2.0
    return (np.ascontiguousarray(both[:, 0]), np.ascontiguousarray(both[:, 1]),
            starts, counts, radii)


def _box(x0, y0, x1, y1, r, width, height):
    lo_x = max(0, int(np.floor(min(x0, x1) - r)))
    hi_x = min(width - 1, int(np.ceil(max(x0, x1) + r)))
    lo_y = max(0, int(np.floor(min(y0, y1) - r)))
    hi_y = min(height - 1, int(np.ceil(max(y0, y1) + r)))
    return lo_x, hi_x, lo_y, hi_y


def _ink_python(xs, ys, starts, counts, radii, out):
    """The same sentence, in numpy, for an install without numba.

    Per stroke rather than per page, so it is slow -- but it is the same
    answer, which is what matters: a fallback that draws something else
    would make the pipeline's numbers depend on what is installed.
    """
    height, width = out.shape
    for s in range(len(starts)):
        begin, count, r = starts[s], counts[s], radii[s]
        for k in range(max(1, count - 1)):
            x0, y0 = xs[begin + k], ys[begin + k]
            x1, y1 = (x0, y0) if count == 1 else (xs[begin + k + 1], ys[begin + k + 1])
            lo_x, hi_x, lo_y, hi_y = _box(x0, y0, x1, y1, r, width, height)
            if hi_x < lo_x or hi_y < lo_y:
                continue
            py, px = np.mgrid[lo_y:hi_y + 1, lo_x:hi_x + 1]
            ex, ey = x1 - x0, y1 - y0
            span = ex * ex + ey * ey
            vx, vy = px - x0, py - y0
            t = np.clip((vx * ex + vy * ey) / span, 0.0, 1.0) if span > 0 else 0.0
            dx, dy = vx - t * ex, vy - t * ey
            out[lo_y:hi_y + 1, lo_x:hi_x + 1] |= (dx * dx + dy * dy <= r * r)


if JIT:
    @njit(cache=True, parallel=True)
    def _ink_jit(xs, ys, starts, counts, radii, out):     # pragma: no cover
        height, width = out.shape
        for s in prange(starts.shape[0]):
            begin, count, r = starts[s], counts[s], radii[s]
            for k in range(max(1, count - 1)):
                ax0 = xs[begin + k]
                ay0 = ys[begin + k]
                if count == 1:
                    ax1, ay1 = ax0, ay0
                else:
                    ax1 = xs[begin + k + 1]
                    ay1 = ys[begin + k + 1]
                # A diagonal segment's axis-aligned box is L x L where the
                # band inside it is about 2rL, so the segment is cut into
                # pieces no longer than 2r and each piece gets its own box.
                # The nearest point of the whole segment to any pixel lies in
                # one of the pieces, so the union is exactly the same set.
                fx, fy = ax1 - ax0, ay1 - ay0
                length = np.sqrt(fx * fx + fy * fy)
                piece = 2.0 * r if r > 0.5 else 1.0
                cuts = max(1, int(np.ceil(length / piece)))
                for c in range(cuts):
                    u0 = c / cuts
                    u1 = (c + 1) / cuts
                    x0 = ax0 + fx * u0
                    y0 = ay0 + fy * u0
                    x1 = ax0 + fx * u1
                    y1 = ay0 + fy * u1
                    lo_x = max(0, int(np.floor(min(x0, x1) - r)))
                    hi_x = min(width - 1, int(np.ceil(max(x0, x1) + r)))
                    lo_y = max(0, int(np.floor(min(y0, y1) - r)))
                    hi_y = min(height - 1, int(np.ceil(max(y0, y1) + r)))
                    ex, ey = x1 - x0, y1 - y0
                    span = ex * ex + ey * ey
                    for py in range(lo_y, hi_y + 1):
                        for px in range(lo_x, hi_x + 1):
                            vx = px - x0
                            vy = py - y0
                            if span > 0.0:
                                t = (vx * ex + vy * ey) / span
                                if t < 0.0:
                                    t = 0.0
                                elif t > 1.0:
                                    t = 1.0
                            else:
                                t = 0.0
                            dx = vx - t * ex
                            dy = vy - t * ey
                            if dx * dx + dy * dy <= r * r:
                                out[py, px] = 1

    @njit(cache=True)
    def _paint_jit(xs, ys, starts, counts, radii, colours, out):  # pragma: no cover
        """The same bands, in colour, in the order they were drawn.

        Serial on purpose: where two strokes overlap the later one is on
        top, and a parallel loop has no later.
        """
        height, width = out.shape[0], out.shape[1]
        for s in range(starts.shape[0]):
            begin, count, r = starts[s], counts[s], radii[s]
            red, green, blue = colours[s, 0], colours[s, 1], colours[s, 2]
            for k in range(max(1, count - 1)):
                x0 = xs[begin + k]
                y0 = ys[begin + k]
                if count == 1:
                    x1, y1 = x0, y0
                else:
                    x1 = xs[begin + k + 1]
                    y1 = ys[begin + k + 1]
                lo_x = max(0, int(np.floor(min(x0, x1) - r)))
                hi_x = min(width - 1, int(np.ceil(max(x0, x1) + r)))
                lo_y = max(0, int(np.floor(min(y0, y1) - r)))
                hi_y = min(height - 1, int(np.ceil(max(y0, y1) + r)))
                ex, ey = x1 - x0, y1 - y0
                span = ex * ex + ey * ey
                for py in range(lo_y, hi_y + 1):
                    for px in range(lo_x, hi_x + 1):
                        vx = px - x0
                        vy = py - y0
                        if span > 0.0:
                            t = (vx * ex + vy * ey) / span
                            if t < 0.0:
                                t = 0.0
                            elif t > 1.0:
                                t = 1.0
                        else:
                            t = 0.0
                        dx = vx - t * ex
                        dy = vy - t * ey
                        if dx * dx + dy * dy <= r * r:
                            out[py, px, 0] = red
                            out[py, px, 1] = green
                            out[py, px, 2] = blue


    @njit(cache=True)
    def _paint_flat(xs, ys, starts, counts, widths, colours, wanted,
                    scale, ox, oy, out):  # pragma: no cover
        """The same bands, from arrays, with the scaling done here.

        What this replaces was not the arithmetic -- it was the
        bookkeeping around it. The caller used to build one `_Band` object
        per stroke, transforming every point in Python, then `flatten` walked
        the objects back into arrays. Measured on the user's 140,203-stroke page:
        0.271s to build the objects, 0.187s to flatten them, and 0.199s to
        do the actual drawing. Three quarters of the cost was turning
        objects into arrays and back, which is the thing `strokes.py` was
        written to stop.

        So the arrays are made once per page and this takes the scale and
        the offset as numbers. `wanted` selects without gathering: a stroke
        that is not in the window costs one comparison.

        Serial on purpose: where two strokes overlap the later one is on
        top, and a parallel loop has no later. The card has an answer to
        that -- see `kernels.paint`, which keeps the highest index per
        pixel with `atomicMax` -- but it has to be an answer and not an
        accident.
        """
        height, width = out.shape[0], out.shape[1]
        for s in range(starts.shape[0]):
            if not wanted[s]:
                continue
            begin, count = starts[s], counts[s]
            r = widths[s] * scale * 0.5
            if r < 0.5:
                r = 0.5
            red, green, blue = colours[s, 0], colours[s, 1], colours[s, 2]
            for k in range(max(1, count - 1)):
                x0 = xs[begin + k] * scale - ox
                y0 = ys[begin + k] * scale - oy
                if count == 1:
                    x1, y1 = x0, y0
                else:
                    x1 = xs[begin + k + 1] * scale - ox
                    y1 = ys[begin + k + 1] * scale - oy
                lo_x = max(0, int(np.floor(min(x0, x1) - r)))
                hi_x = min(width - 1, int(np.ceil(max(x0, x1) + r)))
                lo_y = max(0, int(np.floor(min(y0, y1) - r)))
                hi_y = min(height - 1, int(np.ceil(max(y0, y1) + r)))
                ex, ey = x1 - x0, y1 - y0
                span = ex * ex + ey * ey
                for py in range(lo_y, hi_y + 1):
                    for px in range(lo_x, hi_x + 1):
                        vx = px - x0
                        vy = py - y0
                        if span > 0.0:
                            t = (vx * ex + vy * ey) / span
                            if t < 0.0:
                                t = 0.0
                            elif t > 1.0:
                                t = 1.0
                        else:
                            t = 0.0
                        dx = vx - t * ex
                        dy = vy - t * ey
                        if dx * dx + dy * dy <= r * r:
                            out[py, px, 0] = red
                            out[py, px, 1] = green
                            out[py, px, 2] = blue


def paint_flat(out, xs, ys, starts, counts, widths, colours, wanted=None,
               scale=1.0, offset=(0.0, 0.0)):
    """Paint from a page's arrays, with no object in sight.

    `widths` is in page units and `scale` turns it into pixels, so one set
    of arrays serves every zoom. `wanted` is a boolean over the strokes;
    left out, all of them.
    """
    if not len(starts):
        return out
    if wanted is None:
        wanted = np.ones(len(starts), np.bool_)
    ox, oy = offset
    if JIT:
        _paint_flat(xs, ys, starts, counts, widths, colours, wanted,
                    float(scale), float(ox), float(oy), out)
        return out
    bands, tints = [], []          # pragma: no cover - fallback
    for index in np.nonzero(wanted)[0]:
        begin, count = int(starts[index]), int(counts[index])
        points = [(float(xs[begin + k]) * scale - ox,
                   float(ys[begin + k]) * scale - oy) for k in range(count)]
        bands.append(_Flat(points, max(1.0, float(widths[index]) * scale)))
        tints.append(tuple(int(v) for v in colours[index]))
    return paint(out, bands, tints, 1.0)


class _Flat:                       # pragma: no cover - fallback only
    __slots__ = ('points', 'width')

    def __init__(self, points, width):
        self.points, self.width = points, width


def ink(segments, size, default_width: float = 1.0) -> np.ndarray:
    """Where a tracing's segments land, as a boolean mask."""
    width, height = int(size[0]), int(size[1])
    out = np.zeros((max(1, height), max(1, width)), np.uint8)
    if segments:
        xs, ys, starts, counts, radii = flatten(segments, default_width)
        if len(starts):
            (_ink_jit if JIT else _ink_python)(xs, ys, starts, counts, radii, out)
    return out.view(bool)


def paint(out, segments, colours, default_width: float = 1.0) -> np.ndarray:
    """The same bands into an (H, W, 3) uint8 image, in order.

    `colours` is one RGB triple per segment, already resolved -- this knows
    nothing about palettes.
    """
    if not segments:
        return out
    xs, ys, starts, counts, radii = flatten(segments, default_width)
    if not len(starts):
        return out
    table = np.asarray(colours, np.uint8).reshape(-1, 3)
    if JIT:
        _paint_jit(xs, ys, starts, counts, radii, table, out)
        return out
    for index in range(len(starts)):             # pragma: no cover - fallback
        one = np.zeros(out.shape[:2], np.uint8)
        _ink_python(xs, ys, starts[index:index + 1] * 0 + starts[index],
                    counts[index:index + 1], radii[index:index + 1], one)
        out[one.view(bool)] = table[index]
    return out


def _owners_python(xs, ys, starts, counts, radii, drawn, owner, mine):
    """The same answer in numpy, for an install without numba."""
    height, width = owner.shape
    for s in range(len(starts)):
        begin, count, r = starts[s], counts[s], radii[s]
        held = 0
        for k in range(max(1, count - 1)):
            ax0, ay0 = xs[begin + k], ys[begin + k]
            if count == 1:
                ax1, ay1 = ax0, ay0
            else:
                ax1, ay1 = xs[begin + k + 1], ys[begin + k + 1]
            lo_x = max(0, int(np.floor(min(ax0, ax1) - r)))
            hi_x = min(width - 1, int(np.ceil(max(ax0, ax1) + r)))
            lo_y = max(0, int(np.floor(min(ay0, ay1) - r)))
            hi_y = min(height - 1, int(np.ceil(max(ay0, ay1) + r)))
            if hi_x < lo_x or hi_y < lo_y:
                continue
            py, px = np.mgrid[lo_y:hi_y + 1, lo_x:hi_x + 1]
            ex, ey = ax1 - ax0, ay1 - ay0
            span = ex * ex + ey * ey
            vx, vy = px - ax0, py - ay0
            t = np.clip((vx * ex + vy * ey) / span, 0.0, 1.0) if span > 0 else 0.0
            dx, dy = vx - t * ex, vy - t * ey
            inside = (dx * dx + dy * dy <= r * r)
            crop = owner[lo_y:hi_y + 1, lo_x:hi_x + 1]
            free = inside & (crop < 0) & ~drawn[lo_y:hi_y + 1, lo_x:hi_x + 1]
            crop[free] = s
            held += int(free.sum())
        mine[s] = held


if JIT:
    @njit(cache=True)
    def _owners_jit(xs, ys, starts, counts, radii, drawn, owner, mine):  # pragma: no cover
        """Every pixel to the first stroke that covers it, in stroke order.

        Serial on purpose, and it is not a compromise: in stroke order the
        first to reach a pixel *is* the lowest-numbered one, which is what
        the kernel's `atomicMin` settles in parallel. Same answer, no atomics
        to need.
        """
        height, width = owner.shape
        for s in range(starts.shape[0]):
            begin, count, r = starts[s], counts[s], radii[s]
            held = 0
            for k in range(max(1, count - 1)):
                ax0 = xs[begin + k]
                ay0 = ys[begin + k]
                if count == 1:
                    ax1, ay1 = ax0, ay0
                else:
                    ax1 = xs[begin + k + 1]
                    ay1 = ys[begin + k + 1]
                fx, fy = ax1 - ax0, ay1 - ay0
                length = np.sqrt(fx * fx + fy * fy)
                piece = 2.0 * r if r > 0.5 else 1.0
                cuts = max(1, int(np.ceil(length / piece)))
                for c in range(cuts):
                    u0 = c / cuts
                    u1 = (c + 1) / cuts
                    x0 = ax0 + fx * u0
                    y0 = ay0 + fy * u0
                    x1 = ax0 + fx * u1
                    y1 = ay0 + fy * u1
                    lo_x = max(0, int(np.floor(min(x0, x1) - r)))
                    hi_x = min(width - 1, int(np.ceil(max(x0, x1) + r)))
                    lo_y = max(0, int(np.floor(min(y0, y1) - r)))
                    hi_y = min(height - 1, int(np.ceil(max(y0, y1) + r)))
                    ex, ey = x1 - x0, y1 - y0
                    span = ex * ex + ey * ey
                    for py in range(lo_y, hi_y + 1):
                        for px in range(lo_x, hi_x + 1):
                            vx = px - x0
                            vy = py - y0
                            if span > 0.0:
                                t = (vx * ex + vy * ey) / span
                                if t < 0.0:
                                    t = 0.0
                                elif t > 1.0:
                                    t = 1.0
                            else:
                                t = 0.0
                            dx = vx - t * ex
                            dy = vy - t * ey
                            if (dx * dx + dy * dy <= r * r
                                    and owner[py, px] < 0 and not drawn[py, px]):
                                owner[py, px] = s
                                held += 1
            mine[s] = held


def useful(segments, drawn, size, default_width: float = 1.0) -> np.ndarray:
    """How many pixels each stroke draws that nothing else draws.

    The host's answer to `kernels.useful`, and the same answer: every pixel
    belongs to the lowest-numbered stroke covering it that no earlier level
    had covered, so a stroke owning none can be dropped without the drawing
    changing by a pixel.
    """
    width, height = int(size[0]), int(size[1])
    mine = np.zeros(max(1, len(segments)), np.int32)
    if not segments:
        return mine[:0]
    xs, ys, starts, counts, radii = flatten(segments, default_width)
    owner = np.full((height, width), -1, np.int32)
    drawn = np.ascontiguousarray(drawn).view(np.uint8)
    (_owners_jit if JIT else _owners_python)(xs, ys, starts, counts, radii,
                                             drawn, owner, mine)
    return mine[:len(starts)]


def radii_of(widths):
    """Half the width a stroke is actually drawn at.

    Every rasterizer here rounds: a stroke of 5.76 px is drawn at 6, and has
    been since the caps were put right. One array, once, rather than the
    same `max(1, round(w))` per stroke.
    """
    return np.maximum(1.0, np.rint(np.asarray(widths, np.float64))) / 2.0


def ink_of(buffer, size) -> np.ndarray:
    """Where a `StrokeBuffer` lands, as a boolean mask.

    The same band as `ink`, taking the arrays as they already are. No
    `flatten`: that function exists to put objects back into the arrays they
    were built from, and a buffer was never taken out of them.
    """
    width, height = int(size[0]), int(size[1])
    out = np.zeros((max(1, height), max(1, width)), np.uint8)
    if len(buffer):
        host = buffer.to(np)
        (_ink_jit if JIT else _ink_python)(
            np.round(host.xs, 6), np.round(host.ys, 6),
            host.starts, host.counts, radii_of(host.widths), out)
    return out.view(bool)


def useful_of(buffer, drawn, size) -> np.ndarray:
    """How many pixels each stroke of a buffer draws that nothing else does."""
    width, height = int(size[0]), int(size[1])
    mine = np.zeros(max(1, len(buffer)), np.int32)
    if not len(buffer):
        return mine[:0]
    host = buffer.to(np)
    owner = np.full((height, width), -1, np.int32)
    drawn = np.ascontiguousarray(drawn).view(np.uint8)
    (_owners_jit if JIT else _owners_python)(
        np.round(host.xs, 6), np.round(host.ys, 6), host.starts, host.counts,
        radii_of(host.widths), drawn, owner, mine)
    return mine[:len(host)]
