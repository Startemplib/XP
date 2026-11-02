"""The two inner loops, written in CUDA C++ and compiled at import.

The user's instruction: image work is a naturally parallel job, so do not walk it
one pixel or one region at a time -- open threads, and if Python can embed
another language here, embed it. CuPy can: `RawModule` hands a CUDA C++
source to NVRTC at runtime, so there is no build chain to carry and no
extension to ship, only this file.

Two loops earn it, and they earn it for the same reason -- every element is
independent of every other:

- **the march.** One thread per seed. A rib grows along its own normal until
  the first sample that is not legal or has changed colour, and no rib can
  tell another anything. In numpy this is blocks of 64 steps over every live
  seed at once, which allocates six arrays the size of (seeds x 64) per
  block; 2.25s of the user's page.
- **the band.** One thread per stroke, writing "covered" into the page. Two
  threads landing on the same pixel write the same 1, so a union has no race
  to lose and needs no atomics.

What is *not* here is as deliberate. `findContours` walks a boundary in
order and the order is the point, so it stays on the CPU. So does the PDF
decode, and so does writing the file.

The honest caveat, measured: these kernels are memory-bandwidth bound, not
compute bound. A full-page pass over a uint8 mask is 346 MB of traffic and
the card does ~700 GB/s, so the arithmetic is free and the only number that
matters is how many bytes you touch. That is why residency comes before
cleverness: a page transferred host-to-device costs 0.058s, which is more
than any of these kernels.
"""

import numpy as np

from xp.gpu import backend as gpu
SOURCE = r'''
extern "C" {

/* How far each seed gets before it leaves the legal set, or leaves the
   colour it started in. `use_labels` is 0 when there is no colour map, and
   `labels`/`want` are then not read. */
__global__ void march(const unsigned char* legal,
                      const int* labels, const int* want, const int use_labels,
                      const double* x, const double* y,
                      const double* nx, const double* ny,
                      double* reach,
                      const int n, const int width, const int height,
                      const double step, const double limit)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double sx = x[i], sy = y[i], dx = nx[i], dy = ny[i];
    const int mine = use_labels ? want[i] : 0;
    double t = 0.0;
    while (t + step <= limit) {
        const double u = t + step;
        /* lrint rounds half to even, which is what numpy's rint does, so a
           sample lands on the same pixel either way. */
        const long px = lrint(sx + dx * u);
        const long py = lrint(sy + dy * u);
        if (px < 0 || px >= width || py < 0 || py >= height) break;
        const size_t k = (size_t)py * (size_t)width + (size_t)px;
        if (!legal[k]) break;
        if (use_labels && labels[k] != mine) break;
        t = u;
    }
    reach[i] = t;
}

/* The band of one stroke: every pixel within its own radius of the polyline.
   One thread per stroke; the only write is a 1, so the union is race-free. */
__global__ void band(const double* xs, const double* ys,
                     const long long* starts, const long long* counts,
                     const double* radii,
                     unsigned char* out,
                     const int n, const int width, const int height)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    const long long begin = starts[s];
    const long long count = counts[s];
    const double r = radii[s];
    const double rr = r * r;
    const long long legs = count > 1 ? count - 1 : 1;
    for (long long k = 0; k < legs; ++k) {
        const double ax0 = xs[begin + k];
        const double ay0 = ys[begin + k];
        const double ax1 = count > 1 ? xs[begin + k + 1] : ax0;
        const double ay1 = count > 1 ? ys[begin + k + 1] : ay0;
        /* A segment's axis-aligned box is L x L when it lies diagonally, and
           the band inside it is only about 2rL. So the segment is cut into
           pieces no longer than 2r and each piece gets its own box: the
           nearest point of the whole segment to any pixel lies in one of the
           pieces, so the union over pieces is exactly the same set. */
        const double fx = ax1 - ax0, fy = ay1 - ay0;
        const double len = sqrt(fx * fx + fy * fy);
        const double piece = r > 0.5 ? 2.0 * r : 1.0;
        int cuts = (int)ceil(len / piece);
        if (cuts < 1) cuts = 1;
        for (int c = 0; c < cuts; ++c) {
            const double u0 = (double)c / cuts, u1 = (double)(c + 1) / cuts;
            const double x0 = ax0 + fx * u0, y0 = ay0 + fy * u0;
            const double x1 = ax0 + fx * u1, y1 = ay0 + fy * u1;
            int px0 = (int)floor(fmin(x0, x1) - r); if (px0 < 0) px0 = 0;
            int px1 = (int)ceil(fmax(x0, x1) + r);  if (px1 > width - 1) px1 = width - 1;
            int py0 = (int)floor(fmin(y0, y1) - r); if (py0 < 0) py0 = 0;
            int py1 = (int)ceil(fmax(y0, y1) + r);  if (py1 > height - 1) py1 = height - 1;
            const double ex = x1 - x0, ey = y1 - y0;
            const double span = ex * ex + ey * ey;
            for (int py = py0; py <= py1; ++py) {
                for (int px = px0; px <= px1; ++px) {
                    const double vx = px - x0, vy = py - y0;
                    double t = 0.0;
                    if (span > 0.0) {
                        t = (vx * ex + vy * ey) / span;
                        if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
                    }
                    const double dx = vx - t * ex, dy = vy - t * ey;
                    if (dx * dx + dy * dy <= rr)
                        out[(size_t)py * (size_t)width + (size_t)px] = 1;
                }
            }
        }
    }
}

/* Painting, in parallel, without losing the drawing order.

   A later stroke covers an earlier one. A serial painter gets that for
   free by writing in order; a parallel one has no "later" -- two threads
   writing the same pixel finish in whatever order the scheduler picked,
   and a view that comes out differently on different runs is not a view.

   So no thread writes a colour. Each writes its own index with atomicMax,
   and the highest index wins every pixel -- which is exactly "the last one
   drawn", and it does not depend on when anyone ran. A second pass turns
   indices into colours, one thread per pixel, with no contention at all.

   `wanted` selects without gathering: a stroke outside the window costs
   one load and one branch, which is cheaper than building an array of the
   ones that are inside. `scale` and `ox, oy` are applied here so that one
   set of page-unit arrays serves every zoom. */
__global__ void own_pixels(const double* xs, const double* ys,
                           const long long* starts, const long long* counts,
                           const double* widths, const unsigned char* wanted,
                           int* owner,
                           const double scale, const double ox, const double oy,
                           const int n, const int width, const int height)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    if (!wanted[s]) return;
    const long long begin = starts[s];
    const long long count = counts[s];
    double r = widths[s] * scale * 0.5;
    if (r < 0.5) r = 0.5;
    const double rr = r * r;
    const long long legs = count > 1 ? count - 1 : 1;
    for (long long k = 0; k < legs; ++k) {
        const double ax0 = xs[begin + k] * scale - ox;
        const double ay0 = ys[begin + k] * scale - oy;
        const double ax1 = count > 1 ? xs[begin + k + 1] * scale - ox : ax0;
        const double ay1 = count > 1 ? ys[begin + k + 1] * scale - oy : ay0;
        const double fx = ax1 - ax0, fy = ay1 - ay0;
        const double len = sqrt(fx * fx + fy * fy);
        const double piece = r > 0.5 ? 2.0 * r : 1.0;
        int cuts = (int)ceil(len / piece);
        if (cuts < 1) cuts = 1;
        for (int c = 0; c < cuts; ++c) {
            const double u0 = (double)c / cuts, u1 = (double)(c + 1) / cuts;
            const double x0 = ax0 + fx * u0, y0 = ay0 + fy * u0;
            const double x1 = ax0 + fx * u1, y1 = ay0 + fy * u1;
            int px0 = (int)floor(fmin(x0, x1) - r); if (px0 < 0) px0 = 0;
            int px1 = (int)ceil(fmax(x0, x1) + r);  if (px1 > width - 1) px1 = width - 1;
            int py0 = (int)floor(fmin(y0, y1) - r); if (py0 < 0) py0 = 0;
            int py1 = (int)ceil(fmax(y0, y1) + r);  if (py1 > height - 1) py1 = height - 1;
            const double ex = x1 - x0, ey = y1 - y0;
            const double span = ex * ex + ey * ey;
            for (int py = py0; py <= py1; ++py) {
                for (int px = px0; px <= px1; ++px) {
                    const double vx = px - x0, vy = py - y0;
                    double t = 0.0;
                    if (span > 0.0) {
                        t = (vx * ex + vy * ey) / span;
                        if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
                    }
                    const double dx = vx - t * ex, dy = vy - t * ey;
                    if (dx * dx + dy * dy <= rr)
                        atomicMax(&owner[(size_t)py * (size_t)width + (size_t)px], s);
                }
            }
        }
    }
}

/* Indices to colours. One thread a pixel, nothing shared, no atomics. */
__global__ void tint_pixels(const int* owner, const unsigned char* table,
                            unsigned char* out, const int pixels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixels) return;
    const int who = owner[i];
    if (who < 0) return;
    out[(size_t)i * 3 + 0] = table[(size_t)who * 3 + 0];
    out[(size_t)i * 3 + 1] = table[(size_t)who * 3 + 1];
    out[(size_t)i * 3 + 2] = table[(size_t)who * 3 + 2];
}

/* The drawing, one thread per segment.

   `own_pixels` gives one thread a whole stroke, and that was measured and
   lost: a warp of 32 threads then holds strokes of wildly different lengths
   and box sizes, so it waits for the slowest in it -- 0.15x to 1.2x the
   serial painter, worse the further out the user zoomed. The shape of the
   parallelism was the problem, not the card.

   A thread here owns one **segment of one stroke**: the unit the serial
   painter works in (`raster._paint_flat` loops a segment's bounding box a row
   at a time), so the work is even, and the bounding box and the distance test
   below are that painter's own, worked out from page units and the zoom
   rather than from arrays built for the frame. The answer is not
   approximately the same: it is the same test, evaluated by a different
   thread, and `tests/test_gpu_render.py` holds it to that.

   Draw order survives the parallelism because the owner is the stroke index
   and it is kept with `atomicMax`: the highest index wins every pixel, which
   is "the last one drawn", and it does not depend on when any thread ran. */
__global__ void span_segments(const double* xs, const double* ys,
                              const double* widths,
                              const int* stroke, const long long* at,
                              const long long* next,
                              const unsigned char* wanted,
                              const double scale, const double ox,
                              const double oy, const int n,
                              const int width, const int height, int* owner)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    const int who = stroke[s];
    if (wanted && !wanted[who]) return;
    const double ax = xs[at[s]] * scale - ox;
    const double ay = ys[at[s]] * scale - oy;
    const double bx = xs[next[s]] * scale - ox;
    const double by = ys[next[s]] * scale - oy;
    double rad = widths[who] * scale * 0.5;
    if (rad < 0.5) rad = 0.5;
    const double ex = bx - ax, ey = by - ay;
    const double span = ex * ex + ey * ey;
    int lo_y = (int)floor(fmin(ay, by) - rad);
    int hi_y = (int)ceil(fmax(ay, by) + rad);
    if (lo_y < 0) lo_y = 0;
    if (hi_y > height - 1) hi_y = height - 1;
    int lo_x = (int)floor(fmin(ax, bx) - rad);
    int hi_x = (int)ceil(fmax(ax, bx) + rad);
    if (lo_x < 0) lo_x = 0;
    if (hi_x > width - 1) hi_x = width - 1;
    for (int py = lo_y; py <= hi_y; ++py) {
        const double vy = py - ay;
        int row = py * width;
        for (int px = lo_x; px <= hi_x; ++px) {
            const double vx = px - ax;
            double t = span > 0.0 ? (vx * ex + vy * ey) / span : 0.0;
            if (t < 0.0) t = 0.0;
            else if (t > 1.0) t = 1.0;
            const double dx = vx - t * ex, dy = vy - t * ey;
            if (dx * dx + dy * dy <= rad * rad) {
                atomicMax(&owner[row + px], who);
            }
        }
    }
}

/* Who owns each pixel of the drawing: the lowest-numbered stroke that covers
   it and that nothing had covered before. `atomicMin` makes that answer the
   same however the threads are scheduled, which `atomicExch` would not --
   and a tracing that comes out differently on different runs is not a
   tracing. `drawn` is what earlier levels already put down; a pixel there is
   nobody's to claim. */
__global__ void claim(const double* xs, const double* ys,                      const long long* starts, const long long* counts,
                      const double* radii,
                      const unsigned char* drawn, int* owner,
                      const int n, const int width, const int height)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    const long long begin = starts[s];
    const long long count = counts[s];
    const double r = radii[s];
    const double rr = r * r;
    const long long legs = count > 1 ? count - 1 : 1;
    for (long long k = 0; k < legs; ++k) {
        const double ax0 = xs[begin + k], ay0 = ys[begin + k];
        const double ax1 = count > 1 ? xs[begin + k + 1] : ax0;
        const double ay1 = count > 1 ? ys[begin + k + 1] : ay0;
        const double fx = ax1 - ax0, fy = ay1 - ay0;
        const double len = sqrt(fx * fx + fy * fy);
        const double piece = r > 0.5 ? 2.0 * r : 1.0;
        int cuts = (int)ceil(len / piece); if (cuts < 1) cuts = 1;
        for (int c = 0; c < cuts; ++c) {
            const double u0 = (double)c / cuts, u1 = (double)(c + 1) / cuts;
            const double x0 = ax0 + fx * u0, y0 = ay0 + fy * u0;
            const double x1 = ax0 + fx * u1, y1 = ay0 + fy * u1;
            int px0 = (int)floor(fmin(x0, x1) - r); if (px0 < 0) px0 = 0;
            int px1 = (int)ceil(fmax(x0, x1) + r);  if (px1 > width - 1) px1 = width - 1;
            int py0 = (int)floor(fmin(y0, y1) - r); if (py0 < 0) py0 = 0;
            int py1 = (int)ceil(fmax(y0, y1) + r);  if (py1 > height - 1) py1 = height - 1;
            const double ex = x1 - x0, ey = y1 - y0;
            const double span = ex * ex + ey * ey;
            for (int py = py0; py <= py1; ++py) {
                for (int px = px0; px <= px1; ++px) {
                    const double vx = px - x0, vy = py - y0;
                    double t = 0.0;
                    if (span > 0.0) {
                        t = (vx * ex + vy * ey) / span;
                        if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
                    }
                    const double dx = vx - t * ex, dy = vy - t * ey;
                    if (dx * dx + dy * dy <= rr) {
                        const size_t j = (size_t)py * (size_t)width + (size_t)px;
                        if (!drawn[j]) atomicMin(&owner[j], s);
                    }
                }
            }
        }
    }
}

/* How many pixels each stroke ended up owning. Zero means every pixel it
   covers was already covered by an earlier stroke or an earlier level, so
   dropping it changes the drawing by nothing at all. */
__global__ void tally(const double* xs, const double* ys,
                      const long long* starts, const long long* counts,
                      const double* radii,
                      const int* owner, int* mine,
                      const int n, const int width, const int height)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    const long long begin = starts[s];
    const long long count = counts[s];
    const double r = radii[s];
    const double rr = r * r;
    const long long legs = count > 1 ? count - 1 : 1;
    int held = 0;
    for (long long k = 0; k < legs; ++k) {
        const double ax0 = xs[begin + k], ay0 = ys[begin + k];
        const double ax1 = count > 1 ? xs[begin + k + 1] : ax0;
        const double ay1 = count > 1 ? ys[begin + k + 1] : ay0;
        const double fx = ax1 - ax0, fy = ay1 - ay0;
        const double len = sqrt(fx * fx + fy * fy);
        const double piece = r > 0.5 ? 2.0 * r : 1.0;
        int cuts = (int)ceil(len / piece); if (cuts < 1) cuts = 1;
        for (int c = 0; c < cuts; ++c) {
            const double u0 = (double)c / cuts, u1 = (double)(c + 1) / cuts;
            const double x0 = ax0 + fx * u0, y0 = ay0 + fy * u0;
            const double x1 = ax0 + fx * u1, y1 = ay0 + fy * u1;
            int px0 = (int)floor(fmin(x0, x1) - r); if (px0 < 0) px0 = 0;
            int px1 = (int)ceil(fmax(x0, x1) + r);  if (px1 > width - 1) px1 = width - 1;
            int py0 = (int)floor(fmin(y0, y1) - r); if (py0 < 0) py0 = 0;
            int py1 = (int)ceil(fmax(y0, y1) + r);  if (py1 > height - 1) py1 = height - 1;
            const double ex = x1 - x0, ey = y1 - y0;
            const double span = ex * ex + ey * ey;
            for (int py = py0; py <= py1; ++py) {
                for (int px = px0; px <= px1; ++px) {
                    const double vx = px - x0, vy = py - y0;
                    double t = 0.0;
                    if (span > 0.0) {
                        t = (vx * ex + vy * ey) / span;
                        if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
                    }
                    const double dx = vx - t * ex, dy = vy - t * ey;
                    if (dx * dx + dy * dy <= rr
                        && owner[(size_t)py * (size_t)width + (size_t)px] == s)
                        ++held;
                }
            }
        }
    }
    mine[s] = held;
}

/* The agreed summation order (xp/reduce.py): one thread per block, and
   inside a block the elements in index order. The block sums are added up
   afterwards, also in index order, so nothing about how the threads were
   scheduled can reach the answer. */
__global__ void masked_sum(const float* values, const unsigned char* mask,
                           double* totals, long long* counts,
                           const long long n, const int block)
{
    long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long many = (n + block - 1) / block;
    if (j >= many) return;
    long long lo = j * (long long)block;
    long long hi = lo + block;
    if (hi > n) hi = n;
    double total = 0.0;
    long long held = 0;
    for (long long i = lo; i < hi; ++i) {
        if (mask[i]) { total += (double)values[i]; ++held; }
    }
    totals[j] = total;
    counts[j] = held;
}

/* Rule 4 for a bridge: is a stroke of this width allowed everywhere along
   the segment between two ends? One thread per pair -- a pair knows nothing
   about any other pair -- and it reads the distance field that is already on
   the card for the march. */
__global__ void bridges(const float* field,
                        const double* ax, const double* ay,
                        const double* bx, const double* by,
                        const double* width, const double slack,
                        const double step,
                        unsigned char* ok,
                        const int n, const int w, const int h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double x0 = ax[i], y0 = ay[i], x1 = bx[i], y1 = by[i];
    const double need = width[i] * 0.5 + slack;
    const double dx = x1 - x0, dy = y1 - y0;
    const double len = sqrt(dx * dx + dy * dy);
    int cuts = (int)ceil(len / step);
    if (cuts < 1) cuts = 1;
    for (int c = 0; c <= cuts; ++c) {
        const double t = (double)c / cuts;
        long px = lrint(x0 + dx * t);
        long py = lrint(y0 + dy * t);
        if (px < 0) px = 0; else if (px > w - 1) px = w - 1;
        if (py < 0) py = 0; else if (py > h - 1) py = h - 1;
        if (field[(size_t)py * (size_t)w + (size_t)px] < need) { ok[i] = 0; return; }
    }
    ok[i] = 1;
}

}
'''

THREADS = 256
_module = None
_entry = {}


def ready() -> bool:
    """Is there a card, and did the source compile?

    Compiled once, on the first ask rather than at import: a machine with no
    card should not pay for NVRTC to tell it so.
    """
    global _module
    if not gpu.GPU_AVAILABLE:
        return False
    if _module is None:
        try:
            _module = gpu.xp.RawModule(code=SOURCE, options=('-std=c++11',))
            _module.get_function('march')
            _module.get_function('band')
        except Exception:
            _module = False
    return _module is not False


def _at(name):
    """One handle per kernel, kept.

    `get_function` goes back to the module every time it is asked, and the
    fill asks nine times a page -- 0.15s of looking up something that cannot
    have changed.
    """
    held = _entry.get(name)
    if held is None:
        held = _entry[name] = _module.get_function(name)
    return held


def _grid(n):
    return ((n + THREADS - 1) // THREADS,), (THREADS,)


def _as(array, dtype):
    """On the device, in exactly this dtype, contiguous.

    A raw kernel has no type checking: it takes a pointer and believes the
    signature. Handing it a float32 array where it declares `const double*`
    does not fail -- it reads two floats as one double, which comes out as a
    denormal near zero, and every rib then marched on the spot (its own pixel
    is legal) until it hit the limit. `_seeds` returns its normals in float32
    and its positions in float64, which is exactly the mix that finds this.
    So nothing is trusted here; everything is cast at the door.
    """
    cp = gpu.xp
    if (isinstance(array, cp.ndarray) and array.dtype == dtype
            and array.flags['C_CONTIGUOUS']):
        # Already there, already right. `ascontiguousarray` would copy it
        # anyway, and `labels` is 691 MB -- copying it four times a level
        # cost more than the march it was being passed to.
        return array
    return cp.ascontiguousarray(cp.asarray(array), dtype=dtype)


def march(legal, labels, want, x, y, nx, ny, step, limit):
    """`orthofill._reach`, one thread per seed. Arrays may be on either side;
    what comes back is on the device."""
    if not ready():
        raise RuntimeError('no CUDA module; call kernels.ready() first')
    cp = gpu.xp
    n = int(np.asarray(x).shape[0]) if not hasattr(x, 'shape') else int(x.shape[0])
    reach = cp.zeros(n, cp.float64)
    if n == 0:
        return reach
    use = 1 if labels is not None else 0
    labels = _as(labels, cp.int32) if use else cp.zeros(1, cp.int32)
    want = _as(want, cp.int32) if use else cp.zeros(1, cp.int32)
    legal = _as(legal, cp.uint8)
    grid, block = _grid(n)
    _at('march')(
        grid, block,
        (legal, labels, want, np.int32(use),
         _as(x, cp.float64), _as(y, cp.float64),
         _as(nx, cp.float64), _as(ny, cp.float64), reach,
         np.int32(n), np.int32(legal.shape[1]), np.int32(legal.shape[0]),
         np.float64(step), np.float64(limit)))
    return reach


def useful(xs, ys, starts, counts, radii, drawn, shape):
    """Which of these strokes draw a pixel that nothing else draws.

    The user's words: avoid repeated and overlapping strokes. A level lays its ribs
    a width apart along the boundary, and wherever that boundary curves the
    ribs beside each other cover much of the same ink -- so some of them draw
    nothing that is not already drawn, and a stroke that draws nothing is
    still a stroke in the file, in the editor, and in every pass that comes
    after.

    Every pixel goes to the lowest-numbered stroke that covers it and that no
    earlier level had covered. A stroke that ends up owning none can be
    dropped and the drawing is *pixel for pixel* what it was, because every
    pixel it covered is owned by a stroke that stays.
    """
    if not ready():
        raise RuntimeError('no CUDA module; call kernels.ready() first')
    cp = gpu.xp
    n = int(starts.shape[0])
    if n == 0:
        return cp.zeros(0, cp.int32)
    xs, ys = _as(xs, cp.float64), _as(ys, cp.float64)
    starts, counts = _as(starts, cp.int64), _as(counts, cp.int64)
    radii, drawn = _as(radii, cp.float64), _as(drawn, cp.uint8)
    owner = cp.full(shape, 0x7FFFFFFF, cp.int32)
    mine = cp.zeros(n, cp.int32)
    grid, block = _grid(n)
    args = (xs, ys, starts, counts, radii)
    size = (np.int32(n), np.int32(shape[1]), np.int32(shape[0]))
    _at('claim')(grid, block, args + (drawn, owner) + size)
    _at('tally')(grid, block, args + (owner, mine) + size)
    del owner
    # Handed straight back: `owner` is one int32 per pixel, 691 MB on the user's
    # page, and holding it between levels is the shape of the trouble that
    # once turned a 9.5s page into a 141.8s one.
    cp.get_default_memory_pool().free_all_blocks()
    return mine


def bridges(field, ax, ay, bx, by, width, slack, step):
    """`strokejoin._bridge_is_legal`, one thread per pair.

    The field is the same array the march reads, so when the page is already
    on the card this costs a kernel and nothing else -- where in numpy it is
    a (pairs x samples) block of fancy indexing into a 173-million-pixel
    array, 2.66s of the user's page.
    """
    if not ready():
        raise RuntimeError('no CUDA module; call kernels.ready() first')
    cp = gpu.xp
    n = int(ax.shape[0]) if hasattr(ax, 'shape') else len(ax)
    ok = cp.zeros(max(1, n), cp.uint8)
    if n == 0:
        return ok[:0]
    field = _as(field, cp.float32)
    grid, block = _grid(n)
    _at('bridges')(
        grid, block,
        (field, _as(ax, cp.float64), _as(ay, cp.float64),
         _as(bx, cp.float64), _as(by, cp.float64), _as(width, cp.float64),
         np.float64(slack), np.float64(step), ok,
         np.int32(n), np.int32(field.shape[1]), np.int32(field.shape[0])))
    return ok


def masked_sum(values, mask, many, block):
    """The agreed reduction (`xp/reduce.py`), on the card.

    What comes back is the per-block sums and counts, not the answer: the
    blocks are added up on the host in index order, because *that* order is
    the specification and a second reduction here would be a second order.
    """
    if not ready():
        raise RuntimeError('no CUDA module; call kernels.ready() first')
    cp = gpu.xp
    values = _as(values, cp.float32)
    mask = _as(mask, cp.uint8)
    totals = cp.zeros(many, cp.float64)
    counts = cp.zeros(many, cp.int64)
    grid, threads = _grid(many)
    _at('masked_sum')(grid, threads,
                      (values, mask, totals, counts,
                       np.int64(values.shape[0]), np.int32(block)))
    return totals, counts


def band(xs, ys, starts, counts, radii, shape):
    """`raster.ink`, one thread per stroke, into a device mask."""
    if not ready():
        raise RuntimeError('no CUDA module; call kernels.ready() first')
    cp = gpu.xp
    out = cp.zeros(shape, cp.uint8)
    n = int(starts.shape[0])
    if n == 0:
        return out
    grid, block = _grid(n)
    _at('band')(
        grid, block,
        (_as(xs, cp.float64), _as(ys, cp.float64),
         _as(starts, cp.int64), _as(counts, cp.int64), _as(radii, cp.float64),
         out, np.int32(n), np.int32(shape[1]), np.int32(shape[0])))
    return out


def paint(out, xs, ys, starts, counts, widths, colours, wanted,
          scale, offset, shape):
    """`raster.paint_flat` on the card.

    Two passes and no serialisation: `own_pixels` keeps the highest stroke
    index per pixel with `atomicMax` -- which is the drawing order, and is
    the same answer however the threads were scheduled -- and `tint_pixels`
    turns indices into colours with one thread a pixel and nothing shared.

    `out` is a host (H, W, 3) uint8 image and comes back painted. The page's
    arrays are expected to be on the card already; handing them up per frame
    is what makes a small drawing slower on the GPU than on the CPU.
    """
    if not ready():
        raise RuntimeError('no CUDA module; call kernels.ready() first')
    cp = gpu.xp
    height, width = int(shape[0]), int(shape[1])
    n = int(starts.shape[0])
    owner = cp.full((height, width), -1, cp.int32)
    if n:
        grid, block = _grid(n)
        ox, oy = offset
        _at('own_pixels')(
            grid, block,
            (_as(xs, cp.float64), _as(ys, cp.float64),
             _as(starts, cp.int64), _as(counts, cp.int64),
             _as(widths, cp.float64), _as(wanted, cp.uint8), owner,
             np.float64(scale), np.float64(ox), np.float64(oy),
             np.int32(n), np.int32(width), np.int32(height)))
    painted = cp.asarray(out)
    pixels = height * width
    grid, block = _grid(pixels)
    _at('tint_pixels')(
        grid, block,
        (owner, _as(cp.asarray(colours), cp.uint8), painted, np.int32(pixels)))
    return cp.asnumpy(painted)


def segments_of(starts, counts):
    """Every segment of the drawing, in page units, worked out once per page.

    Which stroke each segment belongs to and the two point indices it runs
    between -- and none of it depends on the zoom. That is the point. The
    first version of this built the geometry per frame, which meant about
    twenty-five array operations and, worse, two device-to-host `.sum()` calls
    to ask how many segments and rows there were. Measured against the serial
    painter, that bookkeeping cost more than the whole drawing: 0.0119s against
    0.0082s at 1x, and it got worse the further out the user zoomed, which is the
    signature of a fixed cost rather than of rasterisation.
    """
    cp = gpu.xp
    n = int(starts.shape[0])
    if not n:
        return None
    per = cp.maximum(1, counts - 1).astype(cp.int64)
    total = int(per.sum())
    if not total:
        return None
    stroke = cp.repeat(cp.arange(n, dtype=cp.int32), per)
    within = cp.arange(total, dtype=cp.int64) - cp.repeat(cp.cumsum(per) - per, per)
    at = starts[stroke] + within
    last = starts[stroke] + counts[stroke] - 1
    nxt = cp.minimum(at + 1, last)
    return (stroke, at.astype(cp.int64), nxt.astype(cp.int64), total)


def paint_segments(owner, xs, ys, starts, counts, widths, wanted, scale,
                   offset, shape, segments=None):
    """`raster.paint_flat` on the card: one thread a segment, geometry in the kernel.

    `own_pixels` gave one thread a whole stroke and lost, and the measurement
    said why: a warp of 32 threads then holds strokes of wildly different
    lengths and box sizes, so it waits for the slowest one in it. A thread here
    owns one **segment**, which is the unit the serial painter works in, and
    computes its own bounding box and its own distance tests from the page's
    units and the zoom. The same bounding box, the same capsule, the same
    comparison -- evaluated by a different thread. `tests/test_gpu_render.py`
    holds the two to being bit-identical on the user's page at four zooms, and they
    are, on all three channels.

    A frame is two kernels and nothing else: who owns each pixel, then indices
    to colours. The segment table is made once per page, the stroke arrays live
    on the card, and `wanted` is the only thing that changes during a gesture.
    """
    if not ready():
        raise RuntimeError('no CUDA module; call kernels.ready() first')
    cp = gpu.xp
    height, width = int(shape[0]), int(shape[1])
    owner.fill(-1)
    if not int(starts.shape[0]):
        return owner
    if segments is None:
        segments = segments_of(starts, counts)
    if segments is None:
        return owner
    stroke, at, nxt, total = segments
    asked = _as(wanted, cp.uint8) if wanted is not None else None
    ox, oy = offset
    grid, block = _grid(total)
    _at('span_segments')(
        grid, block,
        (_as(xs, cp.float64), _as(ys, cp.float64),
         _as(widths, cp.float64), stroke, at, nxt,
         asked if asked is not None else _as(np.ones(1, np.uint8), cp.uint8),
         np.float64(scale), np.float64(ox), np.float64(oy),
         np.int32(total), np.int32(width), np.int32(height), owner))
    return owner


def tint(owner, colours, out):
    """`tint_pixels`: an owner image and a colour table into an RGB image.

    Both are on the card and stay there; `out` is the device buffer the
    caller is holding, so nothing crosses the bus to draw a frame.
    """
    cp = gpu.xp
    pixels = int(owner.size)
    grid, block = _grid(pixels)
    _at('tint_pixels')(grid, block,
                       (owner, _as(colours, cp.uint8), out, np.int32(pixels)))
    return out


def _arrays_of(buffer):
    """A buffer's arrays, on the card, in the dtypes the kernels declare.

    The rounding to six decimals is the one `xopp_io` writes with and the
    rasterizers have always applied; it belongs here, once over the page,
    rather than in whatever built the points.
    """
    cp = gpu.xp
    held = buffer.to(cp)
    return (cp.round(_as(held.xs, cp.float64), 6),
            cp.round(_as(held.ys, cp.float64), 6),
            _as(held.starts, cp.int64), _as(held.counts, cp.int64),
            cp.maximum(1.0, cp.rint(_as(held.widths, cp.float64))) / 2.0)


def band_of(buffer, shape):
    """`band`, taking a `StrokeBuffer` -- no flatten, no per-stroke work."""
    if not len(buffer):
        return gpu.xp.zeros(shape, gpu.xp.uint8)
    return band(*_arrays_of(buffer), shape)


def useful_of(buffer, drawn, shape):
    """`useful`, taking a `StrokeBuffer`."""
    if not len(buffer):
        return gpu.xp.zeros(0, gpu.xp.int32)
    return useful(*_arrays_of(buffer), drawn, shape)
