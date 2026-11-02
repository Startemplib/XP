"""The merge window's painter, on the card.

Three things live on the device and stay there:

  * a page's flattened strokes -- points, starts, counts, widths and colours.
    Uploaded once per page, not once per frame: handing them up again for
    every mouse event is what made a small drawing slower on the card than on
    the processor, and it is the reason this is a class rather than a
    function.
  * the window being drawn: the owner image and the RGB buffer it is tinted
    into, both allocated once and refilled in place.
  * a picture side's page, once it has been rendered, so the PDF layer is
    composited on the card as well.

`raster.paint_flat` is not replaced. It is the definition of what a drawing
looks like, it runs when there is no card, and `tests/test_gpu_render.py`
holds the two to being bit-identical -- not close, identical -- because a
renderer that is nearly the same is a renderer that draws a different page.
"""
from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np

from xp.gpu import backend as gpu
from xp.gpu import kernels


def _one_thread_at_a_time(method):
    """Run a painter method under its own lock. See `Painter.__init__` for why there is one."""

    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    wrapped.__name__ = getattr(method, '__name__', 'wrapped')
    wrapped.__doc__ = method.__doc__
    return wrapped


class Painter:
    """One page's drawing, held on the card and painted into windows."""

    def __init__(self):
        self._key = None
        self._arrays = None
        self._segments = None
        self._owner = None
        self._painted = None
        self._paper = None          # the paper, on the device
        self._paper_host = None     # ...and what it was, to notice a change
        # Counted rather than guessed at: a caller can ask how much crossed
        # the bus, which is the number that says whether residency is working.
        self.uploads = 0
        self.paints = 0
        self.bytes_up = 0
        # **One thread at a time on this painter.** There is exactly one of these -- `views._CARD`,
        # made once -- and two threads reach it: the window paints on the UI thread, and the sharpen
        # worker renders the same window on its own thread (`window._render_for_sharpening` -> 
        # `views.render`). `hold()` reassigns `self._arrays` and `self._segments` while the other
        # thread may be inside `paint()` with the *old* arrays named only by the local variables of
        # that call; the old objects are then freed while the kernels launched from them are still
        # in flight, and the card reads freed memory. Measured as
        # `CUDARuntimeError: cudaErrorIllegalAddress` on one full run of
        # `tests/test_brush_is_local.py` in three, after which every later cupy call in that process
        # fails -- so once it happened the test was measuring a dead context, not a slow brush.
        # Held only around device work, never while waiting for anything else.
        self._lock = threading.RLock()

    # ---- is there a card to use ------------------------------------------

    @staticmethod
    def available() -> bool:
        """Whether the card can be used at all.

        Everything here falls back to `raster.paint_flat` when this is false,
        so the CPU is the answer whenever the card is missing, busy or
        refusing -- not an error the window has to show the user.
        """
        try:
            return bool(kernels.ready())
        except Exception:
            return False

    # ---- what stays resident ---------------------------------------------

    @_one_thread_at_a_time
    def hold(self, key, xs, ys, starts, counts, widths, colours) -> bool:
        """A page's arrays, uploaded once and kept while it is the same page.

        `key` says which page this is and which decisions built it. While it
        is unchanged nothing is sent, which is the whole point.
        """
        if key == self._key and self._arrays is not None:
            return False
        cp = gpu.xp
        arrays = (kernels._as(xs, cp.float64), kernels._as(ys, cp.float64),
                  kernels._as(starts, cp.int64), kernels._as(counts, cp.int64),
                  kernels._as(widths, cp.float64), kernels._as(colours, cp.uint8))
        self._arrays = arrays
        # The segment table is zoom-independent, so it is built once here and
        # never per frame -- including the two device-to-host sums that asking
        # how many segments there are would otherwise cost every frame.
        self._segments = kernels.segments_of(arrays[2], arrays[3])
        self._key = key
        self.uploads += 1
        self.bytes_up += int(sum(a.nbytes for a in arrays))
        return True

    @_one_thread_at_a_time
    def holds(self, key) -> bool:
        return self._key == key and self._arrays is not None

    @_one_thread_at_a_time
    def release(self) -> None:
        """Let the page go. A different document, or the window closing."""
        self._key = None
        self._arrays = None
        self._segments = None
        self._owner = None
        self._painted = None
        self._paper = None

    @property
    def resident_bytes(self) -> int:
        if self._arrays is None:
            return 0
        return int(sum(a.nbytes for a in self._arrays))

    # ---- drawing ---------------------------------------------------------

    @_one_thread_at_a_time
    def paint(self, paper, scale: float, offset: Tuple[float, float],
              wanted, colours=None, device_wanted: bool = False) -> np.ndarray:
        """The window, painted on the card, back as a host uint8 image.

        `paper` is the window as it should look before any stroke: the page's
        paper, or the paper with a picture side already on it. It is uploaded
        only when it changes, so panning across one background sends nothing.

        `wanted` is a boolean over the page's strokes -- which of them are
        being drawn this time. `colours` overrides the resident table, which
        is how the editing palette and the original colours are the same
        painter with different tables rather than two painters.
        """
        cp = gpu.xp
        if self._arrays is None:
            raise RuntimeError('nothing is resident; call hold() first')
        height, width = int(paper.shape[0]), int(paper.shape[1])
        if self._owner is None or self._owner.shape != (height, width):
            self._owner = cp.full((height, width), -1, cp.int32)
            self._painted = None
        if self._painted is None or self._painted.shape != paper.shape:
            self._painted = cp.empty(tuple(paper.shape), cp.uint8)
        # The paper goes up every frame, and the window is reset from the
        # device copy of it. Two earlier versions of this were wrong in
        # opposite directions: skipping the reset when the paper looked the
        # same made `_painted` an accumulator (a second paint came back as the
        # first drawing with the second one on top), and comparing 1.3 MB on
        # the host to decide cost more than sending it. A window is 1-2 MB and
        # the bus moves that in well under a millisecond, while the **strokes**
        # -- 8.5 MB for the user's page -- are what must not be sent again, and are
        # not.
        self._paper = cp.asarray(paper)
        self.bytes_up += int(paper.nbytes)
        self._painted[:] = self._paper
        xs, ys, starts, counts, widths, held = self._arrays
        table = held if colours is None else kernels._as(colours, cp.uint8)
        asked = wanted if device_wanted else kernels._as(wanted, cp.uint8)
        kernels.paint_segments(self._owner, xs, ys, starts, counts, widths, asked,
                               float(scale), offset, (height, width),
                               segments=self._segments)
        kernels.tint(self._owner, table, self._painted)
        self.paints += 1
        return cp.asnumpy(self._painted)

def wash(image, mask, colour, alpha: float) -> np.ndarray:
    """The brush's paint, mixed on the card.

    Exactly what `Image.composite` was doing in the window, integer for
    integer, and the formula was read off PIL rather than guessed at: the
    alpha is `int(v * alpha)` -- truncated, so a full pixel at 0.30 is 76, not
    76.5 -- and the blend is `(over * a + base * (255 - a) + 127) // 255`.

    It matters because the brush is the one thing in the window that moves at
    pointer speed: a stroke is drawn once when a gesture ends, and this is
    drawn on every motion event in between. `tests/test_gpu_render.py` holds it
    to the processor's own arithmetic.
    """
    cp = gpu.xp
    base = cp.asarray(image).astype(cp.int32)
    level = cp.floor(cp.asarray(mask, cp.float32) * np.float32(alpha)
                     ).astype(cp.int32)[:, :, None]
    over = cp.asarray(np.asarray(colour, np.int32))
    mixed = (over * level + base * (255 - level) + 127) // 255
    return cp.asnumpy(mixed.astype(cp.uint8))


def wash_on_the_processor(image, mask, colour, alpha: float) -> np.ndarray:
    """The same blend with PIL, for when there is no card.

    The caller uses this one directly; it is here so the two live next to each
    other and the test can compare them instead of comparing one of them with
    a memory of the other.
    """
    from PIL import Image

    base = image if isinstance(image, Image.Image) else Image.fromarray(image)
    overlay = Image.new('RGB', base.size, tuple(int(v) for v in colour))
    level = Image.fromarray(np.asarray(mask, np.uint8)).point(
        lambda v: int(v * alpha))
    return np.asarray(Image.composite(overlay, base, level), np.uint8)

