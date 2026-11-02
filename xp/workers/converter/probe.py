"""Statistical probe of a rendered page, used to pick conversion parameters
automatically instead of assuming them.

The original pipeline assumed a pure-white background and hand-tuned
thresholds. That breaks on real scans: a page of handwriting on green paper
had its entire background read as green ink, because "not white" was the
test for "is ink". So before converting, measure the page: what the paper
color actually is, how dark the ink is against it, and how thick the
strokes are -- then derive the thresholds from those measurements.

The empirical mappings here (stroke thickness -> adaptive block size, page
area -> min contour area) are deliberately simple and are meant to give a
good starting point for `tune`'s stage-by-stage adjustment, not to replace it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Tuple

import cv2
import numpy as np

from xp.core import geometry
from xp.gpu import backend as gpu


@dataclass
class PageProbe:
    background_bgr: Tuple[int, int, int]
    background_is_white: bool
    ink_fraction: float          # share of pixels that are ink, not paper
    mean_ink_contrast: float     # mean |ink - paper| brightness gap, 0-255
    stroke_thickness_px: float   # estimated pen width at the probed DPI
    ink_colors_bgr: List[Tuple[int, int, int]]  # dominant ink colors, most common first
    suggested_block_size: int
    suggested_c: float
    suggested_min_area: float

    def summary(self) -> str:
        b, g, r = self.background_bgr
        lines = [
            f"paper color      : BGR({b},{g},{r}){'  (white)' if self.background_is_white else '  (tinted)'}",
            f"ink coverage      : {self.ink_fraction * 100:.1f}% of the page",
            f"ink/paper contrast: {self.mean_ink_contrast:.0f} / 255",
            f"stroke thickness  : {self.stroke_thickness_px:.1f}px",
            f"ink colors (BGR) : {self.ink_colors_bgr}",
            f"suggested         : block_size={self.suggested_block_size} "
            f"C={self.suggested_c:.1f} min_area={self.suggested_min_area:.1f}",
        ]
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return asdict(self)


# How far (summed over B, G and R) a pixel must sit from the paper colour to
# count as ink rather than background. The one number that decides what ink
# is, and it lives here because this module's whole job is asking the page
# about itself -- `convert` and `inkdiff` import it. It had three copies
# before, one of them two bare literals in this file.
INK_DELTA = 60


def page_ink(img_bgr, background_bgr=None):
    """The ink in the page itself: far enough from the paper colour.

    This is the only honest answer to "is there ink here?", and it is the one
    `inkdiff` judges the finished tracing by. It is deliberately *not* a
    threshold: no window, no local contrast, nothing that can be set wrong.

    <~ 为什么要有这个函数:自适应阈值是拿一个像素跟它自己窗口里的均值比,
       窗口比笔画窄的时候,笔画内部的均值就等于它自己 —— 内部被判成纸。
       实测 800 DPI、17.3px 的笔、窗口 7:真墨只留下 52%,离纸超过 4px 的
       墨只留下 8.2%,送进追踪器的 mask 本来就是两条轮廓线。
       所以「这里是不是墨」最后要由原图回答,阈值只负责提个初稿。 ~>

    Per channel, accumulating in uint16. The obvious spelling --
    `np.abs(img.astype(np.int16) - paper).sum(axis=2)` -- builds a signed copy
    of the whole page and then another for the absolute value: at 800 DPI that
    is 2.4 GB of temporary to produce one boolean plane, and probe.py was
    paying it on every page.
    """
    if gpu.GPU_AVAILABLE:
        try:
            paper = np.asarray(background_bgr if background_bgr is not None
                               else (255, 255, 255), np.int16)
            on = gpu.xp.asarray(img_bgr).astype(gpu.xp.int16)
            far = gpu.xp.abs(on - gpu.xp.asarray(paper)).sum(axis=2)
            return gpu.to_host(far > INK_DELTA)
        except Exception:
            pass

    paper = np.array(background_bgr if background_bgr is not None else (255, 255, 255),
                     dtype=np.uint8)
    total = np.zeros(img_bgr.shape[:2], np.uint16)
    for channel, value in zip(cv2.split(img_bgr), paper):
        total += cv2.absdiff(channel, np.full_like(channel, int(value)))
    return total > INK_DELTA


def _dominant_color(img_bgr: np.ndarray, bins: int = 16) -> Tuple[int, int, int]:
    """Most common color, found on a coarsely quantized histogram so that
    near-identical shades (scan noise, JPEG ringing) count as one color."""
    step = 256 // bins
    quant = (img_bgr // step).astype(np.int64)
    keys = quant[..., 0] * bins * bins + quant[..., 1] * bins + quant[..., 2]
    counts = np.bincount(keys.reshape(-1), minlength=bins ** 3)
    top = int(counts.argmax())
    b, g, r = top // (bins * bins), (top // bins) % bins, top % bins
    # Return the mean of the pixels in that bin, not the bin corner, so the
    # value is the real paper color rather than a quantized approximation.
    in_bin = keys == top
    mean = img_bgr[in_bin].reshape(-1, 3).mean(axis=0)
    return tuple(int(round(v)) for v in mean)


def _strokes_only(ink_mask: np.ndarray, max_area_fraction: float = 0.02) -> np.ndarray:
    """Drop regions that clearly aren't pen strokes before measuring them:
    anything covering more than `max_area_fraction` of the page, and anything
    touching the image border. That removes the scan's white margin around a
    tinted page (which reads as "not the paper color", hence as ink) and
    solid filled blocks on slides, both of which otherwise dominate the
    thickness estimate -- a page of handwriting measured 781px "strokes"
    purely because of its margin.
    """
    h, w = ink_mask.shape
    max_area = max_area_fraction * h * w
    inside = ink_mask.astype(bool, copy=False)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inside.view(np.uint8), 8)
    left, top = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    wide, tall = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    touches_border = ((left == 0) | (top == 0)
                      | (left + wide >= w) | (top + tall >= h))
    keep = (area <= max_area) & ~touches_border
    keep[0] = False
    if not keep.any():
        return ink_mask
    # Asked of the ink, not of the page. `keep[labels]` reads the whole label
    # image -- 691 MB at 800 DPI -- to answer a question about the 2% of it
    # that is ink: 0.501s against 0.147s for the same array. This runs on the
    # **full-size** page (the pen width is measured there on purpose, see
    # `probe_page`), so it is the one place in the probe that costs anything.
    return geometry.pick(keep, labels, inside)


def _estimate_stroke_thickness(stroke_mask: np.ndarray) -> float:
    """The typical pen width: twice the ink area over its outline length.

    A stroke of length L and width w covers L*w and its outline runs 2L, so
    2*area/perimeter is w. The previous estimate -- twice the 80th percentile
    of the distance transform -- came out about 2.4x too wide on real pages
    (9.9px where the pen measured 4.0), because the percentile is dragged up
    by junctions and filled shapes, where the distance to the edge is large.
    Everything scales off this number (the threshold windows auto-tune tries,
    the smallest mark that must survive, how far the tracer thins), so being
    wrong by that much mis-set all of them.
    """
    if not stroke_mask.any():
        return 1.0
    contours, _ = cv2.findContours(stroke_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    return max(1.0, 2.0 * float(stroke_mask.sum()) / perimeter) if perimeter else 1.0


PROBE_MAX_DIM = 1200


def probe_page(img_bgr: np.ndarray, max_ink_colors: int = 6) -> PageProbe:
    """Measure one rendered page. `img_bgr` is the full-color render.

    Measuring runs on a downscaled copy: connected components and a distance
    transform over a full 800-DPI page (millions of pixels) cost far more
    than the statistics are worth, and none of what's measured here needs
    that resolution. Pixel-denominated results are scaled back up to the
    original resolution before being returned.
    """
    full_bgr = img_bgr
    full_h, full_w = img_bgr.shape[:2]
    probe_scale = 1.0
    longest = max(full_h, full_w)
    if longest > PROBE_MAX_DIM:
        probe_scale = PROBE_MAX_DIM / longest
        img_bgr = cv2.resize(
            img_bgr, (max(1, int(full_w * probe_scale)), max(1, int(full_h * probe_scale))),
            interpolation=cv2.INTER_AREA,
        )

    h, w = img_bgr.shape[:2]

    background = _dominant_color(img_bgr)
    bg = np.array(background, dtype=np.int16)
    background_is_white = bool(np.all(bg >= 240))

    # Ink = far enough from the paper color. Distance in BGR space is good
    # enough here and avoids assuming ink is merely "darker" (it usually is,
    # but light ink on dark paper should still register).
    ink_mask = page_ink(img_bgr, tuple(int(v) for v in bg))
    # Measure color/thickness on real strokes only, but report coverage over
    # the unfiltered mask so `ink_fraction` still describes the whole page.
    stroke_mask = _strokes_only(ink_mask)
    ink_fraction = float(ink_mask.mean())

    if stroke_mask.any():
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        bg_gray = float(cv2.cvtColor(bg.astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_BGR2GRAY)[0, 0])
        mean_ink_contrast = float(np.abs(gray[stroke_mask] - bg_gray).mean())
        ink_pixels = img_bgr[stroke_mask].reshape(-1, 3)
    else:
        mean_ink_contrast = 0.0
        ink_pixels = np.zeros((0, 3), dtype=np.uint8)

    ink_colors: List[Tuple[int, int, int]] = []
    if len(ink_pixels):
        step = 32
        quant = (ink_pixels // step).astype(np.int64)
        keys = quant[:, 0] * 64 + quant[:, 1] * 8 + quant[:, 2]
        uniq, counts = np.unique(keys, return_counts=True)
        for k in uniq[np.argsort(-counts)][:max_ink_colors]:
            mean = ink_pixels[keys == k].mean(axis=0)
            ink_colors.append(tuple(int(round(v)) for v in mean))

    # Pen width is measured on the full-size page, not the downscaled copy:
    # shrinking a 4px stroke to 1px and scaling the answer back up reads about
    # twice too wide, because the shrunken mask is chunkier than the ink.
    if probe_scale < 1:
        thickness = _estimate_stroke_thickness(
            _strokes_only(page_ink(full_bgr, tuple(int(v) for v in bg))))
    else:
        thickness = _estimate_stroke_thickness(stroke_mask)

    # Adaptive threshold works best with a window a few times the stroke
    # width, so it always sees both ink and surrounding paper; keep it odd.
    block = int(max(3, round(thickness * 4)))
    if block % 2 == 0:
        block += 1
    block = min(block, 99)

    # Low contrast needs a smaller offset or the ink gets thresholded away.
    c = float(np.clip(mean_ink_contrast / 12.0, 2.0, 20.0))

    # Drop specks, but nothing as big as a full stop. A dot is about one pen
    # width across, so its contour encloses roughly 0.6-0.8 t^2 (a little less
    # than its pixel count, since the contour runs through the edge pixels'
    # centres). The first version cut at 0.5 t^2 and, measured on a real
    # exercise sheet at 800 dpi, deleted every full stop. t^2/8 keeps dots
    # with room to spare and still drops the sub-half-point fragments that
    # a too-small threshold window leaves inside thick strokes.
    min_area = float(max(1.0, (thickness ** 2) / 8.0))

    return PageProbe(
        background_bgr=background,
        background_is_white=background_is_white,
        ink_fraction=ink_fraction,
        mean_ink_contrast=mean_ink_contrast,
        stroke_thickness_px=thickness,
        ink_colors_bgr=ink_colors,
        suggested_block_size=block,
        suggested_c=c,
        suggested_min_area=min_area,
    )
