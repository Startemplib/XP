"""Conversion profiles: the per-document constants that used to live hardcoded
at the bottom of pdf2xopp_Hand.py / pdf2xopp_Elec.py (DPI, geometry, color
tables), now loadable from JSON instead of copy-pasted per course folder.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from xp.core.document.xopp_io import DEFAULT_PAGE_HEIGHT, DEFAULT_PAGE_WIDTH

# This module IS the profiles package: the code and the defaults it reads live
# in one directory, so a move cannot separate them. One did -- the module went
# to `workers/converter/` while `hand_default.json` stayed behind at
# `xp/profiles/`, and `Profile.named('hand')` began raising
# `FileNotFoundError: 'hand'` in eleven test files. A resource found by walking
# from `__file__` only survives a refactor if it travels with that file.
PROFILES_DIR = Path(__file__).parent

TRACE_MODES = ('ortho-fill', 'onion-fill', 'centerline', 'outline')

# Settings only one tracing mode can read. Anything not named here is read by
# every mode -- stage one (threshold, morphology, specks, the page's own ink),
# the colour pass and the page geometry are shared, because what counts as ink
# must not depend on how it is going to be drawn.
#
# A setting that cannot change the result in the current mode is N/A: the
# panel greys it out and auto-tune spends no evaluation on it. Searching an
# axis with no effect is not merely slow, it is misleading -- the search
# reports a "best" value for a number that did nothing.
MODE_ONLY = {
    'rib_spacing': ('ortho-fill',),
    'rib_width': ('ortho-fill',),
    'rib_rounds': ('ortho-fill',),
    'boundary_width': ('onion-fill',),
    'fill_spacing': ('onion-fill',),
    'fill_overlap': ('onion-fill',),
    'boundary_smooth': ('onion-fill',),
    'boundary_shift': ('onion-fill',),
    'simplify_epsilon': ('onion-fill',),
    'fill_target': ('onion-fill',),
    'remove_invented': ('onion-fill',),
    'extra_target': ('onion-fill',),
    'invented_point_share': ('onion-fill',),
}


def applies(trace: str, name: str) -> bool:
    """Does this setting have any effect in this tracing mode?"""
    modes = MODE_ONLY.get(name)
    return True if modes is None else trace in modes


@dataclass
class Profile:
    mode: str  # "hand" or "elec"
    dpi: int = 300
    # scale/dx/dy: leave as None (the default) to auto-fit each PDF page's
    # actual size onto (page_width, page_height) -- see convert.fit_geometry.
    # The original scripts hardcoded these per document (e.g. DPI/15, DPI/13)
    # as a trial-and-error stand-in for exactly this fit; auto-fit replaces
    # that per-document guesswork for the common case. Set them explicitly
    # only if you need to override the automatic fit.
    scale: Optional[float] = None
    dx: Optional[float] = None
    dy: Optional[float] = None
    min_area: float = 7.0
    line_width: float = 0.07
    max_coord_str_len: int = 30000
    adaptive_block_size: int = 7
    adaptive_c: int = 10
    # Pre/post-processing ported from purify.py, off (0) by default so
    # existing profiles/behavior are unchanged unless explicitly tuned via
    # `xp tune` (see cli.py) -- same 0-10 slider scale and formulas
    # purify.py used, denoise/sharpen applied to grayscale before
    # thresholding, open/close applied to the binary mask before contours.
    denoise: float = 0.0
    sharpen: float = 0.0
    morph_open: float = 0.0
    morph_close: float = 0.0
    # Cut a stroke where the ink color changes along it, 0-10 (0 = off, as
    # the original scripts were: one color per contour). Two pen strokes
    # that cross are one blob, so one contour, and the whole thing otherwise
    # takes the majority color. Higher values allow shorter pieces: at 10 a
    # piece may be half a pen width, near 0 it must be about three.
    color_break: float = 0.0
    # How the ink is turned into strokes:
    #   'centerline'     one stroke down the middle of each pen stroke, at the
    #                    width that ink actually had -- right for handwriting
    #   'onion-fill'     the region peeled inwards ring by ring until it is
    #                    solid: reproduces the ink as area, whatever its shape
    #   'outline'        the boundary only, as the original scripts did: two
    #                    lines per stroke, which at handwriting size merge
    #                    into blobs and fill in the letters
    #   'ortho-fill'     ribs across the region along its medial axis, then a
    #                    disc wherever ink is still bare (orthofill.py). The
    #                    medial axis transform makes the coverage
    #                    constructive rather than iterative, so there is no
    #                    fill loop: measured on the densest square of the user's
    #                    page it leaves 0.000% of the ink bare where the fill
    #                    loop leaves 0.171%, with a third fewer points, in
    #                    half the time.
    trace: str = 'ortho-fill'
    # Stage 15's own settings -- the user's design, and only what it needs.
    #
    #   1 compute where a stroke of this width may stand
    #   2 place ribs there, evenly spaced down the medial axis
    #   3 compute the gap that is left, from the geometry and not a render
    #   4 halve the width and repeat, until the pixel; loose bits get dots
    #   5 draw once, in the region's colour
    #
    # The level *is* the width: the first width is twice the region's mean
    # distance to the paper (so about half the region is left for the levels
    # below), and every level after it is half as wide. That is the whole
    # ladder -- no width to set, and no count of levels to set, because the
    # ladder stops at the narrowest stroke the renderer can express.
    #
    # `rib_spacing` is the one number left: the seed spacing as a fraction of
    # the level's own width. At 1.0 each rib covers one rib-width of the
    # axis, so a level's ribs lie side by side across it -- touching, with
    # nothing between them and nothing on top of itself. Below 1.0 they
    # overlap and no gap of the level's own size can survive; above it the
    # level starts leaving gaps that are its own width to fill.
    rib_spacing: float = 1.0
    # The last step, and the user's: "it can merge at the last step, just connect
    # the nearby one, I can set a para for it -- you know if colour as 1
    # point then connect." Two numbers, and they are not mode-only: joining
    # is about the strokes a list holds, so it applies to whatever drew them.
    #
    # `merge_gap` is how far apart two ends may be, in pen widths; nought
    # turns the step off. `merge_colour` is how far apart their colours may
    # be, as Delta-E in Lab -- the same unit the cluster seams were measured
    # in, so 0 joins only strokes of one identical colour and 2 joins ones a
    # person would not tell apart. The join itself is held to rule 4 like
    # every rib: the bridge is drawn, so it has to be legal where it runs.
    merge_gap: float = 1.0
    merge_colour: float = 2.0
    # <~ rib_width and rib_rounds stood here. The width is now a fact about
    #    the level and the number of levels is a fact about the renderer, so
    #    neither is a setting; and rib_width was a cap that never bound,
    #    which is a dial that moves nothing. Before them, disc_trim,
    #    disc_rounds, rib_curvature and rib_ridge: mine, not the user's, and the
    #    refinements I had bolted beside the user's rule. The user had said plainly not to
    #    keep adding structures, and each of them turned out to be something
    #    a level already does. ~>
    # The last bridge: how the clean region is turned into curves. These are
    # read by 'onion-fill' alone -- see MODE_SETTINGS. All of these are in pen
    # widths or in pixels of the tracing, so the same numbers fit any DPI;
    # geometry.py says what each one does. They are settings rather than
    # constants because each is a fidelity trade, and autotune.py searches
    # them against the page itself once the mask is settled.
    #
    # The outer boundary's own stroke width, in pen widths. This is the one
    # that decides whether the writing comes back the weight it was: the
    # curve is laid half of it inside the true edge, so the stroke covers
    # the outermost band of the ink and nothing outside it.
    boundary_width: float = 0.40
    # Distance between consecutive inward offset curves, in pen widths, and
    # how much wider than that spacing each is stroked -- above 1, so that
    # neighbouring bands overlap and leave no gap down the middle.
    fill_spacing: float = 0.85
    fill_overlap: float = 1.30
    # Sigma of the blur on the distance field before the curves are taken, in
    # pixels: takes the sampling grid out of the boundary.
    boundary_smooth: float = 0.60
    # How far the smoothing that follows may move any one point of a curve,
    # in pixels. A hard per-point bound, so a corner of the writing cannot be
    # rounded off however many passes run.
    boundary_shift: float = 0.35
    # Points sitting on a straight run are dropped within this many pixels.
    # Sub-pixel by design: it thins a dense marching-squares loop, it does
    # not tidy the writing.
    simplify_epsilon: float = 0.30
    # How much of the ink the tracing may still be missing before it stops
    # topping itself up, as a share of the mask. The fill is a loop: draw,
    # rasterize, see what is still paper, draw that too -- so this is the
    # only thing that decides how faithful it ends up, and it is a number
    # rather than a consequence of how the curves happened to land.
    fill_target: float = 0.003
    # The tracing is corrected against the page itself, not against the
    # threshold's opinion of it: a pixel is ink when it is far enough from the
    # paper colour (convert.page_ink), which has no window and no setting that
    # can be wrong. Off only to reproduce an older conversion exactly.
    correct_against_page: bool = True
    # An enclosed region the page calls ink is ink, whatever the threshold
    # made of it -- the inside of a thick stroke comes back this way, while
    # the counter of an `o` is paper in the page and stays a hole.
    recover_interiors: bool = True
    # The other direction: a stroke drawn where the page has no ink is
    # removed. Nothing is guessed -- the tracing has been drawn and the page
    # is there to compare it with.
    remove_invented: bool = True
    # How much ink may be drawn outside the page's own, as a share of it,
    # before another removal round runs.
    extra_target: float = 0.01
    # A stroke is kept when at least this share of its points sit on the
    # page's ink. Below it, the stroke is something the tracer invented.
    invented_point_share: float = 0.5
    # Colour is decided after the shape, once for the page, by clustering the
    # ink in five dimensions -- (L, a, b, x, y) -- and never by a palette: a
    # palette is a standard the user has not set, and rounding to one before it
    # exists would throw away the only measurement there is. See color5d.py.
    colour_5d: bool = True
    # How far apart two colours must be to be two pens -- and **0 means ask
    # the page**, which is the default.
    #
    # How many colours a page was written in is a fact about the page, not a
    # setting: color5d.auto_bandwidth computes the partition at every rung of
    # a ladder, asks each one how much ink it puts in the wrong colour, and
    # takes the simplest partition that stays within a per cent of the best.
    # On the sample page the error is 0.00% up to a bandwidth of 16, 0.16% from
    # 20 to 31 and 31.96% at 38 -- a cliff, and the answer is the rung before
    # it: five pens.
    #
    # Set it to a number to overrule that. The unit is a distance in
    # color5d's direction space (radius 100), so 20 is about 11 degrees of
    # separation -- not a Lab Delta-E.
    colour_bandwidth: float = 0.0
    # How far the clustering looks for a pixel's neighbours, in pen widths.
    # It makes the colour *estimate* robust -- an anti-aliased edge pixel is
    # a mixture, and on its own it votes for a colour nobody wrote -- and it
    # cuts nothing: the clusters are forgotten before a stroke is sliced.
    space_bandwidth: float = 2.5
    # <~ slice_min_run and slice_hysteresis stood here. They were the dials
    #    of a question that no longer gets asked: how finely to cut a
    #    finished stroke where the colour changes along it. Region-First cuts
    #    the ink by colour while it is still a region, so a stroke is born
    #    inside one colour and there is nothing to cut. Both settings are
    #    gone rather than defaulted off -- a dial that cannot move anything
    #    is worse than no dial, as color_break proved for a whole day. ~>
    # When true, probe.py's per-page measurements override adaptive_block_size
    # and adaptive_c. Paper-color detection happens regardless; this only
    # controls whether the threshold settings are auto-derived too, so a
    # profile you tuned by hand stays in charge of its own numbers.
    auto_threshold: bool = False
    # Q base (66 x 138 mm), the page this notebook library is written on and
    # what every conversion and every merge is normalized onto (pagesize.py).
    page_width: float = DEFAULT_PAGE_WIDTH
    page_height: float = DEFAULT_PAGE_HEIGHT
    # Put the page's blue frame onto the template's frame, rather than fitting
    # the page as a whole. The frame is what says where the writing goes; a
    # page box only says how much paper was around it (frame.py).
    use_frame: bool = True
    # "elec" mode: nearest-RGB(BGR) palette, {name: [b, g, r]}
    palette_bgr: Dict[str, List[int]] = field(default_factory=dict)
    # "hand" mode: ordered HSV bands. Each name maps to a LIST of boxes
    # ({"low":[h,s,v], "high":[h,s,v]}); a pixel matches a name if it falls
    # inside ANY of its boxes. Multiple boxes are how hue-wraparound colors
    # (red) and multi-condition colors (black = low-value OR low-saturation)
    # are expressed. Checked in dict insertion order, first match wins, so
    # black/low-saturation should be listed first.
    hsv_ranges: Dict[str, List[Dict[str, List[int]]]] = field(default_factory=dict)

    @classmethod
    def load(cls, path) -> "Profile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    @classmethod
    def mine_path(cls, name: str) -> Path:
        """Where **the user's** saved default for this palette lives.

        Beside the shipped profiles, so it travels with the project -- XP has
        to run from any drive on any machine, and a settings file parked in
        `%APPDATA%` would be left behind by the first move.
        """
        return PROFILES_DIR / f"{name}_mine.json"

    @classmethod
    def named(cls, name: str, shipped: bool = False) -> "Profile":
        """Load a profile by name ('hand', 'elec', ...), or a path to a JSON
        file if `name` isn't one of the built-ins.

        **The user's own saved default wins**, 2026-10-28: *"change this into save
        the parameters as the default, so sometimes I can cook it step by
        step."* The window opens on the settings the user last kept, so a session
        continues where the last one stopped instead of starting from the
        shipped guess every time.

        `shipped=True` asks for the built-in regardless, which is how the user gets
        back to the factory numbers after saving the user's own.
        """
        if not shipped:
            mine = cls.mine_path(name)
            if mine.exists():
                return cls.load(mine)
        builtin = PROFILES_DIR / f"{name}_default.json"
        if builtin.exists():
            return cls.load(builtin)
        return cls.load(name)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def _color_hex_to_bgr(hexcolor: str) -> tuple:
    h = hexcolor.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def map_pixel_elec(pixel_bgr, palette_bgr: Dict[str, List[int]]) -> str:
    """Nearest-neighbor match in BGR space; returns the palette color name."""
    p = np.array(pixel_bgr, dtype=np.int32)
    best_name, best_dist = None, 1e18
    for name, col in palette_bgr.items():
        d = int(np.sum((p - np.array(col, dtype=np.int32)) ** 2))
        if d < best_dist:
            best_name, best_dist = name, d
    return best_name


def map_pixel_hand(pixel_bgr, hsv_ranges: Dict[str, List[Dict[str, List[int]]]],
                    fallback: str) -> str:
    """HSV band classification: first name (in dict order) with a matching
    box wins. See Profile.hsv_ranges for the box format."""
    hsv = cv2.cvtColor(np.uint8([[pixel_bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    for name, boxes in hsv_ranges.items():
        for box in boxes:
            lo, hi = box["low"], box["high"]
            if lo[0] <= h <= hi[0] and lo[1] <= s <= hi[1] and lo[2] <= v <= hi[2]:
                return name
    return fallback


def classify_bgr(profile, colors) -> tuple:
    """Palette class of each BGR sample, for whole arrays at once.

    Same rule as map_pixel_hand / map_pixel_elec one pixel at a time (first
    matching HSV box in order, or nearest palette color), but a page has
    hundreds of thousands of samples and those go through cv2.cvtColor on a
    1x1 image each. Returns (class index per sample, class names).
    """
    colors = np.ascontiguousarray(colors, dtype=np.uint8).reshape(-1, 3)
    if profile.mode == "hand":
        names = list(profile.hsv_ranges)
        fallback = "xopp_black"
        if fallback not in names:
            names.append(fallback)
        out = np.full(len(colors), names.index(fallback), np.int32)
        if len(colors) == 0:
            return out, names
        hsv = cv2.cvtColor(colors.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int16)
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        taken = np.zeros(len(colors), bool)
        for index, name in enumerate(names):
            for box in profile.hsv_ranges.get(name, ()):
                lo, hi = box["low"], box["high"]
                match = (~taken & (h >= lo[0]) & (h <= hi[0]) & (s >= lo[1])
                         & (s <= hi[1]) & (v >= lo[2]) & (v <= hi[2]))
                out[match] = index
                taken |= match
        return out, names
    names = list(profile.palette_bgr)
    if len(colors) == 0:
        return np.zeros(0, np.int32), names
    palette = np.array([profile.palette_bgr[n] for n in names], np.int32)
    distance = ((colors.astype(np.int32)[:, None, :] - palette[None]) ** 2).sum(axis=2)
    return distance.argmin(axis=1).astype(np.int32), names


def calibrate_from_samples(samples_dir, sat_min=5, val_min=20, low_pct=5,
                            high_pct=95) -> Dict[str, List[int]]:
    """Port of ColorProb.py: derive an HSV [low, high] range per subfolder of
    `samples_dir` (one subfolder per ink color, containing cropped screenshots
    of that ink) using percentile statistics over non-background pixels."""
    ranges: Dict[str, List[int]] = {}
    for color_dir in sorted(Path(samples_dir).iterdir()):
        if not color_dir.is_dir():
            continue
        H, S, V = [], [], []
        for img_path in glob.glob(os.path.join(color_dir, "*.png")):
            img = cv2.imread(img_path)
            if img is None:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = hsv[:, :, 0].reshape(-1), hsv[:, :, 1].reshape(-1), hsv[:, :, 2].reshape(-1)
            mask = (s >= sat_min) & (v >= val_min)
            if np.any(mask):
                H.append(h[mask]); S.append(s[mask]); V.append(v[mask])
        if not H:
            continue
        H, S, V = np.concatenate(H), np.concatenate(S), np.concatenate(V)
        low = [int(np.percentile(a, low_pct)) for a in (H, S, V)]
        high = [int(np.percentile(a, high_pct)) for a in (H, S, V)]
        ranges[color_dir.name] = [{"low": low, "high": high}]
    return ranges
