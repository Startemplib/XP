"""Interactive, page-by-page conversion with step-by-step parameter tuning --
a direct port of purify.py's structure.

Two levels, same as purify.py has:

  * the per-item loop: convert one page, preview it, then
    [Enter=accept | s=adjust | q=quit]. purify.py does this per image file;
    here it's per *page*, because a PDF or xopp can run to dozens of pages
    and previewing "the whole document" tells you nothing about any single
    one of them.

  * the stage machine, entered by `s`: denoise -> sharpen -> threshold ->
    morph -> area, one stage at a time, each with its own live preview and
    "Enter=keep / number=new / b=back".

Whatever you tune on a page carries forward as the starting point for the
next one, so a document with consistent scans only needs tuning once, but
every page is still previewed and confirmed on its own.

Needs the optional `arpy` package installed for vp(): this is a
human-in-the-loop tool -- vp() opens a window you look at and answer from
the terminal, exactly like purify.py.
"""

from __future__ import annotations

import copy
from typing import List, Optional

import cv2
import numpy as np
import pymupdf

from xp.workers.converter.convert import (
    _contours_to_strokes,
    apply_morph,
    convert_page_image,
    denoise_and_sharpen,
    page_geometry,
    render_pdf_page,
    usable_dpi,
)
from xp.workers.converter.probe import probe_page
from xp.workers.converter.profiles import Profile
from xp.render.drawing import render_page
from xp.core.document.xopp_io import XoppDocument

try:
    from arpy.visual import vp
except ImportError:
    vp = None


def _require_vp():
    if vp is None:
        raise RuntimeError(
            "arpy.visual.vp isn't importable. Auto-tune needs the optional "
            "`arpy` package installed; without it, tune by hand or skip this step."
        )


def _preview_strokes(strokes: List, profile: Profile) -> np.ndarray:
    """Render stroke elements the way convert would really produce them, as a
    BGR image ready for vp()."""
    doc = XoppDocument.new_empty(page_count=1, width=profile.page_width,
                                  height=profile.page_height)
    for s in strokes:
        doc.pages[0].layer.append(s)
    return np.array(render_page(doc.pages[0]))[:, :, ::-1]  # RGB -> BGR


def _tune_loop(name: str, value: float, apply_fn):
    while True:
        print(f"\n--- tuning {name} --- current={value}")
        v = input("Enter=keep / number=new / b=back: ").strip()

        if v == "":
            return value, "ok"
        if v.lower() == "b":
            return value, "back"

        try:
            v = float(v)
        except ValueError:
            print("Invalid")
            continue

        vp(apply_fn(v))
        value = v


def tune_stages(img_raw: np.ndarray, profile: Profile, scale: float, dx: float,
                 dy: float, background_bgr=None) -> Profile:
    """The stage machine, run against one already-rendered page. Mutates and
    returns a copy of `profile`."""
    _require_vp()
    profile = copy.deepcopy(profile)
    gray0 = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

    denoised = gray0
    sharp = gray0
    binary = None
    morphed = None

    step = 0
    while step < 5:

        # STEP 0: DENOISE
        if step == 0:
            val, state = _tune_loop(
                "denoise", profile.denoise,
                lambda v: denoise_and_sharpen(gray0, v, 0),
            )
            profile.denoise = val
            if state == "back":
                step = max(0, step - 1)
            else:
                denoised = denoise_and_sharpen(gray0, profile.denoise, 0)
                step += 1

        # STEP 1: SHARPEN
        elif step == 1:
            val, state = _tune_loop(
                "sharpen", profile.sharpen,
                lambda v: denoise_and_sharpen(denoised, 0, v),
            )
            profile.sharpen = val
            if state == "back":
                step -= 1
            else:
                sharp = denoise_and_sharpen(denoised, 0, profile.sharpen)
                step += 1

        # STEP 2: THRESHOLD (blockSize, C)
        elif step == 2:
            while True:
                print(f"\n--- threshold (blockSize C) --- "
                      f"current=[{profile.adaptive_block_size} {profile.adaptive_c}]")
                v = input("Enter=keep / 'blockSize C'=new / b=back: ").strip()

                if v == "":
                    binary = cv2.adaptiveThreshold(
                        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                        profile.adaptive_block_size, profile.adaptive_c,
                    )
                    step += 1
                    break
                if v.lower() == "b":
                    step -= 1
                    break

                try:
                    bs_s, c_s = v.split()
                    bs, c = int(bs_s), float(c_s)
                    if bs % 2 == 0:
                        bs += 1
                except ValueError:
                    print("Invalid, expected: blockSize C")
                    continue

                profile.adaptive_block_size, profile.adaptive_c = bs, c
                # Tuning by hand means the auto-derived threshold should stop
                # overriding it.
                profile.auto_threshold = False
                vp(cv2.adaptiveThreshold(
                    sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs, c
                ))

        # STEP 3: MORPH (open, close)
        elif step == 3:
            while True:
                print(f"\n--- morph (open close) --- "
                      f"current=[{profile.morph_open} {profile.morph_close}]")
                v = input("Enter=keep / 'open close'=new / b=back: ").strip()

                if v == "":
                    morphed = apply_morph(binary, profile.morph_open, profile.morph_close)
                    step += 1
                    break
                if v.lower() == "b":
                    step -= 1
                    break

                try:
                    o, c = map(float, v.split())
                except ValueError:
                    print("Invalid, expected: open close")
                    continue

                profile.morph_open, profile.morph_close = o, c
                vp(apply_morph(binary, o, c))

        # STEP 4: AREA (contour-level min_area), previewed as the real result
        elif step == 4:
            def apply_area(v):
                p = copy.deepcopy(profile)
                p.min_area = v
                strokes = _contours_to_strokes(morphed > 0, img_raw, p, scale, dx, dy, background_bgr)
                return _preview_strokes(strokes, p)

            val, state = _tune_loop("min_area", profile.min_area, apply_area)
            profile.min_area = val
            if state == "back":
                step -= 1
            else:
                step += 1

    return profile


def interactive_tune(pdf_path, page_index: int = 0,
                      base_profile: Optional[Profile] = None) -> Profile:
    """Tune on a single page and return the resulting profile."""
    _require_vp()
    profile = copy.deepcopy(base_profile) if base_profile else Profile.named("hand")

    doc = pymupdf.open(pdf_path)
    pdf_page = doc[page_index]
    # Clamp before the geometry: both are given the same dpi, and past the
    # pixel cap MuPDF hands back a white page without raising.
    dpi, note = usable_dpi(pdf_page, profile.dpi)
    if note:
        print(note)
    if dpi != profile.dpi:
        profile.dpi = dpi
    img_raw = render_pdf_page(pdf_page, dpi)
    scale, dx, dy = page_geometry(pdf_page, profile, img_raw)
    background = probe_page(img_raw).background_bgr

    print(f"\n=== Tuning on page {page_index + 1} of {pdf_path} ===")
    profile = tune_stages(img_raw, profile, scale, dx, dy, background)

    print("\n--- FINAL RESULT (actual traced page) ---")
    strokes = convert_page_image(img_raw, profile, scale, dx, dy)
    vp(_preview_strokes(strokes, profile))
    return profile


def interactive_convert(pdf_path, output_path, base_profile: Optional[Profile] = None,
                         pages: Optional[List[int]] = None, auto: bool = True) -> Profile:
    """Convert a PDF page by page, previewing and confirming each one.

    Returns the profile as it stood at the end, so a caller can save the
    tuning that came out of the session.
    """
    _require_vp()
    profile = copy.deepcopy(base_profile) if base_profile else Profile.named("hand")

    doc = pymupdf.open(pdf_path)
    page_indices = pages if pages is not None else list(range(doc.page_count))

    out = XoppDocument.new_empty(
        page_count=doc.page_count, width=profile.page_width, height=profile.page_height
    )
    out_pages = out.pages

    for n, i in enumerate(page_indices, start=1):
        pdf_page = doc[i]
        img_raw = render_pdf_page(pdf_page, profile.dpi)
        scale, dx, dy = page_geometry(pdf_page, profile, img_raw)
        probe = probe_page(img_raw)

        print(f"\n=== page {i + 1}/{doc.page_count}  ({n} of {len(page_indices)} selected) ===")
        print(probe.summary())

        while True:
            strokes = convert_page_image(img_raw, profile, scale, dx, dy, auto=auto)
            print(f"\n{len(strokes)} strokes")
            vp(_preview_strokes(strokes, profile))

            cmd = input("[Enter=accept | s=adjust | q=quit]: ").strip().lower()

            if cmd == "":
                for s in strokes:
                    out_pages[i].layer.append(s)
                break
            if cmd == "q":
                out.save(output_path)
                print(f"Stopped at page {i + 1}. Wrote pages accepted so far to {output_path}")
                return profile
            if cmd == "s":
                profile = tune_stages(img_raw, profile, scale, dx, dy, probe.background_bgr)
                continue
            print("Unrecognized; Enter to accept, s to adjust, q to quit.")

    out.save(output_path)
    print(f"\nWrote {output_path}")
    return profile
