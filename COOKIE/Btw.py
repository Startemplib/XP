#!/usr/bin/env python3
"""Turn the dark pictures inside an XOPP white, and their content black.

An XOPP is a gzip-compressed XML document. Pictures are <image> elements
whose text is a base64 PNG; the page's strokes are separate <stroke>
elements. This script walks the pictures only:

    dark picture (median < 96 or >= 60% dark pixels)
        -> neutral pixels: dark background to white, light text to black
        -> chromatic pixels: kept exactly (red stays red, blue stays blue,
           purple stays purple)

**This is not an RGB or HSV brightness inversion.** V inversion turned a
pure red 255,0,0 into 0,0,0. Here a pixel with chroma (max-min) >= 18 is
colour and is copied unchanged; only the neutral pixels are remapped.

Double-click it and choose an XOPP, or pass one on the command line:

    python Btw.py
    python Btw.py Qpsi.xopp --dry-run
    python Btw.py Qpsi.xopp -o Qpsi_white.xopp
"""
from __future__ import annotations

import argparse
import base64
import gzip
import io
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

INK = 96          # "this is background" for detection
DARK_SHARE = 0.60
BG_DARK = 96      # a pixel below this is a candidate for the dark background
CHROMA = 18       # max-min above this is colour, and colour is never changed
SOFT = 12         # width of the neutral black/white transition, in levels


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _neutral_remap(luminance: np.ndarray, background: float) -> np.ndarray:
    """Dark neutral -> white, light neutral -> black, with a soft edge."""
    threshold = max(float(background) + 32.0, 64.0)
    rise = np.clip((luminance.astype(np.float32) - (threshold - SOFT))
                   / (2.0 * SOFT), 0.0, 1.0)
    return (255.0 * (1.0 - rise)).astype(np.uint8)


def selective_colours(image: Image.Image) -> Image.Image:
    """White background, black neutral content, colour copied untouched.

    `chroma` is max(R,G,B) - min(R,G,B). A red pixel, however dark, has a
    large chroma and is copied exactly. A neutral pixel is remapped from the
    picture's own dark background: below the background threshold it becomes
    white, above it black. This is why red no longer becomes black.
    """
    if image.mode in ("L", "1"):
        grey = np.asarray(image.convert("L"), np.uint8)
        dark = grey < BG_DARK
        if dark.mean() < DARK_SHARE:
            return image.convert("L")
        background = float(np.median(grey[dark]))
        return Image.fromarray(_neutral_remap(grey, background), "L")

    rgba = np.asarray(image.convert("RGBA")).copy()
    rgb = rgba[:, :, :3].astype(np.int16)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    luminance = rgb.mean(axis=2)
    dark = luminance < BG_DARK
    if dark.mean() < DARK_SHARE:
        if image.mode == "RGB":
            return image.copy()
        return image.convert("RGBA")
    background = float(np.median(luminance[dark]))
    neutral = chroma < CHROMA
    remapped = _neutral_remap(luminance, background)
    rgb[neutral] = remapped[neutral, None]
    # The chromatic pixels are not touched: no hue shift, no V inversion.
    rgba[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    if image.mode == "RGB":
        return Image.fromarray(rgba[:, :, :3], "RGB")
    return Image.fromarray(rgba, "RGBA")


def free_name(path: Path) -> Path:
    if not path.exists():
        return path
    for number in range(1, 1000):
        candidate = path.with_name(f"{path.stem}-{number}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise SystemExit(f"no free name beside {path}")


def pick_an_xopp() -> Path | None:
    """The window a double-click opens: one XOPP, or nothing."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    chosen = filedialog.askopenfilename(
        parent=root,
        title="Choose an XOPP whose dark pictures should become white",
        filetypes=[("Xournal++ notebook", "*.xopp"),
                   ("All files", "*.*")],
    )
    root.destroy()
    return Path(chosen) if chosen else None


def process(source: Path, out: Path, dry_run: bool,
            min_dark: float, max_median: float):
    """Invert the dark pictures and write a new XOPP.

    Returns `(out, inspected, inverted)`; raises SystemExit for a file that
    is not an XOPP or for an output that would overwrite the input.
    """
    if not source.is_file():
        raise SystemExit(f"not a file: {source}")
    if out == source:
        raise SystemExit("refusing to overwrite the input XOPP")

    with gzip.open(source, "rb") as handle:
        raw = handle.read()
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise SystemExit(f"not an XOPP (gzip XML): {exc}") from exc

    pages = [element for element in root if local_name(element.tag) == "page"]
    if not pages:
        raise SystemExit("no <page> elements: this is not an XOPP")

    inverted = 0
    inspected = 0
    print(f"{source.name}: {len(pages)} pages")
    print("page  image  native       dark%  median  action")

    for page_number, page in enumerate(pages, start=1):
        for image_number, element in enumerate(
                (e for e in page.iter() if local_name(e.tag) == "image")):
            encoded = (element.text or "").strip()
            if not encoded:
                continue
            inspected += 1
            try:
                picture = Image.open(io.BytesIO(base64.b64decode(encoded)))
                grey = np.asarray(picture.convert("L"), np.uint8)
            except Exception as exc:
                print(f"{page_number:>4}  {image_number:>5}  SKIP  {exc}")
                continue

            dark = float((grey < INK).mean())
            median = float(np.median(grey))
            action = "INVERT" if (dark >= min_dark
                                  or median < max_median) else "keep"
            print(f"{page_number:>4}  {image_number:>5}  "
                  f"{picture.width:>5}x{picture.height:<5}  "
                  f"{dark:>5.1%}  {median:>6.0f}  {action}")

            if action == "INVERT" and not dry_run:
                buffer = io.BytesIO()
                selective_colours(picture).save(buffer, format="PNG")
                element.text = base64.b64encode(buffer.getvalue()).decode("ascii")
                inverted += 1

    print(f"\n{inspected} picture(s); {inverted if not dry_run else 0} inverted")
    if dry_run:
        print("dry run: nothing written")
        return out, inspected, 0

    out = free_name(out)
    xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with gzip.open(out, "wb", compresslevel=6) as handle:
        handle.write(xml)

    # Read the new file back; an output nobody can reopen is not a result.
    with gzip.open(out, "rb") as handle:
        check = ET.fromstring(handle.read())
    check_pages = [e for e in check if local_name(e.tag) == "page"]
    check_images = sum(1 for e in check.iter() if local_name(e.tag) == "image")
    check_strokes = sum(1 for e in check.iter() if local_name(e.tag) == "stroke")
    print(f"wrote {out}")
    print(f"verified: {len(check_pages)} pages, {check_images} pictures, "
          f"{check_strokes} strokes")
    return out, inspected, inverted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xopp", nargs="?", type=Path,
                        help="XOPP to process; omit it to open a file picker")
    parser.add_argument("-o", "--out", type=Path,
                        help="output XOPP (default: <name>_white.xopp)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report only; write nothing")
    parser.add_argument("--min-dark", type=float, default=DARK_SHARE)
    parser.add_argument("--max-median", type=float, default=INK)
    args = parser.parse_args()

    picked = args.xopp is None
    source = pick_an_xopp() if picked else args.xopp
    if source is None:
        print("no file chosen")
        return 0
    source = source.expanduser().resolve()
    out = (args.out or source.with_name(f"{source.stem}_white.xopp")).resolve()

    try:
        out, inspected, inverted = process(
            source, out, args.dry_run, args.min_dark, args.max_median)
    except SystemExit as exc:
        if picked:
            from tkinter import messagebox
            messagebox.showerror("Btw", str(exc))
            return 1
        raise

    if picked:
        from tkinter import messagebox
        if args.dry_run:
            messagebox.showinfo(
                "Btw",
                f"{inspected} picture(s) inspected; dry run, nothing written.")
        else:
            messagebox.showinfo(
                "Btw",
                f"{inspected} picture(s) inspected; {inverted} inverted.\n\n"
                f"Wrote:\n{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
