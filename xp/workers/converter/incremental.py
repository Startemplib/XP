"""Two documents as pixels, so a merge can decide before it converts.

    "Why does merging a PDF with an XOPP still convert the PDF first?
     Shouldn't it compare the pixels first, then convert only the necessary
     parts after I select the merge mode?"

Yes, and it did not. `cli.load_side` sends a `.pdf` straight through
`convert.convert_pdf` before the merge window is even built, so the whole
twenty minutes is spent before anybody has been shown a single difference.
`imagediff` and `patchmerge` were finished and tested and nothing in `xp/`
imported either of them -- the engine had no door. This is the door.

THE DISTINCTION THE OLD PATH DID NOT MAKE, and it is the whole saving:

    rasterizing a PDF        pixels. 61 pages, seconds.
    converting a PDF         strokes. 61 pages, about twenty minutes.

Comparison needs the first. Only `Editable Incremental` needs the second,
and then only for the regions a person picked. Nothing here calls
`convert_pdf`, `orthofill` or any other part of the stroke pipeline; the
import list is the proof, and a test asserts it rather than trusting it.

BOTH SIDES THROUGH THE SAME PAGE BOX. `patchmerge.apply` turns a rectangle
of pixels back into page units with one `scale`, which is only true if the
two sides were rendered through the same box. The base `.xopp` decides it --
it is the structure being added to, and its pages are the ones that have to
stay true -- and the target is resampled onto it. A PDF exported at a
slightly different paper size therefore lines up, instead of registering as
a page-wide change.

WHAT IS NOT DONE HERE. The user's section 8, the coarse-to-fine chunked
rasterizing. Pages are rendered once each at `COMPARE_DPI`; `imagediff`
reduces them to 320px internally to find the shift and to 640px to settle
it, so the comparison itself is already cheap at 0.07s a page. The
rasterizing is the 17.8s, and it is not cut here.
"""

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import ImageColor

from xp.core.imaging import imagediff
from xp.workers.converter import patchmerge
from xp.render.drawing import BACKGROUND_COLOR, render_page
from xp.core.document.xopp_io import XoppDocument

# Pixels per inch for the comparison renders, and for the patches cut out of
# them. A patch is shown at the size the page is printed, so this is also how
# sharp a `Fast Visual` merge looks: 150 is about twice a screen, and about a
# quarter of the pixels the stroke pipeline works at.
COMPARE_DPI = 150

# A page unit in a `.xopp` is a PostScript point, as it is in a PDF.
POINTS_PER_INCH = 72.0


def _as_bgr(image) -> np.ndarray:
    """A PIL RGB image as the BGR array the rest of this works in."""
    return np.asarray(image.convert('RGB'))[:, :, ::-1].copy()


class Sources:
    """A base `.xopp` and a target `.pdf` or `.xopp`, rendered on demand.

    Call it like `render(side, index)` -- that is the shape `patchmerge.plan`
    and `patchmerge.apply` ask for. Pages are rendered when asked for, and
    the last few are kept: `apply` asks again for the handful of pages that
    got patches, and re-rendering those is wasted work. A whole document of
    renders is not kept, because 61 pages at 150 dpi is about 400 MB and the
    last thing this needs is to trade twenty minutes for an out-of-memory.
    """

    def __init__(self, base_path, target_path, dpi: int = COMPARE_DPI,
                 keep: int = 4):
        base_path, target_path = Path(base_path), Path(target_path)
        if base_path.suffix.lower() != '.xopp':
            raise ValueError(
                'the base has to be the .xopp: it is the structure that is '
                'kept, and a PDF has none to keep')
        self.dpi = int(dpi)
        self.scale = self.dpi / POINTS_PER_INCH
        self.document = XoppDocument.load(base_path)
        self.base_path, self.target_path = base_path, target_path
        self._target_doc = None
        self._pdf = None
        if target_path.suffix.lower() == '.pdf':
            import pymupdf
            self._pdf = pymupdf.open(str(target_path))
            self._target_count = self._pdf.page_count
        else:
            self._target_doc = XoppDocument.load(target_path)
            self._target_count = len(self._target_doc.pages)
        self._keep, self._cache = int(keep), {}

    # --- how many pages there are, which is all `plan` asks of them -------

    @property
    def base(self):
        return self.document.pages

    @property
    def target(self):
        return range(self._target_count)

    def close(self):
        if self._pdf is not None:
            self._pdf.close()
            self._pdf = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # --- the renders ------------------------------------------------------

    def page_pixels(self, index: int):
        """The size both sides are rendered at: the base page's box."""
        page = self.document.pages[index]
        return (max(1, round(page.width * self.scale)),
                max(1, round(page.height * self.scale)))

    def __call__(self, side: str, index: int):
        key = (side, index)
        if key in self._cache:
            return self._cache[key]
        made = self._render(side, index)
        if len(self._cache) >= self._keep:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = made
        return made

    def _render(self, side: str, index: int):
        if index >= len(self.document.pages):
            return None
        want = self.page_pixels(index)
        if side == 'base':
            return _as_bgr(render_page(self.document.pages[index],
                                       scale=self.scale))
        if index >= self._target_count:
            return None
        if self._pdf is not None:
            from xp.workers.converter.convert import (render_pdf_page,
                                                       usable_dpi)
            # The page's own ceiling: past it the render is a white page and
            # the comparison would find the whole page missing.
            dpi, _note = usable_dpi(self._pdf[index], self.dpi)
            raw = render_pdf_page(self._pdf[index], dpi)
            return self._placed(raw, index)
        image = _as_bgr(render_page(self._target_doc.pages[index],
                                    scale=self.scale))
        if (image.shape[1], image.shape[0]) != want:
            image = cv2.resize(image, want, interpolation=cv2.INTER_AREA)
        return image

    def _alignment(self, raw, index):
        """How the target's pixels land on the base's.

        Measured from the two renders, not assumed from a profile -- see
        `placement`, which holds the reasoning and is shared with the select
        window so there is one answer to this question and not two.
        """
        from xp.workers.converter import placement
        return placement.place(self('base', index), raw)

    def _placed(self, raw, index):
        """The raw render, put where the base's own page says it goes."""
        want_w, want_h = self.page_pixels(index)
        factor, at_x, at_y = self._alignment(raw, index)
        new_w = max(1, int(round(raw.shape[1] * factor)))
        new_h = max(1, int(round(raw.shape[0] * factor)))
        resized = cv2.resize(
            raw, (new_w, new_h),
            interpolation=cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR)
        page = np.zeros((want_h, want_w, 3), np.uint8)
        # The same paper the base is drawn on. BACKGROUND_COLOR is a name,
        # not a triple, and the arrays here are BGR.
        page[:, :] = ImageColor.getrgb(BACKGROUND_COLOR)[::-1]
        at_x, at_y = int(round(at_x)), int(round(at_y))
        # Clip: ink that runs off the base page keeps what fits, because the
        # base page is the one that has to stay true.
        src_x, src_y = max(0, -at_x), max(0, -at_y)
        put_x, put_y = max(0, at_x), max(0, at_y)
        wide = min(new_w - src_x, want_w - put_x)
        tall = min(new_h - src_y, want_h - put_y)
        if wide > 0 and tall > 0:
            page[put_y:put_y + tall, put_x:put_x + wide] =                 resized[src_y:src_y + tall, src_x:src_x + wide]
        return page


def look(sources: Sources, paper=None, progress: Callable = None):
    """What differs, page by page. Changes nothing and converts nothing."""
    return patchmerge.plan(sources.base, sources.target, sources,
                           paper=paper, progress=progress)


def fast_visual(sources: Sources, made, pages=None, paper=None,
                progress: Callable = None):
    """Put the chosen pages' new ink in as a patch, and nothing else.

    `pages` is the 1-based page numbers the user confirmed; left out, every page
    the plan found something on. Conflicts are never applied: an image cannot
    delete a stroke, and a merge that pretended otherwise would leave the old
    ink live underneath a picture of its absence.
    """
    if pages is not None:
        wanted = set(int(number) for number in pages)
        made = patchmerge.Plan([page for page in made.pages
                                if page.number in wanted])
    return patchmerge.apply(sources.document, made, sources,
                            scale=sources.scale, paper=paper,
                            progress=progress)
