"""Two documents to look at, whatever they are, with nothing converted.

    "Render First -> Select -> Convert Selected Content -> Review -> Merge.
     Do not run PDF -> XOPP conversion when the merge window opens."

The merge window could not do that, and the reason is structural rather than
an oversight:

    class MergeSession:
        def __init__(self, left: XoppDocument, right: XoppDocument, ...)

Both sides had to be documents of strokes, so a `.pdf` had to be converted
before the window could be built -- about twenty minutes for the user's notebook,
spent before anybody had seen anything. Everything downstream of that
constructor is built on strokes: `strokes_for_page`, the views, the brush,
`export`.

A `Side` is the other thing a document can be: **pages you can look at**. It
answers three questions and refuses to answer a fourth.

    pages        how many there are
    page_box     how big one is, in page units
    render       what one looks like, at a scale, optionally one rectangle
                 of it -- pixels, and nothing else

    strokes      the native strokes, and **None for a PDF**

That last one is the whole discipline. A PDF has no strokes; it has ink that
could be *made into* strokes by a conversion that costs twenty minutes.
Returning None is the honest answer and it is what keeps a careless caller
from reintroducing the thing this exists to prevent. Nothing in this module
imports the stroke pipeline, and a test asserts that rather than trusting it.

RENDERING A RECTANGLE, not the page. `render` takes a `box`, and for a PDF
it is passed to PyMuPDF as a clip, so a selection of a corner of a page
rasterizes a corner of a page. That is what makes the cost of a selected
region follow the region rather than the document, which is the user's section 6.

PAGE UNITS. A `.xopp` page unit and a PDF point are both 1/72 inch, so one
`scale` in pixels per unit means the same thing on both sides. Where the two
documents' page boxes differ, whose box wins is the *caller's* decision and
not this module's -- see `incremental.Sources`, which puts the target on the
base's box because there the base is the structure being kept.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from PIL import Image

from xp.render.drawing import BACKGROUND_COLOR, draw_strokes, render_page
from xp.core.document.xopp_io import XoppDocument

# A page unit in a `.xopp` is a PostScript point, as it is in a PDF.
POINTS_PER_INCH = 72.0

XOPP, PDF = 'xopp', 'pdf'


def _as_bgr(image) -> np.ndarray:
    """A PIL RGB image as the BGR array the rest of this works in."""
    return np.asarray(image.convert('RGB'))[:, :, ::-1].copy()


def _the_blue_pieces(page) -> list:
    """The page's strokes that are the frame's blue, parsed -- and nothing else parsed.

    `xp.core.frame._is_blue` is the border detector's own test and it only needs the colour
    attribute. A piece that is not blue cannot be the printed border, so its coordinates are never
    read; a page with no blue at all comes back empty and is skipped whole.
    """
    from xp.core.document.xopp_io import _parse_stroke
    from xp.core.frame import _is_blue

    try:
        # The same call `Page.strokes` makes, so a lazy document is asked in the way it expects.
        elements = [element for element in page.layer.findall('stroke')
                    if _is_blue(element.get('color', ''))]
    except Exception:
        # A page that cannot be asked is left to the slow path, which decides properly.
        return list(page.strokes())
    return [_parse_stroke(element) for element in elements]


class Side:
    """One document, as pages to look at. Opening one converts nothing."""

    def __init__(self, path):
        self.path = Path(path)
        suffix = self.path.suffix.lower()
        if suffix == '.xopp':
            self.kind = XOPP
            # Big ones are opened a page at a time; see `bigxopp`. The user's own
            # notebook is 3 GB of XML and 17.6 million strokes, which is
            # past what one string or one tree can be.
            from xp.core.document.bigxopp import open_document
            self.document = open_document(self.path)
            # The template's printed border is paper, not writing: its
            # centreline is measured for the file's calibration first, then the
            # strokes come out of the copy in memory, so the view, the brush,
            # the matcher and the export all see the same list. See
            # `drop_the_print_frame` and `file_frame`.
            self._removed = [0]
            try:
                self.drop_the_print_frame()
            except Exception:
                # A document too big to walk page by page is still a document;
                # its border, if it has one, is left where it is.
                pass
            self._pdf = None
        elif suffix == '.pdf':
            import pymupdf
            self.kind = PDF
            self.document = None
            self._pdf = pymupdf.open(str(self.path))
        else:
            raise ValueError(f'not a notebook or a PDF: {self.path}')
        self._indexed = {}

    # --- what it is ------------------------------------------------------

    @property
    def pages(self) -> int:
        if self.kind == XOPP:
            return len(self.document.pages)
        return self._pdf.page_count

    def page_box(self, index: int) -> Tuple[float, float]:
        """The page's size in page units -- points, on either kind.

        A lazily-opened document answers this off the page's opening tag,
        because parsing one of the user's pages to read its width would be up to
        223 MB of XML for two numbers.
        """
        if self.kind == XOPP:
            quick = getattr(self.document, 'page_box', None)
            if quick is not None:
                return quick(index)
            page = self.document.pages[index]
            return float(page.width), float(page.height)
        rect = self._pdf[index].rect
        return float(rect.width), float(rect.height)

    def file_frame(self):
        """The frame of the whole file: the pages that have one, or a render.

        **A PDF's geometry belongs to the file.** Some of its pages carry the
        printed border and some do not -- content added later, a crop, a
        different template -- and giving up on the ones without it is what made
        those pages fall back to guessing. One reliable border is enough, and
        the result is what every page of that file is calibrated against.

        Returns `(frame, pages_used, spread)`; `spread` is how far the pages
        that did have a border disagree with each other.
        """
        from xp.core.frame import file_frame as combine

        measured = getattr(self, '_frames', None)
        if measured:
            return combine(list(measured.values()))
        if self.kind == XOPP:
            # No border drawn in the strokes. A notebook written on the
            # template usually names the PDF it was written on, and the
            # template's border is in that file -- the same source the
            # per-page geometry used to look at, now used once for the file.
            from xp.core.frame import Frame, frame_in_pdf

            found = {}
            for index, page in enumerate(self.document.pages[:6]):
                behind = None
                try:
                    from xp.core.frame import background_pdf

                    behind = background_pdf(page, self.document, Path(self.path).parent)
                except Exception:
                    behind = None
                if behind is None:
                    continue
                try:
                    seen = frame_in_pdf(str(behind[0]), behind[1])
                except Exception:
                    seen = None
                if seen is not None:
                    found[index] = Frame(seen.x0, seen.y0, seen.x1, seen.y1,
                                         'the PDF behind a page of this file')
            if found:
                self._frames = found
                # `.values()`, not the dict: `combine` walks frames, and
                # iterating a dict walks its keys -- which are page numbers.
                return combine(list(found.values()))
            return (None, 0, 0.0)
        # A picture: look at a few pages for the template's blue border.
        from xp.core.frame import Frame, centreline_in_image

        found = {}
        scale = 2.0

        def agrees_with_the_first() -> bool:
            """Whether a second page has already given the first page's frame, exactly."""
            if len(found) < 2:
                return False
            frames = list(found.values())
            first = frames[0]
            for other in frames[1:]:
                if (abs(other.x0 - first.x0) > 1e-9 or abs(other.y0 - first.y0) > 1e-9
                        or abs(other.x1 - first.x1) > 1e-9 or abs(other.y1 - first.y1) > 1e-9):
                    return False
            return True

        for index in range(min(self.pages, 6)):
            try:
                seen = centreline_in_image(self.render(index, scale))
            except Exception:
                seen = None
            if seen is not None:
                found[index] = Frame(seen.x0 / scale, seen.y0 / scale,
                                     seen.x1 / scale, seen.y1 / scale,
                                     'seen in a page of this file')
            # **Two pages that agree are the answer.** `combine` of identical frames *is* that frame, so
            # stopping here gives the same box for less work; if they disagree the scan goes on to six,
            # which is what the six pages are for.
            if agrees_with_the_first():
                break
        self._frames = found
        return combine(list(found.values()))

    def page_frame(self, index: int):
        """This page's own frame, measured **on demand** if it has not been yet.

        The strokes of a notebook no longer carry the border, so the frames are
        recorded while the border is removed, for every page. A picture side is
        another matter: `file_frame` scans only the first six pages for the
        template's border, so page 7 and later had no entry -- and `Fit` on one
        of them, told to calibrate from the page on screen, found nothing and
        silently kept the file's calibration. The user asked for that page; it
        costs one render to answer.

        The answer is kept, so pressing Fit twice pays once.
        """
        # **Before the frame is looked for.** Without the cleaning the border is still in the strokes, and this
        # would fall through to `_measure_this_page`, which measures from a render -- a different number, and one
        # that would move the fit.
        self.use(index)
        frames = getattr(self, '_frames', None)
        if frames is None:
            frames = self._frames = {}
        found = frames.get(index)
        if found is not None:
            return found
        found = self._measure_this_page(index)
        if found is not None:
            frames[index] = found
            # The file's frame is the median of the pages: a new page can move
            # it, so nothing that was computed from the old set may be kept.
            self._file_frame_cache = None
        return found

    def _measure_this_page(self, index: int):
        """One page's frame, from its own content, without moving the session.

        A notebook: the PDF behind that page, if it names one (its own strokes
        carry no border any more). A picture: its own pixels, at twice the
        resolution the border is a line at.
        """
        from xp.core.frame import (Frame, background_pdf, centreline_in_image,
                                   centreline_frame, frame_in_pdf,
                                   frame_in_strokes)

        try:
            if self.kind == XOPP:
                if index < 0 or index >= len(self.document.pages):
                    return None
                page = self.document.pages[index]
                seen = frame_in_strokes(page.strokes(), self.page_box(index))
                if seen is not None:
                    seen = centreline_frame(page.strokes(), seen) or seen
                    return Frame(seen.x0, seen.y0, seen.x1, seen.y1,
                                 'measured on this page just now')
                try:
                    behind = background_pdf(page, self.document,
                                            Path(self.path).parent)
                except Exception:
                    behind = None
                if behind is None:
                    return None
                seen = frame_in_pdf(str(behind[0]), behind[1])
                if seen is None:
                    return None
                return Frame(seen.x0, seen.y0, seen.x1, seen.y1,
                             'the PDF behind this page, measured just now')
            scale = 2.0
            if index < 0 or index >= self.pages:
                return None
            seen = centreline_in_image(self.render(index, scale))
            if seen is None:
                return None
            return Frame(seen.x0 / scale, seen.y0 / scale,
                         seen.x1 / scale, seen.y1 / scale,
                         'seen on this page just now')
        except Exception as exc:
            print('[sides] page %d could not be measured: %s' % (index, exc),
                  flush=True)
            return None

    _cleaner = None
    _cleaned = None
    _removed = None

    @property
    def dropped_print_frame(self) -> int:
        """How many border pieces have been taken out **so far**.

        A running count, because cleaning is per page: the number is not known at the end of the open any more,
        and reading a value captured then would say 0 after the whole notebook had been cleaned.
        """
        return sum(getattr(self, '_removed', None) or [0])

    def drop_the_print_frame(self) -> int:
        """Remove the template's printed border from the notebook we loaded.

        **In place, on the copy in memory; the file is not touched.** The border
        is paper -- the user's instruction is that it must never be seen and must not
        be in what is exported -- and a page converted by an older build still
        carries it as strokes: the user's own `TEST_converted-Arra1.xopp` has **3,028**
        blue pieces lying on the frame's rectangle.

        It has to happen here, at the document level, rather than by filtering
        the lists the merger builds: decisions are filed by stroke *index*, and
        a filter applied after the load shifts those indices against the XML the
        export walks -- the decisions would then be applied to the wrong strokes.
        Removing the elements leaves one list, the same one everywhere.

        Returns how many pieces went.
        """
        from xp.core.frame import (centreline_frame, frame_in_strokes,
                                   is_print_frame_piece)

        removed = self._removed = [0]
        removed_by_page = {}
        frames = self._frames = getattr(self, '_frames', None) or {}

        def clean(index: int, page) -> None:
            """One page: measure the border, then take it out.

            The same function is used for the pass below and as the document's
            reload hook, so a page that comes back from the scratch file is
            cleaned exactly as it was the first time.
            """
            # **Only the blue pieces are parsed.** The border is the frame's blue, and both
            # `frame_stroke` and `is_print_frame_piece` test the colour before they build an array
            # (their own comments say why) -- so a stroke that is not blue cannot be the border and
            # its coordinates are never needed. On the user's notebook that is 103,935 strokes across 68
            # pages, 3.7 s of a 4.4 s open, and almost none of them blue.
            strokes = _the_blue_pieces(page)
            if not strokes:
                return
            # Found anywhere on the page, not only on the template's own
            # rectangle: a PDF whose pages were printed or cropped differently
            # carries its border somewhere else, and that is exactly the file
            # that needs calibrating.
            outline = frame_in_strokes(strokes, (page.width, page.height))
            if outline is not None:
                frames[index] = centreline_frame(strokes, outline) or outline
            count = 0
            for stroke in list(strokes):
                if is_print_frame_piece(stroke, outline):
                    page.remove_stroke(stroke)
                    count += 1
            # Reloading the same source page must not count its border twice.
            previous = removed_by_page.get(index, 0)
            removed_by_page[index] = max(previous, count)
            removed[0] += max(0, count - previous)

        if hasattr(self.document, 'cleaner'):
            # A lazy document evicts pages; the hook keeps them clean.
            self.document.cleaner = clean
        self._cleaner = clean
        self._cleaned = set()
        # **Every page is cleaned at open, and that is not laziness for its own sake.** A page's frame is recorded
        # while its border is removed, and `merge.file_frame` combines those frames to calibrate the file: a page
        # nobody touched has no entry, and `_frame_fits` then falls back to the ink. The full scan showed it --
        # `test_fit_and_crop` used one page where it expected two, and `test_frame` reported "fitted to the ink"
        # where it should have said "file calibration". So the eager pass is the default, and the per-page path
        # stays behind `XP_LAZY_CLEAN=1` for whoever makes `file_frame` independent of it.
        import os

        if os.environ.get('XP_LAZY_CLEAN'):
            self.use(0)
        else:
            for index, page in enumerate(self.document.pages):
                self.use(index)
        return removed[0]

    def use(self, index: int) -> None:
        """Make sure this page's printed border is out of the strokes, once.

        Opening cleans the first page; other pages are cleaned before use.
        Successful repeated requests do no work; failures remain retryable.
        """
        cleaned = getattr(self, '_cleaned', None)
        cleaner = getattr(self, '_cleaner', None)
        if cleaned is None or cleaner is None or index in cleaned:
            return
        try:
            cleaner(index, self.document.pages[index])
        except Exception as exc:
            # **A failure is not a cleaning.** The page stays out of `_cleaned`, so the next caller tries again,
            # and the reason is said out loud with the page it happened on -- the same rule the page turn follows.
            print('[sides] page %d could not be cleaned: %s: %s'
                  % (index, type(exc).__name__, exc), flush=True)
            return
        cleaned.add(index)

    def beyond_the_first_layer(self, index: int) -> dict:
        """What this page holds that the merge will not handle, or empty.

        A PDF has no layers, no text and no images of its own -- its page *is*
        a picture, and the merge handles it as one.

        **Pictures are no longer in this list.** They used to be: the merge
        could not see an `<image>`, so it said so. Now the page is rendered with
        its pictures, compared as an image, and the pictures are converted to
        native strokes at export -- so the only thing left to warn about is
        content on another layer, which really is carried through untouched.
        """
        if self.kind != XOPP:
            return {}
        self.use(index)
        try:
            counts = self.document.pages[index].beyond_the_first_layer()
        except Exception:
            return {}
        return {name: count for name, count in counts.items()
                if count and name != 'image'}

    def strokes(self, index: int) -> Optional[List]:
        """The page's native strokes, or None if it has none.

        None for a PDF, always. A PDF has ink, not strokes; turning the one
        into the other is the conversion this whole flow exists to postpone
        until the user has said where and whether. A caller that wants strokes out
        of a PDF has to go and ask for a conversion by name.
        """
        if self.kind == XOPP:
            self.use(index)
            return self.document.pages[index].strokes()
        return None

    # --- what it looks like ----------------------------------------------

    def render(self, index: int, scale: float, box=None) -> np.ndarray:
        """One page, or one rectangle of it, as a BGR array.

        `box` is (left, top, width, height) in the pixels of this `scale`.
        For a PDF it becomes a clip, so rasterizing a corner of a page costs
        a corner of a page; that is what lets the cost of a selection follow
        the selection.
        """
        if self.kind == PDF:
            return self._render_pdf(index, scale, box)
        self.use(index)                     # a page drawn with its printed border in it is a page drawn wrong
        if box is None:
            return _as_bgr(render_page(self.document.pages[index], scale=scale))
        return self._render_xopp_box(index, scale, box)

    def _render_xopp_box(self, index: int, scale: float, box) -> np.ndarray:
        """One rectangle of a notebook page, drawing only what is in it.

        Drawing the page and cropping is what this replaces, and the measure
        said so: on the user's TEST pair a rectangle cost 1.32s against 1.66s for
        the whole page, while the PDF side -- which really does rasterize
        only the clip -- cost 0.04s against 0.22s. A side that charges for
        the whole document when the user points at a corner of it is the thing
        section 6 forbids, whichever side it is.

        A stroke's own width is added to its box before it is judged, or a
        thick stroke whose spine is just outside would lose its edge at the
        rim.
        """
        left, top, width, height = (int(round(value)) for value in box)
        left, top = max(0, left), max(0, top)
        width, height = max(1, width), max(1, height)
        right, bottom = left + width, top + height
        strokes, boxes, half = self._index(index)
        # Vectorised, because the loop was the cost and not the drawing.
        # `page.strokes()` re-parses the page's XML on every call and
        # `Stroke.bbox` re-walks every point of every stroke on every read:
        # on a 140,203-stroke page that was 0.76s to decide what to draw and
        # a fraction of it to draw. Both are done once per page here.
        pad = half * scale + 1.0
        keep = ~((boxes[:, 2] * scale + pad < left)
                 | (boxes[:, 0] * scale - pad > right)
                 | (boxes[:, 3] * scale + pad < top)
                 | (boxes[:, 1] * scale - pad > bottom))
        wanted = [strokes[i] for i in np.nonzero(keep)[0]]
        image = Image.new('RGB', (width, height), BACKGROUND_COLOR)
        # **The page's pictures too**, drawn through the same window: without
        # them a page that carries an embedded picture is shown with a hole in
        # it, which is what made tracing every picture up front look necessary.
        #
        # **The two offset conventions are not the same, and this passed the
        # wrong one.** `draw_strokes` shifts in *pixels* (`x * scale - ox`);
        # `draw_pictures` shifts in *page units* (`(x - ox) * scale`). Handing
        # it `left, top` -- a pixel count -- put a picture at 40 units on a page
        # window starting at 444 px at `(40 - 444) * scale`, four thousand
        # pixels to the left of the canvas: invisible, at every zoom, while the
        # whole-page path (offset 0) drew it perfectly. That is the user's "the
        # embedded picture is not there when I zoom in".
        from xp.render.drawing import draw_pictures

        draw_pictures(image, self.document.pages[index], scale,
                      offset=(left / scale, top / scale))
        draw_strokes(image, wanted, scale, offset=(float(left), float(top)))
        return _as_bgr(image)

    def _index(self, index: int):
        """This page's strokes, their boxes and their half-widths, once.

        Held because a selection is dragged: the window moves and the page
        is drawn again and again, and nothing about the strokes has changed
        between one frame and the next.
        """
        held = self._indexed.get(index)
        if held is not None:
            return held
        strokes = self.document.pages[index].strokes()
        boxes = np.empty((len(strokes), 4), np.float64)
        half = np.empty(len(strokes), np.float64)
        for at, stroke in enumerate(strokes):
            boxes[at] = stroke.bbox
            half[at] = (max(stroke.widths) if stroke.widths else 1.0) / 2.0
        held = (strokes, boxes, half)
        self._indexed[index] = held
        return held

    def _render_pdf(self, index: int, scale: float, box=None) -> np.ndarray:
        import pymupdf

        page = self._pdf[index]
        zoom = scale                       # pixels per point, which is what
        matrix = pymupdf.Matrix(zoom, zoom)  # a page unit is on both sides
        clip = None
        if box is not None:
            left, top, width, height = box
            clip = pymupdf.Rect(left / scale, top / scale,
                                (left + width) / scale, (top + height) / scale)
        pix = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csRGB, clip=clip)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        return image[:, :, ::-1].copy()    # RGB -> BGR

    def render_box(self, index: int, scale: float, box):
        """The part of this page that falls inside `box`, and where it goes.

        `box` is (left, top, width, height) in the pixels of `scale`, and it
        may hang off the page on any side -- it is a window on the merged
        page, and this page may sit anywhere inside that. Returns
        (pixels, put_x, put_y) with the offsets into the window, or None
        when none of the page is in it.

        Only the overlap is rasterized. That is the whole point: a window on
        a page at 8x asked for the whole page was 229 megapixels for the
        patch on screen.
        """
        page_w, page_h = self.page_pixels(index, scale)
        left, top, width, height = (int(round(v)) for v in box)
        x0, y0 = max(0, left), max(0, top)
        x1, y1 = min(page_w, left + width), min(page_h, top + height)
        if x1 <= x0 or y1 <= y0:
            return None
        piece = self.render(index, scale, box=(x0, y0, x1 - x0, y1 - y0))
        return piece, x0 - left, y0 - top

    def page_pixels(self, index: int, scale: float) -> Tuple[int, int]:
        width, height = self.page_box(index)
        return max(1, round(width * scale)), max(1, round(height * scale))

    # --- housekeeping ----------------------------------------------------

    def close(self):
        if self._pdf is not None:
            self._pdf.close()
            self._pdf = None
        shut = getattr(self.document, 'close', None)
        if shut is not None:
            shut()          # a BigDocument has a scratch file to let go of

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self):
        return f'<Side {self.kind} {self.path.name} {self.pages} pages>'


class Pair:
    """The two sides of a merge, in any of the three combinations.

    PDF + XOPP, PDF + PDF and XOPP + XOPP all open the same way, because
    opening is only rendering. Which side is the base -- whose structure is
    kept and whose page box the other is placed on -- is a decision made
    later and by the user, not here and not by which file the user happened to pick
    first.
    """

    def __init__(self, left, right):
        self.left = left if isinstance(left, Side) else Side(left)
        self.right = right if isinstance(right, Side) else Side(right)

    @property
    def pages(self) -> int:
        """The pages both sides have. Position for position; anything
        cleverer is `pagealign`'s business and is settled before this."""
        return min(self.left.pages, self.right.pages)

    @property
    def kinds(self) -> Tuple[str, str]:
        return self.left.kind, self.right.kind

    def side(self, which: str) -> Side:
        if which in ('left', 'base'):
            return self.left
        if which in ('right', 'target'):
            return self.right
        raise ValueError(f'no such side: {which!r}')

    def can_copy_strokes(self) -> bool:
        """Whether the native strokes of one side can simply be taken.

        Only when both sides have strokes to begin with. The user's section 4: if
        the source and target are both XOPP and the chosen content can be
        taken as native strokes, copy them rather than forcing a
        re-vectorisation of ink that is already vector.
        """
        return self.kinds == (XOPP, XOPP)

    def close(self):
        self.left.close()
        self.right.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
