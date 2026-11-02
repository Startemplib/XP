"""Fast Visual Merge: put the new notes on the old document, and nothing else.

The user's second step, and the user's words for what it is:

    "Keep all the pages, layers and strokes of the original XOPP. Compute the
     visual difference. Make image patches only of the regions that are
     genuinely new or changed. Add them to the XOPP at the right place, size
     and layer order. Regions that did not change are not processed at all."

    "The original XOPP is the structural base."

So this does not trace, does not fill, does not join and does not re-trace a
single existing stroke. A page that did not change is copied from the base
document by reference and never looked at again. What it costs follows what
changed, which is the whole reason for it: a full conversion of the sample notebook
is twenty minutes and this is seconds, because sixty of the sixty-one pages
have nothing in them that is new.

**WHAT GOES IN AND WHAT DOES NOT.** An `added` region becomes a patch. A
`removed` or `modified` region does **not**, and this is the part the user was
most specific about: painting an image over a stroke does not delete the
stroke, it hides it. Those regions are reported as conflicts and left for a
person. Silently covering them would produce a document that looks right and
is not, with the old ink still in it and still editable underneath.

**WHAT A PATCH CONTAINS.** The target's own pixels over the whole changed
rectangle, with everything that is not new ink made transparent. Not
`target AND NOT base`: where a new stroke crosses an old one that
subtraction leaves the new stroke cut into pieces, with a hole at every
crossing. The transparency is computed from how far each pixel is from the
paper, so the patch carries the new ink at its own edges and lets the page
through everywhere else.

**WHERE IT GOES.** A layer of its own, on top, named so a merge can be taken
out again. The base document's own layers are not touched, not reordered and
not rewritten.

**WHAT IS NOT DONE HERE.** No page is rendered at full resolution unless it
has a patch to cut out of it, no document is converted, and nothing decides
whether a conflict should be applied -- this reports them and stops. Mode II,
where new regions become editable strokes, is a different module and uses the
same differences.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from xp.core.imaging import imagediff
from xp.core.imaging.inkdiff import INK_DELTA

# The name a merge puts on its layer. One place, because taking a merge out
# again means finding it, and finding it by guessing is how the wrong layer
# gets removed.
MERGE_LAYER = 'XP merge'

# How a patch's transparency is made: a pixel's distance from the paper,
# scaled so that INK_DELTA -- the distance at which this project calls
# something ink -- is already fully opaque. Below it the pixel fades out
# rather than being cut off, which is what keeps an anti-aliased edge
# looking like an edge instead of a staircase.
OPAQUE_AT = float(INK_DELTA)


@dataclass
class PagePlan:
    """What a merge will do to one page, before it does it."""
    number: int                      # 1-based, as the user counts pages
    patches: List[imagediff.Region] = field(default_factory=list)
    conflicts: List[imagediff.Region] = field(default_factory=list)
    shift: Tuple[float, float] = (0.0, 0.0)
    note: str = ''

    @property
    def touched(self) -> bool:
        return bool(self.patches or self.conflicts)


@dataclass
class Plan:
    """What a merge will do, before it does it.

    Made first and shown before anything is written, because the user's section 7
    asks for a preview with the counts in it and because a merge that cannot
    be inspected before it happens is a merge nobody can trust.
    """
    pages: List[PagePlan]

    @property
    def patches(self) -> int:
        return sum(len(p.patches) for p in self.pages)

    @property
    def conflicts(self) -> int:
        return sum(len(p.conflicts) for p in self.pages)

    @property
    def pages_touched(self) -> int:
        return sum(1 for p in self.pages if p.touched)

    def summary(self) -> str:
        if not self.pages:
            return 'nothing to merge'
        said = '%d of %d pages change' % (self.pages_touched, len(self.pages))
        if self.patches:
            said += ', %d patch%s' % (self.patches,
                                      '' if self.patches == 1 else 'es')
        if self.conflicts:
            said += ', %d need a decision' % self.conflicts
        return said


def transparent(image, paper=None, opaque_at=OPAQUE_AT) -> np.ndarray:
    """The image as BGRA, with the paper made invisible.

    A patch has to sit over writing that is already there, so everything in
    it that is not new ink must let the page through. The alpha is the
    pixel's own distance from the paper, which makes a soft edge soft rather
    than cutting it at a threshold and leaving a staircase along every
    stroke.
    """
    if paper is None:
        paper = imagediff.paper_of(image)
    away = np.zeros(image.shape[:2], np.float32)
    for channel, value in zip(cv2.split(image), paper):
        away += cv2.absdiff(channel, np.full_like(channel, int(value)))
    alpha = np.clip(away / max(1.0, opaque_at), 0.0, 1.0)
    out = np.dstack([image, (alpha * 255).astype(np.uint8)])
    return out


def as_png(image) -> bytes:
    ok, buffer = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('the patch could not be encoded')
    return buffer.tobytes()


def plan(base_pages, target_pages, render, paper=None,
         progress: Callable = None) -> Plan:
    """What would change, page by page, without changing anything.

    `base_pages` and `target_pages` are whatever the caller has -- the two
    are only ever touched through `render(side, index)`, which must hand back
    one page as a BGR image at a size the two sides share. Pairing pages, page
    sizes and page order are the documents' business and are settled before
    this is called.
    """
    out = []
    pages = min(len(base_pages), len(target_pages))
    for index in range(pages):
        if progress is not None:
            progress(index + 1, pages)
        base = render('base', index)
        target = render('target', index)
        if base is None or target is None:
            out.append(PagePlan(index + 1, note='no page on one side'))
            continue
        found = imagediff.compare(base, target, paper=paper)
        out.append(PagePlan(
            number=index + 1,
            patches=found.of_kind('added'),
            conflicts=found.conflicts(),
            shift=found.shift,
            note=found.summary()))
    return Plan(out)


def apply(document, made: Plan, render, scale, paper=None,
          layer_name=MERGE_LAYER, progress: Callable = None) -> Dict:
    """Put the patches into `document`, and report what was left alone.

    `document` is the base `XoppDocument` and is edited in place: a new layer
    per page that has patches, and nothing else. No existing stroke, layer,
    image or page attribute is read, moved or rewritten -- the base is the
    structure and this only adds to it.

    `scale` is pixels per page unit for the renders, which is how a
    rectangle of pixels becomes a rectangle of page units. One number,
    because both sides are rendered through the same page box.

    Conflicts are **not** applied. They are counted and handed back, because
    an image cannot delete a stroke and pretending otherwise produces a
    document that looks right with the old ink still live underneath it.
    """
    pages = document.pages
    done = {'patches': 0, 'pages': 0, 'conflicts': 0, 'bytes': 0}
    for page_plan in made.pages:
        done['conflicts'] += len(page_plan.conflicts)
        if not page_plan.patches:
            continue
        index = page_plan.number - 1
        if index >= len(pages):
            continue
        if progress is not None:
            progress(page_plan.number, len(made.pages))
        target = render('target', index)
        if target is None:
            continue
        layer = pages[index].add_layer(layer_name)
        for region in page_plan.patches:
            x, y, w, h = region.box
            crop = region.target
            if crop is None:
                crop = target[y:y + h, x:x + w]
            png = as_png(transparent(crop, paper))
            pages[index].add_image(
                png, (x / scale, y / scale, (x + w) / scale, (y + h) / scale),
                layer)
            done['patches'] += 1
            done['bytes'] += len(png)
        done['pages'] += 1
    return done


def undo(document, layer_name=MERGE_LAYER) -> int:
    """Take a merge out again, by the name it wrote.

    The user's section 6 asks for the merge to be undoable. It is, exactly, because
    a merge only ever adds a layer: removing the layers it named leaves the
    base document byte for byte what it was.
    """
    gone = 0
    for page in document.pages:
        for layer in page.layers():
            if layer.get('name') == layer_name:
                page.element.remove(layer)
                gone += 1
    return gone
