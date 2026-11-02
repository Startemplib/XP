"""What the merged page is made of, kept apart from how it is drawn.

Three things used to be one thing, and that was the bug: which document a
stroke came from, whether it survives the merge, and where it sits in the
drawing order. Left was the document being edited and right was pulled into
it, so "this is right's ink" and "this ink is on top" were the same fact --
and brushing Left over a place where both sides had ink left right's ink
sitting visibly above the left ink it was supposed to give way to.

Here they are separate:

- **Identity** is a `Ref`: which side a stroke came from and its index in
  that side's page. It never changes and nothing is copied to express it.
- **A decision** is what the brush records: for a region, which sides keep
  their ink there. Left, Right, Both and Neither are four values of one
  thing, not layers stacked on each other, so switching between them cannot
  leave a trace of the one before.
- **Drawing order** is a presentation detail, fixed and identical for both
  sides, and carries no meaning at all.

Neither source document is modified. The merged document is built from the
decisions when it is exported, and the same function builds what the Final
Preview shows -- so what you see there is what is written.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

LEFT = 'left'
RIGHT = 'right'
SIDES = (LEFT, RIGHT)

# The four decisions a brush can record about a region.
TAKE_LEFT = 'L'
TAKE_RIGHT = 'R'
TAKE_BOTH = 'B'
TAKE_NONE = 'N'
DECISIONS = (TAKE_LEFT, TAKE_RIGHT, TAKE_BOTH, TAKE_NONE)

# What each decision means for each side. This table is the whole rule.
KEEPS = {
    TAKE_LEFT: {LEFT: True, RIGHT: False},
    TAKE_RIGHT: {LEFT: False, RIGHT: True},
    TAKE_BOTH: {LEFT: True, RIGHT: True},
    TAKE_NONE: {LEFT: False, RIGHT: False},
}


class Ref(NamedTuple):
    """Which stroke, on which side. Identity, not position."""

    side: str
    index: int


@dataclass
class PageDecisions:
    """Every decision recorded about one page of the merge.

    `kept` holds only what has actually been decided; anything absent is
    undecided and falls back to `default_keeps`. Undecided left ink stays
    (your notebook is the base) and undecided right ink does not, so doing
    nothing at all leaves the left document as it was.
    """

    kept: Dict[Ref, bool]
    split: bool = False
    # Bumped by every change. A view that costs a pass over the page can
    # cache itself against this and know, without looking, whether anything
    # it drew has been decided differently since. Zooming and panning do not
    # touch it, which is the whole point: they are what the user does most.
    stamp: int = 0
    # Where the brush has taken from a side that has no strokes to take --
    # a PDF, which is ink until something converts it. One accumulated mask
    # per side, in the merged page's own pixels, exactly the space the brush
    # already works in. Nothing new: it is the brush's own mask, kept
    # instead of being thrown away at the end of the gesture, because on
    # that side there was no stroke to write the decision onto.
    taken: Dict[str, object] = field(default_factory=dict)
    # And where it decided *against* a side, which is the other half of the
    # same decision and was missing. `taken` alone could not express "take
    # neither": nothing was ever recorded, so a PDF stayed on screen after the user
    # had refused it. Filled on every gesture that involves a picture side,
    # for every side, because which side gives way is the whole question.
    # Only ever allocated for merges that have a picture side: a stroke-only
    # merge records nothing here and behaves exactly as it did.
    cleared: Dict[str, object] = field(default_factory=dict)

    def _areas(self, which: str) -> Dict[str, object]:
        return self.taken if which == 'taken' else self.cleared

    def owner(self, shape):
        """One number per pixel: which layer is on top here.

            0  undecided -- draw every side as it stands (black/red/blue)
            1  the left side won this pixel
            2  the right side won this pixel
            3  neither -- the user refused both, so nothing is drawn

        This is the layer relation the review asked for, expressed as one plane
        instead of stacking another mask beside `taken` and `cleared`: those two
        already say everything needed, and this is the same information read as
        "who is on top" rather than "who was taken".

        Both taken at once is Take Both, which keeps both sides -- that is the
        undecided rendering, so it stays 0 rather than becoming a fifth value.
        """
        import numpy as np

        out = np.zeros(shape, np.uint8)
        left_t, right_t = self.taken.get(LEFT), self.taken.get(RIGHT)
        left_c, right_c = self.cleared.get(LEFT), self.cleared.get(RIGHT)
        both = (left_t & right_t) if (left_t is not None and right_t is not None) else None

        for taken, cleared, value in ((left_t, left_c, 1), (right_t, right_c, 2)):
            if taken is None:
                continue
            won = taken
            if cleared is not None:
                won = won & ~cleared
            if both is not None:
                won = won & ~both
            out[won] = value

        if both is not None:
            # Take Both: the user confirmed both sides here. Without this the plane
            # could not tell "I kept both" from "I have not decided", which made
            # the gesture a no-op on a picture side -- the one case where a
            # decision produced no visible change at all.
            out[both] = 4

        if left_c is not None and right_c is not None:
            out[left_c & right_c] = 3
        return out

    def take_area(self, side: str, mask_array, which: str = 'taken'):
        """Add this gesture's pixels to what was taken (or refused) on `side`.

        Returns the indices it actually added, which is what undo needs and
        is a few thousand integers rather than another whole page of bools.
        """
        import numpy as np

        store = self._areas(which)
        held = store.get(side)
        if held is None:
            held = np.zeros(mask_array.shape, bool)
            store[side] = held
        fresh = mask_array & ~held
        held |= fresh
        if fresh.any():
            self.stamp += 1
        return np.flatnonzero(fresh.ravel())

    def remove_area(self, side: str, mask_array, which: str = 'taken'):
        """Take those pixels **out** of what was taken (or refused) on `side`.

        Returns the indices it actually removed, so undo can put them back. Needed
        because the two masks are otherwise additive: a refusal recorded before a
        decision would outlive it, since `owner()` applies the refused pixels last.
        """
        import numpy as np

        held = self._areas(which).get(side)
        if held is None:
            return []
        wanted = np.asarray(mask_array, dtype=bool).ravel()
        flat = held.ravel()
        fresh = np.flatnonzero(wanted & flat)
        if len(fresh):
            flat[fresh] = False
            self.stamp += 1
        return fresh

    def give_back(self, side: str, flat_indices, which: str = 'taken') -> None:
        """Undo: take those pixels out again."""
        held = self._areas(which).get(side)
        if held is None or len(flat_indices) == 0:
            return
        flat = held.ravel()
        flat[flat_indices] = False
        self.stamp += 1

    def take_back(self, side: str, flat_indices, shape, which: str = 'taken') -> None:
        """Redo."""
        import numpy as np

        store = self._areas(which)
        held = store.get(side)
        if held is None:
            held = np.zeros(shape, bool)
            store[side] = held
        held.ravel()[flat_indices] = True
        self.stamp += 1

    def area_taken(self, side: str):
        return self.taken.get(side)

    def area_cleared(self, side: str):
        return self.cleared.get(side)

    @staticmethod
    def default_keeps(ref: Ref, base: str = LEFT) -> bool:
        """Whether an undecided stroke survives: the base side's ink does.

        `base` was the constant LEFT, on the assumption this class's own
        docstring states -- "your notebook is the base". That is false the
        moment the left side is a PDF: a picture side contributes no strokes,
        so "keep the left side" kept nothing at all, and the notebook on the
        right was dropped in full. Measured on the sample pair -- a PDF against
        a 66,147-stroke notebook exported **an empty document**, on every
        page the user had not visited, and said nothing about it.

        So the caller names the base, and `MergeSession` answers with the
        side that actually holds strokes.
        """
        return ref.side == base

    def keeps(self, ref: Ref, base: str = LEFT) -> bool:
        decided = self.kept.get(ref)
        return self.default_keeps(ref, base) if decided is None else decided

    def is_decided(self, ref: Ref) -> bool:
        return ref in self.kept

    def set(self, ref: Ref, keep: Optional[bool]) -> bool:
        """Record (or clear, with None) one decision. True if it changed."""
        if keep is None:
            if self.kept.pop(ref, None) is None:
                return False
            self.stamp += 1
            return True
        if self.kept.get(ref) is keep:
            return False
        self.kept[ref] = keep
        self.stamp += 1
        return True


def apply_decision(page: PageDecisions, decision: str, refs) -> List[Tuple[Ref, Optional[bool]]]:
    """Apply one decision to the strokes a brush covered.

    Returns the previous value of every entry that changed, which is what an
    undo step is made of. Each ref is driven to the value the decision means
    for its side, so the state after depends only on the decision -- never on
    which decision came before it.
    """
    keeps = KEEPS[decision]
    changes: List[Tuple[Ref, Optional[bool]]] = []
    for ref in refs:
        before = page.kept.get(ref)
        if page.set(ref, keeps[ref.side]):
            changes.append((ref, before))
    return changes


def revert(page: PageDecisions, changes) -> List[Ref]:
    """Put back the values `changes` recorded; returns the refs touched."""
    touched = []
    for ref, before in changes:
        page.set(ref, before)
        touched.append(ref)
    return touched
