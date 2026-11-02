"""The page order, and the edits to it, with no window and no file writing.

This is the Arranger's data: which document each page came from, the order
they are in, what is selected, and one undo history for all of it. It knows
nothing about Tk, about thumbnails, or about writing a `.xopp` -- the window
(`arrangergui.py`) is a view over this, and a later phase is what will turn
the order into a file.

Two rules shape the whole module.

**A page is a reference, not a payload.** `PageRef` is `(uid, source,
index)` and nothing else: moving or copying a page moves or copies that
triple, so no stroke is parsed, no XML is rewritten and the original
document is never touched. That is what makes page arrangement a document
*structure* operation rather than a conversion -- and it is why the same
page can appear twice in the order (a duplicate) without either copy being
a rasterization of the other.

**Identity is the uid, never the position.** Positions change on every
drag, and a selection, an undo step or a "move these four" command that
remembered positions would move the wrong pages the moment the order
changed. Every operation takes uids and finds positions itself.

Undo is a snapshot of the order plus the selection. It is deliberately not
an inverse-operation log: the operations here are cheap (a list of a few
hundred small tuples), and a snapshot cannot get an inverse subtly wrong
when a move lands in the gap it came from.
"""

from __future__ import annotations

import itertools
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

# What a page's source answers. Kept as a plain class rather than a protocol
# so a test can hand in a stub with the same three methods and no Tk, no
# document and no disk.

DEFAULT_LABEL_WIDTH = 18


@dataclass(frozen=True)
class PageRef:
    """One page in the arrangement: where it comes from, and which one.

    `uid` is the identity; `source` is the uid of the document it was taken
    from, and `index` is the page inside that document. Two refs with the
    same (source, index) are the same page shown twice -- a duplicate -- and
    that is legal and expected.
    """

    uid: int
    source: int
    index: int


class Source:
    """One imported document, as pages to look at.

    `side` is a `sides.Side` (so XOPP and PDF arrive the same way, and a big
    `.xopp` arrives lazily). Nothing here re-opens the file or re-parses a
    page: the source is a name, a number of pages and a way to render one.
    """

    def __init__(self, uid: int, path, side, label: str):
        self.uid = uid
        self.path = Path(path) if path is not None else None
        self.side = side
        self.label = label

    @property
    def pages(self) -> int:
        return int(self.side.pages)

    @property
    def kind(self) -> str:
        return self.side.kind

    def page_box(self, index: int) -> Tuple[float, float]:
        return self.side.page_box(index)

    def render(self, index: int, scale: float):
        """One page as a BGR array, at `scale` pixels per page unit."""
        return self.side.render(index, scale)

    def strokes(self, index: int):
        """The page's native strokes, or None for a PDF (see sides.Side)."""
        return self.side.strokes(index)

    def close(self) -> None:
        close = getattr(self.side, 'close', None)
        if close is not None:
            close()

    def __repr__(self):
        return f'<Source {self.label!r} {self.kind} {self.pages} pages>'


class Arrangement:
    """The pages, the order, the selection, and the undo history."""

    def __init__(self):
        self._sources: Dict[int, Source] = {}
        self._pages: List[PageRef] = []
        self._selection: Set[int] = set()
        self._next_source = itertools.count(1)
        self._next_page = itertools.count(1)
        self._undo: List[Tuple[Tuple[PageRef, ...], frozenset]] = []
        self._redo: List[Tuple[Tuple[PageRef, ...], frozenset]] = []
        self.clipboard: List[Tuple[int, int]] = []

    # ---------------- reading ----------------

    def __len__(self) -> int:
        return len(self._pages)

    def __iter__(self) -> Iterator[PageRef]:
        return iter(self._pages)

    def __getitem__(self, position: int) -> PageRef:
        return self._pages[position]

    @property
    def sources(self) -> List[Source]:
        return list(self._sources.values())

    def source(self, uid: int) -> Source:
        return self._sources[uid]

    @property
    def pages(self) -> List[PageRef]:
        """The order, as a copy -- so a view cannot reorder by accident."""
        return list(self._pages)

    def position(self, uid: int) -> Optional[int]:
        for index, ref in enumerate(self._pages):
            if ref.uid == uid:
                return index
        return None

    def positions_of(self, uids: Iterable[int]) -> List[int]:
        """Where those pages sit now, ascending -- a test's or a view's map."""
        wanted = set(uids)
        return [index for index, ref in enumerate(self._pages)
                if ref.uid in wanted]

    def ref(self, uid: int) -> Optional[PageRef]:
        index = self.position(uid)
        return None if index is None else self._pages[index]

    @property
    def selection(self) -> Set[int]:
        return set(self._selection)

    def selected(self) -> List[PageRef]:
        return [ref for ref in self._pages if ref.uid in self._selection]

    def selected_positions(self) -> List[int]:
        return [index for index, ref in enumerate(self._pages)
                if ref.uid in self._selection]

    def is_selected(self, uid: int) -> bool:
        return uid in self._selection

    def describe(self, ref: PageRef) -> str:
        source = self._sources[ref.source]
        return f'{source.label} p{ref.index + 1}'

    def order(self) -> List[Tuple[str, int]]:
        """`(source label, 0-based page index)` in the order on screen.

        What a writer will consume (by source path, in this order); a list of
        small tuples so a test can assert a reorder without a document.
        """
        return [(self._sources[ref.source].label, ref.index)
                for ref in self._pages]

    def pages_of(self, source_uid: int) -> int:
        return sum(1 for ref in self._pages if ref.source == source_uid)

    # ---------------- history ----------------

    @contextmanager
    def _edit(self):
        """One user action, one undo step -- even if it changed many pages."""
        before = (tuple(self._pages), frozenset(self._selection))
        yield
        after = (tuple(self._pages), frozenset(self._selection))
        if before[0] != after[0] or before[1] != after[1]:
            self._undo.append(before)
            self._redo.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        pages, selection = self._undo.pop()
        self._redo.append((tuple(self._pages), frozenset(self._selection)))
        self._pages = list(pages)
        self._selection = set(selection)
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        pages, selection = self._redo.pop()
        self._undo.append((tuple(self._pages), frozenset(self._selection)))
        self._pages = list(pages)
        self._selection = set(selection)
        return True

    # ---------------- importing ----------------

    def add_source(self, side, path, label: Optional[str] = None) -> Source:
        """Import a document's pages at the end of the order.

        `side` is anything with `pages`, `page_box`, `render` and `strokes`
        (a `sides.Side` in the window). The pages are added as references;
        no page is parsed here, which is why importing a 495 MB notebook
        costs what opening it costs and nothing per page.
        """
        label = label or self._label_for(path)
        source = Source(next(self._next_source), path, side, label)
        with self._edit():
            self._sources[source.uid] = source
            for index in range(source.pages):
                self._pages.append(PageRef(next(self._next_page), source.uid, index))
            self._selection = set()
        return source

    def _label_for(self, path) -> str:
        stem = Path(path).stem if path is not None else 'document'
        taken = {source.label for source in self._sources.values()}
        if stem not in taken:
            return stem
        for n in itertools.count(2):
            candidate = f'{stem} ({n})'
            if candidate not in taken:
                return candidate
        raise AssertionError('unreachable')

    # ---------------- editing ----------------

    def _resolve(self, uids: Iterable[int]) -> List[PageRef]:
        wanted = set(uids)
        return [ref for ref in self._pages if ref.uid in wanted]

    def move(self, uids: Iterable[int], before: Optional[int] = None) -> bool:
        """Move the pages `uids` together, to just before page `before`.

        `before=None` means the end. Moving a block to where it already is
        is not an edit and does not consume an undo step.
        """
        moving = self._resolve(uids)
        if not moving:
            return False
        moving_ids = {ref.uid for ref in moving}
        if before is not None and before in moving_ids:
            return False
        rest = [ref for ref in self._pages if ref.uid not in moving_ids]
        if before is None:
            at = len(rest)
        else:
            at = next((index for index, ref in enumerate(rest) if ref.uid == before), len(rest))
        reordered = rest[:at] + moving + rest[at:]
        if reordered == self._pages:
            return False                     # already in that gap, in that order
        with self._edit():
            self._pages = reordered
        return True

    def move_units(self, uids: Iterable[int], slot: int) -> bool:
        """Move the pages `uids` as one ordered whole to `slot`.

        `slot` counts pages that are **not** in the selection, which is what
        makes a full selection move predictable: the selection keeps its
        internal order, every other page keeps its relative order, and the
        same slot always means the same gap however scattered the selection
        is. One undo step.
        """
        moving = self._resolve(uids)
        if not moving:
            return False
        ids = {ref.uid for ref in moving}
        rest = [ref for ref in self._pages if ref.uid not in ids]
        slot = max(0, min(int(slot), len(rest)))
        reordered = rest[:slot] + moving + rest[slot:]
        if reordered == self._pages:
            return False
        with self._edit():
            self._pages = reordered
        return True

    def slot_for_index(self, uids: Iterable[int], index: int) -> int:
        """The `move_units` slot for a drop at position `index` in the grid.

        A drop index counts every page; a slot counts only the pages that are
        staying, so the selected pages under the pointer do not count twice.
        """
        ids = set(uids)
        index = max(0, min(int(index), len(self._pages)))
        return sum(1 for ref in self._pages[:index] if ref.uid not in ids)

    def remove(self, uids: Iterable[int]) -> bool:
        """Delete the pages. Only the references go; the sources stay open."""
        gone = {ref.uid for ref in self._resolve(uids)}
        if not gone:
            return False
        with self._edit():
            self._pages = [ref for ref in self._pages if ref.uid not in gone]
            self._selection -= gone
        return True

    def duplicate(self, uids: Iterable[int]) -> List[int]:
        """Copy the selection in place, and select what was made.

        A contiguous run (a Shift range) is duplicated as **one block**, right
        after the last page of the run. A scattered selection (Ctrl picks) is
        duplicated **one by one**: each copy lands immediately after the page
        it copies, so the pairs stay together. One undo step either way.
        """
        chosen = self._resolve(uids)
        if not chosen:
            return []
        positions = [self.position(ref.uid) for ref in chosen]
        contiguous = positions == list(range(positions[0],
                                             positions[0] + len(positions)))
        with self._edit():
            if contiguous:
                copies = [PageRef(next(self._next_page), ref.source, ref.index)
                          for ref in chosen]
                last = positions[-1]
                self._pages[last + 1:last + 1] = copies
            else:
                ids = {ref.uid for ref in chosen}
                copied: List[PageRef] = []
                rebuilt: List[PageRef] = []
                for ref in self._pages:
                    rebuilt.append(ref)
                    if ref.uid in ids:
                        copy = PageRef(next(self._next_page),
                                       ref.source, ref.index)
                        rebuilt.append(copy)
                        copied.append(copy)
                self._pages = rebuilt
                copies = copied
            # Keep the source pages selected too: after a duplicate the
            # originals and their copies are all in the selection.
            self._selection = ({ref.uid for ref in chosen}
                               | {ref.uid for ref in copies})
        return [ref.uid for ref in copies]

    def copy(self, uids: Iterable[int]) -> int:
        """Put the pages on the clipboard, as (source, index) pairs.

        Deliberately content, not uids: pasting into the same arrangement or
        another one later means "this page of this document", and a uid is
        only meaningful inside the arrangement that minted it.
        """
        chosen = self._resolve(uids)
        self.clipboard = [(ref.source, ref.index) for ref in chosen]
        return len(self.clipboard)

    def cut(self, uids: Iterable[int]) -> int:
        """Copy the pages and take them out, as one undo step.

        One step because that is what a cut is: Ctrl+Z after Ctrl+X puts the
        pages back, not half of them. The clipboard is not part of the
        snapshot -- undo restores the pages and leaves what was copied on it,
        which is how every editor behaves and what makes "cut, undo, paste"
        useful.
        """
        chosen = self._resolve(uids)
        if not chosen:
            return 0
        self.clipboard = [(ref.source, ref.index) for ref in chosen]
        gone = {ref.uid for ref in chosen}
        with self._edit():
            self._pages = [ref for ref in self._pages if ref.uid not in gone]
            self._selection -= gone
        return len(chosen)

    def paste(self, before: Optional[int] = None) -> List[int]:
        """Paste the clipboard, after the selection (or at `before`)."""
        if not self.clipboard:
            return []
        at = self._paste_position(before)
        with self._edit():
            made = [PageRef(next(self._next_page), source, index)
                    for source, index in self.clipboard]
            self._pages[at:at] = made
            self._selection = {ref.uid for ref in made}
        return [ref.uid for ref in made]

    def _paste_position(self, before: Optional[int]) -> int:
        if before is not None:
            found = self.position(before)
            return len(self._pages) if found is None else found
        if self._selection:
            return max(self.selected_positions()) + 1
        return len(self._pages)

    # ---------------- selection ----------------

    def select_only(self, uid: int) -> None:
        if self.position(uid) is None:
            return
        self._selection = {uid}

    def toggle(self, uid: int) -> None:
        if self.position(uid) is None:
            return
        if uid in self._selection:
            self._selection.discard(uid)
        else:
            self._selection.add(uid)

    def select_range(self, first: int, last: int) -> None:
        """Select every page between two, in the order shown, inclusive."""
        a, b = self.position(first), self.position(last)
        if a is None or b is None:
            return
        lo, hi = min(a, b), max(a, b)
        self._selection = {ref.uid for ref in self._pages[lo:hi + 1]}

    def select_all(self) -> None:
        self._selection = {ref.uid for ref in self._pages}

    def clear_selection(self) -> None:
        self._selection = set()

    def set_selection(self, uids: Iterable[int]) -> None:
        wanted = set(uids)
        self._selection = {ref.uid for ref in self._pages if ref.uid in wanted}

    # ---------------- housekeeping ----------------

    def close(self) -> None:
        for source in self._sources.values():
            source.close()

    def summary(self) -> str:
        documents = len(self._sources)
        return (f'{documents} document{"s" if documents != 1 else ""}, '
                f'{len(self._pages)} page{"s" if len(self._pages) != 1 else ""}')
