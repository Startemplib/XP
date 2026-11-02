"""A `.xopp` too big to hold, opened a page at a time.

    "XoppDocument.load cannot open it: decompressed it is over 2 GB and
     expat raises OverflowError."

That is the symptom. Measured, the user's notebook is:

    495 MB gzipped, 3.004 GB of XML, 61 pages, 17,613,012 strokes
    the largest page alone has 1,694,566 of them

`ET.fromstring` hands the whole string to expat, which takes an `int`, so
anything past 2^31 bytes cannot be one string. But **streaming the parse is
not the fix**, and saying it is would be the easy lie: `iterparse` without
clearing still builds the whole tree, and a tree of 17.6 million elements is
somewhere around fifteen gigabytes. The string is the first wall, not the
wall.

So the document is never whole in memory. One pass decompresses it to a
scratch file and records where each `<page>` begins and ends; a page is read
back by seeking to it and parsing those bytes alone. Measured against the
two other ways of doing it:

                              open    peak RAM   resident   one page
    all of it in memory       42.9s     3.02 GB    495 MB     0.11s
    split and recompress      45.9s     0.74 GB    495 MB     0.12s
    scratch file and offsets  12.3s     0.05 GB      ~0     0.08-0.69s

The third is faster *and* lighter at every point, because nothing is ever
held that is proportional to the document. That is the acceptance
condition -- RAM must not grow with the size of the file -- and the first
two fail it however well they perform.

WHAT THIS IS NOT. It is not a second reader. Small files go through
`XoppDocument.load` exactly as they always have: it is faster for them, it
is the path every test exercises, and a document that fits should not pay
for one that does not.

THE SCRATCH FILE lives in the system temp directory and goes when the
document is closed. It is the decompressed XML, so it is about six times
the `.xopp` -- three gigabytes for the user's. That is the trade, stated rather
than discovered: three gigabytes of disk for fifty megabytes of RAM.
"""
import gzip
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

from xp.core.document.xopp_io import Page, XoppDocument

# Above this, `XoppDocument.load` is asked for a string expat may refuse and
# a tree that may not fit. The number is the compressed size, because that
# is what can be known without reading the file: the sample notebook is 495 MB
# and decompresses six-fold, and the trouble starts somewhere past 2 GB of
# XML. 200 MB is comfortably below any file that can reach it.
BIG_BYTES = 200 << 20

# How much *source* XML the parsed pages may hold between them. A page is
# worth roughly ten times its XML once it is elements, so this is already
# generous; the point is that it is a bound and not a page count, because
# the user's pages run from 70,496 strokes to 1,694,566.
CACHE_BYTES = 96 << 20

_PAGE_OPEN = b'<page '
_PAGE_CLOSE = b'</page>'
_CHUNK = 16 << 20


def looks_big(path) -> bool:
    """Whether this file should be opened a page at a time."""
    try:
        return os.path.getsize(path) >= BIG_BYTES
    except OSError:
        return False


def open_document(path):
    """`BigDocument` for a file that needs it, `XoppDocument` otherwise."""
    return BigDocument(path) if looks_big(path) else XoppDocument.load(path)


class _Pages:
    """The document's pages, parsed when they are asked for.

    A sequence rather than a list, because building the list is the thing
    that cannot be done. Everything the merge does with `document.pages` --
    index it, take its length, iterate it -- works on this unchanged.
    """

    def __init__(self, owner: 'BigDocument'):
        self._owner = owner

    def __len__(self):
        return len(self._owner.spans)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._owner.page(i)
                    for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        return self._owner.page(index)

    def __iter__(self):
        for index in range(len(self)):
            yield self._owner.page(index)

    def __repr__(self):
        return f'<{len(self)} pages, read on demand>'


class BigDocument:
    """A `.xopp` read a page at a time, with the same surface as a small one.

    `pages`, `root` and `save` are what the merge, the views and the export
    use, and they are all here. What is different is only that no page is
    parsed until something asks for it, and that the pages parsed so far are
    dropped when they add up to more than `CACHE_BYTES` of source.
    """

    def __init__(self, path, scratch_dir=None):
        self.path = Path(path)
        self._scratch = None
        self._file = None
        self.spans: List[Tuple[int, int]] = []
        self._header = b''
        self._footer = b''
        self._cache = {}
        self._order = []
        self._held = 0
        self._evicted = {}
        self._spill = tempfile.TemporaryFile()
        self._unpack(scratch_dir)
        self.root = ET.fromstring(self._header + b'</xournal>')
        self.tree = ET.ElementTree(self.root)
        self.pages = _Pages(self)
        # Set by whoever owns the document, and called on **every** parse -- see
        # `page()`. `None` means nothing has to be done to a page after it is
        # read, which is true for a notebook nobody strips the border from.
        self.cleaner = None

    # ---------------- opening ----------------

    def _unpack(self, scratch_dir=None) -> None:
        """One pass: the XML to a scratch file, and where every page is.

        The boundaries are found in the bytes rather than by parsing,
        because parsing is what cannot be afforded here and `<page ` and
        `</page>` cannot appear inside a `.xopp`'s text -- the coordinates
        and widths are numbers.
        """
        handle, self._scratch = tempfile.mkstemp(
            prefix='xp-big-', suffix='.xml', dir=scratch_dir)
        os.close(handle)
        tail, at, start, seen = b'', 0, None, 0
        with gzip.open(self.path, 'rb') as source, \
                open(self._scratch, 'wb') as out:
            while True:
                chunk = source.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                blob = tail + chunk
                base = seen - len(tail)
                index = 0
                while True:
                    if start is None:
                        found = blob.find(_PAGE_OPEN, index)
                        if found < 0:
                            break
                        if not self.spans and self._header == b'':
                            self._header = blob[:found]
                        start = base + found
                        index = found + 1
                    else:
                        found = blob.find(_PAGE_CLOSE, index)
                        if found < 0:
                            break
                        self.spans.append((start, base + found + len(_PAGE_CLOSE)))
                        start = None
                        index = found + len(_PAGE_CLOSE)
                tail = blob[-16:]
                seen += len(chunk)
        self._file = open(self._scratch, 'rb')
        if not self.spans:
            raise ValueError(f'no pages in {self.path}')

    # ---------------- reading ----------------

    def page(self, index: int) -> Page:
        held = self._cache.get(index)
        if held is not None:
            self._order.remove(index)
            self._order.append(index)
            return held
        if index in self._evicted:
            start, size = self._evicted[index]
            self._spill.seek(start)
            raw = self._spill.read(size)
        else:
            raw = self.page_source(index)
        page = Page(ET.fromstring(raw))
        # **Every load, not only the first.** The cache evicts pages and the
        # next load parses the scratch file again, so anything edited in memory
        # is gone unless it is applied here: the merger's printed-border
        # removal was a one-time pass over whatever happened to be cached, and
        # an evicted page came back with its border on. `cleaner` is set by the
        # owner of the document and must be a pure function of one page.
        cleaner = getattr(self, 'cleaner', None)
        if cleaner is not None:
            try:
                cleaner(index, page)
            except Exception as exc:
                print('[bigxopp] page %d could not be cleaned: %s' % (index, exc),
                      flush=True)
        self._keep(index, page, len(raw))
        return page

    def _keep(self, index: int, page: Page, size: int) -> None:
        self._cache[index] = page
        self._order.append(index)
        self._held += size
        self._sizes = getattr(self, '_sizes', {})
        self._sizes[index] = size
        while self._held > CACHE_BYTES and len(self._order) > 1:
            oldest = self._order[0]
            # Preserve mutable pages on disk before releasing their XML trees.
            old_page = self._cache[oldest]
            data = ET.tostring(old_page.element, encoding='utf-8')
            self._spill.seek(0, 2)
            start = self._spill.tell()
            self._spill.write(data)
            self._evicted[oldest] = (start, len(data))
            self._order.pop(0)
            self._held -= self._sizes.pop(oldest, 0)
            self._cache.pop(oldest)

    def page_source(self, index: int) -> bytes:
        """One page's own XML, straight off the scratch file.

        What `save` writes, so that a page nothing touched goes out byte for
        byte as it came in and is never parsed at all.
        """
        start, end = self.spans[index]
        self._file.seek(start)
        return self._file.read(end - start)

    def page_box(self, index: int) -> Tuple[float, float]:
        """A page's size without parsing its strokes.

        Opening a merge needs every page's size and none of its ink, and a
        page of the user's is up to 223 MB of XML. This reads the attributes off
        the opening tag and stops.
        """
        start, _end = self.spans[index]
        self._file.seek(start)
        head = self._file.read(512)
        cut = head.find(b'>')
        if cut >= 0:
            head = head[:cut + 1]
        try:
            element = ET.fromstring(head.decode('utf-8', 'replace') + '</page>')
        except ET.ParseError:
            return (595.275591, 841.889764)
        return (float(element.get('width', 595.275591)),
                float(element.get('height', 841.889764)))

    # ---------------- writing ----------------

    def save(self, path, compresslevel: int = 1) -> None:
        """Write it out again without ever holding it whole.

        Page by page: a page that has been parsed is serialised from what it
        now is, and a page that was never touched is copied straight off the
        scratch file. That is what keeps saving a 3 GB document from costing
        3 GB of memory, and it is also why an untouched page comes out
        exactly as it went in.

        `compresslevel=1`, the same as `XoppDocument.save` and for the same
        reason, which was written down there and which I had to measure
        again because I used the module default instead of reading it: on
        this document level 9 is 203 seconds. Xournal++ re-saves at its own
        level the next time the user saves from inside it, so there is nothing to
        be had by spending that here.
        """
        header = ET.tostring(self.root, encoding='unicode')
        cut = header.rfind('</xournal>')
        if cut >= 0:
            header = header[:cut]
        from xp.core.document.atomic import atomic_gzip
        with atomic_gzip(path, compresslevel) as out:
            out.write(b"<?xml version='1.0' encoding='UTF-8'?>\n")
            out.write(header.encode('utf-8'))
            for index in range(len(self.spans)):
                page = self._cache.get(index)
                if page is not None:
                    out.write(ET.tostring(page.element, encoding='utf-8'))
                elif index in self._evicted:
                    start, size = self._evicted[index]
                    self._spill.seek(start)
                    out.write(self._spill.read(size))
                else:
                    out.write(self.page_source(index))
                out.write(b'\n')
            out.write(b'</xournal>\n')

    def repair(self) -> int:
        """Nothing to do here, and saying so beats pretending.

        `XoppDocument.repair` walks every stroke to find the one-point ones
        Xournal++ refuses a document over. Walking 17.6 million of them would
        cost what this class exists to avoid, and the strokes in a file XP
        wrote have already been through it once.
        """
        return 0

    # ---------------- housekeeping ----------------

    def close(self) -> None:
        spill = getattr(self, "_spill", None)
        if spill is not None:
            spill.close()
            self._spill = None
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._scratch and os.path.exists(self._scratch):
            try:
                os.remove(self._scratch)
            except OSError:
                pass
        self._scratch = None
        self._cache.clear()
        self._order.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return (f'<BigDocument {self.path.name} {len(self.spans)} pages, '
                f'read on demand>')
