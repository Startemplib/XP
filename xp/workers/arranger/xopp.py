"""A `.xopp` read page by page, with no XP in it.

The file is gzip + XML: one `<xournal>`, then `<page>`s in order. What the
arranger needs from a document is small -- how many pages, how big each one
is, its XML, and a picture of it -- and none of that needs the whole
document in memory.

So opening one does what XP's `bigxopp` does, for the same reason: decompress
once to a scratch file while recording where each `<page ` ... `</page>`
begins and ends. A page is then read by seeking to it; a page is never parsed
to find out how many pages there are, and a 495 MB notebook costs the same
decompression it costs XP (about nine seconds) and nothing per page.

The header before the first page (the XML declaration, `<xournal creator=...
fileversion=...>` and `<title>`) is kept as bytes: the writer puts pages under
the first document's own header, so the output says what version it is
instead of inventing one.
"""

from __future__ import annotations

import gzip
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

CHUNK = 8 << 20
_PAGE_OPEN = b'<page '
_PAGE_CLOSE = b'</page>'
_CLOSE_LEN = len(_PAGE_CLOSE)


class Notebook:
    """One `.xopp`, its pages readable one at a time."""

    kind = 'xopp'

    def __init__(self, path, scratch_dir=None):
        self.path = Path(path)
        self._scratch = None
        self._file = None
        self._spans: List[Tuple[int, int]] = []
        self.header = b''
        self._unpack(scratch_dir)
        if not self._spans:
            self.close()
            raise ValueError(f'no <page> in {self.path}')

    # ---------------- reading ----------------

    def _unpack(self, scratch_dir=None) -> None:
        handle, self._scratch = tempfile.mkstemp(
            prefix='arranger-', suffix='.xml', dir=scratch_dir)
        os.close(handle)
        tail, seen, start = b'', 0, None
        with gzip.open(self.path, 'rb') as source, open(self._scratch, 'wb') as out:
            while True:
                chunk = source.read(CHUNK)
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
                        if not self._spans and not self.header:
                            self.header = blob[:found]
                        start = base + found
                        index = found + 1
                    else:
                        found = blob.find(_PAGE_CLOSE, index)
                        if found < 0:
                            break
                        self._spans.append((start, base + found + _CLOSE_LEN))
                        start = None
                        index = found + _CLOSE_LEN
                tail = blob[-16:]
                seen += len(chunk)
        self._file = open(self._scratch, 'rb')

    @property
    def pages(self) -> int:
        return len(self._spans)

    def page_xml(self, index: int) -> bytes:
        """One page's own XML, exactly as it is in the file."""
        start, end = self._spans[index]
        self._file.seek(start)
        return self._file.read(end - start)

    def page_element(self, index: int) -> ET.Element:
        return ET.fromstring(self.page_xml(index))

    def page_box(self, index: int) -> Tuple[float, float]:
        """Width and height without parsing the page.

        Reading the opening tag and stopping is what keeps a 223 MB page from
        being parsed just to draw a 136 px card.
        """
        start, _end = self._spans[index]
        self._file.seek(start)
        head = self._file.read(1024)
        cut = head.find(b'>')
        if cut >= 0:
            head = head[:cut + 1]
        try:
            element = ET.fromstring(head.decode('utf-8', 'replace') + '</page>')
        except ET.ParseError:
            return (595.275591, 841.889764)
        return (float(element.get('width', 595.275591)),
                float(element.get('height', 841.889764)))

    def stroke_count(self, index: int) -> int:
        """`<stroke` occurrences in the page's bytes -- no tree, no parse."""
        return self.page_xml(index).count(b'<stroke')

    def render(self, index: int, scale: float):
        from .thumb import render_page
        return render_page(self.page_element(index), scale)

    # ---------------- housekeeping ----------------

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._scratch and os.path.exists(self._scratch):
            try:
                os.remove(self._scratch)
            except OSError:
                pass
        self._scratch = None

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
        return f'<Notebook {self.path.name} {self.pages} pages>'
