"""Turning the page order into one `.xopp` file.

    order: (document, page) references  ->  a new document, pages in that order

This is the only place in the package that writes anything, and it writes
exactly one file: the one the person named. Sources are read, never touched.

**A page is copied as bytes, not rebuilt.** A page is `<page width height>`
with its `<background>`, its `<layer>`s (names included) and its `<stroke>`
elements; `Notebook.page_xml` hands those bytes over without parsing the
page. Rebuilding the page from points would mean re-quoting coordinates and
re-deriving widths -- the one way to change ink nobody asked to change -- so
it is not done. The same page twice in the order is the same bytes twice,
which is exactly what a duplicate is.

**The header comes from the first document.** Its XML declaration, `<xournal
creator=... fileversion=...>` and `<title>` are the file's, not a page's;
taking them from a source keeps the output a version Xournal++ recognises
instead of inventing one. The title is replaced, because an arranged document
is not any one of its sources.

Two limits worth stating rather than discovering:

* **A PDF background is made openable.** Xournal++ writes `filename` only on
  the first page of a document and leaves later pages with `<background
  type="pdf" pageno="N"/>`; such a page cannot be first in an arranged file,
  so the document's own filename is copied onto it. A filename that does not
  exist is then pointed at a same-named PDF beside the source notebook or the
  output. Nothing else about a page changes, and a page whose path already
  resolves is copied byte for byte.
* **Pages are not repaired on the way out.** A stroke with fewer than two
  points makes Xournal++ refuse a whole document; this writer copies pages as
  they are, so a source that was never repaired keeps its defect. The test
  that matters here (does Xournal++ open the result) is in
  `tests/test_writer.py`.
"""

from __future__ import annotations

import gzip
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

TITLE = 'XP arranged document'


class Refused(Exception):
    """The writer will not produce this file, and says why."""


@dataclass
class Written:
    path: Path
    pages: int
    documents: int
    bytes: int
    missing_backgrounds: List[str] = field(default_factory=list)


def missing_backgrounds(page_xml: bytes, known=None) -> List[str]:
    """PDF backgrounds this page names that are not on disk.

    Xournal++ refuses the **whole document** when a `<background
    type="pdf">` points at a file it cannot find, so a page whose source was
    already broken makes the arranged file unopenable for that same reason.
    The writer cannot fix it -- the PDF is not in the `.xopp` -- but it can
    say so at save time instead of letting Xournal++ say it later.

    Only the `<background>` tag is parsed; scanning for the tag in the page's
    bytes keeps this off the cost of a 1.7-million-stroke page.
    """
    known = {} if known is None else known
    at = page_xml.find(b'<background')
    if at < 0:
        return []
    end = page_xml.find(b'>', at)
    if end < 0:
        return []
    tag = page_xml[at:end + 1]
    if tag.endswith(b'/>'):
        tag = tag[:-2] + b'>'          # a self-closing tag needs an end tag
    try:
        element = ET.fromstring(tag + b'</background>')
    except ET.ParseError:
        return []
    if element.get('type') != 'pdf':
        return []
    if (element.get('domain') or 'absolute') == 'attach':
        return []                      # attached to the file; not a path
    name = element.get('filename') or ''
    if not name:
        return []
    if name not in known:
        known[name] = Path(name).is_file()
    return [] if known[name] else [name]


def _header(source) -> bytes:
    header = getattr(source.side, 'header', b'')
    if header:
        return header
    # A notebook with no header at all is not something this writer has seen;
    # the smallest thing Xournal++ accepts is better than writing nothing.
    return (b"<?xml version='1.0' encoding='UTF-8'?>\n"
            b'<xournal creator="XP" fileversion="4"><title>' + TITLE.encode() + b'</title>')


_BACKGROUND = re.compile(rb'<background\b[^>]*>')
_FILENAME = re.compile(rb'filename="([^"]*)"')
_DOMAIN = re.compile(rb'domain="([^"]*)"')


def _background_tag(page_xml: bytes):
    match = _BACKGROUND.search(page_xml)
    if match is None:
        return None
    return match.group(0)


def _pdf_template(side):
    """The first PDF filename a document names, for pages that omit it.

    Xournal++ writes `filename` on the first page and leaves later pages with
    `<background type="pdf" pageno="N"/>`; a page like that cannot be the
    first page of an arranged file, because Xournal++ then refuses the whole
    document with "PDF Filename missing!". Copying the document's own
    filename onto those pages keeps the bytes as close as the format allows
    and makes the output openable.
    """
    for index in range(side.pages):
        tag = _background_tag(side.page_xml(index))
        if tag is None or b'type="pdf"' not in tag:
            continue
        name = _FILENAME.search(tag)
        if name is not None and name.group(1):
            return name.group(1), (
                _DOMAIN.search(tag).group(1)
                if _DOMAIN.search(tag) is not None else b'absolute')
    return None


def _repair_background(page_xml: bytes, template):
    """Give a filename-less PDF background the document's filename, or solid."""
    tag = _background_tag(page_xml)
    if tag is None or b'type="pdf"' not in tag:
        return page_xml
    name = _FILENAME.search(tag)
    if name is not None and name.group(1):
        return page_xml
    at = page_xml.find(tag)
    if template is None:
        solid = b'<background type="solid" color="#ffffff"/>'
        return page_xml[:at] + solid + page_xml[at + len(tag):]
    filename, domain = template
    extra = b' filename="' + filename + b'"'
    if _DOMAIN.search(tag) is None:
        extra = b' domain="' + domain + b'"' + extra
    if tag.endswith(b'/>'):
        fixed = tag[:-2] + extra + b'/>'
    else:
        fixed = tag[:-1] + extra + b'>'
    return page_xml[:at] + fixed + page_xml[at + len(tag):]


def _rewrite_background_path(page_xml: bytes, source_folder, target_folder,
                             cache=None):
    """Point a missing PDF background at a same-named file we can see.

    The original path is often from the machine the notebook was written on.
    If a file with the same name sits beside the source notebook (or beside
    the output), Xournal++ should open it instead of asking for a file that
    is not there. Nothing is changed when the path already exists.
    """
    tag = _background_tag(page_xml)
    if tag is None or b'type="pdf"' not in tag:
        return page_xml
    match = _FILENAME.search(tag)
    if match is None or not match.group(1):
        return page_xml
    name = match.group(1).decode('utf-8', 'replace')
    path = Path(name)
    was_relative = not path.is_absolute()
    if was_relative and source_folder is not None:
        path = Path(source_folder) / path
    if path.is_file():
        if not was_relative:
            return page_xml
        # **A relative path only means something beside the file that used
        # it.** Saving the arrangement into another folder made the reference
        # point at `target_folder/scan.pdf`, which is not where the PDF is; the
        # notebook then opened with a background Xournal++ could not find and a
        # dialog asking for it. Naming the resolved file costs a long path and
        # removes the question.
        try:
            same_folder = (target_folder is None
                           or Path(source_folder).resolve() == Path(target_folder).resolve())
        except OSError:
            same_folder = False
        if same_folder:
            return page_xml
        return _with_filename(page_xml, tag, path.resolve().as_posix())
    cache = {} if cache is None else cache
    key = (str(source_folder), path.name.lower())
    if key in cache:
        found = cache[key]
    else:
        found = None
        for folder in (source_folder, target_folder):
            if folder is None:
                continue
            try:
                for child in Path(folder).iterdir():
                    if (child.is_file()
                            and child.name.lower() == path.name.lower()):
                        found = child
                        break
            except OSError:
                continue
            if found is not None:
                break
        cache[key] = found
    if found is None:
        return page_xml
    return _with_filename(page_xml, tag, found.resolve().as_posix())


def _with_filename(page_xml: bytes, tag: bytes, filename: str) -> bytes:
    """The page with this background's `filename` replaced, **escaped**.

    The bytes were pasted into the attribute as they came: a folder called
    `A & B` produced `filename="...A & B/scan.pdf"`, which is not XML, and the
    notebook could not be opened at all -- a save that silently destroys the
    page it was saving. Attribute escaping is the only correct way to put a
    path into an attribute, and `&amp;` is what Xournal++ reads back as `&`.
    """
    from xml.sax.saxutils import escape

    escaped = escape(filename, {'"': '&quot;'}).encode('utf-8')
    fixed = _FILENAME.sub(b'filename="' + escaped + b'"', tag, count=1)
    at = page_xml.find(tag)
    return page_xml[:at] + fixed + page_xml[at + len(tag):]


def build(arrangement, target, progress=None) -> Written:
    """Write the arrangement to `target`, atomically. Returns what was made.

    Atomic means a reader of `target` sees either the previous file or the
    whole new one: the bytes go to a temporary file in the same directory and
    `os.replace` puts them in place. A failure leaves the target untouched
    and removes the temporary file.
    """
    refs = arrangement.pages
    if not refs:
        raise Refused('nothing to save: the order is empty')

    strangers = []
    for ref in refs:
        source = arrangement.source(ref.source)
        if source.kind != 'xopp' and source.label not in strangers:
            strangers.append(source.label)
    if strangers:
        raise Refused('not a .xopp document: ' + ', '.join(strangers))

    target = Path(target)
    for source in arrangement.sources:
        if source.path is None:
            continue
        try:
            same = Path(source.path).resolve() == target.resolve()
        except OSError:
            same = False
        if same:
            raise Refused(f'refusing to write over a source document: {target}')

    target.parent.mkdir(parents=True, exist_ok=True)
    from xp.core.document.atomic import fsync_directory, sweep_partials

    sweep_partials(target)
    handle, temporary = tempfile.mkstemp(prefix='.arrange-', suffix='.xopp',
                                         dir=str(target.parent))
    os.close(handle)
    try:
        first = arrangement.source(refs[0].source)
        head = _header(first)
        seen_backgrounds = {}
        missing = []
        templates = {}
        pdf_names = {}
        written = 0
        with gzip.open(temporary, 'wb', compresslevel=1) as out:
            out.write(head)
            for ref in refs:
                source = arrangement.source(ref.source)
                if source.uid not in templates:
                    templates[source.uid] = _pdf_template(source.side)
                page = source.side.page_xml(ref.index)
                page = _repair_background(page, templates[source.uid])
                page = _rewrite_background_path(
                    page, source.path.parent if source.path else None,
                    target.parent, pdf_names)
                for name in missing_backgrounds(page, seen_backgrounds):
                    if name not in missing:
                        missing.append(name)
                out.write(b'\n')
                out.write(page)
                written += 1
                if progress is not None:
                    progress(written, len(refs))
            out.write(b'\n</xournal>\n')
        size = os.path.getsize(temporary)
        os.replace(temporary, target)
        fsync_directory(target.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise

    documents = len({ref.source for ref in refs})
    return Written(Path(target), len(refs), documents, size, missing)
