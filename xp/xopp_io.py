"""`xp.xopp_io` -- kept as a name, because somebody outside imports it.

The document model moved to `xp.core.document.xopp_io` when the package was
organised by responsibility. Lab/KnowledgeMap/_render_doc.py does
`from xp.xopp_io import XoppDocument`, and that project is not ours to edit,
so the old name goes on working.

New code inside XP should import from `xp.core.document.xopp_io`.
"""
from .core.document.xopp_io import *          # noqa: F401,F403
from .core.document.xopp_io import (          # noqa: F401
    Page,
    Stroke,
    XoppDocument,
    make_stroke_element,
)
