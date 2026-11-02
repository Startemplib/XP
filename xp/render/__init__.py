"""Shared presentation: drawing, rasterising, style, and surfaces.

What is reusable about showing something. A worker's own windows live with
that worker; this package must not import from `xp.workers`, or the
dependency runs backwards.

**Nothing is imported here until it is asked for, and that is not tidiness.**
`xp.render` used to be a single module, and an external consumer imports
`render_page` from it by that name (Lab/KnowledgeMap/_render_doc.py), so the
old spelling has to keep working. Re-exporting it the obvious way --
`from .drawing import render_page` at the top of this file -- made importing
*anything* under `xp.render` pull in `drawing`, which pulls in `raster`,
which pulls in **numba**. Measured: `import xp.render.appearance`, a module
that wants PIL and tkinter and nothing else, cost 0.70 s and dragged numba
into every process that touched a colour. The suite went up 26.6%.

So the names are resolved on first use (PEP 562). `from xp.render import
appearance` costs what it always did; `from xp.render import render_page`
still works, and pays for `drawing` only then.
"""
from __future__ import annotations

# The drawing module's public surface, by name only. Asking for one of these
# loads `.drawing`; importing a sibling module does not.
_FROM_DRAWING = frozenset({
    'BACKGROUND_COLOR',
    'DEFAULT_DISPLAY_SCALE',
    'LEFT_ONLY_COLOR',
    'RIGHT_ONLY_COLOR',
    'STROKE_CHARS_PER_POINT',
    'cut_path',
    'draw_stroke',
    'draw_strokes',
    'path_limit',
    'render_page',
    'segment_ink',
    'to_pixels',
})

__all__ = sorted(_FROM_DRAWING)


def __getattr__(name):
    if name in _FROM_DRAWING:
        from . import drawing
        value = getattr(drawing, name)
        globals()[name] = value        # resolved once, then it is simply there
        return value
    raise AttributeError('module %r has no attribute %r' % (__name__, name))


def __dir__():
    return sorted(set(globals()) | _FROM_DRAWING)
