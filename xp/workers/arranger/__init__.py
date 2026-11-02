"""Arranger: page-level work on `.xopp` notebooks.

Take whole pages out of any number of notebooks, reorder them, duplicate or
delete them, and write one new `.xopp`. Page XML is copied byte for byte, so
nothing is re-traced and nothing is re-rendered. It never merges pages into
each other, never cuts a page, and never re-renders ink into a file: a page is
the bytes it came in as, and only its position in the output changes.

That is a different job from the merger, which decides *within* a page.

**The code lives here now, and it still runs as its own process.** Until
2026-10-21 it was a separate project (`Lab/Arranger`) and this package held
only the adapter that started it. The program has moved in; `bridge` still
starts it with `pythonw -m xp.workers.arranger`, detached, so it keeps its own
Tk root and its own crash domain and a fault in one cannot take the other
down. One tree, two processes -- which is what the dashboard needs anyway,
since its Tk loop is already occupied.

`bridge` is the half XP's launcher and dashboard call. Everything else in this
package is the program.
"""

__all__ = ['arrangement', 'arrangergui', 'xopp', 'thumb', 'pageart', 'preview',
           'writer', 'look', 'bridge']
