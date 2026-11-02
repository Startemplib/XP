"""Where a page is presented: the GPU surfaces and their fallback.

`tkgl` is the default -- a Tk frame that owns the GL context, so
the picture is drawn inside the main window. `glsurface` is the
Qt fallback, a second top-level window placed over the canvas.
Both answer the same four calls (`place`, `place_at`, `present`,
`close`), which is why the caller does not know which it has.
"""
