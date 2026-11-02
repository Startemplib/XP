"""The document page drawn on the card, in the window it belongs to.

This is the surface the merge window uses: the widget **is** the GL window.
`pyopengltk`'s `OpenGLFrame` is a `tk.Frame` whose own HWND gets a WGL context,
so the picture is drawn inside the Tk window -- the same window as the toolbar,
the scrollbars and the canvas -- instead of in a second top-level window laid
over it.

What that deletes, and why it is worth deleting: `glsurface.Surface` wraps Tk's
canvas HWND with `QWindow.fromWinId`, hosts it in a `createWindowContainer`, then
detaches the container and shows it as a window of its own. Everything else in
that file is compensation for that one decision -- `place()` and `place_at()`
(two `SetWindowPos` a frame), `raise_()` after every placement because another
window can sit in front of it, and a hand-back of pointer events because the
surface, being a window of its own, swallows them.

Here the placement is a canvas item's coordinates, there is nothing to raise,
and the pointer and the keyboard never leave Tk's window tree.

Measured on this machine (RTX 4080, GL 4.6): a 1153x498 frame uploads and draws
in 3.4-3.8 ms with `glFinish` and two 1-pixel read-backs in the same loop, i.e.
that is an upper bound on the draw and it does not time the swap; the same frame
through the window-container path presents in 10.6-12.4 ms, a number this file
inherits and `tests/probe_tk_gl.py` re-measures.

The one Windows fact this rests on: `SetPixelFormat` can be applied once per
HWND, and it replaces the window procedure with one from `OPENGL32.DLL` (a plain
`tk.Frame` given a pixel format gets the same one, so it is Windows and not the
library). So a document widget must live as long as its window: this class
creates the frame once and reuses it. Rebuilding the widget is a blank document
for the rest of the session, silently.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

_FRAME_CLASS = None


def available() -> bool:
    """Whether this machine can host the surface at all."""
    try:
        import pyopengltk  # noqa: F401
        from OpenGL import GL  # noqa: F401
    except Exception:
        return False
    return True


def _frame_class():
    """Built on first use, so importing this module costs nothing."""
    global _FRAME_CLASS
    if _FRAME_CLASS is not None:
        return _FRAME_CLASS
    from pyopengltk import OpenGLFrame

    class DocumentFrame(OpenGLFrame):
        def __init__(self, master, **kw):
            super().__init__(master, **kw)
            # pyopengltk learns its size from a `<Configure>` event, and the
            # first `present` arrives before the widget has ever been mapped --
            # so the size has to exist before that event can. Asking
            # `winfo_width()` here answers 1; the request is what was asked for.
            self.width = max(1, int(kw.get('width') or 1))
            self.height = max(1, int(kw.get('height') or 1))
            self.rgb = None            # the frame, uint8 (h, w, 3)
            self.painted = 0
            self.renderer = ''
            self.version = ''
            self.texture = None
            self.texture_size = None
            self.capture_back = False
            self.back_frame = None
            self.surface = None
            self.last_error = None

        def initgl(self):
            from OpenGL import GL
            if not self.renderer:
                self.renderer = (GL.glGetString(GL.GL_RENDERER) or b'').decode('ascii', 'replace')
                self.version = (GL.glGetString(GL.GL_VERSION) or b'').decode('ascii', 'replace')
            GL.glViewport(0, 0, max(1, self.width), max(1, self.height))

        def _upload(self):
            from OpenGL import GL
            h, w = self.rgb.shape[0], self.rgb.shape[1]
            if self.texture is None:
                self.texture = int(GL.glGenTextures(1))
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
            if self.texture_size != (w, h):
                # NEAREST both ways: the frame is already the size of the widget,
                # so this is a straight blit and any filtering is a blur the
                # processor path does not have.
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                self.texture_size = (w, h)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0,
                            GL.GL_RGB, GL.GL_UNSIGNED_BYTE, self.rgb)

        def redraw(self):
            """Draw the frame. Called by pyopengltk's own expose/display path."""
            from OpenGL import GL
            w, h = max(1, self.width), max(1, self.height)
            GL.glViewport(0, 0, w, h)
            # **Cleared to paper, not to the dark ground.** This was
            # (0.06, 0.10, 0.16) -- RGB 15, 26, 41, near black -- so every
            # frame that cleared before its texture was ready showed a dark
            # rectangle exactly the size of the page. That is the black block
            # the user has been reporting: it is not a layer and not ink, it is
            # this clear.
            #
            # The user's instruction, 2026-10-30: *"the temp black in the render of
            # the merger's edit stage should already be white."* A page that
            # is not drawn yet is paper the user has not written on, so paper is
            # what it should look like. Nothing else draws this colour -- the
            # page's own image covers it as soon as there is one -- so the
            # only thing that changes is what a half-finished frame looks
            # like.
            #
            # The merge window is the only user of this surface; the
            # converter and the arranger do not create one.
            if self.rgb is None:
                # **Nothing to draw: draw nothing.** Clearing first and drawing after is what put the clear colour on
                # the screen for a frame -- the flash. The window system keeps the last presented buffer, so returning
                # here leaves the previous frame up, and a clear only ever happens on the very first frame.
                if os.environ.get('XP_GL_TRACE'):
                    print('[gl] a frame arrived with no picture; the last one stays up', flush=True)
                return
            GL.glClearColor(1.0, 1.0, 1.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            if self.rgb is not None:
                try:
                    self._upload()
                    GL.glMatrixMode(GL.GL_PROJECTION)
                    GL.glLoadIdentity()
                    GL.glOrtho(0.0, 1.0, 1.0, 0.0, -1.0, 1.0)
                    GL.glMatrixMode(GL.GL_MODELVIEW)
                    GL.glLoadIdentity()
                    GL.glDisable(GL.GL_DEPTH_TEST)
                    GL.glEnable(GL.GL_TEXTURE_2D)
                    GL.glColor3f(1.0, 1.0, 1.0)
                    GL.glBegin(GL.GL_QUADS)
                    GL.glTexCoord2f(0.0, 0.0)
                    GL.glVertex2f(0.0, 0.0)
                    GL.glTexCoord2f(1.0, 0.0)
                    GL.glVertex2f(1.0, 0.0)
                    GL.glTexCoord2f(1.0, 1.0)
                    GL.glVertex2f(1.0, 1.0)
                    GL.glTexCoord2f(0.0, 1.0)
                    GL.glVertex2f(0.0, 1.0)
                    GL.glEnd()
                    GL.glDisable(GL.GL_TEXTURE_2D)
                    self.painted += 1
                except Exception as exc:       # a blank document beats a dead one
                    self.last_error = repr(exc)
            # **After the draw, before the swap.** `_display` swaps once `redraw` returns, so this is
            # the last moment the finished frame is still in a buffer that can be read.
            if self.capture_back:
                self.capture_back = False
                try:
                    self._keep_the_frame()
                except Exception as exc:
                    self.last_error = repr(exc)

        def read_front(self, x: int, y: int):
            """One pixel of what has actually reached the screen.

            `grabFramebuffer()` is the Qt answer and does not exist here, and
            `PrintWindow` returns black for an OpenGL window. Reading the front
            buffer after the swap is the only read that is about the presented
            picture and not about the back buffer the draw went into.
            """
            from OpenGL import GL
            GL.glReadBuffer(GL.GL_FRONT)
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
            data = (np.uint8 * 4)()
            GL.glReadPixels(int(x), int(y), 1, 1, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)
            return tuple(int(v) for v in data)

        def capture_next(self) -> None:
            """Keep the frame this `redraw` draws, before it is swapped away.

            **What this can and cannot say.** On a desktop that presents, the front buffer answers
            "what is on screen" and this is not needed. On a session whose desktop is a remote or
            virtual display -- this one is `Microsoft Hyper-V Video` plus `Microsoft Remote Display
            Adapter`, and the GL context is `GDI Generic` -- the front buffer returns the clear
            colour whether or not a picture was drawn, and the screen cannot be grabbed at all
            (`ImageGrab`: `screen grab failed`). The back buffer, read between the draw and the swap,
            is then the only pixel evidence there is: it says **the frame that was drawn carries the
            page**, which is the half of "no black page" that lives in this program.
            """
            self.capture_back = True
            self.back_frame = None

        def _keep_the_frame(self) -> None:
            """Copy the back buffer into `self.back_frame`. Called from `redraw`, before the swap."""
            from OpenGL import GL
            width, height = max(1, int(self.width)), max(1, int(self.height))
            buf = np.empty((height, width, 3), np.uint8)
            GL.glReadBuffer(GL.GL_BACK)
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
            GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, buf)
            self.back_frame = np.ascontiguousarray(buf[::-1])      # GL counts from the bottom

    _FRAME_CLASS = DocumentFrame
    return DocumentFrame


class Surface:
    """A GPU surface that is a widget of the Tk window it draws into."""

    def __init__(self, parent, width: int, height: int):
        """`parent` is the canvas. The frame becomes an item of it, on top."""
        self.parent = parent
        self.width, self.height = max(1, int(width)), max(1, int(height))
        self.presented = 0
        self.item = None
        self.canvas = None
        self.swap_interval = None      # set when the driver is asked

        self.frame = _frame_class()(parent, width=self.width, height=self.height)
        self.frame.surface = self

    def set_swap_interval(self, interval: int) -> bool:
        """`wglSwapIntervalEXT`, if the driver offers it; returns whether it did.

        The default is one vblank, so `SwapBuffers` blocks until the next one --
        16.7 ms at 60 Hz, which at this window's size is the *whole* of a
        `present` and is not the cost of the frame. Measured with it off in
        `tests/probe_tk_surface.py`; left on for the window, because a tear in
        the page during a drag is worse than a frame rate of 60.
        """
        try:
            import ctypes

            opengl32 = ctypes.WinDLL('opengl32')
            opengl32.wglGetProcAddress.restype = ctypes.c_void_p
            opengl32.wglGetProcAddress.argtypes = [ctypes.c_char_p]
            address = opengl32.wglGetProcAddress(b'wglSwapIntervalEXT')
            if not address:
                return False
            self.frame.tkMakeCurrent()
            call = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)(address)
            # The function returns TRUE/FALSE, not the interval it replaced.
            if not call(int(interval)):
                return False
            self.swap_interval = int(interval)
            return True
        except Exception:
            return False

    # ---- the same four calls `gui._redraw` makes on the Qt surface ----

    def place(self, canvas) -> None:
        """Record the canvas. There is no window to move: `place_at` does it."""
        self.canvas = canvas

    def place_at(self, canvas, x, y, width, height) -> None:
        """Put the widget over a rectangle given in canvas coordinates.

        A canvas item, so Tk clips and scrolls it with the page -- no
        `SetWindowPos`, no `raise_()`, and nothing for another window to cover.
        """
        self.canvas = canvas
        width, height = max(1, int(width)), max(1, int(height))
        if self.item is None:
            self.item = canvas.create_window(int(x), int(y), anchor='nw',
                                             window=self.frame,
                                             width=width, height=height)
        else:
            # **Move it and size it.** `coords` alone only moves the top-left corner: the widget kept the rectangle it
            # was created with, so a picture composed for a new zoom was drawn into a viewport of the old size and the
            # rest of the page fell back to the canvas. Only when the size really changes -- a widget resize per frame
            # is a round trip a drag cannot afford.
            canvas.coords(self.item, int(x), int(y))
            if getattr(self, '_sized', None) != (width, height):
                self._sized = (width, height)
                try:
                    canvas.itemconfigure(self.item, width=width, height=height)
                    self.frame.configure(width=width, height=height)
                except Exception:
                    pass
            canvas.itemconfigure(self.item, width=width, height=height)
        self.width, self.height = width, height

    def present(self, rgb: np.ndarray, mask=None, colour=None) -> None:
        """Show this window image, optionally with the brush's wash on it.

        `mask` and `colour` given: the blend happens here, on the card, on the
        band the brush actually touched -- the same arithmetic `glsurface`
        does, kept identical so the two paths cannot show different pictures.
        Both `None`: the pixels are already finished and go up as they are.
        """
        pixels = np.ascontiguousarray(rgb, dtype=np.uint8)
        if mask is not None and colour is not None:
            try:
                from xp.gpu import painter as gpurender
                inside = np.nonzero(mask)
                if len(inside[0]):
                    top, bottom = int(inside[0].min()), int(inside[0].max()) + 1
                    left, right = int(inside[1].min()), int(inside[1].max()) + 1
                    crop = np.ascontiguousarray(pixels[top:bottom, left:right])
                    band = np.ascontiguousarray(mask[top:bottom, left:right])
                    pixels = pixels.copy()
                    pixels[top:bottom, left:right] = gpurender.wash(crop, band, colour, 0.30)
            except Exception:
                pass
        self.frame.rgb = pixels
        # pyopengltk's own display path: make the context current, call
        # `redraw`, then swap. Not a bare `redraw()` -- that draws into the back
        # buffer and never shows it, which is how a probe can report a picture
        # that is not on the screen.
        #
        # Not before the widget is mapped either: `wglMakeCurrent` has no
        # window yet, so the GL calls in `redraw` would run against no context.
        # The pixels are kept and the `<Expose>` that follows the first mapping
        # draws them.
        if self.frame.winfo_ismapped():
            self.frame._display()
        else:
            # **A frame presented before the widget is mapped waits for an `<Expose>`.** A scroll produces one; a
            # finished render does not -- which is exactly "it only sharpens when I move the view". Let Tk run its
            # queue and try once more, then leave the pixels for the expose if it is still not up.
            if os.environ.get('XP_GL_TRACE'):
                print('[gl] present on an unmapped frame (%dx%d); asking Tk to map it'
                      % (pixels.shape[1], pixels.shape[0]), flush=True)
            try:
                self.frame.update_idletasks()
            except Exception:
                pass
            if self.frame.winfo_ismapped():
                self.frame._display()
        self.presented += 1

    def capture_next(self) -> None:
        """Ask the frame to keep the next picture it draws, before the swap."""
        self.frame.capture_next()

    def last_drawn(self) -> Optional[np.ndarray]:
        """The last frame this surface drew, or None if nothing was kept.

        Read from the back buffer between the draw and the swap: it is about the drawing, not about
        the presentation. On a desktop that presents, `grab()` is the stronger read.
        """
        return getattr(self.frame, 'back_frame', None)

    @property
    def renderer(self) -> str:
        return getattr(self.frame, 'renderer', '') or ''

    def grab(self) -> np.ndarray:
        """What the widget has actually shown, as (h, w, 3) uint8.

        The front buffer, after the swap -- not the back buffer the draw went
        into, which would report a picture that may never have been shown.
        The Qt route's `grabFramebuffer()` has no equivalent here and
        `PrintWindow` returns black for an OpenGL window, so this is the only
        read that is about the presented picture.
        """
        from OpenGL import GL

        self.frame.tkMakeCurrent()
        w = max(1, int(self.frame.width))
        h = max(1, int(self.frame.height))
        buf = np.empty((h, w, 3), np.uint8)
        GL.glReadBuffer(GL.GL_FRONT)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        GL.glReadPixels(0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, buf)
        return np.ascontiguousarray(buf[::-1])       # GL counts from the bottom

    @property
    def hardware(self) -> bool:
        text = (self.renderer or '').lower()
        return bool(text) and 'llvmpipe' not in text and 'gdi generic' not in text

    def close(self) -> None:
        try:
            if self.item is not None and self.canvas is not None:
                self.canvas.delete(self.item)
        except Exception:
            pass
        try:
            self.frame.destroy()
        except Exception:
            pass
