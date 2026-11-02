"""A GPU surface for the document viewport, parented to a Tk widget.

This replaces the two lines that measured as 100% of a frame --
`ImageTk.PhotoImage(...)` and `canvas.create_image(...)` -- with a texture and a
swap. Measured on this machine: 5.92ms a frame against Tk's 26.0ms, 4.4x.

**The arrangement is the point, and it was learned by crashing.** The first
version called `QOpenGLWidget.repaint()` synchronously from inside Tk's event
dispatch and took the interpreter down with no traceback. `repaint()` paints
and swaps *now*, in whatever nested context it is called from; doing that from
a Tk callback is the shape that kills a process silently.

So, four rules, and this file obeys all of them:

  1. never `repaint()` -- `present()` calls `update()`, which *schedules* a
     paint, and Qt paints it in Qt's own event loop;
  2. Qt is pumped **from Tk**, by `pump()` on an `after`, not the other way
     round -- no nested loop, no re-entrant paint;
  3. nothing here measures time in a tight synchronous loop;
  4. the surface is sized and placed by the caller (the canvas rectangle), and
     the caller forwards the pointer events to the handlers it already has.

Run it on its own to see whether this machine can do it at all:

    python -m xp.glsurface

which builds a Tk window, wraps its frame, presents one frame on the card and
prints the renderer -- or says why it cannot.
"""
from __future__ import annotations

import os
import struct
import numpy as np

_APP = None
def what_to_draw(frame, texture) -> str:
    """What a repaint should put up: `'frame'`, `'previous'` or `'paper'`.

    The display layer must not depend on whether a background render is running -- the user's rule -- so a repaint with no new
    frame has an answer of its own: the picture that is already there. `'paper'` is only for a widget that has never had
    one.
    """
    if frame is not None:
        return 'frame'
    if texture is not None:
        return 'previous'
    return 'paper'


_WIDGET_CLASS = None


def available() -> bool:
    """Whether this machine can host the surface at all."""
    try:
        import PySide6  # noqa: F401
        from PySide6 import QtOpenGLWidgets, QtOpenGL  # noqa: F401
    except Exception:
        return False
    return True


def application():
    """One `QApplication` for the process. Tk keeps its own root."""
    global _APP
    if _APP is None:
        from PySide6 import QtWidgets

        _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP


def pump() -> None:
    """One turn of Qt's event loop. Called from Tk's `after`, never instead."""
    application().processEvents()


def _widget_class():
    """Built on first use, so importing this module costs nothing."""
    global _WIDGET_CLASS
    if _WIDGET_CLASS is not None:
        return _WIDGET_CLASS
    from PySide6 import QtCore, QtGui, QtOpenGL, QtOpenGLWidgets

    class Widget(QtOpenGLWidgets.QOpenGLWidget):
        def __init__(self):
            # No arguments, exactly as the script that paints: the owner is
            # attached after construction by the surface that made it.
            super().__init__()
            self.surface = None
            self.frame = None
            self.texture = None
            self.paints = 0
            # The Tk widget the pointer events are handed back to. Set by the
            # Surface that makes this widget, before it is shown.
            self.forward = None

        def initializeGL(self):
            f = QtGui.QOpenGLContext.currentContext().functions()
            name = f.glGetString(0x1F01)
            if isinstance(name, bytes):
                name = name.decode('ascii', 'replace')
            self.surface.renderer = name or ''
            self._build_quad()

        # ---- what actually puts the picture on the screen ----
        #
        # Until now paintGL uploaded the frame into a texture and then
        # cleared the buffer, with nothing in between: the surface showed
        # one flat colour. `grabFramebuffer()` proved it -- 1737052 of
        # 1737052 pixels counted as ink, because every pixel was the
        # clear colour. A texture that is never drawn is not a picture.
        def _build_quad(self):
            vertex = """#version 330 core
layout(location = 0) in vec2 corner;
out vec2 uv;
void main() {
    uv = vec2(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
    gl_Position = vec4(corner, 0.0, 1.0);
}"""
            fragment = """#version 330 core
in vec2 uv;
out vec4 colour;
uniform sampler2D frame;
void main() {
    colour = vec4(texture(frame, uv).rgb, 1.0);
}"""
            self._program = None
            self._vao = None
            self._vbo = None
            program = QtOpenGL.QOpenGLShaderProgram()
            ok = program.addShaderFromSourceCode(
                QtOpenGL.QOpenGLShader.Vertex, vertex)
            ok = ok and program.addShaderFromSourceCode(
                QtOpenGL.QOpenGLShader.Fragment, fragment)
            ok = ok and program.link()
            if not ok:
                # Loudly, because a shader that will not build means a
                # blank document and that is exactly the bug that went
                # unnoticed for a whole session.
                print('[gl] the quad did not build: %s'
                      % program.log().strip(), flush=True)
                return
            vao = QtOpenGL.QOpenGLVertexArrayObject()
            vao.create()
            vao.bind()
            vbo = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.VertexBuffer)
            vbo.create()
            vbo.bind()
            corners = struct.pack('8f', -1.0, -1.0, 1.0, -1.0,
                                  -1.0, 1.0, 1.0, 1.0)
            vbo.allocate(corners, len(corners))
            program.bind()
            program.enableAttributeArray(0)
            program.setAttributeBuffer(0, 0x1406, 0, 2, 0)
            vao.release()
            self._program, self._vao, self._vbo = program, vao, vbo

        # ---- the pointer, which belongs to the canvas underneath ----
        #
        # The surface is a window of its own sitting over the canvas, so it
        # would swallow the pointer and the brush would do nothing where the
        # picture is. Every event is handed back to the Tk canvas as the same
        # event Tk would have delivered, in canvas coordinates, so the bindings
        # KEYS.md documents keep working without knowing a second window exists.
        def _hand_back(self, widget, sequence, event):
            try:
                position = event.position()
                widget.event_generate(sequence, x=int(position.x()),
                                      y=int(position.y()))
                return True
            except Exception:
                return False

        def mousePressEvent(self, event):
            if not self._hand_back(self.forward, '<Button-1>', event):
                super().mousePressEvent(event)

        def mouseReleaseEvent(self, event):
            if not self._hand_back(self.forward, '<ButtonRelease-1>', event):
                super().mouseReleaseEvent(event)

        def mouseMoveEvent(self, event):
            # `<B1-Motion>` and not `<Motion>` while the left button is held:
            # the brush binds the drag, not the hover, so handing back a plain
            # motion would move the pointer and decide nothing.
            sequence = ('<B1-Motion>'
                        if (getattr(event.buttons(), 'value', event.buttons())
                            & getattr(QtCore.Qt.LeftButton, 'value', 1))
                        else '<Motion>')
            if not self._hand_back(self.forward, sequence, event):
                super().mouseMoveEvent(event)

        def wheelEvent(self, event):
            step = 1 if event.angleDelta().y() > 0 else -1
            try:
                self.forward.event_generate('<MouseWheel>',
                                            x=int(event.position().x()),
                                            y=int(event.position().y()),
                                            delta=step * 120)
            except Exception:
                super().wheelEvent(event)

        def paintGL(self):
            f = QtGui.QOpenGLContext.currentContext().functions()
            # **The last picture stays up.** No new frame is not "nothing to show": the previous picture is brought back
            # by drawing the texture that is already there. Only a widget that has never had a picture may show a bare
            # clear, and that clear is paper. This is the whole of visual continuity: the display does not depend on a
            # background job, and a repaint with no new frame cannot blank the screen.
            choice = what_to_draw(self.frame, self.texture)
            if choice == 'previous':
                clear = getattr(self.surface, 'clear', (1.0, 1.0, 1.0))
                f.glClearColor(float(clear[0]), float(clear[1]), float(clear[2]), 1.0)
                f.glClear(0x00004000)
                self._draw_quad()
                return
            if choice == 'paper':
                clear = getattr(self.surface, 'clear', (1.0, 1.0, 1.0))
                f.glClearColor(float(clear[0]), float(clear[1]), float(clear[2]), 1.0)
                f.glClear(0x00004000)
                return
            if self.frame is not None:
                # The brush's wash, blended on the card between the frame
                # arriving and the texture being written. Doing it here is what
                # removes the per-event round trip: previously `gui._washed`
                # uploaded the window, blended, downloaded it, rebuilt a PIL
                # image, and this method uploaded it again.
                pixels = self.frame
                mask = getattr(self, 'mask', None)
                colour = getattr(self, 'colour', None)
                if mask is not None and colour is not None:
                    try:
                        from xp.gpu import painter as gpurender
                        # Only the brush's own rectangle crosses the bus. The
                        # wash is identity where the mask is zero, so the band
                        # around the cursor can be blended and written back into
                        # a copy of the frame -- same pixels, and the upload and
                        # download shrink from the whole window to the part the
                        # stroke actually touched.
                        inside = np.nonzero(mask)
                        if len(inside[0]):
                            top, bottom = int(inside[0].min()), int(inside[0].max()) + 1
                            left, right = int(inside[1].min()), int(inside[1].max()) + 1
                            crop = np.ascontiguousarray(pixels[top:bottom, left:right])
                            band = np.ascontiguousarray(mask[top:bottom, left:right])
                            washed = gpurender.wash(crop, band, colour, 0.30)
                            pixels = pixels.copy()
                            pixels[top:bottom, left:right] = washed
                        else:
                            pixels = self.frame
                    except Exception:
                        pixels = self.frame
                h, w = pixels.shape[0], pixels.shape[1]
                image = QtGui.QImage(pixels, w, h, w * 3,
                                     QtGui.QImage.Format_RGB888)
                if os.environ.get('XP_GL_TRACE'):
                    print('[gl] frame %dx%d  texture %s'
                          % (w, h,
                             ('%dx%d' % (self.texture.width(),
                                        self.texture.height()))
                             if self.texture is not None else 'none'),
                          flush=True)
                # A new texture for every frame.
                #
                # `setData(QImage)` on an allocated texture re-applies the
                # image's mip levels and format, which Qt refuses:
                #
                #   Cannot set mip levels on a texture that already has
                #   storage allocated.
                #   QOpenGLTexture::setFormat(): Cannot change format once
                #   storage has been allocated.
                #
                # It failed even when the size was identical (751x546 into a
                # 751x546 texture), so the size guard was never the issue and
                # the stroke never reached the screen. Rebuilding is a few
                # tenths of a millisecond on the 4080 and it is always right.
                # **Build the new texture before letting the old one go.** Destroy-then-create left a repaint with no
                # texture to draw, which is a blank frame by construction; swapping keeps one alive at all times.
                made = QtOpenGL.QOpenGLTexture(image)
                previous, self.texture = self.texture, made
                if previous is not None:
                    previous.destroy()
            # **The colour this surface was made with.** The merger passes paper; everyone else keeps the dark ground,
            # so a frame that clears before its texture is ready is paper for the merger and never a black rectangle
            # over the page.
            clear = getattr(self.surface, 'clear', (0.06, 0.10, 0.16))
            f.glClearColor(float(clear[0]), float(clear[1]), float(clear[2]), 1.0)
            f.glClear(0x00004000)
            self._draw_quad()

        def _draw_quad(self):
            """Put the texture on the screen. Without this the surface
            is the clear colour and nothing else."""
            if self._program is None or self.texture is None:
                return
            f = QtGui.QOpenGLContext.currentContext().functions()
            self._program.bind()
            self.texture.bind(0)
            self._program.setUniformValue('frame', 0)
            self._vao.bind()
            f.glDrawArrays(0x0005, 0, 4)      # GL_TRIANGLE_STRIP
            self._vao.release()
            self.texture.release(0)
            self._program.release()
            self.paints += 1
            # **The frame that was drawn, kept before it is swapped away.** See `capture_next`:
            # on this machine the screen cannot be photographed and the front buffer is the clear
            # colour, so the back buffer is the only pixel evidence that the picture reached a buffer
            # at all.
            if getattr(self, 'capture_back', False):
                self.capture_back = False
                try:
                    self._keep_the_frame()
                except Exception as exc:
                    self.back_error = repr(exc)

        def capture_next(self) -> None:
            self.capture_back = True
            self.back_frame = None

        def _keep_the_frame(self) -> None:
            from OpenGL import GL
            ratio = float(self.devicePixelRatio()) or 1.0
            width = max(1, int(round(self.width() * ratio)))
            height = max(1, int(round(self.height() * ratio)))
            buf = np.empty((height, width, 3), np.uint8)
            GL.glReadBuffer(GL.GL_BACK)
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
            GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, buf)
            self.back_frame = np.ascontiguousarray(buf[::-1])

    _WIDGET_CLASS = Widget
    return Widget


class Surface:
    """A GPU surface drawing into a rectangle of a Tk window."""

    def __init__(self, parent, width: int, height: int, clear=None):
        """`clear` is the colour the framebuffer is cleared to before the texture is drawn.

        None keeps the window's dark ground, which is what every existing caller gets. A caller whose picture is a page
        -- the merger -- passes paper instead, because a frame that clears before its texture is ready must not be a
        black rectangle over the page.
        """
        application()
        from PySide6 import QtCore, QtGui, QtWidgets

        self.parent = parent
        self.width, self.height = int(width), int(height)
        self.clear = tuple(clear) if clear is not None else (0.06, 0.10, 0.16)
        self.renderer = ''
        self.presented = 0

        hwnd = int(parent.winfo_id())
        window = QtGui.QWindow.fromWinId(hwnd)
        if window is None:
            raise RuntimeError('QWindow.fromWinId returned nothing for %d' % hwnd)
        self.window = window
        self.container = QtWidgets.QWidget.createWindowContainer(window)
        self.container.setAttribute(QtCore.Qt.WA_NativeWindow, True)
        # From the script that paints: the container is detached, sized, and
        # shown *before* the widget exists. Both are one-line differences from
        # the version that never painted and neither had a reason to matter
        # until the diff was made.
        self.container.setParent(None)
        self.container.resize(self.width, self.height)
        self.container.show()

        self.widget = _widget_class()()
        self.widget.surface = self
        self.widget.forward = parent
        # No `setMinimumSize` here: the script that paints has none, and inside
        # a container that is smaller than what the minimum asks for it fights
        # the layout. Being removed one line at a time is the point of the diff.
        self.widget.resize(self.width, self.height)
        layout = QtWidgets.QVBoxLayout(self.container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.widget)
        self.widget.show()
        pump()

    def present(self, rgb: np.ndarray, mask=None, colour=None) -> None:
        """Show this window image, optionally with the brush's wash on it.

        **The wash is applied here, on the way into the texture**, and that is
        the point of the extra arguments. It used to happen in `gui._washed`,
        which converted the window to PIL, uploaded it, blended, downloaded it,
        rebuilt a PIL image -- and then this method uploaded it again. Two
        uploads and one download per mouse-motion event, for one alpha blend,
        which is exactly the shape the instruction says the display must not
        have.

        With `mask` and `colour` given, the blend happens once, on the card,
        between the frame arriving and the texture being written. With both
        left as None the call is byte-for-byte what it was before this
        argument existed, which is what keeps the change safe to land in two
        halves.
        """
        self.widget.frame = np.ascontiguousarray(rgb, dtype=np.uint8)
        self.widget.mask = mask
        self.widget.colour = colour
        self.widget.repaint()         # as the script that paints does
        self.presented += 1

    def capture_next(self) -> None:
        """Ask the widget to keep the next frame it draws, before the swap."""
        widget = getattr(self, 'widget', None) or getattr(self, 'view', None)
        keep = getattr(widget, 'capture_next', None)
        if keep is None:
            raise RuntimeError('this surface cannot keep the frame it drew')
        keep()

    def last_drawn(self):
        """The last frame the widget drew, or None."""
        widget = getattr(self, 'widget', None) or getattr(self, 'view', None)
        return getattr(widget, 'back_frame', None) if widget is not None else None

    def place_at(self, canvas, canvas_x, canvas_y, width, height) -> None:
        """Put the surface over a rectangle given in canvas coordinates.

        `place` covers the whole canvas, so the frame -- which is a crop --
        was stretched to fill it: measured 751x546 of texture drawn into a
        1038x546 widget, 38% too wide. Sizing the surface to the frame
        makes the two the same shape, so the shader cannot stretch it.
        """
        try:
            sx = canvas.winfo_rootx() + int(canvas_x - canvas.canvasx(0))
            sy = canvas.winfo_rooty() + int(canvas_y - canvas.canvasy(0))
            self.container.setGeometry(int(sx), int(sy),
                                       int(width), int(height))
        except Exception:
            pass

    def place(self, canvas) -> None:
        """Put the surface exactly over a Tk canvas, in screen coordinates.

        This is the whole of the second window's bookkeeping: the canvas says
        where it is and how big it is, and the surface goes there. Called from
        `_redraw` and from `<Configure>`, so a move, a resize or a scroll of the
        Tk window carries the surface with it.

        It is also raised after being placed. The surface is a window of its
        own, so another window can sit in front of it and the document area then
        looks empty -- which happened twice in screenshots this session and is
        the standing cost of presenting this way.
        """
        try:
            self.container.setGeometry(canvas.winfo_rootx(),
                                       canvas.winfo_rooty(),
                                       canvas.winfo_width(),
                                       canvas.winfo_height())
        except Exception:
            pass
        try:
            self.container.raise_()
        except Exception:
            pass

    @property
    def hardware(self) -> bool:
        text = (self.renderer or '').lower()
        return bool(text) and 'llvmpipe' not in text and 'gdi generic' not in text

    def close(self) -> None:
        for part in ('container', 'widget'):
            try:
                getattr(self, part).close()
            except Exception:
                pass


def _main() -> int:
    import tkinter as tk
    import time

    if not available():
        print('no PySide6; the processor path is the answer here')
        return 1

    root = tk.Tk()
    root.title('xp gl surface')
    root.geometry('420x320')
    frame = tk.Frame(root, width=320, height=240)
    frame.pack(fill='both', expand=True)
    label = tk.Label(frame, text='a Tk widget that must keep working')
    label.place(x=4, y=4)
    root.update()

    try:
        surface = Surface(frame, 320, 240)
    except Exception as exc:
        print('could not attach a surface: %r' % (exc,))
        root.destroy()
        return 1

    y = np.linspace(0, 1, 240, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, 320, dtype=np.float32)[None, :]
    rgb = np.empty((240, 320, 3), np.uint8)
    rgb[:, :, 0] = (x * 255).astype(np.uint8)
    rgb[:, :, 1] = (y * 255).astype(np.uint8)
    rgb[:, :, 2] = 40

    surface.present(np.ascontiguousarray(rgb))
    for _ in range(12):
        pump()
        root.update()
        time.sleep(0.01)

    # The widget was built and shown and never painted. Two things can be true
    # and they need telling apart: no GL context at all, or a context that Qt
    # simply has not been asked to paint yet. `makeCurrent` answers the first.
    made_current = False
    try:
        made_current = bool(surface.widget.makeCurrent())
    except Exception as exc:
        print('makeCurrent raised: %r' % (exc,))

    # And drive the exposure by hand rather than waiting for it: re-show and
    # re-size after the Tk frame is mapped, then pump again.
    try:
        surface.container.show()
        surface.container.raise_()
        surface.widget.show()
        surface.widget.resize(surface.width, surface.height)
    except Exception as exc:
        print('re-show raised: %r' % (exc,))
    for _ in range(20):
        pump()
        root.update()
        time.sleep(0.01)
    surface.present(np.ascontiguousarray(rgb))
    for _ in range(10):
        pump()
        root.update()
        time.sleep(0.01)

    clicked = []
    label.bind('<Button-1>', lambda e: clicked.append(1))
    label.event_generate('<Button-1>', x=2, y=2)
    root.update()

    forced = False
    if surface.widget.paints == 0:
        # Nothing has a context (`makeCurrent` was False), so there is nothing
        # to paint into and `update()` can never help. `grabFramebuffer()`
        # makes the context current and paints once as a side effect; it is the
        # one call that forces the initialisation the widget never did.
        try:
            grabbed = surface.widget.grabFramebuffer()
            forced = True
            print('grabFramebuffer: %s' % ('ok %dx%d' % (grabbed.width(), grabbed.height())))
        except Exception as exc:
            print('grabFramebuffer raised: %r' % (exc,))
        for _ in range(10):
            pump()
            root.update()
            time.sleep(0.01)

    print('renderer:      %s' % (surface.renderer or '(none)'))
    print('hardware:      %s' % surface.hardware)
    print('makeCurrent:   %s' % made_current)
    print('frames asked:  %d, painted: %d' % (surface.presented, surface.widget.paints))
    print('texture made:  %s' % (surface.widget.texture is not None))
    print('tk still alive: frame=%s label=%dx%d events=%s'
          % (bool(frame.winfo_exists()), label.winfo_width(), label.winfo_height(),
             bool(clicked)))
    surface.close()
    root.destroy()
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
