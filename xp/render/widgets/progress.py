"""A centred progress panel, and the thread that feeds it.

Lifted from the arranger, which already had one that works, rather than
written again: a panel placed over the window's own canvas, a determinate or
indeterminate bar, a detail line, an error state with a Close button. What is
added here is the half the arranger did not need to share -- running the long
job on a worker thread and pumping its progress into the panel from Tk's own
loop, so the window keeps painting while the work goes on.

The rule the whole thing exists for: **do not invent progress.** A step that
can count what it is doing reports `done / total`; a step that cannot says
what it is doing and leaves the bar pulsing. A bar that crawls to 90% and
waits is a lie, and the point of this panel is to be believed.

Used by the merge window's save, which converts the ink the user took from a PDF
side and then writes the document -- a job of minutes on a dense page, during
which the only feedback used to be the mouse pointer.
"""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

from xp.render.appearance import ABYSS, MUTED, PAPER, TEXT, font as face

FAILED_COLOUR = '#ff7b72'
# Six pixels: a rule of colour under the title, not a native 22-pixel trough.
BAR_HEIGHT = 6


class Progress:
    """The panel. Built once per window, hidden until something starts."""

    def __init__(self, parent, width: int = 430):
        self.parent = parent
        self.panel = tk.Frame(parent, bg=ABYSS, highlightbackground=PAPER,
                              highlightthickness=1)
        # **The same body size the rest of XP writes in.** The title was 12 pt
        # bold and the two lines under it 9 and 8 -- next to a 10 pt panel that
        # reads as "the loading is a different program", and on a 150 % display
        # the 12 pt title is what made the panel look too large for its box.
        self.title = tk.Label(self.panel, text='', bg=ABYSS, fg=TEXT,
                              font=face(10, 'bold'), anchor='w', justify='left')
        self.title.pack(anchor='w', padx=18, pady=(14, 4))
        # **Drawn here, not `ttk.Progressbar`.** On Windows the ttk theme paints
        # its own 22-pixel trough in the system's grey with a green block, and it
        # ignores the colours a style asks for: next to 10 pt text it is the
        # "loading bar is too large" the user pointed at. Three pixels of the panel's
        # own accent colour on a dark trough, and it says the same thing.
        self.bar = tk.Canvas(self.panel, height=BAR_HEIGHT, bg=ABYSS,
                             highlightthickness=0, bd=0, width=width)
        self._maximum = None
        self._value = 0.0
        self._pulse = 0.0
        self._pulsing = False
        self.bar.pack(fill=tk.X, padx=18, pady=(2, 4))
        self.detail = tk.Label(self.panel, text='', bg=ABYSS, fg=MUTED,
                               font=face(10), anchor='w', justify='left',
                               wraplength=width)
        self.detail.pack(anchor='w', padx=18, pady=(2, 4))
        # Details, folded away: the worker, what it was given, and on a
        # failure the exception. Nobody wants it until something goes wrong.
        self.more_button = tk.Button(
            self.panel, text='Details', command=self.toggle_details,
            bg=ABYSS, fg=MUTED, relief='flat', activebackground=PAPER,
            activeforeground=ABYSS, font=face(10), borderwidth=0)
        self.more = tk.Label(self.panel, text='', bg=ABYSS, fg=MUTED,
                             font=face(9), anchor='w', justify='left',
                             wraplength=width)
        self.close_button = tk.Button(
            self.panel, text='Close', command=self.hide, bg=ABYSS, fg=TEXT,
            relief='flat', activebackground=PAPER, activeforeground=ABYSS,
            font=face(10), borderwidth=0)
        self._details_open = False
        self._details_text = ''
        self._started = None
        self.panel.place_forget()

    # ---- showing ---------------------------------------------------------

    def show(self, title: str, maximum=None, detail: str = 'starting...',
             details: str = '') -> None:
        self._started = time.monotonic()
        self._details_text = details
        self.close_button.pack_forget()
        self.more.pack_forget()
        self.more_button.pack_forget()
        self._details_open = False
        self.title.configure(text=title, fg=TEXT)
        self.detail.configure(text=detail, fg=MUTED)
        self._stop_pulsing()
        if not self.bar.winfo_ismapped():
            self.bar.pack(fill=tk.X, padx=18, pady=(2, 4), before=self.detail)
        self._maximum = max(1, int(maximum)) if maximum else None
        self._value = 0.0
        self._paint_bar()
        if self._maximum is None:
            self._pulsing = True
            self._pulse_tick()
        if details:
            self.more_button.pack(anchor='w', padx=14, pady=(0, 10))
        self.panel.place(relx=0.5, rely=0.5, anchor='center')
        self.panel.lift()

    def step(self, done: int, total: int, detail: str = '') -> None:
        """A step that can be counted. `done / total` and a percentage."""
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        self._maximum = total
        self._value = float(done)
        self._stop_pulsing()
        self._paint_bar()
        line = '%d / %d  ·  %d%%%s' % (done, total,
                                       int(round(done * 100.0 / total)),
                                       self._elapsed())
        self.detail.configure(text='%s\n%s' % (detail, line) if detail else line)

    def pulse(self, detail: str) -> None:
        """A step that cannot be counted. Say what it is, not how far."""
        self.detail.configure(text='%s%s' % (detail, self._elapsed()))
        if not self._pulsing:
            self._maximum = None
            self._pulsing = True
            self._pulse_tick()

    def failed(self, title: str, message: str, details: str = '') -> None:
        self._stop_pulsing()
        self.bar.pack_forget()
        self.title.configure(text=title, fg=FAILED_COLOUR)
        self.detail.configure(text=message, fg=FAILED_COLOUR)
        self._details_text = details or self._details_text
        if self._details_text:
            self.more_button.pack(anchor='w', padx=14, pady=(0, 2))
        self.close_button.pack(anchor='e', padx=18, pady=(0, 12))
        self.panel.place(relx=0.5, rely=0.5, anchor='center')
        self.panel.lift()

    def hide(self) -> None:
        self._stop_pulsing()
        self.panel.place_forget()

    # ---- the bar, drawn ------------------------------------------------

    def _paint_bar(self) -> None:
        try:
            width = max(1, self.bar.winfo_width())
        except tk.TclError:
            return
        self.bar.delete('all')
        middle = BAR_HEIGHT // 2
        self.bar.create_rectangle(0, middle - 1, width, middle + 2,
                                  fill='#1c2b3a', outline='')
        if self._maximum is None:
            # A short segment walking to the right: it says "running" without
            # claiming a share.
            span = max(24, int(width * 0.18))
            start = int((width - span) * self._pulse)
            self.bar.create_rectangle(start, middle - 1, start + span,
                                      middle + 2, fill=PAPER, outline='')
            return
        share = 0.0 if not self._maximum else self._value / float(self._maximum)
        share = max(0.0, min(1.0, share))
        if share:
            self.bar.create_rectangle(0, middle - 1, max(2, int(width * share)),
                                      middle + 2, fill=PAPER, outline='')

    def _pulse_tick(self) -> None:
        if not self._pulsing:
            return
        self._pulse = (self._pulse + 0.04) % 1.0
        self._paint_bar()
        try:
            self._pulse_job = self.panel.after(40, self._pulse_tick)
        except tk.TclError:
            self._pulsing = False

    def _stop_pulsing(self) -> None:
        self._pulsing = False
        job = getattr(self, '_pulse_job', None)
        if job is not None:
            try:
                self.panel.after_cancel(job)
            except tk.TclError:
                pass
            self._pulse_job = None

    def toggle_details(self) -> None:
        self._details_open = not self._details_open
        if self._details_open:
            self.more.configure(text=self._details_text)
            self.more.pack(anchor='w', padx=18, pady=(0, 10),
                           before=self.close_button if
                           self.close_button.winfo_ismapped() else None)
        else:
            self.more.pack_forget()

    def _elapsed(self) -> str:
        if self._started is None:
            return ''
        seconds = time.monotonic() - self._started
        if seconds < 60:
            return '   ·   %.0fs' % seconds
        return '   ·   %d:%02d' % (int(seconds // 60), int(seconds % 60))


class Job:
    """One long job on a worker thread, reported through a `Progress`.

    The work function is handed a `say` callable and nothing else it has to
    know about Tk. Everything it says goes through a queue and is drawn from
    Tk's own loop, because touching a widget from the worker thread is how a
    Tk program dies without a traceback.

    `say` takes either `(done, total, text)` for a step that can be counted or
    `(text,)` for one that cannot -- which is the whole of the no-inventing
    rule, expressed as two different calls.
    """

    POLL_MS = 60

    def __init__(self, window, progress: Progress, title: str,
                 work, on_done=None, on_fail=None, maximum=None,
                 details: str = ''):
        self.window = window
        self.progress = progress
        self.title = title
        self.work = work
        self.on_done = on_done
        self.on_fail = on_fail
        self.maximum = maximum
        self.details = details
        self.queue: queue.Queue = queue.Queue()
        self.thread = None
        self.result = None
        self.error = None
        self.cancelled = False
        self.finished = False

    def start(self) -> 'Job':
        self.progress.show(self.title, self.maximum, details=self.details)

        def say(*args):
            self.queue.put(('step', args))

        def run():
            try:
                self.result = self.work(say)
                self.queue.put(('done', None))
            except BaseException as exc:          # reported, never swallowed
                self.error = exc
                self.queue.put(('failed', exc))

        self.thread = threading.Thread(target=run, daemon=True,
                                       name='xp-%s' % self.title)
        self.thread.start()
        self.window.after(self.POLL_MS, self._pump)
        return self

    def _pump(self) -> None:
        latest = None
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == 'step':
                    latest = payload          # only the newest is worth drawing
                elif kind == 'done':
                    self._settle(True)
                    return
                elif kind == 'failed':
                    self._settle(False, payload)
                    return
        except queue.Empty:
            pass
        if latest is not None:
            if len(latest) >= 3:
                self.progress.step(latest[0], latest[1], str(latest[2]))
            elif len(latest) == 2:
                self.progress.step(latest[0], latest[1])
            elif latest:
                self.progress.pulse(str(latest[0]))
        self.window.after(self.POLL_MS, self._pump)

    def _settle(self, ok: bool, error=None) -> None:
        self.finished = True
        if ok:
            self.progress.hide()
            if self.on_done is not None:
                self.on_done(self.result)
            return
        import traceback
        self.progress.failed(
            '%s did not finish' % self.title, str(error) or error.__class__.__name__,
            details=''.join(traceback.format_exception(
                type(error), error, error.__traceback__))[-2000:])
        if self.on_fail is not None:
            self.on_fail(error)
