"""Review selected files and their left/right order before opening a merge.

Two merges are offered here, and the stroke one is the default. They answer
different questions and neither replaces the other:

  stroke by stroke   two documents, decided ink by ink, in the brush window.
                     What you want when both sides are real versions of the
                     same notebook and you have to choose between them.
  image level        one notebook plus what is new in a PDF or another
                     notebook. It compares the two as pictures and patches
                     in only the parts that changed -- it never converts the
                     PDF to strokes, so it costs what changed rather than
                     what the document already contains.
"""
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from xp.render.appearance import decorate

STROKE_MERGE = 'Stroke by stroke (the full merge)'
IMAGE_MERGE = 'Image level - only what changed'

MODE_NOTES = {
    STROKE_MERGE: ('Every stroke on both sides is paired and decided, with '
                   'the brush. Converts a PDF side to strokes first.'),
    IMAGE_MERGE: ('Compares the two as pictures and patches in only what is '
                  'new, keeping the left notebook exactly as it is. Never '
                  'converts the PDF to strokes. Pages are paired by position.'),
}

# **Position, not matching, is the default.** Pages pair 1 with 1 unless the user
# opens the map and says otherwise; this used to be 'auto', so opening a merge
# from this window ran a full content match before the user had seen a single page.
# The automatic part of opening is the *geometry* -- `MergeSession` calibrates
# at startup -- and pairing is the user's to ask for. The map editor is still one
# click away, and the CLI's own default is the same word.
DEFAULT_PAGE_MATCH = 'index'
PAGE_MATCH_CHOICES = ('index', 'manual', 'auto')


def normalized_sources(paths):
    found = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        if path.suffix.lower() not in ('.xopp', '.pdf') or not path.is_file():
            raise ValueError(f'Choose an existing .xopp or .pdf file:\n{path}')
        if path not in found:
            found.append(path)
    if len(found) > 2:
        raise ValueError('Select one file to convert/open, or exactly two files to merge.')
    # A notebook is the natural base when the other side is a PDF.
    if len(found) == 2 and {p.suffix.lower() for p in found} == {'.pdf', '.xopp'}:
        found.sort(key=lambda p: p.suffix.lower() != '.xopp')
    return found


class MergeFilesWindow:
    """Pick the two sides, then open them as a merge or as a read-only
    comparison -- the same two files either way, so it is one window with
    two ways out rather than two windows."""

    def __init__(self, app, paths=(), mode='merge'):
        self.app = app
        self.mode = mode
        paths = normalized_sources(paths)
        self.root = tk.Toplevel(app.root)
        from xp.runtime.shutdown import arm
        arm(self.root)
        self.root.title('XP / Choose sides to ' + ('compare' if mode == 'diff' else 'merge'))
        self.root.geometry('760x300+40+40')
        decorate(self.root)
        body = ttk.Frame(self.root, style='Lab.TFrame', padding=18)
        body.pack(fill='both', expand=True)
        body.columnconfigure(1, weight=1)
        ttk.Label(body, text='TWO SOURCES / ONE NOTEBOOK', style='Title.Lab.TLabel').grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 18))
        self.sides = [tk.StringVar(self.root, str(paths[i]) if len(paths) > i else '') for i in range(2)]
        for row, label in enumerate(('Left / base', 'Right / incoming'), start=1):
            ttk.Label(body, text=label, style='Lab.TLabel').grid(row=row, column=0, sticky='w', padx=(0, 10), pady=8)
            ttk.Entry(body, textvariable=self.sides[row - 1]).grid(row=row, column=1, sticky='ew', pady=8)
            ttk.Button(body, text='Browse...', command=lambda i=row - 1: self.browse(i)).grid(row=row, column=2, padx=(10, 0))
        ttk.Button(body, text='Swap left and right', command=self.swap).grid(row=3, column=1, sticky='w', pady=8)
        # **Position, not matching.** Pages pair 1 with 1 unless the user opens the
        # map and says otherwise; the default used to be 'auto', so opening a
        # merge from this window ran a full content match before the user had seen a
        # single page. The automatic part of opening is the *geometry*
        # (`MergeSession` calibrates at startup); pairing is the user's to ask for.
        self.align = tk.StringVar(self.root, DEFAULT_PAGE_MATCH)
        ttk.Label(body, text='Page matching', style='Lab.TLabel').grid(row=4, column=0, sticky='w')
        self.align_box = ttk.Combobox(body, textvariable=self.align, values=PAGE_MATCH_CHOICES, state='readonly', width=14)
        self.align_box.grid(row=4, column=1, sticky='w')
        row = 5
        if mode != 'diff':
            # The two merges, and the stroke one stays the default: it is the
            # one that decides between two documents, and nothing about the
            # new one replaces it. The user's words -- the current total stroke
            # merge should also remain, this is just a new mode I can select.
            self.merge_mode = tk.StringVar(self.root, STROKE_MERGE)
            ttk.Label(body, text='Merge mode', style='Lab.TLabel').grid(row=5, column=0, sticky='w', pady=8)
            box = ttk.Combobox(body, textvariable=self.merge_mode,
                               values=(STROKE_MERGE, IMAGE_MERGE),
                               state='readonly', width=34)
            box.grid(row=5, column=1, sticky='w', pady=8)
            box.bind('<<ComboboxSelected>>', lambda _e: self._mode_changed())
            self.mode_note = ttk.Label(body, text=MODE_NOTES[STROKE_MERGE], style='Muted.Lab.TLabel', wraplength=520, justify='left')
            self.mode_note.grid(row=6, column=1, columnspan=2, sticky='w')
            row = 7
        else:
            self.merge_mode = None
        note = ('Your originals stay intact; nothing is written.' if mode == 'diff'
                else 'Your originals stay intact. Choose a separate result file next.')
        ttk.Label(body, text=note, style='Muted.Lab.TLabel').grid(row=row, column=0, columnspan=2, sticky='w', pady=14)
        ttk.Button(body, text='Open comparison...' if mode == 'diff' else 'Open merge...',
                   command=self.start).grid(row=row, column=2)
        self.root.lift()

    def _mode_changed(self):
        """The image mode pairs pages by position, so the page-matching box
        would be a control that does nothing. Grey it out rather than leave
        it looking live."""
        by_position = self.merge_mode.get() == IMAGE_MERGE
        self.mode_note.configure(text=MODE_NOTES[self.merge_mode.get()])
        self.align_box.configure(state='disabled' if by_position else 'readonly')

    def browse(self, index):
        filename = filedialog.askopenfilename(parent=self.root, title='Choose a source', filetypes=[('Notebook or PDF', '*.xopp *.pdf')])
        if filename:
            self.sides[index].set(filename)

    def swap(self):
        left, right = (v.get() for v in self.sides)
        self.sides[0].set(right)
        self.sides[1].set(left)

    def add_sources(self, paths):
        """A delayed second Explorer invocation fills the existing empty side."""
        if not self.root.winfo_exists():
            return False
        current = [v.get() for v in self.sides if v.get()]
        if len(current) != 1:
            return False
        combined = normalized_sources(current + list(paths))
        if len(combined) != 2:
            return False
        other = next(path for path in combined if path != Path(current[0]).resolve())
        empty = next(v for v in self.sides if not v.get())
        empty.set(str(other))
        self.root.lift()
        return True

    def start(self):
        if self.app._busy:
            messagebox.showinfo('Please wait', 'Wait for the current command to finish.', parent=self.root)
            return
        try:
            # Validate without reordering: a deliberate swap must be respected.
            sources = [Path(variable.get()).expanduser().resolve() for variable in self.sides]
            if len(normalized_sources(sources)) != 2:
                raise ValueError('Choose two different files.')
        except ValueError as exc:
            messagebox.showerror('Check merge sources', str(exc), parent=self.root)
            return
        if self.mode == 'diff':
            self.app.run_arguments(['diff', str(sources[0]), str(sources[1]), '--align', self.align.get()])
            self.root.destroy()
            return
        if not (self.merge_mode is not None
                and self.merge_mode.get() == IMAGE_MERGE):
            # Opening a comparison asks nothing. This is a window into two
            # documents, not into a commitment: the user may only be looking at
            # them, or trying the brush, and then deciding against the whole
            # thing. The merge window asks where to save when the user commits,
            # and until then it needs no name -- so no name is asked for
            # here and none is passed on.
            self.app.run_arguments(['merge', str(sources[0]), str(sources[1]),
                                    '--align', self.align.get()])
            self.root.destroy()
            return
        # The image merge opens no window: it compares, patches and writes.
        # There is nothing to look at first, so up front is the only time it
        # can ask, and the only time it should.
        output = filedialog.asksaveasfilename(parent=self.root, title='Save merged notebook',
            initialdir=sources[0].parent, initialfile=sources[0].stem + '_merged.xopp',
            defaultextension='.xopp', filetypes=[('Xournal++ notebook', '*.xopp')])
        if not output:
            return
        if Path(output).resolve() in sources or Path(output).suffix.lower() != '.xopp':
            messagebox.showerror('Choose a new notebook', 'Save to a separate .xopp file to preserve both originals.', parent=self.root)
            return
        if sources[0].suffix.lower() != '.xopp':
            messagebox.showerror(
                'Check merge sources',
                'The image-level merge keeps the structure of the left '
                'side, so the left side has to be the .xopp notebook. '
                'Swap the two sides, or use the stroke merge.',
                parent=self.root)
            return
        self.app.run_arguments(['merge', str(sources[0]), str(sources[1]), '-o', output, '--mode', 'image'])
        self.root.destroy()
