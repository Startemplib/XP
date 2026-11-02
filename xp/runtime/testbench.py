"""A bench for trying every part of XP by hand, on real documents.

The automated tests check that the pieces behave; they can't tell you whether
a traced page *looks* right, whether the brush feels right, or whether the
Explorer menu does what you expected. This lists every function XP has, the
real document to try it on, and what a good result looks like -- then runs it
for you and keeps your verdict.

The same list writes TESTING.md (`python -m xp testbench --write-doc`), so the
document and the bench can't drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from xp.render.appearance import ABYSS, decorate

from xp.runtime.paths import ROOT          # the checkout, found not counted
RESULTS = ROOT / 'tests' / 'manual-results.json'

# Real documents, each chosen because it is a different kind of hard.
SAMPLES = {
    'stylus': ('HamiL/HamiL_export.pdf', 'stylus notes, clean colors, 26 pages'),
    'mixed': ('Qpsi/AQ/Quantum notes.pdf', 'phone scan, blue/red/black ink that crosses'),
    'landscape': ('Astromlogy/Transfer/AS.pdf', 'landscape stylus notes in a tall profile'),
    'typed': ('Laperiod/1st/Tutorial/Tutorial 1.pdf', 'typed A4, fine punctuation'),
    'green': ('AMO/Atorbit/Tutorial/tutorial0_solution.pdf', 'scan on green paper, white margin'),
    'notebook': ('Astromlogy/AS.xopp', 'a real notebook, 24 MB'),
    'notebook2': ('Astromlogy/ASsm.xopp', 'the same notebook, edited later'),
}


@dataclass(frozen=True)
class Check:
    area: str
    name: str
    look_for: str
    sample: str = ''
    action: tuple = ()          # ('convert', mode) | ('cli', [...]) | ('manual',)
    extra: str = ''             # a second sample, for the two-sided ones


CHECKS = (
    Check('Convert', 'Auto-tune a page', 'Settings fill in by themselves -- including Denoise, '
          'Sharpen and the morphology, which the search chooses too -- and the status line says '
          'marks kept, ink lost, what it cleaned the page with first, and the boundary and '
          'spacing it settled on. On these notes: 0 marks lost, well under 1% of the ink.',
          'stylus', ('convert', 'auto')),
    Check('Convert', 'Auto-tune a hard page', 'The phone scan is where searching the whole '
          'parameter space earns its seconds. It should take tens of seconds, name the denoise '
          'and closing it picked, and lose visibly less ink than the probe\'s own guess does -- '
          'and it must never lose a mark. Switch Stroke shape to Onion-fill first, so the '
          'boundary and curve settings are searched too.', 'mixed', ('convert', 'auto')),
    Check('Convert', 'Live stages', 'Next stage > advances one stage per press through the '
          'nineteen (Original through Traced result), and each stage redraws after an edit. '
          'The stage list always shows all nineteen; a stage not yet computed is computed '
          'when you aim at it.', 'typed', ('convert', 'live')),
    Check('Convert', 'Difference overlay', 'Red marks ink the tracer lost, blue marks ink it '
          'invented; Show differences turns both off.', 'green', ('convert', 'auto')),
    Check('Convert', 'Synchronized zoom and pan', 'Wheel zooms about the cursor, drag pans, '
          'double-click fits -- and both panes always show the same patch of the page.',
          'landscape', ('convert', 'auto')),
    Check('Convert', 'Color split on crossing ink', 'Raise "Break on color change": pieces of a '
          'crossing keep their own pen, and "Break on color change" in the panel reads the '
          'same value it was set to.', 'mixed', ('convert', 'auto')),
    Check('Convert', 'Stroke shape', 'Switch Stroke shape through Centerline, Onion-fill and '
          'Outline: the traced pane should look more like the page with the first two, and '
          'hollow with the last. Ortho-fill is the default and is not in the list. Thick '
          'strokes must come out solid, thin ones must not fatten.', 'mixed', ('convert', 'auto')),
    Check('Convert', 'Margins and the blue frame', 'Convert a PDF exported with wide margins: '
          'the traced page comes back at notebook size, with the blue frame on the page edge '
          'where the template draws it, not shrunk into the middle.', 'stylus', ('convert', 'auto')),
    Check('Convert', 'Save and load settings', 'Save settings as JSON..., change a value, '
          'Load settings... puts it back.', 'stylus', ('convert', 'auto')),
    Check('Convert', 'Convert the document', 'Convert document... writes a .xopp that Xournal++ '
          'opens, with the strokes the preview showed.', 'stylus', ('convert', 'auto')),
    Check('Measure', 'Probe a page', 'Reports paper color, ink coverage and contrast, pen '
          'thickness, and the settings it would suggest.', 'stylus', ('cli', ['probe', '--pages', '1'])),
    Check('Compare', 'Read-only comparison', 'Opens two notebooks side by side, colored by which '
          'side each stroke is on; the brush does nothing here.', 'notebook', ('diff',), 'notebook2'),
    Check('Merge', 'Brush a decision', 'Pick Left/Right/Both (1/2/3) and drag: what the brush '
          'covered changes colour straight away -- green when the user confirmed ink only one side has, '
          'black when it is common ink that was kept -- and the view repaints with no white gouges.',
          'notebook', ('merge',), 'notebook2'),
    Check('Merge', 'Whole-page actions', 'All Left / All Right / All Both / Clear page take the '
          'whole page one way; the view matches what a fresh page load would draw. '
          '(Ctrl+1/2/3 are the source previews, below -- not whole-page decisions.)',
          'notebook', ('merge',), 'notebook2'),
    Check('Merge', 'Previews of the sources', 'Ctrl+1, Ctrl+2 and Ctrl+3 show the raw left page, '
          'the raw right page and both together, in their own colours. Going round them and back '
          'to Ctrl+1 draws exactly what Ctrl+1 drew the first time, and none of them changes '
          'anything. Esc returns to the editing colours.', 'notebook', ('merge',), 'notebook2'),
    Check('Merge', 'Final preview', 'Brush a few decisions, then press P (or Final preview): the '
          'page appears in its real colours, exactly as it would be exported -- no red, blue or '
          'grey. Esc goes back and the decisions are still there.', 'notebook', ('merge',), 'notebook2'),
    Check('Merge', 'Left brush clears the right', 'Brush Left over a place where both documents '
          'have ink: the left stroke stays and the right one goes, rather than staying visible on '
          'top of it.', 'notebook', ('merge',), 'notebook2'),
    Check('Merge', 'Undo and redo', 'Ctrl+Z / Ctrl+Y step back and forward, held down they repeat, '
          'and the depth is the whole session.', 'notebook', ('merge',), 'notebook2'),
    Check('Merge', 'Keep both versions', 'Ctrl+4 (or the Split into 2 pages button) splits the '
          'page straight away: the merge grows a page, the window moves to the left half, and '
          'the button becomes "Join back".',
          'notebook', ('merge',), 'notebook2'),
    Check('Merge', 'Page mapping', 'With Page matching set to manual, the two-column editor pairs '
          'pages by hand, and inserted pages stay unpaired.', 'notebook', ('merge',), 'notebook2'),
    Check('Merge', 'Re-map mid-merge', 'Press M during a merge: the page map opens, the new pairing '
          'takes effect on the next redraw, and decisions already made are still there.',
          'notebook', ('merge',), 'notebook2'),
    Check('Merge', 'Pair from either side', 'In the page map, click a left page then a right one, or '
          'a right page then a left one -- both make the same link.',
          'notebook', ('merge',), 'notebook2'),
    Check('Merge', 'Pages of different sizes', 'Merge a notebook with a PDF that is not Q base: '
          'the merged page comes out Q base (66 x 138 mm) and the PDF side is placed onto it so '
          'the two inks line up, rather than stretched to fit the page box.',
          'notebook', ('merge',), 'landscape'),
    Check('Merge', 'The fit is placed for you', 'The Fit row says where the other side sits. '
          'F (or the Fit button) measures the two documents\' own Q base frames again and places '
          'the other side from them, falling back to the ink when no frame is visible. There is '
          'no ratio/offset/nudge control and no "every page" checkbox. The two inks should sit '
          'on top of each other, and nothing should become decided by measuring.',
          'notebook', ('merge',), 'notebook2'),
    Check('Merge', 'Save the result', 'Ctrl+S writes a new notebook and leaves both originals '
          'untouched.', 'notebook', ('merge',), 'notebook2'),
    Check('Arrange', 'Arrange pages', 'Arranger opens in a window of its own -- "XP (arranger)" '
          'in the title bar and no console window behind it -- with the pages of both notebooks '
          'as cards. Reorder, duplicate or delete some, then Ctrl+S: it writes '
          '<first document>-ArraN.xopp beside that source and closes itself. The sources must be '
          'untouched and Xournal++ must open the result. XP itself must not have opened a window '
          'for any of this.', 'notebook', ('arrange',), 'notebook2'),
    Check('Shell', 'Explorer menu', 'Right-click a .pdf, then a .xopp: the XP submenu holds '
          'exactly three entries, spelled Converter, Merger and Arranger, and the same three '
          'on both. No fourth item, no "live stages", no "Compare", no "Open in XP" -- those '
          'moved to the console, the dashboard and the shortcuts. Count them. Windows 11 hides '
          'the menu under "Show more options".', '', ('manual',)),
    Check('Shell', 'The two that cannot work say so', 'Right-click a PDF > Arranger: one box '
          'saying it reorders .xopp pages and to convert first -- not an empty arranger, and '
          'no dashboard. Right-click a .xopp > Converter: one box saying it is already a '
          'notebook. Select three files and ask for the wrong one: still ONE box, not three.',
          '', ('manual',)),
    Check('Shell', 'Arrange several at once', 'Select three or more .xopp files, right-click, '
          'XP > Arrange pages...: **one** arranger opens with all of them. Explorer calls the '
          'verb once per file, so three windows would mean the collection is broken.',
          '', ('manual',)),
    Check('Shell', 'Two files at once', 'Select a .xopp and a .pdf, right-click, XP > Merge: both '
          'arrive in one window, notebook on the left.', '', ('manual',)),
    Check('Dashboard', 'Console commands', 'Type a command (e.g. probe "file.pdf" --pages 1) and '
          'its output appears inline, same syntax as the CLI.', '', ('manual',)),
)


def samples_root(override=None):
    """Where the real documents are. Asked for, never guessed from here --
    they stopped being XP's sibling when the tree was tidied."""
    if override:
        return Path(override)
    from xp.runtime.where import material
    return material()


def sample_path(key, root):
    name = SAMPLES.get(key, (None,))[0]
    return (root / name) if name else None


def write_doc(path=ROOT / 'TESTING.md', samples=None):
    """The same checks as a document, for testing away from the bench."""
    root = samples_root(samples)
    lines = ['# Testing XP by hand', '',
             'What the automated tests cannot judge: whether a traced page *looks* right, '
             'whether the brush feels right, whether the Explorer menu does what you expected.',
             '', f'Run the bench with `python -m xp testbench`. Documents come from `{root}`.', '',
             '| # | Area | Function | Document | What a good result looks like |',
             '|---|------|----------|----------|-------------------------------|']
    for number, check in enumerate(CHECKS, 1):
        document = SAMPLES.get(check.sample, ('-', ''))[0]
        if check.extra:
            document += ' + ' + SAMPLES[check.extra][0]
        lines.append(f'| {number} | {check.area} | {check.name} | `{document}` | {check.look_for} |')
    lines += ['', '## The documents, and why each one',
              '', '| Key | File | Why it is here |', '|-----|------|----------------|']
    for key, (name, why) in SAMPLES.items():
        lines.append(f'| {key} | `{name}` | {why} |')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


class TestBench:
    """The checklist as a window: run each function on a real document, and
    keep the verdict."""

    def __init__(self, app, samples=None):
        self.app = app
        self.root_dir = samples_root(samples)
        self.results = json.loads(RESULTS.read_text(encoding='utf-8')) if RESULTS.exists() else {}
        self.root = tk.Toplevel(app.root)
        self.root.title('XP / Test bench')
        decorate(self.root)
        from xp.runtime.shutdown import arm
        arm(self.root)
        self.verdicts = {}
        self.notes = {}
        self._build()
        # **After `_build`, not before.** Asked first, the window is sized while
        # it has no children, and the building that follows re-requests a size
        # from them -- the same order that left the guide at 695x667. The user's order
        # for the windows: maximised. The bench is a table of 31 rows; 1180x760
        # showed about half of it.
        from xp.runtime.shutdown import fill_the_screen

        fill_the_screen(self.root)
        self.root.minsize(1180, 760)

    def _build(self):
        header = ttk.Frame(self.root, style='Lab.TFrame', padding=14)
        header.pack(fill='x')
        ttk.Label(header, text='EVERY FUNCTION, ON REAL DOCUMENTS', style='Title.Lab.TLabel').pack(anchor='w')
        ttk.Label(header, text=f'Documents from {self.root_dir}. Run it, look at it, mark it.',
                  style='Muted.Lab.TLabel').pack(anchor='w')

        footer = ttk.Frame(self.root, style='Lab.TFrame', padding=10)
        footer.pack(side='bottom', fill='x')
        self.status = tk.StringVar(self.root, '')
        ttk.Label(footer, textvariable=self.status, style='Muted.Lab.TLabel').pack(side='left')
        ttk.Button(footer, text='Write TESTING.md', command=self._write_doc).pack(side='right', padx=4)
        ttk.Button(footer, text='Save verdicts', command=self._save).pack(side='right', padx=4)

        canvas = tk.Canvas(self.root, bg=ABYSS, highlightthickness=0)
        scroll = ttk.Scrollbar(self.root, orient='vertical', command=canvas.yview)
        body = ttk.Frame(canvas, style='Lab.TFrame', padding=(14, 0))
        body.bind('<Configure>', lambda _e: canvas.configure(scrollregion=canvas.bbox('all')))
        held = canvas.create_window((0, 0), window=body, anchor='nw')
        # **The rows take the window's width, not their own.** The inner frame
        # was as wide as its content asked (about 690 px), so on a maximised
        # window two thirds of the bench was empty canvas and every description
        # stayed in a narrow column.
        def the_width(event):
            canvas.itemconfigure(held, width=event.width)
            for label in getattr(self, '_detail_labels', []):
                try:
                    # 720, not 380: the radios and the note field live in the
                    # columns to the right of the description, and a wrap wider
                    # than the column's share is a sentence cut mid-word.
                    label.configure(wraplength=max(360, event.width - 720))
                except tk.TclError:
                    pass
        canvas.bind('<Configure>', the_width)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        canvas.bind_all('<MouseWheel>', lambda e: canvas.yview_scroll(-e.delta // 120, 'units'))

        area = None
        for number, check in enumerate(CHECKS, 1):
            if check.area != area:
                area = check.area
                ttk.Label(body, text=area.upper(), style='Muted.Lab.TLabel').pack(anchor='w', pady=(14, 4))
            self._row(body, number, check)

    def _row(self, parent, number, check):
        row = ttk.Frame(parent, style='Lab.TFrame')
        row.pack(fill='x', pady=3)
        row.columnconfigure(1, weight=1)
        missing = self._missing(check)
        ttk.Button(row, text='Run' if not missing else 'n/a', width=6,
                   command=lambda c=check: self._run(c),
                   state='disabled' if missing else 'normal').grid(row=0, column=0, rowspan=2, padx=(0, 10))
        ttk.Label(row, text=f'{number}. {check.name}', style='Lab.TLabel').grid(row=0, column=1, sticky='w')
        detail = check.look_for if not missing else f'{missing} is not in {self.root_dir}'
        told = ttk.Label(row, text=detail, style='Muted.Lab.TLabel', wraplength=620,
                         justify='left')
        told.grid(row=1, column=1, sticky='w')
        if not hasattr(self, '_detail_labels'):
            self._detail_labels = []
        self._detail_labels.append(told)
        key = f'{number}. {check.name}'
        verdict = tk.StringVar(self.root, self.results.get(key, {}).get('verdict', ''))
        self.verdicts[key] = verdict
        for column, (label, value) in enumerate((('pass', 'pass'), ('fail', 'fail'))):
            ttk.Radiobutton(row, text=label, value=value, variable=verdict,
                            style='Lab.TRadiobutton').grid(row=0, column=2 + column, padx=2)
        note = tk.StringVar(self.root, self.results.get(key, {}).get('note', ''))
        self.notes[key] = note
        ttk.Entry(row, textvariable=note, width=34).grid(row=1, column=2, columnspan=2, sticky='ew')

    def _missing(self, check):
        for key in (check.sample, check.extra):
            if key:
                path = sample_path(key, self.root_dir)
                if path is None or not path.exists():
                    return SAMPLES.get(key, ('?',))[0]
        return None

    def _run(self, check):
        first = sample_path(check.sample, self.root_dir)
        second = sample_path(check.extra, self.root_dir)
        kind = check.action[0] if check.action else 'manual'
        try:
            if kind == 'convert':
                self.app.open_conversion(first, mode=check.action[1])
            elif kind == 'cli':
                arguments = list(check.action[1])
                self.app.run_arguments([arguments[0], str(first), *arguments[1:]])
            elif kind in ('merge', 'diff'):
                from xp.workers.merger.gui.opengui import MergeFilesWindow
                MergeFilesWindow(self.app, [first, second], mode=kind)
            elif kind == 'arrange':
                # A separate program: the bench starts it the same way the
                # Explorer verb does and does not wait for it either.
                from xp.workers.arranger.bridge import launch_arranger
                launch_arranger([p for p in (first, second) if p])
            else:
                self.status.set(f'{check.name}: do this one in Explorer, then mark it here.')
                return
            self.status.set(f'Opened: {check.name}')
        except Exception as exc:                      # a bench that dies on a failure is no use
            self.status.set(f'{check.name} raised: {exc}')

    def _save(self):
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        self.results = {key: {'verdict': variable.get(), 'note': self.notes[key].get()}
                        for key, variable in self.verdicts.items()}
        RESULTS.write_text(json.dumps(self.results, indent=2), encoding='utf-8')
        done = sum(1 for v in self.results.values() if v['verdict'])
        failed = sum(1 for v in self.results.values() if v['verdict'] == 'fail')
        self.status.set(f'Saved {done} of {len(CHECKS)} verdicts ({failed} failing) to {RESULTS.name}')

    def _write_doc(self):
        self.status.set(f'Wrote {write_doc(samples=self.root_dir)}')


def run(samples=None):
    from xp.runtime.app import App
    app = App()
    app.root.after(50, lambda: TestBench(app, samples))
    app.run()
