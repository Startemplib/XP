"""The line across the bottom of the panel, cut at every stage.

The user's instruction: *show the time consumed by each step as a line cut by
nodes, the length representing the time, in the bottom bar of the GUI* --
and then, having seen the first one: *do not shrink or omit the stages, I
want a faithfully true line bar indicating the time consumption of each
step.*

So: every stage of the pipeline is on the line, always, and a piece's length
is exactly its share of the computing time. No minimum width, because a
minimum width is a lie about proportion -- nineteen pieces at three pixels
each is six percent of the bar handed out for nothing, and the stages that
need it are precisely the ones that cost nothing. A stage too short to see is
*supposed* to be too short to see; its node is still there, and hovering it
still names it.

    Original      Medial axis          Orthogonal filling      Validation
       |  Probe        |                       |                    |
    o-o-o-o--o---------o-----------------------o--------------------o--o
        5ms            2.9s                    4.0s                 .7s
    previews 1.9s, not counted                           8.0s computing

**What a piece is.** The work that computed that stage. Not the preview
picture the panel drew of it afterwards, and not the time the user spent looking
at it -- the user said that plainly, so there is no wall-clock number on this
widget at all. `Timeline` holds the three apart.

**A pending stage auto-fits.** The user's words: *it is not begun, so the length
means nothing.* So a stage that has not run takes the share its existence
earns -- an equal slice of the bar -- and the stages that have run divide
what is left in proportion to their time. A fixed narrow slot was the first
answer and it was wrong in practice: eighteen un-run stages shared ninety
pixels of the whole bar, and the dashes were too short to see.

**A hollow piece is an inherited one.** A step whose settings did not move
was taken from the cache this run, and is drawn as an outline at the length
it cost when it *did* run -- which is the honest answer to "what does this
step cost on this page".

**The three phases are three colours**, in the panel's own palette.
"""

from __future__ import annotations

import tkinter as tk

from xp.render.appearance import ABYSS, INK, MUTED, PALE, PAPER, TEXT, font

# Which phase each stage belongs to. Kept here rather than imported so this
# module can be read on its own; the tests check it against STAGES.
PHASE_ONE = ('Original', 'Probe', 'Denoise', 'Sharpen', 'Threshold',
             'Morphology', 'Holes to ink', 'Specks', 'Perfect image')
PHASE_TWO = ('Geometry', 'Colour cloud', 'Colour clusters', 'Colours',
             'Region colour map', 'Region correction')
PHASE_THREE = ('Medial axis', 'Orthogonal filling', 'Output validation',
               'Traced result')

# **Three states, and that is all.** The user's convention for this widget:
#   azure  = done, and its length is what it cost;
#   deep blue (VIEWING) = the stage the user is looking at, drawn by `_viewing_mark`;
#   hollow + dotted = not done -- one state, not two.
# The three phase colours and their legend are gone: "the stage category I embedded,
# old bullshit". The phase lists stay because the tests hold them against STAGES.
AZURE = '#3BA7D6'
DONE_COLOUR = AZURE
COLOURS = {1: AZURE, 2: AZURE, 3: AZURE}      # kept for any caller that indexes it

# **The user's numbering, shown on the bar.** 2026-10-27 05:07: "HWEY SOAMMENY
# COLDOR IN THE BAR WHATTHEY MENS AND HWY SOE MANSY STAHE SHDOUE ITFOLRO MY
# SDEFSIN ATST EGS NUMEBR" -- why so many colours, what do they mean, and why
# so many stages when the user's design has seventeen.
#
# It does have seventeen. Nineteen rows appear because **stage 11 is three
# pictures**: what the clustering was handed, how it grouped it, and what
# colour each group came out as. Nothing was numbered on screen, so three
# views of one stage read as three stages. With the user's number in front of the
# name the arithmetic stops being a puzzle.
NUMBER = {
    'Original': 1, 'Probe': 2, 'Denoise': 3, 'Sharpen': 4, 'Threshold': 5,
    'Morphology': 6, 'Holes to ink': 7, 'Specks': 8, 'Perfect image': 9,
    'Geometry': 10,
    'Colour cloud': 11, 'Colour clusters': 11, 'Colours': 11,
    'Region colour map': 12, 'Region correction': 13,
    'Medial axis': 14, 'Orthogonal filling': 15, 'Output validation': 16,
    'Traced result': 17,
}
LAST_STAGE_NUMBER = 17


def numbered(name: str) -> str:
    """`11 Colour cloud`: the user's number in front of the name.

    A stage with no number shows as itself rather than disappearing -- a new
    stage someone forgot to number should look odd, not invisible.
    """
    number = NUMBER.get(name)
    return '%d %s' % (number, name) if number else name

# The rows are stacked from the font's own line height at build time, not
# written here as numbers. Both of the things the user saw -- overlapping text and
# a cropped bottom line -- were hard-coded rows that were right for one face
# and wrong for the next, and a constant that says how tall a line is will
# always eventually disagree with the line.
THICK = 7                  # a piece of the line
# **The stage the user is looking at, marked in dark blue.** The user's order
# (2026-10-26): "the progress bar is cut at every stage; mark the stage being
# viewed in dark blue, and it must move as soon as the user changes stage, without
# restarting any work." Dark blue rather than a phase colour, because it answers
# a different question from the pieces: they say what has been computed, it says
# where the user is standing.
VIEWING = '#123a6b'
NODE = 3                   # the radius of a cut
PENDING = 5                # a stage that has not run yet
GUTTER = 10
GAP = 3                    # between one row and the next
NAME_FONT = font(9)        # the user's face, from the one place that names it
HEIGHT = 84                # a first guess, replaced by the measured one as
                           # soon as the widget can ask Tk about the font


def phase_of(name: str) -> int:
    if name in PHASE_ONE:
        return 1
    if name in PHASE_TWO:
        return 2
    return 3


def _card_headroom():
    """What the card has free, in a few words, or nothing when there is no
    card.

    The user asked for this by name after a conversion died on a 5.02 GiB
    allocation: *"about that exceed VRAM, I guess it is good to set a buffer
    inside -- when it is too large, this can be shown to me when
    converting."* The buffer is `gpu.room`; this is the showing.
    """
    try:
        from xp.gpu import backend as gpu
        if not gpu.GPU_AVAILABLE:
            return ''
        free, total = gpu.xp.cuda.runtime.memGetInfo()
        GiB = 1024.0 ** 3
        return 'card %.1f of %.1f GiB free' % (free / GiB, total / GiB)
    except Exception:
        return ''


def _short(seconds: float) -> str:
    if seconds is None:
        return '-'
    if seconds >= 100.0:
        return '%.0fs' % seconds
    if seconds >= 1.0:
        return '%.1fs' % seconds
    if seconds >= 0.0005:
        return '%dms' % round(seconds * 1000)
    return '0'


class TimingBar(tk.Canvas):
    """Draws a `Timeline`. Reads it, never writes it.

    The widget polls rather than being pushed to: the pipeline runs on a
    worker thread, and a thread that paints a Tk canvas is a crash on
    Windows. Polling at 10 Hz is invisible next to a page that takes seconds.
    """

    def __init__(self, parent, clock, interval=100, **kw):
        super().__init__(parent, height=HEIGHT, background=ABYSS,
                         highlightthickness=0, bd=0, **kw)
        # Either a `Timeline` or something callable that returns the one that
        # is working now. The panel has two -- the stages of a page and the
        # pages of a document -- and a converting panel should be showing the
        # conversion, not the last page the user happened to look at.
        self._source = clock
        self.clock = clock if not callable(clock) else clock()
        self.interval = interval
        self._pieces = []          # (x0, x1, Step) for the hover readout
        # By NAME, not by Step: `steps()` builds fresh objects every redraw,
        # so an identity held across one is never the same object again --
        # which is why hovering a stage that had not run showed nothing.
        self._hover = None
        # The stage on screen, by name (the steps are rebuilt every redraw).
        self.viewing = None
        self._measure_the_face()
        # The pending tick is cancelled when the widget goes, or Tk tries to
        # invoke a callback on a destroyed interpreter and prints
        # `invalid command name "..._tick"` into the console. `redraw` already
        # catches a TclError; this is the one that happens before it is even
        # reached.
        self._waiting = None
        self.bind('<Destroy>', self._stop, add='+')
        self.bind('<Configure>', lambda _e: self.redraw())
        self.bind('<Motion>', self._moved)
        self.bind('<Leave>', lambda _e: self._moved(None))
        self._tick()

    # ---- the layout, from the font ---------------------------------------

    def _measure_the_face(self):
        """Stack the rows on the real line height, and keep the ruler.

        Two name rows above the line, the seconds below it, the notes under
        those. Each row is a full line apart, so two labels on neighbouring
        rows cannot touch however tall the face turns out to be -- which is
        the whole of what went wrong when the face changed.
        """
        import tkinter.font as tkfont

        self._ruler = tkfont.Font(font=NAME_FONT)
        line = self._ruler.metrics('linespace')
        self.name_high = GAP + line // 2
        self.name_low = self.name_high + line + GAP
        self.line_y = self.name_low + line // 2 + GAP + THICK
        self.time_y = self.line_y + THICK // 2 + GAP
        self.note_y = self.time_y + line + GAP
        # **One more line for the key.** The user asked what the colours mean, and the
        # honest answer is that nothing on screen said: this row says it once,
        # where the colours are.
        # Two lines under the notes, not one: the notes are drawn anchored at
        # their own row and reach a full line below its top, so one line of space
        # overlapped them by four pixels (`test_no_two_pieces_of_writing_overlap`
        # measured it).
        self.legend_y = self.note_y + 2 * line
        height = self.legend_y + line + GAP
        self.configure(height=height)
        global HEIGHT
        HEIGHT = height

    def wide(self, text):
        """How wide this string really is in the panel's face."""
        return self._ruler.measure(text)

    # ---- the loop --------------------------------------------------------

    def _tick(self):
        try:
            self.redraw()
        except tk.TclError:
            return                 # the window went away
        self._waiting = self.after(self.interval, self._tick)

    def _stop(self, event=None):
        if event is not None and event.widget is not self:
            return                 # a child's Destroy, not ours
        if self._waiting is not None:
            try:
                self.after_cancel(self._waiting)
            except tk.TclError:
                pass
            self._waiting = None

    # ---- which stage is on screen ---------------------------------------

    def mark_viewing(self, name) -> None:
        """Say which stage is on screen, and move the dark blue mark there.

        **A view change, not a computation.** Nothing here touches the pipeline:
        the mark is moved on the canvas, so stepping from one stage to another
        costs a redraw whether or not a run is in flight -- which is what the user
        asked for: "it moves instantly, and it does not restart the work."
        """
        if name == self.viewing:
            return
        self.viewing = name
        self.redraw()

    # ---- drawing ---------------------------------------------------------

    def redraw(self):
        # **The clock is asked for every redraw when it is a callable.** The constructor
        # froze it once (`self.clock = clock()`), so the bar kept drawing the Timeline it
        # was born with: during a conversion it showed the same "653ms computing / 4 of 19
        # stages" for the whole run, which is exactly what the user photographed and called
        # "the bar isn't moving". The panel hands in a callable precisely because the
        # working Timeline changes with every stage.
        if callable(self._source):
            self.clock = self._source()
        self.delete('all')
        self._pieces = []
        width = max(1, int(self.winfo_width()))
        self._draw_the_key()
        steps = self.clock.steps()
        if not steps or not any(s.measured or s.running for s in steps):
            # **The words go above the line, not on it.** They used to sit
            # on `line_y` because there was nothing else there; now every
            # stage is drawn as a dash across that row, so text on the same
            # row has the line struck through it. Measured on the user's screen:
            # "nothing measured yet" with dashes running through every letter.
            self.create_text(GUTTER, self.name_low, anchor='w', fill=MUTED,
                             font=NAME_FONT,
                             # **"measured", not "run".** A live preview draws a
                             # stage without recording a cost, so the old words
                             # ("no page has run yet") sat under a page that was
                             # plainly on screen.
                             text='nothing measured yet')
            # **And the stages are drawn, equally.** Nothing has been
            # measured, so no stage has a length to show -- which is the
            # reason to give them all the same one and fill the bar, rather
            # than a row of five-pixel ticks nobody can see or hover. The
            # mark is still placed: "where I am" does not depend on anything
            # having been computed yet.
            span = max(1.0, (width - 2 * GUTTER) / max(1, len(steps)))
            x = float(GUTTER)
            self._node(x, True)
            for step in steps:
                self.create_line(x, self.line_y, x + span, self.line_y,
                                 fill=DONE_COLOUR, dash=(1, 2))
                self._pieces.append((x, x + span, step))
                if self.viewing is not None and step.name == self.viewing:
                    self._viewing_mark(x, x + span)
                x += span
                self._node(x, False)
            return
        # **The stage in flight is not counted into the proportion.** It has
        # no finished time -- only however much of it has elapsed so far --
        # so letting it into `total` made the stages that *had* finished
        # share whatever was left of the bar after it, which on a fast first
        # two stages was a hundredth of a pixel each.
        total = sum(s.seconds for s in steps
                    if s.seconds is not None and not s.running)
        if total <= 0.0:
            total = 1e-9
        # Pending stages take a fixed slot each; everything else shares what
        # is left, exactly in proportion. No minimum: a piece's length is its
        # time, and that is the whole promise of the bar.
        # **Too little measured to be proportional?** Then show equal slots.
        # The user's picture: one stage measured mid-run gave that stage 100 % of the width -- a single
        # huge piece and a tail of dots, which reads as "the bar is broken / only one stage
        # exists". Proportional length is the right rule once there is something to be
        # proportional to; below three measured stages it is a claim the numbers cannot support,
        # so every stage gets the same slot and the done ones are filled with azure.
        done = [s for s in steps if s.measured or s.running]
        if len(done) < 3:
            even = max(PENDING, (width - 2 * GUTTER) / max(1, len(steps)))
            x = float(GUTTER)
            self._node(x, True)
            for step in steps:
                if step.running or step.measured:
                    self._piece(x, even, step)
                elif step.running and not step.measured:
                    self._piece(x, even, step)
                else:
                    # The dash fills its slot: "it is not begun, so the
                    # length means nothing" -- a line drawn across the middle
                    # 60 % of the slot is shorter than the stage it stands for
                    # for no reason anyone can read off it.
                    self.create_line(x, self.line_y, x + even,
                                     self.line_y, fill=DONE_COLOUR, dash=(1, 2))
                self._pieces.append((x, x + even, step))
                if self.viewing is not None and step.name == self.viewing:
                    self._viewing_mark(x, x + even)
                x += even
                self._node(min(x, width - GUTTER), step.measured or step.running)
            self._labels(steps)
            self._notes(steps, width)
            if self._hover is not None:
                self._readout(width)
            return
        # **Running counts as waiting here.** A stage in flight has no length
        # to show either, so it takes an equal slice like the ones that have
        # not started; otherwise it was given a slot that came out of nobody's
        # allocation and the bar stopped short of its own end (measured: 704
        # of 880 px).
        waiting = sum(1 for s in steps if s.running or not s.measured)
        # **A stage that has not run auto-fits; it has no length to show.**
        # The user's words: "it is not begun, so the length means nothing" -- and a
        # fixed `PENDING` slot made eighteen un-run stages share ninety pixels
        # of the whole bar, so the dashes were too short to see. Each one now
        # takes the share its existence earns, an equal slice of the bar, and
        # the stages that *have* run divide what is left in proportion to
        # their time. Their ratios are untouched by this: they are all scaled
        # by the same `span`, which is what "the length is exactly the time"
        # has always meant.
        usable = max(1.0, width - 2 * GUTTER)
        count = max(1, len(steps))
        pending_each = (usable / count) if waiting else 0.0
        span = max(1.0, usable - waiting * pending_each)
        x = float(GUTTER)
        self._node(x, True)
        for step in steps:
            if step.running:
                # **`running` wins over `measured`.** The stage that is in flight is also
                # counted as measured (the panel marks it when it starts), so
                # `running and not measured` never matched the one stage that needed it:
                # sampled, `running=['Original']` and `measured=1` at the same time.
                # **The stage that is running has no time yet, so it had no width.**
                # `wide = span * (seconds / total)` with `seconds == 0` is a zero-width
                # piece, which is why the bar looked empty for the whole first stage while
                # the numbers beside it climbed (measured: "0 computing / 1 of 19 / running
                # 870ms ... 2.7s"). It gets an equal slot now, with a marker that moves
                # across it, so the bar itself shows that work is happening.
                import time as _time

                wide = pending_each or max(1.0, span / max(len(steps), 1))
                self._piece(x, wide, step)
                frac = (_time.monotonic() % 1.2) / 1.2
                marker = x + wide * frac
                self.create_line(marker, self.line_y - 3, marker,
                                 self.line_y + THICK + 3, fill=PAPER, width=1)
            elif step.measured:
                wide = span * (step.seconds / total)
                self._piece(x, wide, step)
            else:
                # **Not done, whether it never ran or has no measurement.** One shape,
                # one colour: dotted, hollow. The user's words: hollow and dotted are the same
                # state, so drawing them differently was two answers to one question.
                wide = pending_each
                self.create_line(x, self.line_y, x + wide, self.line_y,
                                 fill=DONE_COLOUR, dash=(1, 2))
            self._pieces.append((x, x + wide, step))
            if self.viewing is not None and step.name == self.viewing:
                self._viewing_mark(x, x + wide)
            x += wide
            # **Never past the right edge.** The last node used to be placed at the raw
            # `x` and could land outside the canvas, on top of the memory text the user
            # photographed. Clamped here, so the bar ends where the bar ends.
            self._node(min(x, width - GUTTER), step.measured or step.running)
        self._labels(steps)

    def _draw_the_key(self) -> None:
        """What the colours and shapes mean, in the bar's own words.

        Three phase colours, a dark-blue marker for where the user is standing, hollow
        for a piece that came out of the cache, dotted for a stage that has not
        run. None of it was written anywhere, which is why the user had to ask.

        **And the phases are drawn in their own colours, not merely named.**
        Writing "phase 1 · phase 2 · phase 3" in grey answers *how many*
        there are and not *which is which*, so a key that said the words was
        still no answer to "what they mens". Each phase gets a swatch in the
        colour the pieces are actually drawn in, with what it does beside it.
        """
        x = GUTTER
        # From the face's own line height, like every other row in here: a
        # swatch written as a pixel count is right for one font and wrong for
        # the next.
        swatch = max(6, self._ruler.metrics('linespace') - 6)
        top = self.legend_y - swatch / 2.0
        # done: azure and solid
        self.create_rectangle(x, top, x + swatch, top + swatch,
                              outline='', fill=DONE_COLOUR, width=1, tags=('legend',))
        x += swatch + 4
        text = 'done, and its length is what it cost'
        self.create_text(x, self.legend_y, anchor='w', fill=MUTED,
                         font=NAME_FONT, text=text, tags=('legend',))
        x += self.wide(text) + 14
        # not done: hollow and dotted
        self.create_rectangle(x, top, x + swatch, top + swatch,
                              outline=DONE_COLOUR, fill='', width=1, dash=(2, 2),
                              tags=('legend',))
        x += swatch + 4
        text = 'not done'
        self.create_text(x, self.legend_y, anchor='w', fill=MUTED,
                         font=NAME_FONT, text=text, tags=('legend',))
        x += self.wide(text) + 14
        # the mark over the stage on screen
        self.create_polygon(x + swatch / 2 - 4, top, x + swatch / 2 + 4, top,
                            x + swatch / 2, top + 5, fill=VIEWING, outline=VIEWING,
                            tags=('legend',))
        x += swatch + 4
        text = 'the stage you are on'
        self.create_text(x, self.legend_y, anchor='w', fill=MUTED,
                         font=NAME_FONT, text=text, tags=('legend',))
        x += self.wide(text) + 14
        # **"17 stages" only when the pieces *are* stages**, and no phase names: the
        # categories are gone.
        if getattr(self.clock, 'noun', 'stages') == 'stages':
            self.create_text(x, self.legend_y, anchor='w', fill=MUTED, font=NAME_FONT,
                             tags=('legend',),
                             text='|  %d stages, 11 has three pictures' % LAST_STAGE_NUMBER)

    def _viewing_mark(self, x0: float, x1: float) -> None:
        """The dark blue mark over the stage on screen."""
        middle = (x0 + x1) / 2.0
        top = self.line_y - THICK - 9
        self.create_line(middle, top, middle, self.line_y + THICK + 9,
                         fill=VIEWING, width=2)
        self.create_polygon(middle - 4, top, middle + 4, top, middle, top + 4,
                            fill=VIEWING, outline=VIEWING)

    def _node(self, x, solid):
        self.create_oval(x - NODE, self.line_y - NODE, x + NODE, self.line_y + NODE,
                         outline=MUTED if solid else '#3a4651',
                         fill=ABYSS, width=1)

    def _piece(self, x, wide, step):
        """**Done is azure and solid; not done is hollow and dotted.** Two shapes only,
        whatever the stage does -- the user's convention for this bar, replacing the three phase
        colours: "the stage category that I embed, old bullshit". A stage that is in
        flight is *not* done, so it is the hollow one."""
        if step.running or not step.measured:
            self.create_rectangle(x, self.line_y - THICK / 2, x + wide,
                                  self.line_y + THICK / 2, outline=DONE_COLOUR,
                                  fill='', width=1, dash=(2, 2))
            if step.running:
                # **A stage in flight has to be seen to move.** Without this the bar is a still
                # picture for as long as the longest stage runs -- the user's "the bar isn't moving",
                # when what was moving was only the small number at the right. The marker walks
                # across the running stage's slot and is tagged, so a test can read its x.
                import time as _time

                sweep = max(1.0, wide)
                marker = x + sweep * ((_time.monotonic() % 1.2) / 1.2)
                self.create_line(marker, self.line_y - THICK - 3, marker,
                                 self.line_y + THICK + 3, fill=PAPER, width=2,
                                 tags=('running-marker',))
        else:
            self.create_rectangle(x, self.line_y - THICK / 2, x + wide,
                                  self.line_y + THICK / 2, outline='', fill=DONE_COLOUR)

    # ---- the writing -----------------------------------------------------

    def _labels(self, steps):
        """Names above the line in two rows, seconds below, nothing touching.

        Nineteen stages in a footer: most pieces are narrower than their own
        name, so a name is placed only where it fits without reaching into
        what is already written. Two rows, because one row fits about six
        names and two fit about eleven. What is left is on hover, which is
        why every piece keeps a hit area whether or not it is wide enough to
        see.
        """
        width = max(1, int(self.winfo_width()))
        taken = {self.name_high: GUTTER - 4, self.name_low: GUTTER - 4}
        for x0, x1, step in self._pieces:
            mid = (x0 + x1) / 2.0
            if step.name == self._hover:
                continue           # drawn last, over everything
            if not step.measured and not step.running:
                continue
            label = numbered(step.name)
            want = self.wide(label)
            for row in (self.name_high, self.name_low):
                left, right = mid - want / 2.0, mid + want / 2.0
                if left >= taken[row] and right <= width - GUTTER:
                    self.create_text(mid, row, anchor='center',
                                     fill=MUTED if step.inherited else TEXT,
                                     font=NAME_FONT, text=label)
                    taken[row] = right + 8
                    if row == self.name_low:
                        # A tick down to its own piece, or the second row is
                        # a list of names rather than labels on a line.
                        self.create_line(mid, row + 5, mid, self.line_y - 5,
                                         fill='#3a4651')
                    break
        # Seconds, under the pieces wide enough to hold them.
        last = GUTTER - 4
        for x0, x1, step in self._pieces:
            if not step.measured or step.inherited:
                continue
            text = _short(step.seconds)
            want = self.wide(text)
            mid = (x0 + x1) / 2.0
            if mid - want / 2.0 >= last and mid + want / 2.0 <= width - GUTTER:
                self.create_text(mid, self.time_y, anchor='n', fill=MUTED,
                                 font=NAME_FONT, text=text)
                last = mid + want / 2.0 + 6
        self._notes(steps, width)
        if self._hover is not None:
            self._readout(width)

    def _notes(self, steps, width):
        """The two numbers that are about the page rather than about a stage.

        No wall clock. The user said not to count the time the user spends looking at a
        stage, and a wall clock is exactly that number, so it is not here --
        what is here is what computing the page has cost, and, separately,
        what the panel spent drawing its own previews, which is real time the user
        waited and no stage did.
        """
        done = sum(1 for s in steps if s.measured)
        tail = '%s computing / %d of %d %s' % (
            _short(self.clock.total()), done, len(steps), self.clock.noun)
        running = self.clock.running()
        if running is not None:
            tail = '%s / running %s' % (tail, _short(running))
        headroom = _card_headroom()
        if headroom:
            tail = '%s / %s' % (tail, headroom)
        right = self.create_text(width - GUTTER, self.note_y, anchor='ne',
                                 fill=PAPER, font=NAME_FONT, text=tail)
        if self.clock.overhead <= 0.0:
            return
        note = 'previews %s, not counted' % _short(self.clock.overhead)
        # Only if it cannot reach what is on the right. Two notes that touch
        # is what the user saw the first time.
        if GUTTER + self.wide(note) + 12 < self.bbox(right)[0]:
            self.create_text(GUTTER, self.note_y, anchor='nw', fill=MUTED,
                             font=NAME_FONT, text=note)

    def _readout(self, width):
        """The hovered stage, written over everything and clear of the edge."""
        held = [(x0, x1, s) for x0, x1, s in self._pieces
                if s.name == self._hover]
        if not held:
            return
        x0, x1, step = held[0]
        if step.measured:
            said = _short(step.seconds)
            if step.inherited:
                said += '   (from the cache this time)'
        elif step.inherited:
            said = 'ran from the cache; never measured here'
        else:
            said = 'not run yet'
        text = '%s   %s' % (step.name, said)
        want = self.wide(text)
        mid = (x0 + x1) / 2.0
        mid = min(max(mid, GUTTER + want / 2.0), width - GUTTER - want / 2.0)
        box = (mid - want / 2.0 - 4, self.name_high - 7,
               mid + want / 2.0 + 4, self.name_high + 7)
        self.create_rectangle(*box, fill=ABYSS, outline='#3a4651')
        self.create_text(mid, self.name_high, anchor='center', fill=TEXT,
                         font=font(9, 'bold'), text=text)

    # ---- hover -----------------------------------------------------------

    def _moved(self, event):
        """Which piece is under the pointer.

        A faithful bar has pieces a fraction of a pixel wide, so the hit area
        is widened to something a hand can hit: the nearest piece within a
        few pixels, which is the only way the names of the short stages are
        reachable at all.
        """
        was = self._hover
        self._hover = None
        if event is not None and self._pieces:
            best, how_far = None, 1e9
            for x0, x1, step in self._pieces:
                if x0 <= event.x <= x1:
                    best, how_far = step, 0.0
                    break
                gap = x0 - event.x if event.x < x0 else event.x - x1
                if gap < how_far:
                    best, how_far = step, gap
            if how_far <= 4.0:
                self._hover = best.name
        if self._hover != was:
            self.redraw()
