"""How long each stage of a page took to **compute**, in pipeline order.

The user asked to see it: a line across the bottom of the panel, cut at every
stage, each piece as long as that stage took. The reason it is worth a module
of its own rather than a print is that it is the only honest answer to "why
is it slow" -- I have twice told the user a time that turned out to be a timing
taken while something else was running, and twice reported a cost from the
wrong part of the pipeline. A bar the user can look at while it runs does not have
that failure mode.

THREE RULES, ALL OF THEM FROM THE USER, AND ALL THREE ARE ABOUT WHAT IS *NOT* COUNTED.

**Only computation.** "Don't count the time I stay in one stage -- count the
time used for computation, not the time I am checking the preview." So the
clock is not a wall clock. It runs from the moment the previous stage's
payload was ready to the moment this one's is, and it is explicitly stopped
for anything that is not the pipeline: the preview picture a stage draws, and
the time between runs while the user is looking at one. There is no wall-clock
number anywhere in this module, because a wall-clock number is the thing the user
said not to show.

**Every stage stays.** "Do not shrink, omit the stages." The timeline is
built on the pipeline's own stage list, so a stage that has not run yet is
present and pending rather than absent. The bar can then be read as the whole
pipeline from the first page onward instead of growing a piece at a time.

**What was measured is kept, and what was never measured is not invented.**
Stepping through the stages one at a time runs the pipeline once per stage,
and the steps before the one the user aimed at come out of the cache computing
nothing. Throwing away their measured cost and drawing a zero would make the
final bar a record of the last run rather than of the page. So a real
measurement survives runs, and a cached delivery marks the stage as inherited
*without* overwriting what it cost when it did run.

And a cached delivery of a stage that has **never** run here records no cost
at all. It used to record the ~0 seconds that handing a picture over takes,
and keep it: a page the user stepped through reported 16.7s where the same page run
straight through reported 22.4s, and the user read the difference as the program
having got slower when it had in fact got three seconds faster. A stage
nothing has measured is unmeasured; it says so and it stays out of the total.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class Step:
    """One stage, and what computing it cost.

    `seconds` is None while nothing has measured the stage: a stage not yet
    measured and a stage that took no time are different things, and a bar
    that drew them the same would be lying about one of them. A stage can be
    `inherited` and unmeasured at once -- it ran, from the cache, and what it
    costs when it really runs is not known here.
    """
    name: str
    seconds: Optional[float] = None
    inherited: bool = False     # this run took it from the cache
    running: bool = False       # this is the one being computed now

    @property
    def measured(self) -> bool:
        return self.seconds is not None


class Timeline:
    """The stages of one page, in pipeline order, with what each one cost.

    Shared between the worker thread that runs the pipeline and the thread
    that draws the panel, so every read and write is under one lock and
    `steps()` hands back a copy. A bar drawn from a list that is being
    appended to while it is drawn is a crash waiting for a slow page.
    """

    __slots__ = ('_lock', '_order', '_cost', '_inherited', '_since',
                 '_running', '_this_run', 'overhead', 'noun', 'clears')

    def __init__(self, order: Sequence[str] = (), noun: str = 'stages'):
        self._lock = threading.Lock()
        self._order = tuple(order)
        # What its pieces are, in a word. The same bar draws the stages of a
        # page and the pages of a document, and "19 of 19 pages" would be a
        # lie about one of them.
        self.noun = noun
        self._cost = {}            # name -> seconds, what computing it took
        self._inherited = set()    # names this run did not recompute
        self._since = time.perf_counter()
        self._running = None       # the stage being computed, if any
        self._this_run = set()
        # **How many times the bar was cleared.** The user's order is that changing a
        # parameter reloads the bar; that is a behaviour, so it needs to be
        # observable. The timeline is `__slots__` and its `reset` cannot be
        # patched from a test (`'Timeline' object attribute 'reset' is
        # read-only`), so it counts its own clears instead.
        self.clears = 0
        self.overhead = 0.0

    # ---- the pipeline's side --------------------------------------------

    def reset(self, order: Sequence[str] = None) -> None:
        """A different page. Everything measured belonged to the old one."""
        with self._lock:
            self.clears += 1
            if order is not None:
                self._order = tuple(order)
            self._cost = {}
            self._inherited = set()
            self._this_run = set()
            self._running = None
            self._since = time.perf_counter()
            self.overhead = 0.0

    def begin(self, first: str = None) -> None:
        """A run starts.

        What was measured is kept -- stepping through the stages runs the
        pipeline once per stage and only the new stage computes anything, so
        clearing here would leave the final bar holding one real number. The
        clock itself restarts, which is what makes the time the user spent looking
        at the last stage uncounted.
        """
        with self._lock:
            self._inherited = set()
            self._this_run = set()
            self._since = time.perf_counter()
            self._running = first if first is not None else self._first_unseen()
            self.overhead = 0.0

    def _first_unseen(self):
        for name in self._order:
            if name not in self._cost:
                return name
        return self._order[0] if self._order else None

    def mark(self, name: str, inherited: bool = False) -> None:
        """A stage's payload is ready.

        A cached delivery does not overwrite a measurement: "this cost
        nothing *this time* because nothing it reads moved" is true, and
        "this stage is free" is not, and only the second one would survive
        into the final bar.
        """
        now = time.perf_counter()
        with self._lock:
            took = max(0.0, now - self._since)
            self._since = now
            if inherited:
                # Marked as taken from the cache, and **not** given a cost.
                # `setdefault(name, took)` recorded a cached delivery as the
                # ~0 seconds it takes to hand a picture over, and kept it for
                # ever -- so a page the user stepped through stage by stage, where
                # everything before the user's stage comes from the cache, reported
                # a total far smaller than the same page run straight
                # through. 16.7s against 22.4s on the user's own screen, and the user
                # read it as the program having got slower when it had got
                # three seconds faster.
                #
                # A stage nothing has ever measured is **unmeasured**. It
                # says so, and it is left out of the total, which is the only
                # answer that is not a number made up out of bookkeeping.
                self._inherited.add(name)
            elif name in self._this_run:
                # Phase one hands a step's pictures over a second time once
                # the step itself has finished. Same stage, not a second cost.
                pass
            else:
                self._cost[name] = took
                self._this_run.add(name)
                self._inherited.discard(name)
            self._running = self._next_after(name)

    def _next_after(self, name):
        if name not in self._order:
            return None
        at = self._order.index(name) + 1
        return self._order[at] if at < len(self._order) else None

    def finish(self) -> None:
        """The run is over. Nothing is being computed.

        Without this the bar keeps a piece growing on the end for ever, and
        that piece would be exactly the time the user spends looking at the stage
        the user stopped on -- the one number the user said not to count.
        """
        with self._lock:
            self._running = None

    def hold(self):
        """Start of something that is not the pipeline -- drawing a preview
        picture, say. Returns a token for `release`."""
        return time.perf_counter()

    def release(self, token) -> None:
        """End of it. The clock is moved forward by as much, so the stage
        being computed is charged for its own work and not for the panel's."""
        now = time.perf_counter()
        with self._lock:
            spent = max(0.0, now - token)
            self.overhead += spent
            self._since += spent

    # ---- the panel's side -----------------------------------------------

    def steps(self) -> List[Step]:
        """Every stage of the pipeline, in order, measured or pending."""
        with self._lock:
            now = time.perf_counter()
            out = []
            for name in self._order:
                if name == self._running:
                    out.append(Step(name, max(0.0, now - self._since),
                                    False, True))
                else:
                    out.append(Step(name, self._cost.get(name),
                                    name in self._inherited, False))
            return out

    def measured(self) -> List[Step]:
        return [s for s in self.steps() if s.measured]

    def total(self) -> float:
        """What computing this page has cost so far. Not a wall clock."""
        with self._lock:
            return sum(self._cost.values())

    def running(self) -> Optional[float]:
        with self._lock:
            if self._running is None:
                return None
            return max(0.0, time.perf_counter() - self._since)
