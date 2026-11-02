"""A global search over a handful of settings, each on its own ladder.

What this is for: choosing a page's conversion settings is a global search
over a space that is small but badly behaved. Denoise interacts with the
threshold window, an opening undoes what a lower offset bought, and the
objective is full of plateaux with cliffs between them -- the point where a
setting starts tracing the paper's grain instead of the ink is a cliff, and
on either side of it nothing changes at all. A sweep of the whole grid is
out: five settings with half a dozen rungs each is thousands of tracings of
an 800-DPI page. A greedy walk stops at the first plateau it reaches.

So: simulated annealing, with a momentum term on the proposals.

The annealing is what gets out of a local basin -- a worse state is accepted
with probability exp(-dE / T), T comes down geometrically, and when the walk
has stopped improving it restarts, alternately back at the best state it has
and at a state drawn from anywhere in the space (basin hopping, which is what
finds a good basin that is nowhere near the one the walk began in).

The momentum is what makes it efficient rather than merely correct. These
axes are largely monotone over long runs -- more threshold offset means more
ink, right up to the cliff where it starts tracing the grain -- and a random
walk crosses a run of n rungs in about n^2 steps. So each axis carries a
signed stride: the step that last worked on it, repeated and lengthened by
one every time it keeps working, and dropped to nothing the moment a move is
refused or the ladder ends. Measured on a 60x60 monotone ladder from one
corner, twelve seeds each, this is not a decoration:

    budget 40   blind 0/12 reached the far corner, momentum  3/12
    budget 60    "    2/12                           "       7/12
    budget 100   "    5/12                           "       9/12

and the mean energy is better at every budget. On a deceptive landscape --
a wide shallow basin next to a narrow deep one -- momentum is neutral
(9/12 both ways at budget 80) and slightly behind on a tight budget, because
coasting can carry the walk past a good basin that was next door. Both are
in tests/test_anneal.py, both as measurements rather than assertions.

**Why not gradient descent, which is the other thing you would reach for.**
The objective is defined on a lattice: each setting is one of a handful of
discrete rungs, chosen because those are the values that mean anything (a
threshold offset is an integer number of grey levels; a morphology kernel is
an odd number of pixels). On a lattice the objective is piecewise constant,
so its gradient is zero wherever it is defined and undefined on the steps
between. A finite-difference "gradient" is then nothing but a two-sided
probe of every axis -- 2N evaluations to decide one move, where annealing
decides a move on one. The momentum part of momentum gradient descent is the
part that carries over, and it is here; the gradient part has nothing to
differentiate.

The search is deterministic given its seed, memoizes every state it has
evaluated (so revisiting one is free and does not count against the budget),
and takes a cancel event and a progress callback because it runs behind a
GUI on pages that take seconds each.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import concurrent.futures
import itertools
import math
import os
import random
import threading
import time


@dataclass(frozen=True)
class Axis:
    """One setting and the rungs it may take, in order.

    Ordered, because the whole search leans on it: a neighbour is a step
    along the ladder, and momentum is a direction along it. Put the rungs in
    the order the setting means something in -- least denoise to most -- not
    in the order they were thought of.
    """
    name: str
    rungs: tuple

    def __len__(self):
        return len(self.rungs)


class Space:
    """The axes, and the translation between a state and real settings.

    A state is a tuple of rung indices, one per axis, which is what the
    search moves around in; `settings` turns one into the numbers the caller
    actually uses. Indices rather than values because a step of one rung is
    the natural move whatever the values are -- the rungs of a threshold
    window are 7, 13, 33, 57, 101, and a search that moved in grey levels
    would spend its whole budget between 7 and 13.
    """

    def __init__(self, axes):
        self.axes = tuple(axes)

    def __len__(self):
        return len(self.axes)

    @property
    def size(self):
        total = 1
        for axis in self.axes:
            total *= len(axis)
        return total

    def settings(self, state) -> dict:
        return {axis.name: axis.rungs[index] for axis, index in zip(self.axes, state)}

    def nearest(self, **values) -> tuple:
        """The state whose rungs are closest to the values given.

        For starting from the settings a page already has, which need not be
        on any ladder -- a profile someone tuned by hand can say 43 where the
        ladder says 33 or 57.
        """
        state = []
        for axis in self.axes:
            if axis.name not in values:
                state.append(0)
                continue
            target = values[axis.name]
            best = min(range(len(axis.rungs)),
                       key=lambda i: abs(float(axis.rungs[i]) - float(target)))
            state.append(best)
        return tuple(state)

    def clamp(self, state) -> tuple:
        return tuple(max(0, min(len(axis) - 1, index))
                     for axis, index in zip(self.axes, state))


@dataclass
class Result:
    state: tuple
    energy: float
    settings: dict
    evaluations: int
    seconds: float
    accepted: int = 0
    restarts: int = 0
    best_at: int = 0                     # which evaluation found the winner
    history: list = field(repr=False, default_factory=list)   # (evaluation, energy, best)

    def summary(self):
        return (f'{self.evaluations} settings tried in {self.seconds:.1f}s, '
                f'best at #{self.best_at} (energy {self.energy:.6g})')


class Cancelled(Exception):
    pass


# The reach of a *random* proposal, at full temperature. Beyond about three
# rungs of a five-rung ladder a proposal is a jump to the other end, which is
# what the restarts are for rather than the proposals.
MAX_STEP = 3
MOMENTUM_BIAS = 0.75      # share of proposals that follow the axis's own velocity
MAX_STRIDE = 8            # longest stride momentum may build up to, in rungs
COOLING = 0.93            # T *= this each step; ~30 steps to a tenth
RESTART_AFTER = 24        # steps with no improvement before reheating from the best
SCALE_SAMPLES = 6         # states scored before the walk, to size the temperature
# Sweeps of the axes held back out of the budget for the polish. Enough for it
# to check every axis on its own once or twice and no more -- a *share* of the
# budget instead took the exploration the walk needs on a landscape whose good
# basin is nowhere near the start (a fifth off the budget took the deceptive
# case in tests/test_anneal.py from 9 of 12 seeds to 3).
POLISH_SWEEPS = 2
# The starting temperature, as a share of the spread of those samples. At a
# quarter, a proposal that costs the whole spread is taken about one time in
# fifty and one costing a fifth of it about two times in five -- which is the
# behaviour wanted: the barrier between two basins is crossable, the floor of
# the good one is not left for nothing.
START_HEAT = 0.25

# How many states may be scored at once. One candidate's energy is a page of
# numpy and OpenCV, and both release the GIL, so threads here are real
# parallelism rather than bookkeeping. Only the parts of the search that have
# no order to them use it -- the seed samples, the polish sweeps, and an
# exhaustive pass -- because the walk's order is the algorithm.
WORKERS = max(1, min(8, os.cpu_count() or 1))

# A space with no more states than this, and no more than the budget, is
# measured whole rather than walked. Below this size the walk is strictly
# worse than looking: it can spend forty evaluations on five states and still
# not have tried one of them.
EXHAUSTIVE_LIMIT = 64


def search(space: Space, energy, start=None, evaluations: int = 200,
           seconds: float = None, temperature: float = None, seed: int = 0,
           progress=None, cancel=None, extra_starts=(), workers=None) -> Result:
    """Anneal `energy` over `space`, and return the best state found.

    `energy` takes a settings dict and returns a float to be made small; it
    may return `math.inf` for a setting that cannot be used at all. `start`
    is a state (not settings) to begin from -- the page's current settings,
    so the search can never end up somewhere that measures worse than what
    was already there. `extra_starts` are further states to score before the
    walk begins: a spread over the space, so it does not start cold.

    `temperature` defaults to a quarter of the spread of the seed energies
    (see START_HEAT). Passing it explicitly skips the scale samples, and is
    for tests that want a known schedule.
    """
    rng = random.Random(seed)
    workers = WORKERS if workers is None else max(1, int(workers))
    began = time.perf_counter()
    cache = {}
    count = [0]
    steps = [0]
    # A proposal that lands on a state already measured costs nothing, so it
    # does not spend the budget -- but it does spend a step, and a walk whose
    # whole neighbourhood is cached would otherwise never stop. The bound is
    # generous: it only bites once the search has run out of anywhere to go.
    step_limit = max(64, evaluations * 20)

    # The walk stops a little early so the polish after it has something to
    # spend: that is where each axis is finally checked on its own, and it is
    # the cheapest part of the search. Leaving it nothing is how a good state
    # ends up one rung off on three axes -- on a six-axis bridge search with
    # forty evaluations the walk took all forty, the polish never ran, and
    # the answer kept a simplification of zero that one sweep would have
    # corrected (four times the points for half a point of ink). What is held
    # back is what the polish needs, not a share of the budget.
    reserve = min(POLISH_SWEEPS * len(space), max(0, evaluations // 4))
    walk_evaluations = max(1, evaluations - reserve)

    def out_of_time():
        return seconds is not None and time.perf_counter() - began >= seconds

    def spent():
        """Out of evaluations or out of time -- the two the caller set."""
        return count[0] >= evaluations or out_of_time()

    def over_budget():
        """The same, for the walk, which stops early to leave the polish room."""
        return count[0] >= walk_evaluations or out_of_time() or steps[0] >= step_limit

    guard = threading.Lock()

    def measure(state):
        state = space.clamp(state)
        with guard:
            if state in cache:
                return state, cache[state]
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        # Outside the lock, because this is the expensive part and holding a
        # lock across it would turn the pool below into a queue. Two threads
        # racing on the same state measure it twice, which is wasteful and
        # harmless; the cache keeps whichever arrived first, so the answer
        # does not depend on the race.
        value = float(energy(space.settings(state)))
        with guard:
            if state not in cache:
                cache[state] = value
                count[0] += 1
                if progress is not None:
                    progress(count[0], evaluations)
            value = cache[state]
        return state, value

    def measure_many(states):
        """Score independent states together, returning them in order.

        Deterministic: which states are scored is decided before any of them
        is measured, and the results come back in the order asked for, so
        which one wins never depends on which thread finished first.

        The batch is trimmed to what the caller's budget still allows, minus
        what is already in the cache, so a budget of one evaluation buys one.
        A batch is the one place a budget can be overrun without noticing --
        the serial loops it replaced checked between every state -- so it is
        checked here instead, once, before anything is launched.
        """
        room = 0 if out_of_time() else max(0, evaluations - count[0])
        allowed = []
        for state in states:
            clamped = space.clamp(state)
            with guard:
                known = clamped in cache
            if known:
                allowed.append(state)          # a lookup, not an evaluation
            elif room:
                allowed.append(state)
                room -= 1
        if not allowed and states:
            # Always at least one. The serial loop this replaced measured the
            # first state before it looked at the budget, and the search has
            # to know what the state it was handed is worth even when it was
            # given nothing to spend -- otherwise it can return a state it
            # never scored.
            allowed = [states[0]]
        states = allowed
        if workers <= 1 or len(states) <= 1:
            return [measure(state) for state in states]
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(workers, len(states))) as pool:
            return list(pool.map(measure, states))

    if start is None:
        start = tuple(len(axis) // 2 for axis in space.axes)

    # Small enough to look at all of it? Then look. This is not a shortcut
    # taken for speed: an exhaustive pass over a space this size cannot miss
    # the best state, and a walk can.
    whole = 1
    for axis in space.axes:
        whole *= len(axis)
    if whole <= min(EXHAUSTIVE_LIMIT, evaluations):
        every = [space.clamp(state) for state in itertools.product(
            *(range(len(axis)) for axis in space.axes))]
        scored = measure_many(every)
        best_state, best_energy = min(scored, key=lambda pair: (pair[1], pair[0]))
        return Result(best_state, best_energy, space.settings(best_state),
                      count[0], time.perf_counter() - began, len(scored), 0,
                      count[0],
                      [(i + 1, value, value) for i, (_s, value) in enumerate(scored)])

    seeds = [start] + [space.clamp(s) for s in extra_starts]
    # Enough seeds scattered over the space to know what the energy's own
    # scale is, because the temperature has to be in those units and there is
    # no way to guess it. Measuring the start alone leaves no scale at all,
    # and a temperature guessed too low is not a slow anneal -- it is a
    # greedy walk with extra steps, which sat in the wrong basin of the
    # deceptive landscape in tests/test_anneal.py and never left it. These
    # are real samples of the objective, so they are not a tax: they are also
    # what stops the search from starting cold.
    if temperature is None and len(seeds) < SCALE_SAMPLES:
        for _ in range(SCALE_SAMPLES - len(seeds)):
            seeds.append(tuple(rng.randrange(len(axis)) for axis in space.axes))
    # All of them at once: independent samples of the objective, taken
    # before anything has been decided, so there is no order to them and
    # nothing to be had by waiting between them.
    scored = measure_many(seeds)
    best_state, best_energy = min(scored, key=lambda pair: pair[1])
    current_state, current_energy = best_state, best_energy
    best_at = count[0]
    history = [(count[0], current_energy, best_energy)]

    if temperature is None:
        finite = [value for _s, value in scored if math.isfinite(value)]
        spread = (max(finite) - min(finite)) if len(finite) > 1 else 0.0
        temperature = max(1e-9, START_HEAT * spread) if spread else 1e-3
    heat = float(temperature)

    # The momentum, one signed integer per axis: the step that worked last
    # time on that axis, which the next proposal repeats and lengthens. It is
    # the lattice version of a velocity -- there is nothing to differentiate
    # here, but there is certainly something to keep going.
    velocity = [0] * len(space)
    accepted = restarts = 0
    since_improvement = 0

    while not over_budget():
        steps[0] += 1
        # One or two axes at a time: one is the ordinary move, and two is
        # what gets past a pair that only pays off together -- more denoise
        # is worse on its own and better with a looser offset.
        how_many = 1 if rng.random() < 0.7 or len(space) < 2 else 2
        moves = rng.sample(range(len(space)), how_many)
        state = list(current_state)
        # How far a random proposal may reach comes down with the
        # temperature: early on it may cross the ladder, later it is a step.
        reach = max(1, min(MAX_STEP, int(round(MAX_STEP * min(1.0, heat / temperature)))))
        for axis in moves:
            if velocity[axis] and rng.random() < MOMENTUM_BIAS:
                state[axis] += velocity[axis]
            else:
                state[axis] += rng.choice((-1, 1)) * rng.randint(1, reach)
        state = space.clamp(state)
        if tuple(state) == current_state:
            for axis in moves:
                velocity[axis] = 0          # it has run into the end of its ladder
            continue

        state, value = measure(state)
        # Standing on a setting that cannot be used at all, every move is an
        # improvement or a draw, and the walk has to be free to take it:
        # `inf - inf` is not a number, so the Metropolis test alone would
        # refuse every one of them and the search would sit on the unusable
        # state it was handed. That is not hypothetical -- it is the state a
        # page too noisy for its current settings starts from.
        if not math.isfinite(current_energy):
            accept = True
        else:
            change = value - current_energy
            accept = change < 0 or (math.isfinite(change) and heat > 0
                                    and rng.random() < math.exp(-change / heat))
        if accept:
            # It worked, so go further the same way next time. A direction
            # that keeps paying off gets a longer stride each step (1, 2, 3,
            # ...), which is what crosses a long monotone run in a handful of
            # moves instead of a random walk's square-root crawl.
            for axis in moves:
                step = state[axis] - current_state[axis]
                if not step:
                    continue
                sign = 1 if step > 0 else -1
                if velocity[axis] * sign > 0:
                    velocity[axis] = sign * min(MAX_STRIDE, abs(velocity[axis]) + 1)
                else:
                    velocity[axis] = sign
            current_state, current_energy = state, value
            accepted += 1
        else:
            for axis in moves:
                velocity[axis] = 0          # it stopped paying off; stop coasting
        if value < best_energy:
            best_state, best_energy, best_at = state, value, count[0]
            since_improvement = 0
        else:
            since_improvement += 1
        history.append((count[0], current_energy, best_energy))
        heat *= COOLING
        if since_improvement >= RESTART_AFTER:
            # The walk has cooled into a basin and is spending its budget
            # re-measuring the floor of it. Reheating where it stands only
            # works when the basin it wants is next door: when the good one
            # is somewhere else entirely, the cost of leaving is every step
            # of the way out, uphill, while the temperature falls -- and a
            # bowl seven rungs across defeats that however patient the
            # schedule. So restarts alternate: one back at the best it has,
            # one at a state drawn from anywhere in the space. That is basin
            # hopping, and it is what finds the narrow deep basin beside the
            # wide shallow one in tests/test_anneal.py.
            restarts += 1
            if restarts % 2 and not over_budget():
                current_state = tuple(rng.randrange(len(axis)) for axis in space.axes)
                current_state, current_energy = measure(current_state)
            else:
                current_state, current_energy = best_state, best_energy
            heat = float(temperature)
            velocity = [0] * len(space)
            since_improvement = 0

    # Polish: sweep each axis on its own, keeping any step that helps, until
    # a whole sweep finds nothing. Annealing lands near a good state rather
    # than on it, and this is the cheap part -- every state it tries that has
    # been tried already is a dictionary lookup.
    #
    # `spent`, not `count`: this used to watch the evaluation count alone, so
    # a caller who gave it no time at all got the walk stopped on its first
    # step and then the whole polish anyway -- which measured *more* settings
    # for a budget of zero seconds than for six of them.
    improved = True
    sweeps = 0
    while improved and not spent() and sweeps < 2 * len(space) + 4:
        improved = False
        sweeps += 1
        # A sweep is every axis's four neighbours, and they are all
        # neighbours of the *same* state, so they can be scored together.
        # Taking the best of the whole sweep instead of the first improvement
        # it stumbles on is a better move as well as a faster one.
        wanted = []
        for axis in range(len(space)):
            for step in (1, -1, 2, -2):
                candidate = tuple(space.clamp(
                    [rung + (step if i == axis else 0)
                     for i, rung in enumerate(best_state)]))
                if candidate != best_state:
                    wanted.append(candidate)
        wanted = list(dict.fromkeys(wanted))
        if not wanted or spent():
            break
        for candidate, value in measure_many(wanted):
            if value < best_energy:
                best_state, best_energy = candidate, value
                best_at, improved = count[0], True

    return Result(best_state, best_energy, space.settings(best_state), count[0],
                  time.perf_counter() - began, accepted, restarts, best_at, history)
