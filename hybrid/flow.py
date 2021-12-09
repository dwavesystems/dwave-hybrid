# Copyright 2018 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import time
import logging
import warnings
import concurrent.futures
import queue
from operator import attrgetter
from functools import partial
from itertools import chain

from dwave.cloud.utils import utcnow

from hybrid.core import Runnable, State, States, stoppable
from hybrid.concurrency import Present, immediate_executor
from hybrid.exceptions import EndOfStream
from hybrid.utils import OceanEncoder
from hybrid import traits

__all__ = [
    'Branch', 'Branches', 'RacingBranches', 'Race', 'ParallelBranches', 'Parallel',
    'Map', 'Reduce', 'Lambda', 'ArgMin', 'Unwind', 'TrackMin',
    'Loop', 'LoopUntilNoImprovement', 'LoopWhileNoImprovement',
    'Identity', 'BlockingIdentity', 'Dup', 'Const', 'Wait', 'Log',
]

logger = logging.getLogger(__name__)


class Branch(traits.NotValidated, Runnable):
    """Sequentially executed :class:`~hybrid.core.Runnable` components.

    Args:
        components (iterable of :class:`~hybrid.core.Runnable`):
            Complete processing sequence to update a current set of samples,
            such as: :code:`decomposer | sampler | composer`.

    Input:
        Defined by the first branch component.

    Output:
        Defined by the last branch component.

    Examples:
        This example runs one iteration of a branch comprising a decomposer,
        local Tabu solver, and a composer. A 10-variable binary quadratic model
        is decomposed by the energy impact of its variables into a 6-variable
        subproblem to be sampled twice with a manually set initial state of
        all -1 values.

        >>> import dimod           # Create a binary quadratic model
        >>> bqm = dimod.BQM({t: 0 for t in range(10)},
        ...                 {(t, (t+1) % 10): 1 for t in range(10)},
        ...                 0, 'SPIN')
        >>> # Run one iteration on a branch
        >>> branch = (EnergyImpactDecomposer(size=6, min_gain=-10) |
        ...           TabuSubproblemSampler(num_reads=2) |
        ...           SplatComposer())
        >>> new_state = branch.next(State.from_sample(min_sample(bqm), bqm))
        >>> print(new_state.subsamples)      # doctest: +SKIP
               4   5   6   7   8   9  energy  num_occ.
           0  +1  -1  -1  +1  -1  +1    -5.0         1
           1  +1  -1  -1  +1  -1  +1    -5.0         1
           [ 2 rows, 6 variables ]

    """

    def __init__(self, components=(), **runopts):
        super(Branch, self).__init__(**runopts)
        self.components = tuple(components)

        if not self.components:
            raise ValueError("branch has to contain at least one component")

        for component in self.components:
            if not isinstance(component, Runnable):
                raise TypeError("expected Runnable component, got {!r}".format(component))

    def __or__(self, other):
        """Sequential composition of runnable components (L-to-R)
        returns a new runnable Branch.
        """
        if isinstance(other, Branch):
            return Branch(components=chain(self, other))
        elif isinstance(other, Runnable):
            return Branch(components=chain(self, (other,)))
        else:
            raise TypeError("only Runnables can be composed into a Branch")

    def __str__(self):
        return " | ".join(map(str, self)) or "(empty branch)"

    def __repr__(self):
        return "{}(components={!r})".format(self.name, tuple(self))

    def __iter__(self):
        return iter(self.components)

    def next(self, state, **runopts):
        """Start an iteration of an instantiated :class:`Branch`.

        Accepts a state and returns a new state.

        Args:
            state (:class:`State`):
                Computation state passed to the first component of the branch.

        Examples:
            This code snippet runs one iteration of a branch to produce a new state::

                new_state = branch.next(core.State.from_sample(min_sample(bqm), bqm)

        """

        runopts['executor'] = immediate_executor

        for component in self.components:
            state = component.run(state, **runopts)

        return state.result()

    def error(self, exc):
        """Pass on the exception from input to the error handler of the first
        runnable in branch.
        """
        return self.next(Present(exception=exc))

    def halt(self):
        """Try terminating all components in an instantiated :class:`Branch`."""
        for component in self.components:
            component.stop()


class Branches(traits.NotValidated, Runnable):
    """Runs multiple workflows of type :class:`~hybrid.core.Runnable` in
    parallel, blocking until all finish.

    Branches operates similarly to :class:`~hybrid.flow.ParallelBranches`,
    but each branch runs on a separate input :class:`~hybrid.core.State`
    (while parallel branches all use the same input state).

    Args:
        *branches ([:class:`~hybrid.core.Runnable`]):
            Runnable branches listed as positional arguments.

    Input:
        :class:`~hybrid.core.States`

    Output:
        :class:`~hybrid.core.States`

    Note:
        :class:`~hybrid.flow.Branches` is also available via implicit
        parallelization binary operator `&`.

    Examples:
        This example runs two branches, a classical tabu search and a random
        sampler, until both terminate::

            Branches(TabuSubproblemSampler(), RandomSubproblemSampler())

        Alternatively::

            TabuSubproblemSampler() & RandomSubproblemSampler()

    """

    def __init__(self, *branches, **runopts):
        super(Branches, self).__init__(**runopts)
        self.branches = tuple(branches)

        if not self.branches:
            raise ValueError("Branches require at least one branch")

        for branch in self.branches:
            if not isinstance(branch, Runnable):
                raise TypeError("expected Runnable branch, got {!r}".format(branch))

    def __and__(self, other):
        """Parallel composition of runnable components returns new Branches."""
        if isinstance(other, Branches):
            return Branches(*chain(self, other))
        elif isinstance(other, Runnable):
            return Branches(*chain(self, (other,)))
        else:
            raise TypeError("only Runnables can be composed into Branches")

    def __str__(self):
        return " & ".join("({})".format(b) for b in self) or "(zero branches)"

    def __repr__(self):
        return "{}{!r}".format(self.name, tuple(self))

    def __iter__(self):
        return iter(self.branches)

    def next(self, states, **runopts):
        futures = [
            branch.run(state.updated(), **runopts)
                for branch, state in zip(self.branches, states)]

        logger.debug("{} running {} branches in parallel".format(
            self.name, len(futures)))

        # wait for all branches to finish
        concurrent.futures.wait(
            futures,
            return_when=concurrent.futures.ALL_COMPLETED)

        # collect resolved states (in original order, not completion order)
        states = States()
        for f in futures:
            states.append(f.result())

        return states

    def halt(self):
        for branch in self.branches:
            branch.stop()


class RacingBranches(traits.NotValidated, Runnable):
    """Runs (races) multiple workflows of type :class:`~hybrid.core.Runnable`
    in parallel, stopping all once the first finishes. Returns the results of
    all, in the specified order.

    Args:
        *branches ([:class:`~hybrid.core.Runnable`]):
            Comma-separated branches.

    Note:
        Each branch runnable is called with run option ``racing_context=True``,
        so it can adapt its behaviour to the context.

    Note:
        `RacingBranches` is also available as `Race`.

    Examples:
        This example runs two branches: a classical tabu search interrupted by
        samples of subproblems returned from a D-Wave system.

        ::

            RacingBranches(
                InterruptableTabuSampler(),
                EnergyImpactDecomposer(size=2)
                | QPUSubproblemAutoEmbeddingSampler()
                | SplatComposer()
            ) | ArgMin()

    """

    def __init__(self, *branches, **runopts):
        self.branches = branches
        super(RacingBranches, self).__init__(**runopts)

        if not self.branches:
            raise ValueError("racing branches requires at least one branch")

    def __str__(self):
        return " !! ".join("({})".format(b) for b in self) or "(zero racing branches)"

    def __repr__(self):
        return "{}{!r}".format(self.name, tuple(self))

    def __iter__(self):
        return iter(self.branches)

    def next(self, state, **runopts):
        runopts.update(racing_context=True)
        futures = [branch.run(state.updated(), **runopts) for branch in self.branches]

        # as soon as one is done, stop all others
        done, _ = concurrent.futures.wait(
            futures,
            return_when=concurrent.futures.FIRST_COMPLETED)

        logger.trace("RacingBranches done set: {}. Stopping remaining.".format(done))
        self.stop()

        # debug info
        idx = futures.index(done.pop())
        branch = self.branches[idx]
        logger.debug("{name} won idx={idx} branch={branch!r}".format(
            name=self.name, idx=idx, branch=branch))

        # collect resolved states (in original order, not completion order!)
        states = States()
        for f in futures:
            states.append(f.result())

        return states

    def halt(self):
        """Terminate an iteration of an instantiated :class:`RacingBranches`."""
        for branch in self.branches:
            branch.stop()


Race = RacingBranches


class Dup(traits.NotValidated, Runnable):
    """Duplicates input :class:`~hybrid.core.State`, n times, into output
    :class:`~hybrid.core.States`.
    """

    def __init__(self, n, *args, **kwargs):
        super(Dup, self).__init__(*args, **kwargs)
        self.n = n

    def __repr__(self):
        return "{}(n={!r})".format(self.name, self.n)

    def next(self, state, **runopts):
        logger.debug("{} cloning input state {} time(s)".format(self.name, self.n))
        return States(*[state.updated() for _ in range(self.n)])


class ParallelBranches(traits.NotValidated, Runnable):
    """Runs multiple workflows of type :class:`~hybrid.core.Runnable` in
    parallel, blocking until all finish.

    Parallel/ParallelBranches operates similarly to :class:`~hybrid.flow.Branches`,
    but every branch re-uses the same input :class:`~hybrid.core.State`.

    Args:
        *branches ([:class:`~hybrid.core.Runnable`]):
            Comma-separated branches.

    Input:
        :class:`~hybrid.core.State`

    Output:
        :class:`~hybrid.core.States`

    Note:
        `Parallel` is implemented as::

            Parallel(*branches) := Dup(len(branches)) | Branches(*branches)

    Note:
        `ParallelBranches` is also available as `Parallel`.

    Examples:
        This example runs two branches, a classical tabu search and a random
        sampler, until both terminate::

            Parallel(
                TabuSubproblemSampler(),
                RandomSubproblemSampler()
            ) | ArgMin()

    """

    def __init__(self, *branches, **runopts):
        super(ParallelBranches, self).__init__(**runopts)
        self.branches = Branches(*branches)
        self.runnable = Dup(len(tuple(self.branches))) | self.branches

    def __repr__(self):
        return "{}{!r}".format(self.name, tuple(self.branches))

    def __iter__(self):
        return iter(self.branches)

    def next(self, state, **runopts):
        runopts['executor'] = immediate_executor
        return self.runnable.run(state, **runopts).result()

    def halt(self):
        return self.runnable.stop()


Parallel = ParallelBranches


class Map(traits.NotValidated, Runnable):
    """Runs a specified :class:`~hybrid.core.Runnable` in parallel on all input
    states.

    Args:
        runnable (:class:`~hybrid.core.Runnable`):
            A runnable executed for every input state.

    Examples:
        This example runs `TabuProblemSampler` on two input states in parallel,
        returning when both are done.

        >>> states = States(State(problem=bqm1), State(problem=bqm2))   # doctest: +SKIP
        >>> Map(TabuProblemSampler()).run(states).result()              # doctest: +SKIP
        [<state_1_with_solution>, <state_2_with_solution>]

    """

    def __init__(self, runnable, **runopts):
        if not isinstance(runnable, Runnable):
            raise TypeError("'runnable' is not instance of Runnable")

        super(Map, self).__init__(**runopts)
        self.runnable = runnable

        # track running computations, so we can stop them on request
        self._futures = []

    def __str__(self):
        return "[]()"

    def __repr__(self):
        return "{}(runnable={!r})".format(self.name, self.runnable)

    def __iter__(self):
        return iter((self.runnable,))

    def next(self, states, **runopts):
        self._futures = [self.runnable.run(state, **runopts) for state in states]

        logger.debug("{} running {!r} on {} input states".format(
            self.name, self.runnable, len(states)))

        concurrent.futures.wait(self._futures,
                                return_when=concurrent.futures.ALL_COMPLETED)

        return States(*(f.result() for f in self._futures))

    def halt(self):
        for future in self._futures:
            future.cancel()


class Reduce(traits.NotValidated, Runnable):
    """Fold-left using the specified :class:`~hybrid.core.Runnable` on a
    sequence of input states, producing a single output state.

    Args:
        runnable (:class:`~hybrid.core.Runnable`):
            A runnable used as the fold-left operator. It should accept a
            2-State input and produce a single State on output.

        initial_state (:class:`~hybrid.core.State`, optional, default=None):
            Optional starting state into which input states will be folded in.
            If undefined, the first input state is used as the `initial_state`.

    """

    def __init__(self, runnable, initial_state=None, **runopts):
        if not isinstance(runnable, Runnable):
            raise TypeError("'runnable' is not instance of Runnable")

        if initial_state is not None and not isinstance(initial_state, State):
            raise TypeError("'initial_state' is not instance of State")

        super(Reduce, self).__init__(**runopts)
        self.runnable = runnable
        self.initial_state = initial_state

    def __str__(self):
        return "Reduce {}".format(self.runnable)

    def __repr__(self):
        return ("{self.name}(runnable={self.runnable!r}, "
                "initial_state={self.initial_state!r}").format(self=self)

    def __iter__(self):
        return iter((self.runnable,))

    def next(self, states, **runopts):
        """Collapse all `states` to a single output state using the `self.runnable`."""

        logger.debug("{} collapsing {} input states with {!r}".format(
            self.name, len(states), self.runnable))

        states = iter(states)

        if self.initial_state is None:
            result = next(states)
        else:
            result = self.initial_state

        runopts['executor'] = immediate_executor

        for state in states:
            result = self.runnable.run(States(result, state), **runopts).result()

        return result


class Lambda(traits.NotValidated, Runnable):
    """Creates a runnable on fly, given just its `next` function (optionally
    `init` and `error` functions can be specified too).

    Args:
        next (callable):
            Implementation of runnable's `next` method, provided as a callable
            (usually a lambda expression for simple operations). Signature of
            the callable has to match the signature of
            :meth:`~hybrid.core.Runnable.next()`; i.e., it accepts two
            arguments: runnable instance and state instance.

        error (callable):
            Implementation of runnable's `error` method.
            See :meth:`~hybrid.core.Runnable.error`.

        init (callable):
            Implementation of runnable's `init` method.
            See :meth:`~hybrid.core.Runnable.init`.

    Note:
        Traits are not enforced, apart from the SISO requirement. Also, note
        `Lambda` runnables can only implement SISO systems.

    Examples:
        This example creates and runs a simple runnable that multiplies state
        variables `a` and `b`, storing them in `c`.

        >>> Lambda(lambda _, s: s.updated(c=s.a * s.b)).run(State(a=2, b=3)).result()     # doctest: +SKIP
        {'a': 2, 'b': 3, 'c': 6}

        This example applies `x += 1` to a sequence of input states.

        >>> Map(Lambda(lambda _, s: s.updated(x=s.x + 1))).run(States(State(x=0), State(x=1))).result()
        [{'x': 1}, {'x': 2}]
    """

    def __init__(self, next, error=None, init=None, **runopts):
        if not callable(next):
            raise TypeError("'next' is not callable")
        if error is not None and not callable(error):
            raise TypeError("'error' is not callable")
        if init is not None and not callable(init):
            raise TypeError("'init' is not callable")

        super(Lambda, self).__init__(**runopts)

        # bind to self
        self.next = partial(next, self, **runopts)
        if error is not None:
            self.error = partial(error, self)
        if init is not None:
            self.init = partial(init, self, **runopts)

        # keep a copy for inspection (without cycles to `self`)
        self._next = next
        self._error = error
        self._init = init

    def __repr__(self):
        return "{}(next={!r}, error={!r}, init={!r})".format(
            self.name, self._next, self._error, self._init)


class ArgMin(traits.NotValidated, Runnable):
    """Selects the best state from a sequence of :class:`~hybrid.core.States`.

    Args:
        key (callable/str):
            Best state is judged according to a metric defined with a `key`.
            The `key` can be a `callable` with a signature::

                key :: (State s, Ord k) => s -> k

            or a string holding a key name/path to be extracted from the input
            state with `operator.attrgetter` method.

            By default, `key == operator.attrgetter('samples.first.energy')`,
            thus favoring states containing a sample with the minimal energy.

    Examples:
        This example runs two branches---a classical tabu search interrupted by
        samples of subproblems returned from a D-Wave system--- and selects the
        state with the minimum-energy sample::

            RacingBranches(
                InterruptableTabuSampler(),
                EnergyImpactDecomposer(size=2)
                | QPUSubproblemAutoEmbeddingSampler()
                | SplatComposer()
            ) | ArgMin()

    """

    def __init__(self, key=None, **runopts):
        """Return the state which minimizes the objective function `key`."""
        super(ArgMin, self).__init__(**runopts)
        if key is None:
            key = 'samples.first.energy'
        if isinstance(key, str):
            key = attrgetter(key)
        self.key = key

    def __str__(self):
        return "[]>"

    def __repr__(self):
        return "{}(key={!r})".format(self.name, self.key)

    def next(self, states, **runopts):
        """Execute one blocking iteration of an instantiated :class:`ArgMin`."""

        # expand `return min(states, key=self.key)` for logging/tracking
        values = [self.key(state) for state in states]
        min_idx = values.index(min(values))

        # debug info
        for idx, val in enumerate(values):
            logger.debug("{name} State(idx={idx}, val={val})".format(
                name=self.name, idx=idx, val=val))

        logger.debug("{name} min_idx={min_idx}".format(
            name=self.name, min_idx=min_idx))

        self.count('branch-%d' % min_idx)

        return states[min_idx]


class TrackMin(traits.NotValidated, Runnable):
    """Tracks and records the best :class:`~hybrid.core.State` according to a
    metric defined with a `key` function; typically this is the minimal state.

    Args:
        key (callable/str, optional, default=None):
            Best state is judged according to a metric defined with a `key`.
            `key` can be a `callable` with a signature::

                key :: (State s, Ord k) => s -> k

            or a string holding a key name/path to be extracted from the input
            state with `operator.attrgetter` method.

            By default, `key == operator.attrgetter('samples.first.energy')`,
            thus favoring states containing a sample with the minimal energy.

        output (bool, optional, default=False):
            Update the output state's `output_key` with the `input_key` of the
            best state seen so far.

        input_key (str, optional, default='samples')
            If `output=True`, then this defines the variable/key name in the
            input state that shall be included in the output state.

        output_key (str, optional, default='samples')
            If `output=True`, then the key under which the `input_key` from the
            best state seen so far is stored in the output state.

    Note:
        If `output` option is turned on, and `output_key` is not changed, the
        output will by default change the state's `samples` on output.

    """

    def __init__(self, key=None, output=False, input_key='samples',
                 output_key='samples', **runopts):
        super(TrackMin, self).__init__(**runopts)
        if key is None:
            key = 'samples.first.energy'
        if isinstance(key, str):
            key = attrgetter(key)
        self.key = key
        self.output = output
        self.output_key = output_key
        self.input_key = input_key

    def __repr__(self):
        return (
            "{self.name}(key={self.key!r}, output={self.output!r}, "
            "input_key={self.input_key!r}, output_key={self.output_key!r})"
        ).format(self=self)

    def _set_new_best(self, state):
        self.best = state

        logger.debug("{} selected state with key={!r} for the new best state".format(
            self.name, self.key(self.best)))
        logger.trace("{} selected {!r} for the new best state".format(
            self.name, self.best))

    def init(self, state, **runopts):
        self._set_new_best(state)

    def next(self, state, **runopts):
        if self.key(state) < self.key(self.best):
            self._set_new_best(state)
            self.count('new-best')

        if self.output:
            return state.updated(**{self.output_key: self.best[self.input_key]})

        return state


@stoppable
class LoopUntilNoImprovement(traits.NotValidated, Runnable):
    """Iterates :class:`~hybrid.core.Runnable` for up to `max_iter` times, or
    until a state quality metric, defined by the `key` function, shows no
    improvement for at least `convergence` number of iterations. Alternatively,
    maximum allowed runtime can be defined with `max_time`, or a custom
    termination Boolean function can be given with `terminate` (a predicate
    on `key`). Loop is always terminated on :exc:`EndOfStream` raised by body
    runnable.

    Args:
        runnable (:class:`~hybrid.core.Runnable`):
            A runnable that's looped over.

        max_iter (int/None, optional, default=None):
            Maximum number of times the `runnable` is run, regardless of other
            termination criteria. This is the upper bound. By default, an upper
            bound on the number of iterations is not set.

        convergence (int/None, optional, default=None):
            Terminates upon reaching this number of iterations with unchanged
            output. By default, convergence is not checked, so the only
            termination criteria is defined with `max_iter`. Setting neither
            creates an infinite loop.

        max_time (float/None, optional, default=None):
            Wall clock runtime termination criterion. Unlimited by default.

        key (callable/str):
            Best state is judged according to a metric defined with a `key`.
            `key` can be a `callable` with a signature::

                key :: (State s, Ord k) => s -> k

            or a string holding a key name/path to be extracted from the input
            state with `operator.attrgetter` method.

            By default, `key == operator.attrgetter('samples.first.energy')`,
            thus favoring states containing a sample with the minimal energy.

        terminate (callable, optional, default=None):
            Loop termination Boolean function (a predicate on `key` value)::

                terminate :: (Ord k) => k -> Bool
    """

    def __init__(self, runnable, max_iter=None, convergence=None,
                 max_time=None, key=None, terminate=None, **runopts):
        super(LoopUntilNoImprovement, self).__init__(**runopts)
        self.runnable = runnable
        self.max_iter = max_iter
        self.max_time = max_time
        self.convergence = convergence

        if key is None:
            key = 'samples.first.energy'
        if isinstance(key, str):
            key = attrgetter(key)
        self.key = key

        if terminate is not None and not callable(terminate):
            raise TypeError("expecting a predicate on 'key' for 'terminate'")
        self.terminate = terminate

    def __str__(self):
        return "Loop over {}".format(self.runnable)

    def __repr__(self):
        return ("{self.name}(runnable={self.runnable!r}, max_iter={self.max_iter!r}, "
                "convergence={self.convergence!r}, max_time={self.max_time!r}, "
                "key={self.key!r}, terminate={self.terminate!r})").format(self=self)

    def __iter__(self):
        return iter((self.runnable,))

    def iteration_update(self, iterno, cnt, inp, out):
        """Implement "converge on unchanging output" behavior:

          - loop `max_iter` times, but bail-out earlier if output doesn't change
            (over input) for `convergence` number of iterations

          - each iteration starts with the previous result state

        Input: relevant counters and I/O states.
        Output: next input state and next counter values
        """
        input_state, input_key = inp
        output_state, output_key = out

        if self.convergence is None:
            return iterno + 1, cnt, output_state

        if output_key == input_key:
            cnt -= 1
        else:
            cnt = self.convergence

        return iterno + 1, cnt, output_state

    def next(self, state, **runopts):
        iterno = 0
        cnt = self.convergence or 0
        input_state = state
        output_state = input_state
        input_key = None
        output_key = None
        start = time.time()

        runopts['executor'] = immediate_executor

        while not self.stop_signal.is_set():

            try:
                output_state = self.runnable.run(input_state, **runopts).result()
            except EndOfStream as exc:
                logger.debug("{name} Iteration(iterno={iterno}) terminating due "
                             "to {exc!r}".format(name=self.name, iterno=iterno, exc=exc))
                break

            if self.convergence or self.terminate:
                input_key = self.key(input_state)
                output_key = self.key(output_state)

            logger.info("{name} Iteration(iterno={iterno}, "
                        "input_state_key={inp}, output_state_key={out})".format(
                            name=self.name, iterno=iterno,
                            inp=input_key, out=output_key))

            iterno, cnt, input_state = self.iteration_update(
                iterno, cnt, (input_state, input_key), (output_state, output_key))

            runtime = time.time() - start

            if self.max_iter is not None and iterno >= self.max_iter:
                break
            if self.max_time is not None and runtime >= self.max_time:
                break
            if self.convergence is not None and cnt <= 0:
                break
            if self.terminate is not None and self.terminate(output_key):
                break

        return output_state

    def halt(self):
        self.runnable.stop()


class Loop(LoopUntilNoImprovement):
    """Alias for :class:`LoopUntilNoImprovement`."""


class SimpleIterator(LoopUntilNoImprovement):
    """Deprecated loop runnable. Use `Loop`/`LoopUntilNoImprovement` instead."""

    def __init__(self, *args, **kwargs):
        super(SimpleIterator, self).__init__(*args, **kwargs)

        warnings.warn("SimpleIterator is deprecated, please use Loop instead.",
                      DeprecationWarning)


class LoopWhileNoImprovement(LoopUntilNoImprovement):
    """Iterates :class:`~hybrid.core.Runnable` until a state quality metric,
    defined by the `key` function, shows no improvement for at least `max_tries`
    number of iterations or until `max_iter` number of iterations is exceeded.
    Alternatively, maximum allowed runtime can be defined with `max_time`, or a
    custom termination Boolean function can be given with `terminate` (a
    predicate on `key`).

    Note:
        Unlike `LoopUntilNoImprovement`/`Loop`, `LoopWhileNoImprovement` will
        run the loop body runnable with the **same input** if output shows no
        improvement (up to `max_tries` times), and it will use the new output
        if it's better than the input.

    Args:
        runnable (:class:`~hybrid.core.Runnable`):
            A runnable that's looped over.

        max_iter (int/None, optional, default=None):
            Maximum number of times the `runnable` is run, regardless of other
            termination criteria. This is the upper bound. By default, an upper
            bound on the number of iterations is not set.

        max_tries (int, optional, default=None):
            Maximum number of times the `runnable` is run for the **same** input
            state. On each improvement, the better state is used for the next
            input state, and the try/trial counter is reset. Defaults to an
            infinite loop (unbounded number of tries).

        max_time (float/None, optional, default=None):
            Wall clock runtime termination criterion. Unlimited by default.

        key (callable/str):
            Best state is judged according to a metric defined with a `key`.
            `key` can be a `callable` with a signature::

                key :: (State s, Ord k) => s -> k

            or a string holding a key name/path to be extracted from the input
            state with `operator.attrgetter` method.

            By default, `key == operator.attrgetter('samples.first.energy')`,
            thus favoring states containing a sample with the minimal energy.

        terminate (callable, optional, default=None):
            Loop termination Boolean function (a predicate on `key` value)::

                terminate :: (Ord k) => k -> Bool
    """

    def __init__(self, runnable, max_iter=None, max_tries=None,
                 max_time=None, key=None, terminate=None, **runopts):
        super(LoopWhileNoImprovement, self).__init__(
            runnable=runnable, max_iter=max_iter, convergence=max_tries,
            max_time=max_time, key=key, terminate=terminate, **runopts)

    def iteration_update(self, iterno, cnt, inp, out):
        """Implement "no-improvement count-down" behavior:

          - loop indefinitely, but bail-out if there's no improvement of output
            over input for `max_tries` number of iterations

          - each iteration uses the same input state, unless there was an improvement
            in this iteration, in which case, use the current output as next input

        Input: relevant counters and I/O states.
        Output: next input state and next counter values
        """
        input_state, input_key = inp
        output_state, output_key = out

        if self.convergence is None:
            return iterno + 1, cnt, output_state

        if output_key >= input_key:
            # no improvement, re-use the same input
            cnt -= 1
            next_input_state = input_state
        else:
            # improvement, use the better output for next input, restart local counter
            cnt = self.convergence
            next_input_state = output_state

        return iterno + 1, cnt, next_input_state


class Unwind(traits.NotValidated, Runnable):
    """Iterates :class:`~hybrid.core.Runnable` until :exc:`.EndOfStream` is
    raised, collecting all output states along the way.

    Note:
        the child runnable is called with run option ``silent_rewind=False``,
        and it is expected to raise :exc:`.EndOfStream` on unwind completion.
    """

    def __init__(self, runnable, **runopts):
        if not isinstance(runnable, Runnable):
            raise TypeError("'runnable' is not instance of Runnable")

        super(Unwind, self).__init__(**runopts)
        self.runnable = runnable

    def __str__(self):
        return "Unwind {}".format(self.runnable)

    def __repr__(self):
        return ("{self.name}(runnable={self.runnable!r}").format(self=self)

    def __iter__(self):
        return iter((self.runnable,))

    def next(self, state, **runopts):
        output = States()
        runopts.update(executor=immediate_executor, silent_rewind=False)

        logger.debug("{} unwinding {!r}".format(self.name, self.runnable))

        while True:
            try:
                state = self.runnable.run(state, **runopts).result()
                output.append(state)
            except EndOfStream:
                break

        logger.debug("{} collected {} states".format(self.name, len(output)))

        return output


@stoppable
class Wait(traits.NotValidated, Runnable):
    """Run indefinitely (effectively blocking branch execution). Has to be
    explicitly stopped.

    Example:
        To effectively exclude one branch from the race, i.e. prevent premature
        stopping of the race between the remaining branches, use :class:`.Wait`
        as the last element in a (fast-executing) racing branch::

            Race(
                Identity() | Wait(),
                InterruptableTabuSampler(),
                SimulatedAnnealingProblemSampler()
            )

        This is functionally identical to::

            Parallel(
                Identity(),
                Race(
                    InterruptableTabuSampler(),
                    SimulatedAnnealingProblemSampler()
                )
            )
    """

    def next(self, state, **runopts):
        self.stop_signal.wait()
        return state.updated()


class Identity(traits.NotValidated, Runnable):
    """Trivial identity runnable. The output is a direct copy of the input."""

    def next(self, state, **runopts):
        return state.updated()


class BlockingIdentity(Wait):
    """Trivial identity runnable that blocks indefinitely before producing
    output, but is interruptable. The output is a direct copy of
    the input, but to receive the output, the block has to be explicitly stopped
    (useful for example in :class:`.RacingBranches` to prevent short-circuiting
    of racing branches with the identity branch).

    ::

        BlockingIdentity := Identity | Wait

    Due to nature of :class:`.Identity`, :class:`.BlockingIdentity` is
    functionally equivalent to :class:`.Wait`.
    """


class Const(traits.NotValidated, Runnable):
    """Set state variables to constant values.

    Args:
        **consts (dict, optional):
            Mapping of state variables to constant values, as keyword arguments.

    Example:
        This example defines a workflow that resets the set of samples before a
        Tabu sampler call in order to avoid using existing samples as initial
        states. Instead, Tabu will use randomly generated initial states::

            random_tabu = Const(samples=None) | TabuProblemSampler(initial_states_generator='random')

    """

    def __init__(self, **consts):
        super(Const, self).__init__()
        self.consts = consts

    def next(self, state, **runopts):
        return state.updated(**self.consts)


class Log(traits.NotValidated, Runnable):
    """Tracks and logs user-defined data (e.g. state metrics) extracted from
    state by the ``key`` function.

    Args:
        key (callable):
            Data/metric(s) extractor. A callable that receives a state, and
            returns a mapping of values to names::

                key: Callable[State] -> Mapping[str, Any]

            Data returned by the key function is stored under ``data`` key of
            the produced log record. In addition to ``data``, UTC ``time`` (in
            ISO-8601 format) and a monotonic float ``timestamp`` (secods since
            the epoch) are always stored in the log record.

        extra (dict, optional):
            Static extra items to add to each log record.

        outfile (file, optional):
            A file opened in text + write/append mode. JSON log record is
            written on each runnable iteration and optionally flushed, depending
            on `buffering` argument.

            Note: Avoid using the same ``outfile`` for multiple
            :class:`~hybrid.flow.Log` runnables, as file write operation is not
            thread-safe.

        buffering (bool, optional, default=True):
            When buffering is set to False, output to `outfile` is flushed on
            each iteration.

        memo (boolean/list/queue.Queue, optional, default=False):
            Set to True to keep track of all log records produced in the
            instance-local :attr:`Log.records` list. Alternatively, provide an
            external list or :class:`queue.Queue` in which all log records will
            be pushed.

        loglevel (int, optional, default=logging.NOTSET):
            When loglevel is set, output the log record to standard logger
            configured for ``hybrid.flow`` namespace.

    """

    def __init__(self, key, extra=None, outfile=None, buffering=False,
                 memo=False, loglevel=logging.NOTSET, **runopts):
        super().__init__(**runopts)
        if not callable(key):
            raise TypeError("callable 'key' expected")
        self.key = key
        self.extra = extra
        self.outfile = outfile
        self.buffering = buffering

        if hasattr(memo, 'append') or hasattr(memo, 'put'):
            self.records = memo
        elif memo:
            self.records = []
        else:
            self.records = None

        self.loglevel = loglevel

    def _append_record(self, record):
        if hasattr(self.records, 'append'):
            self.records.append(record)
        elif hasattr(self.records, 'put'):
            self.records.put(record)
        else:
            raise TypeError("unsupported 'memo' container type")

    def __repr__(self):
        return (f"{self.name}(key={self.key!r}, extra={self.extra!r}, "
                f"outfile={self.outfile!r}, buffering={self.buffering!r}, "
                f"memo={self.memo!r}, loglevel={self.loglevel!r})")

    def next(self, state, **runopts):
        data = self.key(state)
        record = dict(time=utcnow(), timestamp=time.monotonic(), data=data)
        if self.extra is not None:
            record.update(self.extra)
        msg = json.dumps(record, cls=OceanEncoder)

        if self.outfile:
            print(msg, file=self.outfile, flush=not self.buffering)

        if self.records is not None:
            self._append_record(record)

        if self.loglevel:
            logger.log(self.loglevel, f"{self.name} Record({msg})")

        return state
