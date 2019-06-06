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

import time
import logging
import warnings
import concurrent.futures
from operator import attrgetter
from functools import partial
from itertools import chain

import six

from hybrid.core import Runnable, State, States, stoppable
from hybrid.concurrency import Present, immediate_executor
from hybrid.exceptions import EndOfStream
from hybrid import traits

__all__ = [
    'Branch', 'RacingBranches', 'Race', 'ParallelBranches', 'Parallel',
    'Map', 'Reduce', 'Lambda', 'ArgMin', 'Unwind', 'TrackMin',
    'Loop', 'LoopUntilNoImprovement', 'LoopWhileNoImprovement',
    'Identity'
]

logger = logging.getLogger(__name__)


class Branch(Runnable):
    """Sequentially executed :class:`~hybrid.core.Runnable` components.

    Args:
        components (iterable of :class:`~hybrid.core.Runnable`):
            Complete processing sequence to update a current set of samples,
            such as: :code:`decomposer | sampler | composer`.

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

        # patch branch's I/O requirements based on the first and last component

        # be conservative in output requirements, but liberal in input requirements
        #
        # i.e when calculating input requirements, assume the best case scenario,
        # that state is accumulated along the branch; but don't assume that for
        # output

        minimal_inputs = self.components[0].inputs.copy()
        state = self.components[0].inputs.copy()
        # consider connections between all connected components (a, b)
        for a, b in zip(self.components, self.components[1:]):
            # update the "running" state traits, and minimally acceptable input traits
            state |= a.outputs
            missing = b.inputs - state
            minimal_inputs |= missing

            # btw, check dimensionality compatibility
            if a.multi_output != b.multi_input:
                raise TypeError(
                    "mismatched output/input dimensions between {!r} and {!r}".format(a, b))

        self.inputs = minimal_inputs

        # this the minimum we can guarantee
        self.outputs = self.components[-1].outputs

        # this must hold
        self.multi_input = self.components[0].multi_input
        self.multi_output = self.components[-1].multi_output


    def __or__(self, other):
        """Composition of Branch with runnable components (L-to-R) returns a new
        runnable Branch.
        """
        if isinstance(other, Branch):
            return Branch(components=chain(self.components, other.components))
        elif isinstance(other, Runnable):
            return Branch(components=chain(self.components, (other,)))
        else:
            raise TypeError("branch can be composed only with Branch or Runnable")

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


class RacingBranches(Runnable, traits.SIMO):
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

        >>> RacingBranches(                     # doctest: +SKIP
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

        # patch components's I/O requirements based on the subcomponents' requirements

        # ensure i/o dimensionality for all branches is the same
        first = branches[0]
        if not all(b.multi_input == first.multi_input for b in branches[1:]):
            raise TypeError("not all branches have the same input dimensionality")
        if not all(b.multi_output == first.multi_output for b in branches[1:]):
            raise TypeError("not all branches have the same output dimensionality")

        # RB's input has to satisfy all branches' input
        self.inputs = set.union(*(branch.inputs for branch in self.branches))
        self.multi_input = first.multi_input

        # RB's output will be one of the branches' output, but the only guarantee we
        # can make upfront is the largest common subset of all outputs
        self.outputs = set.intersection(*(branch.outputs for branch in self.branches))
        self.multi_output = True

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


class ParallelBranches(Runnable, traits.SIMO):
    """Runs multiple multiple workflows of type :class:`~hybrid.core.Runnable`
    in parallel, blocking until all finish.

    Args:
        *branches ([:class:`~hybrid.core.Runnable`]):
            Comma-separated branches.

    Note:
        `ParallelBranches` is also available as `Parallel`.

    Examples:
        This example runs two branches, a classical tabu search and a random
        sampler, until both terminate.

        >>> Parallel(                     # doctest: +SKIP
                TabuSubproblemSampler(),
                RandomSubproblemSampler()
            ) | ArgMin()

    """

    def __init__(self, *branches, **runopts):
        self.branches = branches
        super(ParallelBranches, self).__init__(**runopts)

        if not self.branches:
            raise ValueError("parallel branches require at least one branch")

        # patch components's I/O requirements based on the subcomponents' requirements

        # ensure i/o dimensionality for all branches is the same
        first = branches[0]
        if not all(b.multi_input == first.multi_input for b in branches[1:]):
            raise TypeError("not all branches have the same input dimensionality")
        if not all(b.multi_output == first.multi_output for b in branches[1:]):
            raise TypeError("not all branches have the same output dimensionality")

        # PB's input has to satisfy all branches' input
        self.inputs = set.union(*(branch.inputs for branch in self.branches))
        self.multi_input = first.multi_input

        # PB's output will be one of the branches' output, but the only guarantee we
        # can make upfront is the largest common subset of all outputs
        self.outputs = set.intersection(*(branch.outputs for branch in self.branches))
        self.multi_output = True

    def __str__(self):
        return " & ".join("({})".format(b) for b in self) or "(zero branches)"

    def __repr__(self):
        return "{}{!r}".format(self.name, tuple(self))

    def __iter__(self):
        return iter(self.branches)

    def next(self, state, **runopts):
        futures = [branch.run(state.updated(), **runopts) for branch in self.branches]

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
        """Terminate an iteration of an instantiated :class:`RacingBranches`."""
        for branch in self.branches:
            branch.stop()


Parallel = ParallelBranches


class Map(Runnable, traits.MIMO):
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

        # patch components's I/O requirements based on the subcomponents' requirements
        # TODO: automate
        self.inputs = runnable.inputs
        self.outputs = runnable.outputs

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

        concurrent.futures.wait(self._futures,
                                return_when=concurrent.futures.ALL_COMPLETED)

        return States(*(f.result() for f in self._futures))

    def halt(self):
        for future in self._futures:
            future.cancel()


class Reduce(Runnable, traits.MISO):
    """Fold-left using the specified :class:`~hybrid.core.Runnable` on a
    sequence of input states, producing a single output state.

    Args:
        runnable (:class:`~hybrid.core.Runnable`):
            A runnable used as the fold-left operator. It should accept a
            2-State input and produce a single State on output.

        initial_state (:class:`State`, optional, default=None):
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

        # preemptively check runnable's i/o dimensionality
        if runnable.validate_input and runnable.validate_output:
            if not runnable.multi_input or runnable.multi_output:
                raise TypeError("runnables must be of multi-input, single-output type")

        # patch components's I/O requirements based on the subcomponents' requirements
        self.multi_input = True
        self.inputs = runnable.inputs
        self.multi_output = False
        self.outputs = runnable.outputs

    def __str__(self):
        return "Reduce {}".format(self.runnable)

    def __repr__(self):
        return ("{self.name}(runnable={self.runnable!r}, "
                "initial_state={self.initial_state!r}").format(self=self)

    def __iter__(self):
        return iter((self.runnable,))

    def next(self, states, **runopts):
        """Collapse all `states` to a single output state using the `self.runnable`."""

        states = iter(states)

        if self.initial_state is None:
            result = next(states)
        else:
            result = self.initial_state

        runopts['executor'] = immediate_executor

        for state in states:
            result = self.runnable.run(States(result, state), **runopts).result()

        return result


class Lambda(Runnable, traits.NotValidated):
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
        {'a': 2, 'b': 3, 'c': 6, ...}

        This example applies `x += 1` to a sequence of input states.

        >>> Map(Lambda(lambda _, s: s.updated(x=s.x + 1))).run(States(State(x=0), State(x=1))).result()   # doctest: +SKIP
        [{'problem': None, 'x': 1, 'samples': None}, {'problem': None, 'x': 2, 'samples': None}]
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


class ArgMin(Runnable, traits.MISO):
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
        state with the minimum-energy sample.

        >>> RacingBranches(                     # doctest: +SKIP
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
        if isinstance(key, six.string_types):
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


class TrackMin(Runnable, traits.SISO):
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

        output_key (str, optional, default='best_samples')
            If `output=True`, then the key under which the `input_key` from the
            best state seen so far is stored in the output state.

    """

    def __init__(self, key=None, output=False, input_key='samples',
                 output_key='best_samples', **runopts):
        super(TrackMin, self).__init__(**runopts)
        if key is None:
            key = 'samples.first.energy'
        if isinstance(key, six.string_types):
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
class LoopUntilNoImprovement(Runnable):
    """Iterates :class:`~hybrid.core.Runnable` for up to `max_iter` times, or
    until a state quality metric, defined by the `key` function, shows no
    improvement for at least `convergence` number of iterations.

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

    """

    def __init__(self, runnable, max_iter=None, convergence=None,
                 max_time=None, key=None, **runopts):
        super(LoopUntilNoImprovement, self).__init__(**runopts)
        self.runnable = runnable
        self.max_iter = max_iter
        self.max_time = max_time
        self.convergence = convergence
        if key is None:
            key = attrgetter('samples.first.energy')
        self.key = key

        # preemptively check runnable's i/o dimensionality
        if runnable.validate_input and runnable.validate_output:
            if runnable.multi_input != runnable.multi_output:
                raise TypeError("runnable's input dimensionality does not match "
                                "the output dimensionality")

        # patch branch's I/O requirements based on the child component's requirements
        self.inputs = self.runnable.inputs
        self.multi_input = self.runnable.multi_input
        self.outputs = self.runnable.outputs
        self.multi_output = self.runnable.multi_output

    def __str__(self):
        return "Loop over {}".format(self.runnable)

    def __repr__(self):
        return ("{self.name}(runnable={self.runnable!r}, max_iter={self.max_iter!r}, "
                "convergence={self.convergence!r}, key={self.key!r})").format(self=self)

    def __iter__(self):
        return iter((self.runnable,))

    def iteration_update(self, iterno, cnt, input_state, output_state):
        """Implement "converge on unchanging output" behavior:

          - loop `max_iter` times, but bail-out earlier if output doesn't change
            (over input) for `convergence` number of iterations

          - each iteration starts with the previous result state

        Input: relevant counters and I/O states.
        Output: next input state and next counter values
        """

        if self.convergence is None:
            return iterno + 1, cnt, output_state

        input_energy = self.key(input_state)
        output_energy = self.key(output_state)

        logger.info("{name} Iteration(iterno={iterno}, output_state_energy={key})".format(
            name=self.name, iterno=iterno, key=output_energy))

        if output_energy == input_energy:
            cnt -= 1
        else:
            cnt = self.convergence

        return iterno + 1, cnt, output_state

    def next(self, state, **runopts):
        iterno = 0
        cnt = self.convergence or 0
        input_state = state
        output_state = input_state
        start = time.time()

        runopts['executor'] = immediate_executor

        while not self.stop_signal.is_set():
            output_state = self.runnable.run(input_state, **runopts).result()

            iterno, cnt, input_state = self.iteration_update(iterno, cnt, input_state, output_state)

            runtime = time.time() - start

            if self.max_iter is not None and iterno >= self.max_iter:
                break
            if self.max_time is not None and runtime >= self.max_time:
                break
            if self.convergence is not None and cnt <= 0:
                break

        return output_state

    def halt(self):
        self.runnable.stop()


class Loop(LoopUntilNoImprovement):
    pass


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

    """

    def __init__(self, runnable, max_iter=None, max_tries=None,
                 max_time=None, key=None, **runopts):
        super(LoopWhileNoImprovement, self).__init__(
            runnable=runnable, max_iter=max_iter, convergence=max_tries,
            max_time=max_time, key=key, **runopts)

    def iteration_update(self, iterno, cnt, input_state, output_state):
        """Implement "no-improvement count-down" behavior:

          - loop indefinitely, but bail-out if there's no improvement of output
            over input for `max_tries` number of iterations

          - each iteration uses the same input state, unless there was an improvement
            in this iteration, in which case, use the current output as next input

        Input: relevant counters and I/O states.
        Output: next input state and next counter values
        """

        if self.convergence is None:
            return iterno + 1, cnt, output_state

        input_energy = self.key(input_state)
        output_energy = self.key(output_state)

        logger.info("{name} Iteration(iterno={iterno}, output_state_energy={key})".format(
            name=self.name, iterno=iterno, key=output_energy))

        if output_energy >= input_energy:
            # no improvement, re-use the same input
            cnt -= 1
            next_input_state = input_state
        else:
            # improvement, use the better output for next input, restart local counter
            cnt = self.convergence
            next_input_state = output_state

        return iterno + 1, cnt, next_input_state


class Unwind(Runnable, traits.SIMO):
    """Iterates :class:`~hybrid.core.Runnable` until :exc:`EndOfStream` is
    raised, collecting all output states along the way.

    Note:
        the child runnable is called with run option ``silent_rewind=False``,
        and it is expected to raise :exc:`EndOfStream` on unwind completion.
    """

    def __init__(self, runnable, **runopts):
        if not isinstance(runnable, Runnable):
            raise TypeError("'runnable' is not instance of Runnable")

        super(Unwind, self).__init__(**runopts)
        self.runnable = runnable

        # preemptively check runnable's i/o dimensionality
        if runnable.validate_input and runnable.validate_output:
            if runnable.multi_input or runnable.multi_output:
                raise TypeError("single input, single output runnable required")

        # patch branch's I/O requirements based on the child component's requirements
        self.inputs = self.runnable.inputs
        self.outputs = self.runnable.outputs

    def __str__(self):
        return "Unwind {}".format(self.runnable)

    def __repr__(self):
        return ("{self.name}(runnable={self.runnable!r}").format(self=self)

    def __iter__(self):
        return iter((self.runnable,))

    def next(self, state, **runopts):
        output = States()
        runopts.update(executor=immediate_executor, silent_rewind=False)

        while True:
            try:
                state = self.runnable.run(state, **runopts).result()
                output.append(state)
            except EndOfStream:
                break

        return output


@stoppable
class Identity(Runnable):
    """Trivial identity runnable. The output is a direct copy of the input."""

    def next(self, state, racing_context=False, **runopts):
        # in a racing context, we don't want to be the winning branch
        if racing_context:
            self.stop_signal.wait()

        return state.updated()
