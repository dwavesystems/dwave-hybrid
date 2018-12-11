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

import logging
import concurrent.futures
from operator import attrgetter
from functools import partial
from itertools import chain

import six

from hybrid.core import Runnable, States, Present
from hybrid import traits

__all__ = ['Branch', 'RacingBranches', 'Race', 'Map', 'Lambda', 'ArgMin', 'Loop', 'SimpleIterator']

logger = logging.getLogger(__name__)


class Branch(Runnable):
    """Sequentially executed :class:`Runnable` components.

    Args:
        components (iterable of :class:`Runnable`): Complete processing sequence to
            update a current set of samples, such as: :code:`decomposer | sampler | composer`.

    Examples:
        This example runs one iteration of a branch comprising a decomposer, local Tabu solver,
        and a composer. A 10-variable binary quadratic model is decomposed by the energy
        impact of its variables into a 6-variable subproblem to be sampled twice
        with a manually set initial state of all -1 values.

        >>> import dimod           # Create a binary quadratic model
        >>> bqm = dimod.BinaryQuadraticModel({t: 0 for t in range(10)},
        ...                                  {(t, (t+1) % 10): 1 for t in range(10)},
        ...                                  0, 'SPIN')
        >>> # Run one iteration on a branch
        >>> branch = (EnergyImpactDecomposer(max_size=6, min_gain=-10) |
        ...           TabuSubproblemSampler(num_reads=2) |
        ...           SplatComposer())
        >>> new_state = branch.next(State.from_sample(min_sample(bqm), bqm))
        >>> print(new_state.subsamples)      # doctest: +SKIP
        Response(rec.array([([-1,  1, -1,  1, -1,  1], -5., 1),
           ([ 1, -1,  1, -1, -1,  1], -5., 1)],
        >>> # Above response snipped for brevity

    """

    def __init__(self, components=(), *args, **kwargs):
        super(Branch, self).__init__(*args, **kwargs)
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

    def next(self, state):
        """Start an iteration of an instantiated :class:`Branch`.

        Accepts a state and returns a new state.

        Args:
            state (:class:`State`):
                Computation state passed to the first component of the branch.

        Examples:
            This code snippet runs one iteration of a branch to produce a new state::

                new_state = branch.next(core.State.from_sample(min_sample(bqm), bqm)

        """
        for component in self.components:
            state = component.run(state, defer=False)
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
    """Runs parallel :class:`Branch` classes.

    Args:
        *branches ([:class:`Runnable`]):
            Comma-separated branches.
        endomorphic (bool):
            Set to ``False`` if you are not sure that the codomain of all branches
            is the domain; for example, if there might be a mix of subproblems
            and problems moving between components.

    Note:
        `RacingBranches` is also available as `Race`.

    Examples:
        This example runs two branches: a classical tabu search interrupted by
        samples of subproblems returned from a D-Wave system.

        >>> RacingBranches(                     # doctest: +SKIP
                InterruptableTabuSampler(),
                EnergyImpactDecomposer(max_size=2)
                | QPUSubproblemAutoEmbeddingSampler()
                | SplatComposer()
            ) | ArgMin()

    """

    def __init__(self, *branches, **kwargs):
        """If known upfront codomain for all branches equals domain, state
        can safely be mixed in with branches' results. Otherwise set
        `endomorphic=False`.
        """
        super(RacingBranches, self).__init__()
        self.branches = branches
        self.endomorphic = kwargs.get('endomorphic', True)

        if not self.branches:
            raise ValueError("racing branches requires at least one branch")

        # patch components's I/O requirements based on the subcomponents' requirements

        # RB's input has to satisfy all branches' input
        self.inputs = set.union(*(branch.inputs for branch in self.branches))

        # RB's output will be one of the branches' output, but the only guarantee we
        # can make upfront is the largest common subset of all outputs
        self.outputs = set.intersection(*(branch.outputs for branch in self.branches))

    def __str__(self):
        return " !! ".join("({})".format(b) for b in self) or "(zero racing branches)"

    def __repr__(self):
        return "{}{!r}".format(self.name, tuple(self))

    def __iter__(self):
        return iter(self.branches)

    def next(self, state):
        """Execute one blocking iteration of an instantiated :class:`RacingBranches`."""

        futures = [branch.run(state.updated()) for branch in self.branches]

        states = States()
        if self.endomorphic:
            states.append(state)

        # as soon as one is done, stop all others
        done, _ = concurrent.futures.wait(
            futures,
            return_when=concurrent.futures.FIRST_COMPLETED)
        self.stop()

        # debug info
        idx = futures.index(done.pop())
        branch = self.branches[idx]
        logger.debug("{name} won idx={idx} branch={branch!r}".format(
            name=self.name, idx=idx, branch=branch))

        # collect resolved states (in original order, not completion order!)
        for f in futures:
            states.append(f.result())

        return states

    def halt(self):
        """Terminate an iteration of an instantiated :class:`RacingBranches`."""
        for branch in self.branches:
            branch.stop()


Race = RacingBranches


class Map(Runnable, traits.MIMO):
    """Runs a specified runnable in parallel on all input states.

    Args:
        runnable (:class:`Runnable`):
            A runnable executed for every input state.

    Examples:
        This example runs `TabuProblemSampler` on two input states in parallel,
        returning when both are done.

        >>> Map(TabuProblemSampler()).run([State(problem=bqm1), State(problem=bqm2)])    # doctest: +SKIP
        [<state_1_with_solution>, <state_2_with_solution>]

    """

    def __init__(self, runnable, *args, **kwargs):
        if not isinstance(runnable, Runnable):
            raise TypeError("'runnable' is not instance of Runnable")

        super(Map, self).__init__(*args, **kwargs)
        self.runnable = runnable

        # patch components's I/O requirements based on the subcomponents' requirements
        # TODO: automate
        self.inputs = runnable.inputs
        self.outputs = runnable.outputs

    def __str__(self):
        return "[]()"

    def __repr__(self):
        return "{}(runnable={!r})".format(self.name, self.runnable)

    def __iter__(self):
        return iter(tuple(self.runnable))

    def next(self, states):
        self._futures = [self.runnable.run(state) for state in states]

        concurrent.futures.wait(self._futures,
                                return_when=concurrent.futures.ALL_COMPLETED)

        return States(*(f.result() for f in self._futures))

    def halt(self):
        for future in self._futures:
            future.cancel()


class Lambda(Runnable):
    """Creates a runnable on fly, given just its `next` function (optionally
    `init` and `error` functions can be specified too).

    Args:
        next (callable):
            Implementation of runnable's `next` method, provided as a callable
            (usually lambda expression for simple operations). Signature of the
            callable has to match the signature of :meth:`Runnable.next()`, i.e.
            it accepts two arguments: runnable instance and state instance.
        error (callable):
            Implementation of runnable's `error` method. See :meth:`Runnable.error`.
        init (callable):
            Implementation of runnable's `init` method. See :meth:`Runnable.init`.

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

    def __init__(self, next, error=None, init=None, *args, **kwargs):
        if not callable(next):
            raise TypeError("'next' is not callable")
        if error is not None and not callable(error):
            raise TypeError("'error' is not callable")
        if init is not None and not callable(init):
            raise TypeError("'init' is not callable")

        super(Lambda, self).__init__(*args, **kwargs)

        # bind to self
        self.next = partial(next, self)
        if error is not None:
            self.error = partial(error, self)
        if init is not None:
            self.init = partial(init, self)

        # keep a copy for inspection (without cycles to `self`)
        self._next = next
        self._error = error
        self._init = init

    def __repr__(self):
        return "{}(next={!r}, error={!r}, init={!r})".format(
            self.name, self._next, self._error, self._init)


class ArgMin(Runnable, traits.MISO):
    """Selects the best state from a sequence of states (:class:`States`).

    Args:
        key (callable/str):
            Best state is judged according to a metric defined with a `key`.
            `key` can be a `callable` with a signature::

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
                EnergyImpactDecomposer(max_size=2)
                | QPUSubproblemAutoEmbeddingSampler()
                | SplatComposer()
            ) | ArgMin()

    """

    def __init__(self, key=None):
        """Return the state which minimizes the objective function `key`."""
        super(ArgMin, self).__init__()
        if key is None:
            key = 'samples.first.energy'
        if isinstance(key, six.string_types):
            key = attrgetter(key)
        self.key = key

    def __str__(self):
        return "[]>"

    def __repr__(self):
        return "{}(key={!r})".format(self.name, self.key)

    def next(self, states):
        """Execute one blocking iteration of an instantiated :class:`ArgMin`."""
        # debug info
        for idx, state in enumerate(states):
            logger.debug("{name} State(idx={idx}, arg={arg})".format(
                name=self.name, idx=idx, arg=self.key(state)))

        return min(states, key=self.key)


class Loop(Runnable):
    """Iterates `runnable` for up to `max_iter` times, or until a state quality
    metric, defined by the `key` function, shows no improvement for at least
    `convergence` time."""

    def __init__(self, runnable, max_iter=1000, convergence=10, key=None):
        super(Loop, self).__init__()
        self.runnable = runnable
        self.max_iter = max_iter
        self.convergence = convergence
        if key is None:
            key = attrgetter('samples.first.energy')
        self.key = key

        # preemptively check runnable's i/o dimensionality
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

    def next(self, state):
        """Execute one blocking iteration of an instantiated :class:`Loop`."""
        last = state
        last_quality = self.key(last)
        cnt = self.convergence

        for iterno in range(self.max_iter):
            state = self.runnable.run(state).result()
            state_quality = self.key(state)

            logger.info("{name} Iteration(iterno={iterno}, best_state_quality={key})".format(
                name=self.name, iterno=iterno, key=state_quality))

            if state_quality == last_quality:
                cnt -= 1
            else:
                cnt = self.convergence
            if cnt <= 0:
                break

            last_quality = state_quality

        return state


SimpleIterator = Loop
