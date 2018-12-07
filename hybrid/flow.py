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

import concurrent.futures
from operator import attrgetter
from functools import partial

import six

from hybrid.core import Runnable, States
from hybrid import traits

import logging
logger = logging.getLogger(__name__)


class RacingBranches(Runnable, traits.SIMO):
    """Runs parallel :class:`Branch` classes.

    Args:
        *branches ([:class:`Runnable`]):
            Comma-separated branches.
        endomorphic (bool):
            Set to ``False`` if you are not sure that the codomain of all branches
            is the domain; for example, if there might be a mix of subproblems
            and problems moving between components.

    Examples:
        This example runs two branches: a classical tabu search interrupted by
        samples of subproblems returned from a D-Wave system.

        >>> RacingBranches(                     # doctest: +SKIP
                InterruptableTabuSampler(),
                EnergyImpactDecomposer(max_size=2)
                | QPUSubproblemAutoEmbeddingSampler()
                | SplatComposer()
            ) | ArgMinFold()

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
        # TODO: automate
        self.inputs = set.union(*(branch.inputs for branch in self.branches))
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

    def stop(self):
        """Terminate an iteration of an instantiated :class:`RacingBranches`."""
        for branch in self.branches:
            branch.stop()


class Map(Runnable, traits.MIMO):
    """Runs a specified runnable in parallel on all input states.

    Args:
        runnable (:class:`Runnable`):
            A runnable executed for every input state.

    Examples:
        This example runs `TabuProblemSampler` on two input states in parallel,
        returning when both are done.

        >>> Map(TabuProblemSampler).run([State(problem=bqm1), State(problem=bqm2)])
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

    def stop(self):
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

    Note: traits are not enforced, apart from the SISO requirement.

    Examples:
        This example creates and runs a simple runnable that multiplies state
        variables `a` and `b`, storing them in `c`.

        >>> Lambda(lambda _, s: s.updated(c=s.a * s.b)).run(State(a=2, b=3)).result()
        {'a': 2, 'b': 3, 'c': 6, ...}

        This example applies `x += 1` to a sequence of input states.

        >>> Map(Lambda(lambda _, s: s.updated(x=s.x + 1))).run(States(State(x=0), State(x=1))).result()
        [{'x': 1, ...}, {'x': 2, ...}]
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


class ArgMinFold(Runnable, traits.MISO):
    """Selects the best state from the list of states (output of
    :class:`RacingBranches`).

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
            ) | ArgMinFold()

    """

    def __init__(self, key=None):
        """Return the state which minimizes the objective function `key`."""
        super(ArgMinFold, self).__init__()
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
        """Execute one blocking iteration of an instantiated :class:`ArgMinFold`."""
        # debug info
        for idx, state in enumerate(states):
            logger.debug("{name} State(idx={idx}, arg={arg})".format(
                name=self.name, idx=idx, arg=self.key(state)))

        return min(states, key=self.key)


class SimpleIterator(Runnable):
    """Iterates `runnable` for up to `max_iter` times, or until a state quality
    metric, defined by the `key` function, shows no improvement for at least
    `convergence` time."""

    def __init__(self, runnable, max_iter=1000, convergence=10, key=None):
        super(SimpleIterator, self).__init__()
        self.runnable = runnable
        self.max_iter = max_iter
        self.convergence = convergence
        if key is None:
            key = attrgetter('samples.first.energy')
        self.key = key

    def __str__(self):
        return "Loop over {}".format(self.runnable)

    def __repr__(self):
        return ("{self.name}(runnable={self.runnable!r}, max_iter={self.max_iter!r}, "
                "convergence={self.convergence!r}, key={self.key!r})").format(self=self)

    def __iter__(self):
        return iter((self.runnable,))

    def next(self, state):
        """Execute one blocking iteration of an instantiated :class:`SimpleIterator`."""
        last = state
        last_key = self.key(last)
        cnt = self.convergence

        for iterno in range(self.max_iter):
            state = self.runnable.run(state).result()
            state_key = self.key(state)

            logger.info("{name} Iteration(iterno={iterno}, best_state_key={key})".format(
                name=self.name, iterno=iterno, key=state_key))

            if state_key == last_key:
                cnt -= 1
            else:
                cnt = self.convergence
            if cnt <= 0:
                break

            last_key = state_key

        return state
