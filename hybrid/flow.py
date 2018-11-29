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

from hybrid.core import Runnable

import logging
logger = logging.getLogger(__name__)


class RacingBranches(Runnable):
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

    def __str__(self):
        return " !! ".join("({})".format(b) for b in self) or "(zero racing branches)"

    def __repr__(self):
        return "{}{!r}".format(self.name, tuple(self))

    def __iter__(self):
        return iter(self.branches)

    def next(self, state):
        """Execute one blocking iteration of an instantiated :class:`RacingBranches`."""

        futures = [branch.run(state.updated()) for branch in self.branches]

        states = []
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


class ArgMinFold(Runnable):
    """Selects the best state from the list of states (output of
    :class:`RacingBranches`).

    Best state is judged according to a metric defined with a "key" function, `key`.
    By default, `key` favors states containing a sample with minimal energy.

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
            key = attrgetter('samples.first.energy')
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
