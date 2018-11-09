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
        branches (:class:`Runnable`): Comma-separated branches.
        endomorphic (bool): Set to False if you are not sure that the codomain of all branches
            is the domain.

    Examples:
        >>> RacingBranches(                  # doctest: +SKIP
            InterruptableTabuSampler(),
            EnergyImpactDecomposer(max_size=2) |
            QPUSubproblemAutoEmbeddingSampler()| SplatComposer()
            ) | ArgMinFold()

    """

    def __init__(self, *branches, endomorphic=True):
        """If known upfront codomain for all branches equals domain, state
        can safely be mixed in with branches' results. Otherwise set
        `endomorphic=False`.
        """
        super(RacingBranches, self).__init__()
        self.branches = branches
        self.endomorphic = endomorphic

    def __str__(self):
        return " !! ".join("({})".format(b) for b in self.branches) or "(zero racing branches)"

    def __repr__(self):
        return "{}({})".format(self.name, ", ".join(map(repr, self.branches)))

    def iterate(self, state):
        futures = [branch.run(state.updated(debug=None)) for branch in self.branches]

        states = []
        if self.endomorphic:
            states.append(state)
        for f in concurrent.futures.as_completed(futures):
            # as soon as one is done, stop all others
            for branch in self.branches:
                branch.stop()
            states.append(f.result())

        return states

    def stop(self):
        for branch in self.branches:
            branch.stop()


class ArgMinFold(Runnable):
    """
    Select the :class:`State` of the :class:`Branch` that minimizes the problem or subproblem.

    Examples:
        >>> RacingBranches(                  # doctest: +SKIP
            InterruptableTabuSampler(),
            EnergyImpactDecomposer(max_size=2) |
            QPUSubproblemAutoEmbeddingSampler()| SplatComposer()
            ) | ArgMinFold()

    """

    def __init__(self, fn=None):
        """Return the state which minimizes the objective function `fn`."""
        super(ArgMinFold, self).__init__()
        if fn is None:
            fn = attrgetter('samples.first.energy')
        self.fn = fn

    def __str__(self):
        return "[]>"

    def __repr__(self):
        return "{}(fn={!r})".format(self.name, self.fn)

    def iterate(self, states):
        # debug info
        for s in states:
            logger.debug("State: arg={arg}, debug={s.debug!r}".format(arg=self.fn(s), s=s))

        return min(states, key=self.fn)


class SimpleIterator(Runnable):

    def __init__(self, runnable, max_iter=1000, convergence=10):
        super(SimpleIterator, self).__init__()
        self.runnable = runnable
        self.max_iter = max_iter
        self.convergence = convergence

    def __str__(self):
        return "Loop over {}".format(self.runnable)

    def __repr__(self):
        return ("{self.name}(runnable={self.runnable!r}, max_iter={self.max_iter!r}, "
                "convergence={self.convergence!r})").format(self=self)

    def iterate(self, state):
        last = state
        cnt = self.convergence

        for iterno in range(self.max_iter):
            state = self.runnable.run(state).result()

            logger.info("iterno={i}, State: energy={s.samples.first.energy}, debug={s.debug!r}".format(i=iterno, s=state))

            if state.samples.first.energy == last.samples.first.energy:
                cnt -= 1
            else:
                cnt = self.convergence
            if cnt <= 0:
                break

            last = state

        return state.updated(debug=dict(n_iter=iterno+1))
