# Copyright 2018 D-Wave Systems Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import concurrent.futures
from operator import attrgetter

from hades.core import Runnable

import logging
logger = logging.getLogger(__name__)


class RacingBranches(Runnable):

    def __init__(self, *branches, endomorphic=True):
        """If known upfront codomain for all branches equals domain, state
        can safely be mixed in with branches' results. Otherwise set
        `endomorphic=False`.
        """
        self.branches = branches
        self.endomorphic = endomorphic

    @property
    def name(self):
        return "{}({})".format(self.__class__.__name__,
                               ", ".join(branch.name for branch in self.branches))

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

    def __init__(self, fn=None):
        """Return the state which minimizes the objective function `fn`."""
        if fn is None:
            fn = attrgetter('samples.first.energy')
        self.fn = fn

    def iterate(self, states):
        # debug info
        for s in states:
            logger.debug("State: arg={arg}, debug={s.debug!r}".format(arg=self.fn(s), s=s))

        return min(states, key=self.fn)


class SimpleIterator(Runnable):

    def __init__(self, runnable, max_iter=1000, convergence=10):
        self.runnable = runnable
        self.max_iter = max_iter
        self.convergence = convergence

    @property
    def name(self):
        return "{}({})".format(self.__class__.__name__, self.runnable.name)

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
