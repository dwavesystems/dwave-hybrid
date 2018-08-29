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
