import concurrent.futures
from operator import attrgetter

from hades.core import Runnable

import logging
logger = logging.getLogger(__name__)


class RacingBranches(Runnable):

    def __init__(self, *branches):
        self.branches = branches

    def iterate(self, state):
        futures = [branch.run(state.updated(ctx=None, debug=None)) for branch in self.branches]

        states = [state]
        for f in concurrent.futures.as_completed(futures):
            # as soon as one is done, stop all others
            for branch in self.branches:
                branch.stop()
            states.append(f.result())

        return states


class ArgMinFold(Runnable):

    def __init__(self, fn=None):
        """Return the state which minimizes the objective function `fn`."""
        if fn is None:
            fn = attrgetter('samples.first.energy')
        self.fn = fn

    def iterate(self, states):
        # debug info
        for s in states:
            logger.debug("State: energy={s.samples.first.energy}, debug={s.debug!r}".format(s=s))

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
