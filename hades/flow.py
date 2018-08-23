import concurrent.futures
from operator import attrgetter

from hades.core import Runnable

import logging
logger = logging.getLogger(__name__)


class RacingBranches(Runnable):

    def __init__(self, branches):
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

    def __init__(self):
        pass

    def iterate(self, states):
        # debug info
        for s in states:
            logger.debug("State: energy={s.samples.first.energy}, debug={s.debug!r}".format(s=s))

        return min(states, key=attrgetter('samples.first.energy'))
