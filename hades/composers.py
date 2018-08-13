from hades.core import Runnable, State, Sample
from hades.utils import updated_sample

import logging
logger = logging.getLogger(__name__)


class SplatComposer(Runnable):
    """Copies subsample values over sample.

    Requires state to have:
    - subsample
    """

    def __init__(self, bqm):
        self.bqm = bqm

    def iterate(self, state):
        composed_sample = updated_sample(state.sample.values, state.ctx['subsample'])
        composed_energy = self.bqm.energy(composed_sample)
        return state.updated(sample=Sample(composed_sample, composed_energy))
