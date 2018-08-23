from hades.core import Runnable, SampleSet
from hades.profiling import tictoc
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

    @tictoc('splat_compose')
    def iterate(self, state):
        # update the first sample in `state.sampleset`, inplace
        # XXX: assume one global sample, one subsample
        # TODO: generalize
        record = state.samples.change_vartype(state.ctx['subsamples'].vartype).record[0]
        sample = updated_sample(record.sample, state.ctx['subsamples'].record[0].sample)
        energy = self.bqm.energy(sample)
        return state.updated(
            samples=SampleSet.from_sample(sample, state.samples.vartype, energy),
            debug=dict(composer=self.__class__.__name__))
