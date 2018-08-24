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
        sample = next(state.samples.change_vartype(state.ctx['subsamples'].vartype).samples())
        subsample = next(state.ctx['subsamples'].samples())
        composed_sample = updated_sample(sample, subsample)
        composed_energy = self.bqm.energy(composed_sample)
        return state.updated(
            samples=SampleSet.from_sample(composed_sample, state.samples.vartype, composed_energy),
            debug=dict(composer=self.__class__.__name__))
