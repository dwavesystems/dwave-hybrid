from hades.core import Runnable, SampleSet
from hades.profiling import tictoc
from hades.utils import updated_sample

import logging
logger = logging.getLogger(__name__)


class SplatComposer(Runnable):
    """A composer that overwrites current samples with subproblem samples.

    Args:
        bqm (:obj:`.BinaryQuadraticModel`): Binary quadratic model.

    Examples:
        This example runs one iteration of a `SplatComposer`, overwriting an initial
        solution to a 6-variable binary quadratic model of all zeros with a solution to
        a 3-variable subproblem that was manually set to all ones.

        >>> import dimod           # Create a binary quadratic model
        >>> bqm = dimod.BinaryQuadraticModel({t: 0 for t in range(6)},
        ...                                  {(t, (t+1) % 6): 1 for t in range(6)},
        ...                                  0, 'BINARY')
        >>> state0 = core.State.from_sample(min_sample(bqm), bqm)
        >>> subsamples = dimod.Response.from_samples([{3: 1, 4: 1, 5: 1}],
        ...                                         {'energy': [-20]}, {}, dimod.BINARY)
        >>> state1.updated(ctx=dict(subsamples=subsamples, debug=dict(sampler="Manual update")))
        >>> composed_state = SplatComposer(bqm).iterate(state1)
        >>> print(composed_state.samples)      # doctest: +SKIP
        Response(rec.array([([0, 0, 0, 1, 1, 1], 2, 1)],
          dtype=[('sample', 'i1', (6,)), ('energy', '<i4'), ('num_occurrences', '<i4')]),
          [0, 1, 2, 3, 4, 5], {}, 'BINARY')
    """

    def __init__(self, bqm):
        self.bqm = bqm

    @tictoc('splat_compose')
    def iterate(self, state):
        """Run one iteration of an instantiated :class:`SplatComposer`.

        Examples:
        
        >>> composed_state = SplatComposer(bqm).iterate(state1)
        """
        # update the first sample in `state.sampleset`, inplace
        # XXX: assume one global sample, one subsample
        # TODO: generalize
        sample = next(state.samples.change_vartype(state.ctx['subsamples'].vartype).samples())
        subsample = next(state.ctx['subsamples'].samples())
        composed_sample = updated_sample(sample, subsample)
        composed_energy = self.bqm.energy(composed_sample)
        return state.updated(
            samples=SampleSet.from_sample(composed_sample, state.samples.vartype, composed_energy),
            debug=dict(composer=self.name))
