from hades.core import Runnable, SampleSet
from hades.profiling import tictoc
from hades.utils import updated_sample

import logging
logger = logging.getLogger(__name__)


class SplatComposer(Runnable):
    """A composer that overwrites current samples with subproblem samples.

    Examples:
        This example runs one iteration of a `SplatComposer`, overwriting an initial
        solution to a 6-variable binary quadratic model of all zeros with a solution to
        a 3-variable subproblem that was manually set to all ones.

        >>> import dimod           # Create a binary quadratic model
        >>> bqm = dimod.BinaryQuadraticModel({t: 0 for t in range(6)},
        ...                                  {(t, (t+1) % 6): 1 for t in range(6)},
        ...                                  0, 'BINARY')
        >>> state0 = State.from_sample(min_sample(bqm), bqm)
        >>> state1 = state0.updated(subsamples=SampleSet.from_sample({3: 1, 4: 1, 5: 1}, 'BINARY'))
        >>> composed_state = SplatComposer().run(state1).result()
        >>> print(composed_state.samples)      # doctest: +SKIP
        Response(rec.array([([0, 0, 0, 1, 1, 1], 1, 2)],
                dtype=[('sample', 'i1', (6,)), ('num_occurrences', '<i8'), ('energy', '<i8')]), [0, 1, 2, 3, 4, 5], {}, 'BINARY')

    """

    @tictoc('splat_compose')
    def iterate(self, state):
        # update the first sample in `state.sampleset`, inplace
        # XXX: assume one global sample, one subsample
        # TODO: generalize
        sample = next(state.samples.change_vartype(state.subsamples.vartype).samples())
        subsample = next(state.subsamples.samples())
        composed_sample = updated_sample(sample, subsample)
        composed_energy = state.problem.energy(composed_sample)
        return state.updated(
            samples=SampleSet.from_sample(composed_sample, state.samples.vartype, composed_energy),
            debug=dict(composer=self.name))
