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

from hades.core import Runnable, SampleSet
from hades.profiling import tictoc
from hades.utils import updated_sample
from hades import traits

import logging
logger = logging.getLogger(__name__)


class IdentityComposer(Runnable, traits.SubproblemComposer):
    """Copy `subsamples` to `samples` verbatim."""

    def __init__(self):
        super(IdentityComposer, self).__init__()

    @tictoc('identity_compose')
    def iterate(self, state):
        return state.updated(samples=state.subsamples, debug=dict(composer=self.name))


class SplatComposer(Runnable, traits.SubproblemComposer):
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

    def __init__(self):
        super(SplatComposer, self).__init__()

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
