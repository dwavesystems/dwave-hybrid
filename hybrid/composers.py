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

import logging

from hybrid.core import Runnable, SampleSet
from hybrid.utils import updated_sample
from hybrid import traits

__all__ = ['IdentityComposer', 'SplatComposer']

logger = logging.getLogger(__name__)


class IdentityComposer(Runnable, traits.SubproblemComposer):
    """Copy `subsamples` to `samples` verbatim."""

    def next(self, state):
        return state.updated(samples=state.subsamples)


class SplatComposer(Runnable, traits.SubproblemComposer):
    """A composer that overwrites current samples with subproblem samples."""

    def next(self, state):
        # update the first sample in `state.sampleset`, inplace
        # XXX: assume one global sample, one subsample
        # TODO: generalize
        sample = next(state.samples.change_vartype(state.subsamples.vartype).samples())
        subsample = next(state.subsamples.samples())
        composed_sample = updated_sample(sample, subsample)
        composed_energy = state.problem.energy(composed_sample)
        return state.updated(
            samples=SampleSet.from_samples(composed_sample, state.samples.vartype, composed_energy))
