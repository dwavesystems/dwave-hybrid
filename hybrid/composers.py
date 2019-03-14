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
from hybrid.utils import updated_sample, flip_energy_gains
from hybrid import traits

__all__ = ['IdentityComposer', 'SplatComposer', 'GreedyPathMerge']

logger = logging.getLogger(__name__)


class IdentityComposer(Runnable, traits.SubsamplesComposer):
    """Copy `subsamples` to `samples` verbatim."""

    def next(self, state, **runopts):
        return state.updated(samples=state.subsamples)


class SplatComposer(Runnable, traits.SubsamplesComposer):
    """A composer that overwrites current samples with subproblem samples.

    Examples:
        See examples on https://docs.ocean.dwavesys.com/projects/hybrid/en/latest/reference/composers.html#examples.
    """

    def next(self, state, **runopts):
        # update the first sample in `state.sampleset`, inplace
        # XXX: assume one global sample, one subsample
        # TODO: generalize
        sample = next(state.samples.change_vartype(state.subsamples.vartype).samples())
        subsample = next(state.subsamples.samples())
        composed_sample = updated_sample(sample, subsample)
        composed_energy = state.problem.energy(composed_sample)
        return state.updated(
            samples=SampleSet.from_samples(composed_sample, state.samples.vartype, composed_energy))


class GreedyPathMerge(Runnable, traits.MISO, traits.SamplesIntaking, traits.SamplesProducing):
    """Dialectic-search merge operation [KS]_. Generates a path from one input state,
    representing the thesis, to another input state, representing the antithesis,
    using a greedy method of single bit flips selected by decreasing energy.

    Returns the best sample on the path, which represents the synthesis.

    Note: only the lowest-energy sample, is considered from either input state.

    Examples:
        See examples on https://docs.ocean.dwavesys.com/projects/hybrid/en/latest/reference/composers.html#examples.

    References
    ----------

    .. [KS] Kadioglu S., Sellmann M. (2009) Dialectic Search. In: Gent I.P. (eds)
        Principles and Practice of Constraint Programming - CP 2009. CP 2009.
        Lecture Notes in Computer Science, vol 5732. Springer, Berlin, Heidelberg
    """

    def next(self, states, **runopts):
        state_thesis, state_antithesis = states
        bqm = state_thesis.problem

        thesis = dict(state_thesis.samples.first.sample)
        thesis_en = state_thesis.samples.first.energy

        antithesis = dict(state_antithesis.samples.first.sample)

        synthesis = thesis.copy()
        synthesis_en = thesis_en

        # input sanity check
        # TODO: convert to hard input validation
        assert len(thesis) == len(antithesis)
        assert state_thesis.problem == state_antithesis.problem

        diff = {v for v in thesis if thesis[v] != antithesis[v]}

        while diff:
            flip_energies = flip_energy_gains(bqm, thesis, diff)
            en, v = flip_energies[-1]

            diff.remove(v)
            thesis[v] = antithesis[v]
            thesis_en += en

            if thesis_en <= synthesis_en:
                # note EQ also, because we want the latest thesis
                synthesis = thesis.copy()
                synthesis_en = thesis_en

        synthesis_samples = SampleSet.from_samples_bqm(synthesis, bqm)

        # calculation sanity check
        assert synthesis_samples.first.energy == synthesis_en

        return state_thesis.updated(samples=synthesis_samples)
