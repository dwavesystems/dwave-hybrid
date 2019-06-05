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
from hybrid.utils import updated_sample, flip_energy_gains, vstack_samplesets
from hybrid import traits

__all__ = ['IdentityComposer', 'SplatComposer', 'GreedyPathMerge', 'MergeSamples', 'SliceSamples']

logger = logging.getLogger(__name__)


class IdentityComposer(Runnable, traits.SubsamplesComposer):
    """Copy `subsamples` to `samples` verbatim."""

    def next(self, state, **runopts):
        return state.updated(samples=state.subsamples)


class SplatComposer(Runnable, traits.SubsamplesComposer):
    """A composer that overwrites current samples with subproblem samples.

    See :ref:`composers-examples`.
    """

    def next(self, state, **runopts):
        # update the first sample in `state.sampleset`, inplace
        # XXX: assume one global sample, one subsample
        # TODO: generalize
        sample = next(iter(state.samples.change_vartype(state.subsamples.vartype).samples()))
        subsample = next(iter(state.subsamples.samples()))
        composed_sample = updated_sample(sample, subsample)
        composed_energy = state.problem.energy(composed_sample)
        return state.updated(
            samples=SampleSet.from_samples(composed_sample, state.samples.vartype, composed_energy))


class GreedyPathMerge(Runnable, traits.MISO, traits.SamplesIntaking, traits.SamplesProducing):
    """Dialectic-search merge operation [KS]_. Generates a path from one input
    state, representing the thesis, to another input state, representing the
    antithesis, using a greedy method of single bit flips selected by decreasing
    energy.

    Returns the best sample on the path, which represents the synthesis.

    Note: only the lowest-energy sample, is considered from either input state.

    See :ref:`composers-examples`.

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


# TODO: move MergeSamples and SliceSamples to `ops` module?

class MergeSamples(Runnable, traits.MISO, traits.SamplesIntaking, traits.SamplesProducing):
    """Merge multiple input states by concatenating samples from all the states
    in to the first state.

    Args:
        aggregate (bool, default=False):
            Aggregate samples after merging.

    Example:
        This example runs two branches, a classical simulated annealing and a
        tabu search, acquiring one sample per branch. It then merges the
        samples, producing the final state with a sampleset of size two.

        >>> import dimod
        >>> import hybrid

        >>> workflow = hybrid.Parallel(
        ...     hybrid.SimulatedAnnealingProblemSampler(num_reads=1),
        ...     hybrid.TabuProblemSampler(num_reads=1)
        ... ) | hybrid.MergeSamples()

        >>> state = hybrid.State.from_problem(
        ...    dimod.BinaryQuadraticModel.from_ising({}, {'ab': 1}))

        >>> result = workflow.run(state).result()
        >>> len(result.samples)
        2
    """

    def __init__(self, aggregate=False, **runopts):
        super(MergeSamples, self).__init__(**runopts)
        self.aggregate = aggregate

    def next(self, states, **runopts):
        if len(states) < 1:
            raise ValueError("no input states")

        samples = vstack_samplesets(*[s.samples for s in states])

        if runopts.pop('aggregate', self.aggregate):
            samples = samples.aggregate()

        return states.first.updated(samples=samples)


class SliceSamples(Runnable, traits.SISO, traits.SamplesIntaking, traits.SamplesProducing):
    """Slice input sampleset acting on samples in a selected order.

    Args:
        start (int, optional, default=None):
            Start index for `slice`.

        stop (int):
            Stop index for `slice`.

        step (int, optional, default=None):
            Step value for `slice`.

        sorted_by (str/None, optional, default='energy'):
            Selects the record field used to sort the samples before slicing.

    Examples:
        Truncate to 5 with lowest energy:

        >>> top5 = SliceSamples(5)

        Truncate to 5 with highest energy:

        >>> bottom5 = SliceSamples(-5, None)

        Slice the sample set ordered by `num_occurrences`, instead by `energy`:

        >>> five_with_highest_num_occurrences = SliceSamples(-5, None, sorted_by='num_occurrences')

        Halve the sample set by selecting only every other sample:

        >>> odd = SliceSamples(None, None, 2)

    """

    def __init__(self, *slice_args, **runopts):
        sorted_by = runopts.pop('sorted_by', 'energy')

        # follow the Python slice syntax
        if slice_args:
            slicer = slice(*slice_args)
        else:
            slicer = slice(None)

        # but also allow extension via kwargs
        start = runopts.pop('start', slicer.start)
        stop = runopts.pop('stop', slicer.stop)
        step = runopts.pop('step', slicer.step)

        super(SliceSamples, self).__init__(**runopts)

        self.slice = slice(start, stop, step)
        self.sorted_by = sorted_by

    def next(self, state, **runopts):
        # allow slice override via runopts
        start = runopts.pop('start', self.slice.start)
        stop = runopts.pop('stop', self.slice.stop)
        step = runopts.pop('step', self.slice.step)
        sorted_by = runopts.pop('sorted_by', self.sorted_by)

        sliced = state.samples.slice(start, stop, step, sorted_by=sorted_by)
        return state.updated(samples=sliced)
