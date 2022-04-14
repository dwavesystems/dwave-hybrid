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

import copy
import random
import logging

import numpy as np
import networkx as nx
import dimod

from hybrid.core import Runnable, SampleSet, States
from hybrid.utils import flip_energy_gains, vstack_samplesets, hstack_samplesets
from hybrid import traits

__all__ = ['IdentityComposer', 'SplatComposer', 'GreedyPathMerge',
           'MergeSamples', 'ExplodeSamples', 'SliceSamples', 'AggregatedSamples',
           'IsoenergeticClusterMove', 'ICM']

logger = logging.getLogger(__name__)


class IdentityComposer(traits.SubsamplesComposer, traits.SISO, Runnable):
    """Copy `subsamples` to `samples` verbatim."""

    def next(self, state, **runopts):
        return state.updated(samples=state.subsamples)


class SplatComposer(traits.SubsamplesComposer, traits.SISO, Runnable):
    """A composer that overwrites current samples with subproblem samples.

    See :ref:`composers-examples`.
    """

    def next(self, state, **runopts):
        # update as many samples possible with partial samples from `state.subsamples`
        # the resulting number of samples will be limited with `len(state.subsamples)`

        samples = hstack_samplesets(state.samples, state.subsamples, bqm=state.problem)

        logger.debug("{name} subsamples (shape={ss_shape!r}) -> samples (shape={s_shape!r}), "
                     "sample energies changed {old_en} -> {new_en}".format(
                         name=self.name,
                         ss_shape=state.subsamples.record.shape,
                         s_shape=state.samples.record.shape,
                         old_en=state.samples.record.energy,
                         new_en=samples.record.energy))

        return state.updated(samples=samples)


class GreedyPathMerge(traits.SamplesProcessor, traits.MISO, Runnable):
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


# TODO: move MergeSamples, SliceSamples and AggregatedSamples to `processors` module?

class MergeSamples(traits.SamplesProcessor, traits.MISO, Runnable):
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

        logger.debug("{name} merging {n} input states into an output state "
                     "with a sample set of size {k}".format(
                         name=self.name, n=len(states), k=len(samples)))

        if runopts.pop('aggregate', self.aggregate):
            samples = samples.aggregate()
            logger.debug("{name} output samples aggregated".format(name=self.name))

        return states.first.updated(samples=samples)


class ExplodeSamples(traits.SamplesProcessor, traits.SIMO, Runnable):
    """Produce one output state per input sample.

    Example:
        This example uses Tabu sampler to produce two samples on a simple
        problem, and then ExplodeSamples to produce two states, each with one
        sample.

        >>> import dimod
        >>> import hybrid

        >>> workflow = hybrid.TabuProblemSampler(num_reads=2) | hybrid.ExplodeSamples()
        >>> state = hybrid.State(problem=dimod.BQM.from_ising({}, {'ab': 1}))

        >>> result = workflow.run(state).result()
        >>> len(result)
        2
    """

    def next(self, state, **runopts):
        samples = state.samples
        if not samples:
            raise ValueError("no input samples")

        states = States()
        n = len(samples)
        for start, stop in zip(range(n), range(1, n+1)):
            sample = samples.slice(start, stop, sorted_by=None)
            states.append(state.updated(samples=sample))

        return states


class SliceSamples(traits.SamplesProcessor, traits.SISO, Runnable):
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

        logger.debug("{} applying slice({}, {}, {}) sorted_by={!r}".format(
            self.name, start, stop, step, sorted_by))

        sliced = state.samples.slice(start, stop, step, sorted_by=sorted_by)

        return state.updated(samples=sliced)


class AggregatedSamples(traits.SamplesProcessor, traits.SISO, Runnable):
    """Aggregates or spreads ("un-aggregates") samples, controlled via
    ``aggregate`` boolean flag.

    Args:
        aggregate (bool, default=True):
            Aggregate samples vs. spread / un-aggregate them.

    """

    def __init__(self, aggregate=True, **runopts):
        super(AggregatedSamples, self).__init__(**runopts)
        self.aggregate = aggregate

    @staticmethod
    def spread(samples):
        """Multiplies each sample its num_occurrences times."""

        record = samples.record
        labels = samples.variables

        sample = np.repeat(record.sample, repeats=record.num_occurrences, axis=0)
        energy = np.repeat(record.energy, repeats=record.num_occurrences, axis=0)
        num_occurrences = np.ones(sum(record.num_occurrences))

        return SampleSet.from_samples(
            samples_like=(sample, labels), vartype=samples.vartype,
            energy=energy, num_occurrences=num_occurrences,
            info=copy.deepcopy(samples.info))

    def next(self, state, **runopts):
        aggregate = bool(runopts.pop('aggregate', self.aggregate))
        logger.debug("{}(aggregate={})".format(self.name, aggregate))

        if aggregate:
            samples = state.samples.aggregate()
        else:
            samples = self.spread(state.samples)

        return state.updated(samples=samples)


class IsoenergeticClusterMove(traits.SamplesProcessor, traits.ProblemSampler,
                              traits.MIMO, Runnable):
    """Isoenergetic cluster move (ICM), also know as Houdayer move.

    ICM creates two new samples from a pair of samples by identifying, among
    connected variables, clusters with exactly complementary values and swapping
    one such randomly chosen cluster between the two samples. The total energy
    of the two samples remains unchanged, yet such moves on variables reasonably
    grouped together can enable better exploration of the solution space.

    Args:
        seed (int, optional, default=None/current time):
            Pseudo-random number generator seed.

    Input:
        :class:`~hybrid.core.States`:
            Two states with at least one sample each. First state should also
            contain a relevant problem.

    Output:
        :class:`~hybrid.core.States`:
            Two states from input with updated first sample in each.

    """

    def __init__(self, seed=None, **runopts):
        super(IsoenergeticClusterMove, self).__init__(**runopts)

        # initialize random seed and store it for reference
        self.seed = seed
        self.random = random.Random(seed)

    def next(self, states):
        """ICM between two first samples in the first two input states."""

        if len(states) > 2:
            raise ValueError("exactly two input states required")

        inp1, inp2 = states
        bqm = inp1.problem

        ss1 = inp1.samples.change_vartype(dimod.BINARY, inplace=False)
        ss2 = inp2.samples.change_vartype(dimod.BINARY, inplace=False)

        # sanity check: we operate on the same set of variables
        if ss1.variables ^ ss2.variables:
            raise ValueError("input samples not over the same set of variables")

        # reorder variables, if necessary
        # (use sequence comparison, not set)
        variables = list(ss1.variables)
        if ss2.variables != variables:
            reorder = [ss2.variables.index(v) for v in variables]
            record = ss2.record[:, reorder]
            ss2 = dimod.SampleSet(record, variables, ss2.info, ss2.vartype)

        # samples' symmetric difference (XOR)
        # (form clusters of input variables with opposite values)
        sample1 = ss1.record.sample[0]
        sample2 = ss2.record.sample[0]
        symdiff = sample1 ^ sample2

        # for cluster detection we'll use a reduced problem graph
        graph = dimod.to_networkx_graph(bqm)
        # note: instead of numpy mask indexing of `notcluster`, we enumerate
        # non-cluster variables manually to avoid conversion of potentially
        # unhashable variable names to numpy types
        notcluster = [v for v, d in zip(variables, symdiff) if d == 0]
        graph.remove_nodes_from(notcluster)

        # pick a random variable that belongs to a cluster, then select the cluster
        node = self.random.choice(list(graph.nodes))
        cluster = nx.node_connected_component(graph, node)

        # flip variables from `cluster` in both input samples
        flipper = np.array([1 if v in cluster else 0 for v in variables])
        ss1.record.sample[0] ^= flipper
        ss2.record.sample[0] ^= flipper

        # change vartype back to input's type
        ss1.change_vartype(inp1.samples.vartype)
        ss2.change_vartype(inp2.samples.vartype)

        # update sampleset's energies
        ss1.record.energy = bqm.energies(ss1)
        ss2.record.energy = bqm.energies(ss2)

        return States(inp1.updated(samples=ss1), inp2.updated(samples=ss2))


ICM = IsoenergeticClusterMove