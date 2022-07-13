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

"""
Classical and quantum :class:`.Runnable`
`dimod <https://docs.ocean.dwavesys.com/en/stable/docs_dimod/sdk_index.html>`_
samplers for problems and subproblems.
"""

import time
import logging
import threading
from collections import namedtuple

import dimod
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import AutoEmbeddingComposite, FixedEmbeddingComposite
from dwave.preprocessing.composites import SpinReversalTransformComposite

from tabu import TabuSampler
from neal import SimulatedAnnealingSampler
from greedy import SteepestDescentSolver
from minorminer.busclique import busgraph_cache

from hybrid.core import Runnable, SampleSet
from hybrid.flow import Loop
from hybrid.utils import random_sample
from hybrid import traits

__all__ = [
    'QPUSubproblemExternalEmbeddingSampler', 'SubproblemCliqueEmbedder',
    'QPUSubproblemAutoEmbeddingSampler', 'ReverseAnnealingAutoEmbeddingSampler',
    'SimulatedAnnealingSubproblemSampler', 'InterruptableSimulatedAnnealingSubproblemSampler',
    'SimulatedAnnealingProblemSampler', 'InterruptableSimulatedAnnealingProblemSampler',
    'TabuSubproblemSampler', 'TabuProblemSampler', 'InterruptableTabuSampler',
    'SteepestDescentSubproblemSampler', 'SteepestDescentProblemSampler',
    'GreedySubproblemSampler', 'GreedyProblemSampler',  # aliases
    'RandomSubproblemSampler',
]

logger = logging.getLogger(__name__)


class QPUSubproblemExternalEmbeddingSampler(traits.SubproblemSampler,
                                            traits.EmbeddingIntaking,
                                            traits.SISO, Runnable):
    r"""A quantum sampler for a subproblem with a defined minor-embedding.

    Note:
        Externally supplied embedding must be present in the input state.

    Args:
        num_reads (int, optional, default=100):
            Number of states (output solutions) to read from the sampler.

        qpu_sampler (:class:`dimod.Sampler`, optional, default=\ :class:`~dwave.system.samplers.DWaveSampler()` ):
            Quantum sampler such as a D-Wave system.

        sampling_params (dict):
            Dictionary of keyword arguments with values that will be used
            on every call of the (external-embedding-wrapped QPU) sampler.

        logical_srt (int, optional, default=False):
            Perform a spin-reversal transform over the logical space.

    See :ref:`samplers-examples`.
    """

    def __init__(self, num_reads=100, qpu_sampler=None, sampling_params=None,
                 logical_srt=False, **runopts):
        super(QPUSubproblemExternalEmbeddingSampler, self).__init__(**runopts)

        self.num_reads = num_reads

        if qpu_sampler is None:
            qpu_sampler = DWaveSampler()
        self.sampler = qpu_sampler

        if sampling_params is None:
            sampling_params = {}
        self.sampling_params = sampling_params

        self.logical_srt = logical_srt

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "qpu_sampler={self.sampler!r}, "
                       "sampling_params={self.sampling_params!r})").format(self=self)

    def next(self, state, **runopts):
        num_reads = runopts.get('num_reads', self.num_reads)
        sampling_params = runopts.get('sampling_params', self.sampling_params)

        params = sampling_params.copy()
        params.update(num_reads=num_reads)

        sampler = FixedEmbeddingComposite(self.sampler, embedding=state.embedding)
        if self.logical_srt:
            params.update(num_spin_reversal_transforms=1)
            sampler = SpinReversalTransformComposite(sampler)
        response = sampler.sample(state.subproblem, **params)

        return state.updated(subsamples=response)


class SubproblemCliqueEmbedder(traits.SubproblemIntaking,
                               traits.EmbeddingProducing,
                               traits.SISO, Runnable):
    """Subproblem-induced-clique embedder on sampler's target graph.

    Args:
        sampler (:class:`dimod.Structured`):
            Structured :class:`dimod.Sampler` such as a
            :class:`~dwave.system.samplers.DWaveSampler`.

    Example:
        To replace :class:`.QPUSubproblemAutoEmbeddingSampler` with a sampler
        that uses fixed clique embedding (adapted to subproblem on each run),
        use ``SubproblemCliqueEmbedder | QPUSubproblemExternalEmbeddingSampler``
        construct::

            from dwave.system import DWaveSampler

            qpu = DWaveSampler()
            qpu_branch = (
                hybrid.EnergyImpactDecomposer(size=50)
                | hybrid.SubproblemCliqueEmbedder(sampler=qpu)
                | hybrid.QPUSubproblemExternalEmbeddingSampler(qpu_sampler=qpu))
    """

    def __init__(self, sampler, **runopts):
        super(SubproblemCliqueEmbedder, self).__init__(**runopts)
        self.sampler = sampler

    def __repr__(self):
        return "{self}(sampler={self.sampler!r})".format(self=self)

    @staticmethod
    def find_clique_embedding(variables, sampler):
        """Given a :class:`dimod.Structured` ``sampler``, and a list of
        variable labels, return a clique embedding.

        Returns:
            dict:
                Clique embedding map with source variables from ``variables``
                and target graph taken from ``sampler``.

        """
        g = sampler.to_networkx_graph()
        return busgraph_cache(g).find_clique_embedding(variables)

    def next(self, state, **runopts):
        embedding = self.find_clique_embedding(
            state.subproblem.variables, self.sampler)
        return state.updated(embedding=embedding)


class QPUSubproblemAutoEmbeddingSampler(traits.SubproblemSampler, traits.SISO, Runnable):
    r"""A quantum sampler for a subproblem with automated heuristic
    minor-embedding.

    Args:
        num_reads (int, optional, default=100):
            Number of states (output solutions) to read from the sampler.

        num_retries (int, optional, default=0):
            Number of times the sampler will retry to embed if a failure occurs.

        qpu_sampler (:class:`dimod.Sampler`, optional, default=\ :class:`~dwave.system.samplers.DWaveSampler()`):
            Quantum sampler such as a D-Wave system. Subproblems that do not fit the
            sampler's structure are minor-embedded on the fly with
            :class:`~dwave.system.composites.AutoEmbeddingComposite`.

        sampling_params (dict):
            Dictionary of keyword arguments with values that will be used
            on every call of the (embedding-wrapped QPU) sampler.

        auto_embedding_params (dict, optional):
            If provided, parameters are passed to the
            :class:`~dwave.system.composites.AutoEmbeddingComposite` constructor
            as keyword arguments.

    See :ref:`samplers-examples`.
    """

    def __init__(self, num_reads=100, num_retries=0, qpu_sampler=None, sampling_params=None,
                 auto_embedding_params=None, **runopts):
        super(QPUSubproblemAutoEmbeddingSampler, self).__init__(**runopts)

        self.num_reads = num_reads
        self.num_retries = num_retries

        if qpu_sampler is None:
            qpu_sampler = DWaveSampler()

        if sampling_params is None:
            sampling_params = {}
        self.sampling_params = sampling_params

        # embed on the fly and only if needed
        if auto_embedding_params is None:
            auto_embedding_params = {}
        self.sampler = AutoEmbeddingComposite(qpu_sampler, **auto_embedding_params)

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "qpu_sampler={self.sampler!r}, "
                       "sampling_params={self.sampling_params!r})").format(self=self)

    def next(self, state, **runopts):
        num_reads = runopts.get('num_reads', self.num_reads)
        sampling_params = runopts.get('sampling_params', self.sampling_params)

        params = sampling_params.copy()
        params.update(num_reads=num_reads)

        num_retries = runopts.get('num_retries', self.num_retries)

        embedding_success = False
        num_tries = 0

        while not embedding_success:
            try:
                num_tries += 1
                response = self.sampler.sample(state.subproblem, **params)
            except ValueError as exc:
                if num_tries <= num_retries:
                    pass
                else:
                    raise exc
            else:
                embedding_success = True

        return state.updated(subsamples=response)

class ReverseAnnealingAutoEmbeddingSampler(traits.SubproblemSampler,
                                           traits.SubsamplesIntaking,
                                           traits.SISO, Runnable):
    r"""A quantum reverse annealing sampler for a subproblem with automated
    heuristic minor-embedding.

    Args:
        num_reads (int, optional, default=100):
            Number of states (output solutions) to read from the sampler.

        anneal_schedule (list(list), optional, default=[[0, 1], [0.5, 0.5], [1, 1]]):
            An anneal schedule defined by a series of pairs of floating-point
            numbers identifying points in the schedule at which to change slope.
            The first element in the pair is time t in microseconds; the second,
            normalized persistent current s in the range [0,1]. The resulting
            schedule is the piecewise-linear curve that connects the provided
            points. For more details, see
            :meth:`~dwave.system.DWaveSampler.validate_anneal_schedule`.

        qpu_sampler (:class:`dimod.Sampler`, optional):
            Quantum sampler such as a D-Wave system. Subproblems that do not fit
            the sampler's structure are minor-embedded on the fly with
            :class:`~dwave.system.composites.AutoEmbeddingComposite`.

            If sampler is not provided, it defaults to::

                qpu_sampler = DWaveSampler(
                    solver=dict(max_anneal_schedule_points__gte=len(anneal_schedule)))

        sampling_params (dict):
            Dictionary of keyword arguments with values that will be used
            on every call of the (embedding-wrapped QPU) sampler.

        auto_embedding_params (dict, optional):
            If provided, parameters are passed to the
            :class:`~dwave.system.composites.AutoEmbeddingComposite` constructor
            as keyword arguments.

    """

    def __init__(self, num_reads=100, anneal_schedule=None, qpu_sampler=None,
                 sampling_params=None, auto_embedding_params=None, **runopts):
        super(ReverseAnnealingAutoEmbeddingSampler, self).__init__(**runopts)

        self.num_reads = num_reads

        if anneal_schedule is None:
            anneal_schedule = [[0, 1], [0.5, 0.5], [1, 1]]
        self.anneal_schedule = anneal_schedule

        if qpu_sampler is None:
            qpu_sampler = DWaveSampler(
                solver=dict(max_anneal_schedule_points__gte=len(self.anneal_schedule)))

        # validate schedule, raising `ValueError` on invalid schedule or
        # `RuntimeError` if anneal schedule not supported by QPU (this could
        # happen only if user provided the `qpu_sampler`)
        qpu_sampler.validate_anneal_schedule(anneal_schedule)

        if sampling_params is None:
            sampling_params = {}
        self.sampling_params = sampling_params

        # embed on the fly and only if needed
        if auto_embedding_params is None:
            auto_embedding_params = {}
        self.sampler = AutoEmbeddingComposite(qpu_sampler, **auto_embedding_params)

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "anneal_schedule={self.anneal_schedule!r}, "
                       "qpu_sampler={self.sampler!r}, "
                       "sampling_params={self.sampling_params!r})").format(self=self)

    def next(self, state, **runopts):
        num_reads = runopts.get('num_reads', self.num_reads)
        anneal_schedule = runopts.get('anneal_schedule', self.anneal_schedule)
        sampling_params = runopts.get('sampling_params', self.sampling_params)

        params = sampling_params.copy()
        params.update(num_reads=num_reads, anneal_schedule=anneal_schedule)

        # TODO: handle more than just the first subsample (not yet supported via API)
        subsamples = self.sampler.sample(
            state.subproblem, initial_state=state.subsamples.first.sample, **params)

        return state.updated(subsamples=subsamples)


class SimulatedAnnealingSubproblemSampler(traits.SubproblemSampler, traits.SISO, Runnable):
    """A simulated annealing sampler for a subproblem.

    Args:
        num_reads (int, optional, default=len(state.subsamples) or 1):
            Number of states (output solutions) to read from the sampler.

        num_sweeps (int, optional, default=1000):
            Number of sweeps or steps.

        beta_range (tuple, optional):
            A 2-tuple defining the beginning and end of the beta schedule, where
            beta is the inverse temperature. The schedule is applied linearly in
            beta. Default range is set based on the total bias associated with
            each node.

        beta_schedule_type (string, optional, default='geometric'):
            Beta schedule type, or how the beta values are interpolated between
            the given 'beta_range'. Supported values are: linear and geometric.

        initial_states_generator (str, 'none'/'tile'/'random', optional, default='random'):
            Defines the expansion of input state subsamples into `initial_states`
            for the simulated annealing, if fewer than `num_reads` subsamples are
            present. See :meth:`~neal.SimulatedAnnealingSampler.sample`.

    See :ref:`samplers-examples`.
    """

    def __init__(self, num_reads=None, num_sweeps=1000,
                 beta_range=None, beta_schedule_type='geometric',
                 initial_states_generator='random', **runopts):
        super(SimulatedAnnealingSubproblemSampler, self).__init__(**runopts)
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range
        self.beta_schedule_type = beta_schedule_type
        self.initial_states_generator = initial_states_generator
        self.sampler = SimulatedAnnealingSampler()
        self._stop_event = threading.Event()

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "num_sweeps={self.num_sweeps!r}, "
                       "initial_states_generator={self.initial_states_generator!r})").format(self=self)

    def next(self, state, **runopts):
        subsamples = self.sampler.sample(
            state.subproblem, num_reads=self.num_reads, num_sweeps=self.num_sweeps,
            beta_range=self.beta_range, beta_schedule_type=self.beta_schedule_type,
            initial_states=state.subsamples,
            initial_states_generator=self.initial_states_generator,
            interrupt_function=lambda: self._stop_event.is_set())
        return state.updated(subsamples=subsamples)

    def halt(self):
        self._stop_event.set()


class InterruptableSimulatedAnnealingSubproblemSampler(SimulatedAnnealingSubproblemSampler):
    """SimulatedAnnealingSubproblemSampler is already interruptable."""
    pass


class SimulatedAnnealingProblemSampler(traits.ProblemSampler, traits.SISO, Runnable):
    """A simulated annealing sampler for a complete problem.

    Args:
        num_reads (int, optional, default=len(state.samples) or 1):
            Number of states (output solutions) to read from the sampler.

        num_sweeps (int, optional, default=1000):
            Number of sweeps or steps.

        beta_range (tuple, optional):
            A 2-tuple defining the beginning and end of the beta schedule, where
            beta is the inverse temperature. The schedule is applied linearly in
            beta. Default range is set based on the total bias associated with
            each node.

        beta_schedule_type (string, optional, default='geometric'):
            Beta schedule type, or how the beta values are interpolated between
            the given 'beta_range'. Supported values are: linear and geometric.

        initial_states_generator (str, 'none'/'tile'/'random', optional, default='random'):
            Defines the expansion of input state samples into `initial_states`
            for the simulated annealing, if fewer than `num_reads` samples are
            present. See :meth:`~neal.SimulatedAnnealingSampler.sample`.

    """

    def __init__(self, num_reads=None, num_sweeps=1000,
                 beta_range=None, beta_schedule_type='geometric',
                 initial_states_generator='random', **runopts):
        super(SimulatedAnnealingProblemSampler, self).__init__(**runopts)
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range
        self.beta_schedule_type = beta_schedule_type
        self.initial_states_generator = initial_states_generator
        self.sampler = SimulatedAnnealingSampler()
        self._stop_event = threading.Event()

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "num_sweeps={self.num_sweeps!r}, "
                       "initial_states_generator={self.initial_states_generator!r})").format(self=self)

    def next(self, state, **runopts):
        samples = self.sampler.sample(
            state.problem, num_reads=self.num_reads, num_sweeps=self.num_sweeps,
            beta_range=self.beta_range, beta_schedule_type=self.beta_schedule_type,
            initial_states=state.samples,
            initial_states_generator=self.initial_states_generator,
            interrupt_function=lambda: self._stop_event.is_set())
        return state.updated(samples=samples)

    def halt(self):
        self._stop_event.set()


class InterruptableSimulatedAnnealingProblemSampler(SimulatedAnnealingProblemSampler):
    """SimulatedAnnealingProblemSampler is already interruptable."""
    pass


class TabuSubproblemSampler(traits.SubproblemSampler, traits.SISO, Runnable):
    """A tabu sampler for a subproblem.

    Args:
        num_reads (int, optional, default=len(state.subsamples) or 1):
            Number of states (output solutions) to read from the sampler.

        tenure (int, optional):
            Tabu tenure, which is the length of the tabu list, or number of
            recently explored solutions kept in memory. Default is a quarter of
            the number of problem variables up to a maximum value of 20.

        timeout (int, optional, default=100):
            Total running time in milliseconds.

        initial_states_generator (str, 'none'/'tile'/'random', optional, default='random'):
            Defines the expansion of input state subsamples into `initial_states`
            for the Tabu search, if fewer than `num_reads` subsamples are
            present. See :meth:`~tabu.TabuSampler.sample`.

    See :ref:`samplers-examples`.
    """

    def __init__(self, num_reads=None, tenure=None, timeout=100,
                 initial_states_generator='random', **runopts):
        super(TabuSubproblemSampler, self).__init__(**runopts)
        self.num_reads = num_reads
        self.tenure = tenure
        self.timeout = timeout
        self.initial_states_generator = initial_states_generator
        self.sampler = TabuSampler()

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "tenure={self.tenure!r}, "
                       "timeout={self.timeout!r}, "
                       "initial_states_generator={self.initial_states_generator!r})").format(self=self)

    def next(self, state, **runopts):
        subsamples = self.sampler.sample(
            state.subproblem, initial_states=state.subsamples,
            initial_states_generator=self.initial_states_generator,
            tenure=self.tenure, timeout=self.timeout, num_reads=self.num_reads)
        return state.updated(subsamples=subsamples)


class TabuProblemSampler(traits.ProblemSampler, traits.SISO, Runnable):
    """A tabu sampler for a binary quadratic problem.

    Args:
        num_reads (int, optional, default=len(state.samples) or 1):
            Number of states (output solutions) to read from the sampler.

        tenure (int, optional):
            Tabu tenure, which is the length of the tabu list, or number of
            recently explored solutions kept in memory. Default is a quarter of
            the number of problem variables up to a maximum value of 20.

        timeout (int, optional, default=100):
            Total running time in milliseconds.

        initial_states_generator (str, 'none'/'tile'/'random', optional, default='random'):
            Defines the expansion of input state samples into `initial_states`
            for the Tabu search, if fewer than `num_reads` samples are
            present. See :meth:`~tabu.TabuSampler.sample`.

    See :ref:`samplers-examples`.
    """

    def __init__(self, num_reads=None, tenure=None, timeout=100,
                 initial_states_generator='random', **runopts):
        super(TabuProblemSampler, self).__init__(**runopts)
        self.num_reads = num_reads
        self.tenure = tenure
        self.timeout = timeout
        self.initial_states_generator = initial_states_generator
        self.sampler = TabuSampler()

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "tenure={self.tenure!r}, "
                       "timeout={self.timeout!r}, "
                       "initial_states_generator={self.initial_states_generator!r})").format(self=self)

    def next(self, state, **runopts):
        sampleset = self.sampler.sample(
            state.problem, initial_states=state.samples,
            initial_states_generator=self.initial_states_generator,
            tenure=self.tenure, timeout=self.timeout, num_reads=self.num_reads)
        return state.updated(samples=sampleset)


class InterruptableTabuSampler(Loop):
    """An interruptable tabu sampler for a binary quadratic problem.

    Args:
        num_reads (int, optional, default=1):
            Number of states (output solutions) to read from the sampler.

        tenure (int, optional):
            Tabu tenure, which is the length of the tabu list, or number of
            recently explored solutions kept in memory. Default is a quarter of
            the number of problem variables up to a maximum value of 20.

        timeout (int, optional, default=20):
            Timeout for non-interruptable operation of tabu search. At the
            completion of each loop of tabu search through its problem
            variables, if this time interval has been exceeded, the search can
            be stopped by an interrupt signal or expiration of the `timeout`
            parameter.

        initial_states_generator (str, 'none'/'tile'/'random', optional, default='random'):
            Defines the expansion of input state samples into `initial_states`
            for the Tabu search, if fewer than `num_reads` samples are
            present. See :meth:`~tabu.TabuSampler.sample`.

        max_time (float, optional, default=None):
            Total running time in milliseconds.

    See :ref:`samplers-examples`.
    """

    def __init__(self, max_time=None, **tabu):
        super(InterruptableTabuSampler, self).__init__(
            TabuProblemSampler(**tabu), max_time=max_time)


class SteepestDescentSubproblemSampler(traits.SubproblemSampler, traits.SISO, Runnable):
    """A steepest descent solver for a subproblem.

    Args:
        num_reads (int, optional, default=len(state.subsamples) or 1):
            Number of states (output solutions) to read from the sampler.

        initial_states_generator (str, 'none'/'tile'/'random', optional, default='random'):
            Defines the expansion of input state subsamples into `initial_states`
            for the steepest descent, if fewer than `num_reads` subsamples are
            present. See :meth:`greedy.sampler.SteepestDescentSolver.sample`.

    See :ref:`samplers-examples`.
    """

    def __init__(self, num_reads=None, initial_states_generator='random', **runopts):
        super(SteepestDescentSubproblemSampler, self).__init__(**runopts)
        self.num_reads = num_reads
        self.initial_states_generator = initial_states_generator
        self.sampler = SteepestDescentSolver()

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "initial_states_generator={self.initial_states_generator!r})").format(self=self)

    def next(self, state, **runopts):
        subsamples = self.sampler.sample(
            state.subproblem, num_reads=self.num_reads,
            initial_states=state.subsamples,
            initial_states_generator=self.initial_states_generator)
        return state.updated(subsamples=subsamples)

# alias
GreedySubproblemSampler = SteepestDescentSubproblemSampler


class SteepestDescentProblemSampler(traits.ProblemSampler, traits.SISO, Runnable):
    """A steepest descent solver for a complete problem.

    Args:
        num_reads (int, optional, default=len(state.samples) or 1):
            Number of states (output solutions) to read from the sampler.

        initial_states_generator (str, 'none'/'tile'/'random', optional, default='random'):
            Defines the expansion of input state samples into `initial_states`
            for the steepest descent, if fewer than `num_reads` samples are
            present. See :meth:`greedy.sampler.SteepestDescentSolver.sample`.

    """

    def __init__(self, num_reads=None, initial_states_generator='random', **runopts):
        super(SteepestDescentProblemSampler, self).__init__(**runopts)
        self.num_reads = num_reads
        self.initial_states_generator = initial_states_generator
        self.sampler = SteepestDescentSolver()

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "initial_states_generator={self.initial_states_generator!r})").format(self=self)

    def next(self, state, **runopts):
        samples = self.sampler.sample(
            state.problem, num_reads=self.num_reads,
            initial_states=state.samples,
            initial_states_generator=self.initial_states_generator)
        return state.updated(samples=samples)

# alias
GreedyProblemSampler = SteepestDescentProblemSampler


class RandomSubproblemSampler(traits.SubproblemSampler, traits.SISO, Runnable):
    """A random sample generator for a subproblem."""

    def next(self, state, **runopts):
        bqm = state.subproblem
        sample = random_sample(bqm)
        sampleset = SampleSet.from_samples(sample,
                                           vartype=bqm.vartype,
                                           energy=bqm.energy(sample))
        return state.updated(subsamples=sampleset)
