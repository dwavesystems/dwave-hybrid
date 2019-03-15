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
Classical and quantum :class:`.Runnable` `dimod <http://dimod.readthedocs.io/en/stable/>`_
samplers for problems and subproblems.
"""

import time
import logging
import threading
from collections import namedtuple

import dimod
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite

from tabu import TabuSampler
from neal import SimulatedAnnealingSampler

from hybrid.core import Runnable, SampleSet
from hybrid.flow import Loop
from hybrid.utils import random_sample
from hybrid import traits

__all__ = [
    'QPUSubproblemExternalEmbeddingSampler', 'QPUSubproblemAutoEmbeddingSampler',
    'ReverseAnnealingAutoEmbeddingSampler',
    'SimulatedAnnealingSubproblemSampler', 'InterruptableSimulatedAnnealingSubproblemSampler',
    'SimulatedAnnealingProblemSampler', 'InterruptableSimulatedAnnealingProblemSampler',
    'TabuSubproblemSampler', 'TabuProblemSampler', 'InterruptableTabuSampler',
    'RandomSubproblemSampler',
]

logger = logging.getLogger(__name__)


class QPUSubproblemExternalEmbeddingSampler(Runnable, traits.SubproblemSampler, traits.EmbeddingIntaking):
    """A quantum sampler for a subproblem with a defined minor-embedding.

    Args:
        num_reads (int, optional, default=100):
            Number of states (output solutions) to read from the sampler.
        qpu_sampler (:class:`dimod.Sampler`, optional, default=DWaveSampler()):
            Quantum sampler such as a D-Wave system.

    Examples:
        See examples on https://docs.ocean.dwavesys.com/projects/hybrid/en/latest/reference/samplers.html#examples.
    """

    def __init__(self, num_reads=100, qpu_sampler=None, **runopts):
        super(QPUSubproblemExternalEmbeddingSampler, self).__init__(**runopts)

        self.num_reads = num_reads
        if qpu_sampler is None:
            qpu_sampler = DWaveSampler()
        self.sampler = qpu_sampler

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "qpu_sampler={self.sampler!r})").format(self=self)

    def next(self, state, **runopts):
        sampler = FixedEmbeddingComposite(self.sampler, embedding=state.embedding)
        response = sampler.sample(state.subproblem, num_reads=self.num_reads)
        return state.updated(subsamples=response)


class QPUSubproblemAutoEmbeddingSampler(Runnable, traits.SubproblemSampler):
    """A quantum sampler for a subproblem with automated heuristic minor-embedding.

    Args:
        num_reads (int, optional, default=100):
            Number of states (output solutions) to read from the sampler.
        qpu_sampler (:class:`dimod.Sampler`, optional, default=EmbeddingComposite(DWaveSampler())):
            Quantum sampler such as a D-Wave system. If sampler is structured,
            it will be converted to unstructured via :class:`~dwave.system.composited.EmbeddingComposite`.

    Examples:
        See examples on https://docs.ocean.dwavesys.com/projects/hybrid/en/latest/reference/samplers.html#examples.
    """

    def __init__(self, num_reads=100, qpu_sampler=None, **runopts):
        super(QPUSubproblemAutoEmbeddingSampler, self).__init__(**runopts)

        self.num_reads = num_reads

        if qpu_sampler is None:
            qpu_sampler = DWaveSampler()

        # convert the structured sampler to unstructured
        if isinstance(qpu_sampler, dimod.Structured):
            self.sampler = EmbeddingComposite(qpu_sampler)
        else:
            self.sampler = qpu_sampler

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "qpu_sampler={self.sampler!r})").format(self=self)

    def next(self, state, **runopts):
        response = self.sampler.sample(state.subproblem, num_reads=self.num_reads)
        return state.updated(subsamples=response)


class ReverseAnnealingAutoEmbeddingSampler(Runnable, traits.SubproblemSampler):
    """A quantum reverse annealing sampler for a subproblem with automated
    heuristic minor-embedding.

    Args:
        num_reads (int, optional, default=100):
            Number of states (output solutions) to read from the sampler.

        qpu_sampler (:class:`dimod.Sampler`, optional, default=EmbeddingComposite(DWaveSampler())):
            Quantum sampler such as a D-Wave system. If sampler is structured,
            it will be converted to unstructured via :class:`~dwave.system.composited.EmbeddingComposite`.

        anneal_schedule (list(list), optional, default=[[0, 1], [0.5, 0.5], [1, 1]]):
            An anneal schedule defined by a series of pairs of floating-point numbers
            identifying points in the schedule at which to change slope. The first
            element in the pair is time t in microseconds; the second, normalized
            persistent current s in the range [0,1]. The resulting schedule is the
            piecewise-linear curve that connects the provided points. For more
            details, see :meth:`~dwave.system.DWaveSampler.validate_anneal_schedule`.
    """

    def __init__(self, num_reads=100, qpu_sampler=None, anneal_schedule=None, **runopts):
        super(ReverseAnnealingAutoEmbeddingSampler, self).__init__(**runopts)

        self.num_reads = num_reads

        if anneal_schedule is None:
            anneal_schedule = [[0, 1], [0.5, 0.5], [1, 1]]
        self.anneal_schedule = anneal_schedule

        if qpu_sampler is None:
            qpu_sampler = DWaveSampler(
                solver={'max_anneal_schedule_points__gte': len(self.anneal_schedule)})

        # validate schedule, raising `ValueError` on invalid schedule or
        # `RuntimeError` if anneal schedule not supported by QPU (this could
        # happen only if user provided the `qpu_sampler`)
        qpu_sampler.validate_anneal_schedule(anneal_schedule)

        # convert the structured sampler to unstructured
        if isinstance(qpu_sampler, dimod.Structured):
            self.sampler = EmbeddingComposite(qpu_sampler)
        else:
            self.sampler = qpu_sampler

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "qpu_sampler={self.sampler!r}, "
                       "anneal_schedule={self.anneal_schedule!r})").format(self=self)

    def next(self, state, **runopts):
        # TODO: handle more than just the first subsample
        response = self.sampler.sample(
            state.subproblem, num_reads=self.num_reads,
            initial_state=state.subsamples.first.sample,
            anneal_schedule=self.anneal_schedule)
        return state.updated(subsamples=response)


class SimulatedAnnealingSubproblemSampler(Runnable, traits.SubproblemSampler):
    """A simulated annealing sampler for a subproblem.

    Args:
        num_reads (int, optional, default=1):
            Number of states (output solutions) to read from the sampler.
        sweeps (int, optional, default=1000):
            Number of sweeps or steps.

    Examples:
        See examples on https://docs.ocean.dwavesys.com/projects/hybrid/en/latest/reference/samplers.html#examples.
    """

    def __init__(self, num_reads=1, sweeps=1000, **runopts):
        super(SimulatedAnnealingSubproblemSampler, self).__init__(**runopts)
        self.num_reads = num_reads
        self.sweeps = sweeps
        self.sampler = SimulatedAnnealingSampler()
        self._stop_event = threading.Event()

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "sweeps={self.sweeps!r})").format(self=self)

    def next(self, state, **runopts):
        subbqm = state.subproblem
        response = self.sampler.sample(
            subbqm, num_reads=self.num_reads, sweeps=self.sweeps,
            interrupt_function=lambda: self._stop_event.is_set())
        return state.updated(subsamples=response)

    def halt(self):
        self._stop_event.set()


class InterruptableSimulatedAnnealingSubproblemSampler(SimulatedAnnealingSubproblemSampler):
    """SimulatedAnnealingSubproblemSampler is already interruptable."""
    pass


class SimulatedAnnealingProblemSampler(Runnable, traits.ProblemSampler):
    """A simulated annealing sampler for a complete problem.

    Args:
        num_reads (int, optional, default=1):
            Number of states (output solutions) to read from the sampler.
        sweeps (int, optional, default=1000):
            Number of sweeps or steps.

    """

    def __init__(self, num_reads=1, sweeps=1000, **runopts):
        super(SimulatedAnnealingProblemSampler, self).__init__(**runopts)
        self.num_reads = num_reads
        self.sweeps = sweeps
        self.sampler = SimulatedAnnealingSampler()
        self._stop_event = threading.Event()

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "sweeps={self.sweeps!r})").format(self=self)

    def next(self, state, **runopts):
        bqm = state.problem
        response = self.sampler.sample(
            bqm, num_reads=self.num_reads, sweeps=self.sweeps,
            interrupt_function=lambda: self._stop_event.is_set())
        return state.updated(samples=response)

    def halt(self):
        self._stop_event.set()


class InterruptableSimulatedAnnealingProblemSampler(SimulatedAnnealingProblemSampler):
    """SimulatedAnnealingProblemSampler is already interruptable."""
    pass


class TabuSubproblemSampler(Runnable, traits.SubproblemSampler):
    """A tabu sampler for a subproblem.

    Args:
        num_reads (int, optional, default=1):
            Number of states (output solutions) to read from the sampler.
        tenure (int, optional):
            Tabu tenure, which is the length of the tabu list, or number of recently
            explored solutions kept in memory. Default is a quarter of the number
            of problem variables up to a maximum value of 20.
        timeout (int, optional, default=20):
            Total running time in milliseconds.

    Examples:
        See examples on https://docs.ocean.dwavesys.com/projects/hybrid/en/latest/reference/samplers.html#examples.
    """

    def __init__(self, num_reads=1, tenure=None, timeout=20, **runopts):
        super(TabuSubproblemSampler, self).__init__(**runopts)
        self.num_reads = num_reads
        self.tenure = tenure
        self.timeout = timeout
        self.sampler = TabuSampler()

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "tenure={self.tenure!r}, "
                       "timeout={self.timeout!r})").format(self=self)

    def next(self, state, **runopts):
        subbqm = state.subproblem
        response = self.sampler.sample(
            subbqm, tenure=self.tenure, timeout=self.timeout, num_reads=self.num_reads)
        return state.updated(subsamples=response)


class TabuProblemSampler(Runnable, traits.ProblemSampler):
    """A tabu sampler for a binary quadratic problem.

    Args:
        num_reads (int, optional, default=1):
            Number of states (output solutions) to read from the sampler.
        tenure (int, optional):
            Tabu tenure, which is the length of the tabu list, or number of recently
            explored solutions kept in memory. Default is a quarter of the number
            of problem variables up to a maximum value of 20.
        timeout (int, optional, default=20):
            Total running time in milliseconds.

    Examples:
        See examples on https://docs.ocean.dwavesys.com/projects/hybrid/en/latest/reference/samplers.html#examples.
    """

    def __init__(self, num_reads=1, tenure=None, timeout=20, **runopts):
        super(TabuProblemSampler, self).__init__(**runopts)
        self.num_reads = num_reads
        self.tenure = tenure
        self.timeout = timeout
        self.sampler = TabuSampler()

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "tenure={self.tenure!r}, "
                       "timeout={self.timeout!r})").format(self=self)

    def next(self, state, **runopts):
        sampleset = self.sampler.sample(
            state.problem, init_solution=state.samples, tenure=self.tenure,
            timeout=self.timeout, num_reads=self.num_reads)
        return state.updated(samples=sampleset)


class InterruptableTabuSampler(Loop):
    """An interruptable tabu sampler for a binary quadratic problem.

    Args:
        num_reads (int, optional, default=1):
            Number of states (output solutions) to read from the sampler.
        tenure (int, optional):
            Tabu tenure, which is the length of the tabu list, or number of recently
            explored solutions kept in memory. Default is a quarter of the number
            of problem variables up to a maximum value of 20.
        timeout (int, optional, default=20):
            Timeout for non-interruptable operation of tabu search. At the completion of
            each loop of tabu search through its problem variables, if this time interval
            has been exceeded, the search can be stopped by an interrupt signal or
            expiration of the `timeout` parameter.
        max_time (float, optional, default=None):
            Total running time in milliseconds.

    Examples:
        See examples on https://docs.ocean.dwavesys.com/projects/hybrid/en/latest/reference/samplers.html#examples.
    """

    def __init__(self, max_time=None, **tabu):
        super(InterruptableTabuSampler, self).__init__(
            TabuProblemSampler(**tabu), max_time=max_time)


class RandomSubproblemSampler(Runnable, traits.SubproblemSampler):
    """A random sample generator for a subproblem."""

    def next(self, state, **runopts):
        bqm = state.subproblem
        sample = random_sample(bqm)
        sampleset = SampleSet.from_samples(sample,
                                           vartype=bqm.vartype,
                                           energy=bqm.energy(sample))
        return state.updated(subsamples=sampleset)
