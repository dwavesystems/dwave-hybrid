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

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite

from tabu import TabuSampler
from neal import SimulatedAnnealingSampler

from hybrid.core import async_executor, Runnable, SampleSet
from hybrid.utils import random_sample
from hybrid import traits

__all__ = [
    'QPUSubproblemExternalEmbeddingSampler', 'QPUSubproblemAutoEmbeddingSampler',
    'SimulatedAnnealingSubproblemSampler', 'InterruptableSimulatedAnnealingSubproblemSampler',
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
    """

    def __init__(self, num_reads=100, qpu_sampler=None):
        super(QPUSubproblemExternalEmbeddingSampler, self).__init__()

        self.num_reads = num_reads
        if qpu_sampler is None:
            qpu_sampler = DWaveSampler()
        self.sampler = qpu_sampler

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "qpu_sampler={self.sampler!r})").format(self=self)

    def next(self, state):
        sampler = FixedEmbeddingComposite(self.sampler, embedding=state.embedding)
        response = sampler.sample(state.subproblem, num_reads=self.num_reads)
        return state.updated(subsamples=response)


class QPUSubproblemAutoEmbeddingSampler(Runnable, traits.SubproblemSampler):
    """A quantum sampler for a subproblem with automated heuristic minor-embedding.

    Args:
        num_reads (int, optional, default=100):
            Number of states (output solutions) to read from the sampler.
        qpu_sampler (:class:`dimod.Sampler`, optional, default=EmbeddingComposite(DWaveSampler())):
            Quantum sampler such as a D-Wave system.
    """

    def __init__(self, num_reads=100, qpu_sampler=None):
        super(QPUSubproblemAutoEmbeddingSampler, self).__init__()

        self.num_reads = num_reads
        if qpu_sampler is None:
            qpu_sampler = EmbeddingComposite(DWaveSampler())
        self.sampler = qpu_sampler

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "qpu_sampler={self.sampler!r})").format(self=self)

    def next(self, state):
        response = self.sampler.sample(state.subproblem, num_reads=self.num_reads)
        return state.updated(subsamples=response)


class SimulatedAnnealingSubproblemSampler(Runnable, traits.SubproblemSampler):
    """A simulated annealing sampler for a subproblem.

    Args:
        num_reads (int, optional, default=1):
            Number of states (output solutions) to read from the sampler.
        sweeps (int, optional, default=1000):
            Number of sweeps or steps.
    """

    def __init__(self, num_reads=1, sweeps=1000):
        super(SimulatedAnnealingSubproblemSampler, self).__init__()
        self.num_reads = num_reads
        self.sweeps = sweeps
        self.sampler = SimulatedAnnealingSampler()
        self._stop_event = threading.Event()

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "sweeps={self.sweeps!r})").format(self=self)

    def next(self, state):
        subbqm = state.subproblem
        response = self.sampler.sample(
            subbqm, num_reads=self.num_reads, sweeps=self.sweeps,
            interrupt_function=lambda: self._stop_event.is_set())
        return state.updated(subsamples=response)

    def stop(self):
        self._stop_event.set()


class InterruptableSimulatedAnnealingSubproblemSampler(SimulatedAnnealingSubproblemSampler):
    """SimulatedAnnealingSubproblemSampler is already interruptable."""
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
    """

    def __init__(self, num_reads=1, tenure=None, timeout=20):
        super(TabuSubproblemSampler, self).__init__()
        self.num_reads = num_reads
        self.tenure = tenure
        self.timeout = timeout
        self.sampler = TabuSampler()

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "tenure={self.tenure!r}, "
                       "timeout={self.timeout!r})").format(self=self)

    def next(self, state):
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
    """

    def __init__(self, num_reads=1, tenure=None, timeout=20):
        super(TabuProblemSampler, self).__init__()
        self.num_reads = num_reads
        self.tenure = tenure
        self.timeout = timeout
        self.sampler = TabuSampler()

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "tenure={self.tenure!r}, "
                       "timeout={self.timeout!r})").format(self=self)

    def next(self, state):
        sampleset = self.sampler.sample(
            state.problem, init_solution=state.samples, tenure=self.tenure,
            timeout=self.timeout, num_reads=self.num_reads)
        return state.updated(samples=sampleset)


class InterruptableTabuSampler(TabuProblemSampler):
    """An interruptable tabu sampler for a binary quadratic problem.

    Args:
        num_reads (int, optional, default=1):
            Number of states (output solutions) to read from the sampler.
        tenure (int, optional):
            Tabu tenure, which is the length of the tabu list, or number of recently
            explored solutions kept in memory. Default is a quarter of the number
            of problem variables up to a maximum value of 20.
        quantum_timeout (int, optional, default=20):
            Timeout for non-interruptable operation of tabu search. At the completion of
            each loop of tabu search through its problem variables, if this time interval
            has been exceeded, the search can be stopped by an interrupt signal or
            expiration of the `timeout` parameter.
        timeout (int, optional, default=20):
            Total running time in milliseconds.
    """

    def __init__(self, quantum_timeout=20, timeout=None, **kwargs):
        self.quantum_timeout = kwargs['timeout'] = quantum_timeout
        super(InterruptableTabuSampler, self).__init__(**kwargs)
        self.max_timeout = timeout
        self._stop_event = threading.Event()

    def __repr__(self):
        return ("{self}(quantum_timeout={self.quantum_timeout!r}, "
                       "timeout={self.timeout!r})").format(self=self)

    def next(self, state):
        start = time.time()
        iterno = 1
        self._stop_event.clear()
        while True:
            state = super(InterruptableTabuSampler, self).next(state)
            runtime = time.time() - start
            timeout = self.max_timeout is not None and runtime >= self.max_timeout
            if self._stop_event.is_set() or timeout:
                break
            iterno += 1
        # TODO: store iterno in local counter
        return state

    def stop(self):
        self._stop_event.set()


class RandomSubproblemSampler(Runnable, traits.SubproblemSampler):
    """A random sample generator for a subproblem."""

    def next(self, state):
        bqm = state.subproblem
        sample = random_sample(bqm)
        sampleset = SampleSet.from_samples(sample,
                                           vartype=bqm.vartype,
                                           energy=bqm.energy(sample))
        return state.updated(subsamples=sampleset)
