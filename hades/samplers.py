import time
import threading
from collections import namedtuple

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

# TODO: pip-ify
from tabu_sampler import TabuSampler
from neal import SimulatedAnnealingSampler

from hades.core import executor, Runnable, State, Sample
from hades.profiling import tictoc
from hades.utils import sample_as_list, sample_as_dict

import logging
logger = logging.getLogger(__name__)


class QPUSubproblemSampler(Runnable):

    def __init__(self, bqm, num_reads=100):
        self.bqm = bqm
        self.num_reads = num_reads
        self.sampler = EmbeddingComposite(DWaveSampler())

    @tictoc('qpu_sample')
    def iterate(self, state):
        response = self.sampler.sample(state.ctx['subproblem'], num_reads=self.num_reads)
        best_response = next(response.data())
        best_sample = sample_as_dict(best_response.sample)
        return state.updated(ctx=dict(subsample=best_sample),
                             debug=dict(source=self.__class__.__name__))


class SimulatedAnnealingSubproblemSampler(Runnable):

    def __init__(self, bqm, num_reads=1, sweeps=1000):
        self.bqm = bqm
        self.num_reads = num_reads
        self.sweeps = sweeps
        self.sampler = SimulatedAnnealingSampler()

    @tictoc('subneal_sample')
    def iterate(self, state):
        subbqm = state.ctx['subproblem']
        response = self.sampler.sample(
            subbqm, num_reads=self.num_reads, sweeps=self.sweeps)
        best_subsample = sample_as_dict(next(response.samples()))
        return state.updated(ctx=dict(subsample=best_subsample),
                             debug=dict(source=self.__class__.__name__))


class TabuSubproblemSampler(Runnable):

    def __init__(self, bqm, num_reads=1, tenure=None, timeout=20):
        self.bqm = bqm
        self.num_reads = num_reads
        self.tenure = tenure
        self.timeout = timeout
        self.sampler = TabuSampler()

    @tictoc('subtabu_sample')
    def iterate(self, state):
        subbqm = state.ctx['subproblem']
        response = self.sampler.sample(
            subbqm, tenure=self.tenure, timeout=self.timeout, num_reads=self.num_reads)
        best_subsample = sample_as_dict(next(response.samples()))
        return state.updated(ctx=dict(subsample=best_subsample),
                             debug=dict(source=self.__class__.__name__))


class TabuProblemSampler(Runnable):

    def __init__(self, bqm, num_reads=1, tenure=None, timeout=20):
        self.bqm = bqm
        self.num_reads = num_reads
        self.tenure = tenure
        self.timeout = timeout
        self.sampler = TabuSampler()

    @tictoc('tabu_sample')
    def iterate(self, state):
        sample = state.sample.values
        response = self.sampler.sample(
            self.bqm, init_solution=sample_as_list(sample), tenure=self.tenure,
            timeout=self.timeout, num_reads=self.num_reads)
        response_datum = next(response.data())
        best_sample = sample_as_dict(response_datum.sample)
        best_energy = response_datum.energy
        return state.updated(sample=Sample(best_sample, best_energy),
                             debug=dict(source=self.__class__.__name__))


class InterruptableTabuSampler(TabuProblemSampler):

    def __init__(self, bqm, quantum_timeout=20, timeout=None, **kwargs):
        kwargs['bqm'] = bqm
        kwargs['timeout'] = quantum_timeout
        super(InterruptableTabuSampler, self).__init__(**kwargs)
        self.max_timeout = timeout
        self._stop_event = threading.Event()

    @tictoc('int_tabu_sample')
    def _interruptable_iterate(self, state):
        start = time.time()
        iterno = 1
        while True:
            state = self.iterate(state)
            runtime = time.time() - start
            timeout = self.max_timeout is not None and runtime >= self.max_timeout
            if self._stop_event.is_set() or timeout:
                break
            iterno += 1
        return state.updated(debug=dict(source=self.__class__.__name__,
                                        runtime=runtime, iterno=iterno))

    def run(self, state):
        self._stop_event.clear()
        return executor.submit(self._interruptable_iterate, state)

    def stop(self):
        self._stop_event.set()
