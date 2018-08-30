import time
import threading
from collections import namedtuple

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite

from tabu import TabuSampler
from neal import SimulatedAnnealingSampler

from hades.core import executor, Runnable, SampleSet
from hades.profiling import tictoc
from hades.utils import random_sample

import logging
logger = logging.getLogger(__name__)


class QPUSubproblemExternalEmbeddingSampler(Runnable):

    def __init__(self, num_reads=100):
        self.num_reads = num_reads
        self.sampler = DWaveSampler()

    @tictoc('qpu_ext_embedding_sample')
    def iterate(self, state):
        sampler = FixedEmbeddingComposite(self.sampler, embedding=state.ctx['embedding'])
        response = sampler.sample(state.ctx['subproblem'], num_reads=self.num_reads)
        return state.updated(ctx=dict(subsamples=response),
                             debug=dict(sampler=self.__class__.__name__))


class QPUSubproblemAutoEmbeddingSampler(Runnable):

    def __init__(self, num_reads=100):
        self.num_reads = num_reads
        self.sampler = EmbeddingComposite(DWaveSampler())

    @tictoc('qpu_auto_embedding_sample')
    def iterate(self, state):
        response = self.sampler.sample(state.ctx['subproblem'], num_reads=self.num_reads)
        return state.updated(ctx=dict(subsamples=response),
                             debug=dict(sampler=self.__class__.__name__))


class SimulatedAnnealingSubproblemSampler(Runnable):

    def __init__(self, num_reads=1, sweeps=1000):
        self.num_reads = num_reads
        self.sweeps = sweeps
        self.sampler = SimulatedAnnealingSampler()

    @tictoc('subneal_sample')
    def iterate(self, state):
        subbqm = state.ctx['subproblem']
        response = self.sampler.sample(
            subbqm, num_reads=self.num_reads, sweeps=self.sweeps)
        return state.updated(ctx=dict(subsamples=response),
                             debug=dict(sampler=self.__class__.__name__))


class TabuSubproblemSampler(Runnable):

    def __init__(self, num_reads=1, tenure=None, timeout=20):
        self.num_reads = num_reads
        self.tenure = tenure
        self.timeout = timeout
        self.sampler = TabuSampler()

    @tictoc('subtabu_sample')
    def iterate(self, state):
        subbqm = state.ctx['subproblem']
        response = self.sampler.sample(
            subbqm, tenure=self.tenure, timeout=self.timeout, num_reads=self.num_reads)
        return state.updated(ctx=dict(subsamples=response),
                             debug=dict(sampler=self.__class__.__name__))


class TabuProblemSampler(Runnable):

    def __init__(self, bqm, num_reads=1, tenure=None, timeout=20):
        self.bqm = bqm
        self.num_reads = num_reads
        self.tenure = tenure
        self.timeout = timeout
        self.sampler = TabuSampler()

    @tictoc('tabu_sample')
    def iterate(self, state):
        response = self.sampler.sample(
            self.bqm, init_solution=state.samples, tenure=self.tenure,
            timeout=self.timeout, num_reads=self.num_reads)
        return state.updated(samples=SampleSet.from_response(response),
                             debug=dict(sampler=self.__class__.__name__))


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
        return state.updated(debug=dict(sampler=self.__class__.__name__,
                                        runtime=runtime, iterno=iterno))

    def run(self, state):
        self._stop_event.clear()
        return executor.submit(self._interruptable_iterate, state)

    def stop(self):
        self._stop_event.set()


class RandomSubproblemSampler(Runnable):

    @tictoc('random_sample')
    def iterate(self, state):
        bqm = state.ctx['subproblem']
        sample = random_sample(bqm)
        response = SampleSet.from_sample_on_bqm(sample, bqm)
        return state.updated(ctx=dict(subsamples=response),
                             debug=dict(sampler=self.__class__.__name__))
