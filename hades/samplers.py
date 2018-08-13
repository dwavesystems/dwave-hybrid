import time
import threading
from collections import namedtuple

import numpy as np
import networkx as nx

import dimod
from dimod import ExactSolver
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import minorminer

# TODO: pip-ify
from tabu_sampler import TabuSampler

from hades.core import executor, Runnable, State, Sample
from hades.profiling import tictoc
from hades.utils import (
    bqm_induced_by, select_localsearch_adversaries, select_random_subgraph,
    updated_sample, sample_dict_to_list)

import logging
logger = logging.getLogger(__name__)


class SimpleQPUSampler(Runnable):

    def __init__(self, bqm, num_reads=100):
        self.bqm = bqm
        self.num_reads = num_reads
        self.sampler = EmbeddingComposite(DWaveSampler())

    def iterate(self, state):
        response = self.sampler.sample(state.ctx['subproblem'], num_reads=self.num_reads)
        best_response = next(response.data())
        best_sample = best_response.sample
        return state.updated(ctx=dict(subsample=best_sample),
                             debug=dict(source=self.__class__.__name__))


class QPUSubproblemSampler(Runnable):

    def __init__(self, bqm, max_n, num_reads=100):
        self.bqm = bqm
        self.max_n = max_n
        self.num_reads = num_reads

        self.graph = nx.Graph(bqm.adj)
        logger.info("Problem graph connected? %r", nx.is_connected(self.graph))

        self.sampler = DWaveSampler()

    @tictoc()
    def _embed(self, variables, sample):
        subbqm = bqm_induced_by(self.bqm, variables, sample)
        source_edgelist = list(subbqm.quadratic) + [(v, v) for v in subbqm.linear]
        _, target_edgelist, target_adjacency = self.sampler.structure
        embedding = minorminer.find_embedding(source_edgelist, target_edgelist)
        bqm_embedded = dimod.embed_bqm(subbqm, embedding, target_adjacency, chain_strength=1.0)
        return embedding, bqm_embedded, subbqm

    @tictoc(name='qpu_iterate')
    def iterate(self, state):
        """Finds a subproblem to send to QPU, solves it, and returns a proposed
        candidate for new global solution (to replace ``sample``).
        """
        sample = state.sample

        # simple strategy to select high-penalty variables (around sample)
        # Beware: variables might NOT be connected. For better results, QPU-sample connected vars
        #frozen = select_localsearch_adversaries(self.bqm, sample, self.max_n, min_gain=0)
        frozen = select_random_subgraph(self.bqm, self.max_n)

        # inspect subgraph connectivity before embedding
        subgraph = nx.Graph(self.graph.subgraph(frozen))
        logger.info("Subgraph (order %d) connected? %r", subgraph.order(), nx.is_connected(subgraph))

        # embed
        embedding, bqm_embedded, subbqm = self._embed(frozen, sample)

        # sample with qpu
        target_response = self.sampler.sample(bqm_embedded, num_reads=self.num_reads)

        # unembed
        response = dimod.unembed_response(target_response, embedding, subbqm)
        response_datum = next(response.data())
        best_sub_sample = response_datum.sample

        composed_sample = updated_sample(sample, best_sub_sample)
        return State(composed_sample, self.bqm.energy(composed_sample), self.__class__.__name__)


class TabuSubproblemSampler(Runnable):

    def __init__(self, bqm, num_reads=1, tenure=None, timeout=20):
        self.bqm = bqm
        self.num_reads = num_reads
        self.tenure = tenure
        self.timeout = timeout
        self.sampler = TabuSampler()

    @tictoc('subtabu_iterate')
    def iterate(self, state):
        subbqm = state.ctx['subproblem']
        response = self.sampler.sample(
            subbqm, tenure=self.tenure, timeout=self.timeout, num_reads=self.num_reads)
        best_subsample = next(response.samples())
        return state.updated(ctx=dict(subsample=best_subsample),
                             debug=dict(source=self.__class__.__name__))


class TabuProblemSampler(Runnable):

    def __init__(self, bqm, num_reads=1, tenure=None, timeout=20):
        self.bqm = bqm
        self.num_reads = num_reads
        self.tenure = tenure
        self.timeout = timeout
        self.sampler = TabuSampler()

    @tictoc('tabu_iterate')
    def iterate(self, state):
        sample = state.sample.values
        response = self.sampler.sample(
            self.bqm, init_solution=sample, tenure=self.tenure,
            timeout=self.timeout, num_reads=self.num_reads)
        response_datum = next(response.data())
        best_sample = sample_dict_to_list(response_datum.sample)
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

    @tictoc('int_tabu_iterate')
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
