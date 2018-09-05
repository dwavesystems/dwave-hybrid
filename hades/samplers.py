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
    """A quantum sampler for a subproblem with a defined minor-embedding.

    Args:
        num_reads (int, optional, default=100):
            Number of states (output solutions) to read from the sampler.
        qpu_sampler (:class:`dimod.Sampler`, optional, default=DWaveSampler()):
            Quantum sampler such as a D-Wave system.

    Examples:
        This example works on a binary quadratic model of two AND gates in series
        by sampling a BQM representing just one of the gates. Output :math:`z` of gate
        :math:`z = x \wedge y` connects to input :math:`a` of gate :math:`c = a \wedge b`.
        An initial state is manually set with invalid solution :math:`x=y=0, z=1; a=b=1, c=0`.
        The state is updated by sampling the subproblem 100 times on a D-Wave system.
        The execution results shown here were three valid solutions to the subproblem; for
        example, :math:`x=0, y=1, z=0` occurred 22 times.

        >>> import dimod
        >>> from dwave.system.samplers import DWaveSampler
        >>> import minorminer
        ...
        >>> # Define a problem and a subproblem
        >>> bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0, 'a': 2.0, 'b': 0.0, 'c': 6.0},
        ...                                  {('y', 'x'): 2.0, ('z', 'x'): -4.0, ('z', 'y'): -4.0,
        ...                                  ('b', 'a'): 2.0, ('c', 'a'): -4.0, ('c', 'b'): -4.0, ('a', 'z'): -4.0},
        ...                                  -1.0, 'BINARY')
        >>> sub_bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0},
        ...                                      {('x', 'y'): 2.0, ('x', 'z'): -4.0, ('y', 'z'): -4.0},
        ...                                      -1.0, dimod.Vartype.BINARY)
        >>> # Find a minor-embedding for the subproblem
        >>> qpu_sampler = DWaveSampler()
        >>> sub_embedding = minorminer.find_embedding(list(sub_bqm.quadratic.keys()), qpu_sampler.edgelist)
        >>> # Set up the sampler with an initial state
        >>> sampler = samplers.QPUSubproblemExternalEmbeddingSampler(num_reads=100)
        >>> state = core.State().from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm)
        >>> state.ctx.update(subproblem=sub_bqm, embedding=sub_embedding)
        >>> # Sample the subproblem on the QPU
        >>> new_state = sampler.iterate(state)
        >>> print(new_state.ctx['subsamples'].record)      # doctest: +SKIP
        [([0, 1, 0], -1., 22) ([0, 0, 0], -1., 47) ([1, 0, 0], -1., 31)]


    """

    def __init__(self, num_reads=100, qpu_sampler=None):
        self.num_reads = num_reads
        if qpu_sampler is None:
            qpu_sampler = DWaveSampler()
        self.sampler = qpu_sampler

    @tictoc('qpu_ext_embedding_sample')
    def iterate(self, state):
        sampler = FixedEmbeddingComposite(self.sampler, embedding=state.ctx['embedding'])
        response = sampler.sample(state.ctx['subproblem'], num_reads=self.num_reads)
        return state.updated(ctx=dict(subsamples=response),
                             debug=dict(sampler=self.name))


class QPUSubproblemAutoEmbeddingSampler(Runnable):
    """A quantum sampler for a subproblem with automated heuristic minor-embedding.

    Args:
        num_reads (int, optional, default=100):
            Number of states (output solutions) to read from the sampler.
        qpu_sampler (:class:`dimod.Sampler`, optional, default=DWaveSampler()):
            Quantum sampler such as a D-Wave system.

    Examples:
        This example works on a binary quadratic model of two AND gates in series
        by sampling a BQM representing just one of the gates. Output :math:`z` of gate
        :math:`z = x \wedge y` connects to input :math:`a` of gate :math:`c = a \wedge b`.
        An initial state is manually set with invalid solution :math:`x=y=0, z=1; a=b=1, c=0`.
        The state is updated by sampling the subproblem 100 times on a D-Wave system.
        The execution results shown here were four valid solutions to the subproblem; for
        example, :math:`x=0, y=0, z=0` occurred 53 times.

        >>> import dimod
        >>> from dwave.system.samplers import DWaveSampler
        ...
        >>> # Define a problem and a subproblem
        >>> bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0, 'a': 2.0, 'b': 0.0, 'c': 6.0},
        ...                                  {('y', 'x'): 2.0, ('z', 'x'): -4.0, ('z', 'y'): -4.0,
        ...                                  ('b', 'a'): 2.0, ('c', 'a'): -4.0, ('c', 'b'): -4.0, ('a', 'z'): -4.0},
        ...                                  -1.0, 'BINARY')
        >>> sub_bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0},
        ...                                      {('x', 'y'): 2.0, ('x', 'z'): -4.0, ('y', 'z'): -4.0},
        ...                                      -1.0, dimod.Vartype.BINARY)
        >>> # Set up the sampler with an initial state
        >>> sampler = samplers.QPUSubproblemAutoEmbeddingSampler(num_reads=100)
        >>> state = core.State().from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm)
        >>> state.ctx.update(subproblem=sub_bqm)
        >>> # Sample the subproblem on the QPU
        >>> new_state = sampler.iterate(state)
        >>> print(new_state.ctx['subsamples'].record)      # doctest: +SKIP
        [([0, 0, 0], -1., 53) ([0, 1, 0], -1., 15) ([1, 0, 0], -1., 31)
         ([1, 1, 1],  1.,  1)]


    """

    def __init__(self, num_reads=100, qpu_sampler=None):
        self.num_reads = num_reads
        if qpu_sampler is None:
            qpu_sampler = DWaveSampler()
        self.sampler = EmbeddingComposite(qpu_sampler)

    @tictoc('qpu_auto_embedding_sample')
    def iterate(self, state):
        response = self.sampler.sample(state.ctx['subproblem'], num_reads=self.num_reads)
        return state.updated(ctx=dict(subsamples=response),
                             debug=dict(sampler=self.name))


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
                             debug=dict(sampler=self.name))


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
                             debug=dict(sampler=self.name))


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
                             debug=dict(sampler=self.name))


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
        return state.updated(debug=dict(sampler=self.name,
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
                             debug=dict(sampler=self.name))
