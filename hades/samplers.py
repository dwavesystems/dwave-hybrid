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
import threading
from collections import namedtuple

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite

from tabu import TabuSampler
from neal import SimulatedAnnealingSampler

from hades.core import async_executor, Runnable, SampleSet
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
        >>> state = core.State.from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm)
        >>> state.update(subproblem=sub_bqm, embedding=sub_embedding)
        >>> # Sample the subproblem on the QPU
        >>> new_state = sampler.iterate(state)
        >>> print(new_state.subsamples.record)      # doctest: +SKIP
        [([0, 1, 0], -1., 22) ([0, 0, 0], -1., 47) ([1, 0, 0], -1., 31)]


    """

    def __init__(self, num_reads=100, qpu_sampler=None):
        self.num_reads = num_reads
        if qpu_sampler is None:
            qpu_sampler = DWaveSampler()
        self.sampler = qpu_sampler

    @tictoc('qpu_ext_embedding_sample')
    def iterate(self, state):
        sampler = FixedEmbeddingComposite(self.sampler, embedding=state.embedding)
        response = sampler.sample(state.subproblem, num_reads=self.num_reads)
        return state.updated(subsamples=response,
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
        >>> state = core.State.from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm)
        >>> state.update(subproblem=sub_bqm)
        >>> # Sample the subproblem on the QPU
        >>> new_state = sampler.iterate(state)
        >>> print(new_state.subsamples.record)      # doctest: +SKIP
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
        response = self.sampler.sample(state.subproblem, num_reads=self.num_reads)
        return state.updated(subsamples=response,
                             debug=dict(sampler=self.name))


class SimulatedAnnealingSubproblemSampler(Runnable):
    """A simulated annealing sampler for a subproblem.

    Args:
        num_reads (int, optional, default=1):
            Number of states (output solutions) to read from the sampler.
        sweeps (int, optional, default=1000):
            Number of sweeps or steps.

    Examples:
        This example works on a binary quadratic model of two AND gates in series
        by sampling a BQM representing just one of the gates. Output :math:`z` of gate
        :math:`z = x \wedge y` connects to input :math:`a` of gate :math:`c = a \wedge b`.
        An initial state is manually set with invalid solution :math:`x=y=0, z=1; a=b=1, c=0`.
        The state is updated by sampling the subproblem 10 times.
        The execution results shown here were valid solutions to the subproblem; for
        example, :math:`x=0, y=1, z=0`.

        >>> import dimod
        >>> from neal import SimulatedAnnealingSampler
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
        >>> sampler = samplers.SimulatedAnnealingSubproblemSampler(num_reads=10)
        >>> state = core.State.from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm)
        >>> state.update(subproblem=sub_bqm)
        >>> # Sample the subproblem
        >>> new_state = sampler.iterate(state)
        >>> print(new_state.subsamples.record)      # doctest: +SKIP
        [([0, 1, 0], -1., 1) ([0, 1, 0], -1., 1) ([0, 0, 0], -1., 1)
        ([0, 0, 0], -1., 1) ([0, 0, 0], -1., 1) ([1, 0, 0], -1., 1)
        ([1, 0, 0], -1., 1) ([0, 0, 0], -1., 1) ([0, 1, 0], -1., 1)
        ([1, 0, 0], -1., 1)]

    """

    def __init__(self, num_reads=1, sweeps=1000):
        self.num_reads = num_reads
        self.sweeps = sweeps
        self.sampler = SimulatedAnnealingSampler()
        self._stop_event = threading.Event()

    @tictoc('subneal_sample')
    def iterate(self, state):
        subbqm = state.subproblem
        response = self.sampler.sample(
            subbqm, num_reads=self.num_reads, sweeps=self.sweeps,
            interrupt_function=lambda: self._stop_event.is_set())
        return state.updated(subsamples=response,
                             debug=dict(sampler=self.name))

    def stop(self):
        self._stop_event.set()


class InterruptableSimulatedAnnealingSubproblemSampler(SimulatedAnnealingSubproblemSampler):
    """SimulatedAnnealingSubproblemSampler is already interruptable."""
    pass


class TabuSubproblemSampler(Runnable):
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
        This example works on a binary quadratic model of two AND gates in series
        by sampling a BQM representing just one of the gates. Output :math:`z` of gate
        :math:`z = x \wedge y` connects to input :math:`a` of gate :math:`c = a \wedge b`.
        An initial state is manually set with invalid solution :math:`x=y=0, z=1; a=b=1, c=0`.
        The state is updated by a tabu search on the subproblem.
        The execution results shown here was a valid solution to the subproblem:
        example, :math:`x=0, y=1, z=0`.

        >>> import dimod
        >>> from tabu import TabuSampler
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
        >>> sampler = samplers.TabuSubproblemSampler(tenure=2, timeout=5)
        >>> state = core.State.from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm)
        >>> state.update(subproblem=sub_bqm)
        >>> # Sample the subproblem
        >>> new_state = sampler.iterate(state)
        >>> print(new_state.subsamples.record)      # doctest: +SKIP
        [([0, 1, 0], -1., 1)]

    """

    def __init__(self, num_reads=1, tenure=None, timeout=20):
        self.num_reads = num_reads
        self.tenure = tenure
        self.timeout = timeout
        self.sampler = TabuSampler()

    @tictoc('subtabu_sample')
    def iterate(self, state):
        subbqm = state.subproblem
        response = self.sampler.sample(
            subbqm, tenure=self.tenure, timeout=self.timeout, num_reads=self.num_reads)
        return state.updated(subsamples=response,
                             debug=dict(sampler=self.name))


class TabuProblemSampler(Runnable):
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
        This example works on a binary quadratic model of two AND gates in series, where
        output :math:`z` of gate :math:`z = x \wedge y` connects to input :math:`a`
        of gate :math:`c = a \wedge b`. An initial state is manually set with invalid
        solution :math:`x=y=0, z=1; a=b=1, c=0`. The state is updated by a tabu search.
        The execution results shown here was a valid solution to the problem:
        example, :math:`x=y=z=a=b=c=1`.

        >>> import dimod
        >>> from tabu import TabuSampler
        ...
        >>> # Define a problem
        >>> bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0, 'a': 2.0, 'b': 0.0, 'c': 6.0},
        ...                                  {('y', 'x'): 2.0, ('z', 'x'): -4.0, ('z', 'y'): -4.0,
        ...                                  ('b', 'a'): 2.0, ('c', 'a'): -4.0, ('c', 'b'): -4.0, ('a', 'z'): -4.0},
        ...                                  -1.0, 'BINARY')
        >>> # Set up the sampler with an initial state
        >>> sampler = samplers.TabuProblemSampler(tenure=2, timeout=5)
        >>> state = State.from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm)
        >>> # Sample the problem
        >>> new_state = sampler.iterate(state)
        >>> print(new_state.samples)      # doctest: +SKIP
        Response(rec.array([([1, 1, 1, 1, 1, 1], -1., 1)],
          dtype=[('sample', 'i1', (6,)), ('energy', '<f8'), ('num_occurrences', '<i4')]),
          ['a', 'b', 'c', 'x', 'y', 'z'], {}, 'BINARY')

    """

    def __init__(self, num_reads=1, tenure=None, timeout=20):
        self.num_reads = num_reads
        self.tenure = tenure
        self.timeout = timeout
        self.sampler = TabuSampler()

    @tictoc('tabu_sample')
    def iterate(self, state):
        response = self.sampler.sample(
            state.problem, init_solution=state.samples, tenure=self.tenure,
            timeout=self.timeout, num_reads=self.num_reads)
        return state.updated(samples=SampleSet.from_response(response),
                             debug=dict(sampler=self.name))


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

    Examples:
        This example works on a binary quadratic model of two AND gates in series, where
        output :math:`z` of gate :math:`z = x \wedge y` connects to input :math:`a`
        of gate :math:`c = a \wedge b`. An initial state is manually set with invalid
        solution :math:`x=y=0, z=1; a=b=1, c=0`. The state is updated by a tabu search.
        The execution results shown here was a valid solution to the problem:
        example, :math:`x=y=z=a=b=c=1`.

        >>> import dimod
        >>> from tabu import TabuSampler
        ...
        >>> # Define a problem
        >>> bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0, 'a': 2.0, 'b': 0.0, 'c': 6.0},
        ...                                  {('y', 'x'): 2.0, ('z', 'x'): -4.0, ('z', 'y'): -4.0,
        ...                                  ('b', 'a'): 2.0, ('c', 'a'): -4.0, ('c', 'b'): -4.0, ('a', 'z'): -4.0},
        ...                                  -1.0, 'BINARY')
        >>> # Set up the sampler with an initial state
        >>> sampler = samplers.InterruptableTabuSampler(tenure=2, quantum_timeout=30, timeout=5000)
        >>> state = core.State.from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm)
        >>> # Sample the problem
        >>> new_state = sampler.run(state)
        >>> new_state  # doctest: +SKIP
        <Future at 0x179eae59898 state=running>
        >>> sampler.stop()
        >>> new_state  # doctest: +SKIP
        <Future at 0x179eae59898 state=finished returned State>
        >>> print(new_state.result())      # doctest: +SKIP
        State(samples=Response(rec.array([([1, 1, 1, 1, 1, 1], -1., 1)],
          dtype=[('sample', 'i1', (6,)), ('energy', '<f8'), ('num_occurrences', '<i4')]),
          ['a', 'b', 'c', 'x', 'y', 'z'], {}, 'BINARY'),
          debug={'sampler': 'InterruptableTabuSampler', 'runtime': 62.85970854759216, 'iterno': 2082})

    """

    def __init__(self, quantum_timeout=20, timeout=None, **kwargs):
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
        return async_executor.submit(self._interruptable_iterate, state)

    def stop(self):
        self._stop_event.set()


class RandomSubproblemSampler(Runnable):
    """A random sample generator for a subproblem.

    Examples:
        This example works on a binary quadratic model of two AND gates in series
        by sampling a BQM representing just one of the gates. Output :math:`z` of gate
        :math:`z = x \wedge y` connects to input :math:`a` of gate :math:`c = a \wedge b`.
        An initial state is manually set with invalid solution :math:`x=y=0, z=1; a=b=1, c=0`.
        The state is updated with a random sample..

        >>> import dimod
        >>> # Define a problem and a subproblem
        >>> bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0, 'a': 2.0, 'b': 0.0, 'c': 6.0},
        ...                                  {('y', 'x'): 2.0, ('z', 'x'): -4.0, ('z', 'y'): -4.0,
        ...                                  ('b', 'a'): 2.0, ('c', 'a'): -4.0, ('c', 'b'): -4.0, ('a', 'z'): -4.0},
        ...                                  -1.0, 'BINARY')
        >>> sub_bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0},
        ...                                      {('x', 'y'): 2.0, ('x', 'z'): -4.0, ('y', 'z'): -4.0},
        ...                                      -1.0, dimod.Vartype.BINARY)
        >>> # Set up the sampler with an initial state
        >>> sampler = samplers.RandomSubproblemSampler()
        >>> state = core.State.from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm)
        >>> state.update(subproblem=sub_bqm)
        >>> # Sample the subproblem a couple of times
        >>> new_state = sampler.iterate(state)
        >>> print(new_state.subsamples.record)      # doctest: +SKIP
        [([0, 0, 0], -1., 1)]
        >>> new_state = sampler.iterate(state)
        >>> print(new_state.subsamples.record)      # doctest: +SKIP
        [([1, 1, 1], 1., 1)]

    """

    @tictoc('random_sample')
    def iterate(self, state):
        bqm = state.subproblem
        sample = random_sample(bqm)
        response = SampleSet.from_sample_on_bqm(sample, bqm)
        return state.updated(subsamples=response,
                             debug=dict(sampler=self.name))
