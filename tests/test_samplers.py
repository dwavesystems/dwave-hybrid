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

import unittest
import time
from operator import attrgetter

import numpy as np
from parameterized import parameterized, parameterized_class

import dimod
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler
from dwave.system.testing import MockDWaveSampler

from hybrid.samplers import *
from hybrid.core import State
from hybrid.testing import mock
from hybrid.utils import random_sample

MockDWaveSampler.to_networkx_graph = DWaveSampler.to_networkx_graph


class MockDWaveReverseAnnealingSampler(MockDWaveSampler):
    """Extend the `dwave.system.testing.MockDWaveSampler` with mock support for
    reverse annealing.
    """
    # TODO: move to dwave-system

    def validate_anneal_schedule(self, anneal_schedule):
        return True

    def sample(self, *args, **kwargs):
        self.anneal_schedule = kwargs.pop('anneal_schedule', None)
        self.initial_state = kwargs.pop('initial_state', None)
        return super(MockDWaveReverseAnnealingSampler, self).sample(*args, **kwargs)

class MockDWaveSamplerCounter(MockDWaveSampler):
    """Extend the `dwave.system.testing.MockDWaveSampler` to count how many times 
    the sampler runs.
    """
    count = 0
    def sample(self, *args, **kwargs):
        self.count += 1
        return super(MockDWaveSamplerCounter, self).sample(*args, **kwargs)


class TestQPUSamplers(unittest.TestCase):

    def test_unstructured_child_sampler(self):
        q = QPUSubproblemAutoEmbeddingSampler(qpu_sampler=SimulatedAnnealingSampler())

        # test sampler stays unstructured
        self.assertFalse(isinstance(q.sampler, dimod.Structured))

        # test sampling works
        bqm = dimod.BinaryQuadraticModel({'a': 1}, {}, 0, 'SPIN')
        init = State.from_subsample({'a': 1}, bqm)
        res = q.run(init).result()
        self.assertEqual(res.subsamples.first.energy, -1)

    def test_structured_child_sampler(self):
        q = QPUSubproblemAutoEmbeddingSampler(qpu_sampler=MockDWaveSampler())

        # test sampler is converted to unstructured
        self.assertFalse(isinstance(q.sampler, dimod.Structured))

        # test sampling works
        bqm = dimod.BinaryQuadraticModel({'a': 1}, {}, 0, 'SPIN')
        init = State.from_subsample({'a': 1}, bqm)
        res = q.run(init).result()
        self.assertEqual(res.subsamples.first.energy, -1)

    def test_auto_embedding_failure(self):
        counter = MockDWaveSamplerCounter()
        q = QPUSubproblemAutoEmbeddingSampler(qpu_sampler=counter)

        target_structure = q.sampler.target_structure
        num_vars = len(target_structure.nodelist) + 1 # source graph will be too large for the target and ensure an embedding failure
        bqm = dimod.BinaryQuadraticModel(num_vars, 'SPIN') 
        init = State.from_subsample(random_sample(bqm), bqm)

        retries = 3
        with self.assertRaises(ValueError):
            result = q.run(init, num_retries=retries).result()

        self.assertEqual(retries + 1, counter.count)

    def test_external_embedding_sampler(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 1}, {})
        init = State.from_subproblem(bqm, embedding={'a': [0]})

        sampler = dimod.StructureComposite(
            SimulatedAnnealingSampler(), nodelist=[0], edgelist=[])

        workflow = QPUSubproblemExternalEmbeddingSampler(qpu_sampler=sampler)

        # run mock sampling
        res = workflow.run(init).result()

        # verify mock sampler received custom kwargs
        self.assertEqual(res.subsamples.first.energy, -1)

    def test_external_embedding_sampler_srt(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 1}, {})
        init = State.from_subproblem(bqm, embedding={'a': [0]})

        sampler = dimod.StructureComposite(
            SimulatedAnnealingSampler(), nodelist=[0], edgelist=[])

        # Test srt option, introduced as placeholder
        # functionality for compatibility with latticeLNLS
        # reference workflows (special case use of extended
        # J-range)
        workflow = QPUSubproblemExternalEmbeddingSampler(qpu_sampler=sampler,logical_srt=True)

        # run mock sampling
        res = workflow.run(init).result()

        # verify mock sampler received custom kwargs
        self.assertEqual(res.subsamples.first.energy, -1)

    @parameterized.expand([['chimera', 2], ['pegasus', 1], ['zephyr', 1]])
    def test_clique_embedder(self, topology_type, expected_chain_length):
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': 1, 'bc': 1, 'ca': 1})
        init = State.from_subproblem(bqm)

        sampler = MockDWaveSampler(topology_type=topology_type)
        workflow = SubproblemCliqueEmbedder(sampler=sampler)

        # run embedding
        res = workflow.run(init).result()

        # verify mock sampler received custom kwargs
        self.assertIn('embedding', res)
        self.assertEqual(len(res.embedding.keys()), 3)

        # embedding a triangle onto a chimera produces 3 x 2-qubit chains;
        # if embedding onto pegasus the chains have length 1.
        self.assertTrue(all(len(e) == expected_chain_length for e in res.embedding.values()))

    def test_reverse_annealing_sampler(self):
        sampler = MockDWaveReverseAnnealingSampler()
        ra = ReverseAnnealingAutoEmbeddingSampler(qpu_sampler=sampler)

        # test sampling works
        bqm = dimod.BinaryQuadraticModel({'a': 1}, {}, 0, 'SPIN')
        state = State.from_subsample({'a': 1}, bqm)
        res = ra.run(state).result()

        self.assertEqual(res.subsamples.first.energy, -1)
        self.assertEqual(sampler.initial_state.popitem()[1], 1)
        self.assertEqual(sampler.anneal_schedule, ra.anneal_schedule)

    def test_custom_qpu_params(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 1}, {})
        init = State.from_subproblem(bqm)

        # define a mock sampler that exposes some parameters of interest
        mock_sampler = mock.MagicMock()
        mock_sampler.parameters = {
            'num_reads': [], 'chain_strength': [], 'future_proof': []}

        qpu_params = dict(chain_strength=2, future_proof=True)

        workflow = QPUSubproblemAutoEmbeddingSampler(
            num_reads=10, qpu_sampler=mock_sampler, sampling_params=qpu_params)

        # run mock sampling
        workflow.run(init).result()

        # verify mock sampler received custom kwargs
        mock_sampler.sample.assert_called_once_with(
            bqm, num_reads=10, **qpu_params)


class TestTabuSamplers(unittest.TestCase):

    def test_tabu_problem_sampler_interface(self):
        bqm = dimod.BinaryQuadraticModel({'a': 1}, {}, 0, 'SPIN')

        workflow = TabuProblemSampler(num_reads=10)

        init = State.from_sample({'a': 1}, bqm)
        result = workflow.run(init).result()

        self.assertEqual(result.samples.first.energy, -1)
        self.assertEqual(len(result.samples), 10)

    def test_tabu_problem_sampler_functionality(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': -1, 'ca': 1}, 0, 'SPIN')

        workflow = TabuProblemSampler()

        # use random sample as initial value
        init = State(problem=bqm, samples=None)
        result = workflow.run(init).result()

        self.assertEqual(result.samples.first.energy, -3)
        self.assertEqual(len(result.samples), 1)

    def test_tabu_problem_sampler_initialization(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1}, 0, 'SPIN')
        sampleset = dimod.SampleSet.from_samples_bqm([{'a': 1, 'b': -1},
                                                      {'a': -1, 'b': 1}], bqm)
        state = State(problem=bqm, samples=sampleset.copy())

        # with timeout=0, TabuSampler should just return the initial_states
        expected = sampleset.record.sample
        result = TabuProblemSampler(timeout=0).run(state).result()

        np.testing.assert_array_equal(result.samples.record.sample, expected)
        self.assertEqual(len(result.samples), 2)

        # test input samples are tiled
        result = TabuProblemSampler(timeout=0, num_reads=4,
                                    initial_states_generator="tile").run(state).result()

        expected = np.tile(sampleset.record.sample, (2,1))

        np.testing.assert_array_equal(result.samples.record.sample, expected)
        self.assertEqual(len(result.samples), 4)

    def test_tabu_subproblem_sampler(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': -1, 'ca': 1}, 0, 'SPIN')

        workflow = TabuSubproblemSampler()

        # use random sample as initial value
        init = State(subproblem=bqm, subsamples=None)
        result = workflow.run(init).result()

        self.assertEqual(result.subsamples.first.energy, -3)
        self.assertEqual(len(result.subsamples), 1)

    def test_tabu_subproblem_sampler_initialization(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1}, 0, 'SPIN')
        sampleset = dimod.SampleSet.from_samples_bqm([{'a': 1, 'b': -1},
                                                      {'a': -1, 'b': 1}], bqm)
        state = State(subproblem=bqm, subsamples=sampleset.copy())

        # with timeout=0, TabuSampler should just return the initial_states
        result = TabuSubproblemSampler(timeout=0).run(state).result()
        expected = sampleset.record.sample

        np.testing.assert_array_equal(result.subsamples.record.sample, expected)
        self.assertEqual(len(result.subsamples), 2)

        # test input samples are tiled
        result = TabuSubproblemSampler(timeout=0, num_reads=4,
                                       initial_states_generator="tile").run(state).result()

        expected = np.tile(sampleset.record.sample, (2,1))

        np.testing.assert_array_equal(result.subsamples.record.sample, expected)
        self.assertEqual(len(result.subsamples), 4)

    def test_interruptable_tabu(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': -1, 'ca': 1}, 0, 'SPIN')

        workflow = InterruptableTabuSampler()

        init = State(problem=bqm)
        future = workflow.run(init)
        while len(workflow.runnable.timers.get('dispatch.next', ())) < 1:
            time.sleep(0)

        workflow.stop()

        self.assertEqual(future.result().samples.first.energy, -3)
        self.assertGreater(len(workflow.timers['dispatch.next']), 0)


class TestSASamplers(unittest.TestCase):

    def test_sa_problem_sampler_interface(self):
        bqm = dimod.BinaryQuadraticModel({'a': 1}, {}, 0, 'SPIN')

        workflow = SimulatedAnnealingProblemSampler(num_reads=10)

        init = State.from_sample({'a': 1}, bqm)
        result = workflow.run(init).result()

        self.assertEqual(result.samples.first.energy, -1)
        self.assertEqual(len(result.samples), 10)

    def test_sa_problem_sampler_functionality(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': -1, 'ca': 1}, 0, 'SPIN')

        workflow = SimulatedAnnealingProblemSampler(num_reads=10)

        # use a random sample as initial value
        init = State(problem=bqm, samples=None)
        result = workflow.run(init).result()

        self.assertEqual(result.samples.first.energy, -3)
        self.assertEqual(len(result.samples), 10)

    def test_sa_problem_sampler_initialization(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1}, 0, 'SPIN')
        sampleset = dimod.SampleSet.from_samples_bqm([{'a': 1, 'b': -1},
                                                      {'a': -1, 'b': 1}], bqm)
        state = State(problem=bqm, samples=sampleset)

        # with timeout=0, TabuSampler should just return the initial_states
        result = SimulatedAnnealingProblemSampler(num_sweeps=0).run(state).result()
        expected = sampleset.record.sample

        self.assertTrue(np.array_equal(result.samples.record.sample, expected))
        self.assertEqual(len(result.samples), 2)

        # test input samples are tiled
        result = SimulatedAnnealingProblemSampler(
            num_sweeps=0, num_reads=4, initial_states_generator="tile").run(state).result()

        expected = np.tile(sampleset.record.sample, (2,1))

        self.assertTrue(np.array_equal(result.samples.record.sample, expected))
        self.assertEqual(len(result.samples), 4)

    def test_sa_subproblem_sampler_initialization(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1}, 0, 'SPIN')
        sampleset = dimod.SampleSet.from_samples_bqm([{'a': 1, 'b': -1},
                                                      {'a': -1, 'b': 1}], bqm)
        state = State(subproblem=bqm, subsamples=sampleset)

        # with timeout=0, TabuSampler should just return the initial_states
        result = SimulatedAnnealingSubproblemSampler(num_sweeps=0).run(state).result()
        expected = sampleset.record.sample

        self.assertTrue(np.array_equal(result.subsamples.record.sample, expected))
        self.assertEqual(len(result.subsamples), 2)

        # test input samples are tiled
        result = SimulatedAnnealingSubproblemSampler(
            num_sweeps=0, num_reads=4, initial_states_generator="tile").run(state).result()

        expected = np.tile(sampleset.record.sample, (2,1))

        self.assertTrue(np.array_equal(result.subsamples.record.sample, expected))
        self.assertEqual(len(result.subsamples), 4)


@parameterized_class(("sampler_cls", "state_gen", "get_samples"), [
    (GreedyProblemSampler, State.from_problem, attrgetter('samples')),
    (GreedySubproblemSampler, State.from_subproblem, attrgetter('subsamples')),
])
class TestGreedySamplers(unittest.TestCase):

    def test_greedy_sampler_interface(self):
        bqm = dimod.BinaryQuadraticModel({'a': 1}, {}, 0, 'SPIN')

        workflow = self.sampler_cls(num_reads=10)

        init = self.state_gen(bqm, {'a': 1})
        result = workflow.run(init).result()

        samples = self.get_samples(result)
        self.assertEqual(samples.first.energy, -1)
        self.assertEqual(len(samples), 10)

    def test_greedy_sampler_functionality(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': -1, 'ca': 1}, 0, 'SPIN')

        workflow = self.sampler_cls(num_reads=10)

        init = self.state_gen(bqm, random_sample)
        result = workflow.run(init).result()

        samples = self.get_samples(result)
        self.assertEqual(samples.first.energy, -3)
        self.assertEqual(len(samples), 10)

    def test_greedy_sampler_initialization(self):
        # a convex section of hyperbolic paraboloid in the Ising space,
        # with global minimum at (-1,-1)
        bqm = dimod.BinaryQuadraticModel.from_ising({0: 2, 1: 2}, {(0, 1): -1})
        sampleset = dimod.SampleSet.from_samples_bqm([{0: 1, 1: -1},
                                                      {0: -1, 1: 1}], bqm)
        ground = dimod.SampleSet.from_samples_bqm([{0: -1, 1: -1},
                                                   {0: -1, 1: -1}], bqm)
        state = self.state_gen(bqm, sampleset)

        # implicit number of samples
        result = self.sampler_cls().run(state).result()
        self.assertEqual(len(self.get_samples(result)), 2)

        # test input samples are tiled
        result = self.sampler_cls(
            num_reads=4, initial_states_generator="tile").run(state).result()

        expected = np.tile(ground.record.sample, (2,1))

        samples = self.get_samples(result)
        self.assertTrue(np.array_equal(samples.record.sample, expected))
        self.assertEqual(len(samples), 4)

    def test_greedy_sampler_is_monotonic(self):
        workflow = self.sampler_cls(num_reads=1)

        bqm = dimod.generators.ran_r(32, 32)
        state = self.state_gen(bqm)

        trials = 32
        for _ in range(trials):
            next_state = workflow.run(state).result()
            self.assertLessEqual(
                self.get_samples(next_state).first.energy,
                self.get_samples(state).first.energy)
            state = next_state
