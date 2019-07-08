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

import numpy as np
import dimod
from neal import SimulatedAnnealingSampler
from dwave.system.testing import MockDWaveSampler

from hybrid.samplers import *
from hybrid.core import State
from hybrid.testing import mock


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
        state = State(problem=bqm, samples=sampleset)

        # with timeout=0, TabuSampler should just return the initial_states
        result = TabuProblemSampler(timeout=0).run(state).result()
        expected = sampleset.record.sample

        self.assertTrue(np.array_equal(result.samples.record.sample, expected))
        self.assertEqual(len(result.samples), 2)

        # test input samples are tiled
        result = TabuProblemSampler(timeout=0, num_reads=4,
                                    initial_states_generator="tile").run(state).result()

        expected = np.tile(sampleset.record.sample, (2,1))

        self.assertTrue(np.array_equal(result.samples.record.sample, expected))
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
        state = State(subproblem=bqm, subsamples=sampleset)

        # with timeout=0, TabuSampler should just return the initial_states
        result = TabuSubproblemSampler(timeout=0).run(state).result()
        expected = sampleset.record.sample

        self.assertTrue(np.array_equal(result.subsamples.record.sample, expected))
        self.assertEqual(len(result.subsamples), 2)

        # test input samples are tiled
        result = TabuSubproblemSampler(timeout=0, num_reads=4,
                                       initial_states_generator="tile").run(state).result()

        expected = np.tile(sampleset.record.sample, (2,1))

        self.assertTrue(np.array_equal(result.subsamples.record.sample, expected))
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
