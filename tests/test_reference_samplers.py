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

import dimod
from dwave.system.testing import MockDWaveSampler

import hybrid
from hybrid.reference.kerberos import KerberosSampler
from hybrid.reference.pa import (
    EnergyWeightedResampler, ProgressBetaAlongSchedule,
    CalculateAnnealingBetaSchedule
)


class TestKerberos(unittest.TestCase):

    def test_basic_operation(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'ca': 1}, 0, dimod.SPIN)
        sampleset = KerberosSampler().sample(
            bqm, max_subproblem_size=1, qpu_sampler=MockDWaveSampler(),
            qpu_params=dict(chain_strength=2))


class TestWeightedResampler(unittest.TestCase):

    def test_sampling(self):
        # for all practical purposes the distribution of energies here should
        # guarantee the last sample always wins
        winner = {'a': 1, 'b': 1}
        skewed = dimod.SampleSet.from_samples(
            [{'a': 0, 'b': 0}, {'a': 0, 'b': 1}, {'a': 1, 'b': 0}, winner],
            energy=[100, 100, 100, -100], vartype='BINARY')

        state = hybrid.State(samples=skewed)

        # cold sampling
        res = EnergyWeightedResampler(beta=1, seed=1234).run(state).result()
        samples = res.samples.aggregate()

        self.assertEqual(len(samples), 1)
        self.assertDictEqual(dict(list(samples.samples())[0]), winner)

        # hot sampling
        res = EnergyWeightedResampler(beta=0, seed=1234).run(state).result()
        samples = res.samples.aggregate()

        self.assertGreater(len(samples), 1)

    def test_beta_use(self):
        ss = dimod.SampleSet.from_samples([{'a': 0}], energy=[0], vartype='SPIN')
        state = hybrid.State(samples=ss)

        # beta not given at all
        with self.assertRaises(ValueError):
            res = EnergyWeightedResampler().run(state).result()

        # beta given on construction
        res = EnergyWeightedResampler(beta=0).run(state).result()
        self.assertEqual(res.samples.info['beta'], 0)

        # beta given on runtime, to run method
        res = EnergyWeightedResampler().run(state, beta=1).result()
        self.assertEqual(res.samples.info['beta'], 1)

        # beta given in state
        state.beta = 2
        res = EnergyWeightedResampler().run(state).result()
        self.assertEqual(res.samples.info['beta'], 2)


class TestPopulationAnnealingUtils(unittest.TestCase):

    def test_beta_progressor(self):
        beta_schedule = [1, 2, 3]

        prog = ProgressBetaAlongSchedule(beta_schedule=beta_schedule)

        betas = []
        while True:
            try:
                betas.append(prog.run(hybrid.State()).result().beta)
            except:
                break

        self.assertEqual(betas, beta_schedule)

    def test_beta_schedule_calc_smoketest(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 1}, {})
        state = hybrid.State.from_problem(bqm)

        # linear interp
        calc = CalculateAnnealingBetaSchedule(length=10, interpolation='linear')
        res = calc.run(state).result()
        self.assertIn('beta_schedule', res)
        self.assertEqual(len(res.beta_schedule), 10)

        # geometric interp
        calc = CalculateAnnealingBetaSchedule(length=10, interpolation='geometric')
        res = calc.run(state).result()
        self.assertIn('beta_schedule', res)
        self.assertEqual(len(res.beta_schedule), 10)
