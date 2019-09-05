# encoding: utf-8

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

import itertools
import unittest

import numpy as np
import networkx as nx
from plucky import pluck

import dimod

from hybrid.core import State, States, SampleSet
from hybrid import traits
from hybrid.composers import (
    IdentityComposer, SplatComposer, GreedyPathMerge,
    MergeSamples, ExplodeSamples, SliceSamples, AggregatedSamples,
    IsoenergeticClusterMove)
from hybrid.utils import min_sample, max_sample, random_sample


class TestIdentityComposer(unittest.TestCase):
    problem = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'ca': 1}, 0, dimod.SPIN)
    samples = [{'a': 1, 'b': 1, 'c': -1}]

    def test_default(self):
        """Subsamples are copied to samples."""

        state = State(
            subproblem=None,
            subsamples=SampleSet.from_samples_bqm(self.samples, self.problem))

        nextstate = IdentityComposer().next(state)
        self.assertEqual(state.subsamples, nextstate.samples)

    def test_traits_enforced(self):
        """Sample composers require `problem`, `samples` and `subsamples`."""

        with self.assertRaises(traits.StateTraitMissingError):
            IdentityComposer().run(State()).result()
        with self.assertRaises(traits.StateTraitMissingError):
            IdentityComposer().run(State(problem=True)).result()
        with self.assertRaises(traits.StateTraitMissingError):
            IdentityComposer().run(State(problem=True, samples=True)).result()
        self.assertTrue(
            IdentityComposer().run(State(problem=True, samples=True, subsamples=True)).result())


class TestSplatComposer(unittest.TestCase):
    problem = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'ca': 1}, 0, dimod.SPIN)
    samples = [{'a': +1, 'b': +1, 'c': +1},
               {'a': -1, 'b': -1, 'c': -1}]
    subproblem = dimod.BinaryQuadraticModel({}, {'bc': 1}, 0, dimod.SPIN)
    subsamples = [{'b': -1, 'c': +1},
                  {'b': +1, 'c': +1}]
    composed = [{'a': +1, 'b': -1, 'c': +1},
                {'a': -1, 'b': +1, 'c': +1}]

    def test_default(self):
        """All subsamples are combined with all the samples."""

        state = State.from_samples(self.samples, self.problem).updated(
            subproblem=self.subproblem,
            subsamples=SampleSet.from_samples_bqm(self.subsamples, self.subproblem))

        nextstate = SplatComposer().next(state)

        self.assertEqual(nextstate.samples,
                         SampleSet.from_samples_bqm(self.composed, self.problem))

    def test_traits_enforced(self):
        """Sample composers require `problem`, `samples` and `subsamples`."""

        with self.assertRaises(traits.StateTraitMissingError):
            SplatComposer().run(State()).result()
        with self.assertRaises(traits.StateTraitMissingError):
            SplatComposer().run(State(problem=True)).result()
        self.assertTrue(
            # problem and samples are included by default
            SplatComposer().run(State(
                problem=self.problem, subproblem=self.subproblem,
                samples=SampleSet.from_samples_bqm(self.samples, self.problem),
                subsamples=SampleSet.from_samples_bqm(self.subsamples, self.subproblem))).result())


class TestGreedyPathMerge(unittest.TestCase):

    def test_basic(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': -1, 'ca': 1}, 0, dimod.SPIN)
        state = State.from_sample(min_sample(bqm), bqm)
        antistate = State.from_sample(max_sample(bqm), bqm)

        result = GreedyPathMerge().run(States(state, antistate)).result()

        self.assertEqual(result.samples.first.energy, -3.0)


class TestMergeSamples(unittest.TestCase):

    def test_single(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1}, 0, dimod.SPIN)

        states = States(State.from_sample({'a': 1, 'b': -1}, bqm))

        state = MergeSamples().run(states).result()

        self.assertEqual(state, states[0])

    def test_multiple(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1}, 0, dimod.SPIN)

        states = States(State.from_sample({'a': 1, 'b': -1}, bqm),
                        State.from_sample({'a': -1, 'b': 1}, bqm))

        expected = State.from_samples([{'a': 1, 'b': -1}, {'a': -1, 'b': 1}], bqm)

        state = MergeSamples().run(states).result()

        self.assertEqual(state, expected)

    def test_aggregation(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1}, 0, dimod.SPIN)

        states = States(State.from_sample({'a': 1, 'b': -1}, bqm),
                        State.from_sample({'a': 1, 'b': -1}, bqm))

        expected = State(
            problem=bqm,
            samples=dimod.SampleSet.from_samples_bqm(
                {'a': 1, 'b': -1}, bqm, num_occurrences=[2]))

        state = MergeSamples(aggregate=True).run(states).result()

        self.assertEqual(state, expected)


class TestExplodeSamples(unittest.TestCase):

    def test_empty(self):
        "At least one input sample is required."

        bqm = dimod.BQM.from_ising({}, {'ab': 1})

        inp = State(problem=bqm, samples=None)
        with self.assertRaises(ValueError):
            ExplodeSamples().run(inp).result()

        inp = State(problem=bqm, samples=SampleSet.empty())
        with self.assertRaises(ValueError):
            ExplodeSamples().run(inp).result()

    def test_single(self):
        "One input sample should produce one output state with that sample."

        bqm = dimod.BQM.from_ising({}, {'ab': 1})

        inp = State.from_sample({'a': 1, 'b': 1}, bqm)

        exp = States(inp.updated())

        out = ExplodeSamples().run(inp).result()

        self.assertEqual(out, exp)

    def test_simple(self):
        "Two output states created for two input samples, in correct order."

        bqm = dimod.BQM.from_ising({}, {'ab': 1})

        inp = State.from_samples([{'a': 1, 'b': 1},
                                  {'a': -1, 'b': 1}], bqm)

        exp = States(State.from_sample({'a': 1, 'b': 1}, bqm),
                     State.from_sample({'a': -1, 'b': 1}, bqm))

        out = ExplodeSamples().run(inp).result()

        self.assertEqual(out, exp)


class TestSliceSamples(unittest.TestCase):

    def test_bottom_n(self):
        energies = list(range(10))
        sampleset = dimod.SampleSet.from_samples(np.ones((10, 1)), dimod.SPIN, energy=energies)
        state = State(samples=sampleset)

        bottom = SliceSamples(3).run(state).result()
        self.assertEqual(bottom.samples, sampleset.truncate(3))

        bottom = SliceSamples().run(state, stop=3).result()
        self.assertEqual(bottom.samples, sampleset.truncate(3))

    def test_top_n(self):
        energies = list(range(10))
        sampleset = dimod.SampleSet.from_samples(np.ones((10, 1)), dimod.SPIN, energy=energies)
        state = State(samples=sampleset)

        top = SliceSamples(-3, None).run(state).result()
        self.assertTrue((top.samples.record.energy == energies[-3:]).all())

        top = SliceSamples().run(state, start=-3).result()
        self.assertTrue((top.samples.record.energy == energies[-3:]).all())

    def test_middle_n(self):
        energies = list(range(10))
        sampleset = dimod.SampleSet.from_samples(np.ones((10, 1)), dimod.SPIN, energy=energies)
        state = State(samples=sampleset)

        mid = SliceSamples(3, -3).run(state).result()
        self.assertTrue((mid.samples.record.energy == energies[3:-3]).all())

        mid = SliceSamples(1, -1).run(state, start=3, stop=-3).result()
        self.assertTrue((mid.samples.record.energy == energies[3:-3]).all())


class TestAggregatedSamples(unittest.TestCase):

    def test_aggregation(self):
        energies = list(range(10))
        sampleset = dimod.SampleSet.from_samples(np.ones((10, 1)), dimod.SPIN, energy=energies)
        state = State(samples=sampleset)

        result = AggregatedSamples(aggregate=True).run(state).result()

        self.assertEqual(len(result.samples), 1)
        self.assertEqual(result.samples.record.sample, np.array([1]))

    def test_spread(self):
        energies = [1, 2]
        occurrences = [3, 2]
        sampleset = dimod.SampleSet.from_samples(
            [{'a': 1}, {'a': 2}], dimod.SPIN,
            energy=energies, num_occurrences=occurrences)
        state = State(samples=sampleset)

        result = AggregatedSamples(aggregate=False).run(state).result()

        # we'll have n=5 samples
        n = sum(occurrences)
        self.assertEqual(len(result.samples), n)

        # samples, energies and num_occurrences must be expanded
        np.testing.assert_array_equal(result.samples.record.sample,
                                      np.array([[1], [1], [1], [2], [2]]))
        np.testing.assert_array_equal(result.samples.record.energy,
                                      np.array([1, 1, 1, 2, 2]))
        np.testing.assert_array_equal(result.samples.record.num_occurrences,
                                      np.ones(n))

        # variables should stay the same
        self.assertEqual(list(sampleset.variables), list(result.samples.variables))


class TestICM(unittest.TestCase):

    @staticmethod
    def total_energy(states):
        """Combined energy of all samples in all states."""
        return sum(float(sum(state.samples.record.energy)) for state in states)

    def test_validation(self):
        bqm1 = dimod.BinaryQuadraticModel({'a': 1}, {}, 0, dimod.SPIN)
        bqm2 = dimod.BinaryQuadraticModel({'b': 1}, {}, 0, dimod.SPIN)
        s1 = State.from_sample({'a': +1}, bqm1)
        s2 = State.from_sample({'b': -1}, bqm2)

        # two input states required
        with self.assertRaises(ValueError):
            inp = States(s1, s1, s1)
            IsoenergeticClusterMove().run(inp).result()

        # variables must match
        with self.assertRaises(ValueError):
            inp = States(s1, s2)
            IsoenergeticClusterMove().run(inp).result()

    def test_triangle_flip(self):
        bqm = dimod.BQM.from_qubo({'ab': 1, 'bc': 1, 'ca': 1})
        s1 = State.from_samples({'a': 0, 'b': 1, 'c': 1}, bqm)
        s2 = State.from_samples({'a': 1, 'b': 0, 'c': 1}, bqm)

        icm = IsoenergeticClusterMove()
        inp = States(s1, s2)
        res = icm.run(inp).result()

        # Expected: ('a', 'b') identified as (the sole) cluster, selected,
        # resulting in variables {'a', 'b'} flipped. Effectively, input states
        # are simply swapped.
        self.assertEqual(res[0].samples, s2.samples)
        self.assertEqual(res[1].samples, s1.samples)

        # verify total samples energy doesn't change after ICM
        self.assertEqual(self.total_energy(inp), self.total_energy(res))

    def test_ising_triangle_flip(self):
        bqm = dimod.BQM.from_ising({}, {'ab': 1, 'bc': 1, 'ca': 1})
        s1 = State.from_samples({'a': -1, 'b': +1, 'c': +1}, bqm)
        s2 = State.from_samples({'a': +1, 'b': -1, 'c': +1}, bqm)

        icm = IsoenergeticClusterMove()
        inp = States(s1, s2)
        res = icm.run(inp).result()

        # Expected: ('a', 'b') identified as (the sole) cluster, selected,
        # resulting in variables {'a', 'b'} flipped. Effectively, input states
        # are simply swapped.
        self.assertEqual(res[0].samples, s2.samples)
        self.assertEqual(res[1].samples, s1.samples)

        # verify total samples energy doesn't change after ICM
        self.assertEqual(self.total_energy(inp), self.total_energy(res))

    def test_small_lattice(self):
        graph = nx.generators.lattice.grid_2d_graph(5, 5)
        bqm = dimod.generators.uniform(graph, vartype=dimod.BINARY, low=1, high=1)
        nodes = sorted(bqm.variables)

        s1 = State.from_samples(dict(zip(nodes, [0, 1, 0, 1, 1,
                                                 0, 0, 0, 1, 0,
                                                 0, 0, 0, 0, 1,
                                                 0, 0, 1, 1, 0,
                                                 1, 0, 1, 0, 0])), bqm)
        s2 = State.from_samples(dict(zip(nodes, [0, 1, 1, 0, 0,
                                                 0, 0, 0, 1, 1,
                                                 0, 1, 1, 0, 1,
                                                 0, 1, 0, 0, 0,
                                                 1, 0, 0, 1, 0])), bqm)

        exp1 = SampleSet.from_samples_bqm(dict(zip(nodes, [0, 1, 0, 1, 1,
                                                           0, 0, 0, 1, 0,
                                                           0, 1, 1, 0, 1,
                                                           0, 1, 0, 0, 0,
                                                           1, 0, 0, 1, 0])), bqm)
        exp2 = SampleSet.from_samples_bqm(dict(zip(nodes, [0, 1, 1, 0, 0,
                                                           0, 0, 0, 1, 1,
                                                           0, 0, 0, 0, 1,
                                                           0, 0, 1, 1, 0,
                                                           1, 0, 1, 0, 0])), bqm)

        icm = IsoenergeticClusterMove(seed=1234)
        inp = States(s1, s2)
        res = icm.run(inp).result()

        self.assertEqual(res[0].samples, exp1)
        self.assertEqual(res[1].samples, exp2)

        # verify total samples energy doesn't change after ICM
        self.assertEqual(self.total_energy(inp), self.total_energy(res))

    def test_bimodal_cluster_sampling_statistics(self):
        bqm = dimod.BQM.from_qubo({'ab': 1, 'bd': 1, 'dc': 1, 'ca': 1})
        nodes = sorted(bqm.variables)

        s1 = State.from_samples(dict(zip(nodes, [0, 1,
                                                 0, 0])), bqm)
        s2 = State.from_samples(dict(zip(nodes, [0, 0,
                                                 1, 0])), bqm)

        exp1 = SampleSet.from_samples_bqm(dict(zip(nodes, [0, 0,
                                                           0, 0])), bqm)
        exp2 = SampleSet.from_samples_bqm(dict(zip(nodes, [0, 1,
                                                           1, 0])), bqm)

        icm = IsoenergeticClusterMove(seed=None)
        inp = States(s1, s2)
        exp = [exp1, exp2]

        # split between [exp1, exp2] and [exp2, exp1] as output samples
        # should be ~50%
        cnt = 0
        n = 100
        for _ in range(n):
            res = icm.run(inp).result()
            r1, r2 = pluck(res, 'samples')

            # test responses are valid
            self.assertIn(r1, exp)
            self.assertIn(r2, exp)

            # verify total samples energy doesn't change after ICM
            self.assertEqual(self.total_energy(inp), self.total_energy(res))

            # count responses
            if r1 == exp1 and r2 == exp2:
                cnt += 1

        self.assertLess(cnt, 0.75 * n)
        self.assertGreater(cnt, 0.25 * n)

    def test_large_sparse(self):
        "Total energy is preserved after ICM on random samples over random graph."

        # random Erdős-Rényi sparse graph with 100 nodes and 10% density
        graph = nx.generators.fast_gnp_random_graph(n=100, p=0.1)
        bqm = dimod.generators.uniform(graph=graph, vartype=dimod.SPIN)
        nodes = sorted(bqm.variables)

        # random input samples
        s1 = State.from_problem(bqm, samples=random_sample)
        s2 = State.from_problem(bqm, samples=random_sample)
        inp = States(s1, s2)

        icm = IsoenergeticClusterMove()
        res = icm.run(inp).result()

        self.assertAlmostEqual(self.total_energy(inp), self.total_energy(res))
