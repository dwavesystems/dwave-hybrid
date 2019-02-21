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

import dimod

from hybrid.core import State, States, SampleSet
from hybrid import traits
from hybrid.composers import IdentityComposer, SplatComposer, GreedyPathMerge
from hybrid.utils import min_sample, max_sample


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
        self.assertTrue(
            # problem and samples are included by default
            IdentityComposer().run(State(subsamples=True)).result())


class TestSplatComposer(unittest.TestCase):
    problem = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'ca': 1}, 0, dimod.SPIN)
    samples = [{'a': 1, 'b': 1, 'c': 1}]
    subproblem = dimod.BinaryQuadraticModel({}, {'bc': 1}, 0, dimod.SPIN)
    subsamples = [{'b': -1, 'c': 1}]

    def test_default(self):
        """First subsample is combined with the first sample."""

        state = State.from_samples(self.samples, self.problem).updated(
            subproblem=self.subproblem,
            subsamples=SampleSet.from_samples_bqm(self.subsamples, self.subproblem))

        nextstate = SplatComposer().next(state)

        sample = {'a': 1, 'b': -1, 'c': 1}
        self.assertEqual(nextstate.samples,
                         SampleSet.from_samples_bqm(sample, self.problem))

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
