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

from hybrid.core import State, SampleSet
from hybrid import traits
from hybrid.composers import IdentityComposer, SplatComposer


class TestIdentityComposer(unittest.TestCase):
    problem = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'ca': 1}, 0, dimod.SPIN)
    samples = [{'a': 1, 'b': 1, 'c': -1}]

    def test_default(self):
        """Subsamples are copied to samples."""

        state = State(
            subproblem=None,
            subsamples=SampleSet.from_bqm_samples(self.problem, self.samples))

        nextstate = IdentityComposer().next(state)
        self.assertEqual(state.subsamples, nextstate.samples)

    def test_traits_enforced(self):
        """Sample composers require `problem`, `subproblem` and `subsamples`."""

        with self.assertRaises(traits.StateTraitMissingError):
            IdentityComposer().run(State()).result()
        with self.assertRaises(traits.StateTraitMissingError):
            IdentityComposer().run(State(problem=True)).result()
        with self.assertRaises(traits.StateTraitMissingError):
            IdentityComposer().run(State(subproblem=True)).result()
        with self.assertRaises(traits.StateTraitMissingError):
            IdentityComposer().run(State(subsamples=True)).result()
        self.assertTrue(
            # problem and samples are included by default
            IdentityComposer().run(State(subproblem=True, subsamples=True)).result())


class TestSplatComposer(unittest.TestCase):
    problem = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'ca': 1}, 0, dimod.SPIN)
    samples = [{'a': 1, 'b': 1, 'c': 1}]
    subproblem = dimod.BinaryQuadraticModel({}, {'bc': 1}, 0, dimod.SPIN)
    subsamples = [{'b': -1, 'c': 1}]

    def test_default(self):
        """First subsample is combined with the first sample."""

        state = State.from_samples(self.samples, self.problem).updated(
            subproblem=self.subproblem,
            subsamples=SampleSet.from_bqm_samples(self.subproblem, self.subsamples))

        nextstate = SplatComposer().next(state)

        sample = {'a': 1, 'b': -1, 'c': 1}
        self.assertEqual(nextstate.samples,
                         SampleSet.from_bqm_sample(self.problem, sample))

    def test_traits_enforced(self):
        """Sample composers require `problem`, `subproblem` and `subsamples`."""

        with self.assertRaises(traits.StateTraitMissingError):
            SplatComposer().run(State()).result()
        with self.assertRaises(traits.StateTraitMissingError):
            SplatComposer().run(State(problem=True)).result()
        with self.assertRaises(traits.StateTraitMissingError):
            SplatComposer().run(State(subproblem=True)).result()
        with self.assertRaises(traits.StateTraitMissingError):
            SplatComposer().run(State(subsamples=True)).result()
        self.assertTrue(
            # problem and samples are included by default
            SplatComposer().run(State(
                problem=self.problem, subproblem=self.subproblem,
                samples=SampleSet.from_bqm_samples(self.problem, self.samples),
                subsamples=SampleSet.from_bqm_samples(self.subproblem, self.subsamples))).result())
