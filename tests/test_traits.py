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

from hybrid.core import State, Runnable
from hybrid.utils import min_sample
from hybrid import traits


class TestRunnableTraits(unittest.TestCase):

    def test_valid_input(self):
        class Component(Runnable, traits.SubproblemIntaking):
            def next(self, state):
                return True

        self.assertTrue(Component().run(State(subproblem=None)).result())

    def test_invalid_input(self):
        class Component(Runnable, traits.SubproblemIntaking):
            def next(self, state):
                return True

        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State()).result()

    def test_valid_output(self):
        class Component(Runnable, traits.SubproblemProducing):
            def next(self, state):
                return state.updated(subproblem=True)

        self.assertTrue(Component().run(State()).result().subproblem)

    def test_invalid_output(self):
        class Component(Runnable, traits.SubproblemProducing):
            def next(self, state):
                return state

        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State()).result()


class TestMultipleTraits(unittest.TestCase):

    def test_explicit_siso_system(self):
        class Component(Runnable, traits.SubproblemIntaking, traits.SubsamplesProducing):
            def next(self, state):
                return state.updated(subsamples=True)

        self.assertTrue(Component().run(State(subproblem=True)).result().subsamples)

        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State()).result()

    def test_explicit_mimo_system(self):
        class Component(Runnable, traits.EmbeddingIntaking, traits.SubproblemIntaking,
                                  traits.SubsamplesProducing, traits.SamplesProducing):
            def next(self, state):
                return state.updated(samples=True, subsamples=True)

        self.assertTrue(Component().run(State(embedding=True, subproblem=True)).result().samples)
        self.assertTrue(Component().run(State(embedding=True, subproblem=True)).result().subsamples)

        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State(embedding=True)).result()
        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State(subproblem=True)).result()

    def test_atypical_output_only_system(self):
        class Component(Runnable, traits.ProblemProducing, traits.EmbeddingProducing):
            def next(self, state):
                return state.updated(problem=True, embedding=True)

        self.assertTrue(Component().run(State()).result().problem)
        self.assertTrue(Component().run(State()).result().embedding)

    def test_composed_traits(self):
        class Component(Runnable, traits.ProblemDecomposer):
            def next(self, state):
                return state.updated(subproblem=True)

        self.assertTrue(Component().run(State(problem=True)).result().subproblem)

        with self.assertRaises(traits.StateTraitMissingError):
            s = State()
            del s['problem']
            Component().run(s).result()

    def test_problem_decomposer_traits(self):
        # ProblemDecomposer ~ ProblemIntaking, SamplesIntaking, SubproblemProducing

        class Component(Runnable, traits.ProblemDecomposer):
            def next(self, state):
                return state

        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State()).result()
        self.assertTrue(
            # problem and samples are included by default
            Component().run(State(subproblem=True)).result())

    def test_subproblem_composer_traits(self):
        # SubproblemComposer ~ SubproblemIntaking, SubsamplesIntaking, ProblemIntaking, SamplesProducing

        class Component(Runnable, traits.SubproblemComposer):
            def next(self, state):
                return state

        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State()).result()
        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State(subproblem=True)).result()
        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State(subsamples=True)).result()
        self.assertTrue(
            # problem and samples are included by default
            Component().run(State(subproblem=True, subsamples=True)).result())
