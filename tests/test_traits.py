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

from hades.core import State, Runnable
from hades.utils import min_sample
from hades import traits


class TestRunnableTraits(unittest.TestCase):

    def test_valid_input(self):
        class Component(Runnable, traits.SubproblemActing):
            def iterate(self, state):
                return True

        self.assertTrue(Component().run(State(subproblem=None)).result())

    def test_invalid_input(self):
        class Component(Runnable, traits.SubproblemActing):
            def iterate(self, state):
                return True

        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State()).result()

    def test_valid_output(self):
        class Component(Runnable, traits.SubproblemProducing):
            def iterate(self, state):
                return state.updated(subproblem=True)

        self.assertTrue(Component().run(State()).result().subproblem)

    def test_invalid_output(self):
        class Component(Runnable, traits.SubproblemProducing):
            def iterate(self, state):
                return state

        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State()).result()


class TestMultipleTraits(unittest.TestCase):

    def test_explicit_siso_system(self):
        class Component(Runnable, traits.SubproblemActing, traits.SubsamplesProducing):
            def iterate(self, state):
                return state.updated(subsamples=True)

        self.assertTrue(Component().run(State(subproblem=True)).result().subsamples)

        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State()).result()

    def test_explicit_mimo_system(self):
        class Component(Runnable, traits.EmbeddingActing, traits.SubproblemActing,
                                  traits.SubsamplesProducing, traits.SamplesProducing):
            def iterate(self, state):
                return state.updated(samples=True, subsamples=True)

        self.assertTrue(Component().run(State(embedding=True, subproblem=True)).result().samples)
        self.assertTrue(Component().run(State(embedding=True, subproblem=True)).result().subsamples)

        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State(embedding=True)).result()
        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State(subproblem=True)).result()

    def test_composed_traits(self):
        class Component(Runnable, traits.ProblemDecomposer):
            def iterate(self, state):
                return state.updated(subproblem=True)

        self.assertTrue(Component().run(State(problem=True)).result().subproblem)

        with self.assertRaises(traits.StateTraitMissingError):
            s = State()
            del s['problem']
            Component().run(s).result()

    def test_problem_decomposer_traits(self):
        # ProblemDecomposer ~ ProblemActing, SamplesActing, SubproblemProducing

        class Component(Runnable, traits.ProblemDecomposer):
            def iterate(self, state):
                return state

        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State()).result()
        self.assertTrue(
            # problem and samples are included by default
            Component().run(State(subproblem=True)).result())

    def test_subproblem_composer_traits(self):
        # SubproblemComposer ~ SubproblemActing, SubsamplesActing, ProblemActing, SamplesProducing

        class Component(Runnable, traits.SubproblemComposer):
            def iterate(self, state):
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
