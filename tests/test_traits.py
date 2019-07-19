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

from hybrid.core import State, States, Runnable
from hybrid.flow import Branch, RacingBranches, Map, Loop, ArgMin
from hybrid.utils import min_sample
from hybrid import traits


class TestRunnableTraits(unittest.TestCase):

    def test_valid_input(self):
        class Component(traits.SubproblemIntaking, Runnable):
            def next(self, state):
                return State()

        self.assertEqual(Component().run(State(subproblem=None)).result(), State())

    def test_invalid_input(self):
        class Component(traits.SubproblemIntaking, Runnable):
            def next(self, state):
                return True

        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State()).result()

    def test_valid_output(self):
        class Component(traits.SubproblemProducing, Runnable):
            def next(self, state):
                return state.updated(subproblem=True)

        self.assertTrue(Component().run(State()).result().subproblem)

    def test_invalid_output(self):
        class Component(traits.SubproblemProducing, Runnable):
            def next(self, state):
                return state

        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State()).result()


class TestMultipleTraits(unittest.TestCase):

    def test_explicit_siso_system(self):
        class Component(traits.SubproblemIntaking, traits.SubsamplesProducing, Runnable):
            def next(self, state):
                return state.updated(subsamples=True)

        self.assertTrue(Component().run(State(subproblem=True)).result().subsamples)

        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State()).result()

    def test_explicit_mimo_system(self):
        class Component(traits.EmbeddingIntaking, traits.SubproblemIntaking,
                        traits.SubsamplesProducing, traits.SamplesProducing, Runnable):
            def next(self, state):
                return state.updated(samples=True, subsamples=True)

        self.assertTrue(Component().run(State(embedding=True, subproblem=True)).result().samples)
        self.assertTrue(Component().run(State(embedding=True, subproblem=True)).result().subsamples)

        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State(embedding=True)).result()
        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State(subproblem=True)).result()

    def test_atypical_output_only_system(self):
        class Component(traits.ProblemProducing, traits.EmbeddingProducing, Runnable):
            def next(self, state):
                return state.updated(problem=True, embedding=True)

        self.assertTrue(Component().run(State()).result().problem)
        self.assertTrue(Component().run(State()).result().embedding)

    def test_composed_traits_input_validation(self):
        class Component(traits.ProblemDecomposer, Runnable):
            def next(self, state):
                return state.updated(subproblem=True)

        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State()).result()
        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State(problem=True)).result()
        self.assertTrue(
            Component().run(State(problem=True, samples=True)).result().subproblem)

    def test_composed_traits_output_validation(self):
        # ProblemDecomposer ~ ProblemIntaking, SamplesIntaking, SubproblemProducing
        class Component(traits.ProblemDecomposer, Runnable):
            def next(self, state):
                return state

        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State(problem=True, samples=True)).result()
        self.assertTrue(
            Component().run(State(problem=True, samples=True, subproblem=True)).result())

    def test_subsamples_composer_traits(self):
        # SubsamplesComposer ~ SamplesIntaking, SubsamplesIntaking, ProblemIntaking, SamplesProducing
        class Component(traits.SubsamplesComposer, Runnable):
            def next(self, state):
                return state

        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State()).result()
        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State(problem=True)).result()
        with self.assertRaises(traits.StateTraitMissingError):
            Component().run(State(problem=True, samples=True)).result()
        self.assertTrue(
            Component().run(State(problem=True, samples=True, subsamples=True)).result())


class TestMultipleStateTraits(unittest.TestCase):

    def test_siso(self):
        class ValidSISO(traits.SISO, Runnable):
            def next(self, state):
                return state

        with self.assertRaises(traits.StateDimensionalityError):
            ValidSISO().run(States()).result()
        self.assertEqual(ValidSISO().run(State(x=1)).result().x, 1)

        class InvalidSISO(traits.SISO, Runnable):
            def next(self, state):
                # should return a single State()
                return States(state, state)

        with self.assertRaises(traits.StateDimensionalityError):
            InvalidSISO().run(State()).result()
            InvalidSISO().run(States()).result()

    def test_mimo_with_specific_state_traits(self):
        # input: list of states with subproblem
        # output: list of states with subsamples
        class SubproblemSamplerMIMO(traits.SubproblemSampler, traits.MIMO, Runnable):
            def next(self, states):
                return States(State(subsamples=1), State(subsamples=2))

        with self.assertRaises(traits.StateDimensionalityError):
            SubproblemSamplerMIMO().run(State()).result()
        with self.assertRaises(traits.StateTraitMissingError):
            SubproblemSamplerMIMO().run(States(State())).result()

        r = SubproblemSamplerMIMO().run(States(State(subproblem=True))).result()
        self.assertEqual(r[0].subsamples, 1)
        self.assertEqual(r[1].subsamples, 2)

    def test_invalid_mimo(self):
        class InvalidMIMO(traits.MIMO, Runnable):
            def next(self, states):
                # should return States()
                return State()

        with self.assertRaises(traits.StateDimensionalityError):
            InvalidMIMO().run(States()).result()

    def test_all_state_traits_enforced(self):
        class A(traits.SubproblemIntaking, traits.EmbeddingProducing, traits.SIMO, Runnable):
            def next(self, state):
                # FAIL: embedding is missing in second state
                return States(State(embedding=1), State())

        with self.assertRaises(traits.StateTraitMissingError):
            A().run(State()).result()

        with self.assertRaises(traits.StateTraitMissingError):
            A().run(State(subproblem=1)).result()

        class B(traits.SubproblemIntaking, traits.EmbeddingProducing, traits.SIMO, Runnable):
            def next(self, state):
                return States(State(embedding=1), State(embedding=2))

        with self.assertRaises(traits.StateTraitMissingError):
            # subproblem missing on input
            B().run(State()).result()

        self.assertEqual(len(B().run(State(subproblem=1)).result()), 2)


class TestTraitsValidationOnOff(unittest.TestCase):

    def test_validated(self):
        class Validated(traits.SISO, Runnable):
            def next(self, state):
                return state

        with self.assertRaises(traits.StateDimensionalityError):
            Validated().run(States()).result()
        self.assertEqual(Validated().run(State(x=1)).result().x, 1)

    def test_not_validated(self):
        class NotValidated(traits.NotValidated, Runnable):
            def next(self, state):
                return state

        self.assertEqual(NotValidated().run(State(x=1)).result().x, 1)
        self.assertEqual(NotValidated().run(States(State(x=1))).result()[0].x, 1)


class TestFlowComponentsTraits(unittest.TestCase):

    def test_branch_with_single_component(self):
        """Traits requirements from inner runnable must be reflected in branch."""
        class ValidSISO(traits.SISO, Runnable):
            def next(self, state):
                return state

        with self.assertRaises(traits.StateDimensionalityError):
            Branch(components=(ValidSISO(),)).run(States()).result()
        self.assertEqual(Branch(components=(ValidSISO(),)).run(State(x=1)).result().x, 1)

        class InvalidSISO(traits.SISO, Runnable):
            def next(self, state):
                return States(state, state)

        with self.assertRaises(traits.StateDimensionalityError):
            Branch(components=(InvalidSISO(),)).run(State()).result()
            Branch(components=(InvalidSISO(),)).run(States()).result()

        # input: list of states with subproblem
        # output: list of states with subsamples
        class SubproblemSamplerMIMO(traits.SubproblemSampler, traits.MIMO, Runnable):
            def next(self, states):
                return States(State(subsamples=1), State(subsamples=2))

        with self.assertRaises(traits.StateDimensionalityError):
            Branch(components=(SubproblemSamplerMIMO(),)).run(State()).result()
        with self.assertRaises(traits.StateTraitMissingError):
            Branch(components=(SubproblemSamplerMIMO(),)).run(States(State())).result()

        r = Branch(components=(SubproblemSamplerMIMO(),)).run(States(State(subproblem=True))).result()
        self.assertEqual(r[0].subsamples, 1)
        self.assertEqual(r[1].subsamples, 2)

    def test_loop(self):
        class MIMOIdent(traits.MIMO, Runnable):
            def next(self, state):
                return state

        wrk = Loop(MIMOIdent(), max_iter=10, convergence=10, key=lambda _: None)
        inp = States(State(idx=0), State(idx=1))

        out = wrk.run(inp).result()
        self.assertEqual(out[0].idx, 0)
        self.assertEqual(out[1].idx, 1)

        with self.assertRaises(traits.StateDimensionalityError):
            wrk.run(State()).result()
