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
