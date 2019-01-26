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
from functools import partial

import dimod

from hybrid.decomposers import (
    EnergyImpactDecomposer, RandomSubproblemDecomposer, RandomConstraintDecomposer)
from hybrid.core import State
from hybrid.utils import min_sample


class TestEnergyImpactDecomposer(unittest.TestCase):
    # minimized when not all vars are equal
    notall = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'ca': 1}, 0, dimod.SPIN)

    def test_one_var(self):
        """First-variable selection works."""

        state = State.from_sample({'a': 1, 'b': 1, 'c': -1}, self.notall)
        eid = EnergyImpactDecomposer(size=1, min_gain=0)
        nextstate = eid.next(state)
        self.assertDictEqual(dict(nextstate.subproblem.linear), {'c': 2})
        self.assertDictEqual(dict(nextstate.subproblem.quadratic), {})

    def test_multi_vars(self):
        """Multiple variables subproblem selection works, without gain limit."""

        state = State.from_sample({'a': 1, 'b': 1, 'c': -1}, self.notall)
        eid = EnergyImpactDecomposer(size=3, min_gain=None)
        nextstate = eid.next(state)
        self.assertDictEqual(dict(nextstate.subproblem.adj), dict(self.notall.adj))

    def test_adaptive_vars(self):
        """Multiple variables subproblem selection works, with gain limit."""

        state = State.from_sample({'a': 1, 'b': 1, 'c': -1}, self.notall)
        eid = EnergyImpactDecomposer(size=3, min_gain=2.0)
        nextstate = eid.next(state)
        self.assertDictEqual(dict(nextstate.subproblem.linear), {'c': 2})
        self.assertDictEqual(dict(nextstate.subproblem.quadratic), {})

    def test_no_vars(self):
        """Failure due to no sub vars available."""

        state = State.from_sample({'a': 1, 'b': 1, 'c': -1}, self.notall)
        eid = EnergyImpactDecomposer(size=3, min_gain=5.0)
        nextstate = eid.next(state)
        self.assertEqual(len(nextstate.subproblem), 0)

    def test_rolling_subproblem(self):
        """Selected number of (non-overlapping) subproblems are unrolled from the input problem."""

        # 10 variables, 0 to 9 when ordered by energy increase on flip
        bqm = dimod.BinaryQuadraticModel({i: i for i in range(10)}, {}, 0.0, 'SPIN')
        sample = {i: 1 for i in range(10)}

        # exactly 5 single-variable problems should be produced
        state = State.from_sample(sample, bqm)
        eid = EnergyImpactDecomposer(size=1, rolling=True, rolling_history=0.5, silent_rewind=False)
        states = list(iter(partial(eid.next, state=state), None))

        self.assertEqual(len(states), 5)
        for idx, state in enumerate(states):
            self.assertEqual(state.subproblem.linear[idx], idx)

    def test_rolling_subproblem_larger_than_rolling_history(self):
        """In case rolling history too small, just one problem is unrolled."""

        # 10 variables, 0 to 9 when ordered by energy increase on flip
        bqm = dimod.BinaryQuadraticModel({i: i for i in range(10)}, {}, 0.0, 'SPIN')
        sample = {i: 1 for i in range(10)}

        # exactly 1 five-variable problems should be produced
        state = State.from_sample(sample, bqm)
        eid = EnergyImpactDecomposer(size=5, rolling=True, rolling_history=0.3, silent_rewind=False)
        states = list(iter(partial(eid.next, state=state), None))

        self.assertEqual(len(states), 1)
        self.assertEqual(len(states[0].subproblem), 5)
        self.assertEqual(list(dict(states[0].subproblem.linear).values()), list(range(0,5)))

        # works even for subproblems as large as the input problem
        eid = EnergyImpactDecomposer(size=len(bqm), rolling=True, rolling_history=0.3, silent_rewind=False)
        states = list(iter(partial(eid.next, state=state), None))

        self.assertEqual(len(states), 1)
        self.assertEqual(len(states[0].subproblem), 10)
        self.assertEqual(list(dict(states[0].subproblem.linear).values()), list(range(0,10)))

        # but adapt to problem size if subproblem larger than problem
        eid = EnergyImpactDecomposer(size=11, rolling=True, rolling_history=0.3, silent_rewind=False)
        states = list(iter(partial(eid.next, state=state), None))

        self.assertEqual(len(states), 1)
        self.assertEqual(len(states[0].subproblem), 10)
        self.assertEqual(list(dict(states[0].subproblem.linear).values()), list(range(0,10)))


class TestRandomSubproblemDecomposer(unittest.TestCase):
    bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1}, 0, dimod.SPIN)
    state = State.from_sample(min_sample(bqm), bqm)

    def test_simple(self):
        runnable = RandomSubproblemDecomposer(size=1)
        state = self.state
        for _ in range(10):
            state = runnable.next(state)
            self.assertEqual(len(state.subproblem.variables), 1)
            self.assertIn(next(iter(state.subproblem.variables)), self.bqm.variables)

    def test_look_and_feel(self):
        self.assertEqual(repr(RandomSubproblemDecomposer(7)), 'RandomSubproblemDecomposer(size=7)')

    def test_validation(self):
        with self.assertRaises(ValueError):
            RandomSubproblemDecomposer(len(self.bqm)+1).run(self.state).result()


class TestConstraintDecomposer(unittest.TestCase):
    def test_typical_construction(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        variables = list('abcdefg')
        constraints = []
        for triplet in itertools.combinations(variables, 3):
            for u, v in itertools.combinations(triplet, 2):
                bqm.add_interaction(u, v, -1)
            constraints.append(triplet)

        rcd = RandomConstraintDecomposer(3, constraints)
        rcd.init(state=State.from_sample(min_sample(bqm), bqm))

        # check that the graph is complete
        G = rcd.constraint_graph
        for i in range(len(constraints)):
            self.assertIn(i, G.nodes)

    def test_next(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        variables = list('abcdefg')
        constraints = []
        for triplet in itertools.combinations(variables, 3):
            for u, v in itertools.combinations(triplet, 2):
                bqm.add_interaction(u, v, -1)
            constraints.append(triplet)

        rcd = RandomConstraintDecomposer(3, constraints)

        state = State.from_sample(min_sample(bqm), bqm)

        newstate = rcd.run(state).result()

        self.assertIn('subproblem', newstate)
        self.assertTrue(len(newstate.subproblem) <= 3)  # correct size
