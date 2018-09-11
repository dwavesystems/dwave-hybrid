import itertools
import unittest

import dimod

from hades.decomposers import EnergyImpactDecomposer, RandomConstraintDecomposer
from hades.core import State
from hades.utils import min_sample


class TestEnergyImpactDecomposer(unittest.TestCase):
    # minimized when not all vars are equal
    notall = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'ca': 1}, 0, dimod.SPIN)

    def test_one_var(self):
        """First-variable selection works."""

        state = State.from_sample({'a': 1, 'b': 1, 'c': -1}, self.notall)
        eid = EnergyImpactDecomposer(self.notall, max_size=1, min_gain=0)
        nextstate = eid.iterate(state)
        self.assertDictEqual(nextstate.subproblem.linear, {'c': 2})
        self.assertDictEqual(nextstate.subproblem.quadratic, {})

    def test_multi_vars(self):
        """Multiple variables subproblem selection works, without gain limit."""

        state = State.from_sample({'a': 1, 'b': 1, 'c': -1}, self.notall)
        eid = EnergyImpactDecomposer(self.notall, max_size=3, min_gain=None)
        nextstate = eid.iterate(state)
        self.assertDictEqual(nextstate.subproblem.adj, self.notall.adj)

    def test_adaptive_vars(self):
        """Multiple variables subproblem selection works, with gain limit."""

        state = State.from_sample({'a': 1, 'b': 1, 'c': -1}, self.notall)
        eid = EnergyImpactDecomposer(self.notall, max_size=3, min_gain=2.0)
        nextstate = eid.iterate(state)
        self.assertDictEqual(nextstate.subproblem.linear, {'c': 2})
        self.assertDictEqual(nextstate.subproblem.quadratic, {})

    def test_no_vars(self):
        """Failure due to no sub vars available."""

        state = State.from_sample({'a': 1, 'b': 1, 'c': -1}, self.notall)
        eid = EnergyImpactDecomposer(self.notall, max_size=3, min_gain=5.0)
        with self.assertRaises(ValueError):
            nextstate = eid.iterate(state)


class TestConstraintDecomposer(unittest.TestCase):
    def test_typical_construction(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        variables = list('abcdefg')
        constraints = []
        for triplet in itertools.combinations(variables, 3):
            for u, v in itertools.combinations(triplet, 2):
                bqm.add_interaction(u, v, -1)
            constraints.append(triplet)

        rcd = RandomConstraintDecomposer(bqm, 3, constraints)

        # check that the graph is complete
        G = rcd.constraint_graph
        for i in range(len(constraints)):
            self.assertIn(i, G.nodes)

    def test_iterate(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        variables = list('abcdefg')
        constraints = []
        for triplet in itertools.combinations(variables, 3):
            for u, v in itertools.combinations(triplet, 2):
                bqm.add_interaction(u, v, -1)
            constraints.append(triplet)

        rcd = RandomConstraintDecomposer(bqm, 3, constraints)

        state = State.from_sample(min_sample(bqm), bqm)

        newstate = rcd.iterate(state)

        self.assertIn('subproblem', newstate)
        self.assertTrue(len(newstate.subproblem) <= 3)  # correct size
