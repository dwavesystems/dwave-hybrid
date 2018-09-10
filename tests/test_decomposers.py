import itertools
import unittest
import math

import dimod

from hades.decomposers import RandomConstraintDecomposer
from hades.core import State
from hades.utils import min_sample


class TestConstraintDecompser(unittest.TestCase):
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

        f = rcd.iterate(state)

        self.assertIn('subproblem', f.ctx)
        self.assertTrue(len(f.ctx['subproblem']) <= 3)  # correct size
