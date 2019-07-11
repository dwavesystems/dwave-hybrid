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

import string
import itertools
import unittest
import collections
from functools import partial

import dimod
import networkx as nx

from hybrid.decomposers import (
    EnergyImpactDecomposer, RandomSubproblemDecomposer,
    RandomConstraintDecomposer, RoofDualityDecomposer)
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

    def test_energy_traverse(self):
        eid = EnergyImpactDecomposer(size=1, traversal='energy')
        priority = collections.OrderedDict((v,v) for v in range(1,4))
        var = eid.traverse(bqm=None, sample=None, ordered_priority=priority, visited=[2,1], size=2)
        self.assertEqual(var, [3])

    def test_bfs_traverse_connected(self):
        eid = EnergyImpactDecomposer(size=None, traversal='bfs')
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'cd': 1, 'da': 1}, 0.0, dimod.SPIN)
        priority = collections.OrderedDict(zip('bdac', itertools.count(0, -1)))
        var = eid.traverse(bqm=bqm, sample=None, ordered_priority=priority, visited=set(), size=3)
        # start from 'b', pick 2 more neighbors
        self.assertEqual(var, set('abc'))

    def test_bfs_traverse_connected_too_small(self):
        eid = EnergyImpactDecomposer(size=None, traversal='bfs')
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'cd': 1, 'da': 1}, 0.0, dimod.SPIN)
        priority = collections.OrderedDict(zip('bdac', itertools.count(0, -1)))
        var = eid.traverse(bqm=bqm, sample=None, ordered_priority=priority, visited=set(), size=5)
        # start from 'b', try to pick more then there is (get complete graph back)
        self.assertEqual(var, set('abcd'))

    def test_bfs_traverse_connected_some_visited(self):
        eid = EnergyImpactDecomposer(size=None, traversal='bfs')
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'cd': 1, 'da': 1, 'bd': 1}, 0.0, dimod.SPIN)
        priority = collections.OrderedDict(zip('bdac', itertools.count(0, -1)))
        var = eid.traverse(bqm=bqm, sample=None, ordered_priority=priority, visited=set('b'), size=3)
        # start with 'b' visited, so use 'd' as root and pick 2 more neighbors
        self.assertEqual(var, set('dac'))

    def test_bfs_traverse_disconnected_some_visited(self):
        eid = EnergyImpactDecomposer(size=None, traversal='bfs')
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'cd': 1, 'da': 1, 'ef': 1, 'fg': 1}, 0.0, dimod.SPIN)
        priority = collections.OrderedDict(zip('abcdefg', itertools.count(0, -1)))
        var = eid.traverse(bqm=bqm, sample=None, ordered_priority=priority, visited=set('abc'), size=3)
        # pick 'd' from first graph component, and 'ef' from the second
        self.assertEqual(var, set('def'))

    def test_bfs_on_sequential_eid_calls_over_disconnected_problem_graph(self):
        # problem graph has two components, each one is 4-node cycle graph
        edges = {'ab': 1, 'bc': 1, 'cd': 1, 'da': 1,
                 'ef': 1, 'fg': 1, 'gh': 1, 'he': 1}
        biases = dict(zip(string.ascii_letters, range(8, 0, -1)))
        bqm = dimod.BinaryQuadraticModel(biases, edges, 0.0, 'SPIN')
        sample = {i: -1 for i in bqm.variables}

        state = State.from_sample(sample, bqm)
        eid = EnergyImpactDecomposer(size=3, traversal='bfs', rolling=True,
                                     rolling_history=1.0, silent_rewind=False)
        states = list(iter(partial(eid.next, state=state), None))

        # energy impact list is: [a..h], so of the 3 subproblems generated,
        # the middle one is disconnected with one var from first group and two
        # variables from the second

        # `c` has higher energy, but it's not connected to `a`, so `d` is picked
        self.assertEqual(set(states[0].subproblem.variables), set('abd'))

        # `c` is picked from the first component, and the seed for the next
        # subproblem is `e`. however, the order of `e`'s neighbors is not defined,
        # so if we need to pick just one, it could be `f` or `h`
        # (note: PFS has a defined order of neighbors)
        self.assertTrue(set('cefh').difference(states[1].subproblem.variables).issubset('fh'))
        self.assertEqual(len(states[1].subproblem.variables), 3)

        # the second component is exhausted in search for 3 variable subproblem
        third = set(states[2].subproblem.variables)
        self.assertTrue(third == set('gh') or third == set('gf'))

    def test_nx_pfs_edgecases(self):
        pfs = EnergyImpactDecomposer._pfs_nodes
        graph = nx.Graph({'a': 'b', 'b': 'c', 'c': 'a'})
        priority = dict(zip('abc', itertools.count())).get

        self.assertEqual(set(pfs(graph, 'a', 0, priority)), set())
        self.assertEqual(set(pfs(graph, 'a', 1, priority)), set('a'))
        self.assertEqual(set(pfs(graph, 'a', 3, priority)), set('abc'))
        self.assertEqual(set(pfs(graph, 'a', 4, priority)), set('abc'))

    def test_nx_pfs_priority_respected(self):
        pfs = EnergyImpactDecomposer._pfs_nodes
        graph = nx.Graph({'a': 'b', 'b': 'c', 'c': 'a'})

        priority = dict(zip('abc', [1, 2, 3])).get
        self.assertEqual(set(pfs(graph, 'a', 2, priority)), set('ac'))
        self.assertEqual(set(pfs(graph, 'b', 2, priority)), set('bc'))
        self.assertEqual(set(pfs(graph, 'c', 2, priority)), set('cb'))

    def test_nx_pfs_deep_search(self):
        pfs = EnergyImpactDecomposer._pfs_nodes

        # two K4 graphs connected with one edge
        graph = nx.complete_graph(4)
        graph.add_edges_from(nx.complete_graph(range(4,8)).edges)
        graph.add_edge(1, 4)

        # use node index for weight/priority
        priority = lambda node: node

        # make sure once the second cluster is within reach, we deplete it
        self.assertEqual(set(pfs(graph, 3, 2, priority)), set([2, 3]))
        self.assertEqual(set(pfs(graph, 3, 3, priority)), set([1, 2, 3]))
        self.assertEqual(set(pfs(graph, 3, 4, priority)), set([1, 2, 3, 4]))
        self.assertEqual(set(pfs(graph, 3, 7, priority)), set([1, 2, 3, 4, 5, 6, 7]))
        self.assertEqual(set(pfs(graph, 3, 8, priority)), set([0, 1, 2, 3, 4, 5, 6, 7]))

    def test_pfs_traverse_connected(self):
        eid = EnergyImpactDecomposer(size=None, traversal='pfs')
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'cd': 1, 'da': 1}, 0.0, dimod.SPIN)
        priority = collections.OrderedDict(zip('bdac', itertools.count(0, -1)))
        var = eid.traverse(bqm=bqm, sample=None, ordered_priority=priority, visited=set(), size=3)
        # start from 'b', pick 2 connected variables, traversed in order of best energy (priority)
        self.assertEqual(var, set('bad'))

    def test_pfs_traverse_connected_too_small(self):
        eid = EnergyImpactDecomposer(size=None, traversal='pfs')
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'cd': 1, 'da': 1}, 0.0, dimod.SPIN)
        priority = collections.OrderedDict(zip('bdac', itertools.count(0, -1)))
        var = eid.traverse(bqm=bqm, sample=None, ordered_priority=priority, visited=set(), size=5)
        # start from 'b', try to pick more then there is (get complete graph back)
        self.assertEqual(var, set('abcd'))

    def test_pfs_traverse_connected_some_visited(self):
        eid = EnergyImpactDecomposer(size=None, traversal='pfs')
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'cd': 1, 'da': 1, 'bd': 1}, 0.0, dimod.SPIN)
        priority = collections.OrderedDict(zip('bdac', itertools.count(0, -1)))
        var = eid.traverse(bqm=bqm, sample=None, ordered_priority=priority, visited=set('b'), size=3)
        # start with 'b' visited, so use 'd' as root and pick 2 more neighbors
        self.assertEqual(var, set('dac'))

    def test_pfs_traverse_disconnected_some_visited(self):
        eid = EnergyImpactDecomposer(size=None, traversal='pfs')
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'cd': 1, 'da': 1, 'ef': 1, 'fg': 1}, 0.0, dimod.SPIN)
        priority = collections.OrderedDict(zip('abcdefg', itertools.count(0, -1)))
        var = eid.traverse(bqm=bqm, sample=None, ordered_priority=priority, visited=set('abc'), size=3)
        # pick 'd' from first graph component, and 'ef' from the second
        self.assertEqual(var, set('def'))

    def test_pfs_on_sequential_eid_calls_over_disconnected_problem_graph(self):
        # problem graph has two components, each one is 4-node cycle graph
        edges = {'ab': 1, 'bc': 1, 'cd': 1, 'da': 1,
                 'ef': 1, 'fg': 1, 'gh': 1, 'he': 1}
        biases = dict(zip(string.ascii_letters, range(8, 0, -1)))
        bqm = dimod.BinaryQuadraticModel(biases, edges, 0.0, 'SPIN')
        sample = {i: -1 for i in bqm.variables}

        state = State.from_sample(sample, bqm)
        eid = EnergyImpactDecomposer(size=3, traversal='pfs', rolling=True,
                                     rolling_history=1.0, silent_rewind=False)
        states = list(iter(partial(eid.next, state=state), None))

        # energy impact list is: [a..h], so of the 3 subproblems generated,
        # the middle one is disconnected with one var from first group and two
        # variables from the second

        # pfs is seeded with `a` and connected nodes in order of energy are picked
        self.assertEqual(set(states[0].subproblem.variables), set('abc'))

        # `d` left from the first component, and the seed for the next
        # subproblem is the next highest in energy `e`.
        # unlike in bfs, the order of `e`'s neighbors is well defined
        self.assertEqual(set(states[1].subproblem.variables), set('def'))

        # the second component is exhausted in search for 3 variable subproblem
        self.assertEqual(set(states[2].subproblem.variables), set('gh'))

    def test_pfs_on_impactful_far_subproblem(self):
        # problem graph has two components, each one is 4-node cycle graph
        # variable flip energy gains order variables: a, h..b
        edges = {'ab': 1, 'bc': 1, 'cd': 1, 'da': 1,
                 'ef': 1, 'fg': 1, 'gh': 1, 'he': 1,
                 'de': 0}
        biases = dict(zip(string.ascii_letters, range(8)))
        biases['a'] += 10
        bqm = dimod.BinaryQuadraticModel(biases, edges, 0.0, 'SPIN')
        sample = {i: -1 for i in bqm.variables}

        state = State.from_sample(sample, bqm)
        eid = EnergyImpactDecomposer(size=5, traversal='pfs')
        result = eid.run(state).result()

        # move towards second cluster and pick the highest energy variables from there
        self.assertEqual(set(result.subproblem.variables), set('adehg'))


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

    def test_next_on_different_sized_constraints(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        variables = list('abcdefg')
        fixed_variables = list('abc')
        size = 3
        constraints = []

        # Set BQM and constraints of varying lengths
        for triplet in itertools.combinations(variables, size):
            for u, v in itertools.combinations(triplet, 2):
                bqm.add_interaction(u, v, -1)
            non_fixed_variables = set(triplet) - set(fixed_variables)
            constraints.append(non_fixed_variables)

        for fixed_variable in fixed_variables:
            bqm.fix_variable(fixed_variable, 1)

        # Get new state
        rcd = RandomConstraintDecomposer(size, constraints)
        state = State.from_sample(min_sample(bqm), bqm)
        newstate = rcd.run(state).result()

        self.assertIn('subproblem', newstate)
        self.assertTrue(len(newstate.subproblem) <= size)  # correct size

    def test_partially_disconnected_constraints(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': 1, 'bc': 1, 'ca': 1})
        size = 2
        constraints = ['a', 'b', 'cb']

        rcd = RandomConstraintDecomposer(size, constraints)
        state = State.from_problem(bqm)
        newstate = rcd.run(state).result()

        self.assertIn('subproblem', newstate)
        self.assertTrue(len(newstate.subproblem) <= size)  # correct size

    def test_completely_disconnected_constraints(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': 1, 'bc': 1, 'ca': 1})
        size = 2
        constraints = ['a', 'b']

        with self.assertRaises(ValueError):
            rcd = RandomConstraintDecomposer(size, constraints)
            state = State.from_problem(bqm)
            _ = rcd.run(state).result()


class TestRoofDualityDecomposer(unittest.TestCase):
    def test_allfixed(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': -1}, {})

        init = State.from_sample({'a': -1}, bqm)  # should be flipped
        new = RoofDualityDecomposer().run(init).result()

        self.assertEqual(new.problem, bqm)
        self.assertEqual(new.subproblem,
                         dimod.BinaryQuadraticModel.from_ising({}, {}, -1))
        self.assertEqual(new.samples.record.sample, [1])

    def test_sampling_mode(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 0}, {})

        init = State.from_samples([{'a': -1}, {'a': 1}], bqm)

        # variable not fixed
        new = RoofDualityDecomposer(sampling_mode=True).run(init).result()
        self.assertEqual(new.problem, bqm)
        self.assertEqual(new.subproblem, bqm)
        self.assertEqual(new.samples, init.samples)

        # variable fixed
        new = RoofDualityDecomposer(sampling_mode=False).run(init).result()
        self.assertEqual(new.problem, bqm)
        self.assertEqual(new.subproblem,
                         dimod.BinaryQuadraticModel.from_ising({}, {}))
        self.assertTrue(len(set(new.samples.record.sample.flatten())), 1)

    def test_energy_correctness(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': -1}, {})

        init = State.from_sample({'a': -1}, bqm)  # should be flipped
        new = RoofDualityDecomposer().run(init).result()

        self.assertEqual(new.samples.record.energy,
                         bqm.energies((new.samples.record.sample,
                                       new.samples.variables)))
