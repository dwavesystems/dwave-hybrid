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
import dwave_networkx as dnx

from hybrid.decomposers import (
    EnergyImpactDecomposer, RandomSubproblemDecomposer,
    RandomConstraintDecomposer, RoofDualityDecomposer, ComponentDecomposer,
    SublatticeDecomposer, make_origin_embeddings,
    _make_cubic_lattice, _make_kings_lattice,
    _zephyr_to_chimeralike, _chimeralike_to_zephyr)
from hybrid.core import State
from hybrid.utils import min_sample, random_sample
from hybrid.exceptions import EndOfStream

from dwave.system.testing import MockDWaveSampler
from dwave.embedding import verify_embedding

class TestComponentDecomposer(unittest.TestCase):
    bqm = dimod.BinaryQuadraticModel({'a': 2, 'b': -1, 'd': 1}, {'bc': 1, 'cd': 1}, 0, dimod.SPIN)
    state = State.from_sample(random_sample(bqm), bqm)

    def test_default(self):
        decomposer = ComponentDecomposer()
        
        state1 = decomposer.next(self.state)
        self.assertIn(dict(state1.subproblem.linear), ({'a': 2}, {'b': -1, 'c': 0, 'd': 1}))

        state2 = decomposer.next(state1)
        self.assertIn(dict(state2.subproblem.linear), ({'a': 2}, {'b': -1, 'c': 0, 'd': 1}))
        self.assertNotEqual(dict(state1.subproblem.linear), dict(state2.subproblem.linear))

        state3 = decomposer.next(state2)  # silent_rewind=True, so rewind w/o raising an assertion
        self.assertIn(dict(state3.subproblem.linear), ({'a': 2}, {'b': -1, 'c': 0, 'd': 1}))

    def test_sort(self):
        decomposer = ComponentDecomposer(key=len)
        
        state1 = decomposer.next(self.state)
        self.assertDictEqual(dict(state1.subproblem.linear), {'b': -1, 'c': 0, 'd': 1})

    def test_sort_reverse(self):
        decomposer = ComponentDecomposer(key=len, reverse=False)
        
        state1 = decomposer.next(self.state)
        self.assertDictEqual(dict(state1.subproblem.linear), {'a': 2})

    def test_no_silent_rewind(self):
        decomposer = ComponentDecomposer(silent_rewind=False)
        state1 = decomposer.next(self.state)
        state2 = decomposer.next(state1)

        with self.assertRaises(EndOfStream):
            state3 = decomposer.next(state2)

    def test_key_func(self):
        def sum_linear_biases(component):
            total = 0
            for v in component:
                total += self.bqm.get_linear(v)
            return total
    
        decomposer = ComponentDecomposer(key=sum_linear_biases)

        state1 = decomposer.next(self.state)
        self.assertDictEqual(dict(state1.subproblem.linear), {'a': 2})

    def test_no_rolling(self):
        decomposer = ComponentDecomposer(rolling=False, key=len)

        state1 = decomposer.next(self.state)
        self.assertDictEqual(dict(state1.subproblem.linear), {'b': -1, 'c': 0, 'd': 1})

        state2 = decomposer.next(state1)
        self.assertDictEqual(dict(state2.subproblem.linear), {'b': -1, 'c': 0, 'd': 1})

    def test_one_component(self):
        bqm = dimod.BinaryQuadraticModel({'a': 1, 'b': -1}, {'ab': 1}, 0, dimod.SPIN)
        state = State.from_sample(random_sample(bqm), bqm)
        
        decomposer = ComponentDecomposer()
        state1 = decomposer.next(state)

        self.assertEqual(bqm, state1.subproblem)

    def test_empty(self):
        bqm = dimod.BinaryQuadraticModel(dimod.SPIN)
        state = State.from_sample(random_sample(bqm), bqm)

        decomposer = ComponentDecomposer()
        state1 = decomposer.next(state)

        self.assertEqual(bqm, state1.subproblem)

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


class TestSublatticeDecomposer(unittest.TestCase):
    """State fields bqm and origin_embeddings alongside optional state fields 
    geometric_offset, problem_dims, exclude_dims and origin_embedding_index
    have non-trivial dependencies that are not fully tested. Standard 
    operational modes are tested for square lattice models.
    """

    def test_trivial_subproblem(self):
        """Check single-variable no-coupler subproblem creation.

        A 2by2 square lattice (a square), with a single variable subsolver,
        is the simplest non-trivial test.
        """

        problem_dims = (2, 2)
        # Vertical edges
        edgelist = [((i, j), (i+1, j))
                    for i in range(problem_dims[0]-1)
                    for j in range(problem_dims[1])]
        # Horizontal edges
        edgelist += [((i, j), (i, j+1))
                     for i in range(problem_dims[0])
                     for j in range(problem_dims[1]-1)]
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {edge: 1 for edge in edgelist})
        origin_embeddings = [{(0, 0): [0]}]
        state = State.from_sample(min_sample(bqm), bqm,
                                  origin_embeddings=origin_embeddings,
                                  problem_dims=problem_dims)

        # Creates subproblems located randomly at (0, 0), (0, 1), (1, 0) and
        # (1, 1).
        runnable = SublatticeDecomposer()
        for _ in range(10):
            state = runnable.next(state)
            self.assertEqual(len(state.subproblem.variables), 1)
            self.assertIn(next(iter(state.subproblem.variables)), bqm.variables)
            self.assertEqual(len(state.embedding), 1)
            self.assertIn(next(iter(state.embedding.keys())), bqm.variables)

    def test_nontrivial_subproblem(self):
        """Check multi-variable multi-coupler subproblem creation.

        A 3by3 square lattice, with a 2x2 lattice subsolver, is the simplest 
        non-trivial test with edges in the decomposed problem.
        """
        problem_dims = (3, 3)
        #Vertical edges
        edgelist = [((i, j), (i+1, j))
                    for i in range(problem_dims[0]-1)
                    for j in range(problem_dims[1])]
        #Horizontal edges
        edgelist += [((i, j), (i, j+1))
                     for i in range(problem_dims[0])
                     for j in range(problem_dims[1]-1)]
        bqm = dimod.BinaryQuadraticModel.from_ising(
            {}, {edge: 1 for edge in edgelist})
        origin_embeddings = [{(i, j): None for i in range(2) for j in range(2)}]
        state = State.from_sample(min_sample(bqm), bqm,
                                  origin_embeddings=origin_embeddings,
                                  problem_dims=problem_dims)

        # Creates one of 3x3 different subsubproblems, some of which are
        # disconnected, and some connected 
        # Run multiple times to prevent coincidental agreement
        runnable = SublatticeDecomposer()
        for _ in range(10):
            state = runnable.next(state)
            self.assertEqual(len(state.subproblem.variables), 4)
            self.assertIn(next(iter(state.subproblem.variables)), bqm.variables)
            self.assertEqual(len(state.embedding), 4)
            self.assertIn(next(iter(state.embedding.keys())), bqm.variables)

    def test_geometric_offset(self):
        """Check exclude_dims parameter at maximum origin_embedding_index.
        
        If geometric offset is zero, the ``embedding`` field should match the 
        ``origin_embedding``.
        """
        problem_dims = (3, 3)
        # Vertical edges
        edgelist = [((i, j), (i+1, j))
                    for i in range(problem_dims[0]-1)
                    for j in range(problem_dims[1])]
        # Horizontal edges
        edgelist += [((i, j), (i, j+1))
                     for i in range(problem_dims[0])
                     for j in range(problem_dims[1]-1)]
        bqm = dimod.BinaryQuadraticModel.from_ising(
            {}, {edge: 1 for edge in edgelist})
        origin_embeddings = [{(i, j): None for i in range(2) for j in range(2)}]
        geometric_offset = (0,0)
        oei = len(origin_embeddings) - 1
        state = State.from_sample(min_sample(bqm), bqm,
                                  origin_embeddings=origin_embeddings,
                                  problem_dims=problem_dims,
                                  geometric_offset=geometric_offset,
                                  origin_embedding_index=oei)

        # Creates a 2x2 subproblem at the origin for final embedding
        # Run multiple times to prevent coincidental agreement
        runnable = SublatticeDecomposer()
        for _ in range(10):
            state = runnable.next(state)
            self.assertEqual(len(state.subproblem.variables), 4)
            self.assertIn(next(iter(state.subproblem.variables)), bqm.variables)
            self.assertEqual(len(state.embedding), 4)
            self.assertIn(next(iter(state.embedding.keys())), bqm.variables)
            self.assertIn(next(iter(state.embedding)), state.origin_embeddings[oei])
            
    def test_exclude_dimensions(self):
        """Check exclude_dims parameter at origin_embedding_index=0.

        If all dimensions are excluded, ``embedding`` field should match the 
        ``origin_embedding``.
        """
        problem_dims = (3, 3)
        # Vertical edges
        edgelist = [((i, j), (i+1, j))
                    for i in range(problem_dims[0]-1)
                    for j in range(problem_dims[1])]
        # Horizontal edges
        edgelist += [((i, j), (i, j+1))
                     for i in range(problem_dims[0])
                     for j in range(problem_dims[1]-1)]
        bqm = dimod.BinaryQuadraticModel.from_ising(
            {}, {edge: 1 for edge in edgelist})
        origin_embeddings = [{(i, j): None for i in range(2) for j in range(2)}]
        exclude_dims = list(range(len(problem_dims)))
        oei = 0
        state = State.from_sample(min_sample(bqm), bqm,
                                  origin_embeddings=origin_embeddings,
                                  problem_dims=problem_dims,
                                  exclude_dims=exclude_dims,
                                  origin_embedding_index=oei)

        # Creates a 2x2 subproblem at the origin for 0th embedding
        # Run multiple times to prevent coincidental agreement
        runnable = SublatticeDecomposer()
        for _ in range(10):
            state = runnable.next(state)
            self.assertEqual(len(state.subproblem.variables), 4)
            self.assertIn(next(iter(state.subproblem.variables)), bqm.variables)
            self.assertEqual(len(state.embedding), 4)
            self.assertIn(next(iter(state.embedding.keys())), bqm.variables)
            self.assertIn(next(iter(state.embedding)), state.origin_embeddings[oei])

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

class TestMakeOriginEmbeddings(unittest.TestCase):
    
    def test_default_embeddings(self):
        """Check default workflow, and self-consistency of default dimensions.
        """
        qpu_sampler = MockDWaveSampler()
        orig_embs = make_origin_embeddings(qpu_sampler=qpu_sampler)
        # Non-empty list of non-empty dictionaries
        self.assertTrue(isinstance(orig_embs, list))
        self.assertTrue(len(orig_embs)>0)
        for orig_emb in orig_embs:
            self.assertTrue(isinstance(orig_emb, dict))
            self.assertTrue(len(orig_emb)>0)

        # variable key should be a tuple of consistent length (function of
        # default qpu_solver topology vector coordinate scheme)
        # chain length should be 1 (native default embedding)
        for orig_emb in orig_embs:
            dict_item = next(iter(orig_emb.items()))
            tuple_length = len(dict_item[0])
            chain_length = len(dict_item[1])
        
            for key, val in orig_emb.items():
                self.assertEqual(len(key), tuple_length)
                self.assertEqual(len(val), chain_length)
  
    def test_all_embedding_shapes(self):
        """Check embeddings match anticipated optimal dimensions for Chimera,
        Pegasus and Zephyr processors with Cubic, Kings and Native embeddings.
        Uses a default processor scale of 4, with either 0 or 15
        edge defects.
        """
        # Expected properties by (topology_type, lattice_topology):
        # tuple length, chain length, number of embeddings
        shape_dicts = {('zephyr', 'zephyr'): {'tl': 4, 'cl': 1, 'ne': 2},
                       ('zephyr', 'kings'): {'tl': 2, 'cl': 2, 'ne': 2},
                       ('zephyr', 'cubic'): {'tl': 3, 'cl': 2, 'ne': 3},
                       ('pegasus', 'pegasus'): {'tl': 5, 'cl': 1, 'ne': 2},
                       ('pegasus', 'cubic'): {'tl': 3, 'cl': 2, 'ne': 3},
                       ('pegasus', 'kings'): {'tl': 2, 'cl': 2, 'ne': 2},
                       ('chimera', 'chimera'): {'tl': 4, 'cl': 1, 'ne': 2},
                       ('chimera', 'cubic'): {'tl': 3, 'cl': 4, 'ne': 3}}
        
        for qpu_top in ['pegasus', 'chimera', 'zephyr']:
            if qpu_top == 'chimera':
                lattice_types = ['cubic', qpu_top, None]
            elif qpu_top == 'pegasus':
                lattice_types = ['cubic', 'kings', qpu_top, None]
            elif qpu_top == 'zephyr':
                lattice_types = ['cubic', 'kings', qpu_top, None]
            else:
                raise ValueError('Unknown qpu topology')
            #Native by default:
            shape_dicts[(qpu_top, None)] = shape_dicts[(qpu_top, qpu_top)] 
            qpu_sampler = MockDWaveSampler(topology_type=qpu_top)

            # pop final 15 edges to exercise edge cover routines.
            # 15 is a worst case upper bound on the number of defects that
            # can be handled given the default exact edge cover routines. 
            for pop_edges in [0, 15]:
                for _ in range(pop_edges):
                    qpu_sampler.edgelist.pop()
                qpu_sampler.properties['couplers'] = qpu_sampler.edgelist

                for lattice_type in lattice_types:
                    orig_embs = make_origin_embeddings(
                        qpu_sampler=qpu_sampler,
                        lattice_type=lattice_type)
                    shape_dict = shape_dicts[(qpu_top, lattice_type)]
                    tuple_length = shape_dict['tl'] 
                    chain_length = shape_dict['cl']
                    self.assertEqual(len(orig_embs), shape_dict['ne'])
                    for orig_emb in orig_embs:
                        for key, val in orig_emb.items():
                            self.assertEqual(len(key), tuple_length)
                            self.assertEqual(len(val), chain_length)

    def test_make_cubic_lattice(self):
        dims = (5,3,4)
        g = _make_cubic_lattice(dims)
        self.assertEqual(g.number_of_nodes(), dims[0]*dims[1]*dims[2])
        self.assertEqual(g.number_of_edges(),
                         (dims[0]-1)*dims[1]*dims[2] +
                         dims[0]*(dims[1]-1)*dims[2] +
                         dims[0]*dims[1]*(dims[2]-1))
        for is_open in itertools.product((0, 1), (0, 1), (0, 1)):
            g = _make_cubic_lattice(dims, is_open=is_open)
            self.assertEqual(g.number_of_nodes(), dims[0]*dims[1]*dims[2])
            self.assertEqual(g.number_of_edges(),
                             (dims[0]-is_open[0])*dims[1]*dims[2] +
                             dims[0]*(dims[1]-is_open[1])*dims[2] +
                             dims[0]*dims[1]*(dims[2]-is_open[2]))
        
    def test_make_kings_lattice(self):
        dims = (5,4)
        g = _make_kings_lattice(dims)
        self.assertEqual(g.number_of_nodes(), dims[0]*dims[1])
        self.assertEqual(g.number_of_edges(),
                         (dims[0]-1)*dims[1] +
                         dims[0]*(dims[1]-1) +
                         2*(dims[0]-1)*(dims[1]-1))
        for is_open in itertools.product((0, 1), (0, 1)):
            g = _make_kings_lattice(dims, is_open=is_open)
            self.assertEqual(g.number_of_nodes(), dims[0]*dims[1])
            self.assertEqual(g.number_of_edges(),
                             (dims[0]-is_open[0])*dims[1] +
                             dims[0]*(dims[1]-is_open[1]) +
                             2*(dims[0]-is_open[0])*(dims[1]-is_open[1]))
        
    def test_all_embeddings_validity(self):
        """Check that embeddings are valid for supported lattice_types.
        Uses a default processor scale of 5, with 15 edge defects.
        """
        # Full scale is 16, a smaller default is used
        qpu_scale = 5
        for qpu_top in ['pegasus', 'chimera', 'zephyr']:
            if qpu_top == 'pegasus':
                qpu_shape = [qpu_scale]
                lattice_types = ['cubic', qpu_top, 'kings']
            elif qpu_top == 'chimera':
                qpu_shape = [qpu_scale, qpu_scale, 4]
                lattice_types = ['cubic', qpu_top]
            elif qpu_top == 'zephyr':
                qpu_shape = [qpu_scale, 4]
                lattice_types = ['cubic', 'kings', qpu_top]

            for lattice_type in lattice_types:
                # proposed_source: a defect free-lattice at sampler
                # scale (hence inclusive of all keys).
                if lattice_type == 'cubic':
                    if qpu_top == 'zephyr':
                        cubic_dims = (qpu_scale, qpu_scale, 16)
                    elif qpu_top == 'pegasus':
                        cubic_dims = (qpu_scale-1, qpu_scale-1, 12)
                    else:
                        cubic_dims = (qpu_scale//2, qpu_scale//2, 8)
                    proposed_source = _make_cubic_lattice(cubic_dims)
                elif lattice_type == 'kings':
                    if qpu_top == 'chimera':
                        continue
                    elif qpu_top == 'pegasus':
                        kings_dims = (3*(qpu_scale-1), 3*(qpu_scale-1))
                    else:
                        kings_dims = (4*qpu_scale, 4*qpu_scale)
                    proposed_source = _make_kings_lattice(kings_dims)
                elif lattice_type == 'zephyr':
                    # Chimeralike system for geometric interpretation
                    proposed_source = dnx.zephyr_graph(qpu_scale, coordinates=True)
                    proposed_source= nx.relabel_nodes(
                        G=proposed_source,
                        mapping={n: _zephyr_to_chimeralike(
                            n, t=qpu_shape[1], half_offset=False) for n in proposed_source})
                elif lattice_type == 'pegasus':
                    proposed_source = dnx.pegasus_graph(qpu_scale,
                                                        nice_coordinates=True)
                else:
                    proposed_source = dnx.chimera_graph(qpu_scale, coordinates=True)

                qpu_sampler = MockDWaveSampler(
                    topology_type=qpu_top,
                    topology_shape=qpu_shape)

                orig_embs = make_origin_embeddings(qpu_sampler=qpu_sampler,
                                                   lattice_type=lattice_type)

                # No defects should be happy with all chains, and all edges,
                # for full graph
                for orig_emb in orig_embs:
                    self.assertEqual(len(list(orig_emb.keys())),
                                     proposed_source.number_of_nodes(),
                                     f'Failed: {qpu_top} {lattice_type}')
                # NB, only emb[0] due to technicality
                orig_emb = orig_embs[0]  
                self.assertTrue(verify_embedding(
                    emb=orig_emb,
                    source=proposed_source,
                    target=qpu_sampler.properties['couplers']),
                                f'{qpu_top} and {lattice_type}')

                # With defects, should be happy on subgraph
                for _ in range(15):
                    qpu_sampler.edgelist.pop()
                orig_embs = make_origin_embeddings(qpu_sampler=qpu_sampler,
                                                   lattice_type=lattice_type)
                for orig_emb in orig_embs:
                    self.assertTrue(verify_embedding(
                        emb=orig_emb,
                        source=proposed_source.subgraph(list(orig_emb.keys())),
                        target=qpu_sampler.properties['couplers']))

    def test_chimeralike_coordinates(self):
        known_mappings = {False: {(0,0,u,t): (u,0,t,0,0)
                              for u in range(2) for t in range(4)},
                          True: {(0,0,0,0): (0,0,2,0,0),
                                 (0,0,0,3): (0,1,1,0,0),
                                 (0,0,1,0): (1,0,2,0,0),
                                 (0,0,1,3): (1,1,1,0,0),
                                 (1,0,0,0): (0,0,2,1,0),
                                 (0,1,0,0): (0,1,2,0,0),
                                 (1,1,1,2): (1,2,0,1,0),
                                 (1,2,0,1): (0,2,3,1,0)}}
        for half_offset, known_mapping in known_mappings.items():
            for k, v in known_mapping.items():
                mapped_k = _chimeralike_to_zephyr(k, half_offset=half_offset)
                self.assertEqual(mapped_k, v)
                mapped_v = _zephyr_to_chimeralike(v, half_offset=half_offset)
                self.assertEqual(mapped_v, k)

    def test_constrained_validity(self):
        """Check that we can constrain an embedding to a given subspace
        """
        # Full scale is 16, a smaller default is used
        qpu_scale = 5
        constrained_scales = {'cubic': (2,2,2),
                              'pegasus': (3,2,3,2,4),  # 2x3 nice-cells
                              'chimera': (1,1,2,4),  # single cell
                              'kings': (2,2)  # single cell
        }
        for qpu_top in ['pegasus', 'chimera', 'zephyr']:
            lattice_types = ['cubic']
            if qpu_top != 'zephyr':
                lattice_types += [qpu_top]
            if qpu_top != 'chimera':
                lattice_types.append('kings')

            if qpu_top == 'pegasus':
                qpu_shape = [qpu_scale]
            elif qpu_top == 'chimera':
                qpu_shape = [qpu_scale, qpu_scale, 4]
            elif qpu_top == 'zephyr':
                qpu_shape = [qpu_scale, 4]

            for lattice_type in lattice_types:
                # proposed_source: a defect free-lattice at sampler
                # scale (hence inclusive of all keys).
                qpu_sampler = MockDWaveSampler(
                    topology_type=qpu_top,
                    topology_shape=qpu_shape)
                cs = constrained_scales[lattice_type]
                orig_embs = make_origin_embeddings(qpu_sampler=qpu_sampler,
                                                   lattice_type=lattice_type,
                                                   problem_dims=cs,
                                                   reject_small_problems=False)
                for orig_emb in orig_embs:
                    from numpy import prod
                    self.assertEqual(len(orig_emb), prod(cs))
                    self.assertFalse(any(any(key[idx] >= bound for idx,bound in enumerate(cs)) for key in orig_emb))
