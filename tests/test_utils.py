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
import dwave_networkx as dnx

from hybrid.utils import (
    chimera_tiles, flip_energy_gains, select_localsearch_adversaries)


class TestEnergyFlipGainUtils(unittest.TestCase):
    # minimized when variables differ
    notall = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'ca': 1}, 0, dimod.SPIN)

    # minimized when B is different from A and C
    notb = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'ca': -1}, 0, dimod.SPIN)

    def test_pos(self):
        """Correctly orders positive gains."""

        # flipping C makes for the highest energy gain
        gains = flip_energy_gains(self.notall, {'a': 1, 'b': 1, 'c': -1})
        self.assertEqual(gains, [(4.0, 'c'), (0.0, 'b'), (0.0, 'a')])
        gains = flip_energy_gains(self.notall, {'a': -1, 'b': -1, 'c': 1})
        self.assertEqual(gains, [(4.0, 'c'), (0.0, 'b'), (0.0, 'a')])

        # flipping any is equally bad
        gains = flip_energy_gains(self.notb, {'a': 1, 'b': -1, 'c': 1})
        self.assertEqual(gains, [(4.0, 'c'), (4.0, 'b'), (4.0, 'a')])

    def test_neg(self):
        """Correctly orders negative gains."""

        # flipping any is equally good
        gains = flip_energy_gains(self.notall, {'a': 1, 'b': 1, 'c': 1})
        self.assertEqual(gains, [(-4.0, 'c'), (-4.0, 'b'), (-4.0, 'a')])

        # flipping B is the worst
        gains = flip_energy_gains(self.notb, {'a': 1, 'b': 1, 'c': 1})
        self.assertEqual(gains, [(0.0, 'c'), (0.0, 'a'), (-4.0, 'b')])

    def test_subset(self):
        """Flip energy is correctly calculated for a subset of variables."""

        gains = flip_energy_gains(self.notb, {'a': 1, 'b': 1, 'c': 1}, {'b'})
        self.assertEqual(gains, [(-4.0, 'b')])

        gains = flip_energy_gains(self.notb, {'a': 1, 'b': 1, 'c': 1}, ['c'])
        self.assertEqual(gains, [(0.0, 'c')])

        gains = flip_energy_gains(self.notb, {'a': 1, 'b': 1, 'c': 1}, 'ab')
        self.assertEqual(gains, [(0.0, 'a'), (-4.0, 'b')])

    def test_localsearch_adversaries(self):
        """When var flip increases energy."""

        defvar = select_localsearch_adversaries(self.notall, {'a': 1, 'b': 1, 'c': -1})
        self.assertEqual(defvar, ['c', 'b', 'a'])

        allvar = select_localsearch_adversaries(self.notall, {'a': 1, 'b': 1, 'c': -1}, max_n=3, min_gain=-1)
        self.assertEqual(allvar, ['c', 'b', 'a'])

        subvar = select_localsearch_adversaries(self.notall, {'a': 1, 'b': 1, 'c': -1}, max_n=2, min_gain=-1)
        self.assertEqual(subvar, ['c', 'b'])

        subvar = select_localsearch_adversaries(self.notall, {'a': 1, 'b': 1, 'c': -1}, min_gain=1)
        self.assertEqual(subvar, ['c'])

        nonevar = select_localsearch_adversaries(self.notall, {'a': 1, 'b': 1, 'c': -1}, min_gain=10)
        self.assertEqual(nonevar, [])

    def test_localsearch_friends(self):
        """When var flip decreases energy."""

        allvar = select_localsearch_adversaries(self.notb, {'a': 1, 'b': 1, 'c': 1}, min_gain=None)
        self.assertEqual(allvar, ['c', 'a', 'b'])

        neutral = select_localsearch_adversaries(self.notb, {'a': 1, 'b': 1, 'c': 1}, min_gain=0.0)
        self.assertEqual(neutral, ['c', 'a'])

        nonevar = select_localsearch_adversaries(self.notb, {'a': 1, 'b': 1, 'c': 1}, min_gain=1.0)
        self.assertEqual(nonevar, [])

        friends = select_localsearch_adversaries(self.notb, {'a': 1, 'b': 1, 'c': 1}, min_gain=-10)
        self.assertEqual(friends, ['c', 'a', 'b'])


class TestChimeraTiles(unittest.TestCase):
    def test_single_target(self):
        bqm = dimod.BinaryQuadraticModel.from_qubo({edge: 1 for edge in dnx.chimera_graph(4).edges})

        tiles = chimera_tiles(bqm, 1, 1, 4)

        self.assertEqual(len(tiles), 16)  # we have the correct number of tiles
        self.assertEqual(set().union(*tiles.values()), set(bqm))  # all of the nodes are present
        for embedding in tiles.values():
            self.assertEqual(set(chain[0] for chain in embedding.values()), set(range(1*1*4*2)))

    def test_even_divisor(self):
        bqm = dimod.BinaryQuadraticModel.from_qubo({edge: 1 for edge in dnx.chimera_graph(4).edges})

        tiles = chimera_tiles(bqm, 2, 2, 4)

        self.assertEqual(len(tiles), 4)  # we have the correct number of tiles
        self.assertEqual(set().union(*tiles.values()), set(bqm))  # all of the nodes are present
        for embedding in tiles.values():
            self.assertEqual(set(chain[0] for chain in embedding.values()), set(range(2*2*4*2)))

    def test_uneven_divisor(self):
        si, sj, st = 3, 3, 4
        ti, tj, tt = 2, 2, 3
        bqm = dimod.BinaryQuadraticModel.from_qubo({edge: 1 for edge in dnx.chimera_graph(si, sj, st).edges})

        tiles = chimera_tiles(bqm, ti, tj, tt)

        self.assertEqual(len(tiles), 8)  # we have the correct number of tiles
        self.assertEqual(set().union(*tiles.values()), set(bqm))  # all of the nodes are present
        for embedding in tiles.values():
            self.assertTrue(set(chain[0] for chain in embedding.values()).issubset(set(range(ti*tj*tt*2))))

    def test_string_labels(self):
        si, sj, st = 2, 2, 3
        ti, tj, tt = 1, 1, 4
        alpha = 'abcdefghijklmnopqrstuvwxyz'

        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

        for u, v in reversed(list(dnx.chimera_graph(si, sj, st).edges)):
            bqm.add_interaction(alpha[u], alpha[v], 1)

        tiles = chimera_tiles(bqm, ti, tj, tt)

        self.assertEqual(len(tiles), 4)  # we have the correct number of tiles
        self.assertEqual(set().union(*tiles.values()), set(bqm))  # all of the nodes are present
        for embedding in tiles.values():
            self.assertTrue(set(chain[0] for chain in embedding.values()).issubset(set(range(ti*tj*tt*2))))
