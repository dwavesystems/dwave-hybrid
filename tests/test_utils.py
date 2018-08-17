import unittest

import dimod
import dwave_networkx as dnx

from hades.utils import chimera_tiles


class Test_chimera_tiles(unittest.TestCase):
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
