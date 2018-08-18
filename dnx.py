from __future__ import division

import unittest
import collections
import math

import dimod
import dwave_networkx as dnx
import networkx as nx

from dwave_networkx.generators.chimera import chimera_coordinates


def canonical_chimera_labeling(G, t=None):
    """
    Returns a mapping from the labels of G to chimera-indexed labeling

    """
    adj = G.adj

    if t is None:
        if hasattr(G, 'edges'):
            num_edges = len(G.edges)
        else:
            num_edges = len(G.quadratic)
        t = _chimera_shore_size(adj, num_edges)

    chimera_indices = {}

    row = col = 0
    root = min(adj, key=lambda v: len(adj[v]))
    horiz, verti = rooted_tile(adj, root, t)
    while len(chimera_indices) < len(adj):

        new_indices = {}

        if row == 0:
            # if we're in the 0th row, we can assign the horizontal randomly
            for si, v in enumerate(horiz):
                new_indices[v] = (row, col, 0, si)
        else:
            # we need to match the row above
            for v in horiz:
                north = [u for u in adj[v] if u in chimera_indices]
                assert len(north) == 1
                i, j, u, si = chimera_indices[north[0]]
                assert i == row - 1 and j == col and u == 0
                new_indices[v] = (row, col, 0, si)

        if col == 0:
            # if we're in the 0th col, we can assign the vertical randomly
            for si, v in enumerate(verti):
                new_indices[v] = (row, col, 1, si)
        else:
            # we need to match the column to the east
            for v in verti:
                east = [u for u in adj[v] if u in chimera_indices]
                assert len(east) == 1
                i, j, u, si = chimera_indices[east[0]]
                assert i == row and j == col - 1 and u == 1
                new_indices[v] = (row, col, 1, si)

        chimera_indices.update(new_indices)

        # get the next root
        root_neighbours = [v for v in adj[root] if v not in chimera_indices]
        if len(root_neighbours) == 1:
            # we can increment the row
            root = root_neighbours[0]
            horiz, verti = rooted_tile(adj, root, t)

            row += 1
        else:
            # need to go back to row 0, and increment the column
            assert not root_neighbours  # should be empty

            # we want (0, col, 1, 0), we could cache this, but for now let's just go look for it
            # the slow way
            vert_root = [v for v in chimera_indices if chimera_indices[v] == (0, col, 1, 0)][0]

            vert_root_neighbours = [v for v in adj[vert_root] if v not in chimera_indices]

            if vert_root_neighbours:

                verti, horiz = rooted_tile(adj, vert_root_neighbours[0], t)
                root = next(iter(horiz))

                row = 0
                col += 1

    return chimera_indices


def rooted_tile(adj, n, t):
    horiz = {n}
    vert = set()

    # get all of the nodes that are two steps away from n
    two_steps = {v for u in adj[n] for v in adj[u] if v != n}

    # find the subset of two_steps that share exactly t neighbours
    for v in two_steps:
        shared = set(adj[n]).intersection(adj[v])

        if len(shared) == t:
            assert v not in horiz
            horiz.add(v)
            vert |= shared

    assert len(vert) == t
    return horiz, vert


def _chimera_shore_size(adj, num_edges):
    # we know |E| = m*n*t*t + (2*m*n-m-n)*t

    num_nodes = len(adj)

    max_degree = max(len(adj[v]) for v in adj)

    if num_nodes == 2 * max_degree:
        return max_degree

    def a(t):
        return -2*t

    def b(t):
        return (t + 2) * num_nodes - 2 * num_edges

    def c(t):
        return -num_nodes

    t = max_degree - 1
    m = (-b(t) + math.sqrt(b(t)**2 - 4*a(t)*c(t))) / (2 * a(t))

    if m.is_integer():
        return t

    return max_degree - 2


class TestRootedTile(unittest.TestCase):

    def test_C33_tiles(self):
        C33 = dnx.chimera_graph(3, 3, 4)

        for root in range(0, len(C33), 8):

            horiz, vert = rooted_tile(C33.adj, root, 4)

            self.assertEqual(horiz, set(range(root, root+4)))
            self.assertEqual(vert, set(range(root+4, root+8)))


class TestCanonicalChimeraLabeling(unittest.TestCase):
    def test_tile_identity(self):
        C1 = dnx.chimera_graph(1)
        coord = chimera_coordinates(1, 1, 4)

        labels = canonical_chimera_labeling(C1)
        labels = {v: coord.int(labels[v]) for v in labels}

        G = nx.relabel_nodes(C1, labels, copy=True)

        self.assertTrue(nx.is_isomorphic(G, C1))
        self.assertEqual(set(G), set(C1))

    def test_bqm_tile_identity(self):
        C1bqm = dimod.BinaryQuadraticModel.from_ising({}, {e: -1 for e in dnx.chimera_graph(1).edges})
        coord = chimera_coordinates(1, 1, 4)

        labels = canonical_chimera_labeling(C1bqm)
        labels = {v: coord.int(labels[v]) for v in labels}

        bqm = C1bqm.relabel_variables(labels, inplace=False)

        self.assertEqual(bqm, C1bqm)

    def test_row_identity(self):
        C41 = dnx.chimera_graph(4, 1)
        coord = chimera_coordinates(4, 1, 4)

        labels = canonical_chimera_labeling(C41)
        labels = {v: coord.int(labels[v]) for v in labels}

        G = nx.relabel_nodes(C41, labels, copy=True)

        self.assertTrue(nx.is_isomorphic(G, C41))

    def test_3x3_identity(self):
        C33 = dnx.chimera_graph(3, 3)
        coord = chimera_coordinates(3, 3, 4)

        labels = canonical_chimera_labeling(C33)
        labels = {v: coord.int(labels[v]) for v in labels}

        G = nx.relabel_nodes(C33, labels, copy=True)

        self.assertTrue(nx.is_isomorphic(G, C33))

    def test_construction_string_labels(self):
        C22 = dnx.chimera_graph(2, 2, 3)
        coord = chimera_coordinates(2, 2, 3)

        alpha = 'abcdefghijklmnopqrstuvwxyz'

        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

        for u, v in reversed(list(C22.edges)):
            bqm.add_interaction(alpha[u], alpha[v], 1)

        assert len(bqm.quadratic) == len(C22.edges)
        assert len(bqm) == len(C22)

        labels = canonical_chimera_labeling(bqm)
        labels = {v: alpha[coord.int(labels[v])] for v in labels}

        bqm2 = bqm.relabel_variables(labels, inplace=False)

        self.assertEqual(bqm, bqm2)

    def test__shore_size_tiles(self):
        for t in range(1, 8):
            G = dnx.chimera_graph(1, 1, t)
            self.assertEqual(_chimera_shore_size(G.adj, len(G.edges)), t)

    def test__shore_size_columns(self):
        # 2, 1, 1 is the same as 1, 1, 2
        for m in range(2, 11):
            for t in range(9, 1, -1):
                G = dnx.chimera_graph(m, 1, t)
                self.assertEqual(_chimera_shore_size(G.adj, len(G.edges)), t)

    def test__shore_size_rectangles(self):
        # 2, 1, 1 is the same as 1, 1, 2
        for m in range(2, 7):
            for n in range(2, 7):
                for t in range(1, 6):
                    G = dnx.chimera_graph(m, n, t)
                    self.assertEqual(_chimera_shore_size(G.adj, len(G.edges)), t)


if __name__ == '__main__':
    unittest.main()
