#!/usr/bin/env python
import os
import random

import dimod
from dwave.system.samplers import DWaveSampler


def generate_random_chimera_problem(adjacency, h_range, j_range, offset=0, vartype=dimod.BINARY):
    """Generate a random chimera problem, with
    h int chosen randomly from h_range, j int chosen randomly from j_range.

    Typically: h_range = [0, 0] and j_range = [-k, +k].

    Args:
        adjacency (dict[/{node: {neighbor_node_1, ...}}): Adjacency dictionary
        h_range (tuple/(upper,lower)): bounds for h
        j_range (tuple/(upper,lower)): bounds for j
        offset (float): energy offset
        vartype (dimod.Vartype): BQM's vartype
    
    Returns:
        dimod.BinaryQuadraticModel
    """

    h = {n: random.randint(*h_range) for n in adjacency.keys()}
    J = {(n,e): random.randint(*j_range)
            for n, edges in adjacency.items()
                for e in edges
                    if e > n}

    return dimod.BinaryQuadraticModel(h, J, offset, vartype)


if __name__ == "__main__":
    # generate 10 random chimera/QPU-structured problems with J's in +/-k

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'problems/random-chimera')
    qpu = DWaveSampler()

    for k in range(1, 11):
        bqm = generate_random_chimera_problem(qpu.adjacency, (0, 0), (-k, k))
        path = os.path.join(outdir, '{}.{:0>2}.qubo'.format(len(bqm), k))

        with open(path, 'w') as fp:
            bqm.to_coo(fp)
