#!/usr/bin/env python

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

import os
import sys
import random

import click
import dimod
import dwave_networkx as dnx


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


@click.command()
@click.option('--size', type=(int, int, int), default=(16, 16, 4),
              help='Size of generated problems. For Chimera, use three-tuple.')
@click.option('--vartype', type=click.Choice(['SPIN', 'BINARY']), default='SPIN',
              help="Generated problems' type (Ising/QUBO).")
@click.option('--count', type=int, default=10,
              help='Number of generated problems.')
@click.option('--format', 'fmt', type=click.Choice(['coo', 'json']), default='coo',
              help='Output format.')
@click.option('--outdir', type=click.Path(exists=True, file_okay=False), required=False,
              help='Output directory. Defaults to stdout.')
def generate_chimera(size, vartype, count, fmt, outdir):
    """Generate `count` of random Chimera-structured problems
    with `size` topology, with zero biases and random J's in +/-k range
    (where k goes from 1 to `count`).
    """

    def store(bqm, fp):
        if fmt == 'coo':
            fp.write(bqm.to_coo(vartype_header=True))
        elif fmt == 'json':
            fp.write(bqm.to_json())

    ext = {'SPIN': 'ising', 'BINARY': 'qubo'}

    adj = dnx.chimera_graph(*size).adj

    for k in range(1, count+1):
        bqm = generate_random_chimera_problem(adj, (0, 0), (-k, k), vartype=vartype)

        if outdir:
            path = os.path.join(outdir, '{}.{:0>2}.{}'.format(len(bqm), k, ext[vartype]))
            with open(path, 'w') as fp:
                store(bqm, fp)
        else:
            store(bqm, sys.stdout)


if __name__ == '__main__':
    generate_chimera()
