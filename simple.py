#!/usr/bin/env python3
"""
- run tabu for some time T with best known solution so far as the init solution (start with random or zeros)
- pick X variables for embedding
  1) if using one-shot-embedding, pick one by one until QPU is filled
  2) if using minorminer or fixed full graph embedding:
     - find the list of "frozen" variables (those that by flipping increase total energy the most)
     - select a connected subgraph somehow
       0) naively use X frozen vars, regardless of their relation
       1) start with the first frozen, do BFS until QPU filled
       2) ?
     (TODO: how to handle boundary conditions? initially, fix all non-frozen vars)
- run X vars on QPU, get S samples back
- see if any of S samples (sub-samples in terms of complete solution) improves the total energy
  - pick the one that improves the energy the most
- loop until no improvements over previous run
"""
from enum import Enum

import dimod

from hades.samplers import (
    Solution, TabuProblemSampler, TabuSubproblemSampler, QPUSubproblemSampler)


class Subsampler(Enum):
    QPU = 1
    TABU = 2


# load problem from qubo file
problem = 'problems/random-chimera/2048.01.qubo'
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp, dimod.BINARY)


# setup subsampler
subsampler_type = Subsampler.QPU

if subsampler_type == Subsampler.TABU:
    subproblem_sampler = TabuSubproblemSampler(bqm, max_n=100, num_reads=1, timeout=20)

elif subsampler_type == Subsampler.QPU:
    subproblem_sampler = QPUSubproblemSampler(bqm, max_n=100, num_reads=100)

else:
    raise ValueError('invalid subsampler')


# setup main branch sampler
problem_sampler = TabuProblemSampler(bqm, tenure=min(20, round(len(bqm) / 4)), timeout=1000)


# initial solution
# TODO: add to bqm: max_node?
best_sample = [0] * (max(bqm.linear.keys()) + 1)
best_energy = bqm.energy(best_sample)
best_solution = Solution(best_sample, best_energy)

max_iter = 10


# iterate
prev_best_solution = best_solution
for iterno in range(max_iter):
    print("\nTabu + Subsampler (%s), iteration #%d..." % (subsampler_type, iterno))

    subsampling_branch = subproblem_sampler.run(best_solution.sample)
    sampling_branch = problem_sampler.run(best_solution.sample)

    tabu_solution = sampling_branch.result()
    print("tabu_solution", tabu_solution)

    composed_solution = subsampling_branch.result()
    print("composed_solution", composed_solution)
    
    # subsampler improves main-solver solution?
    prev_best_solution = best_solution
    if composed_solution.energy < tabu_solution.energy:
        print("!!! Sub-solver improved the main solver's solution:", tabu_solution.energy, "=>", composed_solution.energy)
        best_solution = composed_solution
    else:
        best_solution = tabu_solution

    # termination criteria
    if best_solution.energy == prev_best_solution.energy:
        break
