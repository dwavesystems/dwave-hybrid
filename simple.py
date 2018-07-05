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
- see if any of S samples (sub-samples in terms of complete solution) improve upon the total energy
  - pick the one that improves the energy the most
- loop until no improvements over previous run
"""
import numpy as np
import dimod
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import minorminer
import tabu_solver


def bqm_variables(bqm):
    "Returns all BQM variables."
    return set(bqm.adj.keys())

def get_frozen(bqm, sample):
    """Returns `list[(energy_gain, flip_index)]` in descending order of energy gain
    for flipping qubit with flip_index in sample."""
    frozen = [(bqm.energy(sample[:i] + [1 - bit] + sample[i:]), i) for i, bit in enumerate(sample)]
    frozen.sort(reverse=True)
    return frozen

def frozen_edges(bqm, frozen):
    """Returns a list of all edges in BQM between frozen variables."""
    active = set(i for e, i in frozen)
    edges = [(start, end) for (start, end), coupling in bqm.quadratic.items() if start in active and end in active]
    edges.extend((v, v) for v in bqm.linear if v in active)
    return edges

def frozen_vars(frozen):
    return set(i for e, i in frozen)

def embed(bqm, sampler, current_sample, frozen):
    # fix all variables that are not "frozen" according to sample
    fixed = bqm_variables(bqm) - frozen_vars(frozen)
    subbqm = bqm.copy()
    for v in fixed:
        subbqm.fix_variable(v, current_sample[v])
    
    source_edgelist = list(subbqm.quadratic) + [(v, v) for v in subbqm.linear]
    _, target_edgelist, target_adjacency = sampler.structure
    embedding = minorminer.find_embedding(source_edgelist, target_edgelist)
    bqm_embedded = dimod.embed_bqm(subbqm, embedding, target_adjacency, chain_strength=1.0)
    return embedding, bqm_embedded

def updated_sample(sample, replacements):
    result = sample.copy()
    for k, v in replacements.items():
        result[k] = v
    return result


with open('../qbsolv/tests/qubos/bqp100_1.qubo') as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp, dimod.BINARY)

# get QUBO matrix repr, convert to format that our tabu solver accepts
ud = 0.5 * bqm.to_numpy_matrix()
symm = ud + ud.T #- np.diag(ud.diagonal())
qubo = symm.tolist()

# tabu params
tenure = min(20, round(len(bqm) / 4))
scale_factor = 1
timeout = 1000

# hades params
n_frozen = 40
num_reads = 10000
max_iter = 100

# setup QPU access, get QPU structure
sampler = DWaveSampler()
target_nodelist, target_edgelist, target_adjacency = sampler.structure

# try pure QPU approach, for sanity check
print("Running the complete problem on QPU...")
resp = EmbeddingComposite(sampler).sample(bqm, num_reads=num_reads)
print("=> min energy", next(resp.data(['energy'])).energy)

# initial solution
best_sample = [0] * len(bqm)
best_energy = bqm.energy(best_sample)

# iterate
last_best_energy = best_energy
for iterno in range(max_iter):
    # based on current best_sample, run tabu and QPU on a subproblem in parallel
    print("\nTabu + QPU iteration #%d..." % iterno)

    # freeze high-penalty variables, sample them via QPU
    frozen = get_frozen(bqm, best_sample)
    embedding, bqm_embedded = embed(bqm, sampler, best_sample, frozen[:n_frozen])
    response = sampler.sample(bqm_embedded, num_reads=num_reads)

    # run tabu for 1sec
    # TODO: run for eta_min from previous step?
    r = tabu_solver.TabuSearch(qubo, best_sample, tenure, scale_factor, timeout)
    tabu_energy = r.bestEnergy()
    tabu_sample = list(r.bestSolution())
    print("tabu_energy", tabu_energy)

    # get the best QPU solution and check if any QPU subsolution improves the global solution
    subsamples = dimod.iter_unembed(response.samples(), embedding)
    best_qpu_sample = updated_sample(best_sample, next(subsamples))
    best_qpu_energy = bqm.energy(best_qpu_sample)
    print("best tabu + qpu sample energy", best_qpu_energy)

    # QPU improves tabu solution?
    if best_qpu_energy < tabu_energy:
        print("!!! QPU improved tabu solution:", tabu_energy, "=>", best_qpu_energy)
        best_sample = best_qpu_sample
        best_energy = best_qpu_energy
    else:
        best_sample = tabu_sample
        best_energy = tabu_energy

    if best_energy < last_best_energy:
        last_best_energy = best_energy
    else:
        pass
