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
import networkx as nx
import dimod
from dimod import ExactSolver
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

def extract_bqm(bqm, frozen, state):
    """Return a sub-BQM induced by `variables` on `bqm`, fixing non sub-BQM
    variables (it's enough to fix only variables on boundary to be off by only
    a constant)"""

    # fix all variables that are not "frozen" according to sample
    fixed = bqm_variables(bqm) - frozen_vars(frozen)
    subbqm = bqm.copy()
    for v in fixed:
        subbqm.fix_variable(v, state[v])
    return subbqm

def embed(bqm, sampler, current_sample, frozen):
    subbqm = extract_bqm(bqm, frozen, current_sample)
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

def bqm_to_tabu_qubo(bqm):
    varorder = sorted(list(bqm.adj.keys()))
    ud = 0.5 * bqm.to_numpy_matrix(varorder)
    symm = ud + ud.T #- np.diag(ud.diagonal())
    qubo = symm.tolist()
    return qubo

def tabu_sample_to_bqm_response(sample, bqm):
    varorder = sorted(list(bqm.adj.keys()))
    assert len(sample) == len(varorder)
    return dict(zip(varorder, sample))


with open('../qbsolv/tests/qubos/bqp1000_1.qubo') as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp, dimod.BINARY)

G = nx.Graph(bqm.adj)
print("BQM graph connected?", nx.is_connected(G))

# get QUBO matrix repr, convert to format that our tabu solver accepts
qubo = bqm_to_tabu_qubo(bqm)

# tabu params
tenure = min(20, round(len(bqm) / 4))
scale_factor = 1
timeout = 20

# hades params
n_frozen = 100
num_reads = 100
max_iter = 10

# setup QPU access, get QPU structure
# sampler = DWaveSampler()
# sampler = ExactSolver()
#target_nodelist, target_edgelist, target_adjacency = sampler.structure

# try pure QPU approach, for sanity check
# print("Running the complete problem on QPU...")
# resp = EmbeddingComposite(sampler).sample(bqm, num_reads=num_reads)
# print("=> min energy", next(resp.data(['energy'])).energy)

# initial solution
best_sample = [0] * len(bqm)
best_energy = bqm.energy(best_sample)

# iterate
last_best_energy = best_energy
for iterno in range(max_iter):
    # based on current best_sample, run tabu and QPU on a subproblem in parallel
    print("\nTabu + Subsampler, iteration #%d..." % iterno)

    # freeze high-penalty variables, sample them via QPU
    frozen = get_frozen(bqm, best_sample)[:n_frozen]
    ## inspect subgraph connectivity before embedding
    H = nx.Graph(G.subgraph(frozen_vars(frozen)))
    print("subgraph (order %d) connected?" % H.order(), nx.is_connected(H))

    subbqm = extract_bqm(bqm, frozen, best_sample)
    r = tabu_solver.TabuSearch(bqm_to_tabu_qubo(subbqm), [0]*n_frozen, 20, scale_factor, 5000)
    best_sub_sample = tabu_sample_to_bqm_response(r.bestSolution(), subbqm)

    # run tabu for 1sec
    # TODO: run for eta_min from previous step?
    r = tabu_solver.TabuSearch(qubo, best_sample, tenure, scale_factor, timeout)
    tabu_energy = r.bestEnergy()
    tabu_sample = list(r.bestSolution())
    print("tabu_energy", tabu_energy)

    # get the best QPU solution and check if any QPU subsolution improves the global solution
    best_composed_sample = updated_sample(best_sample, best_sub_sample)
    best_composed_energy = bqm.energy(best_composed_sample)
    print("best known + qpu sample energy", best_composed_energy)

    # QPU improves tabu solution?
    if best_composed_energy < tabu_energy:
        print("!!! Sub-solver improved the main solver's solution:", tabu_energy, "=>", best_composed_energy)
        best_sample = best_composed_sample
        best_energy = best_composed_energy
    else:
        best_sample = tabu_sample
        best_energy = tabu_energy

    if best_energy < last_best_energy:
        last_best_energy = best_energy
    else:
        pass
