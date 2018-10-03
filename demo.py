#!/usr/bin/env python
"""
Kerberos prototype: runs N samplers in parallel.
Some samplers might me interruptable.
"""

import dimod
from hades.samplers import (
    QPUSubproblemExternalEmbeddingSampler, QPUSubproblemAutoEmbeddingSampler,
    SimulatedAnnealingSubproblemSampler,
    TabuSubproblemSampler, TabuProblemSampler, InterruptableTabuSampler)
from hades.decomposers import (
    RandomSubproblemDecomposer, IdentityDecomposer,
    TilingChimeraDecomposer, EnergyImpactDecomposer)
from hades.composers import SplatComposer
from hades.core import State, SampleSet
from hades.flow import RacingBranches, ArgMinFold, SimpleIterator
from hades.utils import min_sample, max_sample, random_sample


#problem = 'problems/random-chimera/2048.01.qubo'
#problem = 'problems/random-chimera/8192.01.qubo'
#problem = 'problems/qbsolv/bqp1000_1.qubo'
problem = 'problems/ac3/ac3_00.txt'
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp)


iteration = RacingBranches(
    InterruptableTabuSampler(),
    #TabuProblemSampler(timeout=1000),
    #IdentityDecomposer() | SimulatedAnnealingSubproblemSampler(num_reads=1, sweeps=1000) | SplatComposer(),
    #RandomSubproblemDecomposer(size=100) | TabuSubproblemSampler(num_reads=1, timeout=500) | SplatComposer(),
    #RandomSubproblemDecomposer(size=100) | QPUSubproblemAutoEmbeddingSampler(num_reads=200) | SplatComposer(),
    EnergyImpactDecomposer(max_size=50, min_diff=50) | QPUSubproblemAutoEmbeddingSampler(num_reads=200) | SplatComposer(),
    #TilingChimeraDecomposer(size=(16,16,4)) | QPUSubproblemExternalEmbeddingSampler(num_reads=100) | SplatComposer(),
    #TilingChimeraDecomposer(size=(16,16,4)) | SimulatedAnnealingSubproblemSampler(num_reads=1, sweeps=1000) | SplatComposer(),
    EnergyImpactDecomposer(max_size=100, min_diff=50) | SimulatedAnnealingSubproblemSampler(num_reads=1, sweeps=1000) | SplatComposer(),
) | ArgMinFold()

main = SimpleIterator(iteration, max_iter=10, convergence=3)

init_state = State.from_sample(min_sample(bqm), bqm)

solution = main.run(init_state).result()

print("Solution: energy={s.samples.first.energy}, debug={s.debug!r}".format(s=solution))
