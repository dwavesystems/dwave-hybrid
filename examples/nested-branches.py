#!/usr/bin/env python
import sys

import dimod

from hades.samplers import (
    SimulatedAnnealingSubproblemSampler,
    TabuSubproblemSampler, InterruptableTabuSampler)
from hades.decomposers import EnergyImpactDecomposer, IdentityDecomposer
from hades.composers import SplatComposer
from hades.core import State
from hades.flow import RacingBranches, ArgMinFold, SimpleIterator
from hades.utils import min_sample


problem = sys.argv[1]
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp)


iteration = RacingBranches(
    IdentityDecomposer(bqm) | SimulatedAnnealingSubproblemSampler() | SplatComposer(bqm),
    EnergyImpactDecomposer(bqm, max_size=50, min_diff=50)
        | RacingBranches(
            SimulatedAnnealingSubproblemSampler(sweeps=1000),
            TabuSubproblemSampler(tenure=20, timeout=10),
            endomorphic=False
        )
        | ArgMinFold(lambda state: state.ctx['subsamples'].record[0].energy)
        | SplatComposer(bqm)
) | ArgMinFold()

main = SimpleIterator(iteration, max_iter=10, convergence=3)

init_state = State.from_sample(min_sample(bqm), bqm)

solution = main.run(init_state).result()

print("Solution: energy={s.samples.first.energy}, debug={s.debug!r}".format(s=solution))
