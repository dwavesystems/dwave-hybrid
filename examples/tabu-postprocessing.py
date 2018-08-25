#!/usr/bin/env python
import sys

import dimod

from hades.samplers import (
    QPUSubproblemAutoEmbeddingSampler, TabuProblemSampler, InterruptableTabuSampler)
from hades.decomposers import EnergyImpactDecomposer
from hades.composers import SplatComposer
from hades.core import State, SampleSet
from hades.flow import RacingBranches, ArgMinFold, SimpleIterator
from hades.utils import min_sample, max_sample, random_sample


problem = sys.argv[1]
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp)


# Run Tabu in parallel with QPU, but post-process QPU samples with very short Tabu
iteration = RacingBranches(
    InterruptableTabuSampler(bqm),
    EnergyImpactDecomposer(bqm, max_size=50, min_diff=50)
    | QPUSubproblemAutoEmbeddingSampler(num_reads=100)
    | SplatComposer(bqm)
    | TabuProblemSampler(bqm, timeout=1)
) | ArgMinFold()

main = SimpleIterator(iteration, max_iter=10, convergence=3)

init_state = State.from_sample(min_sample(bqm), bqm)

solution = main.run(init_state).result()

print("Solution: energy={s.samples.first.energy}, debug={s.debug!r}".format(s=solution))
