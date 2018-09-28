#!/usr/bin/env python

# Copyright 2018 D-Wave Systems Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

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
    InterruptableTabuSampler(),
    EnergyImpactDecomposer(max_size=50, min_diff=50)
    | QPUSubproblemAutoEmbeddingSampler(num_reads=100)
    | SplatComposer()
    | TabuProblemSampler(timeout=1)
) | ArgMinFold()

main = SimpleIterator(iteration, max_iter=10, convergence=3)

init_state = State.from_sample(min_sample(bqm), bqm)

solution = main.run(init_state).result()

print("Solution: energy={s.samples.first.energy}, debug={s.debug!r}".format(s=solution))
