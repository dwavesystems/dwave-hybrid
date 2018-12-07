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

from __future__ import print_function

import sys

import dimod

from hybrid.samplers import (
    QPUSubproblemAutoEmbeddingSampler, TabuProblemSampler, InterruptableTabuSampler)
from hybrid.decomposers import EnergyImpactDecomposer
from hybrid.composers import SplatComposer
from hybrid.core import State, SampleSet
from hybrid.flow import RacingBranches, ArgMin, SimpleIterator
from hybrid.utils import min_sample, max_sample, random_sample


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
) | ArgMin()

main = SimpleIterator(iteration, max_iter=10, convergence=3)

init_state = State.from_sample(min_sample(bqm), bqm)

solution = main.run(init_state).result()

print("Solution: energy={s.samples.first.energy}".format(s=solution))
