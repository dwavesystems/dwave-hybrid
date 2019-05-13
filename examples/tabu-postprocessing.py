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
import hybrid


# load a problem
problem = sys.argv[1]
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp)


# run Tabu in parallel with QPU, but post-process QPU samples with very short Tabu
iteration = hybrid.RacingBranches(
    hybrid.Identity(),
    hybrid.InterruptableTabuSampler(),
    hybrid.EnergyImpactDecomposer(size=50)
    | hybrid.QPUSubproblemAutoEmbeddingSampler(num_reads=100)
    | hybrid.SplatComposer()
    | hybrid.TabuProblemSampler(timeout=1)
) | hybrid.ArgMin()

main = hybrid.Loop(iteration, max_iter=10, convergence=3)


# run the workflow
init_state = hybrid.State.from_sample(hybrid.utils.min_sample(bqm), bqm)
solution = main.run(init_state).result()

# show results
print("Solution: sample={.samples.first}".format(solution))
