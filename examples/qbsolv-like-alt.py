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


# define a qbsolv-like workflow
def merge_substates(_, substates):
    a, b = substates
    return a.updated(subsamples=hybrid.hstack_samplesets(a.subsamples, b.subsamples))

subproblems = hybrid.Unwind(
    hybrid.EnergyImpactDecomposer(size=50, rolling_history=0.15)
)

qpu = hybrid.Map(
    hybrid.QPUSubproblemAutoEmbeddingSampler()
) | hybrid.Reduce(
    hybrid.Lambda(merge_substates)
) | hybrid.SplatComposer()

random = hybrid.Map(
    hybrid.RandomSubproblemSampler()
) | hybrid.Reduce(
    hybrid.Lambda(merge_substates)
) | hybrid.SplatComposer()

subsampler = hybrid.Parallel(qpu, random, endomorphic=False) | hybrid.ArgMin()

iteration = hybrid.Race(
    hybrid.Identity(),
    hybrid.InterruptableTabuSampler(),
    subproblems | subsampler
) | hybrid.ArgMin()

main = hybrid.Loop(iteration, max_iter=10, convergence=3)


# run the workflow
init_state = hybrid.State.from_sample(hybrid.min_sample(bqm), bqm)
solution = main.run(init_state).result()

# show execution profile
hybrid.profiling.print_counters(main)

# show results
print("Solution: sample={.samples.first}".format(solution))
