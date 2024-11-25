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

import sys

import dimod
import hybrid


# load a problem
problem = sys.argv[1]
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp)


# construct a workflow that races Simulated Annealing against SA/Tabu on a subproblem
iteration = hybrid.Race(
    hybrid.SimulatedAnnealingProblemSampler(),
    hybrid.EnergyImpactDecomposer(size=50)
        | hybrid.RacingBranches(
            hybrid.SimulatedAnnealingSubproblemSampler(num_sweeps=1000),
            hybrid.TabuSubproblemSampler(tenure=20, timeout=10))
        | hybrid.ArgMin('subsamples.first.energy')
        | hybrid.SplatComposer()
) | hybrid.ArgMin('samples.first.energy')
main = hybrid.Loop(iteration, max_iter=10, convergence=3)


# run the workflow
init_state = hybrid.State.from_sample(hybrid.utils.min_sample(bqm), bqm)
solution = main.run(init_state).result()

# show results
print("""
Solution:
    energy={s.samples.first.energy}
    sample={s.samples.first.sample}
""".format(s=solution))
