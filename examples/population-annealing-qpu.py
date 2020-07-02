#!/usr/bin/env python

# Copyright 2019 D-Wave Systems Inc.
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

import neal
import dimod
import hybrid

from hybrid.reference.pt import FixedTemperatureSampler
from hybrid.reference.pa import (
    CalculateAnnealingBetaSchedule, ProgressBetaAlongSchedule, EnergyWeightedResampler)


# load a problem
problem = sys.argv[1]
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp)

print("BQM: {} nodes, {} edges, {:.2f} density".format(
    len(bqm), len(bqm.quadratic), hybrid.bqm_density(bqm)))


# sweeps per fixed-temperature sampling step
num_sweeps = 1000

# number of generations, or temperatures to progress through
num_iter = 20

# population size
num_samples = 20

# QPU initial sampling: limits the PA workflow to QPU-sized problems
qpu_init = (
    hybrid.IdentityDecomposer()
    | hybrid.QPUSubproblemAutoEmbeddingSampler(num_reads=num_samples)
    | hybrid.IdentityComposer()
) | hybrid.AggregatedSamples(False)

# PA workflow: after initial beta schedule estimation, we do `num_iter` steps
# (one per beta/temperature) of fixed-temperature sampling / weighted resampling
workflow = qpu_init | CalculateAnnealingBetaSchedule(length=num_iter) | hybrid.Loop(
    ProgressBetaAlongSchedule() | FixedTemperatureSampler(num_sweeps=num_sweeps) | EnergyWeightedResampler(),
    max_iter=num_iter
)

# run the workflow
state = hybrid.State.from_problem(bqm)
solution = workflow.run(state).result()

# show execution profile
hybrid.profiling.print_counters(workflow)

# show results
print("Solution: sample={0.samples.first}, energy={0.samples.first.energy}".format(solution))
