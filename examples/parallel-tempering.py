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
import math
import random

import numpy as np

import neal
import dimod
import hybrid

from hybrid.reference.pt import FixedTemperatureSampler, SwapReplicasDownsweep


# load a problem
problem = sys.argv[1]
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp)

print("BQM: {} nodes, {} edges, {:.2f} density".format(
    len(bqm), len(bqm.quadratic), hybrid.bqm_density(bqm)))


# PT workflow, one approach: store temperatures/betas in states

n_sweeps = 10000
n_replicas = 10
n_iterations = 10

# states are randomly initialized
state = hybrid.State.from_problem(bqm)

# get a reasonable beta range
beta_hot, beta_cold = neal.default_beta_range(bqm)

# generate betas for all branches/replicas
betas = np.geomspace(beta_hot, beta_cold, n_replicas)

# create n_replicas with geometric distribution of betas (inverse temperature)
replicas = hybrid.States(*[state.updated(beta=b) for b in betas])

# run replicas update/swap for n_iterations
# (after each update/sampling step, do n_replicas-1 swap operations)
update = hybrid.Map(FixedTemperatureSampler(num_sweeps=n_sweeps))
swap = SwapReplicasDownsweep()
workflow = hybrid.Loop(update | swap, max_iter=n_iterations) \
         | hybrid.MergeSamples(aggregate=True)

solution = workflow.run(replicas).result()

# show execution profile
hybrid.profiling.print_counters(workflow)

# show results
print("Solution: sample={0.samples.first}, energy={0.samples.first.energy}".format(solution))
