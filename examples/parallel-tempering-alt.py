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

from __future__ import print_function

import sys
import math
import random

import numpy as np

import neal
import dimod
import hybrid

from hybrid.reference.pt import FixedTemperatureSampler
from hybrid.reference.pt import SweepDownReplicasSwap


# load a problem
problem = sys.argv[1]
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp)

print("BQM: {} nodes, {} edges, {:.2f} density".format(
    len(bqm), len(bqm.quadratic), hybrid.bqm_density(bqm)))


# PT workflow: temperature/beta is a property of a branch

n_sweeps = 10000
n_replicas = 10
n_iterations = 10

# replicas are initialized with random samples
state = hybrid.State.from_problem(bqm)
replicas = hybrid.States(*[state.updated() for _ in range(n_replicas)])

# get a reasonable beta range
beta_hot, beta_cold = neal.default_beta_range(bqm)

# generate betas for all branches/replicas
betas = np.geomspace(beta_hot, beta_cold, n_replicas)

# run replicas update/swap for n_iterations
# (after each update/sampling step, do n_replicas-1 swap operations)
update = hybrid.Branches(*[FixedTemperatureSampler(beta=beta, num_sweeps=n_sweeps) for beta in betas])
swap = SweepDownReplicasSwap(betas)
workflow = hybrid.Loop(update | swap, max_iter=n_iterations) \
         | hybrid.MergeSamples(aggregate=True)

solution = workflow.run(replicas).result()

# show execution profile
hybrid.profiling.print_counters(workflow)

# show results
print("Solution: sample={0.samples.first}, energy={0.samples.first.energy}".format(solution))
