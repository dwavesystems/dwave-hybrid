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
    QPUSubproblemAutoEmbeddingSampler, InterruptableTabuSampler)
from hybrid.decomposers import EnergyImpactDecomposer
from hybrid.composers import SplatComposer
from hybrid.core import State
from hybrid.flow import RacingBranches, ArgMin, Loop
from hybrid.utils import min_sample


# load a problem
problem = sys.argv[1]
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp)

# define the solver
iteration = RacingBranches(
    InterruptableTabuSampler(),
    EnergyImpactDecomposer(size=50, rolling=True, rolling_history=0.15)
    | QPUSubproblemAutoEmbeddingSampler()
    | SplatComposer()
) | ArgMin()
main = Loop(iteration, max_iter=10, convergence=3)

# run solver
init_state = State.from_sample(min_sample(bqm), bqm)
solution = main.run(init_state).result()

# show results
print("Solution: sample={s.samples.first}".format(s=solution))
