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
import operator

import dimod

from hybrid.samplers import (
    SimulatedAnnealingSubproblemSampler,
    TabuSubproblemSampler, InterruptableTabuSampler)
from hybrid.decomposers import EnergyImpactDecomposer, IdentityDecomposer
from hybrid.composers import SplatComposer
from hybrid.core import State
from hybrid.flow import RacingBranches, ArgMin, Loop
from hybrid.utils import min_sample


problem = sys.argv[1]
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp)


iteration = RacingBranches(
    IdentityDecomposer() | SimulatedAnnealingSubproblemSampler() | SplatComposer(),
    EnergyImpactDecomposer(size=50)
        | RacingBranches(
            SimulatedAnnealingSubproblemSampler(sweeps=1000),
            TabuSubproblemSampler(tenure=20, timeout=10),
            endomorphic=False
        )
        | ArgMin(operator.attrgetter('subsamples.first.energy'))
        | SplatComposer()
) | ArgMin()

main = Loop(iteration, max_iter=10, convergence=3)

init_state = State.from_sample(min_sample(bqm), bqm)

solution = main.run(init_state).result()

print("Solution: energy={s.samples.first.energy}".format(s=solution))
