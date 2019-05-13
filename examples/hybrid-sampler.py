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

import dimod
import hybrid


# load a problem
problem = sys.argv[1]
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp)


# define the workflow
workflow = hybrid.Loop(
    hybrid.RacingBranches(
        hybrid.Identity(),
        hybrid.InterruptableTabuSampler(),
        hybrid.EnergyImpactDecomposer(size=50, rolling=True, traversal='bfs')
        | hybrid.QPUSubproblemAutoEmbeddingSampler()
        | hybrid.SplatComposer()) | hybrid.ArgMin(), convergence=3)


# create a dimod sampler that runs the workflow and sample
result = hybrid.HybridSampler(workflow).sample(bqm)

# show results
print("Solution: sample={.first}".format(result))
