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

"""Dialectic search [1][2] algorithm works roughly as follows: it tries to
greedily improve a given assignment (the thesis), then greedily improve a
randomized modification (an antithesis) of the current assignment, and finally
greedily improve a combination of the two assignments (the synthesis). If this
new assignment is at least as good, it replaces the current assignment. If this
process does not produce improved results for a set period of time, the search
continues from the modified assignment instead.

[1] Kadioglu S., Sellmann M. (2009) Dialectic Search. In: Gent I.P. (eds)
    Principles and Practice of Constraint Programming - CP 2009. CP 2009.
    Lecture Notes in Computer Science, vol 5732. Springer, Berlin, Heidelberg

[2] http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.186.3521&rep=rep1&type=pdf
"""

from __future__ import print_function

import sys

import dimod
import hybrid


# load a problem
problem = sys.argv[1]
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp)


# construct a Dialectic Search workflow
generate_antithesis = ( hybrid.IdentityDecomposer()
                      | hybrid.RandomSubproblemSampler()
                      | hybrid.SplatComposer()
                      | hybrid.TabuProblemSampler())

generate_synthesis = ( hybrid.GreedyPathMerge()
                     | hybrid.TabuProblemSampler())

min_tracker = hybrid.TrackMin()

local_update = hybrid.LoopWhileNoImprovement(
    hybrid.Parallel(generate_antithesis) | generate_synthesis | min_tracker,
    max_tries=10)

global_update = hybrid.Loop(generate_antithesis | local_update, max_iter=10)


# run the workflow
init_state = hybrid.State.from_sample(hybrid.min_sample(bqm), bqm)
final_state = global_update.run(init_state).result()

# show execution profile
hybrid.profiling.print_counters(global_update)

# show results
print("Solution: sample={.samples.first}".format(min_tracker.best))
