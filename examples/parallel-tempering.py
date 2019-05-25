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
import math
import random

import numpy as np

import neal
import dimod
import hybrid


# load a problem
problem = sys.argv[1]
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp)

n = len(bqm)
m = len(bqm.quadratic)
d = 200.0 * m / n / (n - 1)
print("BQM: {} nodes, {} edges, {:.2f}% density".format(n, m, d))


# PT workflow, one approach: store temperatures/betas in states

class FixedTemperatureSampler(hybrid.Runnable, hybrid.traits.SISO):
    """PT propagate/update step."""

    def next(self, state, **runopts):
        num_reads = 20

        # neal bugfix
        samples = state.samples.change_vartype('SPIN', inplace=False)

        # construct initial_states from current samples (tile and/or trim)
        labels_map = dict(zip(samples.variables, range(len(samples.variables))))
        samples_array = samples.record.sample
        if samples_array.shape[0] < num_reads:
            samples_array = np.tile(samples_array, (num_reads // samples_array.shape[0] + 1, 1))
        if samples_array.shape[0] > num_reads:
            samples_array = samples_array[:num_reads]

        initial_states = (samples_array, labels_map)
        new_samples = neal.SimulatedAnnealingSampler().sample(
            state.problem, initial_states=initial_states,
            beta_range=(state.beta, state.beta), beta_schedule_type='linear',
            num_reads=num_reads, sweeps=1000).aggregate()

        return state.updated(samples=new_samples)


class SwapReplicas(hybrid.Runnable, hybrid.traits.MIMO):
    """PT swap replicas step."""

    def next(self, states, **runopts):
        # choose a random state, and propose a swap with the adjacent state

        i = random.choice(range(len(states) - 1))
        j = i + 1

        state_i = states[i]
        state_j = states[j]

        beta_diff = state_i.beta - state_j.beta
        energy_diff = state_i.samples.first.energy - state_j.samples.first.energy

        p = random.uniform(0, 1)
        w = min(1, math.exp(beta_diff * energy_diff))
        if p < w:
            # swap samples for replicas i and j
            states[i] = state_i.updated(samples=state_j.samples)
            states[j] = state_j.updated(samples=state_i.samples)

        return states


n_replicas = 10
n_iterations = 10

# every state is randomly initialized
state = hybrid.State.from_problem(bqm, samples=hybrid.random_sample)

# create n_replicas with geometric distribution of betas (inverse temperature)
replicas = hybrid.States(
    *[state.updated(beta=b) for b in np.geomspace(1, 0.05, n_replicas)])

# run replicas update/swap for n_iterations
# (after each update/sampling step, do n_replicas-1 swap operations)
workflow = hybrid.Loop(
    hybrid.Map(FixedTemperatureSampler())
    | hybrid.Loop(SwapReplicas(), max_iter=n_replicas-1), max_iter=n_iterations
) | hybrid.MergeSamples(aggregate=True)

solution = workflow.run(replicas).result()

# show execution profile
hybrid.profiling.print_counters(workflow)

# show results
print("Solution: sample={0.samples.first}, energy={0.samples.first.energy}".format(solution))
