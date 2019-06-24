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

"""Parallel tempering support and reference workflow implementation."""

import math
import random

import numpy as np
import neal

import dimod
import hybrid

__all__ = [
    'FixedTemperatureSampler',
    'RandomPairSampleSwap', 'SweepDownReplicasSwap'
]

class FixedTemperatureSampler(hybrid.traits.SISO, hybrid.Runnable):
    """Parallel tempering propagate/update step.

    The temperature (`beta`) can be specified upon object construction, and/or
    given externally in the input state.

    On each call, run fixed temperature (~`1/beta`) simulated annealing
    for `num_sweeps` (seeded by input sample(s)), effectively producing a new
    state by sampling from a Boltzmann distribution at the given temperature.
    """

    def __init__(self, beta=None, num_sweeps=10000, **runopts):
        super(FixedTemperatureSampler, self).__init__(**runopts)
        self.beta = beta
        self.num_sweeps = num_sweeps

    def next(self, state, **runopts):
        beta = state.get('beta', self.beta)
        new_samples = neal.SimulatedAnnealingSampler().sample(
            state.problem, initial_states=state.samples,
            beta_range=(beta, beta), beta_schedule_type='linear',
            num_sweeps=self.num_sweeps).aggregate()

        return state.updated(samples=new_samples)


class RandomPairSampleSwap(hybrid.traits.MIMO, hybrid.Runnable):
    """Parallel tempering swap replicas step.

    On each call, choose a random input state (replica), and probabilistically
    accept a swap with the adjacent state (replica). If swap is accepted, **only
    samples** contained in the selected states exchanged.
    """

    def swap_pair(self, states, i, j):
        """One pair of states' (i, j) samples probabilistic swap."""

        s_i, s_j = states[i], states[j]
        beta_diff = s_i.beta - s_j.beta
        energy_diff = s_i.samples.first.energy - s_j.samples.first.energy

        # since `min(1, math.exp(beta_diff * energy_diff))` can overflow,
        # we need to move `min` under `exp`
        w = math.exp(min(0, beta_diff * energy_diff))
        p = random.uniform(0, 1)
        if w > p:
            # swap samples for replicas i and j
            states[i].samples, states[j].samples = s_j.samples, s_i.samples

        return states

    def next(self, states, **runopts):
        i = random.choice(range(len(states) - 1))
        j = i + 1

        return self.swap_pair(states, i, j)


class SweepDownReplicasSwap(hybrid.traits.MIMO, hybrid.Runnable):
    """Parallel tempering swap replicas step.

    On each call, sweep down and probabilistically swap all adjacent pairs
    of replicas (input states).
    """

    def __init__(self, betas, **runopts):
        super(SweepDownReplicasSwap, self).__init__(**runopts)
        self.betas = betas

    def swap_pair(self, states, i, j):
        """One pair of states (i, j) probabilistic swap."""

        beta_diff = self.betas[i] - self.betas[j]
        energy_diff = states[i].samples.first.energy - states[j].samples.first.energy

        # since `min(1, math.exp(beta_diff * energy_diff))` can overflow,
        # we need to move `min` under `exp`
        w = math.exp(min(0, beta_diff * energy_diff))
        p = random.uniform(0, 1)
        if w > p:
            # swap samples for replicas i and j
            states[i], states[j] = states[j], states[i]

        return states

    def next(self, states, **runopts):
        for i in range(len(states) - 1):
            states = self.swap_pair(states, i, i + 1)

        return states
