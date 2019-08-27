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

"""Parallel tempering support and a reference workflow implementation."""

import math
import random

import numpy as np
import neal

import dimod
import hybrid

__all__ = [
    'FixedTemperatureSampler',
    'SwapReplicaPairRandom', 'SwapReplicasDownsweep',
    'ParallelTempering', 'HybridizedParallelTempering'
]


class FixedTemperatureSampler(hybrid.traits.SISO, hybrid.Runnable):
    """Parallel tempering propagate/update step.

    The temperature (`beta`) can be specified upon object construction, and/or
    given externally (dynamically) in the input state.

    On each call, run fixed temperature (~`1/beta`) simulated annealing
    for `num_sweeps` (seeded by input sample(s)), effectively producing a new
    state by sampling from a Boltzmann distribution at the given temperature.

    Args:
        beta (float, optional):
            Inverse of constant sampling temperature. If not supplied on
            construction, it must be present in the input state.

        num_sweeps (int, optional, default=10k):
            Number of fixed temperature sampling sweeps.

        num_reads (int, optional, default=len(state.samples)):
            Number of samples produced. If undefined, inferred from the size of
            the input sample set.

        aggregate (bool, optional, default=False):
            Aggregate samples (duplicity stored in ``num_occurrences``).

        seed (int, optional, default=None):
            Pseudo-random number generator seed.

    """

    def __init__(self, beta=None, num_sweeps=10000, num_reads=None,
                 aggregate=False, seed=None, **runopts):
        super(FixedTemperatureSampler, self).__init__(**runopts)
        self.beta = beta
        self.num_sweeps = num_sweeps
        self.num_reads = num_reads
        self.aggregate = aggregate
        self.seed = seed

    def next(self, state, **runopts):
        beta = state.get('beta', self.beta)
        seed = runopts.pop('seed', self.seed)
        aggregate = runopts.pop('aggregate', self.aggregate)

        new_samples = neal.SimulatedAnnealingSampler().sample(
            state.problem, initial_states=state.samples,
            beta_range=(beta, beta), beta_schedule_type='linear',
            num_reads=self.num_reads, initial_states_generator='tile',
            num_sweeps=self.num_sweeps, seed=seed)

        if aggregate:
            new_samples = new_samples.aggregate()

        return state.updated(samples=new_samples)


class SwapReplicaPairRandom(hybrid.traits.MIMO, hybrid.Runnable):
    """Parallel tempering swap replicas step.

    On each call, choose a random input state (replica), and probabilistically
    accept a swap with the adjacent state (replica). If swap is accepted, **only
    samples** contained in the selected states are exchanged.

    Betas can be supplied in constructor, or otherwise they have to present in
    the input states.

    Args:
        betas (list(float), optional):
            List of betas (inverse temperature), one for each input state. If
            not supplied, betas have to be present in the input states.

        seed (int, default=None):
            Pseudo-random number generator seed.

    """

    def __init__(self, betas=None, seed=None, **runopts):
        super(SwapReplicaPairRandom, self).__init__(**runopts)
        self.betas = betas
        self.seed = seed
        self.random = random.Random(seed)

    def swap_pair(self, betas, states, i, j):
        """One pair of states' (i, j) samples probabilistic swap."""

        beta_diff = betas[i] - betas[j]
        energy_diff = states[i].samples.first.energy - states[j].samples.first.energy

        # since `min(1, math.exp(beta_diff * energy_diff))` can overflow,
        # we need to move `min` under `exp`
        w = math.exp(min(0, beta_diff * energy_diff))
        p = self.random.uniform(0, 1)
        if w > p:
            # swap samples for replicas i and j
            states[i].samples, states[j].samples = states[j].samples, states[i].samples

        return states

    def next(self, states, **runopts):
        betas = self.betas
        if betas is None:
            betas = [state.beta for state in states]

        i = self.random.choice(range(len(states) - 1))
        j = i + 1

        return self.swap_pair(betas, states, i, j)


class SwapReplicasDownsweep(SwapReplicaPairRandom):
    """Parallel tempering swap replicas step.

    On each call, sweep down and probabilistically swap all adjacent pairs
    of replicas (input states).

    Betas can be supplied in constructor, or otherwise they have to present in
    the input states.

    Args:
        betas (list(float), optional):
            List of betas (inverse temperature), one for each input state. If
            not supplied, betas have to be present in the input states.
    """

    def __init__(self, betas=None, **runopts):
        super(SwapReplicasDownsweep, self).__init__(betas=betas, **runopts)

    def next(self, states, **runopts):
        betas = self.betas
        if betas is None:
            betas = [state.beta for state in states]

        for i in range(len(states) - 1):
            states = self.swap_pair(betas, states, i, i + 1)

        return states


class SpawnParallelTemperingReplicas(hybrid.traits.SIMO, hybrid.Runnable):
    """Expand the input state into N states enriched with betas. Betas are
    calculated from the input problem.

    Args:
        num_replicas (int):
            Number of replicas (output states) to create off of the input state.
    """

    def __init__(self, num_replicas, **runopts):
        super(SpawnParallelTemperingReplicas, self).__init__(**runopts)
        self.num_replicas = num_replicas

    def next(self, state, **runopts):
        bqm = state.problem

        # get a reasonable beta range
        beta_hot, beta_cold = neal.default_beta_range(bqm)

        # generate betas for all branches/replicas
        betas = np.geomspace(beta_hot, beta_cold, self.num_replicas)

        # create num_replicas with betas spaced with geometric progression
        states = hybrid.States(*[state.updated(beta=b) for b in betas])

        return states


#
# A few PT workflow generators. Should be treated as Runnable classes
#

def ParallelTempering(num_sweeps=10000, num_replicas=10,
                      max_iter=None, max_time=None, convergence=3):
    """Parallel tempering workflow generator.

    Args:
        num_sweeps (int, optional):
            Number of sweeps in the fixed temperature sampling.

        num_replicas (int, optional):
            Number of replicas (parallel states / workflow branches).

        max_iter (int/None, optional):
            Maximum number of iterations of the update/swaps loop.

        max_time (int/None, optional):
            Maximum wall clock runtime (in seconds) allowed in the update/swaps
            loop.

        convergence (int/None, optional):
            Number of times best energy of the coldest replica has to repeat
            before we terminate.

    Returns:
        Workflow (:class:`~hybrid.core.Runnable` instance).

    """

    # expand single input state into `num_replicas` replica states
    preprocess = SpawnParallelTemperingReplicas(num_replicas=num_replicas)

    # fixed temperature sampling on all replicas in parallel
    update = hybrid.Map(FixedTemperatureSampler(num_sweeps=num_sweeps))

    # replica exchange step: do the top-down sweep over adjacent pairs
    # (good hot samples sink to bottom)
    swap = SwapReplicasDownsweep()

    # loop termination key function
    def key(states):
        if states is not None:
            return states[-1].samples.first.energy

    # replicas update/swap until Loop termination criteria reached
    loop = hybrid.Loop(
        update | swap,
        max_iter=max_iter, max_time=max_time, convergence=convergence, key=key)

    # collapse all replicas (although the bottom one should be the best)
    postprocess = hybrid.MergeSamples(aggregate=True)

    workflow = preprocess | loop | postprocess

    return workflow


def HybridizedParallelTempering(num_sweeps=10000, num_replicas=10,
                                max_iter=None, max_time=None, convergence=3):
    """Parallel tempering workflow generator.

    Args:
        num_sweeps (int, optional):
            Number of sweeps in the fixed temperature sampling.

        num_replicas (int, optional):
            Number of replicas (parallel states / workflow branches).

        max_iter (int/None, optional):
            Maximum number of iterations of the update/swaps loop.

        max_time (int/None, optional):
            Maximum wall clock runtime (in seconds) allowed in the update/swaps
            loop.

        convergence (int/None, optional):
            Number of times best energy of the coldest replica has to repeat
            before we terminate.

    Returns:
        Workflow (:class:`~hybrid.core.Runnable` instance).

    """

    # expand single input state into `num_replicas` replica states
    preprocess = SpawnParallelTemperingReplicas(num_replicas=num_replicas)

    # QPU branch: limits the PT workflow to QPU-sized problems
    qpu = (
        hybrid.IdentityDecomposer()
        | hybrid.QPUSubproblemAutoEmbeddingSampler()
        | hybrid.IdentityComposer()
    )

    # use QPU as the hottest temperature sampler and `num_replicas-1` fixed-temperature-samplers
    update = hybrid.Branches(
        qpu,
        *[FixedTemperatureSampler(num_sweeps=num_sweeps) for _ in range(num_replicas-1)])

    # replica exchange step: do the top-down sweep over adjacent pairs
    # (good hot samples sink to bottom)
    swap = SwapReplicasDownsweep()

    # loop termination key function
    def key(states):
        if states is not None:
            return states[-1].samples.first.energy

    # replicas update/swap until Loop termination criteria reached
    loop = hybrid.Loop(
        update | swap,
        max_iter=max_iter, max_time=max_time, convergence=convergence, key=key)

    # collapse all replicas (although the bottom one should be the best)
    postprocess = hybrid.MergeSamples(aggregate=True)

    workflow = preprocess | loop | postprocess

    return workflow
