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

"""Population annealing support and a reference workflow implementation."""

import warnings

import numpy as np

import neal
import dimod
import hybrid

__all__ = ['EnergyWeightedResampler',
           'ProgressBetaAlongSchedule',
           'CalculateAnnealingBetaSchedule',
           'PopulationAnnealing',
           'HybridizedPopulationAnnealing']


class EnergyWeightedResampler(hybrid.traits.SISO, hybrid.Runnable):
    """Sample from the input sample set according to a distribution defined with
    sample energies (with replacement) and temperature/beta difference between
    current and previous population:

        p ~ exp(-sample.energy / delta_temperature) ~ exp(-delta_beta * sample.energy)

    Args:
        delta_beta (float):
            Inverse of sampling temperature difference between current and
            previous population. Can be defined on sampler construction, on run
            method invocation, or in the input state's ``delta_beta`` variable.

        beta (float):
            Deprecated. Use 'delta_beta' instead.

        seed (int, default=None):
            Pseudo-random number generator seed.

    Returns:
        Input state with new samples. The lower the energy of an input sample,
        the higher will be its relative frequency in the output sample set. 
    """

    def __init__(self, delta_beta=None, seed=None, beta=None, **runopts):
        super(EnergyWeightedResampler, self).__init__(**runopts)

        if beta is not None:
            warnings.warn("'beta' has been replaced with 'delta_beta'.",
                          DeprecationWarning)
            if delta_beta is not None:
                warnings.warn("Ignoring 'beta' since 'delta_beta' is specified.",
                              SyntaxWarning)
            else:
                delta_beta = beta

        self.delta_beta = delta_beta
        self.seed = seed
        self.random = np.random.RandomState(seed)

    def next(self, state, **runopts):
        delta_beta = runopts.get('delta_beta', self.delta_beta)
        delta_beta = state.get('delta_beta', delta_beta)
        if delta_beta is None:
            raise ValueError('delta_beta must be given on construction or during run-time')

        ss = state.samples

        # calculate weights (note: to avoid overflow, we offset energy, as it
        # cancels out during probability calc)
        min_energy = ss.record.energy.min()
        w = np.exp(-delta_beta * (ss.record.energy - min_energy))
        p = w / sum(w)

        # resample
        idx = self.random.choice(len(ss), len(ss), p=p)
        record = ss.record[idx]
        info = ss.info.copy()
        info.update(beta=state.beta, delta_beta=delta_beta)
        new_samples = dimod.SampleSet(record, ss.variables, info, ss.vartype)

        return state.updated(samples=new_samples)


class ProgressBetaAlongSchedule(hybrid.traits.SISO, hybrid.Runnable):
    """Sets ``beta`` and ``delta_beta`` state variables according to a schedule
    given on construction or in state at first run call.

    Args:
        beta_schedule (iterable(float)):
            The beta schedule. State's ``beta``/``delta_beta`` are iterated
            according to the beta schedule.

    Raises:
        :exc:`~hybrid.exceptions.EndOfStream` when beta schedule is depleted.
    """

    def __init__(self, beta_schedule=None, **runopts):
        super(ProgressBetaAlongSchedule, self).__init__(**runopts)
        self.beta_schedule = beta_schedule

    def init(self, state, **runopts):
        beta_schedule = state.get('beta_schedule', self.beta_schedule)
        self.beta_schedule = iter(beta_schedule)

    def next(self, state, **runopts):
        try:
            next_beta = next(self.beta_schedule)
        except StopIteration:
            raise hybrid.exceptions.EndOfStream

        beta = state.get('beta', next_beta)
        delta_beta = next_beta - beta
        return state.updated(beta=next_beta, delta_beta=delta_beta)


class CalculateAnnealingBetaSchedule(hybrid.traits.SISO, hybrid.Runnable):
    """Calculate a best-guess beta schedule estimate for annealing methods,
    based on magnitudes of biases of the input problem, and the requested method
    of interpolation.

    Args:
        length (int):
            Length of the produced beta schedule.

        interpolation (str, optional, default='linear'):
            Interpolation used between the hot and the cold beta. Supported
            values are:

            * linear
            * geometric

        beta_range (tuple[float], optional):
            A 2-tuple defining the beginning and end of the beta schedule,
            where beta is the inverse temperature. The schedule is derived by
            interpolating the range with ``interpolation`` method. Default range
            is set based on the total bias associated with each node (see
            :meth:`neal.default_beta_range`).

    """

    def __init__(self, length=2, interpolation='linear', beta_range=None, **runopts):
        super(CalculateAnnealingBetaSchedule, self).__init__(**runopts)
        self.length = length
        self.interpolation = interpolation
        self.beta_range = beta_range

    def next(self, state, **runopts):
        bqm = state.problem

        if self.beta_range is None:
            # get a reasonable beta range
            beta_range = neal.default_beta_range(bqm)
        else:
            beta_range = self.beta_range

        beta_hot, beta_cold = beta_range

        # generate betas
        if self.interpolation == 'linear':
            beta_schedule = np.linspace(beta_hot, beta_cold, self.length)
        elif self.interpolation == 'geometric':
            beta_schedule = np.geomspace(beta_hot, beta_cold, self.length)
        else:
            raise ValueError("Beta schedule type {} not implemented".format(self.interpolation))

        # store the schedule in output state
        return state.updated(beta_schedule=beta_schedule)


def PopulationAnnealing(num_reads=100, num_iter=100, num_sweeps=100,
                        beta_range=None):
    """Population annealing workflow generator.

    Args:
        num_reads (int):
            Size of the population of samples.

        num_iter (int):
            Number of temperatures over which we iterate fixed-temperature
            sampling / resampling.

        num_sweeps (int):
            Number of sweeps in the fixed temperature sampling step.

        beta_range (tuple[float], optional):
            A 2-tuple defining the beginning and end of the beta
            schedule, where beta is the inverse temperature. Passed to
            :class:`.CalculateAnnealingBetaSchedule` for linear schedule
            generation.

    Returns:
        Workflow (:class:`~hybrid.core.Runnable` instance).
    """

    # PA workflow: after initial beta schedule estimation, we do `num_iter` steps
    # (one per beta/temperature) of fixed-temperature sampling / weighted resampling

    schedule_init = CalculateAnnealingBetaSchedule(
        length=num_iter, beta_range=beta_range, interpolation='linear')

    workflow = schedule_init | hybrid.Loop(
        ProgressBetaAlongSchedule()
        | hybrid.FixedTemperatureSampler(num_sweeps=num_sweeps, num_reads=num_reads)
        | EnergyWeightedResampler(),
        max_iter=num_iter
    )

    return workflow


def HybridizedPopulationAnnealing(num_reads=100, num_iter=100, num_sweeps=100,
                                  beta_range=None):
    """Workflow generator for population annealing initialized with QPU samples.

    Args:
        num_reads (int):
            Size of the population of samples.

        num_iter (int):
            Number of temperatures over which we iterate fixed-temperature
            sampling / resampling.

        num_sweeps (int):
            Number of sweeps in the fixed temperature sampling step.

        beta_range (tuple[float], optional):
            A 2-tuple defining the beginning and end of the beta
            schedule, where beta is the inverse temperature. Passed to
            :class:`.CalculateAnnealingBetaSchedule` for linear schedule
            generation.

    Returns:
        Workflow (:class:`~hybrid.core.Runnable` instance).
    """

    # QPU initial sampling: limits the PA workflow to QPU-sized problems
    qpu_init = (
        hybrid.IdentityDecomposer()
        | hybrid.QPUSubproblemAutoEmbeddingSampler(num_reads=num_reads)
        | hybrid.IdentityComposer()
    ) | hybrid.AggregatedSamples(False)

    # PA workflow: after initial QPU sampling and initial beta schedule estimation,
    # we do `num_iter` steps (one per beta/temperature) of fixed-temperature
    # sampling / weighted resampling

    schedule_init = CalculateAnnealingBetaSchedule(
        length=num_iter, beta_range=beta_range, interpolation='linear')

    workflow = qpu_init | schedule_init | hybrid.Loop(
        ProgressBetaAlongSchedule()
        | hybrid.FixedTemperatureSampler(num_sweeps=num_sweeps)
        | EnergyWeightedResampler(),
        max_iter=num_iter
    )

    return workflow
