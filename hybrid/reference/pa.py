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

import numpy as np

import neal
import dimod
import hybrid

__all__ = ['EnergyWeightedResampler',
           'ProgressBetaAlongSchedule',
           'CalculateAnnealingBetaSchedule']


class EnergyWeightedResampler(hybrid.traits.SISO, hybrid.Runnable):
    """Sample from the input sample set according to a distribution defined with
    sample energies (with replacement):

        p ~ exp(-sample.energy / temperature) ~ exp(-beta * sample.energy)

    Args:
        beta (float):
            Inverse of sampling temperature. Can be defined on sampler
            construction, on run method invocation, or in the input state's
            ``beta`` variable.

    Returns:
        Input state with new samples. The lower the energy of an input sample,
        the higher will be its relative frequency in the output sample set. 
    """

    def __init__(self, beta=None, **runopts):
        super(EnergyWeightedResampler, self).__init__(**runopts)
        self.beta = beta

    def next(self, state, **runopts):
        beta = runopts.get('beta', self.beta)
        beta = state.get('beta', beta)
        if beta is None:
            raise ValueError('beta must be given on construction or during run-time')

        ss = state.samples

        # calculate weights
        w = np.exp(-beta * ss.record.energy)
        p = w / sum(w)

        # resample
        idx = np.random.choice(len(ss), len(ss), p=p)
        record = ss.record[idx]
        info = ss.info.copy()
        info.update(beta=beta)
        new_samples = dimod.SampleSet(record, ss.variables, info, ss.vartype)

        return state.updated(samples=new_samples)


class ProgressBetaAlongSchedule(hybrid.traits.SISO, hybrid.Runnable):
    """Given the beta schedule (on construction or in state on first run),
    this runnable will set the ``beta`` state variable to a value according to
    the schedule.

    Args:
        beta_schedule (iterable(float)):
            The beta schedule. State's ``beta`` is iterated according to the
            beta schedule.
    """

    def __init__(self, beta_schedule=None, **runopts):
        super(ProgressBetaAlongSchedule, self).__init__(**runopts)
        self.beta_schedule = beta_schedule

    def init(self, state, **runopts):
        beta_schedule = state.get('beta_schedule', self.beta_schedule)
        self.beta_schedule = iter(beta_schedule)

    def next(self, state, **runopts):
        return state.updated(beta=next(self.beta_schedule))


class CalculateAnnealingBetaSchedule(hybrid.traits.SISO, hybrid.Runnable):
    """Calculate a good estimate of beta schedule for annealing methods, based
    on magnitudes of biases of the input problem.

    Args:
        length (int):
            Length of the produced beta schedule.

        interpolation (str, optional, default='geometric'):
            Interpolation used between the hot and the cold beta. Supported
            values are:

            * linear
            * geometric
    """

    def __init__(self, length=2, interpolation='geometric', **runopts):
        super(CalculateAnnealingBetaSchedule, self).__init__(**runopts)
        self.length = length
        self.interpolation = interpolation

    def next(self, state, **runopts):
        bqm = state.problem

        # get a reasonable beta range
        beta_hot, beta_cold = neal.default_beta_range(bqm)

        # generate betas
        if self.interpolation == 'linear':
            beta_schedule = np.linspace(beta_hot, beta_cold, self.length)
        elif self.interpolation == 'geometric':
            beta_schedule = np.geomspace(beta_hot, beta_cold, self.length)
        else:
            raise ValueError("Beta schedule type {} not implemented".format(self.interpolation))

        # store the schedule in output state
        return state.updated(beta_schedule=beta_schedule)
