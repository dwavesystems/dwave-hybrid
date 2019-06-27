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

import dimod
import hybrid

__all__ = ['EnergyWeightedResampler']


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
