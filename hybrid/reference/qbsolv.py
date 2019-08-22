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

"""QBSolv inspired simple workflows."""

import dimod
import hybrid

__all__ = ['SimplifiedQbsolv']


def SimplifiedQbsolv(max_iter=10, max_time=None, convergence=3,
                     energy_threshold=None, max_subproblem_size=30):
    """Races a Tabu solver and a QPU-based sampler of flip-energy-impact induced
    subproblems.

    For arguments description see: :class:`~hybrid.reference.kerberos.Kerberos`.
    """

    energy_reached = None
    if energy_threshold is not None:
        energy_reached = lambda en: en <= energy_threshold

    workflow = hybrid.Loop(
        hybrid.Race(
            hybrid.InterruptableTabuSampler(),
            hybrid.EnergyImpactDecomposer(
                size=max_subproblem_size, rolling=True, rolling_history=0.15)
            | hybrid.QPUSubproblemAutoEmbeddingSampler()
            | hybrid.SplatComposer()
        ) | hybrid.ArgMin() | hybrid.TrackMin(output=True),
        max_iter=max_iter, max_time=max_time,
        convergence=convergence, terminate=energy_reached)

    return workflow
