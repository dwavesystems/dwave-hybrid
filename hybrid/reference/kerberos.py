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

"""
Kerberos hybrid sampler runs 3 sampling branches in parallel. In each iteration,
best results from tabu search and simulated annealing are combined with best
results from QPU sampling a subproblem.
"""

import dimod
import hybrid

__all__ = ['Kerberos', 'KerberosSampler']


def Kerberos(max_iter=100, max_time=None, convergence=3, energy_threshold=None,
             sa_reads=1, sa_sweeps=10000, tabu_timeout=500,
             qpu_reads=100, qpu_sampler=None, qpu_params=None,
             max_subproblem_size=50):
    """An opinionated hybrid asynchronous decomposition sampler for problems of
    arbitrary structure and size. Runs Tabu search, Simulated annealing and QPU
    subproblem sampling (for high energy impact problem variables) in parallel
    and returns the best samples.

    Kerberos workflow is used by :class:`KerberosSampler`.

    Termination Criteria Args:

        max_iter (int):
            Number of iterations in the hybrid algorithm.

        max_time (float/None, optional, default=None):
            Wall clock runtime termination criterion. Unlimited by default.

        convergence (int):
            Number of iterations with no improvement that terminates sampling.

        energy_threshold (float, optional):
            Terminate when this energy threshold is surpassed. Check is
            performed at the end of each iteration.

    Simulated Annealing Parameters:

        sa_reads (int):
            Number of reads in the simulated annealing branch.

        sa_sweeps (int):
            Number of sweeps in the simulated annealing branch.

    Tabu Search Parameters:

        tabu_timeout (int):
            Timeout for non-interruptable operation of tabu search (time in
            milliseconds).

    QPU Sampling Parameters:

        qpu_reads (int):
            Number of reads in the QPU branch.

        qpu_sampler (:class:`dimod.Sampler`, optional, default=DWaveSampler()):
            Quantum sampler such as a D-Wave system.

        qpu_params (dict):
            Dictionary of keyword arguments with values that will be used
            on every call of the QPU sampler.

        max_subproblem_size (int):
            Maximum size of the subproblem selected in the QPU branch.

    Returns:
        Workflow (:class:`~hybrid.core.Runnable` instance).

    """

    energy_reached = None
    if energy_threshold is not None:
        energy_reached = lambda en: en <= energy_threshold

    iteration = hybrid.Race(
        hybrid.BlockingIdentity(),
        hybrid.InterruptableTabuSampler(
            timeout=tabu_timeout),
        hybrid.InterruptableSimulatedAnnealingProblemSampler(
            num_reads=sa_reads, num_sweeps=sa_sweeps),
        hybrid.EnergyImpactDecomposer(
            size=max_subproblem_size, rolling=True, rolling_history=0.3, traversal='bfs')
            | hybrid.QPUSubproblemAutoEmbeddingSampler(
                num_reads=qpu_reads, qpu_sampler=qpu_sampler, sampling_params=qpu_params)
            | hybrid.SplatComposer()
    ) | hybrid.ArgMin()

    workflow = hybrid.Loop(iteration, max_iter=max_iter, max_time=max_time,
                           convergence=convergence, terminate=energy_reached)

    return workflow


class KerberosSampler(dimod.Sampler):
    """An opinionated dimod-compatible hybrid asynchronous decomposition sampler
    for problems of arbitrary structure and size.

    Examples:
        This example solves a two-variable Ising model.

        >>> import dimod
        >>> import hybrid
        >>> response = hybrid.KerberosSampler().sample_ising(
        ...                     {'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})    # doctest: +SKIP
        >>> response.data_vectors['energy']      # doctest: +SKIP
        array([-1.5])

    """

    properties = None
    parameters = None
    runnable = None

    def __init__(self):
        self.parameters = {
            'num_reads': [],
            'init_sample': [],
            'max_iter': [],
            'max_time': [],
            'convergence': [],
            'energy_threshold': [],
            'sa_reads': [],
            'sa_sweeps': [],
            'tabu_timeout': [],
            'qpu_reads': [],
            'qpu_sampler': [],
            'qpu_params': [],
            'max_subproblem_size': []
        }
        self.properties = {}

    def sample(self, bqm, init_sample=None, num_reads=1, **kwargs):
        """Run Tabu search, Simulated annealing and QPU subproblem sampling (for
        high energy impact problem variables) in parallel and return the best
        samples.

        Sampling Args:

            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            init_sample (:class:`~dimod.SampleSet`, callable, ``None``):
                Initial sample set (or sample generator) used for each "read".
                Use a random sample for each read by default.

            num_reads (int):
                Number of reads. Each sample is the result of a single run of the
                hybrid algorithm.

        Termination Criteria Args:

            max_iter (int):
                Number of iterations in the hybrid algorithm.

            max_time (float/None, optional, default=None):
                Wall clock runtime termination criterion. Unlimited by default.

            convergence (int):
                Number of iterations with no improvement that terminates sampling.

            energy_threshold (float, optional):
                Terminate when this energy threshold is surpassed. Check is
                performed at the end of each iteration.

        Simulated Annealing Parameters:

            sa_reads (int):
                Number of reads in the simulated annealing branch.

            sa_sweeps (int):
                Number of sweeps in the simulated annealing branch.

        Tabu Search Parameters:

            tabu_timeout (int):
                Timeout for non-interruptable operation of tabu search (time in
                milliseconds).

        QPU Sampling Parameters:

            qpu_reads (int):
                Number of reads in the QPU branch.

            qpu_sampler (:class:`dimod.Sampler`, optional, default=DWaveSampler()):
                Quantum sampler such as a D-Wave system.

            qpu_params (dict):
                Dictionary of keyword arguments with values that will be used
                on every call of the QPU sampler.

            max_subproblem_size (int):
                Maximum size of the subproblem selected in the QPU branch.

        Returns:
            :obj:`~dimod.SampleSet`: A `dimod` :obj:`.~dimod.SampleSet` object.

        """

        if callable(init_sample):
            init_state_gen = lambda: hybrid.State.from_sample(init_sample(), bqm)
        elif init_sample is None:
            init_state_gen = lambda: hybrid.State.from_sample(hybrid.random_sample(bqm), bqm)
        elif isinstance(init_sample, dimod.SampleSet):
            init_state_gen = lambda: hybrid.State.from_sample(init_sample, bqm)
        else:
            raise TypeError("'init_sample' should be a SampleSet or a SampleSet generator")

        self.runnable = Kerberos(**kwargs)

        samples = []
        energies = []
        for _ in range(num_reads):
            init_state = init_state_gen()
            final_state = self.runnable.run(init_state)
            # the best sample from each run is one "read"
            ss = final_state.result().samples
            ss.change_vartype(bqm.vartype, inplace=True)
            samples.append(ss.first.sample)
            energies.append(ss.first.energy)

        return dimod.SampleSet.from_samples(samples, vartype=bqm.vartype, energy=energies)
