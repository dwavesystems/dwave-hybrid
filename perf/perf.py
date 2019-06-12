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

"""Performance tests."""

import os
import sys
import json
from itertools import chain
from functools import partial
from collections import OrderedDict
from glob import glob

import dimod
import hybrid

from dwave.system import DWaveSampler, EmbeddingComposite
from dwave_qbsolv import QBSolv


problems_per_group = 10

problems = list(chain(
    sorted(glob('../problems/qbsolv/bqp50_*'))[:problems_per_group],
    sorted(glob('../problems/qbsolv/bqp100_*'))[:problems_per_group],
    sorted(glob('../problems/qbsolv/bqp250_*'))[:problems_per_group],
    sorted(glob('../problems/qbsolv/bqp500_*'))[:problems_per_group],
    sorted(glob('../problems/qbsolv/bqp1000_*'))[:problems_per_group],
    sorted(glob('../problems/qbsolv/bqp2500_*'))[:problems_per_group],
    sorted(glob('../problems/random-chimera/2048*'))[:problems_per_group],
    sorted(glob('../problems/random-chimera/8192*'))[:problems_per_group],
    sorted(glob('../problems/ac3/*'))[:problems_per_group],
))

workflows = [
    ("10s-tabu",
        lambda **kw: hybrid.TabuProblemSampler(timeout=10000)),

    ("10k-sa",
        lambda **kw: (hybrid.IdentityDecomposer()
                      | hybrid.SimulatedAnnealingSubproblemSampler(sweeps=10000)
                      | hybrid.SplatComposer())),

    ("qbsolv-like",
        lambda qpu, **kw: hybrid.Loop(hybrid.Race(
            hybrid.InterruptableTabuSampler(timeout=200),
            hybrid.EnergyImpactDecomposer(size=50, rolling=True, rolling_history=0.15)
            | hybrid.QPUSubproblemAutoEmbeddingSampler(qpu_sampler=qpu)
            | hybrid.SplatComposer()
        ) | hybrid.ArgMin(), max_iter=100, convergence=10)),

    ("tiling-chimera",
        lambda qpu, **kw: hybrid.Loop(hybrid.Race(
            hybrid.InterruptableTabuSampler(timeout=200),
            hybrid.TilingChimeraDecomposer(size=(16,16,4))
            | hybrid.QPUSubproblemExternalEmbeddingSampler(qpu_sampler=qpu)
            | hybrid.SplatComposer(),
        ) | hybrid.ArgMin(), max_iter=100, convergence=10)),
]

samplers = [
    ("kerberos",
        lambda init_sample, energy_threshold, **kw: partial(
            hybrid.KerberosSampler().sample,
            init_sample=init_sample, energy_threshold=energy_threshold)),

    ("qbsolv-classic",
        lambda **kw: QBSolv().sample),

    ("qbsolv-qpu",
        lambda qpu, **kw: partial(QBSolv().sample, solver=EmbeddingComposite(qpu))),
]


def run(problems, workflows, samplers, n_runs=1, targets=None):
    results = OrderedDict()
    targets = targets or {}

    def workflow_runner(bqm, factory, qpu=None, **kwargs):
        workflow = factory(qpu=qpu)
        init_state = hybrid.State.from_sample(hybrid.min_sample(bqm), bqm)

        with hybrid.tictoc() as timer:
            samples = workflow.run(init_state).result().samples

        return samples, timer

    def sampler_runner(bqm, factory, energy_threshold=None, qpu=None, **kwargs):
        sampler = factory(
            qpu=qpu,
            init_sample=lambda: hybrid.min_sample(bqm),
            energy_threshold=energy_threshold
        )

        with hybrid.tictoc() as timer:
            samples = sampler(bqm)

        return samples, timer

    # reuse the cloud client
    qpu = DWaveSampler()

    for path in problems:
        problem = os.path.splitext(os.path.basename(path))[0]

        target_energy = targets.get(problem)
        results[problem] = OrderedDict()

        with open(path) as fp:
            bqm = dimod.BinaryQuadraticModel.from_coo(fp)

        for runner, solvers in [(workflow_runner, workflows),
                                (sampler_runner, samplers)]:
            for solver_name, factory in solvers:
                run_results = []

                for run in range(n_runs):
                    case = '{!r} with {!r}, run={!r}, target={!r}'.format(
                        problem, solver_name, run, target_energy)

                    try:
                        samples, timer = runner(bqm, factory, qpu=qpu,
                                                energy_threshold=target_energy)

                    except Exception as exc:
                        print("FAILED {case}: {exc!r}".format(**locals()))
                        run_results.append(dict(error=repr(exc)))

                    else:
                        print("case={case!r}"
                              " energy={samples.first.energy!r},"
                              " wallclock={timer.dt!r}".format(**locals()))
                        run_results.append(dict(energy=samples.first.energy,
                                                wallclock=timer.dt))

                results[problem][solver_name] = run_results

    return results


if __name__ == "__main__":
    # Usage: $0 [target_energies_json]
    # Outputs info to stdout, json to stderr.

    # load target energies, if provided
    targets = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as fp:
            targets = json.load(fp)

    results = run(problems[:1], workflows, samplers, n_runs=1, targets=targets)

    print(json.dumps(results), file=sys.stderr)
