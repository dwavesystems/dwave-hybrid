#!/usr/bin/env python
import os
import sys
import json
from glob import glob
from itertools import chain
from collections import OrderedDict

import dimod

from hybrid.core import State
from hybrid.utils import min_sample
from hybrid.profiling import tictoc
from hybrid.samplers import TabuProblemSampler, SimulatedAnnealingSubproblemSampler
from hybrid.decomposers import IdentityDecomposer
from hybrid.composers import SplatComposer
from hybrid.reference.kerberos import KerberosSampler


problems_per_group = None

problems = list(chain(
    sorted(glob('problems/qbsolv/bqp50_*'))[:problems_per_group],
    sorted(glob('problems/qbsolv/bqp100_*'))[:problems_per_group],
    sorted(glob('problems/qbsolv/bqp250_*'))[:problems_per_group],
    sorted(glob('problems/qbsolv/bqp500_*'))[:problems_per_group],
    sorted(glob('problems/qbsolv/bqp1000_*'))[:problems_per_group],
    sorted(glob('problems/qbsolv/bqp2500_*'))[:problems_per_group],

    sorted(glob('problems/random-chimera/2048*'))[:problems_per_group],
    sorted(glob('problems/random-chimera/8192*'))[:problems_per_group],

    sorted(glob('problems/ac3/*'))[:problems_per_group],
))

workflows = [
    ("10s-tabu",
        lambda **kw: TabuProblemSampler(timeout=10000)),

    ("10k-sa",
        lambda **kw: IdentityDecomposer() | SimulatedAnnealingSubproblemSampler(sweeps=10000) | SplatComposer()),
]

samplers = [
    ("kerberos",
        lambda **kw: KerberosSampler()),
]


def run(problems, workflows, samplers, n_runs=1, targets=None):
    results = OrderedDict()
    targets = targets or {}

    def workflow_runner(bqm, workflow, **kwargs):
        init_state = State.from_sample(min_sample(bqm), bqm)

        with tictoc() as timer:
            samples = workflow.run(init_state).result().samples

        return samples, timer

    def sampler_runner(bqm, sampler, energy_threshold=None, **kwargs):
        with tictoc() as timer:
            samples = sampler.sample(bqm, init_sample=lambda: min_sample(bqm),
                                     energy_threshold=energy_threshold)

        return samples, timer

    for problem in problems:
        results[problem] = OrderedDict()

        problem_nice_name = os.path.splitext(os.path.basename(problem))[0]
        target_energy = targets.get(problem_nice_name)

        with open(problem) as fp:
            bqm = dimod.BinaryQuadraticModel.from_coo(fp)

        for runner, solvers in [(workflow_runner, workflows), (sampler_runner, samplers)]:
            for name, factory in solvers:
                run_results = []

                for run in range(n_runs):
                    case = '{!r} with {!r}, run={!r}'.format(problem, name, run)

                    try:
                        samples, timer = runner(bqm, factory(),
                                                energy_threshold=target_energy)

                    except Exception as exc:
                        raise
                        print("FAILED {case}: {exc!r}".format(**locals()))
                        run_results.append(repr(exc))

                    else:
                        print("case={case!r}"
                            " energy={samples.first.energy!r},"
                            " wallclock={timer.dt!r}".format(**locals()))
                        run_results.append(dict(energy=samples.first.energy,
                                                wallclock=timer.dt))

                results[problem][name] = run_results

    return results


if __name__ == "__main__":
    # Usage: $0 [target_energies_json]
    # Outputs info to stdout, json to stderr.

    # load target energies, if provided
    targets = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as fp:
            targets = json.load(fp)

    results = run(problems[:1], workflows[:0], samplers, n_runs=1, targets=targets)

    print(json.dumps(results), file=sys.stderr)
