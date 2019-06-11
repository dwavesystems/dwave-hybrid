#!/usr/bin/env python
import sys
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


def run(problems, workflows, samplers):
    results = OrderedDict()

    def run_workflow(bqm, workflow):
        init_state = State.from_sample(min_sample(bqm), bqm)

        with tictoc() as timer:
            samples = workflow.run(init_state).result().samples

        return samples, timer

    def run_sampler(bqm, sampler):
        with tictoc() as timer:
            samples = sampler.sample(bqm, init_sample=lambda: min_sample(bqm))

        return samples, timer

    for problem in problems:
        results[problem] = OrderedDict()

        with open(problem) as fp:
            bqm = dimod.BinaryQuadraticModel.from_coo(fp)

        for runner, solvers in [(run_workflow, workflows), (run_sampler, samplers)]:
            for name, factory in solvers:
                case = '{!r} with {!r}'.format(problem, name)

                try:
                    samples, timer = runner(bqm, factory())

                except Exception as exc:
                    raise
                    print("FAILED {case}: {exc!r}".format(**locals()))
                    results[problem][name] = repr(exc)

                else:
                    print("case={case!r}"
                        " energy={samples.first.energy!r},"
                        " wallclock={timer.dt!r}".format(**locals()))
                    results[problem][name] = dict(
                        energy=samples.first.energy,
                        wallclock=timer.dt)

    return results


if __name__ == "__main__":
    import json
    results = run(problems[:1], workflows[:0], samplers)
    print(json.dumps(results), file=sys.stderr)
