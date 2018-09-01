#!/usr/bin/env python
"""Performance tests."""

from itertools import chain
from glob import glob

import dimod
from dwave.system.samplers import DWaveSampler

from hades.samplers import (
    QPUSubproblemExternalEmbeddingSampler, QPUSubproblemAutoEmbeddingSampler,
    SimulatedAnnealingSubproblemSampler, RandomSubproblemSampler,
    TabuSubproblemSampler, TabuProblemSampler, InterruptableTabuSampler)
from hades.decomposers import (
    RandomSubproblemDecomposer, IdentityDecomposer,
    TilingChimeraDecomposer, EnergyImpactDecomposer)
from hades.composers import SplatComposer
from hades.core import State, SampleSet
from hades.flow import RacingBranches, ArgMinFold, SimpleIterator
from hades.utils import min_sample
from hades.profiling import tictoc


problems = chain(
    sorted(glob('problems/qbsolv/bqp100_*'))[:5],
    sorted(glob('problems/qbsolv/bqp2500_*'))[:5],
    sorted(glob('problems/random-chimera/2048*'))[:5],
    sorted(glob('problems/random-chimera/8192*'))[:5],
    sorted(glob('problems/ac3/*'))[:5],
)

solver_factories = [
    ("10 second Tabu",
        lambda bqm, **kw: TabuProblemSampler(bqm, timeout=10000)),

    ("10k sweeps Simulated Annealing",
        lambda bqm, **kw: IdentityDecomposer(bqm) | SimulatedAnnealingSubproblemSampler(sweeps=10000) | SplatComposer(bqm)),

    ("qbsolv-like solver",
        lambda bqm, qpu, **kw: SimpleIterator(RacingBranches(
            InterruptableTabuSampler(bqm, quantum_timeout=200),
            EnergyImpactDecomposer(bqm, max_size=50, min_diff=30)
            | QPUSubproblemAutoEmbeddingSampler(qpu_sampler=qpu)
            | SplatComposer(bqm)
        ) | ArgMinFold(), max_iter=100, convergence=10)),

    ("tiling chimera solver",
        lambda bqm, qpu, **kw: SimpleIterator(RacingBranches(
            InterruptableTabuSampler(bqm, quantum_timeout=200),
            TilingChimeraDecomposer(bqm, size=(16,16,4))
            | QPUSubproblemExternalEmbeddingSampler(qpu_sampler=qpu)
            | SplatComposer(bqm),
        ) | ArgMinFold(), max_iter=100, convergence=10)),
]


def run(problems, solver_factories):
    # reuse the cloud client
    qpu = DWaveSampler()

    for problem in problems:
        with open(problem) as fp:
            bqm = dimod.BinaryQuadraticModel.from_coo(fp)

        for name, factory in solver_factories:
            case = '{!r} with {!r}'.format(problem, name)

            try:
                solver = factory(bqm)
                init_state = State.from_sample(min_sample(bqm), bqm)

                with tictoc(case) as timer:
                    solution = solver.run(init_state).result()

            except Exception as exc:
                print("{case}: {exc!r}".format(**locals()))

            else:
                print("case={case!r}"
                      " energy={solution.samples.first.energy!r},"
                      " wallclock={timer.dt!r}".format(**locals()))


if __name__ == "__main__":
    run(problems, solver_factories)
