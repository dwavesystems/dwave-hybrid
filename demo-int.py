#!/usr/bin/env python
"""
Demo of using an interruptable solver in combination with variable-running QPU solver.
"""

import concurrent.futures
from operator import attrgetter

import dimod
from hades.samplers import (
    Solution, QPUSubproblemSampler, TabuSubproblemSampler, TabuProblemSampler, InterruptableTabuSampler)


problem = 'problems/random-chimera/2048.01.qubo'
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp, dimod.BINARY)


samplers = [
    InterruptableTabuSampler(bqm),
    QPUSubproblemSampler(bqm, max_n=100, num_reads=100),
]


max_iter = 10
best = Solution([0] * (max(bqm.linear.keys()) + 1))

for iterno in range(max_iter):
    branches = [sampler.run(best.sample) for sampler in samplers]

    solutions = []
    for f in concurrent.futures.as_completed(branches):
        # as soon as one is done (QPU), stop all others (main tabu)
        for s in samplers:
            s.stop()
        solutions.append(f.result())

    best = min(solutions, key=attrgetter('energy'))

    # debug info
    print("iterno={}, solutions:".format(iterno))
    for s in solutions:
        print("- energy={s.energy}, source={s.source!r}, meta={s.meta}".format(s=s))
    print("\nBEST: energy={s.energy}, source={s.source!r}, meta={s.meta}\n".format(s=best))
