#!/usr/bin/env python
"""
Kerberos prototype: runs N samplers in parallel.
Some samplers might me interruptable.
"""

import concurrent.futures
from operator import attrgetter

import dimod
from hades.samplers import (
    QPUSubproblemSampler, TabuSubproblemSampler, TabuProblemSampler, InterruptableTabuSampler)
from hades.core import BranchState


problem = 'problems/random-chimera/2048.01.qubo'
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp, dimod.BINARY)


samplers = [
    InterruptableTabuSampler(bqm),
    TabuProblemSampler(bqm, timeout=1000),
    TabuSubproblemSampler(bqm, max_n=400, num_reads=1, timeout=500),
    QPUSubproblemSampler(bqm, max_n=400, num_reads=200),
]


max_iter = 100
best = BranchState([0] * (max(bqm.linear.keys()) + 1))

last = BranchState(energy=1e100)
cnt = 3
for iterno in range(max_iter):
    branches = [sampler.run(best) for sampler in samplers]

    solutions = []
    for f in concurrent.futures.as_completed(branches):
        # as soon as one is done, stop all others
        for s in samplers:
            s.stop()
        solutions.append(f.result())

    best = min(solutions, key=attrgetter('energy'))

    # debug info
    print("iterno={}, solutions:".format(iterno))
    for s in solutions:
        print("- energy={s.energy}, source={s.source!r}, meta={s.meta}".format(s=s))
    print("\nBEST: energy={s.energy}, source={s.source!r}, meta={s.meta}\n".format(s=best))

    if best.energy >= last.energy:
        cnt -= 1
    if cnt <= 0:
        break
    last = best
