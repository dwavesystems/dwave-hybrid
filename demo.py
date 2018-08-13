#!/usr/bin/env python
"""
Kerberos prototype: runs N samplers in parallel.
Some samplers might me interruptable.
"""

import concurrent.futures
from operator import attrgetter

import dimod
from hades.samplers import (
    QPUSubproblemSampler, SimpleQPUSampler,
    TabuSubproblemSampler, TabuProblemSampler, InterruptableTabuSampler)
from hades.decomposers import RandomSubproblemDecomposer
from hades.composers import SplatComposer
from hades.core import State, Sample


problem = 'problems/random-chimera/2048.01.qubo'
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp, dimod.BINARY)


samplers = [
    InterruptableTabuSampler(bqm),
    #TabuProblemSampler(bqm, timeout=1),
    #RandomSubproblemDecomposer(bqm, size=400) | TabuSubproblemSampler(bqm, num_reads=1, timeout=500) | SplatComposer(bqm),
    #QPUSubproblemSampler(bqm, max_n=400, num_reads=200),
    RandomSubproblemDecomposer(bqm, size=400) | SimpleQPUSampler(bqm, num_reads=200) | SplatComposer(bqm)
]


max_iter = 10
best = State(Sample([0] * (max(bqm.linear.keys()) + 1)))

last = State(Sample({}, energy=1e100))
cnt = 10
for iterno in range(max_iter):
    branches = [sampler.run(best) for sampler in samplers]

    solutions = []
    for f in concurrent.futures.as_completed(branches):
        # as soon as one is done, stop all others
        for s in samplers:
            s.stop()
        solutions.append(f.result())

    best = min(solutions, key=attrgetter('sample.energy'))

    # debug info
    print("iterno={}, solutions:".format(iterno))
    for s in solutions:
        print("- energy={s.sample.energy}, debug={s.debug!r}".format(s=s))
    print("\nBEST: energy={s.sample.energy}, debug={s.debug!r}\n".format(s=best))

    if best.sample.energy >= last.sample.energy:
        cnt -= 1
    if cnt <= 0:
        break
    last = best
