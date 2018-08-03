#!/usr/bin/env python
"""
Kerberos prototype: runs three samplers (generally n) in parallel.
"""

import concurrent.futures
from operator import attrgetter

import dimod
from hades.samplers import (
    Solution, QPUSubproblemSampler, TabuSubproblemSampler, TabuProblemSampler)


problem = 'problems/random-chimera/2048.01.qubo'
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp, dimod.BINARY)


samplers = [
    TabuProblemSampler(bqm, timeout=1000),
    TabuSubproblemSampler(bqm, max_n=100, num_reads=1, timeout=1000),
    QPUSubproblemSampler(bqm, max_n=100, num_reads=100),
]


max_iter = 10
best = Solution([0] * (max(bqm.linear.keys()) + 1))

for iterno in range(max_iter):
    branches = [sampler.run(best.sample) for sampler in samplers]
    solutions = [f.result() for f in concurrent.futures.as_completed(branches)]
    best = min(solutions, key=attrgetter('energy'))
    print("iterno={}, energy={s.energy}, source={s.source!r}".format(iterno, s=best))
