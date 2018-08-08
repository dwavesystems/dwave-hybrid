#!/usr/bin/env python
"""
Kerberos prototype: runs three samplers (generally n) in parallel.
"""

import concurrent.futures
from operator import attrgetter

import dimod
from hades.samplers import (
    QPUSubproblemSampler, TabuSubproblemSampler, TabuProblemSampler)
from hades.core import BranchState


problem = 'problems/random-chimera/2048.01.qubo'
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp, dimod.BINARY)


samplers = [
    TabuProblemSampler(bqm, timeout=1000),
    TabuSubproblemSampler(bqm, max_n=400, num_reads=1, timeout=500),
    QPUSubproblemSampler(bqm, max_n=400, num_reads=200),
]


max_iter = 10
best = BranchState([0] * (max(bqm.linear.keys()) + 1))

for iterno in range(max_iter):
    branches = [sampler.run(best) for sampler in samplers]
    states = [f.result() for f in concurrent.futures.as_completed(branches)]
    best = min(states, key=attrgetter('energy'))

    # debug info
    print("iterno={}, states:".format(iterno))
    for s in states:
        print("- energy={s.energy}, source={s.source!r}, meta={s.meta}".format(s=s))
    print("\nBEST: energy={s.energy}, source={s.source!r}, meta={s.meta}\n".format(s=best))
