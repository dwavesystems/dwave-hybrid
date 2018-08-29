#!/usr/bin/env python
from os.path import dirname
import sys
sys.path.insert(0, dirname(dirname(__file__)))

import dimod
from kerberos import KerberosSampler

problem = sys.argv[1]
with open(problem) as fp:
    bqm = dimod.BinaryQuadraticModel.from_coo(fp)

solution = KerberosSampler().sample(bqm, max_iter=10, convergence=3)

print("Solution: {!r}".format(solution.record))
