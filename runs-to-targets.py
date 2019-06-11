#!/usr/bin/env python
import os
import sys
import json
import plucky

all_data = json.load(sys.stdin)

targets = {}

for path, problem_data in all_data.items():
    problem_name = os.path.splitext(os.path.basename(path))[0]

    for solver, runs in problem_data.items():
        best_energy = min(plucky.pluck(runs, 'energy'))

        targets[problem_name] = best_energy

print(json.dumps(targets, indent=4))
