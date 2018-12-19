# Copyright 2018 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import dimod
from neal import SimulatedAnnealingSampler
from dwave.system.testing import MockDWaveSampler

from hybrid.samplers import *
from hybrid.core import State


class TestQPUSamplers(unittest.TestCase):

    def test_unstructured_child_sampler(self):
        q = QPUSubproblemAutoEmbeddingSampler(qpu_sampler=SimulatedAnnealingSampler())

        # test sampler stays unstructured
        self.assertFalse(isinstance(q.sampler, dimod.Structured))

        # test sampling works
        bqm = dimod.BinaryQuadraticModel({'a': 1}, {}, 0, 'SPIN')
        init = State.from_subsample({'a': 1}, bqm)
        res = q.run(init).result()
        self.assertEqual(res.subsamples.first.energy, -1)

    def test_structured_child_sampler(self):
        q = QPUSubproblemAutoEmbeddingSampler(qpu_sampler=MockDWaveSampler())

        # test sampler is converted to unstructured
        self.assertFalse(isinstance(q.sampler, dimod.Structured))

        # test sampling works
        bqm = dimod.BinaryQuadraticModel({'a': 1}, {}, 0, 'SPIN')
        init = State.from_subsample({'a': 1}, bqm)
        res = q.run(init).result()
        self.assertEqual(res.subsamples.first.energy, -1)
