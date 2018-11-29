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

from hybrid.core import Runnable, Branch, State
from hybrid.flow import RacingBranches, ArgMinFold, SimpleIterator
from hybrid.profiling import tictoc, iter_inorder, walk_inorder
from hybrid.testing import mock
from hybrid.composers import *
from hybrid.samplers import *
from hybrid.decomposers import *


class TestCoreRunnablesIterable(unittest.TestCase):

    class RunnableA(Runnable): pass
    class RunnableB(Runnable): pass

    @staticmethod
    def children(runnable):
        return [str(x) for x in runnable]

    def test_runnable(self):
        self.assertEqual(list(Runnable()), [])

    def test_branch(self):
        # explicit branch construction
        self.assertEqual(self.children(Branch(components=(self.RunnableA(),))), ['RunnableA'])
        # implicit + order
        self.assertEqual(self.children(self.RunnableA() | self.RunnableB()), ['RunnableA', 'RunnableB'])

    def test_racingbranches(self):
        rb = RacingBranches(self.RunnableA(), self.RunnableB())
        self.assertEqual(self.children(rb), ['RunnableA', 'RunnableB'])

    def test_argminfold(self):
        self.assertEqual(self.children(ArgMinFold()), [])

    def test_simpleiterator(self):
        r = SimpleIterator(self.RunnableA())
        self.assertEqual(self.children(r), ['RunnableA'])

    def test_concrete_runnables(self):
        # composers
        self.assertEqual(self.children(IdentityComposer()), [])
        self.assertEqual(self.children(SplatComposer()), [])
        # sample of samplers
        self.assertEqual(self.children(QPUSubproblemAutoEmbeddingSampler(qpu_sampler=False)), [])
        self.assertEqual(self.children(SimulatedAnnealingSubproblemSampler()), [])
        self.assertEqual(self.children(TabuProblemSampler()), [])
        self.assertEqual(self.children(InterruptableTabuSampler()), [])
        # sample of decomposers
        self.assertEqual(self.children(IdentityDecomposer()), [])
        self.assertEqual(self.children(EnergyImpactDecomposer(max_size=1)), [])
        self.assertEqual(self.children(RandomSubproblemDecomposer(size=1)), [])


class TestTictoc(unittest.TestCase):

    def test_ctx_mgr(self):
        with mock.patch('hybrid.profiling.perf_counter', side_effect=[0, 1]):
            with tictoc() as t:
                pass
            self.assertEqual(t.tick, 0)
            self.assertEqual(t.dt, 1)

    def test_decorator(self):
        with mock.patch('hybrid.profiling.perf_counter', side_effect=[0, 1]):
            def f():
                pass
            deco = tictoc('f')
            ff = deco(f)
            ff()

            self.assertEqual(deco.tick, 0)
            self.assertEqual(deco.dt, 1)


class TestRunnableWalkers(unittest.TestCase):

    def test_iter_walk(self):
        flow = SimpleIterator(RacingBranches(Runnable(), Runnable()) | ArgMinFold())
        names = [r.name for r in iter_inorder(flow)]
        self.assertEqual(names, ['SimpleIterator', 'Branch', 'RacingBranches', 'Runnable', 'Runnable', 'ArgMinFold'])

    def test_callback_walk(self):
        flow = SimpleIterator(RacingBranches(Runnable(), Runnable()) | ArgMinFold())
        names = []
        walk_inorder(flow, visit=lambda r, _: names.append(r.name))
        self.assertEqual(names, ['SimpleIterator', 'Branch', 'RacingBranches', 'Runnable', 'Runnable', 'ArgMinFold'])


class TestCounters(unittest.TestCase):

    def test_counter_called(self):
        class Ident(Runnable):
            def next(self, state):
                with self.count('my-counter'):
                    return state

        r = Ident()
        r.run(State()).result()

        self.assertEqual(len(r.counters['my-counter']), 1)
