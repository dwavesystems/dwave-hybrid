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

from dwave.system.testing import MockDWaveSampler

from hybrid.core import Runnable, State
from hybrid.flow import Branch, RacingBranches, ArgMin, Loop
from hybrid.profiling import tictoc, iter_inorder, walk_inorder, make_timeit, make_count
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

    def test_argmin(self):
        self.assertEqual(self.children(ArgMin()), [])

    def test_loop(self):
        r = Loop(self.RunnableA())
        self.assertEqual(self.children(r), ['RunnableA'])

    def test_concrete_runnables(self):
        # composers
        self.assertEqual(self.children(IdentityComposer()), [])
        self.assertEqual(self.children(SplatComposer()), [])
        # sample of samplers
        self.assertEqual(self.children(QPUSubproblemAutoEmbeddingSampler(qpu_sampler=MockDWaveSampler())), [])
        self.assertEqual(self.children(SimulatedAnnealingSubproblemSampler()), [])
        self.assertEqual(self.children(TabuProblemSampler()), [])
        # sample of decomposers
        self.assertEqual(self.children(IdentityDecomposer()), [])
        self.assertEqual(self.children(EnergyImpactDecomposer(size=1)), [])
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
        flow = Loop(RacingBranches(Runnable(), Runnable()) | ArgMin())
        names = [r.name for r in iter_inorder(flow)]
        self.assertEqual(names, ['Loop', 'Branch', 'RacingBranches', 'Runnable', 'Runnable', 'ArgMin'])

    def test_callback_walk(self):
        flow = Loop(RacingBranches(Runnable(), Runnable()) | ArgMin())
        names = []
        walk_inorder(flow, visit=lambda r, _: names.append(r.name))
        self.assertEqual(names, ['Loop', 'Branch', 'RacingBranches', 'Runnable', 'Runnable', 'ArgMin'])


class TestTimers(unittest.TestCase):

    def test_basic(self):
        timers = {}
        time = make_timeit(timers)
        with time('a'):
            _ = 1

        self.assertSetEqual(set(timers.keys()), {'a'})
        self.assertEqual(len(timers['a']), 1)

    def test_nested_timers(self):
        timers = {}
        time = make_timeit(timers)
        with time('a'):
            with time('b'):
                with time('c'):
                    with time('b'):
                        _ = 2 ** 50

        self.assertSetEqual(set(timers.keys()), {'a', 'b', 'c'})
        self.assertEqual(len(timers['a']), 1)
        self.assertEqual(len(timers['b']), 2)
        self.assertEqual(len(timers['c']), 1)

    def test_runnable_timer_called(self):
        class Ident(Runnable):
            def next(self, state):
                with self.timeit('my-timer'):
                    return state

        r = Ident()
        r.run(State()).result()

        self.assertEqual(len(r.timers['my-timer']), 1)

    def test_runnable_default_timer_value(self):
        self.assertEqual(Runnable().timers['my-timer'], [])

class TestCounters(unittest.TestCase):

    def test_basic(self):
        counters = {}
        count = make_count(counters)

        count('a')
        try:
            raise ZeroDivisionError
        except:
            count('b')
        finally:
            count('a', inc=3)

        self.assertSetEqual(set(counters.keys()), {'a', 'b'})
        self.assertEqual(counters['a'], 4)
        self.assertEqual(counters['b'], 1)

    def test_runnable_counter_called(self):
        class Ident(Runnable):
            def next(self, state):
                self.count('my-counter', 3)
                self.count('your-counter')
                return state

        r = Ident()
        r.run(State()).result()

        self.assertEqual(r.counters['my-counter'], 3)
        self.assertEqual(r.counters['your-counter'], 1)

    def test_runnable_default_counter_value(self):
        self.assertEqual(Runnable().counters['my-counter'], 0)
