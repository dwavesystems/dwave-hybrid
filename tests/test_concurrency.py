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

import os
import unittest

import dimod
import hybrid
from hybrid.core import State
from hybrid.concurrency import Present, Future, ImmediateExecutor, immediate_executor
from hybrid.testing import RunTimeAssertionMixin
from hybrid.utils import cpu_count


class TestPresent(unittest.TestCase):

    def test_res(self):
        for val in 1, 'x', True, False, State(problem=1), lambda: None:
            f = Present(result=val)
            self.assertIsInstance(f, Future)
            self.assertTrue(f.done())
            self.assertEqual(f.result(), val)

    def test_exc(self):
        for exc in ValueError, KeyError, ZeroDivisionError:
            f = Present(exception=exc())
            self.assertIsInstance(f, Future)
            self.assertTrue(f.done())
            self.assertRaises(exc, f.result)

    def test_invalid_init(self):
        self.assertRaises(ValueError, Present)


class TestImmediateExecutor(unittest.TestCase):

    def test_submit_res(self):
        ie = ImmediateExecutor()
        f = ie.submit(lambda x: not x, True)
        self.assertIsInstance(f, Present)
        self.assertIsInstance(f, Future)
        self.assertEqual(f.result(), False)

    def test_submit_exc(self):
        ie = ImmediateExecutor()
        f = ie.submit(lambda: 1/0)
        self.assertIsInstance(f, Present)
        self.assertIsInstance(f, Future)
        self.assertRaises(ZeroDivisionError, f.result)


class TestMultithreading(unittest.TestCase, RunTimeAssertionMixin):

    def test_concurrent_tabu_samples(self):
        t1 = hybrid.TabuProblemSampler(timeout=1000)
        t2 = hybrid.TabuProblemSampler(timeout=2000)
        workflow = hybrid.Parallel(t1, t2)

        bqm = dimod.BinaryQuadraticModel({'a': 1}, {}, 0, 'BINARY')
        state = hybrid.State.from_problem(bqm)

        with self.assertRuntimeWithin(1900, 2500):
            workflow.run(state).result()

    @unittest.skipUnless(cpu_count() >= 2, "at least two threads required")
    def test_concurrent_sa_samples(self):
        s1 = hybrid.SimulatedAnnealingProblemSampler(num_reads=1000, num_sweeps=10000)
        s2 = hybrid.SimulatedAnnealingProblemSampler(num_reads=1000, num_sweeps=10000)
        p = hybrid.Parallel(s1, s2)

        bqm = dimod.BinaryQuadraticModel({'a': 1}, {}, 0, 'BINARY')
        state = hybrid.State.from_problem(bqm)

        def time_runnable(runnable, init):
            runnable.run(init).result()
            return sum(runnable.timers['dispatch.next'])

        t_s1 = time_runnable(s1, state)
        t_s2 = time_runnable(s2, state)
        t_p = time_runnable(p, state)

        # parallel execution must not be slower than the longest running branch + 75%
        # NOTE: the extremely weak upper bound was chosen so we don't fail on the
        # unreliable/inconsistent CI VMs, and yet to show some concurrency does exist
        t_expected_max = max(t_s1, t_s2) * 1.75

        self.assertLess(t_p, t_expected_max)
