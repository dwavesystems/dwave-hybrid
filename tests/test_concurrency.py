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

        with self.assertRuntimeWithin(2000, 2500):
            workflow.run(state).result()
