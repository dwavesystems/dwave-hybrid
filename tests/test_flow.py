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

import time
import itertools
import unittest
import threading
import operator

import dimod

from hybrid.flow import RacingBranches, ArgMinFold, SimpleIterator
from hybrid.core import State, Runnable, Present
from hybrid.utils import min_sample, max_sample


class TestRacingBranches(unittest.TestCase):

    def test_look_and_feel(self):
        br = Runnable(), Runnable()
        rb = RacingBranches(*br)
        self.assertEqual(rb.name, 'RacingBranches')
        self.assertEqual(str(rb), '(Runnable) !! (Runnable)')
        self.assertEqual(repr(rb), 'RacingBranches(Runnable(), Runnable())')
        self.assertEqual(tuple(rb), br)

    def test_stopped(self):
        class Fast(Runnable):
            def next(self, state):
                time.sleep(0.1)
                return state.updated(x=state.x + 1)

        class Slow(Runnable):
            def init(self, state):
                self.time_to_stop = threading.Event()

            def next(self, state):
                self.time_to_stop.wait()
                return state.updated(x=state.x + 2)

            def stop(self):
                self.time_to_stop.set()

        # standard case
        rb = RacingBranches(Slow(), Fast(), Slow())
        res = rb.run(State(x=0)).result()
        self.assertEqual([s.x for s in res], [0, 2, 1, 2])

        # branches' outputs are of a different type that the inputs
        # (i.e. non-endomorphic racing branches)
        rb = RacingBranches(Slow(), Fast(), Slow(), endomorphic=False)
        res = rb.run(State(x=0)).result()
        self.assertEqual([s.x for s in res], [2, 1, 2])


class TestArgMinFold(unittest.TestCase):

    def test_look_and_feel(self):
        fold = ArgMinFold(key=False)
        self.assertEqual(fold.name, 'ArgMinFold')
        self.assertEqual(str(fold), '[]>')
        self.assertEqual(repr(fold), "ArgMinFold(key=False)")

        fold = ArgMinFold(key=min)
        self.assertEqual(repr(fold), "ArgMinFold(key=<built-in function min>)")

    def test_default_fold(self):
        bqm = dimod.BinaryQuadraticModel({'a': 1}, {}, 0, dimod.SPIN)
        states = [
            State.from_sample(min_sample(bqm), bqm),    # energy: -1
            State.from_sample(max_sample(bqm), bqm),    # energy: +1
        ]
        best = ArgMinFold().run(Present(result=states)).result()
        self.assertEqual(best.samples.first.energy, -1)

    def test_custom_fold(self):
        bqm = dimod.BinaryQuadraticModel({'a': 1}, {}, 0, dimod.SPIN)
        states = [
            State.from_sample(min_sample(bqm), bqm),    # energy: -1
            State.from_sample(max_sample(bqm), bqm),    # energy: +1
        ]
        fold = ArgMinFold(key=lambda s: -s.samples.first.energy)
        best = fold.run(Present(result=states)).result()
        self.assertEqual(best.samples.first.energy, 1)


class TestSimpleIterator(unittest.TestCase):

    def test_basic(self):
        class Inc(Runnable):
            def next(self, state):
                return state.updated(cnt=state.cnt + 1)

        it = SimpleIterator(Inc(), max_iter=100, convergence=100, key=lambda _: None)
        s = it.run(State(cnt=0)).result()

        self.assertEqual(s.cnt, 100)
