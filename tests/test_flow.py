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
from hybrid.core import State, Runnable
from hybrid.utils import min_sample


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
