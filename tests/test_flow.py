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
import copy
from concurrent import futures

import dimod

from hybrid.flow import (
    Branch, RacingBranches, ParallelBranches,
    ArgMin, Map, Reduce, Lambda, Unwind, TrackMin,
    LoopUntilNoImprovement, LoopWhileNoImprovement, SimpleIterator, Loop,
    Identity
)
from hybrid.core import State, States, Runnable, Present
from hybrid.utils import min_sample, max_sample
from hybrid.profiling import tictoc
from hybrid.exceptions import EndOfStream
from hybrid.testing import mock
from hybrid import traits


class TestBranch(unittest.TestCase):

    def test_empty(self):
        with self.assertRaises(ValueError):
            Branch()

    def test_composition(self):
        class A(Runnable):
            def next(self, state):
                return state.updated(x=state.x + 1)
        class B(Runnable):
            def next(self, state):
                return state.updated(x=state.x * 7)

        a, b = A(), B()
        s = State(x=1)

        b1 = Branch(components=(a, b))
        self.assertEqual(b1.components, (a, b))
        self.assertEqual(b1.run(s).result().x, (s.x + 1) * 7)

        b2 = b1 | b | a
        self.assertEqual(b2.components, (a, b, b, a))
        self.assertEqual(b2.run(s).result().x, (s.x + 1) * 7 * 7 + 1)

        with self.assertRaises(TypeError):
            b1 | 1

    def test_look_and_feel(self):
        class A(Runnable): pass
        class B(Runnable): pass

        b = A() | B()
        self.assertEqual(b.name, 'Branch')
        self.assertEqual(str(b), 'A | B')
        self.assertEqual(repr(b), 'Branch(components=(A(), B()))')
        self.assertEqual(tuple(b), b.components)
        self.assertIsInstance(b, Branch)
        self.assertIsInstance(b | b, Branch)

    def test_error_prop(self):
        class ErrorSilencer(Runnable):
            def next(self, state):
                return state
            def error(self, exc):
                return State(error=True)

        class Identity(Runnable):
            def next(self, state):
                return state

        branch = ErrorSilencer() | Identity()
        s1 = Present(exception=KeyError())
        s2 = branch.run(s1).result()

        self.assertEqual(s2.error, True)

    def test_stop(self):
        class Stoppable(Runnable):
            def init(self, state):
                self.stopped = False
            def next(self, state):
                return state
            def halt(self):
                self.stopped = True

        branch = Branch([Stoppable()])
        branch.run(State())
        branch.stop()

        self.assertTrue(next(iter(branch)).stopped)


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
            def next(self, state, **runopts):
                time.sleep(0.1)
                return state.updated(x=state.x + 1)

        class Slow(Runnable):
            def init(self, state, **runopts):
                self.time_to_stop = threading.Event()

            def next(self, state, **runopts):
                self.time_to_stop.wait()
                return state.updated(x=state.x + 2)

            def halt(self):
                self.time_to_stop.set()

        # default case
        rb = RacingBranches(Slow(), Fast(), Slow())
        res = rb.run(State(x=0)).result()
        self.assertEqual([s.x for s in res], [2, 1, 2])

        # "endomorphic case"
        rb = RacingBranches(Identity(), Slow(), Fast(), Slow())
        res = rb.run(State(x=0)).result()
        self.assertEqual([s.x for s in res], [0, 2, 1, 2])


class TestParallelBranches(unittest.TestCase):

    def test_look_and_feel(self):
        br = Runnable(), Runnable()
        pb = ParallelBranches(*br)
        self.assertEqual(pb.name, 'ParallelBranches')
        self.assertEqual(str(pb), '(Runnable) & (Runnable)')
        self.assertEqual(repr(pb), 'ParallelBranches(Runnable(), Runnable())')
        self.assertEqual(tuple(pb), br)

    def test_basic(self):
        class Fast(Runnable):
            def next(self, state):
                time.sleep(0.1)
                return state.updated(x=state.x + 1)

        class Slow(Runnable):
            def next(self, state):
                time.sleep(0.2)
                return state.updated(x=state.x + 2)

        # "endomorphic case"
        pb = ParallelBranches(Identity(), Slow(), Fast(), Slow())
        res = pb.run(State(x=0)).result()
        self.assertEqual([s.x for s in res], [0, 2, 1, 2])

        # default case
        pb = ParallelBranches(Slow(), Fast(), Slow())
        res = pb.run(State(x=0)).result()
        self.assertEqual([s.x for s in res], [2, 1, 2])

    def test_parallel_independent_execution(self):
        class Component(Runnable):
            def __init__(self, runtime):
                super(Component, self).__init__()
                self.runtime = runtime
            def next(self, state):
                time.sleep(self.runtime)
                return state

        # make sure all branches really run in parallel
        pb = ParallelBranches(
            Component(1), Component(1), Component(1), Component(1), Component(1))
        with tictoc() as tt:
            pb.run(State()).result()

        # total runtime has to be smaller that the sum of individual runtimes
        self.assertTrue(1 <= tt.dt <= 2)


class TestArgMin(unittest.TestCase):

    def test_look_and_feel(self):
        fold = ArgMin(key=False)
        self.assertEqual(fold.name, 'ArgMin')
        self.assertEqual(str(fold), '[]>')
        self.assertEqual(repr(fold), "ArgMin(key=False)")

        fold = ArgMin(key=min)
        self.assertEqual(repr(fold), "ArgMin(key=<built-in function min>)")

    def test_default_fold(self):
        bqm = dimod.BinaryQuadraticModel({'a': 1}, {}, 0, dimod.SPIN)
        states = States(
            State.from_sample(min_sample(bqm), bqm),    # energy: -1
            State.from_sample(max_sample(bqm), bqm),    # energy: +1
        )
        best = ArgMin().run(states).result()
        self.assertEqual(best.samples.first.energy, -1)

    def test_custom_fold(self):
        bqm = dimod.BinaryQuadraticModel({'a': 1}, {}, 0, dimod.SPIN)
        states = States(
            State.from_sample(min_sample(bqm), bqm),    # energy: -1
            State.from_sample(max_sample(bqm), bqm),    # energy: +1
        )
        fold = ArgMin(key=lambda s: -s.samples.first.energy)
        best = fold.run(states).result()
        self.assertEqual(best.samples.first.energy, 1)


class TestTrackMin(unittest.TestCase):

    def test_look_and_feel(self):
        tracker = TrackMin(key=False, output=False, output_key='a', input_key='a')
        self.assertEqual(tracker.name, 'TrackMin')
        self.assertEqual(str(tracker), 'TrackMin')
        self.assertEqual(repr(tracker), "TrackMin(key=False, output=False, input_key='a', output_key='a')")

    def test_default_tracking(self):
        """Best seen state is kept (default: state with sampleset with the lowest energy)"""

        bqm = dimod.BinaryQuadraticModel({'a': 1}, {}, 0, dimod.SPIN)
        min_state = State.from_sample(min_sample(bqm), bqm)    # energy: -1
        max_state = State.from_sample(max_sample(bqm), bqm)    # energy: +1

        tracker = TrackMin()
        _ = tracker.run(max_state).result()
        self.assertEqual(tracker.best.samples.first.energy, +1)
        _ = tracker.run(min_state).result()
        self.assertEqual(tracker.best.samples.first.energy, -1)
        _ = tracker.run(max_state).result()
        self.assertEqual(tracker.best.samples.first.energy, -1)

    def test_custom_key(self):
        """Custom key function works, here best state has the highest energy."""

        bqm = dimod.BinaryQuadraticModel({'a': 1}, {}, 0, dimod.SPIN)
        states = States(
            State.from_sample(min_sample(bqm), bqm),    # energy: -1
            State.from_sample(max_sample(bqm), bqm),    # energy: +1
        )

        tracker = TrackMin(key=lambda s: -s.samples.first.energy)
        for state in states:
            tracker.run(state).result()
        self.assertEqual(tracker.best.samples.first.energy, +1)

    def test_output(self):
        """Best state is properly output, for a custom key."""

        state1 = State(a=1, en=1)
        state2 = State(a=2, en=0)

        tracker = TrackMin(key=lambda s: s.en, output=True, input_key='a', output_key='best')
        result1 = tracker.run(state1).result()
        self.assertEqual(result1.best, 1)
        result2 = tracker.run(state2).result()
        self.assertEqual(result2.best, 2)


class TestLoopUntilNoImprovement(unittest.TestCase):

    def test_max_iter(self):
        class Inc(Runnable):
            def next(self, state):
                return state.updated(cnt=state.cnt + 1)

        # iterate for `max_iter`
        it = LoopUntilNoImprovement(Inc(), max_iter=100, convergence=1000, key=lambda _: None)
        s = it.run(State(cnt=0)).result()
        self.assertEqual(s.cnt, 100)

        # `key` function not needed if `convergence` undefined
        it = LoopUntilNoImprovement(Inc(), max_iter=100, convergence=None)
        s = it.run(State(cnt=0)).result()
        self.assertEqual(s.cnt, 100)

        # `convergence` not needed for simple finite loop
        it = LoopUntilNoImprovement(Inc(), max_iter=100)
        s = it.run(State(cnt=0)).result()
        self.assertEqual(s.cnt, 100)

    def test_convergence(self):
        class Inc(Runnable):
            def next(self, state):
                return state.updated(cnt=state.cnt + 1)

        it = LoopUntilNoImprovement(Inc(), max_iter=1000, convergence=100, key=lambda _: None)
        s = it.run(State(cnt=0)).result()

        self.assertEqual(s.cnt, 100)

    def test_timeout(self):
        class Inc(Runnable):
            def next(self, state):
                return state.updated(cnt=state.cnt + 1)

        # timeout after exactly two runs
        with mock.patch('time.time', side_effect=itertools.count(), create=True):
            loop = LoopUntilNoImprovement(Inc(), max_time=2)
            state = loop.run(State(cnt=0)).result()
            self.assertEqual(state.cnt, 2)

        # timeout after the second run
        with mock.patch('time.time', side_effect=itertools.count(), create=True):
            loop = LoopUntilNoImprovement(Inc(), max_time=2.5)
            state = loop.run(State(cnt=0)).result()
            self.assertEqual(state.cnt, 3)

    def test_finite_loop(self):
        class Inc(Runnable):
            def next(self, state):
                return state.updated(cnt=state.cnt + 1)

        it = Loop(Inc(), 10)
        s = it.run(State(cnt=0)).result()

        self.assertEqual(s.cnt, 10)

    def test_infinite_loop_stops_before_first_run(self):
        """An infinite loop can be preemptively stopped (before starting)."""

        class Inc(Runnable):
            def next(self, state):
                return state.updated(cnt=state.cnt + 1)

        loop = Loop(Inc())
        loop.stop()
        state = loop.run(State(cnt=0))

        self.assertEqual(state.result().cnt, 0)

    def test_infinite_loop_stops(self):
        """An infinite loop can be stopped after at least one iteration."""

        class Inc(Runnable):
            def __init__(self):
                super(Inc, self).__init__()
                self.first_run = threading.Event()

            def next(self, state):
                self.first_run.set()
                return state.updated(cnt=state.cnt + 1)

        loop = Loop(Inc())
        state = loop.run(State(cnt=0))

        # make sure loop body runnable is run at least once, then issue stop
        loop.runnable.first_run.wait(timeout=1)
        loop.stop()

        self.assertTrue(state.result().cnt >= 1)

    def test_infinite_loop_runs_after_stop(self):
        """An infinite loop can be started again after being stopped."""

        class Inc(Runnable):
            def __init__(self):
                super(Inc, self).__init__()
                self.first_run = threading.Event()

            def next(self, state):
                self.first_run.set()
                return state.updated(cnt=state.cnt + 1)

        loop = Loop(Inc())
        state1 = loop.run(State(cnt=0))

        # make sure loop body runnable is run at least once, then issue stop
        loop.runnable.first_run.wait(timeout=1)
        loop.stop()

        # check the state AND make sure next() finishes (see #111)
        self.assertTrue(state1.result().cnt >= 1)

        # reset our test event object
        loop.runnable.first_run.clear()

        # run loop again
        state2 = loop.run(State(cnt=0))

        # make sure loop body runnable is run at least once, then issue stop
        loop.runnable.first_run.wait(timeout=1)
        loop.stop()

        self.assertTrue(state2.result().cnt >= 1)

    def test_infinite_loop_over_interruptable_runnable(self):
        """Stop signal must propagate to child runnable."""

        class IntInc(Runnable):
            def __init__(self):
                super(IntInc, self).__init__()
                self.time_to_stop = threading.Event()
                self.first_run = threading.Event()

            def next(self, state):
                self.first_run.set()
                self.time_to_stop.wait()
                return state.updated(cnt=state.cnt + 1)

            def halt(self):
                self.time_to_stop.set()

        loop = Loop(IntInc())
        s = loop.run(State(cnt=0))

        # make sure loop body runnable is run at least once
        loop.runnable.first_run.wait()

        loop.stop()

        self.assertTrue(s.result().cnt >= 1)

    def test_validation(self):
        class simo(Runnable, traits.SIMO):
            def next(self, state):
                return States(state, state)

        with self.assertRaises(TypeError):
            LoopUntilNoImprovement(simo())


class TestLoopWhileNoImprovement(unittest.TestCase):

    def test_no_improvement_tries(self):
        class Inc(Runnable):
            def next(self, state):
                return state.updated(cnt=state.cnt + 1)

        loop = LoopWhileNoImprovement(Inc(), max_tries=10, key=lambda _: 0)
        state = loop.run(State(cnt=0)).result()

        self.assertEqual(len(loop.runnable.timers['dispatch.next']), 10)
        self.assertEqual(state.cnt, 1)

    def test_max_iter(self):
        class Inc(Runnable):
            def next(self, state):
                return state.updated(cnt=state.cnt + 1)

        loop = LoopWhileNoImprovement(Inc(), max_iter=5, max_tries=10, key=lambda _: 0)
        state = loop.run(State(cnt=0)).result()

        self.assertEqual(len(loop.runnable.timers['dispatch.next']), 5)
        self.assertEqual(state.cnt, 1)

    def test_infinite_loop_stops(self):
        """An infinite loop can be stopped after 10 iterations."""

        class Countdown(Runnable):
            """Countdown runnable that sets a semaphore on reaching zero."""

            def __init__(self):
                super(Countdown, self).__init__()
                self.ring = threading.Event()

            def next(self, state):
                output = state.updated(cnt=state.cnt - 1)
                if output.cnt <= 0:
                    self.ring.set()
                return output

        countdown = Countdown()
        loop = LoopWhileNoImprovement(countdown)
        state = loop.run(State(cnt=10))

        # stop only AFTER countdown reaches zero (10 iterations)
        # timeout in case Countdown failed before setting the flag
        countdown.ring.wait(timeout=1)
        loop.stop()

        self.assertTrue(state.result().cnt <= 0)

    def test_runs_with_improvement(self):
        class Inc(Runnable):
            def next(self, state):
                return state.updated(cnt=state.cnt + 1)

        loop = LoopWhileNoImprovement(Inc(), max_tries=100, key=lambda state: -min(3, state.cnt))
        state = loop.run(State(cnt=0)).result()

        self.assertEqual(len(loop.runnable.timers['dispatch.next']), 103)
        self.assertEqual(state.cnt, 4)

    def test_validation(self):
        class simo(Runnable, traits.SIMO):
            def next(self, state):
                return States(state, state)

        with self.assertRaises(TypeError):
            LoopWhileNoImprovement(simo())


class TestMap(unittest.TestCase):

    def test_isolated(self):
        class Inc(Runnable):
            def next(self, state):
                return state.updated(cnt=state.cnt + 1)

        states = States(State(cnt=1), State(cnt=2))
        result = Map(Inc()).run(states).result()

        self.assertEqual(result[0].cnt, states[0].cnt + 1)
        self.assertEqual(result[1].cnt, states[1].cnt + 1)

    def test_branch(self):
        class Inc(Runnable):
            def next(self, state):
                return state.updated(cnt=state.cnt + 1)

        states = States(State(cnt=1), State(cnt=2))
        branch = Map(Inc()) | ArgMin('cnt')
        result = branch.run(states).result()

        self.assertEqual(result.cnt, states[0].cnt + 1)

    def test_input_validation(self):
        with self.assertRaises(TypeError):
            Map(False)
        with self.assertRaises(TypeError):
            Map(lambda: None)
        with self.assertRaises(TypeError):
            Map(Runnable)
        self.assertIsInstance(Map(Runnable()), Runnable)


class TestReduce(unittest.TestCase):

    class Sum(Runnable, traits.MISO):
        def next(self, states):
            a, b = states
            return a.updated(val=a.val + b.val)

    def test_basic(self):
        states = States(State(val=1), State(val=2), State(val=3))
        result = Reduce(self.Sum()).run(states).result()

        self.assertIsInstance(result, State)
        self.assertEqual(result.val, 1+2+3)

    def test_initial_state(self):
        initial = State(val=10)
        states = States(State(val=1), State(val=2))
        result = Reduce(self.Sum(), initial_state=initial).run(states).result()

        self.assertEqual(result.val, 10+1+2)

    def test_unstructured_runnable(self):
        initial = State(val=10)
        states = States(State(val=2), State(val=3))

        multiply = Lambda(next=lambda self, s: s[0].updated(val=s[0].val * s[1].val))
        result = Reduce(multiply, initial_state=initial).run(states).result()

        self.assertEqual(result.val, 10*2*3)

    def test_input_validation(self):
        with self.assertRaises(TypeError):
            Reduce(False)
        with self.assertRaises(TypeError):
            Reduce(Runnable)
        self.assertIsInstance(Reduce(self.Sum()), Runnable)


class TestLambda(unittest.TestCase):

    def test_basic_runnable(self):
        runnable = Lambda(lambda _, s: s.updated(c=s.a * s.b))
        state = State(a=2, b=3)
        result = runnable.run(state).result()

        self.assertEqual(result.c, state.a * state.b)

    def test_error_and_init(self):
        runnable = Lambda(
            next=lambda self, state: state.updated(c=state.a * state.b),
            error=lambda self, exc: State(error=exc),
            init=lambda self, state: setattr(self, 'first', state.c)
        )

        # test init
        state = State(a=2, b=3, c=0)
        result = runnable.run(state).result()

        self.assertEqual(runnable.first, 0)
        self.assertEqual(result.c, state.a * state.b)

        # test error prop
        exc = ZeroDivisionError()
        result = runnable.run(Present(exception=exc)).result()

        self.assertEqual(result.error, exc)

    def test_map_lambda(self):
        states = States(State(cnt=1), State(cnt=2))
        result = Map(Lambda(lambda _, s: s.updated(cnt=s.cnt + 1))).run(states).result()

        self.assertEqual(result[0].cnt, states[0].cnt + 1)
        self.assertEqual(result[1].cnt, states[1].cnt + 1)

    def test_input_validation(self):
        with self.assertRaises(TypeError):
            Lambda(False)
        with self.assertRaises(TypeError):
            Lambda(lambda: None, False)
        with self.assertRaises(TypeError):
            Lambda(lambda: None, lambda: None, False)
        self.assertIsInstance(Lambda(lambda: None, lambda: None, lambda: None), Runnable)


class TestUnwind(unittest.TestCase):

    def test_basic(self):
        class Streamer(Runnable):
            def next(self, state, **runopts):
                if state.cnt <= 0:
                    raise EndOfStream
                return state.updated(cnt=state.cnt - 1)

        r = Unwind(Streamer())
        states = r.run(State(cnt=3)).result()

        # states should contain 3 states with cnt=3..0
        self.assertEqual(len(states), 3)
        for idx, state in enumerate(states):
            self.assertEqual(state.cnt, 2-idx)

    def test_validation(self):
        class simo(Runnable, traits.SIMO):
            def next(self, state):
                return States(state, state)

        with self.assertRaises(TypeError):
            Unwind(simo())


class TestIdentity(unittest.TestCase):

    def test_basic(self):
        ident = Identity()
        state = State(x=1, y='a', z=[1,2,3])

        inp = copy.deepcopy(state)
        out = ident.run(state).result()

        self.assertEqual(inp, out)
        self.assertFalse(out is inp)

    def test_interruptable(self):
        ident = Identity()
        state = State(x=1)
        out = ident.run(state, racing_context=True)

        # ident block should not finish
        done, not_done = futures.wait({out}, timeout=0.1)
        self.assertEqual(len(done), 0)
        self.assertEqual(len(not_done), 1)

        # until we stop it
        ident.stop()

        done, not_done = futures.wait({out}, timeout=0.1)
        self.assertEqual(len(done), 1)
        self.assertEqual(len(not_done), 0)

        self.assertEqual(out.result().x, 1)
