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
import concurrent.futures
import logging
import itertools

import mock
import dimod
from tabu import TabuSampler

import hybrid
from hybrid.core import (
    PliableDict, State, States, SampleSet, Runnable, Branch, stoppable,
    HybridSampler, HybridRunnable, HybridProblemRunnable, HybridSubproblemRunnable
)
from hybrid.concurrency import Present, Future, immediate_executor
from hybrid.decomposers import IdentityDecomposer
from hybrid.composers import IdentityComposer
from hybrid.samplers import TabuProblemSampler
from hybrid.utils import min_sample, sample_as_dict
from hybrid.testing import isolated_environ, RunTimeAssertionMixin
from hybrid.exceptions import RunnableError


class TestPliableDict(unittest.TestCase):

    def test_construction(self):
        self.assertDictEqual(PliableDict(), {})
        self.assertDictEqual(PliableDict(x=1), {'x': 1})
        self.assertDictEqual(PliableDict(**{'x': 1}), {'x': 1})
        self.assertDictEqual(PliableDict({'x': 1, 'y': 2}), {'x': 1, 'y': 2})

    def test_setter(self):
        d = PliableDict()
        d.x = 1
        self.assertDictEqual(d, {'x': 1})

    def test_getter(self):
        d = PliableDict(x=1)
        self.assertEqual(d.x, 1)
        self.assertEqual(d.y, None)


class TestSampleSet(unittest.TestCase):

    def test_default(self):
        ss = dimod.SampleSet.from_samples([1], dimod.SPIN, [0])
        self.assertEqual(ss, SampleSet(ss.record, ss.variables, ss.info, ss.vartype))

    def test_empty(self):
        empty = SampleSet.empty()
        self.assertEqual(len(empty), 0)

        impliedempty = SampleSet()
        self.assertEqual(impliedempty, empty)

    def test_from_samples_bqm(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1}, 0, dimod.SPIN)
        ss = SampleSet.from_samples_bqm({'a': 1, 'b': -1}, bqm)
        self.assertEqual(ss.first.energy, -1)

    def test_hstack(self):
        a = SampleSet.from_samples(
                [{'a': 0, 'b': 1, 'c': 0}, {'a': 1, 'b': 0, 'c': 1}], vartype='BINARY', energy=[0, 1])
        b = SampleSet.from_samples(
                [{'d': 1, 'e': 0, 'f': 1}, {'d': 0, 'e': 1, 'f': 0}], vartype='BINARY', energy=[1, 0])
        c = SampleSet.from_samples(
                [{'d': -1, 'e': 1, 'f': 1}], vartype='SPIN', energy=0)

        m = a.hstack(b, c)

        self.assertEqual(len(m), 1)
        self.assertDictEqual(dict(m.first.sample), {'a': 0, 'b': 1, 'c': 0, 'd': 0, 'e': 1, 'f': 1})

    def test_vstack(self):
        a = SampleSet.from_samples(
                [{'a': 1, 'b': 0, 'c': 0}, {'a': 0, 'b': 1, 'c': 0}], vartype='BINARY', energy=[3, 1])
        b = SampleSet.from_samples(
                [{'a': 0, 'b': 0, 'c': 1}, {'a': 1, 'b': 0, 'c': 1}], vartype='BINARY', energy=[2, 0])
        c = SampleSet.from_samples(
                [{'a': -1, 'b': 1, 'c': -1}], vartype='SPIN', energy=4)

        m = a.vstack(b, c)

        self.assertEqual(len(m), 5)
        self.assertEqual(
            list(m.samples()),
            [
                {'a': 1, 'b': 0, 'c': 1},   # b[1], en=0
                {'a': 0, 'b': 1, 'c': 0},   # a[1], en=1
                {'a': 0, 'b': 0, 'c': 1},   # b[0], en=2
                {'a': 1, 'b': 0, 'c': 0},   # a[0], en=3
                {'a': 0, 'b': 1, 'c': 0}    # c[0] in BINARY, en=4
            ]
        )


class TestState(unittest.TestCase):

    def test_construction(self):
        self.assertDictEqual(State(), dict())
        self.assertEqual(State(samples=[1]).samples, [1])
        self.assertEqual(State(problem={'a': 1}).problem, {'a': 1})
        self.assertEqual(State(debug={'a': 1}).debug, {'a': 1})

    def test_from_samples(self):
        s1 = [0, 1]
        s2 = {0: 1, 1: 0}
        bqm = dimod.BinaryQuadraticModel({0: 1, 1: 2}, {}, 0.0, 'BINARY')
        self.assertEqual(State.from_sample(s1, bqm).samples.first.energy, 2.0)
        self.assertEqual(State.from_sample(s2, bqm).samples.first.energy, 1.0)
        self.assertEqual(State.from_sample(s1, bqm, beta=0.5).beta, 0.5)
        self.assertEqual(State.from_samples([s1, s1], bqm).samples.first.energy, 2.0)
        self.assertEqual(State.from_samples([s2, s2], bqm).samples.first.energy, 1.0)
        self.assertEqual(State.from_samples([s1, s1], bqm, beta=0.5).beta, 0.5)
        self.assertEqual(State.from_samples([sample_as_dict(s1), s2], bqm).samples.first.energy, 1.0)

    def test_from_subsamples(self):
        s1 = [0, 1]
        s2 = {0: 1, 1: 0}
        bqm = dimod.BinaryQuadraticModel({0: 1, 1: 2}, {}, 0.0, 'BINARY')
        self.assertEqual(State.from_subsample(s1, bqm).subsamples.first.energy, 2.0)
        self.assertEqual(State.from_subsample(s2, bqm).subsamples.first.energy, 1.0)
        self.assertEqual(State.from_subsample(s2, bqm, beta=0.5).beta, 0.5)
        self.assertEqual(State.from_subsamples([s1, s1], bqm).subsamples.first.energy, 2.0)
        self.assertEqual(State.from_subsamples([s2, s2], bqm).subsamples.first.energy, 1.0)
        self.assertEqual(State.from_subsamples([s2, s2], bqm, beta=0.5).beta, 0.5)
        self.assertEqual(State.from_subsamples([sample_as_dict(s1), s2], bqm).subsamples.first.energy, 1.0)

    def test_from_problem(self):
        sample = {0: 1, 1: 0}
        bqm = dimod.BinaryQuadraticModel({0: 1, 1: 2}, {}, 0.0, 'BINARY')
        self.assertEqual(State.from_problem(bqm).samples.first.energy, 0.0)
        self.assertEqual(State.from_problem(bqm, beta=0.5).beta, 0.5)
        self.assertEqual(State.from_problem(bqm, samples=hybrid.utils.max_sample).samples.first.energy, 3.0)
        self.assertEqual(State.from_problem(bqm, samples=sample), State.from_samples(sample, bqm))
        self.assertEqual(State.from_problem(bqm, samples=[sample]), State.from_samples([sample], bqm))

    def test_from_subproblem(self):
        subsample = {0: 1, 1: 0}
        bqm = dimod.BinaryQuadraticModel({0: 1, 1: 2}, {}, 0.0, 'BINARY')
        self.assertEqual(State.from_subproblem(bqm).subsamples.first.energy, 0.0)
        self.assertEqual(State.from_subproblem(bqm, beta=0.5).beta, 0.5)
        self.assertEqual(State.from_subproblem(bqm, subsamples=hybrid.utils.max_sample).subsamples.first.energy, 3.0)
        self.assertEqual(State.from_subproblem(bqm, subsamples=subsample), State.from_subsamples(subsample, bqm))
        self.assertEqual(State.from_subproblem(bqm, subsamples=[subsample]), State.from_subsamples([subsample], bqm))

    def test_updated(self):
        a = SampleSet.from_samples([1,0,1], 'SPIN', 0)
        b = SampleSet.from_samples([0,1,0], 'SPIN', 0)
        s1 = State(samples=a)
        s2 = State(samples=b, emb={'a': {'b': 1}}, debug={'x': 1})
        s3 = State(debug={'x': {'y': {'z': [1]}}})

        # test simple replace
        self.assertDictEqual(s1.updated(), s1)
        self.assertDictEqual(s1.updated(samples=b), State(samples=b))
        self.assertDictEqual(s2.updated(emb={'b': 1}).emb, {'b': 1})
        self.assertDictEqual(s1.updated(samples=b, debug=dict(x=1), emb={'a': {'b': 1}}), s2)

        # test recursive merge of `debug`
        self.assertDictEqual(s1.updated(debug=dict(x=1)).debug, {'x': 1})
        self.assertDictEqual(s2.updated(debug=dict(x=2)).debug, {'x': 2})
        self.assertDictEqual(s2.updated(debug=dict(y=2)).debug, {'x': 1, 'y': 2})
        self.assertDictEqual(s2.updated(debug=dict(y=2)).debug, {'x': 1, 'y': 2})

        self.assertDictEqual(s3.updated(debug={'x': {'y': {'z': [2]}}}).debug, {'x': {'y': {'z': [2]}}})
        self.assertDictEqual(s3.updated(debug={'x': {'y': {'w': 2}}}).debug, {'x': {'y': {'z': [1], 'w': 2}}})

        # test clear
        self.assertEqual(s2.updated(emb=None).emb, None)
        self.assertEqual(s2.updated(debug=None).debug, None)

    def test_copy(self):
        s1 = State(a=PliableDict(x=1))
        s2 = s1.copy()
        self.assertEqual(s1, s2)

        s1.a.x = 2
        self.assertNotEqual(s1, s2)
        self.assertNotEqual(id(s1), id(s2))
        self.assertEqual(s1.a.x, 2)
        self.assertEqual(s2.a.x, 1)


class TestStates(unittest.TestCase):

    def test_construction(self):
        self.assertEqual(States(1, 2), [1, 2])

    def test_state_api(self):
        states = States(State(x=1), State(y=1))

        self.assertEqual(states.result(), states)
        self.assertEqual(states.updated(x=2), States(State(x=2), State(x=2, y=1)))

    def test_first(self):
        self.assertEqual(States(1).first, 1)
        self.assertEqual(States(1, 2).first, 1)
        with self.assertRaises(IndexError):
            States().first


class TestRunnable(unittest.TestCase):

    def test_look_and_feel(self):
        r = Runnable()
        self.assertEqual(r.name, 'Runnable')
        self.assertEqual(str(r), 'Runnable')
        self.assertEqual(repr(r), 'Runnable()')
        self.assertEqual(tuple(r), tuple())
        self.assertRaises(NotImplementedError, r.next, State())
        self.assertIsNone(r.stop())
        self.assertIsInstance(r | r, Branch)

    def test_simple_run(self):
        r = Runnable()

        # async run with valid state
        f = r.run(State())
        self.assertIsInstance(f, Future)
        self.assertNotIsInstance(f, Present)
        self.assertRaises(NotImplementedError, f.result)

        # sync run with valid state
        f = r.run(State(), executor=immediate_executor)
        self.assertIsInstance(f, Present)
        self.assertRaises(NotImplementedError, f.result)

        # run with error state, check exc is propagated (default)
        f = r.run(Present(exception=ZeroDivisionError()))
        self.assertRaises(ZeroDivisionError, f.result)

        class MyRunnable(Runnable):
            def init(self, state):
                self.first = state.problem
            def next(self, state):
                return state.updated(problem=state.problem + 1)

        r = MyRunnable()
        s1 = State(problem=1)
        s2 = r.run(s1).result()

        self.assertEqual(r.first, s1.problem)
        self.assertEqual(s2.problem, s1.problem + 1)

    def test_error_prop(self):
        class MyRunnable(Runnable):
            def next(self, state):
                return state
            def error(self, exc):
                return State(error=True)

        r = MyRunnable()
        s1 = Present(exception=KeyError())
        s2 = r.run(s1).result()

        self.assertEqual(s2.error, True)

    def test_chaining(self):
        class Inc(Runnable):
            def next(self, state):
                return state.updated(x=state.x + 1)

        class Pow(Runnable):
            def __init__(self, exp):
                super(Pow, self).__init__()
                self.exp = exp

            def next(self, state):
                return state.updated(x=state.x ** self.exp)

        b = Inc() | Pow(3)

        s1 = State(x=1)
        s2 = b.run(s1).result()

        self.assertEqual(s2.x, (1 + 1) ** 3)

    def test_runopts_propagate(self):
        class Add(Runnable):
            def next(self, state, const):
                return state.updated(x=state.x + const)

        add = Add(const=8)
        s1 = State(x=5)
        s2 = add.run(s1).result()

        self.assertEqual(s2.x, s1.x + add.runopts['const'])

    def test_runopts_are_overridden(self):
        class Affine(Runnable):
            def init(self, state, scale, **kwargs):
                self.final_scale = scale

            def next(self, state, scale, offset):
                return state.updated(x=scale * state.x + offset)

        add = Affine(scale=3, offset=7)
        s1 = State(x=2)
        s2 = add.run(s1, scale=5).result()

        # scale is from run-time, offset is from construction-time
        self.assertEqual(s2.x, 5 * s1.x + 7)

        # init() should get the same
        self.assertEqual(add.final_scale, 5)


class TestStoppableDecorator(unittest.TestCase, RunTimeAssertionMixin):

    @stoppable
    class StoppableSleeper(Runnable):
        def next(self, state, timeout=None, **runopts):
            if timeout is not None:
                timeout /= 1000.0
            self.stop_signal.wait(timeout=timeout)
            return state

    def test_interface(self):
        sleeper = self.StoppableSleeper(timeout=100)
        state = State()

        with self.assertRuntimeWithin(0, 500):
            sleeper.run(state).result()

        with self.assertRuntimeWithin(0, 500):
            r = sleeper.run(state)
            sleeper.stop()
            r.result()

    def test_it_stops(self):
        sleeper = self.StoppableSleeper(timeout=None)
        state = State()

        # runs indefinitely
        r = sleeper.run(state)
        self.assertFalse(r.done())
        self.assertFalse(sleeper.stop_signal.is_set())

        # until stopped
        sleeper.stop()
        self.assertTrue(sleeper.stop_signal.is_set())

        # when it returns; but not before the decorator resets the stop signal
        r.result()
        self.assertFalse(sleeper.stop_signal.is_set())


class TestHybridSampler(unittest.TestCase):

    def test_workflow_runs(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 1}, {})
        sampler = HybridSampler(TabuProblemSampler())
        ss = sampler.sample(bqm)

        self.assertEqual(list(ss.samples()), [{'a': -1}])
        self.assertNotIn('state', ss.info)

    def test_return_state(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 1}, {})
        sampler = HybridSampler(TabuProblemSampler())
        ss = sampler.sample(bqm, return_state=True)

        self.assertEqual(list(ss.samples()), [{'a': -1}])
        self.assertIn('state', ss.info)
        self.assertEqual(ss.info['state'].problem, bqm)

    def test_sampling_params(self):

        # a simple runnable that optionally resets the output samples
        class Workflow(Runnable):
            def next(self, state, reset=False):
                if reset:
                    return state.updated(samples=hybrid.SampleSet.empty())
                return state

        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 1}, {})
        sampler = HybridSampler(Workflow())

        # control run
        ss = sampler.sample(bqm)
        self.assertEqual(len(list(ss.samples())), 1)

        # test runopts propagation
        ss = sampler.sample(bqm, reset=True)
        self.assertEqual(len(list(ss.samples())), 0)

    def test_validation(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 1}, {})
        sampler = TabuProblemSampler()

        with self.assertRaises(TypeError):
            HybridSampler()

        with self.assertRaises(TypeError):
            HybridSampler(1)

        with self.assertRaises(TypeError):
            HybridSampler(sampler).sample(1)

        with self.assertRaises(ValueError):
            HybridSampler(sampler).sample(bqm, initial_sample={1: 2})

        ss = HybridSampler(sampler).sample(bqm, initial_sample={'a': 1})
        self.assertEqual(list(ss.samples()), [{'a': -1}])

        ss = HybridSampler(sampler).sample(bqm, initial_sample={'a': -1})
        self.assertEqual(list(ss.samples()), [{'a': -1}])


class TestHybridRunnable(unittest.TestCase):
    bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'ca': -1}, 0, dimod.SPIN)
    init_state = State.from_sample(min_sample(bqm), bqm)

    def test_generic(self):
        runnable = HybridRunnable(TabuSampler(), fields=('problem', 'samples'))
        response = runnable.run(self.init_state)

        self.assertIsInstance(response, concurrent.futures.Future)
        self.assertEqual(response.result().samples.record[0].energy, -3.0)

    def test_validation(self):
        with self.assertRaises(TypeError):
            HybridRunnable(1, 'ab')

        with self.assertRaises(ValueError):
            HybridRunnable(TabuSampler(), None)

        with self.assertRaises(ValueError):
            HybridRunnable(TabuSampler(), ('a'))

        self.assertIsInstance(HybridRunnable(TabuSampler(), 'ab'), HybridRunnable)
        self.assertIsInstance(HybridRunnable(TabuSampler(), ('a', 'b')), HybridRunnable)
        self.assertIsInstance(HybridRunnable(TabuSampler(), ['a', 'b']), HybridRunnable)

    def test_problem_sampler_runnable(self):
        runnable = HybridProblemRunnable(TabuSampler())
        response = runnable.run(self.init_state)

        self.assertIsInstance(response, concurrent.futures.Future)
        self.assertEqual(response.result().samples.record[0].energy, -3.0)

    def test_subproblem_sampler_runnable(self):
        runnable = HybridSubproblemRunnable(TabuSampler())
        state = self.init_state.updated(subproblem=self.bqm)
        response = runnable.run(state)

        self.assertIsInstance(response, concurrent.futures.Future)
        self.assertEqual(response.result().subsamples.record[0].energy, -3.0)

    def test_runnable_composition(self):
        runnable = IdentityDecomposer() | HybridSubproblemRunnable(TabuSampler()) | IdentityComposer()
        response = runnable.run(self.init_state)

        self.assertIsInstance(response, concurrent.futures.Future)
        self.assertEqual(response.result().samples.record[0].energy, -3.0)


class TestLogging(unittest.TestCase):

    def test_init(self):
        self.assertTrue(hasattr(logging, 'TRACE'))
        self.assertEqual(getattr(logging, 'TRACE'), 5)

        logger = logging.getLogger(__name__)
        self.assertTrue(callable(logger.trace))

    def test_loglevel_from_env(self):
        logger = logging.getLogger(__name__)

        def ll_check(env, name):
            with isolated_environ(remove_dwave=True):
                os.environ[env] = name
                hybrid._apply_loglevel_from_env(logger)
                self.assertEqual(logger.getEffectiveLevel(), getattr(logging, name.upper()))

        levels = ('trace', 'debug', 'info', 'warning', 'error', 'critical')
        combinations = itertools.product(
            ['DWAVE_HYBRID_LOG_LEVEL', 'dwave_hybrid_log_level'],
            itertools.chain((l.lower() for l in levels), (l.upper() for l in levels)))

        for env, name in combinations:
            ll_check(env, name)

    def test_trace_logging(self):
        logger = logging.getLogger(__name__)
        hybrid._create_trace_loglevel(logging)  # force local _trace override

        with isolated_environ(remove_dwave=True):
            # trace on
            os.environ['DWAVE_HYBRID_LOG_LEVEL'] = 'trace'
            hybrid._apply_loglevel_from_env(logger)
            with mock.patch.object(logger, '_log') as m:
                logger.trace('test')
                m.assert_called_with(logging.TRACE, 'test', ())

            # trace off
            os.environ['DWAVE_HYBRID_LOG_LEVEL'] = 'debug'
            hybrid._apply_loglevel_from_env(logger)
            with mock.patch.object(logger, '_log') as m:
                logger.trace('test')
                self.assertFalse(m.called)


class TestExceptions(unittest.TestCase):

    def test_init(self):
        self.assertEqual(RunnableError('msg', 1).state, 1)
