import unittest

import dimod

from hades.core import PliableDict, State, SampleSet, HybridSampler
from hades.samplers import TabuProblemSampler


class TestSampleSet(unittest.TestCase):
    pass


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


class TestState(unittest.TestCase):

    def test_construction(self):
        self.assertDictEqual(State(), dict(samples=None, problem=None, debug={}))
        self.assertEqual(State(samples=[1]).samples, [1])
        self.assertEqual(State(problem={'a': 1}).problem, {'a': 1})
        self.assertEqual(State(debug={'a': 1}).debug, {'a': 1})

    def test_updated(self):
        a = SampleSet.from_sample([1,0,1], 'SPIN', 0)
        b = SampleSet.from_sample([0,1,0], 'SPIN', 0)
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



class TestHybridSampler(unittest.TestCase):

    def test_simple(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'ca': -1}, 0, dimod.SPIN)
        sampler = HybridSampler(TabuProblemSampler())
        response = sampler.sample(bqm)

        self.assertEqual(response.record[0].energy, -3.0)

    def test_validation(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': 1, 'ca': -1}, 0, dimod.SPIN)
        sampler = TabuProblemSampler()

        with self.assertRaises(TypeError):
            HybridSampler()

        with self.assertRaises(TypeError):
            HybridSampler(sampler).sample(1)

        with self.assertRaises(ValueError):
            HybridSampler(sampler).sample(bqm, initial_sample={1: 2})

        response = HybridSampler(sampler).sample(bqm, initial_sample={'a': 1, 'b': 1, 'c': 1})
        self.assertEqual(response.record[0].energy, -3.0)

        response = HybridSampler(sampler).sample(bqm, initial_sample={'a': -1, 'b': 1, 'c': -1})
        self.assertEqual(response.record[0].energy, -3.0)
