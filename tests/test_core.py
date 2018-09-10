import unittest

from hades.core import State, SampleSet


class TestSampleSet(unittest.TestCase):
    pass


class TestState(unittest.TestCase):

    def test_construction(self):
        self.assertEqual(State()._asdict(), dict(samples=None, ctx={}, debug={}))
        self.assertEqual(State([1]).samples, [1])
        self.assertEqual(State(ctx={'a': 1}).ctx, {'a': 1})
        self.assertEqual(State(debug={'a': 1}).debug, {'a': 1})

    def test_update(self):
        a = SampleSet.from_sample([1,0,1], 'SPIN', 0)
        b = SampleSet.from_sample([0,1,0], 'SPIN', 0)
        s1 = State(a)
        s2 = State(b, ctx={'a': {'b': 1}})

        # test simple replace
        self.assertEqual(s1.updated(), s1)
        self.assertEqual(s2.updated(ctx={}).ctx, {'a': {'b': 1}})
        self.assertEqual(s2.updated(ctx=None).ctx, {})
        self.assertEqual(s1.updated(ctx='test'), s1._replace(ctx='test'))
        self.assertEqual(s1.updated(ctx={'a': 1}), s1._replace(ctx={'a': 1}))

        # test recursive merge
        self.assertEqual(s2.updated(), s2)
        self.assertEqual(s2.updated(samples=[1,1,1]).samples, [1,1,1])
        self.assertEqual(s2.updated(ctx=None).updated(ctx={'x': 1}).ctx, {'x': 1})
        self.assertEqual(s2.updated(ctx={'a': 2}).ctx, {'a': 2})
        self.assertEqual(s2.updated(ctx={'a': {'b': 2}}).ctx, {'a': {'b': 2}})
        self.assertEqual(s2.updated(ctx={'a': {'c': 2}}).ctx, {'a': {'c': 2}})
