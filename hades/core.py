from collections import namedtuple


# TODO: abstract as singleton executor under hades namespace
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)


_BranchState = namedtuple('BranchState', 'sample energy source meta')

class BranchState(_BranchState):
    """Identical to _BranchState namedtuple, with added default values/kwargs."""

    def __new__(_cls, sample=None, energy=None, source=None, meta=None):
        return _BranchState.__new__(_cls, sample, energy, source, meta)


class Runnable(object):
    """Runnable component can be run for one iteration at a time. Iteration
    might be stopped, but implementing stop support is not required."""

    def __init__(self, *args, **kwargs):
        super(Runnable, self).__init__(*args, **kwargs)

    def iterate(self, state):
        """Accepts a state and returns a new state."""
        raise NotImplementedError

    def run(self, state):
        return executor.submit(self.iterate, state)

    def stop(self):
        pass

    def __or__(self, other):
        """Composition of runnable components (L-to-R) returns a new Runnable."""
        return Branch(components=(self, other))


class Branch(Runnable):
    def __init__(self, components, *args, **kwargs):
        super(Branch, self).__init__(*args, **kwargs)
        self._components = components

    def iterate(self, state):
        components = iter(self._components)
        state = next(components)(state).result()
        for component in components:
            state = component(state).result()
        return state
