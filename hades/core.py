from collections import namedtuple


# TODO: abstract as singleton executor under hades namespace
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)


_Sample = namedtuple('_Sample', 'values energy')

class Sample(_Sample):
    """Sample namedtuple that includes optional energy, in addition to
    variable values."""

    def __new__(_cls, values, energy=None):
        return _Sample.__new__(_cls, values, energy)


# TODO: replace `sample` with population of `Sample`s
_State = namedtuple('State', 'sample ctx debug')

class State(_State):
    """Computation state passed along a branch between connected components.
    The structure is fixed, but fields are mutable. Components can store
    context into `ctx` and debugging/tracing info into `debug`.

    NB: Identical to _State namedtuple, with added default values/kwargs.
    """

    def __new__(_cls, sample=None, ctx=None, debug=None):
        """`sample` is Sample, `ctx` and `debug` are `dict`."""
        if ctx is None:
            ctx = {}
        if debug is None:
            debug = {}
        return _State.__new__(_cls, sample, ctx, debug)

    def updated(self, **kwargs):
        """Returns updated state. `sample` should be of type `Sample`, and
        `ctx`/`debug` dictionaries with items to add/update in state's
        `ctx`/`debug`."""
        self.ctx.update(kwargs.pop('ctx', {}))
        self.debug.update(kwargs.pop('debug', {}))
        return self._replace(**kwargs)


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
