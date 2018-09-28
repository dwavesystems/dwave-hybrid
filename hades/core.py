# Copyright 2018 D-Wave Systems Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

from collections import namedtuple
from itertools import chain
from copy import deepcopy
import operator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, Executor

from plucky import merge
import dimod


class ImmediateExecutor(Executor):

    def submit(self, fn, *args, **kwargs):
        """Blocking version of `Executor.submit()`. Returns resolved `Future`."""
        # TODO: (re)combine with our global async_executor object, probably introduce
        # customizable underlying executor (e.g. thread/process/celery/network)
        try:
            return Present(result=fn(*args, **kwargs))
        except Exception as exc:
            return Present(exception=exc)


# TODO: abstract and make customizable to support other types of executors
async_executor = ThreadPoolExecutor(max_workers=4)
immediate_executor = ImmediateExecutor()


class SampleSet(dimod.Response):
    """Set of samples and any other data returned by dimod samplers.

    Args:
        record (:obj:`numpy.recarray`)
            Samples and data as a NumPy record array. The 'sample', 'energy' and 'num_occurrences'
            fields are required. 'sample' field is a 2D NumPy int8 array where each row is a
            sample and each column represents the value of a variable.
        labels (list):
            List of variable labels.
        info (dict):
            Information about the response as a whole formatted as a dict.
        vartype (:class:`.Vartype`/str/set):
            Variable type for the response. Accepted input values:

            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

    """

    def __eq__(self, other):
        # TODO: merge into dimod.Response together with the updated docstring
        return (self.vartype == other.vartype and self.info == other.info
            and self.variable_labels == other.variable_labels
            and self.record == other.record)

    @property
    def first(self):
        """Return the `Sample(sample={...}, energy, num_occurrences)` with
        lowest energy.
        """
        # TODO: merge into dimod.Response
        return next(self.data(sorted_by='energy', name='Sample'))

    @classmethod
    def from_sample(cls, sample, vartype, energy=None, num_occurrences=1):
        """Convenience method for constructing a SampleSet from one raw (dict)
        sample.
        """
        return cls.from_samples(
            samples_like=[sample],
            vectors={'energy': [energy], 'num_occurrences': [num_occurrences]},
            info={}, vartype=vartype)

    @classmethod
    def from_sample_on_bqm(cls, sample, bqm):
        """Convenience method for constructing a SampleSet from one raw (dict)
        sample with energy calculated from the BQM.
        """
        return cls.from_sample(sample, bqm.vartype, bqm.energy(sample))

    @classmethod
    def from_response(cls, response):
        """Convenience method for constructing a SampleSet from a dimod response.
        """
        return cls.from_future(response, result_hook=lambda x: x)


class PliableDict(dict):
    """Dictionary subclass with attribute accessors acting as item accessors.

    Example:

        >>> d = PliableDict(x=1)
        >>> d.y = 2
        >>> d
        {'x': 1, 'y': 2}

        >>> d.z is None
        True
    """

    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, value):
        self[name] = value


class State(PliableDict):
    """Computation state passed along a branch between connected components."""

    def __init__(self, *args, **kwargs):
        """State is a `PliableDict` (`dict` subclass) which always contains
        at least three keys: `samples`, `problem` and `debug`.
        """
        super(State, self).__init__(*args, **kwargs)
        self.setdefault('samples', None)
        self.setdefault('problem', None)
        self.setdefault('debug', PliableDict())

    def copy(self):
        """Simple deep copy if itself. Functionally identical to
        `State.updated()`.
        """
        return deepcopy(self)

    def updated(self, **kwargs):
        """Return a (deep) copy of itself, updated from `kwargs`.

        It has `dict.update` semantics with immutability of `sorted`. One
        exception (currently) is for the `debug` key, for which we do a
        depth-unlimited recursive merge.

        Example:

            >>> state = State()
            >>> state
            {'debug': {}, 'problem': None, 'samples': None}

            >>> newstate = state.updated(problem="test")
            >>> newstate
            {'debug': {}, 'problem': 'test', 'samples': None}

        """

        overwrite = lambda a,b: b

        # use a special merge strategy for `debug` (op=overwrite, max_depth=None)
        debug = merge(self.get('debug', {}), kwargs.get('debug', {}), op=overwrite)
        if debug is not None:
            kwargs['debug'] = PliableDict(debug)

        return State(merge(self, kwargs, max_depth=1, op=overwrite))

    def result(self):
        """Implement `concurrent.Future`-compatible result resolution interface.

        Also, explicitly defining this method prevents accidental definition of
        `State.result` method via attribute setters, which might prevent result
        resolution in some edge cases.
        """
        return self

    @classmethod
    def from_sample(cls, sample, bqm):
        """Convenience method for constructing State from raw (dict) sample;
        energy is calculated from the BQM, and State.problem is also set to that
        BQM.
        """
        return cls(problem=bqm,
                   samples=SampleSet.from_sample(sample,
                                                 vartype=bqm.vartype,
                                                 energy=bqm.energy(sample)))


class Present(Future):
    """Already resolved Future object."""

    def __init__(self, result=None, exception=None):
        super(Present, self).__init__()
        if result:
            self.set_result(result)
        elif exception:
            self.set_exception(exception)
        else:
            raise ValueError("can't provide both 'result' and 'exception'")


class RunnableError(Exception):
    """Generic Runnable exception error that includes the error context, in
    particular, the `State` that caused the runnable component to fail."""

    def __init__(self, message, state):
        super(RunnableError, self).__init__(message)
        self.state = state


class Runnable(object):
    """Component that can be run for an iteration such as samplers and branches.

    Implementations must support the iterate or run methods, stop is not required.

    Examples:
        This example runs a tabu search on a binary quadratic model. An initial state is
        manually set with invalid solution :math:`x=y=0, z=1; a=b=1, c=0` and an updated
        state is created by running the sampler for one iteration.

        >>> import dimod           # Create a binary quadratic model
        >>> bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0, 'a': 2.0, 'b': 0.0, 'c': 6.0},
        ...                                  {('y', 'x'): 2.0, ('z', 'x'): -4.0, ('z', 'y'): -4.0,
        ...                                  ('b', 'a'): 2.0, ('c', 'a'): -4.0, ('c', 'b'): -4.0, ('a', 'z'): -4.0},
        ...                                  -1.0, 'BINARY')
        >>> # Set up the sampler runnable
        >>> sampler = samplers.TabuProblemSampler(bqm, tenure=2, timeout=5)
        >>> # Run one iteration of the sampler
        >>> new_state = sampler.iterate(core.State.from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm))
        >>> print(new_state.samples)      # doctest: +SKIP
        Response(rec.array([([1, 1, 1, 1, 1, 1], -1., 1)],
          dtype=[('sample', 'i1', (6,)), ('energy', '<f8'), ('num_occurrences', '<i4')]),
          ['a', 'b', 'c', 'x', 'y', 'z'], {}, 'BINARY')
    """

    def __init__(self, *args, **kwargs):
        super(Runnable, self).__init__(*args, **kwargs)

    @property
    def name(self):
        """Return the class name of an instantiated :class:`Runnable`.

        Examples:
            This code-snippet returns the class name of an instantiated sampler.

            >>> print(sampler.name)      # doctest: +SKIP
            TabuProblemSampler

        """
        return self.__class__.__name__

    def init(self, state):
        """Run prior to the first iterate/run, with the first state received.

        Default to NOP.
        """
        pass

    def iterate(self, state):
        """Start a blocking iteration of an instantiated :class:`Runnable`.

        Accepts a state and returns a new state.

        Args:
            state (:class:`State`): Computation state passed between connected components.

        Examples:
            This code snippet runs one iteration of a sampler to produce a new state::

                new_state = sampler.iterate(core.State.from_sample({'x': 0, 'y': 0}, bqm))

        """
        raise NotImplementedError

    def error(self, exc):
        """Called when previous component raised an exception (instead of new state).

        Must return a valid new `State`, or raise an exception.

        Default to re-raise of input exception. Runnable errors must be explicitly silenced.
        """
        raise exc

    def dispatch(self, future):
        """Dispatch `state` got by resolving `future` to either `iterate` or `error`.

        Args:
            state (:class:`concurrent.futures.Future`-like object): :class:`State` future.

        Returns state from `iterate`/`error`, or passes-thru an exception raised there.
        Blocks on `state` resolution and `iterate`/`error` execution .
        """

        try:
            state = future.result()
        except Exception as exc:
            return self.error(exc)

        if not getattr(self, '_initialized', False):
            self.init(state)
            setattr(self, '_initialized', True)

        return self.iterate(state)

    def run(self, state, defer=True):
        """Execute the next step/iteration of an instantiated :class:`Runnable`.

        Accepts a state in a :class:`~concurrent.futures.Future`-like object and
        return a new state in a :class:`~concurrent.futures.Future`-like object.

        Args:
            state (:class:`State`):
                Computation state future-lookalike passed between connected components.

            defer (bool, optional, default=True):
                Return result future immediately, and run the computation asynchronously.
                If set to false, block on computation, and return the resolved future.

        Examples:
            This code snippet runs one iteration of a sampler to produce a new state::

                new_state = sampler.run(core.State.from_sample({'x': 0, 'y': 0}, bqm))

        """
        if defer:
            executor = async_executor
        else:
            executor = immediate_executor

        return executor.submit(self.dispatch, state)

    def stop(self):
        """Terminate an iteration of an instantiated :class:`Runnable`."""
        pass

    def __or__(self, other):
        """Composition of runnable components (L-to-R) returns a new runnable Branch."""
        return Branch(components=(self, other))


class Branch(Runnable):
    """Sequentially executed :class:`Runnable` components.

    Args:
        components (iterable of :class:`Runnable`): Complete processing sequence to
            update a current set of samples, such as: :code:`decomposer | sampler | composer`.

    Examples:
        This example runs one iteration of a branch comprising a decomposer, a D-Wave system,
        and a composer. A 10-variable binary quadratic model is decomposed by the energy
        impact of its variables into a 6-variable subproblem to be sampled twice
        with a manually set initial state of all -1 values.

        >>> import dimod           # Create a binary quadratic model
        >>> bqm = dimod.BinaryQuadraticModel({t: 0 for t in range(10)},
        ...                                  {(t, (t+1) % 10): 1 for t in range(10)},
        ...                                  0, 'SPIN')
        >>> # Run one iteration on a branch
        >>> branch = EnergyImpactDecomposer(bqm, max_size=6, min_gain=-10) |
        ...                  QPUSubproblemAutoEmbeddingSampler(num_reads=2) |
        ...                  SplatComposer(bqm)
        >>> new_state = branch.iterate(core.State.from_sample(min_sample(bqm), bqm)
        >>> print(new_state.subsamples)      # doctest: +SKIP
        Response(rec.array([([-1,  1, -1,  1, -1,  1], -5., 1),
           ([ 1, -1,  1, -1, -1,  1], -5., 1)],
        >>> # Above response snipped for brevity

    """

    def __init__(self, components=(), *args, **kwargs):
        super(Branch, self).__init__(*args, **kwargs)
        self.components = tuple(components)

    def __or__(self, other):
        """Composition of Branch with runnable components (L-to-R) returns a new
        runnable Branch.
        """
        if isinstance(other, Branch):
            return Branch(components=chain(self.components, other.components))
        elif isinstance(other, Runnable):
            return Branch(components=chain(self.components, (other,)))
        else:
            raise TypeError("branch can be composed only with Branch or Runnable")

    @property
    def name(self):
        """Return the :class:`Runnable` classes in a branch.

        Examples:
            This code-snippet returns the classes in an instantiated branch.

            >>> print(branch.name)      # doctest: +SKIP
            EnergyImpactDecomposer | QPUSubproblemAutoEmbeddingSampler | SplatComposer
        """
        return " | ".join(component.name for component in self.components)

    def iterate(self, state):
        """Start an iteration of an instantiated :class:`Branch`.

        Accepts a state and returns a new state.

        Args:
            state (:class:`State`):
                Computation state passed to the first component of the branch.

        Examples:
            This code snippet runs one iteration of a branch to produce a new state::

                new_state = branch.iterate(core.State.from_sample(min_sample(bqm), bqm)

        """
        for component in self.components:
            state = component.run(state, defer=False)
        return state.result()

    def error(self, exc):
        """Be explicit about propagating input error."""
        raise exc

    def stop(self):
        """Try terminating all components in an instantiated :class:`Branch`."""
        for component in self.components:
            component.stop()
