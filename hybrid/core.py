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

from collections import namedtuple
from itertools import chain
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, Executor
import operator
import logging

from plucky import merge
import dimod

from hybrid.traits import StateTraits
from hybrid.utils import min_sample, sample_as_dict
from hybrid.profiling import make_count

logger = logging.getLogger(__name__)


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


class PliableDict(dict):
    """Dictionary subclass with attribute accessors acting as item accessors.

    Example:

        >>> d = PliableDict(x=1)
        >>> d.y = 2
        >>> d                     # doctest: +SKIP
        {'x': 1, 'y': 2}
        >>> d.z is None
        True
    """

    # some attribute lookups will be delegated to superclass, to handle things like pickling
    _delegated = frozenset(('__reduce_ex__', '__reduce__',
                            '__getstate__', '__setstate__',
                            '__getinitargs__', '__getnewargs__'))

    def __getattr__(self, name):
        if name in self._delegated:
            return super(PliableDict, self).__getattr__(name)

        return self.get(name)

    def __setattr__(self, name, value):
        self[name] = value


class SampleSet(dimod.SampleSet):
    """The `dimod.SampleSet` extended with a few helper methods."""

    def __init__(self, *args, **kwargs):
        if not args and not kwargs:
            # construct empty SampleSet
            empty = self.empty()
            super(SampleSet, self).__init__(empty.record, empty.variables,
                                            empty.info, empty.vartype)
        else:
            super(SampleSet, self).__init__(*args, **kwargs)

    @classmethod
    def empty(cls):
        return cls.from_samples([], vartype=dimod.SPIN, energy=0)

    @classmethod
    def from_bqm_sample(cls, bqm, sample):
        return cls.from_bqm_samples(bqm, [sample])

    @classmethod
    def from_bqm_samples(cls, bqm, samples):
        """Construct SampleSet from samples on BQM, filling in vartype and
        energies.
        """
        return cls.from_samples(
            samples,
            vartype=bqm.vartype,
            energy=[bqm.energy(sample) for sample in samples])


class State(PliableDict):
    """Computation state passed along a branch between connected components.

    State is a `dict` subclass and contains at least keys `samples`, `problem`.

    Examples:
        >>> import dimod           # Create a binary quadratic model
        >>> bqm = dimod.BinaryQuadraticModel({0: -1, 1: -1}, {(0, 1): 2}, 0.0, dimod.BINARY)
        >>> hybrid.core.State.from_sample(hybrid.utils.min_sample(bqm), bqm)   # doctest: +SKIP
        {'problem': BinaryQuadraticModel({0: -1, 1: -1}, {(0, 1): 2}, 0.0, Vartype.BINARY),
         'samples': SampleSet(rec.array([([0, 0], 0., 1)],
            dtype=[('sample', 'i1', (2,)), ('energy', '<f8'), ('num_occurrences', '<i4')]), [0, 1], {}, 'BINARY')}
    """

    def __init__(self, *args, **kwargs):
        """State is a `PliableDict` (`dict` subclass) which usually contains
        at least two keys: `samples` and `problem`.
        """
        super(State, self).__init__(*args, **kwargs)
        self.setdefault('samples', None)
        self.setdefault('problem', None)

    def copy(self):
        """Simple deep copy of itself. Functionally identical to
        `State.updated()`.
        """
        return deepcopy(self)

    def updated(self, **kwargs):
        """Return a (deep) copy of the state, updated from `kwargs`.

        It has `dict.update` semantics with immutability of `sorted`. Currently an
        exception is the `debug` key, if it exists, for which a depth-unlimited
        recursive merge is executed.

        Example:

            >>> state = State()
            >>> state
            {problem': None, 'samples': None}
            >>> newstate = state.updated(problem="test")
            >>> newstate
            {problem': 'test', 'samples': None}
        """

        overwrite = lambda a,b: b

        # use a special merge strategy for `debug` (op=overwrite, max_depth=None)
        debug = merge(self.get('debug', {}), kwargs.get('debug', {}), op=overwrite)
        if debug:
            self.setdefault('debug', PliableDict())
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
        """Convenience method for constructing a state from a raw (dict) sample.

        Energy is calculated from the binary quadratic model (BQM), and `State.problem`
        is also set to that BQM.

        Example:

            >>> import dimod
            >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': 0.5, 'bc': 0.5, 'ca': 0.5})
            >>> state = State.from_sample({'a': -1, 'b': -1, 'c': -1}, bqm)

        """
        return cls.from_samples([sample], bqm)

    @classmethod
    def from_samples(cls, samples, bqm):
        """Convenience method for constructing a state from raw (dict) samples.

        Per-sample energy is calculated from the binary quadratic model (BQM), and
        `State.problem` is set to the BQM.

        Example:

            >>> import dimod
            >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': 0.5, 'bc': 0.5, 'ca': 0.5})
            >>> state = State.from_samples({'a': -1, 'b': -1, 'c': -1},
            ...                            {'a': -1, 'b': -1, 'c': 1}, bqm)
        """
        return cls(problem=bqm, samples=SampleSet.from_bqm_samples(bqm, samples))


class Present(Future):
    """Already resolved :class:`~concurrent.futures.Future` object.

    Users should treat this class as just another :class:`~concurrent.futures.Future`,
    the difference being an implementation detail: :class:`Present` is "resolved" at
    construction time.

    See the example of the :meth:`Runnable.run` method.
    """

    def __init__(self, result=None, exception=None):
        super(Present, self).__init__()
        if result is not None:
            self.set_result(result)
        elif exception is not None:
            self.set_exception(exception)
        else:
            raise ValueError("can't provide both 'result' and 'exception'")


class Runnable(StateTraits):
    """Component such as samplers and branches that can be run for an iteration.

    Examples:
        This example runs a tabu search on a binary quadratic model. An initial state is
        manually set to :math:`x=y=0, z=1; a=b=1, c=0` and an updated
        state is created by running the sampler for one iteration.

        >>> import dimod           # Create a binary quadratic model
        >>> bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0, 'a': 2.0, 'b': 0.0, 'c': 6.0},
        ...                                  {('y', 'x'): 2.0, ('z', 'x'): -4.0, ('z', 'y'): -4.0,
        ...                                  ('b', 'a'): 2.0, ('c', 'a'): -4.0, ('c', 'b'): -4.0, ('a', 'z'): -4.0},
        ...                                  -1.0, 'BINARY')
        >>> # Set up the sampler runnable
        >>> sampler = samplers.TabuProblemSampler(bqm, tenure=2, timeout=5)
        >>> # Run one iteration of the sampler
        >>> new_state = sampler.next(core.State.from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm))
        >>> print(new_state.samples)      # doctest: +SKIP
        Response(rec.array([([1, 1, 1, 1, 1, 1], -1., 1)],
          dtype=[('sample', 'i1', (6,)), ('energy', '<f8'), ('num_occurrences', '<i4')]),
          ['a', 'b', 'c', 'x', 'y', 'z'], {}, 'BINARY')

    """

    def __init__(self, *args, **kwargs):
        super(Runnable, self).__init__(*args, **kwargs)

        self.counters = {}
        self.count = make_count(self.counters, prefix=self.name, loglevel=logging.TRACE)

    def __str__(self):
        return self.name

    def __repr__(self):
        return "{}()".format(self.name)

    def __iter__(self):
        return iter(tuple())

    @property
    def name(self):
        """Return the :class:`Runnable` class name."""
        return self.__class__.__name__

    def init(self, state):
        """Run prior to the first next/run, with the first state received.

        Default to NOP.
        """
        pass

    def next(self, state):
        """Execute one blocking iteration of an instantiated :class:`Runnable`.

        Args:
            state (:class:`State`): Computation state passed between connected components.

        Returns:
            :class:`State`: The new state.

        Examples:
            This code snippet runs one iteration of a sampler to produce a new state::

                new_state = sampler.next(core.State.from_sample({'x': 0, 'y': 0}, bqm))

        """
        raise NotImplementedError

    def error(self, exc):
        """Return a valid new state or raise an exception.

        Called when the previous component raised an exception instead of generating a
        new state.

        The default implementation raises again the input exception. Runnable errors
        must be explicitly silenced.
        """
        raise exc

    def dispatch(self, future):
        """Dispatch state from resolving `future` to either `next` or `error` methods.

        Args:
            state (:class:`concurrent.futures.Future`-like object): :class:`State` future.

        Returns state from :meth:`next` or :meth:`error`, or passes through an exception
        raised there.

        Blocks on state resolution and execution of :meth:`next` or :meth:`error`.
        """

        with self.count('dispatch.resolve'):
            try:
                state = future.result()
            except Exception as exc:
                with self.count('dispatch.resolve.error'):
                    return self.error(exc)

        if not getattr(self, '_initialized', False):
            with self.count('dispatch.init'):
                self.init(state)
            setattr(self, '_initialized', True)

        self.validate_input_state_traits(state)

        with self.count('dispatch.next'):
            new_state = self.next(state)

        self.validate_output_state_traits(new_state)

        return new_state

    def run(self, state, defer=True):
        """Execute the next step/iteration of an instantiated :class:`Runnable`.

        Accepts a state in a :class:`~concurrent.futures.Future`-like object and
        returns a new state in a :class:`~concurrent.futures.Future`-like object.

        Args:
            state (:class:`State`):
                Computation state future-like object passed between connected components.

            defer (bool, optional, default=True):
                If True, returns a future immediately and runs the computation
                asynchronously. If False, blocks on computation until the resolved
                future is returned.

        Examples:
            These two code snippets run one iteration of a sampler to produce a new state.
            The second sets the 'defer' argument to False.

            >>> sampler.run(State.from_sample(min_sample(bqm), bqm))   # doctest: +SKIP
            <Future at 0x20cbe22ea20 state=running>

            >>> sampler.run(State.from_sample(min_sample(bqm), bqm), defer=False)   # doctest: +SKIP
            <Present at 0x20ca68cd2b0 state=finished returned State>
        """

        if defer:
            executor = async_executor
        else:
            executor = immediate_executor

        with self.count('dispatch'):
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
        >>> new_state = branch.next(core.State.from_sample(min_sample(bqm), bqm)
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

    def __str__(self):
        return " | ".join(map(str, self)) or "(empty branch)"

    def __repr__(self):
        return "{}(components={!r})".format(self.name, tuple(self))

    def __iter__(self):
        return iter(self.components)

    def next(self, state):
        """Start an iteration of an instantiated :class:`Branch`.

        Accepts a state and returns a new state.

        Args:
            state (:class:`State`):
                Computation state passed to the first component of the branch.

        Examples:
            This code snippet runs one iteration of a branch to produce a new state::

                new_state = branch.next(core.State.from_sample(min_sample(bqm), bqm)

        """
        for component in self.components:
            state = component.run(state, defer=False)
        return state.result()

    def error(self, exc):
        """Pass on the exception from input to the error handler of the first
        runnable in branch.
        """
        return self.next(Present(exception=exc))

    def stop(self):
        """Try terminating all components in an instantiated :class:`Branch`."""
        for component in self.components:
            component.stop()


class HybridSampler(dimod.Sampler):
    """Produces a `dimod.Sampler` from a `hybrid.Runnable`-based sampler.

    Args:
        runnable_solver (`Runnable`):
            Hybrid runnable, likely composed, that accepts a binary quadratic
            model in the input state and produces sample(s) in the output state.

    Example:
        This example produces a :std:doc:`dimod <dimod:index>` sampler from
        :class:`~hybrid.samplers.TabuProblemSampler` and uses its `sample_ising`
        mixin to solve a simple Ising problem.

        >>> hybrid_sampler = TabuProblemSampler()
        >>> dimod_sampler = HybridSampler(hybrid_sampler)
        >>> solution = dimod_sampler.sample_ising({}, {'ab': 0.5, 'bc': 0.5, 'ca': 0.5})
        >>> solution.first.energy
        -0.5
    """

    properties = None
    parameters = None

    def __init__(self, runnable_solver):
        """Construct the sampler off of a (composed) `Runnable` BQM solver."""
        if not isinstance(runnable_solver, Runnable):
            raise TypeError("'sampler' should be 'hybrid.Runnable'")
        self._runnable_solver = runnable_solver

        self.parameters = {'initial_sample': []}
        self.properties = {}

    def sample(self, bqm, initial_sample=None):
        """Sample from a binary quadratic model using composed runnable sampler.

        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            initial_sample (dict, default=None):
                `bqm`-compatible sample used for initial state construction.
                Defaults to `hybrid.utils.min_sample(bqm)`.

        Returns:
            :obj:`~dimod.Response`: A `dimod` :obj:`.~dimod.Response` object.

        """
        if not isinstance(bqm, dimod.BinaryQuadraticModel):
            raise TypeError("'bqm' should be BinaryQuadraticModel")

        if initial_sample is None:
            initial_sample = min_sample(bqm)
        else:
            initial_sample = sample_as_dict(initial_sample)

        if len(initial_sample) != len(bqm):
            raise ValueError("size of 'initial_sample' incompatible with 'bqm'")

        initial_state = State.from_sample(initial_sample, bqm)
        final_state = self._runnable_solver.run(initial_state)

        return dimod.Response.from_future(final_state, result_hook=lambda f: f.result().samples)


class HybridRunnable(Runnable):
    """Produces a `hybrid.Runnable` from a `dimod.Sampler` (dual of `HybridSampler`).

    The runnable samples from a problem defined in a state field named `fields[0]`
    and populates the state field referred to by `fields[1]`.

    Args:
        sampler (:class:`dimod.Sampler`):
            dimod-compatible sampler which is run on every iteration of the runnable.
        fields (tuple(str, str)):
            Input and output state field names.
        **sample_kwargs (dict):
            Sampler-specific parameters passed to sampler on every call/iteration.

    Example:
        This example creates a :class:`Runnable` from dimod sampler
        :std:doc:`TabuSampler <tabu:index>`, runs it on an Ising model, and finds
        the lowest energy.

        >>> from tabu import TabuSampler
        >>> import dimod
        >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': 0.5, 'bc': 0.5, 'ca': 0.5})
        >>> runnable = HybridRunnable(TabuSampler(), fields=('subproblem', 'subsamples'), timeout=100)
        >>> state0 = State(subproblem=bqm, subsamples=SampleSet.from_bqm_sample(bqm, min_sample(bqm)))
        >>> state = runnable.run(state0)
        >>> state.result()['subsamples'].first.energy     # doctest: +SKIP
        -0.5

    """

    def __init__(self, sampler, fields, **sample_kwargs):
        super(HybridRunnable, self).__init__()

        if not isinstance(sampler, dimod.Sampler):
            raise TypeError("'sampler' should be 'dimod.Sampler'")
        try:
            assert len(tuple(fields)) == 2
        except:
            raise ValueError("'fields' should be two-tuple with input/output state fields")

        self.sampler = sampler
        self.input, self.output = fields
        self.sample_kwargs = sample_kwargs

        # manually add traits
        self.inputs.add(self.input)
        self.outputs.add(self.output)

    def next(self, state):
        response = self.sampler.sample(state[self.input], **self.sample_kwargs)
        return state.updated(**{self.output: response})


class HybridProblemRunnable(HybridRunnable):
    """Produces a `hybrid.Runnable` from a `dimod.Sampler` (dual of `HybridSampler`).

    The runnable samples from `state.problem` and populates `state.samples`.

    See an example in :class:`hybrid.core.HybridRunnable`.
    """

    def __init__(self, sampler, **sample_kwargs):
        super(HybridProblemRunnable, self).__init__(
            sampler, fields=('problem', 'samples'), **sample_kwargs)


class HybridSubproblemRunnable(HybridRunnable):
    """Produces a `hybrid.Runnable` from a `dimod.Sampler` (dual of `HybridSampler`).

    The runnable samplea from `state.subproblem` and populates `state.subsamples`.

    See an example in :class:`hybrid.core.HybridRunnable`.
    """

    def __init__(self, sampler, **sample_kwargs):
        super(HybridSubproblemRunnable, self).__init__(
            sampler, fields=('subproblem', 'subsamples'), **sample_kwargs)
