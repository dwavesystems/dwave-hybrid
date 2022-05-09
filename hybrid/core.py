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

import logging
import threading
from collections import namedtuple, defaultdict
from copy import deepcopy

from plucky import merge
import dimod

from hybrid import traits
from hybrid.utils import min_sample, sample_as_dict, hstack_samplesets, vstack_samplesets, cpu_count
from hybrid.profiling import make_timeit, make_count
from hybrid.concurrency import Future, Present, Executor, immediate_executor, thread_executor

__all__ = [
    'SampleSet', 'State', 'States', 'Runnable', 'HybridSampler',
    'HybridRunnable', 'HybridProblemRunnable', 'HybridSubproblemRunnable',
    'stoppable'
]

logger = logging.getLogger(__name__)


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
    """The `dimod.SampleSet` extended with a few helper methods.

    Note: this is basically a staging area for new `dimod.SampleSet` features
    before we merge them upstream.
    """

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

    def hstack(self, *others):
        """Combine the first sample in this SampleSet with first samples in all
        other SampleSets. Energy is reset to zero, and vartype is cast to the
        local vartype (first sampleset's vartype).
        """
        return hstack_samplesets(self, *others)

    def vstack(self, *others):
        return vstack_samplesets(self, *others)


class State(PliableDict):
    """Computation state passed along a branch between connected components.

    State is a :class:`dict` subclass and usually contains at least two keys:
    ``samples`` and ``problem``.

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

    def copy(self):
        """Simple deep copy of itself. Functionally identical to
        `State.updated()`.
        """
        return deepcopy(self)

    def updated(self, **kwargs):
        """Return a (deep) copy of the state, updated from `kwargs`.

        This method has `dict.update` semantics with immutability of `sorted`.
        Currently an exception is the `debug` key, if it exists, for which a
        depth-unlimited recursive merge is executed.

        Example:

            >>> state = State()
            >>> state
            {}
            >>> newstate = state.updated(problem="test")
            >>> newstate
            {'problem': 'test'}
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
    def from_sample(cls, sample, bqm, **kwargs):
        """Convenience method for constructing a state from a raw (dict) sample.

        Energy is calculated from the binary quadratic model (BQM), and
        `State.problem` is also set to that BQM.

        Args:
            sample (:class:`dimod.SampleLike`):
                A single sample. For recognized formats, see
                :class:`~dimod.typing.SampleLike`.

            bqm (:class:`dimod.BinaryQuadraticModel`):
                Binary quadratic model compatible with the sample provided.

            **kwargs:
                Arbitrary state variables to be set.

        Example:

            >>> import dimod
            >>> bqm = dimod.BQM.from_ising({}, {'ab': 0.5, 'bc': 0.5, 'ca': 0.5})
            >>> state = State.from_sample({'a': -1, 'b': -1, 'c': -1}, bqm)

        """
        return cls.from_samples(sample, bqm, **kwargs)

    @classmethod
    def from_samples(cls, samples, bqm, **kwargs):
        """Convenience method for constructing a state from raw (dict) samples.

        Per-sample energy is calculated from the binary quadratic model (BQM),
        and `State.problem` is set to the BQM.

        Args:
            samples (:class:`dimod.SamplesLike`):
                Collection of samples. For recognized formats, see
                :class:`~dimod.typing.SamplesLike`.

            bqm (:class:`dimod.BinaryQuadraticModel`):
                Binary quadratic model compatible with samples provided.

            **kwargs:
                Arbitrary state variables to be set.

        Example:

            >>> import dimod
            >>> bqm = dimod.BQM.from_ising({}, {'ab': 0.5, 'bc': 0.5, 'ca': 0.5})
            >>> state = State.from_samples([{'a': -1, 'b': -1, 'c': -1},
            ...                             {'a': -1, 'b': -1, 'c': 1}], bqm)
        """
        return cls(problem=bqm,
                   samples=SampleSet.from_samples_bqm(samples, bqm), **kwargs)

    @classmethod
    def from_subsample(cls, subsample, bqm, **kwargs):
        """Similar to :meth:`.from_sample`, but initializes `subproblem` and
        `subsamples`.
        """
        return cls.from_subsamples(subsample, bqm, **kwargs)

    @classmethod
    def from_subsamples(cls, subsamples, bqm, **kwargs):
        """Similar to :meth:`.from_samples`, but initializes `subproblem` and
        `subsamples`.
        """
        return cls(subproblem=bqm,
                   subsamples=SampleSet.from_samples_bqm(subsamples, bqm), **kwargs)

    @classmethod
    def from_problem(cls, bqm, samples=None, **kwargs):
        """Convenience method for constructing a state from (possibly only)
        a BQM.
        """

        if samples is None:
            samples = min_sample

        if callable(samples):
            samples_like = samples(bqm)
        else:
            samples_like = samples

        return cls.from_samples(samples_like, bqm, **kwargs)

    @classmethod
    def from_subproblem(cls, bqm, subsamples=None, **kwargs):
        """Convenience method for constructing a state from (possibly only)
        a subproblem BQM.
        """

        if subsamples is None:
            subsamples = min_sample

        if callable(subsamples):
            subsamples_like = subsamples(bqm)
        else:
            subsamples_like = subsamples

        return cls.from_subsamples(subsamples_like, bqm, **kwargs)


class States(list):
    """List of states."""

    def __init__(self, *args):
        super(States, self).__init__(args)

    def result(self):
        return self

    @property
    def first(self):
        return self[0]

    def updated(self, **kwargs):
        """Return a (deep) copy of the states, updated from `kwargs`."""
        return States(*(state.updated(**kwargs) for state in self))


class Runnable(traits.StateTraits):
    """Components such as samplers and branches that can be run for an iteration.

    Args:
        **runopts (dict):
            Keyword arguments passed down to each `Runnable.run` call.

    Note:
        The base class :class:`~hybrid.core.Runnable` does not enforce traits
        validation. To enable validation, derive your subclass from one of the
        state structure, I/O dimensionality, or I/O validation mixins in
        :mod:`~hybrid.traits`.

    Examples:
        This example runs a tabu search on a binary quadratic model. An initial
        state is manually set to :math:`x=y=0, z=1; a=b=1, c=0` and an updated
        state is created by running the sampler for one iteration.

        >>> import dimod           # Create a binary quadratic model
        >>> bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0, 'a': 2.0, 'b': 0.0, 'c': 6.0},
        ...                                  {('y', 'x'): 2.0, ('z', 'x'): -4.0, ('z', 'y'): -4.0,
        ...                                  ('b', 'a'): 2.0, ('c', 'a'): -4.0, ('c', 'b'): -4.0, ('a', 'z'): -4.0},
        ...                                  -1.0, 'BINARY')
        >>> # Set up the sampler runnable
        >>> sampler = TabuProblemSampler(tenure=2, timeout=5)
        >>> # Run one iteration of the sampler
        >>> new_state = sampler.next(State.from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm))
        >>> print(new_state.samples)      # doctest: +SKIP
              a  b  c  x  y  z  energy  num_occ.
           0  1  1  1  1  1  1    -1.0         1
           [ 1 rows, 6 variables ]

    """

    def __init__(self, **runopts):
        super(Runnable, self).__init__()

        self.runopts = runopts

        self.timers = defaultdict(list)
        self.timeit = make_timeit(self.timers, prefix=self.name, loglevel=logging.TRACE)

        self.counters = defaultdict(int)
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

    def init(self, state, **runopts):
        """Run prior to the first next/run, with the first state received.

        Default to NOP.
        """
        pass

    def next(self, state, **runopts):
        """Execute one blocking iteration of an instantiated :class:`Runnable`
        with a valid state as input.

        Args:
            state (:class:`State`):
                Computation state passed between connected components.

        Returns:
            :class:`State`: The new state.

        Examples:
            This code snippet runs one iteration of a sampler to produce a new
            state::

                new_state = sampler.next(core.State.from_sample({'x': 0, 'y': 0}, bqm))

        """
        raise NotImplementedError

    def error(self, exc):
        """Execute one blocking iteration of an instantiated :class:`Runnable`
        with an exception as input.

        Called when the previous component raised an exception instead of
        generating a new state.

        The default implementation raises again the input exception. Runnable
        errors must be explicitly silenced.
        """
        raise exc

    def halt(self):
        """Called by `stop()`. Override this method (instead of `stop`) to
        handle stopping of one blocking call of `next`. Defaults to NOP.
        """
        pass

    def dispatch(self, future, **kwargs):
        """Dispatch state from resolving `future` to either `next` or `error`
        methods.

        Args:
            state (:class:`concurrent.futures.Future`-like object):
                :class:`State` future.

        Returns state from :meth:`next` or :meth:`error`, or passes through an
        exception raised there.

        Blocks on state resolution and execution of :meth:`next` or
        :meth:`error`.
        """

        with self.timeit('dispatch.resolve'):
            try:
                state = future.result()
            except Exception as exc:
                with self.timeit('dispatch.resolve.error'):
                    return self.error(exc)

        if not getattr(self, '_initialized', False):
            with self.timeit('dispatch.init'):
                self.init(state, **kwargs)
            setattr(self, '_initialized', True)

        self.validate_input_state_traits(state)

        with self.timeit('dispatch.next'):
            new_state = self.next(state, **kwargs)

        self.validate_output_state_traits(new_state)

        return new_state

    def run(self, state, **kwargs):
        """Execute the next step/iteration of an instantiated :class:`Runnable`.

        Accepts a state in a :class:`~concurrent.futures.Future`-like object and
        returns a new state in a :class:`~concurrent.futures.Future`-like object.

        Args:
            state (:class:`State`):
                Computation state future-like object passed between connected
                components.

            executor (:class:`~concurrent.futures.Executor`, optional, default=None):
                The Executor to which the execution of this block is scheduled.
                By default `hybrid.concurrency.thread_executor` is used.

        Examples:
            These two code snippets run one iteration of a sampler to produce a
            new state. The first is an asynchronous call and the second a
            blocking call.

            >>> sampler.run(State.from_sample(min_sample(bqm), bqm))   # doctest: +SKIP
            <Future at 0x20cbe22ea20 state=running>

            >>> sampler.run(State.from_sample(min_sample(bqm), bqm),
            ...             executor=hybrid.immediate_executor)   # doctest: +SKIP
            <Present at 0x20ca68cd2b0 state=finished returned State>
        """

        # merge deferred local runopts with explicit kwarg options
        runopts = self.runopts.copy()
        runopts.update(kwargs)

        # based on merged runopts, decide on the executor
        executor = runopts.pop('executor', None)
        if executor is None:
            executor = thread_executor

        if not isinstance(executor, Executor):
            raise TypeError("expecting 'concurrent.futures.Executor' subclass "
                            "instance for 'executor'")

        with self.timeit('dispatch'):
            return executor.submit(self.dispatch, state, **runopts)

    def stop(self):
        """Terminate an iteration of an instantiated :class:`Runnable`."""
        return self.halt()

    def __or__(self, other):
        """Composition of runnable components (L-to-R) returns a new
        runnable Branch."""
        return Branch(components=(self, other))

    def __and__(self, other):
        """Parallel composition of runnable components returns new Branches."""
        if isinstance(other, Branches):
            return Branches(self, *other)
        elif isinstance(other, Runnable):
            return Branches(self, other)
        else:
            raise TypeError("only Runnables can be composed into Branches")


def stoppable(cls):
    """Extends a `Runnable` subclass with a `stop_signal` semaphore/event, and
    amends the existing `halt` and `next` methods to signal stop via the
    semaphore, and reset the stop signal on run completion, respectively.

    Example:
        A Runnable block that accepts `timeout` and on `run` blocks for up to
        `timeout` seconds. It can be interrupted via call to `stop`, in which
        case blocks shorter than the timeout interval::

            @stoppable
            class StoppableSleeper(Runnable):
                def next(self, state, timeout=None, **runopts):
                    self.stop_signal.wait(timeout=timeout)
                    return state

            >>> sleeper = StoppableSleeper(timeout=30)
            >>> sleeper.run(state)
            <Future at 0x7fbb575e6f60 state=running>

            >>> sleeper.stop()
            >>> sleeper.timers
            <snipped>
                'dispatch.next': [13.224211193970405]

    """
    if not issubclass(cls, Runnable):
        raise TypeError("Runnable subclass expected as the decorated class")

    orig_init = cls.__init__
    orig_halt = cls.halt
    orig_next = cls.next

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.stop_signal = threading.Event()

    def halt(self):
        result = orig_halt(self)
        self.stop_signal.set()
        return result

    def next(self, state, **runopts):
        result = orig_next(self, state, **runopts)
        self.stop_signal.clear()
        return result

    cls.__init__ = __init__
    cls.halt = halt
    cls.next = next

    return cls


class HybridSampler(dimod.Sampler):
    """Produces a `dimod.Sampler` from a `hybrid.Runnable`-based sampler.

    Args:
        workflow (`Runnable`):
            Hybrid workflow, likely composed, that accepts a binary quadratic
            model in the input state and produces sample(s) in the output state.

    Example:
        This example produces a :std:doc:`dimod <oceandocs:docs_dimod/sdk_index>`
        sampler from :class:`~hybrid.samplers.TabuProblemSampler` and uses its
        `sample_ising` mixin to solve a simple Ising problem.

        >>> hybrid_sampler = TabuProblemSampler()
        >>> dimod_sampler = HybridSampler(hybrid_sampler)
        >>> solution = dimod_sampler.sample_ising({}, {'ab': 0.5, 'bc': 0.5, 'ca': 0.5})
        >>> solution.first.energy
        -0.5
    """

    properties = None
    parameters = None

    def __init__(self, workflow):
        """Construct the sampler off of a (composed) `Runnable` BQM solver."""
        if not isinstance(workflow, Runnable):
            raise TypeError("'sampler' should be 'hybrid.Runnable'")
        self._workflow = workflow

        self.parameters = {'initial_sample': []}
        self.properties = {}

    def sample(self, bqm, initial_sample=None, return_state=False, **params):
        """Sample from a binary quadratic model using composed runnable sampler.

        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            initial_sample (dict, default=None):
                `bqm`-compatible sample used for initial state construction.
                Defaults to `hybrid.utils.min_sample(bqm)`.

            return_state (bool, optional, default=False):
                If True, the final state is added to :attr:`dimod.SampleSet.info`
                of the returned sample set. Note that if a `state` key
                already exists in the sample set then it is overwritten.

            **params (dict):
                Sampling parameters passed down to the underlying workflow as
                run-time parameters.

        Note:
            Sampling via hybrid workflow is run asynchronously, and a sample set
            is returned as soon as workflow starts running. A blocking result
            resolve occurres on the first attribute access to the returned
            sample set.

        Returns:
            :class:`~dimod.SampleSet`:
                Possibly yet unresolved sample set.

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
        final_state = self._workflow.run(initial_state, **params)

        def result_hook(state):
            resolved = state.result()
            ss = resolved.samples
            if return_state:
                # note: this creates a cyclic reference to `samples` from the
                # sample set via `info['state']`, but that shouldn't be a
                # problem for GC
                ss.info.update(state=resolved)
            return ss

        return dimod.SampleSet.from_future(final_state, result_hook=result_hook)


class HybridRunnable(Runnable):
    """Produces a `hybrid.Runnable` from a `dimod.Sampler` (dual of
    `HybridSampler`).

    The runnable samples from a problem defined in a state field named
    `fields[0]` and populates the state field referred to by `fields[1]`.

    Args:
        sampler (:class:`dimod.Sampler`):
            dimod-compatible sampler which is run on every iteration of the
            runnable.

        fields (tuple(str, str)):
            Input and output state field names.

        **sample_kwargs (dict):
            Sampler-specific parameters passed to sampler on every call.

    Example:
        This example creates a :class:`Runnable` from dimod sampler
        :std:doc:`TabuSampler <oceandocs:docs_tabu/sdk_index>`, runs it on an Ising model, and
        finds the lowest energy.

        >>> from tabu import TabuSampler
        >>> import dimod
        >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': 0.5, 'bc': 0.5, 'ca': 0.5})
        >>> runnable = HybridRunnable(TabuSampler(), fields=('subproblem', 'subsamples'), timeout=100)
        >>> state0 = State(subproblem=bqm, subsamples=SampleSet.from_samples_bqm(min_sample(bqm), bqm))
        >>> state = runnable.run(state0)
        >>> state.result()['subsamples'].first.energy     # doctest: +SKIP
        -0.5

    """

    def __init__(self, sampler, fields, **sample_kwargs):
        super(HybridRunnable, self).__init__(**sample_kwargs)

        if not isinstance(sampler, dimod.Sampler):
            raise TypeError("'sampler' should be 'dimod.Sampler'")
        try:
            assert len(tuple(fields)) == 2
        except:
            raise ValueError("'fields' should be two-tuple with input/output state fields")

        self.sampler = sampler
        self.input, self.output = fields

    def next(self, state, **sample_kwargs):
        known_params = self.sampler.parameters
        sample_params = {k: v for k, v in sample_kwargs.items() if k in known_params}
        response = self.sampler.sample(state[self.input], **sample_params)
        return state.updated(**{self.output: response})


class HybridProblemRunnable(HybridRunnable):
    """Produces a `hybrid.Runnable` from a `dimod.Sampler` (dual of
    `HybridSampler`).

    The runnable that samples from `state.problem` and populates
    `state.samples`.

    See an example in :class:`hybrid.core.HybridRunnable`. An example of the
    duality with `HybridSampler` is::

        HybridProblemRunnable(HybridSampler(TabuProblemSampler())) == TabuProblemSampler()
    """

    def __init__(self, sampler, **sample_kwargs):
        super(HybridProblemRunnable, self).__init__(
            sampler, fields=('problem', 'samples'), **sample_kwargs)


class HybridSubproblemRunnable(HybridRunnable):
    """Produces a `hybrid.Runnable` from a `dimod.Sampler` (dual of
    `HybridSampler`).

    The runnable that samples from `state.subproblem` and populates
    `state.subsamples`.

    See an example in :class:`hybrid.core.HybridRunnable`.
    """

    def __init__(self, sampler, **sample_kwargs):
        super(HybridSubproblemRunnable, self).__init__(
            sampler, fields=('subproblem', 'subsamples'), **sample_kwargs)


# deferred import, due to flow.* deps on core.*
from hybrid.flow import Branch, Branches
