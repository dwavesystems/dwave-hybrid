from collections import namedtuple
from itertools import chain
from copy import deepcopy
import operator

# TODO: abstract as singleton executor under hades namespace
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

from plucky import merge
import dimod


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

    @classmethod
    def from_sample(cls, sample, bqm):
        """Convenience method for constructing State from raw (dict) sample;
        energy is calculated from BQM.
        """
        return cls(samples=SampleSet.from_sample(sample,
                                                 vartype=bqm.vartype,
                                                 energy=bqm.energy(sample)))


class Present(object):
    """Already resolved Future-like object.

    Very limited in Future-compatibility. We implement only the minimum here.
    """

    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result

    def done(self):
        return True


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

    def run(self, state):
        """Start an asynchronous iteration of an instantiated :class:`Runnable`.

        Accepts a state in a :class:`future` and return a new state in a :class:`future`.

        Args:
            state (:class:`State`): Computation state passed between connected components.

        Examples:
            This code snippet runs one iteration of a sampler to produce a new state::

                new_state = sampler.run(core.State.from_sample({'x': 0, 'y': 0}, bqm))

        """
        try:
            state = state.result()
        except:
            pass
        return executor.submit(self.iterate, state)

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
            state = component.iterate(state)
        return state

    def stop(self):
        """Try terminating all components in an instantiated :class:`Branch`."""
        for component in self.components:
            component.stop()
