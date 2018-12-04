.. _samplers:

========
Samplers
========

.. automodule:: hybrid.samplers

Classes
=======

.. autoclass:: QPUSubproblemExternalEmbeddingSampler
.. autoclass:: QPUSubproblemAutoEmbeddingSampler
.. autoclass:: SimulatedAnnealingSubproblemSampler
.. autoclass:: TabuSubproblemSampler
.. autoclass:: TabuProblemSampler
.. autoclass:: InterruptableTabuSampler
.. autoclass:: RandomSubproblemSampler

Examples
========

QPUSubproblemExternalEmbeddingSampler
-------------------------------------

This example works on a binary quadratic model of two AND gates in series
by sampling a BQM representing just one of the gates. Output :math:`z` of gate
:math:`z = x \wedge y` connects to input :math:`a` of gate :math:`c = a \wedge b`.
An initial state is manually set with invalid solution :math:`x=y=0, z=1; a=b=1, c=0`.
The state is updated by sampling the subproblem 100 times on a D-Wave system.
The execution results shown here were three valid solutions to the subproblem; for
example, :math:`x=0, y=1, z=0` occurred 22 times.

.. code-block:: python

    import dimod
    import minorminer
    from dwave.system.samplers import DWaveSampler
    from hybrid.samplers import QPUSubproblemExternalEmbeddingSampler
    from hybrid.core import State

    # Define a problem and a subproblem
    bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0, 'a': 2.0, 'b': 0.0, 'c': 6.0},
                                     {('y', 'x'): 2.0, ('z', 'x'): -4.0, ('z', 'y'): -4.0,
                                     ('b', 'a'): 2.0, ('c', 'a'): -4.0, ('c', 'b'): -4.0, ('a', 'z'): -4.0},
                                     -1.0, 'BINARY')
    sub_bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0},
                                         {('x', 'y'): 2.0, ('x', 'z'): -4.0, ('y', 'z'): -4.0},
                                         -1.0, dimod.Vartype.BINARY)

    # Find a minor-embedding for the subproblem
    qpu_sampler = DWaveSampler()
    sub_embedding = minorminer.find_embedding(list(sub_bqm.quadratic.keys()), qpu_sampler.edgelist)

    # Set up the sampler with an initial state
    sampler = QPUSubproblemExternalEmbeddingSampler(num_reads=100)
    state = State.from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm)
    state.update(subproblem=sub_bqm, embedding=sub_embedding)

::

    # Sample the subproblem on the QPU (REPL)
    >>> new_state = sampler.run(state).result()
    >>> print(new_state.subsamples.record)
    [([0, 1, 0], -1., 22) ([0, 0, 0], -1., 47) ([1, 0, 0], -1., 31)]


QPUSubproblemAutoEmbeddingSampler
---------------------------------

This example works on a binary quadratic model of two AND gates in series
by sampling a BQM representing just one of the gates. Output :math:`z` of gate
:math:`z = x \wedge y` connects to input :math:`a` of gate :math:`c = a \wedge b`.
An initial state is manually set with invalid solution :math:`x=y=0, z=1; a=b=1, c=0`.
The state is updated by sampling the subproblem 100 times on a D-Wave system.
The execution results shown here were four valid solutions to the subproblem; for
example, :math:`x=0, y=0, z=0` occurred 53 times.

.. code-block:: python

    import dimod
    from hybrid.samplers import QPUSubproblemAutoEmbeddingSampler
    from hybrid.core import State

    # Define a problem and a subproblem
    bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0, 'a': 2.0, 'b': 0.0, 'c': 6.0},
                                    {('y', 'x'): 2.0, ('z', 'x'): -4.0, ('z', 'y'): -4.0,
                                     ('b', 'a'): 2.0, ('c', 'a'): -4.0, ('c', 'b'): -4.0, ('a', 'z'): -4.0},
                                    -1.0, 'BINARY')
    sub_bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0},
                                        {('x', 'y'): 2.0, ('x', 'z'): -4.0, ('y', 'z'): -4.0},
                                        -1.0, dimod.Vartype.BINARY)

    # Set up the sampler with an initial state
    sampler = QPUSubproblemAutoEmbeddingSampler(num_reads=100)
    state = State.from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm)
    state.update(subproblem=sub_bqm)

::

    # Sample the subproblem on the QPU (REPL)
    >>> new_state = sampler.run(state).result()
    >>> print(new_state.subsamples.record)
    [([0, 0, 0], -1., 53) ([0, 1, 0], -1., 15) ([1, 0, 0], -1., 31) ([1, 1, 1],  1.,  1)]


SimulatedAnnealingSubproblemSampler
-----------------------------------

This example works on a binary quadratic model of two AND gates in series
by sampling a BQM representing just one of the gates. Output :math:`z` of gate
:math:`z = x \wedge y` connects to input :math:`a` of gate :math:`c = a \wedge b`.
An initial state is manually set with invalid solution :math:`x=y=0, z=1; a=b=1, c=0`.
The state is updated by sampling the subproblem 10 times.
The execution results shown here were valid solutions to the subproblem; for
example, :math:`x=0, y=1, z=0`.

.. code-block:: python

    import dimod
    from hybrid.samplers import SimulatedAnnealingSubproblemSampler
    from hybrid.core import State

    # Define a problem and a subproblem
    bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0, 'a': 2.0, 'b': 0.0, 'c': 6.0},
                                    {('y', 'x'): 2.0, ('z', 'x'): -4.0, ('z', 'y'): -4.0,
                                    ('b', 'a'): 2.0, ('c', 'a'): -4.0, ('c', 'b'): -4.0, ('a', 'z'): -4.0},
                                    -1.0, 'BINARY')
    sub_bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0},
                                        {('x', 'y'): 2.0, ('x', 'z'): -4.0, ('y', 'z'): -4.0},
                                        -1.0, dimod.Vartype.BINARY)

    # Set up the sampler with an initial state
    sampler = SimulatedAnnealingSubproblemSampler(num_reads=10)
    state = State.from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm)
    state.update(subproblem=sub_bqm)

::

    # Sample the subproblem (REPL)
    >>> new_state = sampler.run(state).result()
    >>> print(new_state.subsamples.record)
    [([0, 1, 0], -1., 1) ([0, 1, 0], -1., 1) ([0, 0, 0], -1., 1)
    ([0, 0, 0], -1., 1) ([0, 0, 0], -1., 1) ([1, 0, 0], -1., 1)
    ([1, 0, 0], -1., 1) ([0, 0, 0], -1., 1) ([0, 1, 0], -1., 1)
    ([1, 0, 0], -1., 1)]


TabuSubproblemSampler
---------------------

This example works on a binary quadratic model of two AND gates in series
by sampling a BQM representing just one of the gates. Output :math:`z` of gate
:math:`z = x \wedge y` connects to input :math:`a` of gate :math:`c = a \wedge b`.
An initial state is manually set with invalid solution :math:`x=y=0, z=1; a=b=1, c=0`.
The state is updated by a tabu search on the subproblem.
The execution results shown here was a valid solution to the subproblem:
example, :math:`x=0, y=1, z=0`.

.. code-block:: python

    import dimod
    from hybrid.samplers import TabuSubproblemSampler
    from hybrid.core import State

    # Define a problem and a subproblem
    bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0, 'a': 2.0, 'b': 0.0, 'c': 6.0},
                                     {('y', 'x'): 2.0, ('z', 'x'): -4.0, ('z', 'y'): -4.0,
                                      ('b', 'a'): 2.0, ('c', 'a'): -4.0, ('c', 'b'): -4.0, ('a', 'z'): -4.0},
                                     -1.0, 'BINARY')
    sub_bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0},
                                         {('x', 'y'): 2.0, ('x', 'z'): -4.0, ('y', 'z'): -4.0},
                                         -1.0, dimod.Vartype.BINARY)

    # Set up the sampler with an initial state
    sampler = TabuSubproblemSampler(tenure=2, timeout=5)
    state = State.from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm)
    state.update(subproblem=sub_bqm)

::

    >>> # Sample the subproblem (REPL)
    >>> print(new_state.subsamples.record)
    [([0, 1, 0], -1., 1)]


TabuProblemSampler
------------------

This example works on a binary quadratic model of two AND gates in series, where
output :math:`z` of gate :math:`z = x \wedge y` connects to input :math:`a`
of gate :math:`c = a \wedge b`. An initial state is manually set with invalid
solution :math:`x=y=0, z=1; a=b=1, c=0`. The state is updated by a tabu search.
The execution results shown here was a valid solution to the problem:
example, :math:`x=y=z=a=b=c=1`.

.. code-block:: python

    import dimod
    from hybrid.samplers import TabuProblemSampler
    from hybrid.core import State

    # Define a problem
    bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0, 'a': 2.0, 'b': 0.0, 'c': 6.0},
                                     {('y', 'x'): 2.0, ('z', 'x'): -4.0, ('z', 'y'): -4.0,
                                      ('b', 'a'): 2.0, ('c', 'a'): -4.0, ('c', 'b'): -4.0, ('a', 'z'): -4.0},
                                     -1.0, 'BINARY')

    # Set up the sampler with an initial state
    sampler = TabuProblemSampler(tenure=2, timeout=5)
    state = State.from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm)

::

    # Sample the problem (REPL)
    >>> new_state = sampler.run(state).result()
    >>> print(new_state.samples)
    Response(rec.array([([1, 1, 1, 1, 1, 1], -1., 1)],
        dtype=[('sample', 'i1', (6,)), ('energy', '<f8'), ('num_occurrences', '<i4')]),
        ['a', 'b', 'c', 'x', 'y', 'z'], {}, 'BINARY')


InterruptableTabuSampler
------------------------

This example works on a binary quadratic model of two AND gates in series, where
output :math:`z` of gate :math:`z = x \wedge y` connects to input :math:`a`
of gate :math:`c = a \wedge b`. An initial state is manually set with invalid
solution :math:`x=y=0, z=1; a=b=1, c=0`. The state is updated by a tabu search.
The execution results shown here was a valid solution to the problem:
example, :math:`x=y=z=a=b=c=1`.

.. code-block:: python

    import dimod
    from hybrid.samplers import InterruptableTabuSampler
    from hybrid.core import State

    # Define a problem
    bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0, 'a': 2.0, 'b': 0.0, 'c': 6.0},
                                     {('y', 'x'): 2.0, ('z', 'x'): -4.0, ('z', 'y'): -4.0,
                                      ('b', 'a'): 2.0, ('c', 'a'): -4.0, ('c', 'b'): -4.0, ('a', 'z'): -4.0},
                                     -1.0, 'BINARY')

    # Set up the sampler with an initial state
    sampler = InterruptableTabuSampler(tenure=2, quantum_timeout=30, timeout=5000)
    state = State.from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm)

::

    >>> # Sample the problem (REPL)
    >>> new_state = sampler.run(state)
    >>> new_state
    <Future at 0x179eae59898 state=running>
    >>> sampler.stop()
    >>> new_state
    <Future at 0x179eae59898 state=finished returned State>
    >>> print(new_state.result())
    State(samples=Response(rec.array([([1, 1, 1, 1, 1, 1], -1., 1)],
        dtype=[('sample', 'i1', (6,)), ('energy', '<f8'), ('num_occurrences', '<i4')]),
        ['a', 'b', 'c', 'x', 'y', 'z'], {}, 'BINARY'))


RandomSubproblemSampler
-----------------------

This example works on a binary quadratic model of two AND gates in series
by sampling a BQM representing just one of the gates. Output :math:`z` of gate
:math:`z = x \wedge y` connects to input :math:`a` of gate :math:`c = a \wedge b`.
An initial state is manually set with invalid solution :math:`x=y=0, z=1; a=b=1, c=0`.
The state is updated with a random sample..

.. code-block:: python

    import dimod
    from hybrid.samplers import RandomSubproblemSampler
    from hybrid.core import State

    # Define a problem and a subproblem
    bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0, 'a': 2.0, 'b': 0.0, 'c': 6.0},
                                     {('y', 'x'): 2.0, ('z', 'x'): -4.0, ('z', 'y'): -4.0,
                                      ('b', 'a'): 2.0, ('c', 'a'): -4.0, ('c', 'b'): -4.0, ('a', 'z'): -4.0},
                                     -1.0, 'BINARY')
    sub_bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0},
                                         {('x', 'y'): 2.0, ('x', 'z'): -4.0, ('y', 'z'): -4.0},
                                         -1.0, dimod.Vartype.BINARY)

    # Set up the sampler with an initial state
    sampler = RandomSubproblemSampler()
    state = State.from_sample({'x': 0, 'y': 0, 'z': 1, 'a': 1, 'b': 1, 'c': 0}, bqm)
    state.update(subproblem=sub_bqm)

::

    # Sample the subproblem a couple of times (REPL)
    >>> new_state = sampler.run(state).result()
    >>> print(new_state.subsamples.record)
    [([0, 0, 0], -1., 1)]
    >>> new_state = sampler.run(state).result()
    >>> print(new_state.subsamples.record)
    [([1, 1, 1], 1., 1)]
