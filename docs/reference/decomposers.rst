.. _decomposers:

===========
Decomposers
===========

.. automodule:: hybrid.decomposers

Classes
=======

.. autoclass:: EnergyImpactDecomposer
.. autoclass:: IdentityDecomposer
.. autoclass:: RandomConstraintDecomposer
.. autoclass:: RandomSubproblemDecomposer
.. autoclass:: TilingChimeraDecomposer


Examples
========

EnergyImpactDecomposer
----------------------

This example iterates twice on a 10-variable binary quadratic model with a
random initial sample set. `size` configuration limits the subproblem
in the first iteration to the first 4 variables shown in the output of
`flip_energy_gains`.

.. code-block:: python

    import dimod
    from hybrid.decomposers import EnergyImpactDecomposer
    from hybrid.core import State
    from hybrid.utils import min_sample, flip_energy_gains

    bqm = dimod.BinaryQuadraticModel({t: 0 for t in range(10)},
                                     {(t, (t+1) % 10): 1 for t in range(10)},
                                     0, 'BINARY')

    decomposer = EnergyImpactDecomposer(size=4, rolling=True, rolling_history=1.0)
    state0 = State.from_sample(min_sample(bqm), bqm)

::

    >>> flip_energy_gains(bqm, state0.samples.first.sample)
    [(0, 9), (0, 8), (0, 7), (0, 6), (0, 5), (0, 4), (0, 3), (0, 2), (0, 1), (0, 0)]
    >>> state1 = decomposer.run(state0).result()
    >>> list(state1.subproblem.variables)
    [8, 7, 9, 6]
    >>> state2 = decomposer.run(state1).result()
    >>> list(state2.subproblem.variables)
    [2, 3, 4, 5]


RandomSubproblemDecomposer
--------------------------

This example decomposes a 6-variable binary quadratic model with a
random initial sample set to create a 3-variable subproblem.

.. code-block:: python

    import dimod
    from hybrid.decomposers import RandomSubproblemDecomposer
    from hybrid.core import State
    from hybrid.utils import random_sample

    bqm = dimod.BinaryQuadraticModel(
        {t: 0 for t in range(6)}, {(t, (t+1) % 6): 1 for t in range(6)}, 0, 'BINARY')

    decomposer = RandomSubproblemDecomposer(bqm, size=3)
    state0 = State.from_sample(random_sample(bqm), bqm)
    state1 = decomposer.run(state0).result()

::

    >>> print(state1.subproblem)
    BinaryQuadraticModel({2: 1.0, 3: 0.0, 4: 0.0}, {(2, 3): 1.0, (3, 4): 1.0}, 0.0, Vartype.BINARY)


TilingChimeraDecomposer
-----------------------

This example decomposes a 2048-variable Chimera structured binary quadratic model
read from a file into 2x2x4-lattice subproblems.

.. code-block:: python

    import dimod
    from hybrid.decomposers import TilingChimeraDecomposer
    from hybrid.core import State
    from hybrid.utils import random_sample

    with open('problems/random-chimera/2048.09.qubo', 'r') as fp:
        bqm = dimod.BinaryQuadraticModel.from_coo(fp)

    decomposer = TilingChimeraDecomposer(size=(2,2,4))
    state0 = State.from_sample(random_sample(bqm), bqm)
    state1 = decomposer.run(state0).result()

::

    >>> print(state1.subproblem)
    BinaryQuadraticModel({0: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: -3.0, 1: 0.0, 2: 0.0, 3: -4.0, 1024: -7.0, 1028: 0.0,
    >>> # Snipped above response for brevity
    >>> state1 = decomposer.run(state0).result()
    >>> print(state1.subproblem)
    BinaryQuadraticModel({8: 3.0, 12: 0.0, 13: 2.0, 14: -11.0, 15: -3.0, 9: 4.0, 10: 0.0, 11: 0.0, 1032: 0.0,
    >>> # Snipped above response for brevity


RandomConstraintDecomposer
--------------------------

This example decomposes a 4-variable binary quadratic model that represents
three serial NOT gates into 2-variable subproblems. The expected decomposition
should use variables that represent one of the NOT gates rather than two
arbitrary variables.

.. code-block:: python

    import dimod
    from hybrid.decomposers  RandomConstraintDecomposer
    from hybrid.core import State
    from hybrid.utils import random_sample

    bqm = dimod.BinaryQuadraticModel({'w': -2.0, 'x': -4.0, 'y': -4.0, 'z': -2.0},
                                     {('w', 'x'): 4.0, ('x', 'y'): 4.0, ('y', 'z'): 4.0},
                                     3.0, 'BINARY')

    decomposer = RandomConstraintDecomposer(2, [{'w', 'x'}, {'x', 'y'}, {'y', 'z'}])
    state0 = State.from_sample(random_sample(bqm), bqm)
    state1 = decomposer.run(state0).result()

::

    >>> print(state1.subproblem)
    BinaryQuadraticModel({'z': -2.0, 'y': 0.0}, {('z', 'y'): 4.0}, 0.0, Vartype.BINARY)
