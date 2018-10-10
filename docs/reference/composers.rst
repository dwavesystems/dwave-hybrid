.. _composers:

=========
Composers
=========

.. automodule:: hades.composers

Class
=====

.. autoclass:: SplatComposer

Examples
========

SplatComposer
-------------

This example runs one iteration of a `SplatComposer`, overwriting an initial
solution to a 6-variable binary quadratic model of all zeros with a solution to
a 3-variable subproblem that was manually set to all ones.

.. code-block:: python

    import dimod
    from hades.composers import SplatComposer
    from hades.core import State, SampleSet
    from hades.utils import min_sample

    bqm = dimod.BinaryQuadraticModel({t: 0 for t in range(6)},
                                     {(t, (t+1) % 6): 1 for t in range(6)},
                                     0, 'BINARY')

    composer = SplatComposer()
    state0 = State.from_sample(min_sample(bqm), bqm)
    state1 = state0.updated(subsamples=SampleSet.from_sample({3: 1, 4: 1, 5: 1}, 'BINARY'))

    composed_state = composer.run(state1).result()

::

    >>> print(composed_state.samples)
    Response(rec.array([([0, 0, 0, 1, 1, 1], 1, 2)],
            dtype=[('sample', 'i1', (6,)), ('num_occurrences', '<i8'), ('energy', '<i8')]), [0, 1, 2, 3, 4, 5], {}, 'BINARY')
