.. _composers:

=========
Composers
=========

.. automodule:: hybrid.composers

Class
=====

.. autoclass:: SplatComposer
.. autoclass:: GreedyPathMerge

Examples
========

SplatComposer
-------------

This example runs one iteration of a `SplatComposer` composer, overwriting an initial
solution to a 6-variable binary quadratic model of all zeros with a solution to
a 3-variable subproblem that was manually set to all ones.

.. code-block:: python

    import dimod
    from hybrid.composers import SplatComposer
    from hybrid.core import State, SampleSet
    from hybrid.utils import min_sample

    bqm = dimod.BinaryQuadraticModel({t: 0 for t in range(6)},
                                     {(t, (t+1) % 6): 1 for t in range(6)},
                                     0, 'BINARY')

    composer = SplatComposer()
    state0 = State.from_sample(min_sample(bqm), bqm)
    state1 = state0.updated(subsamples=SampleSet.from_samples({3: 1, 4: 1, 5: 1}, 'BINARY', 0.0))

    composed_state = composer.run(state1).result()

::

    >>> print(composed_state.samples)
    Response(rec.array([([0, 0, 0, 1, 1, 1], 1, 2)],
            dtype=[('sample', 'i1', (6,)), ('num_occurrences', '<i8'), ('energy', '<i8')]), [0, 1, 2, 3, 4, 5], {}, 'BINARY')

GreedyPathMerge
---------------

This example runs one iteration of a `GreedyPathMerge` composer on a thesis and antithesis
:class:`~hybrid.core.State` to find a ground state of a square graph.
By inverting the state of variable :math:`d` and :math:`c` in `samples_d` and then variable :math:`a` of
the lowest energy sample of `samples_a` (second sample), the composer finds a path between
these two samples that contains the ground state shown on the right of the top figure.

.. figure:: ../_static/square_problem.png
  :name: SquareProblem
  :scale: 70 %
  :alt: Block diagram

  Square problem with two ground states.

.. figure:: ../_static/square_problem_path.png
  :name: SquareProblemPath
  :scale: 70 %
  :alt: Block diagram

  Path from thesis to antithesis.
.. code-block:: python

    import dimod
    bqm = dimod.BinaryQuadraticModel({}, {'ab': 1.0, 'bc': 1.0, 'cd': 1.0, 'da': 1}, 0, 'SPIN')
    samples_d = {'a': 1, 'b': 1, 'c': -1, 'd': -1}
    samples_a = [{'a': -1, 'b': -1, 'c': 1, 'd': 1}, {'a': -1, 'b': 1, 'c': 1, 'd': 1}]
    states = [hybrid.State.from_samples(samples_d, bqm),
              hybrid.State.from_samples(samples_a, bqm)]
    synthesis = GreedyPathMerge().next(states)

::

    >>> print(synthesis.samples)
           a   b   c   d  energy  num_occ.
       0  +1  +1  +1  +1    -4.0         1
       [ 1 rows, 4 variables ]
