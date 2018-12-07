.. _using:

===================
Using the Framework
===================

This section helps you quickly use a provided reference sampler to solve arbitrary-sized
problems and then shows you how to build (hybrid) workflows using provided components.

Reference Hybrid Sampler: Kerberos
==================================

*dwave-hybrid* includes a reference example sampler built using the framework:
Kerberos is a dimod-compatible hybrid asynchronous decomposition sampler that enables
you to solve problems of arbitrary structure and size. It finds best samples
by running in parallel tabu search, simulated annealing, and D-Wave subproblem sampling
on problem variables that have high-energy impact.

The example below uses Kerberos to solve a large QUBO.

>>> import dimod
>>> from hybrid.reference.kerberos import KerberosSampler
>>> with open('../problems/random-chimera/8192.01.qubo') as problem:
...     bqm = dimod.BinaryQuadraticModel.from_coo(problem)
>>> len(bqm)
8192
>>> solution = KerberosSampler().sample(bqm, max_iter=10, convergence=3)   # doctest: +SKIP
>>> solution.first.energy     # doctest: +SKIP
-4647.0

Building Workflows
==================

As shown in the :ref:`overview` section, you build hybrid solvers by arranging components such
as samplers in a workflow.

Building Blocks
---------------

The basic components---building blocks---you use are based on the :class:`.Runnable`
class: decomposers, samplers, and composers. Such components input a set of samples,
a :class:`.SampleSet`, and output updated samples. A :class:`State` associated
with such an iteration of a component holds the problem, samples, and optionally
additional information.

The following example demonstrates a simple workflow that uses just one :class:`.Runnable`,
a sampler representing the classical tabu search algorithm, to solve a problem
(fully classically, without decomposition). The example solves a small problem of a
triangle graph of nodes identically coupled. An initial :class:`.State` of all-zero
samples is set as a starting point. The solution, `new_state` is derived from a single
iteration of the `TabuProblemSampler` :class:`.Runnable`.

>>> import dimod
>>> # Define a problem
>>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': 0.5, 'bc': 0.5, 'ca': 0.5})
>>> # Set up the sampler with an initial state
>>> sampler = TabuProblemSampler(tenure=2, timeout=5)
>>> state = State.from_sample({'a': 0, 'b': 0, 'c': 0}, bqm)
>>> # Sample the problem
>>> new_state = sampler.run(state).result()
>>> print(new_state.samples)                     # doctest: +SKIP
SampleSet(rec.array([([ 1, -1, -1], -0.5, 1)],
          dtype=[('sample', 'i1', (3,)), ('energy', '<f8'), ('num_occurrences', '<i4')]),
          ['a', 'b', 'c'], {}, 'SPIN')

Flow Structuring
----------------

The framework provides classes for structuring workflows that use the "building-block"
components. As shown in the :ref:`overview` section, you can create a *branch* of :class:`Runnable`
classes; for example :code:`decomposer | sampler | composer`, which delegates part
of a problem to a sampler such as the D-Wave system.

The following example a branch comprising a decomposer, local Tabu solver, and a composer.
A 10-variable binary quadratic model is decomposed by the energy impact of its variables
into a 6-variable subproblem to be sampled twice. An initial state of all -1 values
is set using the utility function :meth:`~hybrid.utils.min_sample`.

>>> import dimod           # Create a binary quadratic model
>>> bqm = dimod.BinaryQuadraticModel({t: 0 for t in range(10)},
...                                  {(t, (t+1) % 10): 1 for t in range(10)},
...                                  0, 'SPIN')
>>> branch = (EnergyImpactDecomposer(max_size=6, min_gain=-10) |
...           TabuSubproblemSampler(num_reads=2) |
...           SplatComposer())
>>> new_state = branch.next(State.from_sample(min_sample(bqm), bqm))
>>> print(new_state.subsamples)      # doctest: +SKIP
Response(rec.array([([-1,  1, -1,  1, -1,  1], -5., 1),
   ([ 1, -1,  1, -1, -1,  1], -5., 1)],
>>> # Above response snipped for brevity

Such :class:`.Branch` classes can be run in parallel using the :class:`.RacingBranches` class.
From the outputs of these parallel branches, :class:`.ArgMin` selects a new current sample.
And instead of a single iteration on the sample set, you can use the :class:`.SimpleIterator`
to iterate a set number of times or until a convergence criteria is met.

This next example solves a binary quadratic model by iteratively producing best samples.
Similar to :std:doc:`qbsolv <qbsolv:index>`, it employs both tabu search on the entire
problem and a D-Wave system on subproblems. In addition to building-block components
such as employed above, this example also uses infrastructure classes to manage the
decomposition and parallel running of branches.


.. include:: ../../README.rst
  :start-after: example-start-marker
  :end-before: example-end-marker
