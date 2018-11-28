============
Introduction
============

**dwave-hybrid** provides a framework for iterating arbitrary-sized sets of samples
through parallel solvers to find an optimal solution.

Section `Reference Hybrid Sampler: Kerberos`_ below demonstrates using a provided
reference sampler built with this framework, Kerberos, to solve a problem too large
to :term:`minor-embed` on the D-Wave system. Section `Example Sampler Designs`_ below
shows an example of using the framework to build a hybrid sampler that, similar to
:std:doc:`qbsolv <qbsolv:index>`, can employ tabu search on a whole problem
while submitting parts of the problem to a D-Wave system.

Framework Overview
==================

The :ref:`HybridBlockDiagram` figure below shows an example configuration. Samples
are iterated over four parallel solvers. The top branch represents a classical tabu
search that runs on the entire problem until interrupted by another branch completing.
These use different decomposers to parcel out parts of the current sample
set (iteration :math:`i`) to samplers such as a D-Wave system (second-highest branch)
or another structure of parallel simulated annealing and tabu search. A generic
representation of a branch's components---decomposer, sampler, and composer---is
shown in the lowest branch. A user-defined criterion selects from current samples
and solver outputs a sample set for iteration :math:`i+1`.

.. figure:: ./_static/HybridBlockDiagram.png
  :name: HybridBlockDiagram
  :scale: 70 %
  :alt: Block diagram

  Schematic Representation

The framework provides classes of infrastructure and building-blocks for
easily configuring experimental hybrid, asynchronous decomposition samplers.
:class:`.SimpleIterator` iterates over a
user-defined structure: the :class:`.RacingBranches` that runs parallel
:class:`.Branch` classes. These contain :class:`.Runnable` components---decomposers,
samplers, and composers---that process a :class:`.SampleSet` of current (initial
or best) samples and maintain a computation :class:`.State`. Based on the outputs
of these parallel branches, :class:`.ArgMinFold` selects a new current sample.

Example Sampler Designs
=======================

This first example uses the framework to solve a small problem of a triangle graph
of nodes identically coupled. This simple example uses only the classical tabu search
algorithm run on the entire problem. Only a sampler :class:`.Runnable` component and
:class:`.State` building blocks are needed.

>>> import dimod
>>> from hybrid.samplers import TabuProblemSampler
>>> from hybrid.core import State
>>> # Define a problem
>>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': 0.5, 'bc': 0.5, 'ca': 0.5})
>>> # Set up the sampler with an initial state
>>> sampler = TabuProblemSampler(tenure=2, timeout=5)
>>> state = State.from_sample({'a': 0, 'b': 0, 'c': 0}, bqm)
>>> # Sample the problem
>>> new_state = sampler.run(state).result()
>>> print(new_state.samples)
SampleSet(rec.array([([ 1, -1, -1], -0.5, 1)],
          dtype=[('sample', 'i1', (3,)), ('energy', '<f8'), ('num_occurrences', '<i4')]),
          ['a', 'b', 'c'], {}, 'SPIN')

This next example solves a binary quadratic model by iteratively producing best samples.
Similar to :std:doc:`qbsolv <qbsolv:index>`, it employs both tabu search on the entire
problem and a D-Wave system on subproblems. In addition to building-block components
such as employed above, this example also uses infrastructure classes to manage the
decomposition and parallel running of branches.


.. include:: ../README.rst
  :start-after: example-start-marker
  :end-before: example-end-marker

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
>>> with open('problems/random-chimera/8192.01.qubo') as problem:
...     bqm = dimod.BinaryQuadraticModel.from_coo(problem)
>>> len(bqm)
8192
>>> solution = KerberosSampler().sample(bqm, max_iter=10, convergence=3)
>>> solution.first.energy     # doctest: +SKIP
-4647.0
