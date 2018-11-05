============
Introduction
============

**dwave-hybrid** provides a framework for iterating arbitrary-sized sets of samples
through parallel solvers to find an optimal solution.


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

The framework provides infrastructure classes and building-block classes for
easily configuring experimental hybrid, asynchronous decomposition samplers.
:class:`.SimpleIterator` iterates over a
user-defined structure: the :class:`.RacingBranches` that runs parallel
:class:`.Branch` classes. These contain :class:`.Runnable` components---decomposers,
samplers, and composers---that process a :class:`.SampleSet` of current (initial
or best) samples and maintain a computation :class:`.State`. Based on the outputs
of these parallel branches, :class:`.ArgMinFold` selects a new current sample.

You can use these to customize your components.

Example
=======

This example solves a binary quadratic model by iteratively producing the best samples


.. include:: ../README.rst
  :start-after: example-start-marker
  :end-before: example-end-marker
