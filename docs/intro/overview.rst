.. _overview:

========
Overview
========

The *dwave-hybrid* framework enables you to quickly design and test workflows that
iterate sets of samples through samplers to solve arbitrary QUBOs. Large problems
can be decomposed and two or more solution techniques can run in parallel.

The :ref:`HybridBlockDiagram` figure below shows an example configuration. Samples
are iterated over four parallel solvers. The top **branch** represents a classical tabu
search that runs on the entire problem until interrupted by another branch completing.
These use different decomposers to parcel out parts of the current sample
set (iteration :math:`i`) to samplers such as a D-Wave system (second-highest branch)
or another structure of parallel simulated annealing and tabu search. A generic
representation of a branch's components---decomposer, sampler, and composer---is
shown in the lowest branch. A user-defined criterion selects from current samples
and solver outputs a sample set for iteration :math:`i+1`.

.. figure:: ../_static/HybridBlockDiagram.png
  :name: HybridBlockDiagram
  :scale: 70 %
  :alt: Block diagram

  Schematic Representation

You can use the framework to run a provided hybrid solver or to configure workflows using
provided components such as tabu samplers and energy-based decomposers.

You can also use the framework to build your own components to incorporate into your
workflow.
