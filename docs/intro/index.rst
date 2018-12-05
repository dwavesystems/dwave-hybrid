.. _intro:

============
Introduction
============

**dwave-hybrid** provides a framework for iterating arbitrary-sized sets of samples
through parallel solvers to find an optimal solution.

For the documentation of a particular code element, see the :ref:`reference` section. 
This introduction gives an overview of the package; steps you through using it,
starting with running a provided hybrid solver that handles arbitrary-sized QUBOs; and
points out the way to developing your own components in the framework.

* :ref:`overview` presents the framework and explains key concepts.
* :ref:`using` shows how to use the framework. You can quickly get started by using a
  provided reference sampler built with this framework, Kerberos, to solve a problem too
  large to :term:`minor-embed` on a D-Wave system. Next, use the framework
  to build (hybrid) workflows; for example, a solver similar to :std:doc:`qbsolv <qbsolv:index>`,
  which can employ tabu search on a whole problem while submitting parts of the problem
  to a D-Wave system.

* :ref:`developing` guides you to developing your own hybrid components.

.. toctree::
   :maxdepth: 2
   :hidden:

   overview
   using
   developing
