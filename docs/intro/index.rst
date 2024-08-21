.. _intro_hybrid:

============
Introduction
============

**dwave-hybrid** provides a framework for iterating arbitrary-sized sets of 
samples through parallel solvers to find an optimal solution.

For the documentation of a particular code element, see the 
:ref:`reference_hybrid` section. This introduction gives an overview of the 
package; steps you through using it, starting with running a provided hybrid
solver that handles arbitrary-sized QUBOs; and points out the way to developing 
your own components in the framework.

*   :ref:`overview_hybrid` presents the framework and explains key concepts.

*   :ref:`using_framework` shows how to use the framework. You can quickly get 
    started by using a provided reference sampler built with this framework, 
    :class:`Kerberos <oceandocs:hybrid.reference.kerberos.KerberosSampler>`, to 
    solve a problem too large to :term:`minor-embed` on a D-Wave system. Next, 
    use the framework to build (hybrid) workflows; for example, a workflow for 
    larger-than-QPU lattice-structured problems.

*   :ref:`developing_hybrid` guides you to developing your own hybrid components.

*   :ref:`reference_examples_hybrid` describes some workflow examples included in the code.

.. toctree::
    :maxdepth: 2
    :hidden:

    overview
    using
    developing
    reference_examples
