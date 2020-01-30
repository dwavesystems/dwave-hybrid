.. _reference_examples_hybrid:

==================
Reference Examples
==================

The `examples <https://github.com/dwavesystems/dwave-hybrid/tree/master/examples>`_
directory of the code includes implementations of some :ref:`reference_workflows`
you can incorporate as provided into your application and also use to jumpstart
your development of custom workflows.

A typical first use of dwave-hybrid might be to simply use the Kerberos reference
sampler to solve a QUBO, as shown in :ref:`using_framework`. Next, you might tune its configurable
parameters, described under :ref:`reference_workflows`.

To further improve performance, you can step up from using a generic
workflow to one tailored for your application and its problem. As a first step you can
modify a reference workflow with existing components. After that, you can implement your
own components as described in :ref:`developing_hybrid`.
