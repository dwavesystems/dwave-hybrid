.. _developing:

=========================
Developing New Components
=========================

The *dwave-hybrid* framework enables you to build your own components to incorporate into your
workflow.

The key superclass is the :class:`~hybrid.core.Runnable` class: the basic components---samplers,
decomposers, composers---and the flow-structuring components such as branches inherit
from this class. A :class:`~hybrid.core.Runnable` is run for an iteration in which it updates
the :class:`~hybrid.core.State` it receives. Typical methods are `run` or `next` to execute an
iteration and `stop` to terminate the :class:`~hybrid.core.Runnable`.

The :ref:`core` and :ref:`flow` sections describe, respectively the basic :class:`~hybrid.core.Runnable`
classes (building blocks) and flow-structuring ones and their methods. If you are
implementing these methods for your own :class:`~hybrid.core.Runnable` class, see comments in
the code.

The :ref:`conversion` section describes the :class:`~hybrid.core.HybridRunnable`
class you can use to produce a :class:`~hybrid.core.Runnable` sampler based on
a :std:doc:`dimod <dimod:index>` sampler.

The :ref:`utilities` section provides a list of useful utility methods. 
