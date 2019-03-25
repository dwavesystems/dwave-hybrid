.. _developing:

=========================
Developing New Components
=========================

The *dwave-hybrid* framework enables you to build your own components to incorporate into your
workflow.

The key superclass is the :class:`~hybrid.core.Runnable` class: all basic components---samplers,
decomposers, composers---and flow-structuring components such as branches inherit
from this class. A :class:`~hybrid.core.Runnable` is run for an iteration in which it updates
the :class:`~hybrid.core.State` it receives. Typical methods are `run` or `next` to execute an
iteration and `stop` to terminate the :class:`~hybrid.core.Runnable`.

The :ref:`core` and :ref:`flow` sections describe, respectively, the basic :class:`~hybrid.core.Runnable`
classes (building blocks) and flow-structuring ones and their methods. If you are
implementing these methods for your own :class:`~hybrid.core.Runnable` class, see comments in
the code.

The :ref:`racingBranches1` graphic below shows the top-down composition (tree structure) of a hybrid loop.

.. figure:: ../_static/tree.png
  :name: Tree
  :scale: 65 %
  :alt: Top-Down Composition

  Top-Down Composition

.. traits-start-marker

State traits are verified for all :class:`~hybrid.core.Runnable` objects that inherit
from :class:`~hybrid.traits.StateTraits` or its subclasses. Verification includes:

(1) Minimal checks of workflow construction (composition of :class:`~hybrid.core.Runnable` classes)
(2) Runtime checks

All built-in :class:`~hybrid.core.Runnable` classes declare state traits requirements that are
either independent (for simple ones) or derived from a child workflow. Traits of a new
:class:`~hybrid.core.Runnable` must be expressed and modified at construction time by its parent.
When developing new :class:`~hybrid.core.Runnable` classes, constructing composite traits can be
nontrivial for some advanced flow-control runnables.

.. traits-end-marker

The :ref:`conversion` section describes the :class:`~hybrid.core.HybridRunnable`
class you can use to produce a :class:`~hybrid.core.Runnable` sampler based on
a :std:doc:`dimod <dimod:index>` sampler.

The :ref:`utilities` section provides a list of useful utility methods.
