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

.. Some trait verification is applied to :class:`~hybrid.core.Runnable` objects, at a minimum
   verifying correct inputs and outputs (full verification on some branches may be overly complex
   to implement). If you develop advanced flow-control classes, and override an :code:`__init__` method,
   call the parent class's :code:`__init__` with a `super` construct as done in the subclasses
   of the :class:`~hybrid.traits.StateTraits` class.

<Unedited> State traits verification is done for all :class:`~hybrid.core.Runnable` objects that inherit 
from :class:`~hybrid.traits.StateTraits` or its subclasses:
(1) Minimal checks at workflow (composition of runnables) construction
(2) Definite checks at run time.
Constructing composite traits when composing runnables might not be trivial. The parent runnable
must express child's traits and modify them at construction time.
All built-in runnables declare state traits requirements which are either independent (for simple runnables)
or derived from the child workflow. It's recommended that developers declare state traits requirements,
but, especially for advanced flow-control runnables, that might be too burdensome.

The :ref:`conversion` section describes the :class:`~hybrid.core.HybridRunnable`
class you can use to produce a :class:`~hybrid.core.Runnable` sampler based on
a :std:doc:`dimod <dimod:index>` sampler.

The :ref:`utilities` section provides a list of useful utility methods.
