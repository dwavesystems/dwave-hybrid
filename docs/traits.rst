.. _hybrid_traits:

======
Traits
======

.. start_hybrid_traits

State traits are verified for all :class:`~hybrid.core.Runnable` objects that
inherit from :class:`~hybrid.traits.StateTraits` or its subclasses. Verification
includes:

(1) Minimal checks of workflow construction (composition of
    :class:`~hybrid.core.Runnable` classes)
(2) Runtime checks

All built-in :class:`~hybrid.core.Runnable` classes declare state traits
requirements that are either independent (for simple ones) or derived from a
child workflow. Traits of a new :class:`~hybrid.core.Runnable` must be expressed
and modified at construction time by its parent. When developing new
:class:`~hybrid.core.Runnable` classes, constructing composite traits can be
nontrivial for some advanced flow-control runnables.

.. end_hybrid_traits

.. automodule:: hybrid.traits
    :members:
