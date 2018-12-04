.. _core:

===============
Building Blocks
===============

Building-block classes for hybrid workflows together with the provided
:ref:`infrastructure <flow>` classes.

.. automodule:: hybrid.core

Classes
=======

.. autoclass:: Branch
.. autoclass:: Present
.. autoclass:: Runnable
.. autoclass:: State


Properties
==========

.. autosummary::
   :toctree: generated/

   Runnable.name
   SampleSet.first


Methods
=======

For users, the methods of main interest are `run` (and occasionally `stop`) for
:class:`Runnable` and its subclasses and the methods of generating states for
the :class:`State` class. Developers should understand and might implement versions
of the `init`, `next`, `error`, and `stop` methods.

.. autosummary::
   :toctree: generated/

   Branch.error
   Branch.next
   Branch.stop
   Runnable.dispatch
   Runnable.error
   Runnable.next
   Runnable.run
   Runnable.stop
   State.updated
   State.from_sample
   State.from_samples


Hybrid Runnables and Dimod Samplers
===================================

These classes handle conversion between `dwave-hybrid` and :std:doc:`dimod <dimod:index>`.

Classes
-------

.. autoclass:: HybridSampler
.. autoclass:: HybridRunnable
.. autoclass:: HybridProblemRunnable
.. autoclass:: HybridSubproblemRunnable
