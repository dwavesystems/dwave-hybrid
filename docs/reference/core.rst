.. _core:

===============
Building Blocks
===============

Building-block classes for configuring hybrid, asynchronous decomposition samplers,
together with the provided :ref:`infrastructure <flow>` classes.

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

For users of building-block classes alone (not controlled by elements of
:ref:`infrastructure <flow>` classes), the methods of main interest are `run`
and `next` plus methods of generating states. Developers might implement versions
of methods such as `error`.

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


Hybrid/Dimod Conversion
=======================

These classes handle conversion between `dwave-hybrid` and :std:doc:`dimod <dimod:index>`.

Classes
-------

.. autoclass:: HybridSampler
.. autoclass:: HybridRunnable
.. autoclass:: HybridProblemRunnable
.. autoclass:: HybridSubproblemRunnable
