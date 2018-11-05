.. _core:

===============
Building Blocks
===============

.. automodule:: hybrid.core

Classes
=======

.. autoclass:: Branch
.. autoclass:: Present
.. autoclass:: Runnable
.. autoclass:: SampleSet
.. autoclass:: State


Properties
==========

.. autosummary::
   :toctree: generated/

   Branch.name
   Runnable.name
   SampleSet.first


Methods
=======

.. autosummary::
   :toctree: generated/

   Branch.iterate
   Branch.stop
   Present.done
   Present.result
   Runnable.iterate
   Runnable.run
   Runnable.stop
   State.copy
   State.from_sample
   State.updated
