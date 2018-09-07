.. _core:

===============
Building Blocks
===============

.. automodule:: hades.core

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
   SampleSet.from_response
   SampleSet.from_sample
   SampleSet.from_sample_on_bqm
   State.copy
   State.from_sample
   State.updated
