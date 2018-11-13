.. _core:

===============
Building Blocks
===============

Building-block classes that can be used with the infrastructure classes
provided by the dwave-hybrid framework for configuring hybrid, asynchronous
decomposition samplers.

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

   SampleSet.first


Methods
=======

.. autosummary::
   :toctree: generated/

   Runnable.run
   Runnable.stop
   State.updated
   State.from_sample
   State.from_samples
