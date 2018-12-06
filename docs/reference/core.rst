.. _core:

==========
Primitives
==========

Basic building-block classes and superclasses for hybrid workflows.

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
