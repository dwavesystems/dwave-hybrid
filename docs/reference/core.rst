.. _core:

==========
Primitives
==========

Basic building-block classes and superclasses for hybrid workflows.

.. automodule:: hybrid.core

Classes
=======

.. autoclass:: Present
.. autoclass:: Runnable
.. autoclass:: State
.. autoclass:: States


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

   Runnable.dispatch
   Runnable.error
   Runnable.init
   Runnable.halt
   Runnable.next
   Runnable.run
   Runnable.stop
   State.updated
   State.from_sample
   State.from_samples
