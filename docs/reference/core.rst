.. _core_hybrid:

==========
Primitives
==========

Basic building-block classes and superclasses for hybrid workflows.

.. automodule:: hybrid.core

Classes
=======

.. autoclass:: State
.. autoclass:: States
.. autoclass:: SampleSet
.. autoclass:: Runnable


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
   SampleSet.empty
   SampleSet.hstack
   SampleSet.vstack
   State.copy
   State.updated
   State.result
   State.from_problem
   State.from_subproblem
   State.from_sample
   State.from_samples
   State.from_subsample
   State.from_subsamples
   States.first
   States.updated
