=====
Hades
=====

.. index-start-marker

A general, minimal Python framework for building hybrid asynchronous decomposition
samplers for quadratic unconstrained binary optimization (QUBO) problems.
It facilitates experimentation with structures and parameters for
tailoring a decomposition solver to a problem.

The framework enables rapid development and insight into expected performance
of productized versions of its experimental prototypes.
It does not provide real-time performance.

.. index-end-marker


Installation or Building
========================

.. installation-start-marker

**Package not yet available on PyPI.** Install in developer (edit) mode::

    pip install -e git+https://github.com/dwavesystems/hades.git#egg=hades

or from source::

    git clone https://github.com/dwavesystems/hades.git
    cd hades
    python setup.py install

.. installation-end-marker


Example
=======

.. example-start-marker

.. code-block:: python

    import dimod
    from hades.samplers import (
        QPUSubproblemAutoEmbeddingSampler, InterruptableTabuSampler)
    from hades.decomposers import EnergyImpactDecomposer
    from hades.composers import SplatComposer
    from hades.core import State
    from hades.flow import RacingBranches, ArgMinFold, SimpleIterator
    from hades.utils import min_sample

    # Construct a problem
    bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': -1, 'ca': 1}, 0, dimod.SPIN)

    # Define the solver
    iteration = RacingBranches(
        InterruptableTabuSampler(),
        EnergyImpactDecomposer(max_size=2)
        | QPUSubproblemAutoEmbeddingSampler()
        | SplatComposer()
    ) | ArgMinFold()
    main = SimpleIterator(iteration, max_iter=10, convergence=3)

    # Solve the problem
    init_state = State.from_sample(min_sample(bqm), bqm)
    solution = main.run(init_state).result()

    # Print results
    print("Solution: sample={s.samples.first}, debug={s.debug!r}".format(s=solution))


.. example-end-marker
