=====
Hades
=====

.. index-start-marker

Minimal and general Python framework for building hybrid asynchronous
decomposition samplers for quadratic unconstrained binary optimization (QUBO)
problems. Enables users to easily experiment with different
choices (structural and parametric), thus tailoring the decomposition solver
algorithm to their specific problem. Helps with rapid development and experimenting,
not real time/wall clock performance (rather: offline performance).

.. index-end-marker


Installation or Building
========================

.. installation-start-marker

Package not yet available on PyPI. Install in developer (edit) mode::

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

    # construct a problem
    bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': -1, 'ca': 1}, 0, dimod.SPIN)

    # define the solver
    iteration = RacingBranches(
        InterruptableTabuSampler(bqm),
        EnergyImpactDecomposer(bqm, max_size=2)
        | QPUSubproblemAutoEmbeddingSampler()
        | SplatComposer(bqm)
    ) | ArgMinFold()
    main = SimpleIterator(iteration, max_iter=10, convergence=3)

    # run solver
    init_state = State.from_sample(min_sample(bqm), bqm)
    solution = main.run(init_state).result()

    # show results
    print("Solution: sample={s.samples.first}, debug={s.debug!r}".format(s=solution))


.. example-end-marker
