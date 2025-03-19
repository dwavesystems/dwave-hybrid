.. image:: https://badge.fury.io/py/dwave-hybrid.svg
    :target: https://badge.fury.io/py/dwave-hybrid
    :alt: Latest version on PyPI

.. image:: https://circleci.com/gh/dwavesystems/dwave-hybrid.svg?style=shield
    :target: https://circleci.com/gh/dwavesystems/dwave-hybrid
    :alt: Linux/MacOS/Windows build status

.. image:: https://img.shields.io/codecov/c/github/dwavesystems/dwave-hybrid/master.svg
    :target: https://codecov.io/gh/dwavesystems/dwave-hybrid
    :alt: Code coverage

.. image:: https://img.shields.io/pypi/pyversions/dwave-hybrid.svg?style=flat
    :target: https://pypi.org/project/dwave-hybrid/
    :alt: Supported Python versions


============
dwave-hybrid
============

.. start_hybrid_about

A general, minimal Python framework for building hybrid asynchronous
decomposition samplers for quadratic unconstrained binary optimization (QUBO)
problems.

*dwave-hybrid* facilitates three aspects of solution development:

*   Hybrid approaches to combining quantum and classical compute resources
*   Evaluating a portfolio of algorithmic components and problem-decomposition
    strategies
*   Experimenting with workflow structures and parameters to obtain the best
    application results

The framework enables rapid development and insight into expected performance
of productized versions of its experimental prototypes.

Your optimized algorithmic components and other contributions to this project
are welcome!

.. end_hybrid_about


Installation or Building
========================

Install from a package on PyPI::

    pip install dwave-hybrid

or from source in development mode::

    git clone https://github.com/dwavesystems/dwave-hybrid.git
    cd dwave-hybrid
    pip install -e .


Testing
=======

Install test requirements and run ``unittest``::

    pip install -r tests/requirements.txt
    python -m unittest


Example
=======

.. start_hybrid_example

.. code-block:: python

    import dimod
    import hybrid

    # Construct a problem
    bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': -1, 'ca': 1}, 0, dimod.SPIN)

    # Define the workflow
    iteration = hybrid.RacingBranches(
        hybrid.InterruptableTabuSampler(),
        hybrid.EnergyImpactDecomposer(size=2)
        | hybrid.QPUSubproblemAutoEmbeddingSampler()
        | hybrid.SplatComposer()
    ) | hybrid.ArgMin()
    workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=3)

    # Solve the problem
    init_state = hybrid.State.from_problem(bqm)
    final_state = workflow.run(init_state).result()

    # Print results
    print("Solution: sample={.samples.first}".format(final_state))

.. end_hybrid_example


Documentation
=============

Documentation for latest stable release included in Ocean is available
`here <https://docs.dwavequantum.com/en/latest/ocean/api_ref_hybrid>`_.


License
=======

Released under the Apache License 2.0. See `<LICENSE>`_ file.


Contributing
============

Ocean's `contributing guide <https://docs.dwavequantum.com/en/latest/ocean/contribute.html>`_
has guidelines for contributing to Ocean packages.