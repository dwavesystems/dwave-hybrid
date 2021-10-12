.. image:: https://badge.fury.io/py/dwave-hybrid.svg
    :target: https://badge.fury.io/py/dwave-hybrid
    :alt: Last version on PyPI

.. image:: https://circleci.com/gh/dwavesystems/dwave-hybrid.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/dwave-hybrid
    :alt: Linux/Mac build status

.. image:: https://ci.appveyor.com/api/projects/status/porqyytww2elwjv8/branch/master?svg=true
    :target: https://ci.appveyor.com/project/dwave-adtt/dwave-hybrid/branch/master
    :alt: Windows build status

.. image:: https://img.shields.io/codecov/c/github/dwavesystems/dwave-hybrid/master.svg
    :target: https://codecov.io/gh/dwavesystems/dwave-hybrid
    :alt: Code coverage

.. image:: https://readthedocs.com/projects/d-wave-systems-dwave-hybrid/badge/?version=latest
    :target: https://docs.ocean.dwavesys.com/projects/hybrid/en/latest/
    :alt: Documentation status

.. image:: https://img.shields.io/pypi/pyversions/dwave-hybrid.svg?style=flat
    :target: https://pypi.org/project/dwave-hybrid/
    :alt: PyPI - Python Version


=============
D-Wave Hybrid
=============

.. index-start-marker

A general, minimal Python framework for building hybrid asynchronous decomposition
samplers for quadratic unconstrained binary optimization (QUBO) problems.

*dwave-hybrid* facilitates three aspects of solution development:

*   Hybrid approaches to combining quantum and classical compute resources
*   Evaluating a portfolio of algorithmic components and problem-decomposition strategies
*   Experimenting with workflow structures and parameters to obtain the best application results

The framework enables rapid development and insight into expected performance
of productized versions of its experimental prototypes.

Your optimized algorithmic components and other contributions to this project are welcome!

.. index-end-marker


Installation or Building
========================

.. installation-start-marker

Install from a package on PyPI::

    pip install dwave-hybrid

or from source::

    git clone https://github.com/dwavesystems/dwave-hybrid.git
    cd dwave-hybrid
    pip install -r requirements.txt
    python setup.py install

.. installation-end-marker


Example
=======

.. example-start-marker

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


.. example-end-marker


License
=======

Released under the Apache License 2.0. See `<LICENSE>`_ file.

Contributing
============

Ocean's `contributing guide <https://docs.ocean.dwavesys.com/en/stable/contributing.html>`_
has guidelines for contributing to Ocean packages.
