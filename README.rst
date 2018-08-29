=====
Hades
=====

.. index-start-marker

Hybrid asynchronous decomposition sampler for quadratic unconstrained binary
optimization (QUBO) problems.

.. index-end-marker

Installation or Building
========================

.. installation-start-marker

A wheel might be available for your system on PyPI. Source distributions are provided as well.

.. code-block:: python

    pip install hybrid-sampler


Alternatively, you can build the library with setuptools.

.. code-block:: bash

    pip install -r python/requirements.txt
    python setup.py install

.. installation-end-marker

Example
=======

.. example-start-marker

>>> import dimod
>>> from kerberos import KerberosSampler
>>> # Create the problem
>>> Q = {(0, 0): -1, (1, 1): -1, (0, 1): 2}
>>> bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset = 0.0)
>>> # Run the solver
>>> solution = KerberosSampler().sample(bqm, max_iter=10, convergence=3)
>>> print(solution)   # doctest: +SKIP
Response(rec.array([([0, 1], -1., 1)],
         dtype=[('sample', 'i1', (2,)), ('energy', '<f8'), ('num_occurrences', '<i4')]),
         [0, 1], {}, 'BINARY')

.. example-end-marker
