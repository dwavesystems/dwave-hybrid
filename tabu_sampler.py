import random

import tabu_solver
from dimod.core.sampler import Sampler
from dimod.response import Response
from dimod.vartypes import Vartype


class TabuSampler(Sampler):
    """A dimod sampler wrapper for the Tabu solver.

    Examples:
        This example solves a two-variable Ising model.

        >>> import dimod
        >>> response = TabuSampler().sample_ising(
        ...                     {'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
        >>> response.data_vectors['energy']
        array([-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5])

    """

    properties = None
    parameters = None

    def __init__(self):
        self.parameters = {'tenure': [],
                           'scale_factor': [],
                           'timeout': [],
                           'num_reads': []}
        self.properties = {}

    def sample(self, bqm, init_solution=None, tenure=None, scale_factor=1, timeout=20, num_reads=1):
        """Run Tabu search on `bqm` and return the best solution found within
        `timeout` milliseconds.

        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.
            init_solution (list, optional):
                List of 0/1 values, which defines the initial state of each variable.
                Defaults to random sample.
            tenure (int, optional):
                Tabu tenure. Defaults to: min(20, num_vars / 4).
            scale_factor (number, optional):
                Scaling factor for biases/couplings in BQM. Internally, BQM is
                converted to QUBO matrix, and elements are stored as long ints
                using ``internal_q = long int (q * scale_factor)``.
            timeout (int, optional):
                Total running time in milliseconds.
            num_reads (int, optional): Number of reads. Each sample is the result of
                a single run of the simulated annealing algorithm.

        Returns:
            :obj:`~dimod.Response`: A `dimod` :obj:`.~dimod.Response` object.

        Examples:
            This example provides samples for a two-variable QUBO model.

            >>> import dimod
            >>> sampler = dimod.TabuSampler()
            >>> Q = {(0, 0): -1, (1, 1): -1, (0, 1): 2}
            >>> bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset = 0.0)
            >>> response = sampler.sample(bqm)
            >>> response.data_vectors['energy']
            array([-1., -1.])

        """

        # input checking and defaults calculation
        if init_solution is not None and len(init_solution) != len(bqm):
            raise ValueError("'init_solution' dimension different from BQM")

        if tenure is None:
            tenure = int(max(min(20, len(bqm) / 4), 1))
        if not isinstance(tenure, int):
            raise TypeError("'tenure' should be an integer in range [1, num_vars - 1]")
        if not 0 < tenure < len(bqm):
            raise ValueError("'tenure' should be an integer in range [1, num_vars - 1]")

        if not isinstance(num_reads, int):
            raise TypeError("'samples' should be a positive integer")
        if num_reads < 1:
            raise ValueError("'samples' should be a positive integer")

        bqm = bqm.change_vartype(Vartype.BINARY, inplace=False)
        qubo = self._bqm_to_tabu_qubo(bqm)

        # run Tabu search
        samples = []
        energies = []
        for _ in range(num_reads):
            if init_solution is None:
                init_sample = [random.randint(0, 1) for _ in range(len(bqm))]
            else:
                init_sample = init_solution
            r = tabu_solver.TabuSearch(qubo, init_sample, tenure, scale_factor, timeout)
            sample = self._tabu_sample_to_bqm_sample(list(r.bestSolution()), bqm)
            energy = bqm.energy(sample)
            samples.append(sample)
            energies.append(energy)

        response = Response.from_dicts(samples, {'energy': energies}, vartype=Vartype.BINARY)
        response.change_vartype(bqm.vartype, inplace=True)
        return response

    def _bqm_to_tabu_qubo(self, bqm):
        varorder = sorted(list(bqm.adj.keys()))
        ud = 0.5 * bqm.to_numpy_matrix(varorder)
        # Note: normally, conversion would be: `ud + ud.T - numpy.diag(ud.diagonal())`,
        # but the Tabu solver we're using requires slightly different qubo matrix.
        symm = ud + ud.T
        qubo = symm.tolist()
        return qubo

    def _tabu_sample_to_bqm_sample(self, sample, bqm):
        varorder = sorted(list(bqm.adj.keys()))
        assert len(sample) == len(varorder)
        return dict(zip(varorder, sample))


if __name__ == "__main__":
    import dimod
    from pprint import pprint

    print("TabuSampler:")
    bqm = dimod.BinaryQuadraticModel({'a': 0.0, 'b': -1.0, 'c': 0.5}, {('a', 'b'): -1.0, ('b', 'c'): 1.5}, 1, dimod.BINARY)
    response = TabuSampler().sample(bqm, num_reads=10)
    pprint(list(response.data()))

    print("ExactSolver:")
    response = dimod.ExactSolver().sample(bqm)
    pprint(list(response.data()))

    print("Sampling Ising:")
    response = TabuSampler().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1}, num_reads=10)
    pprint(list(response.data()))
