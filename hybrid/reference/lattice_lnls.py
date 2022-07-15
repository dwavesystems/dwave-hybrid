# Copyright 2022 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Greedy large neighborhood local search workflows for lattices."""

import hybrid
import dimod
from dwave.system import DWaveSampler

__all__ = ['LatticeLNLS','LatticeLNLSSampler']


def LatticeLNLS(topology,
                exclude_dims=None,
                qpu_sampler=None,
                energy_threshold=None,
                max_iter=128,
                max_time=None,
                convergence=None,
                qpu_params=None,
                workflow_type='qpu-only'):
    '''Implements lattice workflows as described in `Hybrid quantum annealing for
    larger-than-QPU lattice-structured problems <https://arxiv.org/abs/2202.03044>`_.

    LatticeLNLS workflow is used by :class:`LatticeLNLSSampler`.
    Note that to operate this workflow a minimal set of lattice specific state
    variables must be instantiated:

    1. bqm: A :obj:`~dimod.BinaryQuadraticModel`, with variables indexed geometrically

    2. origin_embeddings: see :class:`~hybrid.decomposers.make_origin_embeddings`

    3. problem_dims: see :class:`~hybrid.decomposers.SublatticeDecomposer`

    Args:

        topology (str):
            A lattice topology (e.g. 'cubic'), consistent with bqm.

            Supported values:

                * 'pegasus' (``qpu_sampler`` must be pegasus-structured)

                * 'cubic' (``qpu_sampler`` must be pegasus of chimera-structured)

                * 'chimera' (``qpu_sampler`` must be chimera-structured)

        qpu_sampler (:class:`dimod.Sampler`, optional, default=DWaveSampler()):
            Sampler such as a D-Wave system.

        qpu_params (dict, optional, default = ``{'num_reads': 25, 'annealing_time': 100}``):
            Dictionary of keyword arguments with values that will be used
            on every call of the QPU sampler.
            A local copy of the parameter is made. If the dictionary does not
            include 'num_reads', it is defaulted as 25, if dictionary
            does not include 'annealing_time', it is defaulted as 100.

        workflow_type (str, optional):
            Options are:

               * 'qpu-only'
                   Default workflow of this paper

               * 'qpu+post-process'
                   Steepest greedy descent over the subspace is run
                   sequentially on samples returned by the QPU.

               * 'qpu+parallel-process'
                   Steepest greedy descent on the full space is run in
                   parallel with the QPU; the best result is accepted on
                   each iteration.

        max_iter (int, optional, default=128):
            Number of iterations in the hybrid algorithm.

        max_time (float/None, optional):
            Wall clock runtime termination criterion. Unlimited by default.

        convergence (int, optional):
            Number of iterations with no improvement that terminates sampling.

        energy_threshold (float, optional):
            Terminate when this energy threshold is surpassed. Check is
            performed at the end of each iteration.

    Returns:
        Workflow (:class:`~hybrid.core.Runnable` instance).

    See also:
        :class:`~hybrid.decomposers.make_origin_embeddings`

        :class:`~hybrid.decomposers.SublatticeDecomposer`

        Jack Raymond et al, `Hybrid quantum annealing for larger-than-QPU
        lattice-structured problems <https://arxiv.org/abs/2202.03044>`_
    '''
    if exclude_dims is None:
        exclude_dims = []
    if qpu_params is None:
        qpu_params = {'num_reads': 25, 'annealing_time': 100}
    if qpu_sampler is None:
        qpu_sampler = DWaveSampler()
    qpu_params0 = qpu_params.copy()
    if 'num_reads' not in qpu_params0:
        qpu_params0['num_reads'] = 25
    if 'annealing_time' not in qpu_params0:
        qpu_params0['annealing_time'] = 100
    qpu_branch = (hybrid.decomposers.SublatticeDecomposer()
                  | hybrid.QPUSubproblemExternalEmbeddingSampler(
                      qpu_sampler=qpu_sampler,
                      sampling_params=qpu_params0,
                      num_reads=qpu_params0['num_reads']))

    if workflow_type == 'qpu-only':
        per_it_runnable =  (qpu_branch| hybrid.SplatComposer())
    elif workflow_type == 'qpu+post-process':
        per_it_runnable = (qpu_branch
                           | hybrid.SteepestDescentSubProblemSampler()
                           | hybrid.SplatComposer())
    elif workflow_type == 'qpu+parallel-process':
        per_it_runnable = (
            hybrid.Parallel(
                qpu_branch | hybrid.SplatComposer(),
                hybrid.SteepestDescentProblemSampler())
            | hybrid.ArgMin())
    else:
        raise ValueError('Unknown workflow type')
    if energy_threshold is not None:
        energy_reached = lambda en: en <= energy_threshold
    else:
        energy_reached = None
    #Iterate to a termination criteria, integrate proposal if energy lowered:
    workflow = hybrid.Loop(per_it_runnable | hybrid.TrackMin(output=True),
                           max_iter=max_iter, terminate=energy_reached,
                           convergence=convergence, max_time=max_time)
    return workflow


class LatticeLNLSSampler(dimod.Sampler):
    """A dimod-compatible hybrid decomposition sampler for problems of lattice
    structure.

    For workflow and lattice related arguments, see:
    :class:`~hybrid.reference.lattice_lnls.LatticeLNLS`.

    Examples:
        This example solves a cubic-structured BQM using the default QPU.
        An 18x18x18 cubic-lattice ferromagnet is created, and sampled
        by the lattice workflow.

        >>> import dimod
        >>> import hybrid
        >>> from dwave.system import DWaveSampler
        >>> topology = 'cubic'
        >>> qpu_sampler = DWaveSampler()                        # doctest: +SKIP
        >>> sampler = hybrid.LatticeLNLSSampler()               # doctest: +SKIP
        >>> cuboid = (18,18,18)
        >>> edge_list = ([((i,j,k),((i+1)%cuboid[0],j,k)) for i in range(cuboid[0])
        ...              for j in  range(cuboid[1]) for k in range(cuboid[2])]
        ...             + [((i,j,k),(i,(j+1)%cuboid[1],k)) for i in range(cuboid[0])
        ...                for j in  range(cuboid[1]) for k in range(cuboid[2])]
        ...             + [((i,j,k),(i,j,(k+1)%cuboid[2])) for i in range(cuboid[0])
        ...                for j in range(cuboid[1]) for k in range(cuboid[2])])
        >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {e: -1 for e in edge_list})
        >>> EGS = -len(edge_list)
        >>> qpu_params={'num_reads': 25,
        ...             'annealing_time': 100,
        ...             'auto_scale': False,
        ...             'chain_strength': 2}
        >>> response = sampler.sample(topology='cubic', bqm=bqm,
        ...                           problem_dims=cuboid,
        ...                           energy_threshold=EGS,
        ...                           qpu_sampler=qpu_sampler,
        ...                           qpu_params=qpu_params)                           # doctest: +SKIP
        >>> response.data_vectors['energy']                      # doctest: +SKIP
        array([-17496])

    See also:
        :class:`~hybrid.decomposers.make_origin_embeddings`

        :class:`~hybrid.decomposers.SublatticeDecomposer`

        Jack Raymond et al, `Hybrid quantum annealing for larger-than-QPU
        lattice-structured problems <https://arxiv.org/abs/2202.03044>`_
    """

    properties = None
    parameters = None
    runnable = None
    origin_embedding = None

    def __init__(self):
        #Minimum requirements for dimod compatibility are used.
        #Certain parameters might be initialized in principle and
        #shared amongst many sampling processes.
        self.parameters = {
            'origin_embeddings': None
        }
        self.properties = {}

    def sample(self, topology, bqm, problem_dims, exclude_dims=None,
               reject_small_problems=True, qpu_sampler=None,
               init_sample=None, num_reads=1, **kwargs):
        """Solve large subspaces of a lattice structured problem sequentially
        integrating proposals greedily to arrive at a global or local minima of
        the bqm.

        Args:
            bqm (:class:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from. Keying of variables
                must be appropriate to the lattice class.

            init_sample (:class:`~dimod.SampleSet`, callable, ``None``):
                Initial sample set (or sample generator) used for each "read".
                Use a random sample for each read by default.

            num_reads (int):
                Number of reads. Each sample is the result of a single run of
                the hybrid algorithm.

            problem_dims (tuple of ints):
                Lattice dimensions (e.g. cubic case (18,18,18)).

            exclude_dims (list of ints, optional):
                Subspaces are selected by geometric displacement. In the case of
                cellular-level displacements only dimensions indexing cell-displacements
                are considered. The defaults are topology dependent:

                * 'chimera': [2,3] (u,k chimera coordinates are not displaced).

                * 'pegasus': [0,3,4] (t,u,k nice pegasus coordinates are not displaced).

                * 'cubic': [] all dimensions are displaced.

            reject_small_problems (bool, optional, default=True):
                If the subsolver is bigger than the target problem, raise an
                error by default (True), otherwise quietly shrink the embedding
                to be no larger than the target problem.

            additional workflow arguments:
                per :class:`~hybrid.reference.lattice_lnls.LatticeLNLS`.

        Returns:
            :class:`~dimod.SampleSet`: A `dimod` :class:`.~dimod.SampleSet` object.

        See also:
            :class:`~hybrid.decomposers.make_origin_embeddings`

            :class:`~hybrid.decomposers.SublatticeDecomposer`

            Jack Raymond et al, `Hybrid quantum annealing for larger-than-QPU
            lattice-structured problems <https://arxiv.org/abs/2202.03044>`_
        """
        if qpu_sampler is None:
            qpu_sampler = DWaveSampler()

        if exclude_dims is None:
            if topology == 'chimera':
                exclude_dims = [2,3]
            elif topology == 'pegasus':
                exclude_dims = [0,3,4]
            else:
                exclude_dims = []
                #Recreate on each call, no reuse:
        self.origin_embeddings = hybrid.make_origin_embeddings(
            qpu_sampler, topology, problem_dims=problem_dims,
            reject_small_problems=reject_small_problems)

        if callable(init_sample):
            init_state_gen = lambda: hybrid.State.from_sample(
                init_sample(),
                bqm,
                problem_dims=problem_dims,
                exclude_dims=exclude_dims,
                origin_embeddings=self.origin_embeddings)
        elif init_sample is None:
            init_state_gen = lambda: hybrid.State.from_sample(
                hybrid.random_sample(bqm),
                bqm,
                problem_dims=problem_dims,
                exclude_dims=exclude_dims,
                origin_embeddings=self.origin_embeddings)
        elif isinstance(init_sample, dimod.SampleSet):
            init_state_gen = lambda: hybrid.State.from_sample(
                init_sample,
                bqm,
                problem_dims=problem_dims,
                exclude_dims=exclude_dims,
                origin_embeddings=self.origin_embeddings)
        else:
            raise TypeError("'init_sample' should be a SampleSet or a SampleSet generator")

        #Recreate on each call, no reuse:
        self.runnable = LatticeLNLS(topology=topology,
                                    qpu_sampler=qpu_sampler,
                                    exclude_dims=exclude_dims,
                                    **kwargs)

        samples = []
        energies = []
        for _ in range(num_reads):
            init_state = init_state_gen()
            final_state = self.runnable.run(init_state)
            # the best sample from each run is one "read"
            ss = final_state.result().samples
            ss.change_vartype(bqm.vartype, inplace=True)
            samples.append(ss.first.sample)
            energies.append(ss.first.energy)

        return dimod.SampleSet.from_samples(samples, vartype=bqm.vartype,
                                            energy=energies)
