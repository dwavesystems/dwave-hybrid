# Copyright 2018 D-Wave Systems Inc.
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

import logging
import collections
import random
import itertools
from heapq import heappush, heappop
from functools import partial
from itertools import product

import numpy as np
import networkx as nx

import dimod
from dimod.traversal import connected_components
import dwave_networkx as dnx
import dwave.preprocessing

from hybrid.core import Runnable, State
from hybrid.exceptions import EndOfStream
from hybrid import traits
from hybrid.utils import (
    bqm_induced_by, flip_energy_gains, select_random_subgraph,
    chimera_tiles)

__all__ = [
    'IdentityDecomposer', 'ComponentDecomposer', 'EnergyImpactDecomposer', 
    'RandomSubproblemDecomposer', 'TilingChimeraDecomposer', 
    'RandomConstraintDecomposer', 'RoofDualityDecomposer',
    'SublatticeDecomposer', 'make_origin_embeddings',
]

logger = logging.getLogger(__name__)


class IdentityDecomposer(traits.ProblemDecomposer, traits.SISO, Runnable):
    """Selects a subproblem that is a full copy of the problem."""

    def next(self, state, **runopts):
        return state.updated(subproblem=state.problem)

class ComponentDecomposer(traits.ProblemDecomposer, traits.SISO, Runnable):
    """Selects a subproblem of variables that make up a connected component.

    Args:
        rolling (bool, optional, default=True):
            If True, successive calls for the same problem (with possibly
            different samples) produce subproblems on different components,
            selected by rolling down the list of all components.

        key (callable, optional, default=None):
            Extracts a comparison key from each component of the problem. The 
            comparison keys determine which component will be used to build the 
            subproblem.

        reverse (bool, optional, default=None):
            Only applies when `key` is specified. If False, the components' comparison 
            keys will be sorted in increasing order. If True or unspecified, they will
            be sorted in decreasing order.

        silent_rewind (bool, optional, default=True):
            If False, raises :exc:`EndOfStream` when resetting/rewinding the
            subproblem generator once all components have been used. 
    
    See :ref:`decomposers-examples`.

    """

    def __init__(self, rolling=True, silent_rewind=True, key=None, reverse=None, **runopts):
        super(ComponentDecomposer, self).__init__(**runopts)

        self.rolling = rolling
        self.silent_rewind = silent_rewind
        self.key = key

        if reverse is None:
            self.reverse = True
        else:
            self.reverse = reverse

        self._rolling_bqm = None
        self._iter_components = None

    def __repr__(self):
        return ("{self}(rolling={self.rolling!r}, "
                "silent_rewind={self.silent_rewind!r}, "
                "key={self.key!r}, "
                "reverse={self.reverse!r})").format(self=self)

    def _get_iter_components(self, bqm):
        components = connected_components(bqm)

        if self.rolling and self.key:
            return iter(sorted(components, key=self.key, reverse=self.reverse))
        else:
            return components

    def next(self, state, **runopts):
        silent_rewind = runopts.get('silent_rewind', self.silent_rewind)

        bqm = state.problem

        if bqm.num_variables <= 1:
            return state.updated(subproblem=bqm)

        if self.rolling:
            if bqm != self._rolling_bqm:
                # This is the first time this problem was called
                self._rolling_bqm = bqm
                self._iter_components = self._get_iter_components(bqm)

            try:
                component = next(self._iter_components)
            except StopIteration:
                # We've already used every component in this problem
                if not silent_rewind:
                    self._rolling_bqm = None # Reset to be ready for subsequent call
                    raise EndOfStream

                # Rewind    
                self._iter_components = self._get_iter_components(bqm)
                component = next(self._iter_components)

        else:
            self._iter_components = self._get_iter_components(bqm)

            if self.key is None:
                component = next(self._iter_components)
            else:
                if self.reverse:
                    component = max(self._iter_components, key=self.key)
                else:
                    component = min(self._iter_components, key=self.key)
        
        sample = state.samples.change_vartype(bqm.vartype).first.sample
        subbqm = bqm_induced_by(bqm, component, sample)
        
        return state.updated(subproblem=subbqm)

    
class EnergyImpactDecomposer(traits.ProblemDecomposer, traits.SISO, Runnable):
    """Selects a subproblem of variables maximally contributing to the problem
    energy.

    The selection currently implemented does not ensure that the variables are
    connected in the problem graph.

    Args:
        size (int):
            Nominal number of variables in the subproblem. Actual subproblem can
            be smaller, depending on other parameters (e.g. `min_gain`).

        min_gain (int, optional, default=-inf):
            Minimum reduction required to BQM energy, given the current sample.
            A variable is included in the subproblem only if inverting its
            sample value reduces energy by at least this amount.

        rolling (bool, optional, default=True):
            If True, successive calls for the same problem (with possibly
            different samples) produce subproblems on different variables,
            selected by rolling down the list of all variables sorted by
            decreasing impact.

        rolling_history (float, optional, default=1.0):
            Fraction of the problem size, as a float in range 0.0 to 1.0, that
            should participate in the rolling selection. Once reached,
            subproblem unrolling is reset.
            
        silent_rewind (bool, optional, default=True):
            If False, raises :exc:`EndOfStream` when resetting/rewinding the
            subproblem generator upon the reset condition for unrolling.

        traversal (str, optional, default='energy'):
            Traversal algorithm used to pick a subproblem of `size` variables.
            Options are:

            energy:
                Use the next `size` variables in the list of variables ordered
                by descending energy impact.

            bfs:
                Breadth-first traversal seeded by the next variable in the
                energy impact list.

            pfs:
                Priority-first traversal seeded by variables from the energy
                impact list, proceeding with the variable on the search boundary
                that has the highest energy impact.

    See :ref:`decomposers-examples`.
    """

    @classmethod
    def _energy(cls, bqm, sample, ordered_priority, visited, size):
        return list(itertools.islice(
                (v for v in ordered_priority if v not in visited), size))

    @classmethod
    def _bfs_nodes(cls, graph, source, size, **kwargs):
        """Traverse `graph` with BFS starting from `source`, up to `size` nodes.
        Return an iterator of subgraph nodes (including source node).
        """
        if size < 1:
            return iter(())

        return itertools.chain(
            (source,),
            itertools.islice((v for u, v in nx.bfs_edges(graph, source)), size-1)
        )

    @classmethod
    def _pfs_nodes(cls, graph, source, size, priority):
        """Priority-first traversal of `graph` starting from `source` node,
        returning up to `size` nodes iterable. Node priority is determined by
        `priority(node)` callable. Nodes with higher priority value are
        traversed before nodes with lower priority.
        """
        if size < 1:
            return iter(())

        # use min-heap to implement (max) priority queue
        # use insertion order to break priority tie
        queue = []
        counter = itertools.count()
        push = lambda priority, node: heappush(queue, (-priority, next(counter), node))
        pop = partial(heappop, queue)

        visited = set()
        enqueued = set()
        push(priority(source), source)

        while queue and len(visited) < size:
            _, _, node = pop()

            if node in visited:
                continue

            visited.add(node)

            for neighbor in graph[node]:
                if neighbor not in enqueued:
                    enqueued.add(neighbor)
                    push(priority(neighbor), neighbor)

        return iter(visited)

    @classmethod
    def _iterative_graph_search(cls, bqm, sample, ordered_priority, visited, size, method):
        """Traverse `bqm` graph using multi-start graph search `method`, until
        `size` variables are selected. Each subgraph is seeded from
        `ordered_priority` ordered dictionary.

        Note: a lot of room for optimization. Nx graph could be cached,
        or we could use a search/traverse method (BFS/PFS) which accepts a
        "node mask" - set of visited nodes.
        """
        graph = dimod.to_networkx_graph(bqm)
        graph.remove_nodes_from(visited)

        variables = set()
        order = iter(ordered_priority)

        while len(variables) < size and len(graph):
            # find the next untraversed variable in (energy) order
            try:
                source = next(order)
                while source in visited or source in variables:
                    source = next(order)
            except StopIteration:
                break

            # get a subgraph induced by source
            nodes = list(
                method(graph, source, size - len(variables), priority=ordered_priority.get))
            variables.update(nodes)

            # in next iteration we traverse a reduced BQM graph
            graph.remove_nodes_from(nodes)

        return variables

    def __init__(self, size, min_gain=None,
                 rolling=True, rolling_history=1.0, silent_rewind=True,
                 traversal='energy', **runopts):

        traversers = {
            'energy': self._energy,
            'bfs': partial(self._iterative_graph_search, method=self._bfs_nodes),
            'pfs': partial(self._iterative_graph_search, method=self._pfs_nodes),
        }

        super(EnergyImpactDecomposer, self).__init__(**runopts)

        if rolling and rolling_history < 0.0 or rolling_history > 1.0:
            raise ValueError("rolling_history must be a float in range [0.0, 1.0]")

        if traversal not in traversers:
            raise ValueError("traversal mode not supported: {}".format(traversal))

        self.size = size
        self.min_gain = min_gain
        self.rolling = rolling
        self.rolling_history = rolling_history
        self.silent_rewind = silent_rewind
        self.traverse = traversers[traversal]

        # variables unrolled so far
        self._unrolled_vars = set()
        self._rolling_bqm = None

        # variable energy impact caching
        self._variable_priority = collections.OrderedDict()
        self._prev_sample = None

    def __repr__(self):
        return (
            "{self}(size={self.size!r}, min_gain={self.min_gain!r}, "
            "rolling={self.rolling!r}, rolling_history={self.rolling_history!r}, "
            "silent_rewind={self.silent_rewind!r})"
        ).format(self=self)

    def _rewind_rolling(self, state):
        self._unrolled_vars.clear()
        self._rolling_bqm = state.problem
        self._rolling_sample = state.sample

    def next(self, state, **runopts):
        # run time options override
        silent_rewind = runopts.get('silent_rewind', self.silent_rewind)

        bqm = state.problem
        sample = state.samples.change_vartype(bqm.vartype).first.sample

        size = self.size
        if size > len(bqm):
            logger.debug("{} subproblem size greater than the problem size, "
                         "adapting to problem size".format(self.name))
            size = len(bqm)

        bqm_changed = bqm != self._rolling_bqm
        sample_changed = sample != self._prev_sample

        if bqm_changed:
            self._rewind_rolling(state)

        if sample_changed:
            self._prev_sample = sample

        # cache energy impact calculation per (bqm, sample)
        if bqm_changed or sample_changed or not self._variable_priority:
            impact = flip_energy_gains(bqm, sample, min_gain=self.min_gain)
            self._variable_priority = collections.OrderedDict((v, en) for en, v in impact)

        if self.rolling:
            if len(self._unrolled_vars) >= self.rolling_history * len(bqm):
                logger.debug("{} reset rolling at unrolled history size {}".format(
                    self.name, len(self._unrolled_vars)))
                self._rewind_rolling(state)
                # reset before exception, to be ready on a subsequent call
                if not silent_rewind:
                    raise EndOfStream

        # pick variables for the next subproblem
        next_vars = self.traverse(bqm, sample,
                                  ordered_priority=self._variable_priority,
                                  visited=self._unrolled_vars,
                                  size=size)

        logger.debug("{} selected {} subproblem variables: {!r}".format(
            self.name, len(next_vars), next_vars))

        if self.rolling:
            self._unrolled_vars.update(next_vars)

        # induce sub-bqm based on selected variables and global sample
        subbqm = bqm_induced_by(bqm, next_vars, sample)
        return state.updated(subproblem=subbqm)


class RandomSubproblemDecomposer(traits.ProblemDecomposer, traits.SISO, Runnable):
    """Selects a subproblem of `size` random variables.

    The selection currently implemented does not ensure that the variables are
    connected in the problem graph.

    Args:
        size (int):
            Number of variables in the subproblem.

    See :ref:`decomposers-examples`.
    """

    def __init__(self, size, **runopts):
        super(RandomSubproblemDecomposer, self).__init__(**runopts)

        self.size = size

    def __repr__(self):
        return "{self}(size={self.size!r})".format(self=self)

    def next(self, state, **runopts):
        bqm = state.problem

        size = self.size
        if size > len(bqm):
            logger.debug("{} subproblem size greater than the problem size, "
                         "adapting to problem size".format(self.name))
            size = len(bqm)

        variables = select_random_subgraph(bqm, size)
        sample = state.samples.change_vartype(bqm.vartype).first.sample
        subbqm = bqm_induced_by(bqm, variables, sample)

        logger.debug("{} selected {} subproblem variables: {!r}".format(
            self.name, len(variables), variables))
        return state.updated(subproblem=subbqm)


class SublatticeDecomposer(traits.ProblemDecomposer, traits.SISO, Runnable):
    """Selects a lattice-structured subproblem.

    This decomposer requires the input state to contain fields ``bqm`` and
    ``origin_embeddings``; only the keys (the variables/nodes) from 
    ``origin_embeddings`` are used.
    The decomposer can also use the optional state fields ``problem_dims``,
    ``exclude_dims``, ``geometric_offset`` and ``origin_embedding_index``.

    By default ``geometric_offset`` is assigned uniformly at random on the range
    given by ``problem_dims``.
    By default ``origin_embedding_index`` is assigned uniformly at random
    on the range ``[0,len(state.origin_embeddings))``.
    The random number generator can be initialized with the class variable
    ``seed``.

    If ``problem_dims`` is a state field, geometrically offset variables are
    wrapped around boundaries according to assumed periodic boundary condition.
    This is a required field when the ``geometric_offset`` is not specified.
    Note that, the origin embedding must specify a lattice smaller than the
    target lattice.

    Args:
        seed (int, default=None):
            Pseudo-random number generator seed.

    Returns:
        ``subproblem`` and ``embedding`` state fields

    See also:
        :class:`~hybrid.reference.lattice_lnls.LatticeLNLS`

        :class:`~hybrid.reference.lattice_lnls.LatticeLNLSSampler`

        :class:`~hybrid.decomposers.make_origin_embeddings`

        :ref:`decomposers-examples`

        Jack Raymond et al, `Hybrid quantum annealing for larger-than-QPU
        lattice-structured problems <https://arxiv.org/abs/2202.03044>`_
    """

    def __init__(self, seed=None, **runopts):
        super(SublatticeDecomposer, self).__init__(**runopts)
        self.random = np.random.RandomState(seed)

    def __repr__(self):
        return "{self}(random={self.random!r})".format(self=self)

    def next(self, state, **runopts):
        bqm = state.problem

        if 'geometric_offset' not in state:
            # Select uniformly at random amongst available geometric offsets
            geometric_offset = [self.random.randint(dim) for dim in state.problem_dims]
            # Do not offset excluded dimensions
            if 'exclude_dims' in state:
                for dim in state.exclude_dims:
                    if dim < 0 or dim >= len(geometric_offset):
                        raise ValueError('exclude_dimension state variable '
                                         'indexes an invalid dimension')
                    geometric_offset[dim] = 0
        else:
            if len(state.problem_dims) != len(state.geometric_offset):
                raise ValueError('problem_dimension and geometric_offset state '
                                 'variables are of incompatible length')
            for idx, offset in enumerate(state.geometric_offset):
                if not (offset < state.problem_dims[idx] and 0 <= offset):
                    raise ValueError(
                        'geometric_offset state variable values are outside the '
                        f'lattice allowed ranges [0, problem_dimension[idx]), idx={idx}')
            geometric_offset = state.geometric_offset

        def key_transform(initial_coordinates):
            # The geometric keys are offset, with wrapping about periodic
            # boundary conditions.
            final_coordinates = list(initial_coordinates)
            if 'problem_dims' in state:
                for idx, val in enumerate(geometric_offset):
                    final_coordinates[idx] += val
                    final_coordinates[idx] %= state.problem_dims[idx]
            else:
                for idx, val in enumerate(geometric_offset):
                    final_coordinates[idx] += val
            return tuple(final_coordinates)

        # For now we explicitly encode different automorphism as different
        # origin_embeddings, but is would be natural to allow symmetry
        # operations (automorphisms) with respect to some fixed embedding
        # and lattice class.
        if 'origin_embedding_index' not in state:
            #Select uniformly at random amongst available embeddings:
            origin_embedding_index = self.random.randint(
                len(state.origin_embeddings))
        else:
            if (state.origin_embedding_index > len(state.origin_embeddings) or
                state.origin_embedding_index < -len(state.origin_embeddings)):
                raise ValueError(
                    'embedding_index state variable specifies an '
                    'origin_embeddings element beyond the list range')
            origin_embedding_index = state.origin_embedding_index

        # Create the embedding:
        embedding = {key_transform(key): value
                     for key, value
                     in state.origin_embeddings[origin_embedding_index].items()}

        # Create the associated subproblem, conditioned on best boundary sample
        # values:
        variables = embedding.keys()
        sample = state.samples.change_vartype(bqm.vartype).first.sample
        subbqm = bqm_induced_by(bqm, variables, sample)
        logger.debug("{} selected {} subproblem variables: {!r}".format(
            self.name, len(variables), variables))

        return state.updated(subproblem=subbqm, embedding=embedding)


class RoofDualityDecomposer(traits.ProblemDecomposer, traits.ProblemSampler,
                            traits.SISO, Runnable):
    """Selects a subproblem with variables that cannot be fixed by roof duality.

    Roof duality finds a lower bound for the minimum of a quadratic polynomial.
    It can also find minimizing assignments for some of the polynomial's
    variables; these fixed variables take the same values in all optimal
    solutions [BHT]_ [BH]_. A quadratic pseudo-Boolean function can be
    represented as a network to find the lower bound through network-flow
    computations. This decomposer can also use maximum flow in the implication
    network to fix variables. Consequently, you can find an assignment for the
    remaining variables that attains the optimal value.

    Args:
        sampling_mode (bool, optional, default=True):
            In sampling mode, only roof-duality is used. When `sampling_mode` is
            false, strongly connected components are used to fix more variables,
            but in some optimal solutions these variables may take different
            values.

    .. [BHT] Boros, E., P.L. Hammer, G. Tavares. Preprocessing of Unconstraint
        Quadratic Binary Optimization. Rutcor Research Report 10-2006, April,
        2006.

    .. [BH] Boros, E., P.L. Hammer. Pseudo-Boolean optimization. Discrete
        Applied Mathematics 123, (2002), pp. 155-225

    """
    def __init__(self, sampling_mode=True, **runopts):
        super(RoofDualityDecomposer, self).__init__(**runopts)
        self.sampling_mode = sampling_mode

    def __repr__(self):
        return "{self.name}(sampling_mode={self.sampling_mode!r})".format(self=self)

    def next(self, state, **runopts):
        bqm = state.problem
        sampleset = state.samples

        _, fixed_vars = dwave.preprocessing.roof_duality(bqm, strict=self.sampling_mode)

        # make a new bqm of everything not-fixed
        subbqm = bqm.copy()
        subbqm.fix_variables(fixed_vars)

        # update the existing state with the fixed variables
        newsampleset = sampleset.copy()
        for v, val in fixed_vars.items():
            # index lookups on variables are fast for SampleSets
            newsampleset.record.sample[:, newsampleset.variables.index(v)] = val

        # make sure the energies reflect the changes
        newsampleset.record.energy = bqm.energies(newsampleset)

        return state.updated(subproblem=subbqm, samples=newsampleset)


class TilingChimeraDecomposer(traits.ProblemDecomposer, traits.EmbeddingProducing,
                              traits.SISO, Runnable):
    """Returns sequential Chimera lattices that tile the initial problem.

    A Chimera lattice is an m-by-n grid of Chimera tiles, where each tile is a
    bipartite graph with shores of size t. The problem is decomposed into a
    sequence of subproblems with variables belonging to the Chimera lattices
    that tile the problem Chimera lattice. For example, a 2x2 Chimera lattice
    could be tiled 64 times (8x8) on a fully-yielded D-Wave 2000Q system
    (16x16).

    Args:
        size (int, optional, default=(4,4,4)):
            Size of the Chimera lattice as (m, n, t), where m is the number of
            rows, n the columns, and t the size of shore in the Chimera lattice.

        loop (Bool, optional, default=True):
            Cycle continually through the tiles.

    See :ref:`decomposers-examples`.
    """

    def __init__(self, size=(4,4,4), loop=True, **runopts):
        """Size C(n,m,t) defines a Chimera subgraph returned with each call."""
        super(TilingChimeraDecomposer, self).__init__(**runopts)
        self.size = size
        self.loop = loop
        self.blocks = None

    def __repr__(self):
        return "{self}(size={self.size!r}, loop={self.loop!r})".format(self=self)

    def init(self, state, **runopts):
        self.blocks = iter(chimera_tiles(state.problem, *self.size).items())
        if self.loop:
            self.blocks = itertools.cycle(self.blocks)

    def next(self, state, **runopts):
        """Each call returns a subsequent block of size `self.size` Chimera cells."""
        bqm = state.problem
        pos, embedding = next(self.blocks)
        variables = embedding.keys()
        sample = state.samples.change_vartype(bqm.vartype).first.sample
        subbqm = bqm_induced_by(bqm, variables, sample)
        return state.updated(subproblem=subbqm, embedding=embedding)


class RandomConstraintDecomposer(traits.ProblemDecomposer, traits.SISO, Runnable):
    """Selects variables randomly as constrained by groupings.

    By grouping related variables, the problem's structure can guide the random
    selection of variables so subproblems are related to the problem's
    constraints.

    Args:
        size (int):
            Number of variables in the subproblem.

        constraints (list[set]):
            Groups of variables in the BQM, as a list of sets, where each set is
            associated with a constraint.

    See :ref:`decomposers-examples`.
    """

    def __init__(self, size, constraints, **runopts):
        super(RandomConstraintDecomposer, self).__init__(**runopts)

        self.size = size

        if not isinstance(constraints, collections.abc.Sequence):
            raise TypeError("constraints should be a list of containers")
        if any(len(const) > size for const in constraints):
            raise ValueError("size must be able to contain the largest constraint")
        self.constraints = constraints

    def __repr__(self):
        return "{self}(size={self.size!r}, constraints={self.constraints!r})".format(self=self)

    def init(self, state, **runopts):
        if self.size > len(state.problem):
            raise ValueError("subproblem size cannot be greater than the problem size")

        # get the connectivity between the constraint components
        self.constraint_graph = CG = nx.Graph()
        for ci, const in enumerate(self.constraints):
            for i in range(ci+1, len(self.constraints)):
                if any(v in const for v in self.constraints[i]):
                    CG.add_edge(i, ci)

        if len(CG) < 1:
            raise ValueError("constraint graph empty")

    def next(self, state, **runopts):
        CG = self.constraint_graph
        size = self.size
        constraints = self.constraints
        bqm = state.problem

        # get a random constraint to start with
        n = random.choice(list(CG.nodes))

        if len(constraints[n]) > size:
            raise NotImplementedError

        # starting from our constraint, do a breadth-first search adding constraints until our max
        # size is reached
        variables = set(constraints[n])
        for _, ci in nx.bfs_edges(CG, n):
            proposed = [v for v in constraints[ci] if v not in variables]
            if len(proposed) + len(variables) <= size:
                variables.union(proposed)
            if len(variables) == size:
                # can exit early
                break

        sample = state.samples.change_vartype(bqm.vartype).first.sample
        subbqm = bqm_induced_by(bqm, variables, sample)
        return state.updated(subproblem=subbqm)


def _all_minimal_covers(edgelist):
    """Create all minimal covers, provided graphs contain fewer than 17 edges."""

    if len(edgelist) > 16:
        raise ValueError(
            'len(edgelist) > 16. Exponential scaling of this method means it '
            'should only be used for small graphs. The use case for this function '
            'is in finding coverings over unyielded lattice subgraphs, which in '
            'the current online generation of processors are sufficiently small '
            'so as to allow a brute force method.')

    minimum_coverings = []
    for cover in map(set, product(*edgelist)):
        newmin = []
        for s in minimum_coverings:
            if s < cover:
                break
            elif cover < s:
                pass
            else:
                newmin.append(s)
        else:
            minimum_coverings = newmin + [cover]

    return minimum_coverings


def _unyielded_conditional_edges(emb, source, target):
    """Adaption of minorminer.utils.diagnose_embedding
    Simplification is possible because nodes match by construction,
    and each chain is connected by construction.
    """
    labels = {}
    for x in source:
        for q in emb[x]:
            labels[q] = x

    yielded = nx.Graph()
    yielded.add_edges_from((labels[e[0]], labels[e[1]]) for e in target.edges
                           if e[0] in labels and e[1] in labels)

    unyielded_edge_set = [e for e in source.edges()
                          if not yielded.has_edge(e[0],e[1])]

    return unyielded_edge_set


def _yield_limited_origin_embedding(origin_embedding, proposed_source, target):
    """An unconditional edge defect is an unyielded edge for which one of the
    two qubits is yielded. These must be eliminated (by removing additional
    qubits) for operation of ```SublatticeDecomposer``. Minimizing the
    number of variables removed is a minimum vertex problem over the graph of
    unconditional edge defects. Because defects are rare in available
    processors, it can be solved by brute force.
    """
    # A fully yielded subgraph of the target problem
    proposed_source = proposed_source.subgraph(list(origin_embedding.keys()))

    # Edges not yielded over this subgraph:
    unyielded_edge_set = _unyielded_conditional_edges(emb=origin_embedding,
                                                      source=proposed_source,
                                                      target=target)

    # Remove minimal number of nodes such that edges yielded on subgraph:
    G = nx.Graph()
    G.add_edges_from(unyielded_edge_set)
    for subgraph_nodes in nx.algorithms.components.connected_components(G):
        cover = _all_minimal_covers(G.subgraph(subgraph_nodes).edges)
        for v in cover[0]:
            del origin_embedding[v]

    # Restrict to giant component. 
    proposed_source = proposed_source.subgraph(list(origin_embedding.keys()))
    max_cc = max(nx.connected_components(proposed_source), key=len)
    if len(max_cc) == 0:
        raise ValueError('The proposed origin embedding contains no variables. '
                         'This is likely caused by a key error (not using '
                         'geometrically appropriate variable keys.')
    origin_embedding = {k: v for k, v in origin_embedding.items() if k in max_cc}
    return origin_embedding


def _make_cubic_lattice(dimensions):
    """Returns an open boundary cubic lattice graph as function of lattice
    dimensions. Helper function for ``make_origin_embeddings``.
    """
    cubic_lattice_graph = nx.Graph()
    cubic_lattice_graph.add_edges_from([((x, y, z), (x+1, y, z))
                                        for x in range(dimensions[0]-1)
                                        for y in range(dimensions[1])
                                        for z in range(dimensions[2])
                                        ])
    cubic_lattice_graph.add_edges_from([((x, y, z), (x, y+1, z))
                                        for x in range(dimensions[0])
                                        for y in range(dimensions[1]-1)
                                        for z in range(dimensions[2])
                                        ])
    cubic_lattice_graph.add_edges_from([((x, y, z), (x, y, z+1))
                                        for x in range(dimensions[0])
                                        for y in range(dimensions[1])
                                        for z in range(dimensions[2]-1)
                                        ])
    return cubic_lattice_graph


def make_origin_embeddings(qpu_sampler=None, lattice_type=None,
                           problem_dims=None, reject_small_problems=True,
                           allow_unyielded_edges=False):
    """Creates optimal embeddings for a lattice.

    An embedding is a dictionary specifying the mapping from each lattice
    variable to a set of qubits (chain). The embeddings created are compatible
    with the topology and shape of a specified ``qpu_sampler``.

    Args:
        qpu_sampler (:class:`dimod.Sampler`, optional):
            Quantum sampler such as a D-Wave system. If not specified, the
            :class:`~dwave.system.samplers.DWaveSampler` sampler class is used
            to select a QPU solver with a topology compatible with the specified
            ``lattice_type`` (e.g. an Advantage system for a 'pegasus' lattice
            type).

        lattice_type (str, optional, default=qpu_sampler.properties['topology']['type']):
            Options are:
                * "cubic"
                    Embeddings compatible with the schemes arXiv:2009.12479 and
                    arXiv:2003.00133 are created for a ``qpu_sampler`` of
                    topology type either 'pegasus' or 'chimera'.

                * "pegasus"
                    Embeddings are chain length one (minimal and native).
                    If ``qpu_sampler`` topology type is 'pegasus', maximum
                    scale subgraphs are embedded using the ``nice_coordinates``
                    vector labeling scheme for variables.

                * "chimera"
                    Embeddings are chain length one (minimal and native).
                    If ``qpu_sampler`` topology type is 'chimera', maximum
                    scale chimera subgraphs are embedded using the chimera
                    vector labeling scheme for variables.

        problem_dims (tuple of ints, optional):
            origin_embeddings over the solver are constrained to not exceed
            these dimensions.

            For cubic lattices, dimension bounds are specified as (x, y, z).

            For pegasus lattices, bounds are specified in nice coordinate
            (t, y, x, u, k): a P16 solver has bounds (3,15,15,2,4)

            For chimera lattices, bounds are specified in chimera coordinates
            (i,j,u,k): a C16 solver has bounds (16,16,2,4).

            Where embedded variables exceed these bounds the embedding is
            either truncated, or raises an error, according to the flag
            ``reject_small_problems``.

            Note that when the embedding scale exactly matches the problem
            scale, problems can also occur for the case of periodic boundary
            conditions that are not supported by these embeddings, which assume
            open boundary induced problems in general.

        reject_small_problems (bool, optional, True):
            If the target problem dimensions are smaller than the dimensions
            allowed by the origin embeddings, raise an error.
            Since these routines are intended to support workflows beyond
            solvable scale, a user override is required for the inappopriate
            use case. This flag is only invoked if problem_dims is specified.

        allow_unyielded_edges (bool, optional, False):
            A requirement for certain sublattice based methods is that any two
            variables connected in the lattice must also be connected in the
            embedding. If two variables allow yielded chains, but are not
            connected, then by default at least one of the two variables is
            treated as unyielded. If the argument is set to False then yielded
            chains of incomplete connectivity are returned as part of the
            embedding.

    Returns:
        A list of embeddings. Each embedding is a dictionary, mapping
        geometric problem keys to sets of qubits (chains) compatible with
        the ``qpu_sampler``.

    Examples:
        This example creates a list of three cubic lattice embeddings
        compatible with the default online system. These three embeddings are
        related by rotation of the lattice: for a Pegasus P16 system the
        embeddings are for lattices of size (15,15,12), (12,15,15) and (15,12,15)
        respectively.

        >>> from dwave.system.samplers import DWaveSampler   # doctest: +SKIP
        >>> sampler = DWaveSampler()  # doctest: +SKIP
        >>> embeddings = make_origin_embeddings(qpu_sampler=sampler,
        ...                                     lattice_type='cubic')  # doctest: +SKIP

    See also:
        :class:`~hybrid.reference.lattice_lnls.LatticeLNLS`

        :class:`~hybrid.reference.lattice_lnls.LatticeLNLSSampler`

        :class:`~hybrid.decomposers.SublatticeDecomposer`

        Jack Raymond et al, `Hybrid quantum annealing for larger-than-QPU
        lattice-structured problems <https://arxiv.org/abs/2202.03044>`_
    """
    if qpu_sampler is None:
        if lattice_type == 'pegasus' or lattice_type == 'chimera':
            qpu_sampler = DWaveSampler(solver={'topology__type': lattice_type})
        else:
            qpu_sampler = DWaveSampler()

    qpu_type = qpu_sampler.properties['topology']['type']
    if lattice_type is None:
        lattice_type = qpu_type
    qpu_shape = qpu_sampler.properties['topology']['shape']

    target = nx.Graph()
    # 'couplers' and 'qubits' property fields are preferred to
    # edgelist and nodelist properties when available.
    # Mutability of the former is a useful feature for masking
    # on the fly.
    if 'couplers' in qpu_sampler.properties:
        target.add_edges_from(qpu_sampler.properties['couplers'])
    else:
        target.add_edges_from(qpu_sampler.edgelist)
    if qpu_type == lattice_type:
        # Fully yielded fully utilized native topology problem.
        # This method is also easily adapted to work for any chain-length 1
        # embedding
        if 'qubits' in qpu_sampler.properties:
            origin_embedding = {q: [q] for q in qpu_sampler.properties['qubits']}
        else:
            origin_embedding = {q: [q] for q in qpu_sampler.nodelist}
        if lattice_type == 'pegasus':
            # Trimming to nice_coordinate supported embeddings is not a unique,
            # options, it has some advantages and some disadvantages:
            proposed_source = dnx.pegasus_graph(qpu_shape[0], nice_coordinates=True)
            proposed_source = nx.relabel_nodes(
                proposed_source,
                {q: dnx.pegasus_coordinates(qpu_shape[0]).nice_to_linear(q)
                 for q in proposed_source.nodes()})
            lin_to_vec = dnx.pegasus_coordinates(qpu_shape[0]).linear_to_nice

        elif lattice_type == 'chimera':
            proposed_source = dnx.chimera_graph(qpu_shape[0],
                                                qpu_shape[1],
                                                qpu_shape[2])
            lin_to_vec = dnx.chimera_coordinates(qpu_shape[0]).linear_to_chimera
        else:
            raise ValueError(
                f'Unsupported native processor topology {qpu_type}. '
                'Support for Zephyr and other topologies is straightforward to '
                'add subject to standard dwave_networkx library tool availability.')

    elif lattice_type == 'cubic':
        if qpu_type == 'pegasus':
            vec_to_lin = dnx.pegasus_coordinates(qpu_shape[0]).pegasus_to_linear
            L = qpu_shape[0] - 1
            dimensions = [L, L, 12]
            # See arXiv:2003.00133
            origin_embedding = {(x, y, z): [vec_to_lin((0, x, z+4, y)),
                                            vec_to_lin((1, y+1, 7-z, x))]
                                for x in range(L)
                                for y in range(L)
                                for z in range(8)
                                if target.has_edge(vec_to_lin((0, x, z+4, y)),
                                                   vec_to_lin((1, y+1, 7-z, x)))}
            origin_embedding.update({(x, y, z): [vec_to_lin((0, x+1, z-8, y)),
                                                 vec_to_lin((1, y, 19-z, x))]
                                     for x in range(L)
                                     for y in range(L)
                                     for z in range(8, 12)
                                     if target.has_edge(vec_to_lin((0, x+1, z-8, y)),
                                                        vec_to_lin((1, y, 19-z, x)))})
        elif qpu_type == 'chimera':
            vec_to_lin = dnx.chimera_coordinates(qpu_shape[0],
                                                 qpu_shape[1],
                                                 qpu_shape[2]).chimera_to_linear
            L = qpu_shape[0] // 2 
            dimensions = [L, L, 8]
            # See arxiv:2009.12479, one choice amongst many
            origin_embedding = {(x, y, z):
                                [vec_to_lin(coord) for coord in [(2*x+1, 2*y, 0, z),
                                                                 (2*x, 2*y, 0, z),
                                                                 (2*x, 2*y, 1, z),
                                                                 (2*x, 2*y+1, 1, z)]]
                                for x in range(L) for y in range(L) for z in range(4)
                                if target.has_edge(vec_to_lin((2*x+1, 2*y, 0, z)),
                                                   vec_to_lin((2*x, 2*y, 0, z)))
                                and target.has_edge(vec_to_lin((2*x, 2*y, 0, z)),
                                                    vec_to_lin((2*x, 2*y, 1, z)))
                                and target.has_edge(vec_to_lin((2*x, 2*y, 1, z)),
                                                    vec_to_lin((2*x, 2*y+1, 1, z)))
                                }
            origin_embedding.update({(x, y, 4+z):
                                     [vec_to_lin(coord) for coord in [(2*x+1, 2*y, 1, z),
                                                                      (2*x+1, 2*y+1, 1, z),
                                                                      (2*x+1, 2*y+1, 0, z),
                                                                      (2*x, 2*y+1, 0, z)]]
                                     for x in range(L)
                                     for y in range(L)
                                     for z in range(4)
                                     if target.has_edge(vec_to_lin((2*x+1, 2*y, 1, z)),
                                                        vec_to_lin((2*x+1, 2*y+1, 1, z)))
                                     and target.has_edge(vec_to_lin((2*x+1, 2*y+1, 1, z)),
                                                         vec_to_lin((2*x+1, 2*y+1, 0, z)))
                                     and target.has_edge(vec_to_lin((2*x+1, 2*y+1, 0, z)),
                                                         vec_to_lin((2*x,2*y+1,0,z)))
                                     })
        else:
            raise ValueError(f'Unsupported qpu_sampler topology {qpu_type} '
                             'for cubic lattice solver')

        proposed_source = _make_cubic_lattice(dimensions)
    else:
        raise ValueError('Unsupported combination of lattice_type '
                         'and qpu_sampler topology')

    if not allow_unyielded_edges:
        origin_embedding = _yield_limited_origin_embedding(origin_embedding,
                                                           proposed_source,
                                                           target)

    if qpu_type == lattice_type:
        # Convert keys to standard vector scheme:
        origin_embedding = {lin_to_vec(node): origin_embedding[node]
                            for node in origin_embedding}
    else:
        # Keys are already in geometric format:
        pass

    # We can propose additional embeddings. Or we can use symmetries of the
    # target graph (automorphisms), to create additional embedding options.
    # This is important in the cubic case, because the subregion shape and
    # embedding features are asymmetric in the x, y and z directions.
    # Various symmetries can be exploited in all lattices.
    origin_embeddings = [origin_embedding]
    if lattice_type == 'cubic':
        # A rotation is sufficient for demonstration purposes:
        origin_embeddings.append({(key[2], key[0], key[1]): value
                                  for key, value in origin_embedding.items()})
        origin_embeddings.append({(key[1], key[2], key[0]): value
                                  for key, value in origin_embedding.items()})
        problem_dim_spec = 3
    elif lattice_type == 'pegasus':
        # A horizontal to vertical flip is sufficient for demonstration purposes.
        # Flip north-east to south-west axis (see draw_pegasus):
        L = qpu_shape[0]
        origin_embeddings.append(
            {(key[0], L-2-key[2], L-2-key[1], 1-key[3], 3-key[4]): value
             for key,value in origin_embedding.items()})
        problem_dim_spec = 5
    elif lattice_type == 'chimera':
        # A horizontal to vertical flip is sufficient for demonstration purposes:
        origin_embeddings.append({(key[1], key[0], 1-key[2], key[3]): value
                                  for key,value in origin_embedding.items()})
        problem_dim_spec = 4
    else:
        raise ValueError('Unsupported lattice_type')
    if problem_dims is not None:
        if len(problem_dims) != problem_dim_spec:
            raise ValueError('len(problem_dims) is incompatible with'
                             'the lattice type')
        else:
            pass
        for origin_embedding in origin_embeddings:
            rem_list = {key for key in origin_embedding
                        if any(key[idx]>=problem_dims[idx]
                               for idx in range(problem_dim_spec))}

            if len(rem_list) > 0:
                if reject_small_problems:
                    raise ValueError('embedding scale exceeds '
                                     'that of the problem target (problem_dims)')
                else:
                    [origin_embedding.pop(key) for key in rem_list]
            else:
                pass
    else:
        pass
    return origin_embeddings
