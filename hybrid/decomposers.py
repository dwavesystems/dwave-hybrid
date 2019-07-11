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

import dimod
import networkx as nx

from hybrid.core import Runnable, State
from hybrid.exceptions import EndOfStream
from hybrid import traits
from hybrid.utils import (
    bqm_induced_by, flip_energy_gains, select_random_subgraph,
    chimera_tiles)

__all__ = [
    'IdentityDecomposer', 'EnergyImpactDecomposer', 'RandomSubproblemDecomposer',
    'TilingChimeraDecomposer', 'RandomConstraintDecomposer',
    'RoofDualityDecomposer',
]

logger = logging.getLogger(__name__)


class IdentityDecomposer(traits.ProblemDecomposer, traits.SISO, Runnable):
    """Selects a subproblem that is a full copy of the problem."""

    def next(self, state, **runopts):
        return state.updated(subproblem=state.problem)


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
        graph = bqm.to_networkx_graph()
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

        fixed_vars = dimod.fix_variables(bqm, sampling_mode=self.sampling_mode)

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

        if not isinstance(constraints, collections.Sequence):
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
