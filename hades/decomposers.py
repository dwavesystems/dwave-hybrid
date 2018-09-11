import logging
from itertools import cycle
import collections
import random

import networkx as nx

from hades.core import Runnable, State
from hades.profiling import tictoc
from hades.utils import (
    bqm_induced_by, select_localsearch_adversaries, select_random_subgraph,
    chimera_tiles)


logger = logging.getLogger(__name__)


class EnergyImpactDecomposer(Runnable):
    """Selects a subproblem of variables maximally contributing to the problem energy.

    The selection currently implemented does not ensure that the variables are connected
    in the problem graph.

    Args:
        bqm (:obj:`.BinaryQuadraticModel`):
            Binary quadratic model (BQM).
        max_size (int):
            Maximum number of variables in the subproblem.
        min_gain (int, optional, default=0):
            Minimum reduction required to BQM energy, given the current sample. A variable
            is included in the subproblem only if inverting its sample value reduces energy
            by at least this ammount.
        min_diff (int, optional, default=1):
            Minimum number of variables that did not partake in the previous iteration.
        stride (int, optional, default=1):
            Number of variables to shift each step of identifying subproblem variables that meet
            the `min_diff` criteria. A shift of 3, for example, skips three high
            contributors to the current and previous iteration in favor of selecting three
            variables not in the previous iteration.

    Examples:
        This example iterates twice on a 10-variable binary quadratic model with a
        random initial sample set. `min_gain` configuration limits the subproblem
        in the first iteration to the first 4 variables shown in the output of
        `flip_energy_gains`; `min_diff` configuration should bring in new variables
        (this edge case worsens the second pick)

        >>> import dimod           # Create a binary quadratic model
        >>> bqm = dimod.BinaryQuadraticModel({t: 0 for t in range(10)},
        ...                                  {(t, (t+1) % 10): 1 for t in range(10)},
        ...                                  0, 'BINARY')
        >>> decomposer = EnergyImpactDecomposer(bqm, max_size=8, min_gain=1, min_diff=2)
        >>> state0 = core.State.from_sample(random_sample(bqm), bqm)
        >>> flip_energy_gains(bqm, state0.samples.first.sample)     # doctest: +SKIP
        [(1, 8), (1, 6), (1, 2), (1, 1), (0, 7), (-1, 9), (-1, 5), (-1, 3), (-1, 0), (-2, 4)]
        >>> state1 = decomposer.iterate(state0)
        >>> print(state1.ctx['subproblem'])       # doctest: +SKIP
        BinaryQuadraticModel({9: 2, 3: 1, 5: 2, 1: 1}, {}, 0.0, Vartype.BINARY)
        >>> state2 = decomposer.iterate(state1)
        >>> print(state2.ctx['subproblem'])      # doctest: +SKIP
        BinaryQuadraticModel({1: 1, 3: 1}, {}, 0.0, Vartype.BINARY)

    """

    def __init__(self, bqm, max_size, min_gain=0.0, min_diff=1, stride=1):
        if max_size > len(bqm):
            raise ValueError("subproblem size cannot be greater than the problem size")
        if min_diff > max_size or min_diff < 0:
            raise ValueError("min_diff must be nonnegative and less than max_size")

        self.bqm = bqm
        self.max_size = max_size
        self.min_gain = min_gain
        self.min_diff = min_diff
        self.stride = stride

        # variables from previous iteration
        self._prev_vars = set()

    @tictoc('energy_impact_decompose')
    def iterate(self, state):
        # select a new subset of `max_size` variables, making sure they differ
        # from previous iteration by at least `min_diff` variables
        sample = state.samples.change_vartype(self.bqm.vartype).first.sample
        variables = select_localsearch_adversaries(
            self.bqm, sample, min_gain=self.min_gain)

        # TODO: soft fail strategy? skip one iteration or relax vars selection?
        if len(variables) < self.min_diff:
            raise ValueError("less than min_diff variables identified as"
                             " contributors to min_gain energy increase")

        offset = 0
        next_vars = set(variables[offset : offset+self.max_size])
        while len(next_vars ^ self._prev_vars) < self.min_diff:
            offset += self.stride
            next_vars = set(variables[offset : offset+self.max_size])

        logger.debug("Select variables: %r (diff from prev = %r)",
                     next_vars, next_vars ^ self._prev_vars)
        self._prev_vars = next_vars

        # induce sub-bqm based on selected variables and global sample
        subbqm = bqm_induced_by(self.bqm, next_vars, sample)
        return state.updated(subproblem=subbqm,
                             debug=dict(decomposer=self.name))


class RandomSubproblemDecomposer(Runnable):
    """Select a subproblem of `size` random variables.

    The selection currently implemented does not ensure that the variables are connected
    in the problem graph.

    Args:
        bqm (:obj:`.BinaryQuadraticModel`):
            Binary quadratic model (BQM).
        size (int):
            Number of variables in the subproblem.

    Examples:
        This example decomposes a 6-variable binary quadratic model with a
        random initial sample set to create a 3-variable subproblem.

        >>> import dimod           # Create a binary quadratic model
        >>> bqm = dimod.BinaryQuadraticModel({t: 0 for t in range(6)},
        ...             {(t, (t+1) % 6): 1 for t in range(6)}, 0, 'BINARY')
        >>> decomposer = RandomSubproblemDecomposer(bqm, size=3)
        >>> state0 = core.State.from_sample(random_sample(bqm), bqm)
        >>> state1 = decomposer.iterate(state0)
        >>> print(state1.ctx['subproblem'])
        BinaryQuadraticModel({2: 1.0, 3: 0.0, 4: 0.0}, {(2, 3): 1.0, (3, 4): 1.0}, 0.0, Vartype.BINARY)

    """

    def __init__(self, bqm, size):
        # TODO: add min_diff support (like in EnergyImpactDecomposer)
        if size > len(bqm):
            raise ValueError("subproblem size cannot be greater than the problem size")

        self.bqm = bqm
        self.size = size

    @tictoc('random_decompose')
    def iterate(self, state):
        variables = select_random_subgraph(self.bqm, self.size)
        sample = state.samples.change_vartype(self.bqm.vartype).first.sample
        subbqm = bqm_induced_by(self.bqm, variables, sample)
        return state.updated(subproblem=subbqm,
                             debug=dict(decomposer=self.name))


class IdentityDecomposer(Runnable):
    """Selects a subproblem that is a copy of the problem."""

    def __init__(self, bqm):
        self.bqm = bqm

    @tictoc('identity_decompose')
    def iterate(self, state):
        return state.updated(subproblem=self.bqm,
                             debug=dict(decomposer=self.name))


class TilingChimeraDecomposer(Runnable):
    """Returns sequential Chimera lattices that tile the initial problem.

    A Chimera lattice is an m-by-n grid of Chimera tiles, where each tile is a bipartite graph
    with shores of size t. The problem is decomposed into a sequence of subproblems with variables
    belonging to the Chimera lattices that tile the problem Chimera lattice. For example,
    a 2x2 Chimera lattice could be tiled 64 times (8x8) on a fully-yielded D-Wave 2000Q system (16x16).

    Args:
        bqm (:obj:`.BinaryQuadraticModel`):
            Binary quadratic model (BQM).
        size (int, optional, default=(4,4,4)):
            Size of the Chimera lattice as (m, n, t), where m is the number of rows,
            n the columns, and t the size of shore in the Chimera lattice.
        loop (Bool, optional, default=True):
            Cycle continually through the tiles.

    Examples:
        This example decomposes a 2048-variable Chimera structured binary quadratic model
        read from a file into 2x2x4-lattice subproblems.

        >>> import dimod           # Import a Chimera-structured binary quadratic model
        >>> with open('2048.09.qubo', 'r') as file:    # doctest: +SKIP
        ...     bqm = dimod.BinaryQuadraticModel.from_coo(file)
        >>> decomposer = TilingChimeraDecomposer(bqm, size=(2,2,4))   # doctest: +SKIP
        >>> state0 = core.State.from_sample(random_sample(bqm), bqm)  # doctest: +SKIP
        >>> state1 = decomposer.iterate(state0)    # doctest: +SKIP
        >>> print(state1.ctx['subproblem'])        # doctest: +SKIP
        BinaryQuadraticModel({0: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: -3.0, 1: 0.0, 2: 0.0, 3: -4.0, 1024: -7.0, 1028: 0.0,
        >>> # Snipped above response for brevity
        >>> state1 = decomposer.iterate(state0)    # doctest: +SKIP
        >>> print(state1.ctx['subproblem'])        # doctest: +SKIP
        BinaryQuadraticModel({8: 3.0, 12: 0.0, 13: 2.0, 14: -11.0, 15: -3.0, 9: 4.0, 10: 0.0, 11: 0.0, 1032: 0.0,

    """

    def __init__(self, bqm, size=(4,4,4), loop=True):
        """Size C(n,m,t) defines a Chimera subgraph returned with each call."""
        self.bqm = bqm
        self.size = size
        self.blocks = iter(chimera_tiles(self.bqm, *self.size).items())
        if loop:
            self.blocks = cycle(self.blocks)

    @tictoc('tiling_chimera_decompose')
    def iterate(self, state):
        """Each call returns a subsequent block of size `self.size` Chimera cells."""
        pos, embedding = next(self.blocks)
        variables = embedding.keys()
        sample = state.samples.change_vartype(self.bqm.vartype).first.sample
        subbqm = bqm_induced_by(self.bqm, variables, sample)
        return state.updated(subproblem=subbqm, embedding=embedding,
                             debug=dict(decomposer=self.name))


class RandomConstraintDecomposer(Runnable):
    """Pick variables randomly by chunks

    Args:
        bqm:

        size:

        constraints (list[set]):
            A list of sets where each set is a group of variables in the bqm associated with a
            constraint.

    """

    def __init__(self, bqm, size, constraints):
        self.bqm = bqm

        if size > len(bqm):
            raise ValueError("subproblem size cannot be greater than the problem size")
        self.size = size

        if not isinstance(constraints, collections.Sequence):
            raise TypeError("constraints should be a list of containers")
        if any(len(const) > size for const in constraints):
            raise ValueError("size must be able to contain the largest constraint")
        self.constraints = constraints

        # get the connectivity between the constraint components
        self.constraint_graph = CG = nx.Graph()
        for ci, const in enumerate(constraints):
            for i in range(ci+1, len(constraints)):
                if any(v in const for v in constraints[i]):
                    CG.add_edge(i, ci)

    @tictoc('random_constraint_decomposer')
    def iterate(self, state):
        CG = self.constraint_graph
        size = self.size
        bqm = self.bqm
        constraints = self.constraints

        # get a random constraint to start with.
        # for some reason random.choice(CG.nodes) does not work, so we rely on the fact that our
        # graph is index-labeled
        n = random.choice(range(len(CG)))

        if len(constraints[n]) > size:
            raise NotImplementedError

        # starting from our constraint, do a breadth-first search adding constraints until our max
        # size is reached
        variables = set(constraints[n])
        for _, ci in nx.bfs_edges(CG, n):
            proposed = [v for v in constraints[ci] if v not in variables]
            if len(proposed) + len(variables) <= size:
                variables.add(proposed)
            if len(variables) == size:
                # can exit early
                break

        sample = state.samples.change_vartype(self.bqm.vartype).first.sample
        subbqm = bqm_induced_by(self.bqm, variables, sample)
        return state.updated(subproblem=subbqm,
                             debug=dict(decomposer=self.name))
