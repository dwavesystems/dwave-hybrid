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
    """Selects up to `max_size` variables that contribute the most to energy
    increase.

    Note: currently, list of variables not connected in problem graph might be
    returned.
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
        return state.updated(ctx=dict(subproblem=subbqm),
                             debug=dict(decomposer=self.name))


class RandomSubproblemDecomposer(Runnable):
    """Selects a random subproblem of size `size`. The subproblem is possibly
    not connected.
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
        return state.updated(ctx=dict(subproblem=subbqm),
                             debug=dict(decomposer=self.name))


class IdentityDecomposer(Runnable):
    """Copies problem to subproblem."""

    def __init__(self, bqm):
        self.bqm = bqm

    @tictoc('identity_decompose')
    def iterate(self, state):
        return state.updated(ctx=dict(subproblem=self.bqm),
                             debug=dict(decomposer=self.name))


class TilingChimeraDecomposer(Runnable):
    """Returns sequential tile slices of the initial BQM."""

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
        return state.updated(ctx=dict(subproblem=subbqm, embedding=embedding),
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
        return state.updated(ctx=dict(subproblem=subbqm),
                             debug=dict(decomposer=self.name))
