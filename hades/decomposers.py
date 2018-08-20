from itertools import cycle

from hades.core import Runnable, State
from hades.profiling import tictoc
from hades.utils import (
    bqm_induced_by, select_localsearch_adversaries, select_random_subgraph,
    chimera_tiles)


class EnergyImpactDecomposer(Runnable):
    """Selects up to `max_size` variables that contribute the most to energy
    increase.

    Note: currently, list of variables not connected in problem graph might be
    returned.
    """

    def __init__(self, bqm, max_size, min_gain=0.0, stride=1):
        self.bqm = bqm
        self.max_size = max_size
        self.min_gain = min_gain
        self.stride = stride

        # variables from previous iteration
        self._prev_vars = set()

    @tictoc('energy_impact_decompose')
    def iterate(self, state):
        # select new subset of max_size variables, making sure they differ from
        # previous iteration (on collision, move one stride right)
        variables = select_localsearch_adversaries(
            self.bqm, state.sample.values, min_gain=self.min_gain)
        candidate_vars = set(variables[:self.max_size])
        if candidate_vars == self._prev_vars:
            variables = set(variables[self.stride:][:self.max_size])
        else:
            variables = candidate_vars
        self._prev_vars = variables

        # induce sub-bqm based on selected variables and global sample
        subbqm = bqm_induced_by(self.bqm, variables, state.sample.values)
        return state.updated(ctx=dict(subproblem=subbqm),
                             debug=dict(decomposer=self.__class__.__name__))


class RandomSubproblemDecomposer(Runnable):
    """Selects a random subproblem of size `size`. The subproblem is possibly
    not connected.
    """

    def __init__(self, bqm, size):
        self.bqm = bqm
        self.size = size

    @tictoc('random_decompose')
    def iterate(self, state):
        variables = select_random_subgraph(self.bqm, self.size)
        subbqm = bqm_induced_by(self.bqm, variables, state.sample.values)
        return state.updated(ctx=dict(subproblem=subbqm),
                             debug=dict(decomposer=self.__class__.__name__))


class IdentityDecomposer(Runnable):
    """Copies problem to subproblem."""

    def __init__(self, bqm):
        self.bqm = bqm

    @tictoc('identity_decompose')
    def iterate(self, state):
        return state.updated(ctx=dict(subproblem=self.bqm),
                             debug=dict(decomposer=self.__class__.__name__))


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
        subbqm = bqm_induced_by(self.bqm, variables, state.sample.values)
        return state.updated(ctx=dict(subproblem=subbqm, embedding=embedding),
                             debug=dict(decomposer=self.__class__.__name__))
