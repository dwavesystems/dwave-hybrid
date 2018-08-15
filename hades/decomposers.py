from hades.core import Runnable, State
from hades.profiling import tictoc
from hades.utils import (
    bqm_induced_by, select_localsearch_adversaries, select_random_subgraph)


class EnergyImpactDecomposer(Runnable):
    """Selects up to `max_size` variables that contribute the most to energy
    increase.

    Note: currently, list of variables not connected in problem graph might be
    returned.
    """

    def __init__(self, bqm, max_size, min_gain=0.0):
        self.bqm = bqm
        self.max_size = max_size
        self.min_gain = min_gain

    @tictoc('energy_impact_decompose')
    def iterate(self, state):
        variables = select_localsearch_adversaries(
            self.bqm, state.sample.values, self.max_size, min_gain=self.min_gain)
        subbqm = bqm_induced_by(self.bqm, variables, state.sample.values)
        return state.updated(ctx=dict(subproblem=subbqm))


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
        return state.updated(ctx=dict(subproblem=subbqm))


class IdentityDecomposer(Runnable):
    """Copies problem to subproblem."""

    def __init__(self, bqm):
        self.bqm = bqm

    @tictoc('identity_decompose')
    def iterate(self, state):
        return state.updated(ctx=dict(subproblem=self.bqm))
