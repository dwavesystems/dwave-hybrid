from __future__ import division

import random

import six
import dimod
import numpy

from dnx import canonical_chimera_labeling


def bqm_reduced_to(bqm, variables, state, keep_offset=True):
    """Return a sub-BQM, which is ``bqm`` reduced to ``variables``, by fixing
    all non sub-BQM variables.

    Note:
        Optimized for ``len(variables) ~ len(bqm)`` (fixing very little vars).
    """

    # fix complement of ``variables```
    fixed = set(bqm.variables).difference(variables)
    subbqm = bqm.copy()
    for v in fixed:
        subbqm.fix_variable(v, state[v])

    if not keep_offset:
        subbqm.remove_offset()

    return subbqm


def bqm_induced_by(bqm, variables, state):
    """Return sub-BQM that includes only ``variables``, and boundary is fixed
    according to ``state``.

    Args:
        bqm (:class:`dimod.BinaryQuadraticModel`):
            Original BQM.

        variables (list/set);
            Variables of the subgraph.

        state (dict/list):
            Mapping of variable labels to values. If labels are sequential integers
            ``state`` may be a list. State is required only for variables on boundary
            (variables in BQM graph connected with ``variables``).

    Returns:
        Sub-graph (sub-bqm) induced by ``variables`` on ``bqm``. Only variables on
        boundary (adjacent to any of internal variables) are fixed according to
        ``state``. BQM offset is set to zero.

    Note:
        Optimized for ``len(variables) << len(bqm)`` (fixing majority of vars).

    """

    variables = set(variables)

    # create empty BQM and copy in a subgraph induced by `variables`
    subbqm = dimod.BinaryQuadraticModel({}, {}, 0.0, bqm.vartype)

    for u in variables:
        bias = bqm.linear[u]
        for v, j in bqm.adj[u].items():
            if v in variables:
                subbqm.add_interaction(u, v, j / 2.0)
            else:
                bias += j * state[v]
        subbqm.add_variable(u, bias)

    # no point in having offset since we're fixing only variables on boundary
    subbqm.remove_offset()

    return subbqm


def bqm_edges_between_variables(bqm, variables):
    """Returns a list of all edges as tuples in ``bqm`` between ``variables``.
    Nodes/variables are included as (v, v).
    """
    variables = set(variables)
    edges = [(start, end) for (start, end), coupling in bqm.quadratic.items() if start in variables and end in variables]
    edges.extend((v, v) for v in bqm.linear if v in variables)
    return edges


def flip_energy_gains_naive(bqm, sample):
    """Return `list[(energy_gain, flip_index)]` in descending order
    for flipping qubit with flip_index in sample.

    Note: Grossly inefficient! Use `flip_energy_gains_iterative` which traverses
    variables, updating energy delta based on previous var value and neighbors.
    """

    if bqm.vartype is dimod.BINARY:
        flip = lambda val: 1 - val
    elif bqm.vartype is dimod.SPIN:
        flip = lambda val: -val
    else:
        raise ValueError("vartype not supported")

    base = bqm.energy(sample)
    sample = sample_as_list(sample)
    energy_gains = [(bqm.energy(sample[:i] + [flip(val)] + sample[i+1:]) - base, i) for i, val in enumerate(sample)]
    energy_gains.sort(reverse=True)
    return energy_gains


def flip_energy_gains_iterative(bqm, sample):
    """Return `list[(energy_gain, flip_index)]` in descending order
    for flipping qubit with flip_index in sample.

    Args:
        bqm (:class:`dimod.BinaryQuadraticModel`):
            BQM of type dimod.BINARY

        sample (list/dict):
            Perturbation base (0/1 values for QUBO and -1/+1 for Ising model)

    Note:
        Comparison with the naive approach (bqm size ~ 2k, random sample)::

            >>> %timeit flip_energy_gains_naive(bqm, sample)
            3.35 s ± 37.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

            >>> %timeit flip_energy_gains_iterative(bqm, sample)
            3.52 ms ± 20.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

        Three orders of magnitude faster.

        Subnote: using list comprehension speeds-up the iterative approach by
        only 2%, so we're using the standard loop (a lot more readable).
    """

    if bqm.vartype is dimod.BINARY:
        # val is 0, flips to 1 => delta +1
        # val is 1, flips to 0 => delta -1
        delta = lambda val: 1 - 2 * val
    elif bqm.vartype is dimod.SPIN:
        # val is -1, flips to +1 => delta +2
        # val is +1, flips to -1 => delta -2
        delta = lambda val: -2 * val
    else:
        raise ValueError("vartype not supported")

    energy_gains = []
    sample = sample_as_dict(sample)
    for idx, val in sample.items():
        contrib = bqm.linear[idx] + sum(w * sample[neigh] for neigh, w in bqm.adj[idx].items())
        energy_gains.append((contrib * delta(val), idx))

    energy_gains.sort(reverse=True)
    return energy_gains


flip_energy_gains = flip_energy_gains_iterative


def select_localsearch_adversaries(bqm, sample, max_n=None, min_gain=None):
    """Returns a list of up to ``max_n`` variables from ``bqm`` that have a high
    energy gain (at least ``min_gain``) for single bit flip, and thus are
    considered tabu for tabu.
    """
    var_gains = flip_energy_gains(bqm, sample)

    if max_n is None:
        max_n = len(sample)
    if min_gain is None:
        variables = [var for _, var in var_gains]
    else:
        variables = [var for en, var in var_gains if en >= min_gain]

    return variables[:max_n]


def select_random_subgraph(bqm, n):
    return random.sample(bqm.linear.keys(), n)


def chimera_tiles(bqm, m, n, t):
    """Map a given bqm to a set of chimera-structured tiles defined by (m, n, t).

    Returns:
        dict: The keys are the tile coordinates (row, col, aisle) and the values are partial
        embeddings, each mapping part of the bqm to a chimera tile as would
        be generated by dnx.chimera_graph(m, n, t).

    """
    try:
        chimera_indices = canonical_chimera_labeling(bqm)
    except AssertionError:
        raise ValueError("non-Chimera structured problem")

    max_m = max(i for i, _, _, _ in chimera_indices.values()) + 1
    max_n = max(j for _, j, _, _ in chimera_indices.values()) + 1
    max_t = max(k for _, _, _, k in chimera_indices.values()) + 1

    tile_rows = -(max_m // -m)  # ceiling division
    tile_columns = -(max_n // -n)
    tile_shore_length = -(max_t // -t)

    tiles = {(row, col, aisle): {}
             for row in range(tile_rows)
             for col in range(tile_columns)
             for aisle in range(tile_shore_length)}

    for v, (si, sj, u, sk) in chimera_indices.items():
        row = si % tile_rows  # which tile
        i = si // tile_rows  # which row within the tile

        col = sj % tile_columns
        j = sj // tile_columns

        aisle = sk % tile_shore_length
        k = sk // tile_shore_length

        tiles[(row, col, aisle)][v] = [((n*i + j)*2 + u)*t + k]

    return tiles


def updated_sample(sample, replacements):
    """Returns a copy of ``sample`` (which is a dict-like object), with
    variables changed according to ``replacements``.
    """
    result = sample_as_dict(sample).copy()
    for k, v in sample_as_dict(replacements).items():
        result[k] = v
    return result


def sample_as_list(sample):
    """Convert dict-like ``sample`` (list/dict/dimod.SampleView),
    ``map: idx -> var``, to ``list: var``.
    """
    if isinstance(sample, list):
        return sample
    if isinstance(sample, numpy.ndarray):
        return sample.tolist()
    indices = sorted(dict(sample).keys())
    if len(indices) > 0 and indices[-1] - indices[0] + 1 != len(indices):
        raise ValueError("incomplete sample dict")
    return [sample[k] for k in indices]


def sample_as_dict(sample):
    """Convert list-like ``sample`` (list/dict/dimod.SampleView),
    ``list: var``, to ``map: idx -> var``.
    """
    if isinstance(sample, dict):
        return sample
    if isinstance(sample, (list, numpy.ndarray)):
        sample = enumerate(sample)
    return dict(sample)


@dimod.decorators.vartype_argument('vartype')
def random_sample_seq(size, vartype):
    """Return random sample of `size` in length, with values from `vartype`."""
    values = list(vartype.value)
    return {i: random.choice(values) for i in range(size)}


def random_sample(bqm):
    values = list(bqm.vartype.value)
    return {i: random.choice(values) for i in bqm.variables}


def min_sample(bqm):
    value = min(bqm.vartype.value)
    return {i: value for i in bqm.variables}


def max_sample(bqm):
    value = max(bqm.vartype.value)
    return {i: value for i in bqm.variables}
