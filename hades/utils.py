import random

import six
import dimod


def bqm_variables(bqm):
    """Returns all BQM variables.
    To be replaced with BQM.variables in new dimod."""
    return six.viewkeys(bqm.linear)


def bqm_reduced_to(bqm, variables, state, keep_offset=True):
    """Return a sub-BQM, which is ``bqm`` reduced to ``variables``, by fixing
    all non sub-BQM variables.

    Note:
        Optimized for ``len(variables) ~ len(bqm)`` (fixing very little vars).
    """

    # fix complement of ``variables```
    fixed = bqm_variables(bqm).difference(variables)
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
    """Returns `list[(energy_gain, flip_index)]` in descending order
    for flipping qubit with flip_index in sample.

    Note: Grossly inefficient! Use `flip_energy_gains_iterative` which traverses
    bits, updating energy delta based on previous bit and neighbors.
    """
    base = bqm.energy(sample)
    energy_gains = [(bqm.energy(sample[:i] + [1 - bit] + sample[i+1:]) - base, i) for i, bit in enumerate(sample)]
    energy_gains.sort(reverse=True)
    return energy_gains


def flip_energy_gains_iterative(bqm, sample):
    """Returns `list[(energy_gain, flip_index)]` in descending order
    for flipping qubit with flip_index in sample.

    Args:
        bqm (:class:`dimod.BinaryQuadraticModel`):
            BQM of type dimod.BINARY

        sample (list):
            Perturbation base (as 0/1 binary values)

    Note:
        Comparison with the naive approach (bqm size ~ 2k, random sample)::

            >>> %timeit flip_energy_gains_naive(bqm, sample)
            3.35 s ± 37.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

            >>> %timeit flip_energy_gains_iterative(bqm, sample)
            3.52 ms ± 20.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

        Three orders of magnitude faster.

        Subnote: using list comprehension speeds-up the iterative approach by
        only 2%, so we're using the standard loop (a lot more readable).

    TODO: generalize to support all dimod vartypes (add SPIN).
    """
    energy_gains = []
    for idx, val in enumerate(sample):
        # val is 0, flips to 1 => delta +1
        # val is 1, flips to 0 => delta -1
        delta = 1 - 2 * val
        contrib = bqm.linear[idx] + sum(w * sample[neigh] for neigh, w in bqm.adj[idx].items())
        energy_gains.append((contrib * delta, idx))

    energy_gains.sort(reverse=True)
    return energy_gains


flip_energy_gains = flip_energy_gains_iterative


def select_localsearch_adversaries(bqm, sample, max_n, min_gain=0.0):
    """Returns a list of up to ``max_n`` variables from ``bqm`` that have a high
    energy gain (at least ``min_gain``) for single bit flip, and thus are
    considered tabu for tabu.
    """
    variables = [idx for en, idx in flip_energy_gains(bqm, sample) if en >= min_gain]
    return variables[:max_n]


def select_random_subgraph(bqm, n):
    return random.sample(bqm.linear.keys(), n)


def updated_sample(sample, replacements):
    """Returns a copy of ``sample`` (which is a dict-like object), with
    variables changed according to ``replacements``.
    """
    result = sample.copy()
    for k, v in replacements.items():
        result[k] = v
    return result


def sample_as_list(sample):
    """Convert dict-like ``sample`` (list/dict/dimod.SampleView),
    ``map: idx -> var``, to ``list: var``.
    """
    if isinstance(sample, list):
        return sample
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
    if isinstance(sample, list):
        sample = enumerate(sample)
    return dict(sample)
