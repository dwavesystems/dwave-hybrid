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


def flip_energy_gains(bqm, sample):
    """Returns `list[(energy_gain, flip_index)]` in descending order
    for flipping qubit with flip_index in sample.
    """
    base = bqm.energy(sample)
    energy_gains = [(bqm.energy(sample[:i] + [1 - bit] + sample[i:]) - base, i) for i, bit in enumerate(sample)]
    energy_gains.sort(reverse=True)
    return energy_gains


def select_localsearch_adversaries(bqm, sample, max_n, min_gain=0.0):
    """Returns a list of up to ``max_n`` variables from ``bqm`` that have a high
    energy gain (at least ``min_gain``) for single bit flip, and thus are
    considered tabu for tabu.
    """
    variables = [idx for en, idx in flip_energy_gains(bqm, sample) if en >= min_gain]
    return variables[:max_n]


def updated_sample(sample, replacements):
    """Returns a copy of ``sample`` (which is a list-like object), with
    variables changed according to ``replacements.
    """
    result = sample.copy()
    for k, v in replacements.items():
        result[k] = v
    return result


def sample_dict_to_list(sample):
    """Convert ``sample``, ``dict: idx -> var``, to ``list: var``."""
    indices = sorted(sample.keys())
    if len(indices) > 0 and indices[-1] - indices[0] + 1 != len(indices):
        raise ValueError("incomplete sample dict")
    return [sample[k] for k in indices]


def sample_list_to_dict(sample):
    """Convert ``sample``, ``list: var``, to ``dict: idx -> var``."""
    return dict(enumerate(sample))
