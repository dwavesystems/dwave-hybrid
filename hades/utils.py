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


def bqm_induced_with(bqm, variables, state):
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

    # no point in having offset since we fixing only variables on boundary
    subbqm.remove_offset()

    return subbqm
