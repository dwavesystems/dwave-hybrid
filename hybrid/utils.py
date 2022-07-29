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

import json
import random
import datetime

import numpy

import dimod
from dwave_networkx.algorithms import canonical_chimera_labeling


def cpu_count():    # pragma: no cover
    try:
        import os
        # doesn't exist in python2, and can return None
        return os.cpu_count() or 1
    except AttributeError:
        pass

    try:
        import multiprocessing
        # doesn't have to be implemented
        return multiprocessing.cpu_count()
    except NotImplementedError:
        pass

    return 1


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types.

    Supported types:
     - basic numeric types: booleans, integers, floats
     - arrays: ndarray, recarray
    """

    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.bool_):
            return bool(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()

        return super().default(obj)


class OceanEncoder(NumpyEncoder):
    """JSON encoder for common Ocean types.

    Use for convenience output/logging/inspection only, not for robust
    serialization (not all info required for reconstruction is dumped)!

    Supported types:
     - numpy types (see :class:`~hybrid.utils.NumpyEncoder`)
     - BinaryQuadraticModel (serialized as "numpy vectors")
     - SampleSet
    """

    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, dimod.BinaryQuadraticModel):
            # in dimod > 0.10, to_numpy_vectors returns a namedtuple. until we
            # drop support for dimod < 0.10, we have to be explicit here
            fields = ("linear", "quadratic", "offset", "labels")
            vec = obj.to_numpy_vectors(sort_indices=True, sort_labels=True, return_labels=True)
            return dict(zip(fields, vec))
        elif isinstance(obj, dimod.SampleSet):
            return obj.to_serializable()

        return super().default(obj)


def bqm_density(bqm):
    """Calculate BQM's graph density."""

    n = len(bqm)
    m = len(bqm.quadratic)
    max_m = n * (n - 1) / 2

    return m / max_m


def bqm_reduced_to(bqm, variables, sample, keep_offset=True):
    """Reduce a binary quadratic model by fixing values of some variables.

    The function is optimized for ``len(variables) ~ len(bqm)``, that is,
    for small numbers of fixed variables.

    Args:
        bqm (:class:`dimod.BinaryQuadraticModel`):
            Binary quadratic model (BQM).

        variables (list/set);
            Subset of variables to keep in the reduced BQM.

        sample (dict/list):
            Mapping of variable labels to values or a list when labels are
            sequential integers. Must include all variables not specified in
            `variables`.

        keep_offset (bool, optional, default=True):
            If false, set the reduced binary quadratic model's offset to zero;
            otherwise, uses the caluclated energy offset.

    Returns:
            :class:`dimod.BinaryQuadraticModel`: A reduced BQM.

    Examples:
        This example reduces a 3-variable BQM to two variables.

        >>> import dimod
        >>> bqm = dimod.BQM({}, {'ab': -1, 'bc': -1, 'ca': -1}, 0, 'BINARY')
        >>> sample = {'a': 1, 'b': 1, 'c': 0}
        >>> subbqm = bqm_reduced_to(bqm, ['a', 'b'], sample)
        >>> len(subbqm)
        2

    """

    # fix complement of ``variables```
    fixed = set(bqm.variables).difference(variables)
    subbqm = bqm.copy()
    for v in fixed:
        subbqm.fix_variable(v, sample[v])

    if not keep_offset:
        subbqm.remove_offset()

    return subbqm


def bqm_induced_by(bqm, variables, sample):
    """Induce a binary quadratic model by fixing values of boundary variables.

    The function is optimized for ``len(variables) << len(bqm)``, that is, for
    fixing the majority of variables.

    Args:
        bqm (:class:`dimod.BinaryQuadraticModel`):
            Binary quadratic model (BQM).

        variables (list/set);
            Subset of variables to keep in the reduced BQM, typically a
            subgraph.

        sample (dict/list):
            Mapping of variable labels to values or a list when labels
            are sequential integers. Values are required only for boundary
            variables, that is, for variables with interactions with `variables`
            (having edges with non-zero quadratic biases connected to the
            subgraph).

    Returns:
        :class:`dimod.BinaryQuadraticModel`: A BQM induced by fixing values of
        those variables adjacent to its subset of variables and setting the
        energy offset to zero.

    Examples:
        This example induces a 2-variable BQM from a 6-variable path
        graph---the subset of nodes 2 and 3 of nodes 0 to 5---by fixing values
        of boundary variables 1 and 4.

        >>> import dimod
        >>> import networkx as nx
        >>> bqm = dimod.BinaryQuadraticModel({},
        ...     {edge: edge[0] + 0.5 for edge in set(nx.path_graph(6).edges)}, 0, 'BINARY')
        >>> sample = {1: 3, 4: 3}
        >>> len(bqm_induced_by(bqm, [2, 3], sample))
        2

    """

    variables = set(variables)

    # create empty BQM and copy in a subgraph induced by `variables`
    subbqm = dimod.BinaryQuadraticModel.empty(bqm.vartype)

    for u in variables:
        bias = bqm.linear[u]
        for v, j in bqm.adj[u].items():
            if v in variables:
                subbqm.add_interaction(u, v, j / 2.0)
            else:
                bias += j * sample[v]
        subbqm.add_variable(u, bias)

    # no point in having offset since we're fixing only variables on boundary
    subbqm.offset = 0

    return subbqm


def bqm_edges_between_variables(bqm, variables):
    """Return edges connecting specified variables of a binary quadratic model.

    Args:
        bqm (:class:`dimod.BinaryQuadraticModel`):
            Binary quadratic model (BQM).

        variables (list/set):
            Subset of variables in the BQM.

    Returns:
        list: All edges connecting `variables` as tuples plus the variables
        themselves as tuples (v, v).

    Examples:
        This example returns connecting edges between 3 nodes of a BQM based on
        a 4-variable path graph.

        >>> import dimod
        >>> bqm = dimod.BQM({}, {(0, 1): 1, (1, 2): 1, (2, 3): 1}, 0, 'BINARY')
        >>> edges = bqm_edges_between_variables(bqm, {0, 1, 3})
        >>> edges       # doctest: +SKIP
        [(0, 1), (0, 0), (1, 1), (3, 3)]    # note: order not defined

    """
    variables = set(variables)
    edges = [(start, end) for (start, end), coupling in bqm.quadratic.items()
                if start in variables and end in variables]
    edges.extend((v, v) for v in bqm.linear if v in variables)
    return edges


def flip_energy_gains(bqm, sample, variables=None, min_gain=None):
    """Order variable flips by descending contribution to energy changes in a
    BQM.

    Args:
        bqm (:class:`dimod.BinaryQuadraticModel`):
            Binary quadratic model (BQM).

        sample (list/dict):
            Sample values as returned by dimod samplers (0 or 1 values for
            `dimod.BINARY` and -1 or +1 for `dimod.SPIN`)

        variables (sequence, optional, default=None):
            Consider only flips of these variables. If undefined, consider all
            variables in `sample`.

        min_gain (float, optional, default=None):
            Minimum required energy increase from flipping a sample value to
            return its corresponding variable.

    Returns:
        list: Energy changes in descending order, in the format of tuples
            (energy_gain, variable), for flipping the given sample value
            for each variable.

    Examples:
        This example returns the variable with maximum contribution to energy
        for the given sample.

        >>> import dimod
        >>> bqm = dimod.BQM({}, {'ab': 0, 'bc': 1, 'cd': 2}, 0, 'SPIN')
        >>> flip_energy_gains(bqm, {'a': -1, 'b': 1, 'c': 1, 'd': -1})[0][1]
        'd'

    """

    if bqm.vartype is dimod.BINARY:
        # val 0 flips to 1 => delta +1
        # val 1 flips to 0 => delta -1
        delta = lambda val: 1 - 2 * val
    elif bqm.vartype is dimod.SPIN:
        # val -1 flips to +1 => delta +2
        # val +1 flips to -1 => delta -2
        delta = lambda val: -2 * val
    else:
        raise ValueError("vartype not supported")

    sample = sample_as_dict(sample)

    if variables is None:
        variables = iter(sample)

    if min_gain is None:
        min_gain = float('-inf')

    energy_gains = []
    for v in variables:
        val = sample[v]
        contrib = bqm.linear[v] + sum(w * sample[neigh] for neigh, w in bqm.adj[v].items())
        en = contrib * delta(val)
        if en >= min_gain:
            energy_gains.append((en, v))

    energy_gains.sort(reverse=True, key=lambda p: (p[0], bqm.variables.index(p[1])))
    return energy_gains


def select_localsearch_adversaries(bqm, sample, max_n=None, min_gain=None):
    """Find variable flips that contribute high energy changes to a BQM.

    Args:
        bqm (:class:`dimod.BinaryQuadraticModel`):
            Binary quadratic model (BQM).

        sample (list/dict):
            Sample values as returned by dimod samplers (0 or 1 values for
            `dimod.BINARY` and -1 or +1 for `dimod.SPIN`)

        max_n (int, optional, default=None):
            Maximum contributing variables to return. By default, returns any
            variable for which flipping its sample value results in an energy
            gain of `min_gain`.

        min_gain (float, optional, default=None):
            Minimum required energy increase from flipping a sample value to
            return its corresponding variable.

    Returns:
        list: Up to `max_n` variables for which flipping the corresponding
        sample value increases the BQM energy by at least `min_gain`.

    Examples:
        This example returns 2 variables (out of up to 3 allowed) for which
        flipping sample values changes BQM energy by 1 or more. The BQM has
        energy gains of  0, -2, 2, 4 for variables a, b, c, d respectively for
        the given sample.

        >>> import dimod
        >>> bqm = dimod.BQM({}, {'ab': 0, 'bc': 1, 'cd': 2}, 0, 'SPIN')
        >>> select_localsearch_adversaries(
        ...     bqm, {'a': -1, 'b': 1, 'c': 1, 'd': -1}, max_n=3, min_gain=1)
        ['d', 'c']

    """
    var_gains = flip_energy_gains(bqm, sample, min_gain=min_gain)

    if max_n is None:
        max_n = len(sample)

    variables = [var for _, var in var_gains]

    return variables[:max_n]


def select_random_subgraph(bqm, n):
    """Select randomly `n` variables of the specified binary quadratic model.

    Args:
        bqm (:class:`dimod.BinaryQuadraticModel`):
            Binary quadratic model (BQM).

        n (int):
            Number of requested variables. Must be between 0 and `len(bqm)`.

    Returns:
        list: `n` variables selected randomly from the BQM.

    Examples:
        This example returns 2 variables of a 4-variable BQM.

        >>> import dimod
        >>> bqm = dimod.BQM({}, {'ab': 0, 'bc': 1, 'cd': 2}, 0, 'BINARY')
        >>> select_random_subgraph(bqm, 2)      # doctest: +SKIP
        ['d', 'b']

    """
    return random.sample(list(bqm.variables), n)


def chimera_tiles(bqm, m, n, t):
    """Map a binary quadratic model to a set of Chimera tiles.

    A Chimera lattice is an m-by-n grid of Chimera tiles, where each tile is a
    bipartite graph with shores of size t.

    Args:
        bqm (:obj:`.BinaryQuadraticModel`): Binary quadratic model (BQM).
        m (int): Rows.
        n (int): Columns.
        t (int): Size of shore.

    Returns:
        dict: Map as a dict where keys are tile coordinates (row, column, aisle)
        and values are partial embeddings of part of the BQM to a Chimera tile.
        Embeddings are those that would be generated by dwave_networkx's
        chimera_graph() function.

    Examples:
        This example maps a 1-by-2 Chimera-derived BQM to 2 side-by-side tiles.

        >>> import dwave_networkx as dnx
        >>> import dimod
        >>> G = dnx.chimera_graph(1, 2)     # Create a Chimera-based BQM
        >>> bqm = dimod.BinaryQuadraticModel({}, {edge: edge[0] for edge in G.edges}, 0, 'BINARY')
        >>> chimera_tiles(bqm, 1, 1, 4)     # doctest: +SKIP
        {(0, 0, 0): {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7]},
         (0, 1, 0): {8: [0], 9: [1], 10: [2], 11: [3], 12: [4], 13: [5], 14: [6], 15: [7]}}
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
    """Update a copy of a sample with replacement values.

    Args:
        sample (list/dict):
            Sample values as returned by dimod samplers to be copied.

        replacements (list/dict):
            Sample values to replace in the copied `sample`.

    Returns:
        list/dict: Copy of `sample` overwritten by specified values.

    Examples:
        >>> sample = {'a': 1, 'b': 1}
        >>> updated_sample(sample, {'b': 2})       # doctest: +SKIP
        {'a': 1, 'b': 2}

    """
    result = sample_as_dict(sample).copy()
    for k, v in sample_as_dict(replacements).items():
        result[k] = v
    return result


def sample_as_list(sample):
    """Return sample object in list format.

    Args:
        sample (list/dict/dimod.SampleView): Sample object formatted as a list,
        Numpy array, dict, or as returned by dimod samplers. Variable labeling
        must be numerical.

    Returns:
        list: Copy of `sample` formatted as a list.

    Examples:
        >>> sample = {0: 1, 1: 1}
        >>> sample_as_list(sample)
        [1, 1]

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
    """Return sample object in dict format.

    Args:
        sample (list/dict/dimod.SampleView): Sample object formatted as a list,
        Numpy array, dict, or as returned by dimod samplers.

    Returns:
        list: Copy of `sample` formatted as a dict, with variable indices as keys.

    Examples:
        >>> sample = [1, 2, 3]
        >>> sample_as_dict(sample)     # doctest: +SKIP
        {0: 1, 1: 2, 2: 3}

    """
    if isinstance(sample, dict):
        return sample
    if isinstance(sample, (list, numpy.ndarray)):
        sample = enumerate(sample)
    return dict(sample)


@dimod.decorators.vartype_argument('vartype')
def random_sample_seq(size, vartype):
    """Return a random sample.

    Args:
        size (int):
            Sample size (number of variables).

        vartype (:class:`dimod.Vartype`):
            Variable type; for example, `Vartype.SPIN`, `BINARY`, or `{-1, 1}`.

    Returns:
        dict: Random sample of `size` in length, with values from `vartype`.

    Examples:
        >>> random_sample_seq(4, dimod.BINARY)      # doctest: +SKIP
        {0: 0, 1: 1, 2: 0, 3: 0}

    """
    values = list(vartype.value)
    return {i: random.choice(values) for i in range(size)}


def random_sample(bqm):
    """Return a random sample for a binary quadratic model.

    Args:
        bqm (:obj:`.BinaryQuadraticModel`):
            Binary quadratic model (BQM).

    Returns:
        dict: A sample with random values for the BQM.

    Examples:
        >>> import dimod
        >>> bqm = dimod.BQM({}, {'ab': -1, 'bc': -1, 'ca': -1}, 0, 'BINARY')
        >>> random_sample(bqm)     # doctest: +SKIP
        {'a': 0, 'b': 1, 'c': 1}

    """
    values = list(bqm.vartype.value)
    return {i: random.choice(values) for i in bqm.variables}


def min_sample(bqm):
    """Return a sample with all variables set to the minimal value for a binary
    quadratic model.

    Args:
        bqm (:obj:`.BinaryQuadraticModel`):
            Binary quadratic model (BQM).

    Returns:
        dict: A sample with minimal values for all variables of the BQM.

    Examples:
        >>> import dimod
        >>> bqm = dimod.BQM({}, {'ab': -1, 'bc': -1, 'ca': -1}, 0, 'BINARY')
        >>> min_sample(bqm)     # doctest: +SKIP
        {'a': 0, 'b': 0, 'c': 0}

    """
    value = min(bqm.vartype.value)
    return {i: value for i in bqm.variables}


def max_sample(bqm):
    """Return a sample with all variables set to the maximal value for a binary
    quadratic model.

    Args:
        bqm (:obj:`.BinaryQuadraticModel`):
            Binary quadratic model (BQM).

    Returns:
        dict: A sample with maximal values for all variables of the BQM.

    Examples:
        >>> import dimod
        >>> bqm = dimod.BQM({}, {'ab': -1, 'bc': -1, 'ca': -1}, 0, 'BINARY')
        >>> max_sample(bqm)     # doctest: +SKIP
        {'a': 1, 'b': 1, 'c': 1}

    """
    value = max(bqm.vartype.value)
    return {i: value for i in bqm.variables}


def hstack_samplesets(base, *others, **kwargs):
    """Horizontally combine samples in `base` sampleset with samples in all the
    other samplesets provided in `*others`.

    Set of variables in the resulting sampleset is union of all variables in
    all joined samplesets.

    Number of samples in the resulting sampleset is set to the number of samples
    in the smallest input sampleset.

    Resulting sampleset inherits vartype from `bqm` (or `base` sampleset if
    `bqm` is undefined), and has energy calculated on `bqm` (or zero if `bqm` is
    undefined).
    """

    bqm = kwargs.pop('bqm', None)

    if bqm is None:
        vartype = base.vartype
    else:
        vartype = bqm.vartype

    samplesets = [base] + list(others)

    # calculate final set of variables
    variables = base.variables
    for sampleset in others:
        variables |= sampleset.variables

    # calculate final number of samples
    num_samples = min(len(sampleset) for sampleset in samplesets)

    # determine final dtype
    dtype = None
    dtypes = set(ss.record.sample.dtype.name for ss in samplesets)
    if len(dtypes) == 1:
        dtype = dtypes.pop()

    # prepare empty result sampleset
    samples = numpy.empty((num_samples, len(variables)), dtype=dtype)

    # copy over samplesets, one by one, from left to right
    for ss in samplesets:
        ss.change_vartype(vartype)
        mask = [variables.index(v) for v in ss.variables]
        samples[:, mask] = ss.record.sample[:num_samples]

    if bqm is None:
        energies = 0
    else:
        energies = bqm.energies((samples, variables))

    return dimod.SampleSet.from_samples(
        (samples, variables), energy=energies, vartype=vartype,
        info=None, num_occurrences=None, aggregate_samples=False,
        sort_labels=False)


def vstack_samplesets(*samplesets):
    """Vertically combine `*samplesets`. All samples must be over the same set
    of variables.
    """
    return dimod.sampleset.concatenate(samplesets)
