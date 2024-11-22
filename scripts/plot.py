# Copyright 2019 D-Wave Systems Inc.
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

"""Plot utility function featured in dwave-hybrid webinar [1]_ is included here
for completeness (for lack of a better place).

.. [1] https://www.youtube.com/watch?v=EW44reo8Bn0
"""

import networkx as nx

# use this for inline jupyter lab plots:
# %matplotlib widget

import matplotlib

# `Agg` and `TkAgg` are known to segfault in interactive sessions
# matplotlib.use('Agg')
# matplotlib.use('TkAgg')

# `Qt5Agg` works good in ipython/console
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt


def plot(G, subgraph=None, pos=None):
    if pos is None:
        pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        pos = nx.spring_layout(G)

    if not subgraph:
        graph_edge_alpha = 0.2
        graph_node_alpha = 0.9

    else:
        graph_edge_alpha = 0.05
        graph_node_alpha = 0.1

        subgraph_edge_alpha = 0.4
        subgraph_node_alpha = 0.9

    fig = plt.figure()

    # edges for full graph
    nx.draw_networkx_edges(G, pos, alpha=graph_edge_alpha)

    # edges for subgraph
    if subgraph:
        nx.draw_networkx_edges(subgraph, pos, alpha=subgraph_edge_alpha)

    # nodes for full graph
    nodelist = G.nodes.keys()
    max_degree = max(dict(G.degree).values())

    # (top 70% of reds cmap)
    normalized_degrees = [0.3 + 0.7 * G.degree[n] / max_degree for n in nodelist]
    node_color = plt.cm.Reds([0] + normalized_degrees)[1:]
    nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_size=80, node_color=node_color, alpha=graph_node_alpha)

    # nodes for subgraph
    if subgraph:
        sub_nodelist = subgraph.nodes.keys()
        sub_node_color = [node_color[n] for n in sub_nodelist]
        nx.draw_networkx_nodes(subgraph, pos, nodelist=sub_nodelist, node_size=80, node_color=sub_node_color, alpha=subgraph_node_alpha)

    plt.axis('off')
    plt.show()
