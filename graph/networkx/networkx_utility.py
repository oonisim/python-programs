"""
NetworkX utility module
"""
import logging
import sys
from typing import (
    List,
    Dict,
    Set,
    Tuple,
    Iterable,
    Any,
    Union
)

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def draw_graph(
        graph,
        seed: Union[int, np.random.RandomState, None] = None,
        k: float = 0.5,
        figsize: Tuple[float, float] = plt.gcf().get_size_inches()
) -> Dict[Any, np.ndarray]:
    """Draw graph
    https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html

    Args:
        graph: graph
        seed: RandomState instance or None for deterministic node layouts.
        k: Optimal distance between nodes. Increase this value to move nodes farther apart.
        figsize: Matplotlib figure size (inch, inch)
    """
    pos = nx.spring_layout(graph, k=k, seed=seed)
    plt.figure(figsize=figsize)
    nx.draw(
        graph,
        pos=pos,
        node_size=500,
        node_color="cyan",
        with_labels=True,
        # font_weight='bold'
    )
    return pos


def list_loops_in_directed_graph(graph: nx.DiGraph) -> List[Tuple[str, str, int]]:
    """Find loops in the directed graph
    Args:
        graph: directed graph
    Returns: A list of directed edges indicating the path taken for the loop. [] if no loop.
    """
    # try:
    #     find_cycle(G) return one cycle only.
    #     return nx.find_cycle(graph, orientation=None)
    # except nx.NetworkXNoCycle:
    #     return []

    return nx.simple_cycles(graph)


def get_nodes_in_loops_in_directed_graph(graph: nx.DiGraph) -> Set[str]:
    """Get nodes forming a loop
    Args:
        graph: graph
    Returns: A set of nodes included in any loops.
    """
    nodes: Set[str] = set()
    for route in list_loops_in_directed_graph(graph):
        nodes.update(set(route))

    return nodes


def get_source_nodes_reachable_to_target_nodes(graph: nx.DiGraph, targets: Iterable):
    """Get a set of nodes that can be reachable to the taget nodes.
    Args:
        graph: graph
        targets: target nodes
    Returns: Set of nodes reachable to the target nodes or empty set if none exists.
    """
    assert targets is not None
    result = set()
    for node in targets:
        result.update(nx.ancestors(graph, node))

    return result


def list_reverse_topological_sorted_paths_in_graph(graph: nx.DiGraph) -> List[List[str]]:
    """List all the independent sub graph paths with reverse-topologically sorted
    Objective:
        Get the order of nodes in each sub-graph where the depended on comes before depending.
        Analogy is the order of tasks to execute. If task x depends on y, and y on z, then
        the order is (z, y, x) so that the dependencies of x have been resolved before x.

        Topological sorts generates (x, y, z), hence reverse it.

    Topological Sort:
        Need a DAG (acyclic = no cycles in the graph) to be able to topologically order
        by arranging the vertices as a linear ordering.
        DAG -> Topologically-sort-able and vice-verta.

    References:
        * https://networkx.org/nx-guides/content/algorithms/dag/index.html.
        * https://networkx.org/documentation/stable/reference/algorithms/dag.html

    NetworkX.weakly_connected_components(G) provides a node group for each sub graph.
    G cannot include a loop, hence nodes in loops as well as the nodes that can reach
    them need to be excluded.

    Args:
        graph: parent graph including sub graphs
    Returns: list of paths, each of which is reverse topologically sorted
    """
    paths: List[List[str]] = []
    graph_to_proces: nx.DiGraph = graph

    # --------------------------------------------------------------------------------
    # Exclude nodes in loop and those nodes that can reach them.
    # --------------------------------------------------------------------------------
    if not nx.is_directed_acyclic_graph(graph):
        excludes: Set[str] = get_nodes_in_loops_in_directed_graph(graph)
        excludes.update(get_source_nodes_reachable_to_target_nodes(graph, excludes))
        includes = set(graph.nodes()) - excludes

        if len(includes) <= 0:
            return paths
        else:
            graph_to_proces = graph.subgraph(includes)

    # --------------------------------------------------------------------------------
    # List reverse-topologically-sorted paths
    # --------------------------------------------------------------------------------
    for group in sorted(nx.weakly_connected_components(graph_to_proces), key=len, reverse=True):
        subgraph = graph.subgraph(group)
        path = list(reversed(list(nx.topological_sort(subgraph))))
        paths.append(path)

    return paths