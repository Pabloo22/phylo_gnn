from numpy.typing import NDArray
import numpy as np

from phylo_gnn.encoding import VectorTree


def get_distances(
    vector_tree: VectorTree,
    edge_indices: dict[tuple[str, str, str], NDArray[np.int64]],
) -> dict[tuple[str, str, str], NDArray[np.float32]]:
    """Returns the distance between the nodes in the edge index.

    This function can be used to get edge attributes for the edges in the
    edge index.
    """
    result = {}
    for edge_type, edge_index in edge_indices.items():
        distances = vector_tree.get_distances(edge_index[0], edge_index[1])
        result[edge_type] = distances
    return result


def get_topological_distances(
    vector_tree: VectorTree,
    edge_indices: dict[tuple[str, str, str], NDArray[np.int64]],
) -> dict[tuple[str, str, str], NDArray[np.float32]]:
    """Returns the topological distance between the nodes in the edge index.

    This function can be used to get edge attributes for the edges in the
    edge index.
    """
    result = {}
    for edge_type, edge_index in edge_indices.items():
        distances = vector_tree.get_topological_distances(
            edge_index[0], edge_index[1]
        )
        result[edge_type] = distances
    return result
