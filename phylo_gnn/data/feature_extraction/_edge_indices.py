from torch_geometric.typing import EdgeType  # type: ignore
from numpy.typing import NDArray
import numpy as np

from phylo_gnn.data.feature_extraction import (
    EdgeIndexExtractor,
    NodeNames,
    EdgeNames,
    EdgeIndicesExtractor,
    VectorTree,
)


def get_node_is_in_level_edge_index(
    vector_tree: VectorTree,
) -> NDArray[np.int64]:
    """Creates the edge index for the 'is_in_level' relation.

    This edge index connects nodes to their respective level nodes:

    [node_id, level_id].

    Args:
        vector_tree: A VectorTree object.

    Returns:
        The edge index for the 'is_in_level' relation.
    """
    node_ids = np.arange(vector_tree.num_nodes, dtype=np.int64)
    level_ids = vector_tree.levels
    return np.stack((node_ids, level_ids), axis=0)


def get_level_has_node_edge_index(
    vector_tree: VectorTree,
) -> NDArray[np.int64]:
    """Creates the edge index for the 'has_node' relation.

    This edge index connects levels to their respective node ids:

    [level_id, node_id].

    Args:
        vector_tree: A VectorTree object.

    Returns:
        The edge index for the 'has_node' relation.
    """
    node_ids = np.arange(vector_tree.num_nodes, dtype=np.int64)
    level_ids = vector_tree.levels
    return np.stack((level_ids, node_ids), axis=0)


EDGE_INDEX_EXTRACTORS_MAPPING: dict[EdgeType, EdgeIndexExtractor] = {
    (
        NodeNames.NODE.value,
        EdgeNames.HAS_PARENT.value,
        NodeNames.NODE.value,
    ): lambda vector_tree: vector_tree.children_to_parent_edge_index,
    (
        NodeNames.NODE.value,
        EdgeNames.HAS_CHILD.value,
        NodeNames.NODE.value,
    ): lambda vector_tree: vector_tree.parent_to_children_edge_index,
    (
        NodeNames.NODE.value,
        EdgeNames.IS_IN_LEVEL.value,
        NodeNames.NODE.value,
    ): get_node_is_in_level_edge_index,
}


def get_edge_indices_extractor(
    edge_types: list[EdgeType],
) -> EdgeIndicesExtractor:
    """Returns a function that extracts edge indices for the given edge types.

    Args:
        edge_types: List of edge types to extract indices for.

    Returns:
        A function that takes a vector tree and returns a dictionary of edge
        indices.
    """
    return lambda vector_tree: {
        edge_type: EDGE_INDEX_EXTRACTORS_MAPPING[edge_type](vector_tree)
        for edge_type in edge_types
    }
