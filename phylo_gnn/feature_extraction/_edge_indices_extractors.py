from numpy.typing import NDArray
import numpy as np

from phylo_gnn.feature_extraction import VectorTree, EdgeNames, NodeNames


def get_basic_edge_index(
    vector_tree: VectorTree,
) -> dict[tuple[str, str, str], NDArray[np.int64]]:
    """Returns (node, has_parent, node) and (node, has_child, node) edge
    indices types."""

    return {
        (
            NodeNames.NODE.value,
            EdgeNames.HAS_PARENT.value,
            NodeNames.NODE.value,
        ): vector_tree.children_to_parent_edge_index,
        (
            NodeNames.NODE.value,
            EdgeNames.HAS_CHILD.value,
            NodeNames.NODE.value,
        ): vector_tree.parent_to_children_edge_index,
    }
