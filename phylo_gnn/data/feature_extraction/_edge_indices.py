from torch_geometric.typing import EdgeType  # type: ignore

from phylo_gnn.data.feature_extraction import (
    EdgeIndexExtractor,
    NodeNames,
    EdgeNames,
    EdgeIndicesExtractor,
)


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
