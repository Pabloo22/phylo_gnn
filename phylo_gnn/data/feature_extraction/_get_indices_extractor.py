from torch_geometric.typing import EdgeType  # type: ignore

from phylo_gnn.data.feature_extraction import (
    EDGE_INDEX_EXTRACTORS_MAPPING,
    EdgeIndicesExtractor,
)


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
