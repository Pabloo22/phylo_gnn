from torch_geometric.typing import EdgeType  # type: ignore

from phylo_gnn.data.feature_extraction import (
    EdgeIndexExtractor,
    NodeNames,
    EdgeNames,
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
