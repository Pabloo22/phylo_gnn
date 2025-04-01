from phylo_gnn.encoding import VectorTree, EdgeType, NodeType


def get_basic_edge_index(vector_tree: VectorTree):
    """Returns (node, has_parent, node) and (node, has_child, node) edge
    indices types."""
