import numpy as np
from numpy.typing import NDArray
from phylo_gnn.feature_extraction import VectorTree, NodeNames


def _initialize_node_features(
    vector_tree: VectorTree,
    num_features: int,
) -> NDArray[np.float32]:
    return np.zeros((vector_tree.num_nodes, num_features), dtype=np.float32)


def get_basic_node_features(
    vector_tree: VectorTree,
) -> NDArray[np.float32]:
    """Returns an X matrix of node features:

    - branch_length: Branch lengths of the tree.
    - distance_to_root: Distance from each node to the root.
    - distance_to_leaves: Distance from each node to the farthest leaf.
    - topological_distance_to_leaves: Number of edges from each node to the
        farthest leaf.
    - subtree_branch_lengths: Sum of branch lengths in the subtree rooted at
      each node.
    - level: Level of each node in the tree.
    - position in level: Position of each node within its level computed by
      ladderizing the tree according to subtree_branch_lengths.
    """
    # Initialize the feature matrix
    node_features = _initialize_node_features(vector_tree, 7)

    # Fill in the features
    node_features[:, 0] = vector_tree.branch_lengths
    node_features[:, 1] = vector_tree.distance_to_root
    node_features[:, 2] = vector_tree.distance_to_leaves
    node_features[:, 3] = vector_tree.topological_distance_to_leaves
    node_features[:, 4] = vector_tree.subtree_sum_branch_lengths
    node_features[:, 5] = vector_tree.levels
    node_features[:, 6] = vector_tree.set_positions_in_level()

    return node_features


def divide_into_internal_and_leaf_nodes(
    vector_tree: VectorTree,
    node_features: NDArray[np.float32],
) -> dict[str, NDArray[np.float32]]:
    leaves_indices = vector_tree.leaves_indices
    leaves_indices_mask = np.zeros(vector_tree.num_nodes, dtype=bool)
    leaves_indices_mask[leaves_indices] = True
    return {
        NodeNames.INTERNAL.value: node_features[~leaves_indices_mask],
        NodeNames.LEAF.value: node_features[leaves_indices_mask],
    }
