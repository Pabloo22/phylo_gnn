import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest

from phylo_gnn.encoding import VectorTree


def test_initialization_and_num_nodes(vector_tree_levelorder: VectorTree):
    """Test basic initialization and the num_nodes property."""
    tree = vector_tree_levelorder
    assert isinstance(tree, VectorTree)
    assert tree.num_nodes == 5  # Root, 1 internal, 3 leaves


def test_from_newick_levelorder_structure(vector_tree_levelorder: VectorTree):
    """Test branch lengths and children indices for levelorder."""
    tree = vector_tree_levelorder
    expected_branch_lengths = np.array(
        [0.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32
    )
    expected_children_indices = [
        [1, 2],  # Root (0) children
        [3, 4],  # Internal node (1) children
        [],  # Leaf C (2) children
        [],  # Leaf A (3) children
        [],  # Leaf B (4) children
    ]

    assert_allclose(tree.branch_lengths, expected_branch_lengths)
    assert tree.children_indices == expected_children_indices


def test_parent_indices(vector_tree_levelorder: VectorTree):
    """Test the parent_indices property."""
    tree = vector_tree_levelorder
    expected_parent_indices = np.array([-1, 0, 0, 1, 1], dtype=np.int64)
    assert_array_equal(tree.parent_indices, expected_parent_indices)


def test_levels_and_max_level(vector_tree_levelorder: VectorTree):
    """Test the levels and max_level properties."""
    tree = vector_tree_levelorder
    expected_levels = np.array([0, 1, 1, 2, 2], dtype=np.int64)
    assert_array_equal(tree.levels, expected_levels)
    assert tree.max_level == 2


def test_iter_by_level_forward(vector_tree_levelorder: VectorTree):
    """Test iterating through levels from root to leaves."""
    tree = vector_tree_levelorder
    expected_levels = {
        0: np.array([0]),
        1: np.array([1, 2]),
        2: np.array([3, 4]),
    }
    count = 0
    for level, indices in tree.iter_by_level(reverse=False):
        assert level in expected_levels
        assert_array_equal(indices, expected_levels[level])
        count += 1
    assert count == tree.max_level + 1


def test_iter_by_level_reverse(vector_tree_levelorder: VectorTree):
    """Test iterating through levels from leaves to root."""
    tree = vector_tree_levelorder
    expected_levels = {
        2: np.array([3, 4]),
        1: np.array([1, 2]),
        0: np.array([0]),
    }
    count = 0
    for level, indices in tree.iter_by_level(reverse=True):
        assert level in expected_levels
        assert_array_equal(indices, expected_levels[level])
        count += 1
    assert count == tree.max_level + 1


def test_subtree_sum_branch_lengths(vector_tree_levelorder: VectorTree):
    """Test calculation of subtree branch length sums."""
    tree = vector_tree_levelorder
    # Expected sums (calculated bottom-up):
    # Leaves A, B, C (indices 3, 4, 2): 0.0
    # Internal node (index 1): branch(A) + branch(B) + subtree(A) + subtree(B)
    #                         = 1.0 + 1.0 + 0.0 + 0.0 = 2.0
    # Root (index 0): branch(Internal) + branch(C) + subtree(Internal) +
    # subtree(C)
    #                 = 1.0 + 2.0 + 2.0 + 0.0 = 5.0
    expected_sums = np.array([5.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert_allclose(tree.subtree_sum_branch_lengths, expected_sums, atol=1e-6)


def test_dist_to_root(vector_tree_levelorder: VectorTree):
    """Test calculation of distance from each node to the root."""
    tree = vector_tree_levelorder
    # Expected distances:
    # Root (0): 0.0
    # Internal (1): dist(Root) + branch(Internal) = 0.0 + 1.0 = 1.0
    # C (2): dist(Root) + branch(C) = 0.0 + 2.0 = 2.0
    # A (3): dist(Internal) + branch(A) = 1.0 + 1.0 = 2.0
    # B (4): dist(Internal) + branch(B) = 1.0 + 1.0 = 2.0
    expected_distances = np.array([0.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float32)
    assert_allclose(tree.distance_to_root, expected_distances, atol=1e-6)


def test_dist_to_leaves(vector_tree_levelorder: VectorTree):
    tree = vector_tree_levelorder
    # Expected distances (calculated bottom-up):
    # Leaves A, B, C (indices 3, 4, 2): 0.0
    # Internal node (1): max(dist_leaf(A)+branch(A), dist_leaf(B)+branch(B))
    #                  = max(0.0+1.0, 0.0+1.0) = 1.0
    # Root node (0): max(dist_leaf(Internal)+branch(Internal), dist_leaf(C)+
    # branch(C))
    #              = max(1.0 + 1.0, 0.0 + 2.0) = max(2.0, 2.0) = 2.0
    expected_distances = np.array([2.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert_allclose(
        tree.distance_to_leaves, expected_distances, atol=1e-6
    )


def test_leaves_indices_and_num_leaves(vector_tree_levelorder: VectorTree):
    """Test identification of leaf nodes and their count."""
    tree = vector_tree_levelorder
    expected_leaves_indices = sorted([2, 3, 4])  # Order might vary, so sort
    assert sorted(tree.leaves_indices) == expected_leaves_indices
    assert tree.num_leaves == 3


def test_inverse_levels(vector_tree_levelorder: VectorTree):
    """Test the inverse_levels property."""
    tree = vector_tree_levelorder
    # max_level = 2
    # levels = [0, 1, 1, 2, 2]
    expected_inverse_levels = np.array(
        [2, 1, 1, 0, 0], dtype=np.int64
    )  # max_level - levels
    assert_array_equal(tree.inverse_levels, expected_inverse_levels)


def test_single_node_tree():
    """Test a tree consisting of only a single root node."""
    tree = VectorTree.from_newick("A:0;")

    assert tree.num_nodes == 1
    assert_allclose(tree.branch_lengths, np.array([0.0], dtype=np.float32))
    assert tree.children_indices == [[]]
    assert_array_equal(tree.parent_indices, np.array([-1], dtype=np.int64))
    assert_array_equal(tree.levels, np.array([0], dtype=np.int64))
    assert tree.max_level == 0
    assert_allclose(
        tree.subtree_sum_branch_lengths, np.array([0.0], dtype=np.float32)
    )
    assert_allclose(tree.distance_to_root, np.array([0.0], dtype=np.float32))
    assert_allclose(tree.distance_to_leaves, np.array([0.0], dtype=np.float32))
    assert tree.leaves_indices == [0]
    assert tree.num_leaves == 1
    assert_array_equal(tree.inverse_levels, np.array([0], dtype=np.int64))


def test_positions_in_level(vector_tree_levelorder: VectorTree):
    """Test the positions_in_level property based on ladderization."""
    tree = vector_tree_levelorder
    # Expected positions:
    # Root (0): 0
    # Level 1 Children of Root (0): Node 1 (sum 2.0), Node 2 (sum 0.0)
    # -> Sorted [1, 2] -> Pos [0, 1]
    # Level 2 Children of Node 1: Node 3 (sum 0.0), Node 4 (sum 0.0)
    # -> Sorted [3, 4] (tie-break by index) -> Pos [0, 1]
    expected_positions = np.array([0, 0, 1, 0, 1], dtype=np.int64)
    assert_array_equal(tree.set_positions_in_level(), expected_positions)


def test_num_leaves_array(vector_tree_levelorder: VectorTree):
    """Test the num_leaves_array property."""
    tree = vector_tree_levelorder
    # Expected leaf counts:
    # Node 0 (root): 3 leaves (2, 3, 4)
    # Node 1: 2 leaves (3, 4)
    # Node 2: 1 leaf (itself)
    # Node 3: 1 leaf (itself)
    # Node 4: 1 leaf (itself)
    expected_num_leaves = np.array([3, 2, 1, 1, 1], dtype=np.int64)
    assert_array_equal(tree.num_leaves_array, expected_num_leaves)


def test_num_leaves_array_single_node():
    """Test num_leaves_array for a single node tree."""
    tree = VectorTree.from_newick("A:0;")
    expected_num_leaves = np.array([1], dtype=np.int64)
    assert_array_equal(tree.num_leaves_array, expected_num_leaves)


def test_mrca_empty_input(sample_vector_tree: VectorTree):
    """Test MRCA calculation with empty input arrays."""
    nodes_1 = np.array([], dtype=np.int64)
    nodes_2 = np.array([], dtype=np.int64)
    expected = np.array([], dtype=np.int64)
    result = sample_vector_tree.get_most_recent_common_ancestors(
        nodes_1, nodes_2
    )
    assert_array_equal(result, expected)


def test_mrca_identical_nodes(sample_vector_tree: VectorTree):
    """Test MRCA when input nodes in a pair are identical."""
    nodes_1 = np.array([3, 6, 0, 1], dtype=np.int64)
    nodes_2 = np.array([3, 6, 0, 1], dtype=np.int64)
    expected = np.array(
        [3, 6, 0, 1], dtype=np.int64
    )  # MRCA is the node itself
    result = sample_vector_tree.get_most_recent_common_ancestors(
        nodes_1, nodes_2
    )
    assert_array_equal(result, expected)


def test_mrca_direct_ancestor(sample_vector_tree: VectorTree):
    """Test MRCA when one node is a direct ancestor of the other."""
    # Pairs: (descendant, ancestor)
    nodes_1 = np.array([3, 7, 1, 4, 0], dtype=np.int64)
    nodes_2 = np.array(
        [1, 2, 0, 1, 0], dtype=np.int64
    )  # Corresponding ancestors
    expected = np.array(
        [1, 2, 0, 1, 0], dtype=np.int64
    )  # Expected MRCA is the ancestor
    result = sample_vector_tree.get_most_recent_common_ancestors(
        nodes_1, nodes_2
    )
    assert_array_equal(result, expected)

    # Pairs: (ancestor, descendant) - order shouldn't matter
    nodes_1_rev = np.array([1, 2, 0, 1, 0], dtype=np.int64)
    nodes_2_rev = np.array([3, 7, 1, 4, 0], dtype=np.int64)
    expected_rev = np.array([1, 2, 0, 1, 0], dtype=np.int64)
    result_rev = sample_vector_tree.get_most_recent_common_ancestors(
        nodes_1_rev, nodes_2_rev
    )
    assert_array_equal(result_rev, expected_rev)


def test_mrca_siblings(sample_vector_tree: VectorTree):
    """Test MRCA for sibling nodes."""
    # Pairs of siblings
    nodes_1 = np.array([3, 6, 1], dtype=np.int64)
    nodes_2 = np.array([4, 7, 2], dtype=np.int64)  # Corresponding siblings
    expected = np.array(
        [1, 2, 0], dtype=np.int64
    )  # Expected MRCA is the parent
    result = sample_vector_tree.get_most_recent_common_ancestors(
        nodes_1, nodes_2
    )
    assert_array_equal(result, expected)


def test_mrca_cousins_and_different_levels(sample_vector_tree: VectorTree):
    """Test MRCA for nodes at same or different levels (cousins)."""
    # Pairs: (node_level_2, node_level_2) -> MRCA level 0 or 1
    nodes_1 = np.array([3, 4, 3, 6], dtype=np.int64)
    nodes_2 = np.array([5, 6, 7, 5], dtype=np.int64)
    expected = np.array([1, 0, 0, 0], dtype=np.int64)  # MRCAs are 1, 0, 0, 0
    result = sample_vector_tree.get_most_recent_common_ancestors(
        nodes_1, nodes_2
    )
    assert_array_equal(result, expected)

    # Pairs: (node_level_1, node_level_2) -> MRCA level 0 or 1
    nodes_1_diff = np.array([1, 2, 1], dtype=np.int64)
    nodes_2_diff = np.array([6, 5, 7], dtype=np.int64)
    expected_diff = np.array([0, 0, 0], dtype=np.int64)  # MRCAs are 0, 0, 0
    result_diff = sample_vector_tree.get_most_recent_common_ancestors(
        nodes_1_diff, nodes_2_diff
    )
    assert_array_equal(result_diff, expected_diff)


def test_mrca_involving_root(sample_vector_tree: VectorTree):
    """Test MRCA when one node is the root or the MRCA is the root."""
    nodes_1 = np.array([0, 0, 3, 1], dtype=np.int64)
    nodes_2 = np.array([0, 5, 6, 2], dtype=np.int64)
    expected = np.array(
        [0, 0, 0, 0], dtype=np.int64
    )  # MRCA involving root is root
    result = sample_vector_tree.get_most_recent_common_ancestors(
        nodes_1, nodes_2
    )
    assert_array_equal(result, expected)


def test_mrca_mixed_cases(sample_vector_tree: VectorTree):
    """Test MRCA with a mix of different relationship types."""
    nodes_1 = np.array([3, 1, 4, 7, 6, 0], dtype=np.int64)
    nodes_2 = np.array(
        [
            3,  # Identical
            0,  # Ancestor
            6,  # Cousin (MRCA root)
            2,  # Ancestor
            7,  # Sibling (MRCA parent 2)
            0,
        ],  # Identical (root)
        dtype=np.int64,
    )
    expected = np.array([3, 0, 0, 2, 2, 0], dtype=np.int64)
    result = sample_vector_tree.get_most_recent_common_ancestors(
        nodes_1, nodes_2
    )
    assert_array_equal(result, expected)


def test_mrca_different_input_types(sample_vector_tree: VectorTree):
    """Test MRCA with Python lists and different numpy int types."""
    nodes_1_list = [3, 7]
    nodes_2_list = [6, 5]
    expected_list = np.array([0, 0], dtype=np.int64)
    result_list = sample_vector_tree.get_most_recent_common_ancestors(
        nodes_1_list, nodes_2_list
    )
    assert_array_equal(result_list, expected_list)
    assert result_list.dtype == np.int64  # Implementation should return int64

    nodes_1_int32 = np.array([4, 1], dtype=np.int32)
    nodes_2_int32 = np.array([7, 2], dtype=np.int32)
    expected_int32 = np.array([0, 0], dtype=np.int64)
    result_int32 = sample_vector_tree.get_most_recent_common_ancestors(
        nodes_1_int32, nodes_2_int32
    )
    assert_array_equal(result_int32, expected_int32)
    assert result_int32.dtype == np.int64


def test_mrca_different_lengths_raises_error(sample_vector_tree: VectorTree):
    """Test that providing arrays of different lengths raises ValueError."""
    nodes_1 = np.array([1, 2], dtype=np.int64)
    nodes_2 = np.array([3], dtype=np.int64)
    with pytest.raises(
        ValueError, match="Input arrays must have the same length"
    ):
        sample_vector_tree.get_most_recent_common_ancestors(nodes_1, nodes_2)
