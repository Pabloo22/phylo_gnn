import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from phylo_gnn.data import VectorTree  # Assuming your module structure


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
    assert_allclose(tree.dist_to_root, expected_distances, atol=1e-6)


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
    assert_allclose(tree.dist_to_leaves, expected_distances, atol=1e-6)


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
    assert_allclose(tree.dist_to_root, np.array([0.0], dtype=np.float32))
    assert_allclose(tree.dist_to_leaves, np.array([0.0], dtype=np.float32))
    assert tree.leaves_indices == [0]
    assert tree.num_leaves == 1
    assert_array_equal(tree.inverse_levels, np.array([0], dtype=np.int64))
