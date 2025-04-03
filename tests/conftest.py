# conftest.py
import pytest
import numpy as np
from phylo_gnn.feature_extraction import VectorTree


@pytest.fixture(scope="module")
def simple_newick_string():
    """Provides a simple Newick string for testing."""
    # Expected levelorder traversal (IDs assigned by ete3 based on this):
    # 0: Root (dist=0.0) -> Children [1, 2]
    # 1: Internal Node (dist=1.0) -> Children [3, 4]
    # 2: Node C (dist=2.0) -> Children [] (Leaf)
    # 3: Node A (dist=1.0) -> Children [] (Leaf)
    # 4: Node B (dist=1.0) -> Children [] (Leaf)
    return "((A:1.0, B:1.0):1.0, C:2.0);"


@pytest.fixture(scope="module")
def vector_tree_levelorder(simple_newick_string):
    """Provides a VectorTree instance created with levelorder traversal."""
    return VectorTree.from_newick(
        simple_newick_string, traverse_order="levelorder"
    )


@pytest.fixture(scope="module")
def sample_vector_tree() -> VectorTree:
    """
    Creates a sample VectorTree instance for testing.
    Tree structure:
          0 (root)
         / \
        1   2
       /|\ / \
      3 4 5 6 7
    Level Order: 0, 1, 2, 3, 4, 5, 6, 7
    Parents:    -1, 0, 0, 1, 1, 1, 2, 2
    Levels:      0, 1, 1, 2, 2, 2, 2, 2
    """
    branch_lengths = np.array(
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
    )
    children_indices = [
        [1, 2],  # Node 0 children
        [3, 4, 5],  # Node 1 children
        [6, 7],  # Node 2 children
        [],  # Node 3 children (leaf)
        [],  # Node 4 children (leaf)
        [],  # Node 5 children (leaf)
        [],  # Node 6 children (leaf)
        [],  # Node 7 children (leaf)
    ]
    tree = VectorTree(branch_lengths, children_indices)
    return tree
