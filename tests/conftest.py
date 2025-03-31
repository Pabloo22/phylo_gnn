# conftest.py
import pytest

from phylo_gnn.data import VectorTree


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
