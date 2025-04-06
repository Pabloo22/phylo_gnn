from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cached_property, partial
from typing import Any

import numpy as np
from numpy.typing import NDArray
import ete3  # type: ignore[import-untyped]

from phylo_gnn.ete3_utils import ID_ATTR, set_node_ids


def extract_neg_subtree_sum_branch_lengths(
    node_idx: int, vector_tree: VectorTree
) -> float:
    """Extracts the negative sum of branch lengths in the subtree for a given
    node.

    Args:
        node_idx: The index of the node for which to calculate diversity
        criteria.
    """
    node_subtree_sum = vector_tree.subtree_sum_branch_lengths[node_idx]
    return -node_subtree_sum


_DEFAULT_SORTING_CRITERIA = extract_neg_subtree_sum_branch_lengths


class VectorTree:

    def __init__(
        self,
        branch_lengths: NDArray[np.float32],
        children_indices: list[list[int]],
    ):
        self.branch_lengths = branch_lengths
        self.children_indices = children_indices
        self._positions_by_level: None | NDArray[np.int64] = None

    @property
    def position_in_level(self) -> NDArray[np.int64]:
        """Returns the position of each node in its level.

        The position is defined as the index of the node in the list of
        children of its parent. The root node has no parent, so its position
        is set to -1.
        """
        if self._positions_by_level is None:
            raise ValueError(
                "Position in level is not computed yet. "
                "Call `set_position_in_level` first."
            )
        return self._positions_by_level

    @property
    def num_nodes(self) -> int:
        return len(self.branch_lengths)

    @classmethod
    def from_newick(
        cls,
        newick: str,
        traverse_order: str = "levelorder",
        nwk_format: int = 1,
        **kwargs,
    ) -> VectorTree:
        tree = ete3.TreeNode(newick, format=nwk_format, **kwargs)
        return cls.from_ete3(tree, traverse_order)

    @classmethod
    def from_ete3(
        cls,
        tree: ete3.Tree,
        traverse_order: str = "levelorder",
    ) -> VectorTree:
        set_node_ids(tree, traverse_strategy=traverse_order)
        branch_lengths = np.array(
            [node.dist for node in tree.traverse(traverse_order)],
            dtype=np.float32,
        )
        children_indices = [
            [getattr(child, ID_ATTR) for child in node.get_children()]
            for node in tree.traverse(traverse_order)
        ]
        return cls(branch_lengths, children_indices)

    @cached_property
    def parent_indices(self) -> NDArray[np.int64]:
        """Returns an array of parent indices for each node.

        The root node has no parent, so its index is set to -1.
        """
        parent_indices = np.full(self.num_nodes, -1, dtype=np.int64)
        for parent, children in enumerate(self.children_indices):
            for child in children:
                parent_indices[child] = parent
        return parent_indices

    @cached_property
    def levels(self) -> NDArray[np.int64]:
        """The level of each node in the tree.

        The root node is at level 0, its children are at level 1, and so on.
        The level of a node is defined as the number of edges on the path from
        the root to that node.
        """
        level = np.zeros(self.num_nodes, dtype=np.int64)
        root_id = np.where(self.parent_indices == -1)[0][0]
        current_level = [root_id]
        next_level = []
        while current_level:
            for node in current_level:
                for child in self.children_indices[node]:
                    level[child] = level[node] + 1
                    next_level.append(child)
            current_level = next_level
            next_level = []
        return level

    def iter_by_level(self, reverse: bool = False):
        """Iterates through the nodes of the tree by level.

        Args:
            reverse:
                If True, iterate from leaves to root (max level to 0).
                If False, iterate from root to leaves (0 to max level).

        Yields:
            (level, indices) where indices is an array of node indices
            at that level.
        """
        level_range = (
            range(self.max_level, -1, -1)
            if reverse
            else range(self.max_level + 1)
        )
        for level in level_range:
            level_mask = self.levels == level
            level_nodes = np.where(level_mask)[0]
            yield level, level_nodes

    @cached_property
    def max_level(self) -> int:
        return int(np.max(self.levels))

    @cached_property
    def is_ultrametric(self) -> bool:
        """Checks if the tree is ultrametric.

        An ultrametric tree is one where all leaves are equidistant from the
        root.
        """
        return np.allclose(
            self.distance_to_root[self.leaves_indices],
            self.distance_to_root[self.leaves_indices][0],
        )

    @cached_property
    def subtree_sum_branch_lengths(self) -> NDArray[np.float32]:
        """Computes the sum of branch lengths in the subtree rooted at each
        node.

        Returns:
            numpy array with the sum of branch lengths for each node's subtree
        """
        subtree_sums = np.zeros_like(self.branch_lengths)
        for level, level_nodes in self.iter_by_level(reverse=True):
            if level == 0:  # Skip root level since it has no parent
                continue

            parents = self.parent_indices[level_nodes]
            assert all(
                parents >= 0
            ), "All nodes should have a parent except the root"

            # Add branch lengths and subtree sums
            np.add.at(subtree_sums, parents, self.branch_lengths[level_nodes])
            np.add.at(subtree_sums, parents, subtree_sums[level_nodes])
        return subtree_sums

    @cached_property
    def distance_to_root(self) -> NDArray[np.float32]:
        """Returns the sum of branch lengths from each node to the root."""
        distances = np.zeros_like(self.branch_lengths)
        for level, level_nodes in self.iter_by_level():
            if level == 0:  # Skip root node
                continue
            parents = self.parent_indices[level_nodes]
            distances[level_nodes] = (
                distances[parents] + self.branch_lengths[level_nodes]
            )
        return distances

    @cached_property
    def topological_distance_to_root(self) -> NDArray[np.float32]:
        """Returns the topological distance (number of edges) from each node
        to the root."""
        distances = np.zeros(self.num_nodes, dtype=np.float32)
        for level, level_nodes in self.iter_by_level():
            if level == 0:  # Skip root node
                continue
            parents = self.parent_indices[level_nodes]
            distances[level_nodes] = distances[parents] + 1
        return distances

    @cached_property
    def distance_to_leaves(self) -> NDArray[np.float32]:
        """Returns the distance from each node to its descendant leaves.

        The distance is defined as the maximum distance from the node to any
        of its descendant leaves.
        """
        return self._get_distance_to_leaves(topological=False)

    @cached_property
    def topological_distance_to_leaves(self) -> NDArray[np.float32]:
        """Returns the distance from each node to its descendant leaves.

        The distance is defined as the maximum distance from the node to any
        of its descendant leaves.
        """
        return self._get_distance_to_leaves(topological=True)

    def _get_distance_to_leaves(
        self, topological: bool = False
    ) -> NDArray[np.float32]:
        """Returns the distance from each node to its descendant leaves.

        The distance is defined as the maximum distance from the node to any
        of its descendant leaves.
        """
        if self.is_ultrametric:
            # More efficient calculation for ultrametric trees
            return self._get_distance_to_leaves_in_ultrametric_tree()
        distances = np.zeros_like(self.branch_lengths)
        branch_lengths = (
            self.branch_lengths
            if not topological
            else np.ones_like(self.branch_lengths)
        )
        for _, level_nodes in self.iter_by_level(reverse=True):
            for node in level_nodes:
                children = self.children_indices[node]
                if not children:
                    continue
                child_distances = [
                    float(distances[child] + branch_lengths[child])
                    for child in children
                ]
                distances[node] = max(child_distances)
        return distances

    def _get_distance_to_leaves_in_ultrametric_tree(
        self, topological: bool = False
    ) -> NDArray[np.float32]:
        """More vectorized version of dist_to_leaves for ultrametric trees."""
        # For ultrametric trees, the distance from any node to its descendant
        # leaves is the same for all leaves in its subtree
        distances = np.zeros_like(self.branch_lengths)
        branch_lengths = (
            self.branch_lengths
            if not topological
            else np.ones_like(self.branch_lengths)
        )
        for _, level_nodes in self.iter_by_level(reverse=True):
            non_leaf_nodes = [
                node for node in level_nodes if not self.is_leaf(node)
            ]
            first_children = [
                self.children_indices[node][0] for node in non_leaf_nodes
            ]
            distances[non_leaf_nodes] = (
                distances[first_children] + branch_lengths[first_children]
            )
        return distances

    @cached_property
    def leaves_indices(self) -> list[int]:
        """Returns the indices of the leaves in the tree."""
        return [node for node in range(self.num_nodes) if self.is_leaf(node)]

    @property
    def num_leaves(self) -> int:
        """Returns the number of leaves in the tree."""
        return len(self.leaves_indices)

    @cached_property
    def num_leaves_array(self) -> NDArray[np.int64]:
        """Returns the number of leaves under each node's subtree."""
        num_leaves = np.zeros(self.num_nodes, dtype=np.int64)
        num_leaves[self.leaves_indices] = 1
        for level, level_nodes in self.iter_by_level(reverse=True):
            if level == 0:  # Root has no parent
                continue
            parents = self.parent_indices[level_nodes]
            np.add.at(num_leaves, parents, num_leaves[level_nodes])

        assert np.max(num_leaves) == self.num_leaves, (
            f"Max number of leaves {np.max(num_leaves)} "
            f"does not match expected {self.num_leaves}"
            f" for {self.num_nodes} nodes"
            f" and {self.max_level} levels."
            f" Num leaves array: {num_leaves}"
        )
        assert np.min(num_leaves) == 1
        return num_leaves

    def is_leaf(self, node_idx: int) -> bool:
        """Checks if a given node is a leaf.

         Args:
             node_idx: The index of the node to check.

        Returns:
             True if the node is a leaf, False otherwise.
        """
        return not self.children_indices[node_idx]

    @cached_property
    def inverse_levels(self) -> NDArray[np.int64]:
        """Returns the inverse levels of the tree.

        The inverse level of a node is defined as the maximum level minus the
        level of that node.
        """
        return self.max_level - self.levels

    def get_siblings(self, node: int) -> list[int]:
        """Returns the sibling nodes of a given node.

        Args:
            node: The index of the node to find siblings for.

        Returns:
            A list of sibling node indices.
        """
        parent = self.parent_indices[node]
        if parent == -1:
            return []
        return [
            child for child in self.children_indices[parent] if child != node
        ]

    def set_positions_in_level(
        self,
        sorting_criteria: Callable = _DEFAULT_SORTING_CRITERIA,
    ) -> NDArray[np.int64]:
        """Rank nodes in each level based on a key function.

        Args:
            key:
                A callable that takes a node index and a current ranks array
                and returns a tuple of values to sort by.
            reverse_iter:
                If True, iterate from leaves to root
                (max level to 0). If False, iterate from root to leaves
                (0 to max level).

        Returns:
            An array of ranks for each node in the tree.
        """
        sorting_criteria = partial(sorting_criteria, vector_tree=self)

        def key_fn(node_idx: int) -> tuple[int, Any, int]:
            parent_rank = self.position_in_level[self.parent_indices[node_idx]]
            return parent_rank, sorting_criteria(node_idx), node_idx

        try:
            self._positions_by_level = np.zeros(self.num_nodes, dtype=np.int64)
            for level, level_nodes in self.iter_by_level():
                if level == 0:
                    continue
                sorted_nodes = sorted(level_nodes, key=key_fn)
                for pos, node in enumerate(sorted_nodes):
                    self._positions_by_level[node] = pos
        except Exception as e:
            self._positions_by_level = None
            raise e

        return self._positions_by_level

    def get_most_recent_common_ancestor(self, node1: int, node2: int) -> int:
        """Returns the most recent common ancestor of two nodes."""
        if node1 == node2:
            return node1
        ancestor_1 = node1
        ancestor_2 = node2
        while ancestor_1 != ancestor_2:
            if self.levels[ancestor_1] > self.levels[ancestor_2]:
                ancestor_1 = self.parent_indices[ancestor_1]
            else:
                ancestor_2 = self.parent_indices[ancestor_2]
            if ancestor_1 == -1 or ancestor_2 == -1:
                raise ValueError(
                    f"Nodes {node1} and {node2} are not in the same tree."
                )
        return ancestor_1

    def get_most_recent_common_ancestors(
        self,
        nodes_1: NDArray[np.integer] | Sequence[int],
        nodes_2: NDArray[np.integer] | Sequence[int],
    ) -> NDArray[np.int64]:
        """Returns the most recent common ancestors of two sets of nodes."""
        # Ensure input arrays have the same length
        if len(nodes_1) != len(nodes_2):
            raise ValueError("Input arrays must have the same length")

        # Convert inputs to numpy arrays with int64 dtype
        n1 = np.asarray(nodes_1, dtype=np.int64)
        n2 = np.asarray(nodes_2, dtype=np.int64)

        # Initialize result array
        result = np.empty_like(n1)

        # Fast path: same nodes are their own MRCA
        same = n1 == n2
        result[same] = n1[same]

        # Only process different nodes
        diff = ~same
        if not np.any(diff):
            return result

        # Extract the different nodes for processing
        a1 = n1[diff].copy()
        a2 = n2[diff].copy()

        # Get levels for equalization
        l1 = self.levels[a1]
        l2 = self.levels[a2]

        # Equalize levels first by moving deeper nodes up
        while np.any(l1 != l2):
            deeper1 = l1 > l2
            deeper2 = l2 > l1

            a1[deeper1] = self.parent_indices[a1[deeper1]]
            l1[deeper1] -= 1
            a2[deeper2] = self.parent_indices[a2[deeper2]]
            l2[deeper2] -= 1

        # Now move both up until they meet
        while np.any(a1 != a2):
            not_equal = a1 != a2
            a1[not_equal] = self.parent_indices[a1[not_equal]]
            a2[not_equal] = self.parent_indices[a2[not_equal]]

            # Check if any nodes reached root without finding MRCA
            invalid = (a1 == -1) | (a2 == -1)
            if np.any(invalid):
                raise ValueError("Some nodes do not share a common ancestor")

        # Store results
        result[diff] = a1

        return result

    def get_distance(self, node_1: int, node_2: int) -> float:
        """Returns the distance between two nodes."""
        if node_1 == node_2:
            return 0.0
        ancestor = self.get_most_recent_common_ancestor(node_1, node_2)
        return (
            self.distance_to_root[node_1]
            + self.distance_to_root[node_2]
            - 2 * self.distance_to_root[ancestor]
        )

    def get_distances(
        self,
        nodes_1: NDArray[np.integer] | Sequence[int],
        nodes_2: NDArray[np.integer] | Sequence[int],
    ) -> NDArray[np.float32]:
        """Returns the distances between two sets of nodes.

        Args:
            nodes_1:
                First set of node indices
            nodes_2:
                Second set of node indices. Must have the same length as
                nodes_1.

        Returns:
            Array of distances between corresponding pairs of nodes.
        """
        if len(nodes_1) != len(nodes_2):
            raise ValueError("Input arrays must have the same length")
        mrcas = self.get_most_recent_common_ancestors(nodes_1, nodes_2)
        distances = (
            self.distance_to_root[nodes_1]
            + self.distance_to_root[nodes_2]
            - 2 * self.distance_to_root[mrcas]
        ).astype(np.float32)
        return distances

    def get_topological_distances(
        self,
        nodes_1: NDArray[np.integer] | Sequence[int],
        nodes_2: NDArray[np.integer] | Sequence[int],
    ) -> NDArray[np.float32]:
        """Returns the topological distances between two sets of nodes.

        Args:
            nodes_1:
                First set of node indices
            nodes_2:
                Second set of node indices. Must have the same length as
                nodes_1.

        Returns:
            Array of topological distances between corresponding pairs of
            nodes.
        """
        if len(nodes_1) != len(nodes_2):
            raise ValueError("Input arrays must have the same length")
        mrcas = self.get_most_recent_common_ancestors(nodes_1, nodes_2)
        distances = (
            self.topological_distance_to_leaves[nodes_1]
            + self.topological_distance_to_leaves[nodes_2]
            - 2 * self.topological_distance_to_leaves[mrcas]
        ).astype(np.float32)
        return distances

    @cached_property
    def is_binary(self) -> bool:
        """Checks if the tree is binary.

        A binary tree is a tree where each node has at most two children.
        """
        return all(len(children) <= 2 for children in self.children_indices)

    @cached_property
    def depth(self) -> int:
        """Returns the depth of the tree.

        The depth of a tree is the length of the longest topological path
        from the root to a leaf.
        """
        return int(np.max(self.levels)) + 1

    @cached_property
    def height(self) -> float:
        """Returns the height of the tree.

        The height of a tree is the length of the longest path from the
        root to any leaf, measured in branch lengths.
        """
        return float(np.max(self.distance_to_root))

    @cached_property
    def parent_to_children_edge_index(self) -> NDArray[np.int64]:
        """Returns the edge index for parent-to-children connections in COO
        format."""
        edge_index = []
        for parent, children in enumerate(self.children_indices):
            for child in children:
                edge_index.append([parent, child])
        return np.array(edge_index, dtype=np.int64).T

    @cached_property
    def children_to_parent_edge_index(self) -> NDArray[np.int64]:
        """Returns the edge index for children-to-parent connections in COO
        format."""
        edge_index = self.parent_to_children_edge_index
        return np.vstack((edge_index[1], edge_index[0]))
