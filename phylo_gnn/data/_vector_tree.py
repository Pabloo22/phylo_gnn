from __future__ import annotations

from functools import cached_property

import numpy as np
from numpy.typing import NDArray
import ete3  # type: ignore[import-untyped]

from phylo_gnn.ete3_utils import ID_ATTR, set_node_ids


class VectorTree:

    def __init__(
        self,
        branch_lengths: NDArray[np.float32],
        children_indices: list[list[int]],
    ):
        self.branch_lengths = branch_lengths
        self.children_indices = children_indices

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
    def dist_to_root(self) -> NDArray[np.float32]:
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
    def dist_to_leaves(self) -> NDArray[np.float32]:
        distances = np.zeros_like(self.branch_lengths)
        for _, level_nodes in self.iter_by_level(reverse=True):
            for node in level_nodes:
                children = self.children_indices[node]
                if children:
                    # Find maximum distance to present through any child
                    child_distances = [
                        float(distances[child] + self.branch_lengths[child])
                        for child in children
                    ]
                    distances[node] = max(child_distances)

        return distances

    @cached_property
    def leaves_indices(self) -> list[int]:
        """Returns the indices of the leaves in the tree."""
        return [
            node
            for node in range(self.num_nodes)
            if not self.children_indices[node]
        ]

    @property
    def num_leaves(self) -> int:
        """Returns the number of leaves in the tree."""
        return len(self.leaves_indices)

    @cached_property
    def inverse_levels(self) -> NDArray[np.int64]:
        """Returns the inverse levels of the tree.

        The inverse level of a node is defined as the maximum level minus the
        level of that node.
        """
        return self.max_level - self.levels
