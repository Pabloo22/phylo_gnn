from ete3 import Tree  # type: ignore[import-untyped]

ID_ATTR = "id"


def set_node_ids(
    tree: Tree,
    traverse_strategy: str = "levelorder",
    count_leaves: bool = True,
) -> None:
    """Sets the id attribute for each node in the tree.

    Args:
        tree: An ``ete3`` tree object.
    """
    for i, node in enumerate(tree.traverse(traverse_strategy)):
        if node.is_leaf() and not count_leaves:
            continue
        node.add_feature(ID_ATTR, i)
