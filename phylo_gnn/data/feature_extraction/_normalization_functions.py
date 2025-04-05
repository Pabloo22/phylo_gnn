import numpy as np
from numpy.typing import NDArray

from phylo_gnn.data.feature_extraction import NormalizationFunction, VectorTree


def div_by_max_level(
    x: NDArray[np.float32],
    vector_tree: VectorTree,
) -> NDArray[np.float32]:
    """Divides the values by the maximum level of the tree."""
    max_level = vector_tree.max_level
    if max_level == 0:
        return x
    return x / float(max_level)


def div_by_num_nodes_in_level(
    x: NDArray[np.float32],
    vector_tree: VectorTree,
) -> NDArray[np.float32]:
    """Divides the values by the number of nodes in the level.

    Args:
        x:
            The values to be divided. It must be a 1D
            array with the same length as the number of nodes in the tree.
        vector_tree:
            The vector tree object.

    Returns:
        The divided values.
    """
    for _, level_nodes in vector_tree.iter_by_level():
        num_nodes_in_level = len(level_nodes)
        x[level_nodes] = x[level_nodes] / num_nodes_in_level
    return x


NORMALIZATION_FUNCTIONS_MAPPING: dict[str, NormalizationFunction] = {
    "minmax": lambda x, _: (x - x.min()) / (x.max() - x.min()),
    "log1p": lambda x, _: np.log1p(x),
    "zscore": lambda x, _: (x - x.mean()) / x.std(),
    "div_by_max_level": div_by_max_level,
    "div_by_num_nodes_in_level": div_by_num_nodes_in_level,
}
