from collections.abc import Callable
import enum

import torch
from torch_geometric.data import HeteroData  # type: ignore
import numpy as np
from numpy.typing import NDArray

from phylo_gnn.feature_extraction import VectorTree


EdgeIndices = dict[tuple[str, str, str], NDArray[np.int64]]
EncodingFunction = Callable[[str, NDArray], HeteroData]
NodeFeatureExtractor = Callable[[VectorTree], dict[str, NDArray[np.float32]]]
EdgeIndicesExtractor = Callable[[VectorTree], EdgeIndices]
EdgeFeatureExtractor = Callable[
    [VectorTree, EdgeIndices], dict[tuple[str, str, str], NDArray[np.float32]]
]
TargetProcessor = (
    Callable[[NDArray], torch.Tensor]
    | Callable[[NDArray], dict[str, torch.Tensor]]
    | Callable[[NDArray], dict[tuple[str, str, str], torch.Tensor]]
)


class NodeType(str, enum.Enum):
    NODE = "node"
    INTERNAL = "internal"
    LEAF = "leaf"


class EdgeType(str, enum.Enum):
    HAS_PARENT = "has_parent"
    HAS_CHILD = "has_child"
    HAS_SIBLING = "has_sibling"
    HAS_ANCESTOR = "has_ancestor"
    HAS_DESCENDANT = "has_descendant"
    HAS_ADJACENT_LEVEL_POSITION = "has_adjacent_level_position"
