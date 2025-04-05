from collections.abc import Callable
import enum
from typing import NamedTuple

import torch
from torch_geometric.data import HeteroData  # type: ignore
from torch_geometric.typing import EdgeType  # type: ignore
import numpy as np
from numpy.typing import NDArray

from phylo_gnn.data.feature_extraction import VectorTree


EdgeIndices = dict[EdgeType, NDArray[np.int64]]
EncodingFunction = Callable[[str, NDArray], HeteroData]
NodeFeatureExtractor = Callable[[VectorTree], dict[str, NDArray[np.float32]]]
EdgeIndicesExtractor = Callable[[VectorTree], EdgeIndices]
EdgeFeaturesExtractor = Callable[
    [VectorTree, EdgeIndices], dict[EdgeType, NDArray[np.float32]]
]
TargetProcessor = (
    Callable[[NDArray], torch.Tensor]
    | Callable[[NDArray], dict[str, torch.Tensor]]
    | Callable[[NDArray], dict[EdgeType, torch.Tensor]]
)
NormalizationFunction = Callable[[NDArray, VectorTree], NDArray[np.float32]]
EdgeIndexExtractor = Callable[[VectorTree], NDArray[np.int64]]
EdgeFeatureExtractor = Callable[
    [VectorTree, NDArray[np.int64]], NDArray[np.float32]
]


class NodeNames(str, enum.Enum):
    NODE = "node"
    INTERNAL = "internal"
    LEAF = "leaf"


class EdgeNames(str, enum.Enum):
    HAS_PARENT = "has_parent"
    HAS_CHILD = "has_child"
    HAS_SIBLING = "has_sibling"
    HAS_ANCESTOR = "has_ancestor"
    HAS_DESCENDANT = "has_descendant"
    HAS_ADJACENT_LEVEL_POSITION = "has_adjacent_level_position"


class FeaturePipeline(NamedTuple):
    feature_name: str
    normalization_fn_name: str | None
