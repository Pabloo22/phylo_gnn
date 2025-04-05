from collections.abc import Iterable
from collections import defaultdict
from numpy.typing import NDArray
import numpy as np
from torch_geometric.typing import EdgeType  # type: ignore

from phylo_gnn.data.feature_extraction import VectorTree, EdgeFeatureExtractor


def get_distances(
    vector_tree: VectorTree,
    edge_index: NDArray[np.int64],
) -> NDArray[np.float32]:
    """Returns the distance between the nodes in the edge index.

    This function can be used to get edge attributes for the edges in the
    edge index.
    """
    distances = vector_tree.get_distances(edge_index[0], edge_index[1])
    return distances


def get_topological_distances(
    vector_tree: VectorTree,
    edge_index: NDArray[np.int64],
) -> NDArray[np.float32]:
    """Returns the topological distance between the nodes in the edge
    index."""
    distances = vector_tree.get_topological_distances(
        edge_index[0], edge_index[1]
    )
    return distances


def get_composite_edge_feature_extractor(
    extractors: Iterable[EdgeFeatureExtractor],
) -> EdgeFeatureExtractor:
    """Concatenates the edge features from the extractors."""

    def concat(
        vector_tree: VectorTree,
        edge_indices: dict[EdgeType, NDArray[np.int64]],
    ) -> dict[EdgeType, NDArray[np.float32]]:
        features: dict[EdgeType, list[NDArray[np.float32]]] = defaultdict(list)
        for extractor in extractors:
            edge_features = extractor(vector_tree, edge_indices)
            for edge_type, feature in edge_features.items():
                if feature.ndim == 1:
                    feature = feature[:, np.newaxis]
                features[edge_type].append(feature)
        features_combined = {
            edge_type: np.concatenate(features[edge_type], axis=1)
            for edge_type in features
        }
        return features_combined

    return concat
