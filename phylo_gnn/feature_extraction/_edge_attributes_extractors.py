from collections.abc import Iterable
from collections import defaultdict
from numpy.typing import NDArray
import numpy as np

from phylo_gnn.feature_extraction import VectorTree, EdgeFeatureExtractor


def get_distances(
    vector_tree: VectorTree,
    edge_indices: dict[tuple[str, str, str], NDArray[np.int64]],
) -> dict[tuple[str, str, str], NDArray[np.float32]]:
    """Returns the distance between the nodes in the edge index.

    This function can be used to get edge attributes for the edges in the
    edge index.
    """
    result = {}
    for edge_type, edge_index in edge_indices.items():
        distances = vector_tree.get_distances(edge_index[0], edge_index[1])
        result[edge_type] = distances
    return result


def get_topological_distances(
    vector_tree: VectorTree,
    edge_indices: dict[tuple[str, str, str], NDArray[np.int64]],
) -> dict[tuple[str, str, str], NDArray[np.float32]]:
    """Returns the topological distance between the nodes in the edge index."""
    result = {}
    for edge_type, edge_index in edge_indices.items():
        distances = vector_tree.get_topological_distances(
            edge_index[0], edge_index[1]
        )
        result[edge_type] = distances
    return result


def get_composite_edge_feature_extractor(
    extractors: Iterable[EdgeFeatureExtractor],
) -> EdgeFeatureExtractor:
    """Concatenates the edge features from the extractors."""

    def concat(
        vector_tree: VectorTree,
        edge_indices: dict[tuple[str, str, str], NDArray[np.int64]],
    ) -> dict[tuple[str, str, str], NDArray[np.float32]]:
        features: dict[tuple[str, str, str], list[NDArray[np.float32]]] = (
            defaultdict(list)
        )
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
