from collections.abc import Iterable
from numpy.typing import NDArray
import numpy as np
from torch_geometric.typing import EdgeType  # type: ignore

from phylo_gnn.data.feature_extraction import (
    VectorTree,
    EdgeFeatureExtractor,
    NORMALIZATION_FUNCTIONS_MAPPING,
    FeaturePipeline,
    EdgeFeaturesExtractor,
)


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


EDGE_FEATURE_EXTRACTORS_MAPPING: dict[str, EdgeFeatureExtractor] = {
    "distances": get_distances,
    "topological_distances": get_topological_distances,
}


def get_edge_feature_extractor(
    feature_pipelines: dict[EdgeType, Iterable[FeaturePipeline]],
) -> EdgeFeaturesExtractor:
    """Creates an edge feature extractor based on the provided feature
    pipelines.

    Args:
        feature_pipelines: A dictionary where keys are edge types and values
            are iterable of feature pipelines.

    Returns:
        A function that takes a VectorTree object and
            returns a dictionary of edge features.
    """

    def edge_feature_extractor(
        vector_tree: VectorTree,
        edge_indices_dict: dict[EdgeType, NDArray[np.int64]],
    ) -> dict[str, NDArray[np.float32]]:
        """Extracts edge features from a VectorTree object.

        Args:
            vector_tree (VectorTree): The input VectorTree object.
            edge_indices_dict (dict[EdgeType, NDArray[np.int64]]): A dictionary
                mapping edge types to their respective edge indices.

        Returns:
            dict[str, NDArray[np.float32]]: A dictionary mapping edge types to
                their respective feature arrays.
        """
        edge_features_dict = {}
        for edge_type, pipelines in feature_pipelines.items():
            arrays = []
            for pipeline in pipelines:
                feature_array = EDGE_FEATURE_EXTRACTORS_MAPPING[
                    pipeline.feature_name
                ](vector_tree, edge_indices_dict[edge_type])
                if pipeline.normalization_fn_name is not None:
                    feature_array = NORMALIZATION_FUNCTIONS_MAPPING[
                        pipeline.normalization_fn_name
                    ](feature_array, vector_tree)
                arrays.append(feature_array)
            edge_features_dict[edge_type] = np.concatenate(arrays, axis=1)
        return edge_features_dict

    return edge_feature_extractor
