from collections.abc import Iterable, Mapping
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


def get_level2node_position_in_level(
    vector_tree: VectorTree,
    edge_index: NDArray[np.int64],
) -> NDArray[np.float32]:
    """Returns the position in level for the nodes in the edge index."""
    levels, node_indices = edge_index
    if len(np.unique(levels)) > len(np.unique(node_indices)):
        raise ValueError(
            "First row must represent levels and second row "
            "must represent node indices. "
            f"First row: {levels}\n"
            f"Second row: {node_indices}"
        )
    return vector_tree.position_in_level[node_indices]


def get_node2level_position_in_level(
    vector_tree: VectorTree,
    edge_index: NDArray[np.int64],
) -> NDArray[np.float32]:
    node_indices, levels = edge_index
    if len(np.unique(levels)) > len(np.unique(node_indices)):
        raise ValueError(
            "Second row must represent node indices and first "
            "row must represent levels."
        )
    return vector_tree.position_in_level[node_indices]


def get_distance_between_levels(
    vector_tree: VectorTree,
    edge_index: NDArray[np.int64],
) -> NDArray[np.float32]:
    """Returns the distance between the levels in the edge index.

    This function can be used to get edge attributes for the edges in the
    edge index.
    """
    avg_distance_to_root_by_level = vector_tree.avg_attr_by_level(
        "distance_to_root"
    )
    distance_between_levels_matrix = np.abs(
        avg_distance_to_root_by_level[edge_index[0]]
        - avg_distance_to_root_by_level[edge_index[1]]
    )
    return distance_between_levels_matrix.astype(np.float32)


EDGE_FEATURE_EXTRACTORS_MAPPING: dict[str, EdgeFeatureExtractor] = {
    "distances": get_distances,
    "topological_distances": get_topological_distances,
    "level2node_position_in_level": get_level2node_position_in_level,
    "node2level_position_in_level": get_node2level_position_in_level,
    "distance_between_levels": get_distance_between_levels,
}


def get_edge_feature_extractor(
    feature_pipelines: Mapping[EdgeType, Iterable[FeaturePipeline]],
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
        edge_indices_dict: Mapping[EdgeType, NDArray[np.int64]],
    ) -> dict[str, NDArray[np.float32]]:
        """Extracts edge features from a VectorTree object.

        Args:
            vector_tree: The input VectorTree object.
            edge_indices_dict: A dictionary
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
                # Ensure feature_array is 2D before appending
                if feature_array.ndim == 1:
                    feature_array = feature_array.reshape(-1, 1)
                arrays.append(feature_array)
            edge_features_dict[edge_type] = np.concatenate(arrays, axis=1)
        return edge_features_dict

    return edge_feature_extractor
