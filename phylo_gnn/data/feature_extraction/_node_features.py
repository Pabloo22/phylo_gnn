from collections.abc import Mapping
import numpy as np
from numpy.typing import NDArray

from phylo_gnn.data.feature_extraction import (
    FeaturePipeline,
    NodeFeatureExtractor,
    VectorTree,
    NORMALIZATION_FUNCTIONS_MAPPING,
)


def get_node_feature_extractor(
    feature_pipelines: Mapping[str, list[FeaturePipeline]],
) -> NodeFeatureExtractor:
    """Creates a node feature extractor based on the provided feature
    pipelines.

    Args:
        feature_pipelines: A dictionary
            where keys are node types and values are lists of feature
            pipelines.

    Returns:
        NodeFeatureExtractor: A function that takes a VectorTree object and
            returns a dictionary of node features.
    """

    def node_feature_extractor(
        vector_tree: VectorTree,
    ) -> dict[str, NDArray[np.float32]]:
        """Extracts node features from a VectorTree object.

        Args:
            vector_tree (VectorTree): The input VectorTree object.

        Returns:
            dict[str, NDArray[np.float32]]: A dictionary mapping node types to
                their respective feature arrays.
        """
        node_features_dict = {}
        for node_type, pipelines in feature_pipelines.items():
            arrays = []
            for pipeline in pipelines:
                feature_array = get_node_feature_array(
                    vector_tree, pipeline.feature_name
                )
                if pipeline.normalization_fn_name is not None:
                    feature_array = NORMALIZATION_FUNCTIONS_MAPPING[
                        pipeline.normalization_fn_name
                    ](feature_array, vector_tree)
                arrays.append(feature_array)
            node_features_dict[node_type] = np.concatenate(arrays, axis=1)
        return node_features_dict

    return node_feature_extractor


def get_node_feature_array(
    vector_tree: VectorTree, feature_name: str
) -> NDArray[np.float32]:
    if feature_name == "position_in_level":
        return np.astype(vector_tree.set_positions_in_level(), np.float32)
    return getattr(vector_tree, feature_name)
