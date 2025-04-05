import torch
from torch_geometric.data import HeteroData  # type: ignore
import numpy as np
from numpy.typing import NDArray

from phylo_gnn.data.feature_extraction import (
    VectorTree,
    TargetProcessor,
    NodeFeatureExtractor,
    get_graph_classification_target,
    EdgeFeatureExtractor,
    EdgeIndicesExtractor,
    EncodingFunction,
)


def get_encoding_function(
    node_feature_extractor: NodeFeatureExtractor,
    edge_indices_extractor: EdgeIndicesExtractor | None = None,
    target_processor: TargetProcessor = get_graph_classification_target,
    edge_attribute_extractor: EdgeFeatureExtractor | None = None,
) -> EncodingFunction:

    def encoding_function(newick: str, target_row: NDArray) -> HeteroData:
        tree_vector = VectorTree.from_newick(newick)
        hetero_data = HeteroData()

        node_features_dict = node_feature_extractor(tree_vector)
        _add_node_features(hetero_data, node_features_dict)

        edge_indices_dict = (
            edge_indices_extractor(tree_vector)
            if edge_indices_extractor is not None
            else {}
        )
        _add_edge_indices(hetero_data, edge_indices_dict)

        edge_attributes_dict = (
            edge_attribute_extractor(tree_vector, edge_indices_dict)
            if edge_attribute_extractor is not None
            else {}
        )
        _add_edge_attributes(hetero_data, edge_attributes_dict)

        target = target_processor(target_row)
        _add_target(hetero_data, target)

        return hetero_data

    return encoding_function


def _add_node_features(
    hetero_data: HeteroData,
    node_features_dict: dict[str, NDArray[np.float32]],
):
    """Adds node features to the hetero_data object."""
    for node_type, node_features in node_features_dict.items():
        hetero_data[node_type].x = node_features


def _add_edge_indices(
    hetero_data: HeteroData,
    edge_indices_dict: dict[tuple[str, str, str], NDArray[np.int64]],
):
    """Adds edge indices to the hetero_data object."""
    for edge_type, edge_indices in edge_indices_dict.items():
        hetero_data[edge_type].edge_index = edge_indices


def _add_edge_attributes(
    hetero_data: HeteroData,
    edge_attributes_dict: dict[tuple[str, str, str], NDArray[np.float32]],
):
    """Adds edge attributes to the hetero_data object."""
    for edge_type, edge_attributes in edge_attributes_dict.items():
        assert (
            hetero_data[edge_type].edge_index.shape[1]
            == edge_attributes.shape[0]
        ), (
            "Edge attributes and edge indices must have the same number of "
            f"edges. Got {edge_attributes.shape[0]} and "
            f"{hetero_data[edge_type].edge_index.shape[1]} respectively."
        )
        hetero_data[edge_type].edge_attr = edge_attributes


def _add_target(
    hetero_data: HeteroData,
    target: (
        torch.Tensor
        | dict[str, torch.Tensor]
        | dict[tuple[str, str, str], torch.Tensor]
    ),
):
    """Adds target to the hetero_data object."""
    if isinstance(target, dict):
        for key, value in target.items():
            hetero_data[key].y = value
    else:
        hetero_data.y = target
