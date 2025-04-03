from torch_geometric.data import HeteroData  # type: ignore
import torch


def get_node_features_dict(hetero_data: HeteroData) -> dict[str, torch.Tensor]:
    return {
        node_type: hetero_data[node_type].x
        for node_type in hetero_data.node_types
    }


def get_edge_attributes_dict(
    hetero_data: HeteroData, attr_name: str = "edge_attr"
):
    """Extracts edge attributes from a HeteroData object into a dictionary.

    Args:
        hetero_data: A torch_geometric.data.HeteroData object
        attr_name: The attribute name to extract (default: 'edge_attr')

    Returns:
        dict: A dictionary mapping edge types to their attribute tensors
    """
    return {
        edge_type: hetero_data[edge_type][attr_name]
        for edge_type in hetero_data.edge_types
        if attr_name in hetero_data[edge_type]
    }
