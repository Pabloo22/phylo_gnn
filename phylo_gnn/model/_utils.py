from typing import Callable, Optional, List
from torch_geometric.data import HeteroData  # type: ignore
from torch_geometric.typing import EdgeType  # type: ignore
import torch
from torch import nn


STR_TO_ACTIVATION = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


def get_node_features_dict(hetero_data: HeteroData) -> dict[str, torch.Tensor]:
    return {
        node_type: hetero_data[node_type].x
        for node_type in hetero_data.node_types
    }


def get_edge_attributes_dict(
    hetero_data: HeteroData, attr_name: str = "edge_attr"
) -> dict[EdgeType, torch.Tensor]:
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


def get_edge_indices_dict(
    hetero_data: HeteroData,
) -> dict[EdgeType, torch.Tensor]:
    """Extracts edge indices from a HeteroData object into a dictionary.

    Args:
        hetero_data: A torch_geometric.data.HeteroData object

    Returns:
        dict: A dictionary mapping edge types to their index tensors
    """
    return {
        edge_type: hetero_data[edge_type].edge_index
        for edge_type in hetero_data.edge_types
    }


def get_batch_dict(
    hetero_data: HeteroData,
) -> tuple[dict[str, torch.Tensor], dict[EdgeType, torch.Tensor]]:
    """Extract batch assignment information from HeteroData object."""
    batch_dict = {}
    edge_batch_dict = {}

    # Extract node batch information
    for node_type in hetero_data.node_types:
        if hasattr(hetero_data[node_type], "batch"):
            batch_dict[node_type] = hetero_data[node_type].batch

    # Extract edge batch information
    for edge_type in hetero_data.edge_types:
        if hasattr(hetero_data[edge_type], "batch"):
            edge_batch_dict[edge_type] = hetero_data[edge_type].batch

    return batch_dict, edge_batch_dict


def get_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.1,
    activation: Callable[[], nn.Module] | str = "relu",
    output_activation: Optional[Callable[[], nn.Module]] = None,
) -> nn.Sequential:
    """Create a Multi-Layer Perceptron with customizable architecture.

    Args:
        input_dim:
            Input dimension
        output_dim:
            Output dimension
        hidden_dims:
            List of hidden layer dimensions (None for single-layer network)
        dropout:
            Dropout probability (0 for no dropout)
        activation:
            Activation function to use between layers
        output_activation:
            Optional activation function for the output layer

    Returns:
        nn.Sequential: The constructed MLP
    """
    if isinstance(activation, str):
        activation = STR_TO_ACTIVATION.get(activation.lower(), "")
    if isinstance(activation, str):
        raise ValueError(
            f"Unsupported activation function: {activation}. "
            "Supported functions are: "
            f"{', '.join(STR_TO_ACTIVATION.keys())}"
        )

    layers: list[nn.Module] = []

    # If no hidden dimensions provided, create a single-layer network
    if hidden_dims is None or len(hidden_dims) == 0:
        layers.append(nn.Linear(input_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        return nn.Sequential(*layers)

    # Build hidden layers
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim

    # Add output layer
    layers.append(nn.Linear(current_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())

    return nn.Sequential(*layers)
