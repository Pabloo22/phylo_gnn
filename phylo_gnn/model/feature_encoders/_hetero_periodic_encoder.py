import torch
from torch import nn
from torch_geometric.typing import EdgeType, NodeType  # type: ignore

from phylo_gnn.model.feature_encoders import BaseEncoder, MultiPeriodicEncoder


class HeteroPeriodicEncoder(BaseEncoder):
    """Heterogeneous Periodic Encoder for graph node and edge features.

    Uses a different MultiPeriodicEncoder for each node and edge type in the
    heterogeneous graph.

    Args:
        node_input_dims: Dictionary mapping node types to input dimensions
        node_output_dims: Dictionary mapping node types to output dimensions,
            or a single integer
        edge_input_dims: Dictionary mapping edge types to input dimensions
            (optional)
        edge_output_dims: Dictionary mapping edge types to output dimensions,
            or a single integer (optional)
        sigma: Standard deviation for initializing frequencies
        concat: Whether to concatenate or sum the encoded dimensions
    """

    def __init__(
        self,
        node_input_dims: dict[NodeType, int],
        node_output_dims: dict[NodeType, int] | int,
        edge_input_dims: dict[EdgeType, int] | None = None,
        edge_output_dims: dict[EdgeType, int] | int | None = None,
        sigma: float = 1.0,
        concat: bool = True,
        **kwargs,
    ):
        super().__init__(
            node_input_dims=node_input_dims,
            node_output_dims=node_output_dims,
            edge_input_dims=edge_input_dims,
            edge_output_dims=edge_output_dims,
            sigma=sigma,
            concat=concat,
            **kwargs,
        )

        self.sigma = sigma
        self.concat = concat

        # Process node_output_dims to ensure it's a dictionary
        if isinstance(node_output_dims, int):
            self.node_output_dims_dict = {
                node_type: node_output_dims for node_type in node_input_dims
            }
        else:
            self.node_output_dims_dict = node_output_dims

        # Create a MultiPeriodicEncoder for each node type
        self.node_encoders = nn.ModuleDict(
            {
                node_type: MultiPeriodicEncoder(
                    input_size=input_dim,
                    output_size=self.node_output_dims_dict[node_type],
                    sigma=sigma,
                    concat=concat,
                )
                for node_type, input_dim in node_input_dims.items()
            }
        )

        # Process edge_output_dims if edge encoding is requested
        self.edge_encoders = nn.ModuleDict()
        self.edge_type_to_key = {}

        if edge_input_dims is not None and edge_output_dims is not None:
            if isinstance(edge_output_dims, int):
                edge_output_dims_dict = {
                    edge_type: edge_output_dims
                    for edge_type in edge_input_dims
                }
            else:
                edge_output_dims_dict = edge_output_dims

            # Create a MultiPeriodicEncoder for each edge type
            for edge_type, input_dim in edge_input_dims.items():
                # Convert tuple to string key for nn.ModuleDict
                src, rel, dst = edge_type
                key = f"{src}___{rel}___{dst}"
                self.edge_type_to_key[edge_type] = key

                self.edge_encoders[key] = MultiPeriodicEncoder(
                    input_size=input_dim,
                    output_size=edge_output_dims_dict[edge_type],
                    sigma=sigma,
                    concat=concat,
                )

    def forward(  # type: ignore[override]
        self,
        node_features_dict: dict[NodeType, torch.Tensor],
        edge_attributes_dict: (
            dict[EdgeType, torch.Tensor] | None
        ) = None,
    ) -> tuple[
        dict[NodeType, torch.Tensor],
        dict[EdgeType, torch.Tensor] | None,
    ]:
        """Forward pass for heterogeneous graph node and edge features.

        Args:
            node_features_dict:
                Dictionary mapping node types to feature tensors
            edge_attributes_dict:
                Dictionary mapping edge types to attribute tensors

        Returns:
            Tuple of (encoded node features dict, encoded edge attributes dict)
        """
        # Validate input node types
        for node_type in node_features_dict:
            if node_type not in self.node_encoders:
                raise ValueError(
                    f"Node type '{node_type}' not found in encoder. "
                    f"Available node types: {list(self.node_encoders.keys())}"
                )

        # Encode node features
        encoded_node_features = {
            node_type: self.node_encoders[node_type](features)
            for node_type, features in node_features_dict.items()
        }

        # Encode edge attributes if available
        encoded_edge_attributes = None
        if edge_attributes_dict is not None and len(self.edge_encoders) > 0:
            encoded_edge_attributes = {}
            for edge_type, attrs in edge_attributes_dict.items():
                # Validate edge type
                if edge_type not in self.edge_type_to_key:
                    raise ValueError(
                        f"Edge type '{edge_type}' not found in encoder. "
                        f"Available: {list(self.edge_type_to_key.keys())}"
                    )

                # Get the string key for this edge type
                key = self.edge_type_to_key[edge_type]
                encoded_edge_attributes[edge_type] = self.edge_encoders[key](
                    attrs
                )

        return encoded_node_features, encoded_edge_attributes
