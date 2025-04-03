import torch
from torch_geometric.typing import EdgeType, NodeType  # type: ignore
from torch_geometric.nn import (  # type: ignore
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from phylo_gnn.model.readouts import BaseReadout
from phylo_gnn.model import get_mlp


class SimpleReadout(BaseReadout):
    """Simple readout that aggregates node and edge features per graph,
    concatenates the results, and processes them with an MLP.
    """

    def __init__(
        self,
        node_input_dims: dict[NodeType, int],
        edge_input_dims: dict[EdgeType, int] | None = None,
        output_dim: int = 6,
        aggregator: str = "sum",
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(
            node_input_dims=node_input_dims,
            edge_input_dims=edge_input_dims,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            aggregator=aggregator,  # Save in hparams
            **kwargs,
        )

        valid_aggregators = ["sum", "max", "mean"]
        if aggregator not in valid_aggregators:
            raise ValueError(
                f"Invalid aggregator: {aggregator}. "
                f"Supported aggregators: {valid_aggregators}"
            )

        self.aggregator = aggregator

        mlp_input_dim = sum(node_input_dims.values())
        if edge_input_dims is not None:
            mlp_input_dim += sum(edge_input_dims.values())

        self.mlp = get_mlp(
            input_dim=mlp_input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
        )

    def _get_pooling_function(self):
        """Get the appropriate PyG pooling function based on aggregator."""
        pooling_functions = {
            "sum": global_add_pool,
            "max": global_max_pool,
            "mean": global_mean_pool,
        }
        if self.aggregator not in pooling_functions:
            raise ValueError(f"Unsupported aggregator: {self.aggregator}")
        return pooling_functions[self.aggregator]

    def forward(
        self,
        node_features_dict: dict[NodeType, torch.Tensor],
        edge_attributes_dict: dict[EdgeType, torch.Tensor] | None = None,
        batch_dict: dict[NodeType, torch.Tensor] | None = None,
        edge_batch_dict: dict[EdgeType, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward pass aggregating node and edge features.

        Args:
            node_features_dict: Dict mapping node types to feature tensors
                [num_nodes, feature_dim]
            edge_attributes_dict: Dict mapping edge types to attribute tensors
                [num_edges, feature_dim]
            batch_dict: Dict mapping node types to batch assignment tensors
                [num_nodes]
            edge_batch_dict: Dict mapping edge types to batch assignment
                tensors [num_edges]

        Returns:
            Tensor of output embeddings [batch_size, output_dim]
        """
        pool_func = self._get_pooling_function()
        all_features = []

        # Process node features
        for node_type in sorted(node_features_dict.keys()):
            features = node_features_dict[node_type]

            # Get batch indices for this node type
            if batch_dict is not None and node_type in batch_dict:
                batch = batch_dict[node_type]
            elif self.training and batch_dict is None:
                raise ValueError(
                    "Batch information is required during training. "
                    "Please provide batch_dict."
                )
            else:
                # If batch information is not available, assume single graph
                batch = torch.zeros(
                    features.size(0), dtype=torch.long, device=features.device
                )

            # Aggregate features per graph
            agg_features = pool_func(features, batch)
            all_features.append(agg_features)

        # Process edge attributes if provided
        if edge_attributes_dict is not None:
            for edge_type in sorted(edge_attributes_dict.keys()):
                attributes = edge_attributes_dict[edge_type]

                # Get batch indices for this edge type
                if (
                    edge_batch_dict is not None
                    and edge_type in edge_batch_dict
                ):
                    batch = edge_batch_dict[edge_type]
                elif self.training and edge_batch_dict is None:
                    raise ValueError(
                        "Batch information is required during training. "
                        "Please provide edge_batch_dict or set "
                        "edge_attributes_dict to None."
                    )
                else:
                    # If batch information is not available, assume single
                    # graph
                    batch = torch.zeros(
                        attributes.size(0),
                        dtype=torch.long,
                        device=attributes.device,
                    )

                # Aggregate features per graph
                agg_attributes = pool_func(attributes, batch)
                all_features.append(agg_attributes)

        # Handle case where no features were found
        if not all_features:
            raise ValueError("No node or edge features were provided")

        # Concatenate all aggregated features along feature dimension
        # Shape: [batch_size, total_feature_dim]
        batched_features = torch.cat(all_features, dim=1)

        # Apply MLP to get logits
        # Shape: [batch_size, output_dim]
        logits = self.mlp(batched_features)

        return logits
