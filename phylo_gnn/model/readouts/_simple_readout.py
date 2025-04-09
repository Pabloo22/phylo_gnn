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
        aggregator: str = "all",
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_edge_features: bool = True,
        **kwargs,
    ):
        mlp_input_dim = sum(node_input_dims.values())
        if edge_input_dims is not None and use_edge_features:
            mlp_input_dim += sum(edge_input_dims.values())

        valid_aggregators = ["sum", "max", "mean", "all"]
        if aggregator not in valid_aggregators:
            raise ValueError(
                f"Invalid aggregator: {aggregator}. "
                f"Supported aggregators: {valid_aggregators}"
            )

        self.aggregator = aggregator

        if aggregator == "all":
            mlp_input_dim *= 3
        if hidden_dims is None:
            hidden_dims = self._default_hidden_dims(
                input_dim=mlp_input_dim, output_dim=output_dim
            )
        super().__init__(
            node_input_dims=node_input_dims,
            edge_input_dims=edge_input_dims,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            aggregator=aggregator,  # Save in hparams
            mlp_input_dim=mlp_input_dim,
            use_edge_features=use_edge_features,
            **kwargs,
        )
        self.use_edge_features = use_edge_features
        self.mlp = get_mlp(
            input_dim=mlp_input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
        )

    def _default_hidden_dims(
        self, input_dim: int, output_dim: int = 6
    ) -> list[int]:
        hidden_dims = []
        current_dim = input_dim // 2
        min_dim = max(output_dim * 3, 8)
        while current_dim >= min_dim:
            hidden_dims.append(current_dim)
            current_dim //= 2
        return hidden_dims

    @staticmethod
    def _aggregate_and_concat_all(
        x: torch.Tensor, batch: torch.Tensor | None, size: int | None = None
    ) -> torch.Tensor:
        """Check if all aggregators are used and concatenated."""
        sum_agg = global_add_pool(x, batch, size)
        mean_agg = global_mean_pool(x, batch, size)
        max_agg = global_max_pool(x, batch, size)
        return torch.cat([sum_agg, mean_agg, max_agg], dim=1)

    def _get_pooling_function(self):
        """Get the appropriate PyG pooling function based on aggregator."""
        pooling_functions = {
            "sum": global_add_pool,
            "max": global_max_pool,
            "mean": global_mean_pool,
            "all": self._aggregate_and_concat_all,
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
        """Forward pass aggregating node and edge features."""
        pool_func = self._get_pooling_function()
        all_features = []

        # Process node features
        for node_type in sorted(node_features_dict.keys()):
            features = node_features_dict[node_type]

            if batch_dict is None or node_type not in batch_dict:
                raise ValueError(
                    f"Batch information is missing for node type: {node_type}"
                )
            batch = batch_dict[node_type]

            # Aggregate features per graph
            agg_features = pool_func(features, batch)
            all_features.append(agg_features)

        # Process edge attributes if provided
        if edge_attributes_dict is not None and self.use_edge_features:
            for edge_type in sorted(edge_attributes_dict.keys()):
                attributes = edge_attributes_dict[edge_type]

                if edge_batch_dict is None or edge_type not in edge_batch_dict:
                    raise ValueError(
                        f"Batch information is missing for: {edge_type}"
                    )
                batch = edge_batch_dict[edge_type]

                # Aggregate features per graph
                agg_attributes = pool_func(attributes, batch)
                all_features.append(agg_attributes)

        if not all_features:
            raise ValueError("No node or edge features were provided")

        batched_features = torch.cat(all_features, dim=1)
        logits = self.mlp(batched_features)
        return logits
