from torch import nn
import torch
from torch_geometric.nn import HeteroConv, GATv2Conv  # type: ignore
from torch_geometric.typing import EdgeType  # type: ignore

from phylo_gnn.model.message_passing import BaseMessagePassing
from phylo_gnn.model import STR_TO_ACTIVATION, get_mlp


class HGATv2MessagePassing(BaseMessagePassing):
    def __init__(
        self,
        node_input_dims: dict[str, int],
        node_output_dims: dict[str, int] | int,
        edge_types: list[EdgeType] | None = None,
        edge_input_dims: dict[EdgeType, int] | None = None,
        edge_output_dims: dict[EdgeType, int] | int | None = None,
        num_layers: int = 3,
        heads: int = 4,
        concat_heads: bool = True,
        dropout: float = 0.1,
        aggregation: str = "sum",
        activation: str = "elu",
        add_self_loops: bool = True,
        layer_norm: bool = True,
        negative_slope: float = 0.2,
        mlp_num_layers: int = 1,
        mlp_dropout: float | None = None,
        mlp_activation: str = "elu",
        **kwargs,
    ):
        if aggregation == "cat":
            raise NotImplementedError(
                "Concatenation aggregation is not implemented yet."
            )
        final_node_output_dims = self._get_final_node_output_dims(
            node_input_dims=node_input_dims, node_output_dims=node_output_dims
        )
        final_edge_output_dims = self._get_final_edge_output_dims(
            edge_input_dims=edge_input_dims,
            edge_output_dims=edge_output_dims,
        )
        if mlp_dropout is None:
            mlp_dropout = dropout
        super().__init__(
            node_input_dims=node_input_dims,
            node_output_dims=final_node_output_dims,
            edge_types=edge_types,
            edge_input_dims=edge_input_dims,
            edge_output_dims=final_edge_output_dims,
            heads=heads,
            concat_heads=concat_heads,
            dropout=dropout,
            aggregation=aggregation,
            add_self_loops=add_self_loops,
            activation=activation,
            layer_norm=layer_norm,
            negative_slope=negative_slope,
            mlp_num_layers=mlp_num_layers,
            mlp_dropout=mlp_dropout,
            mlp_activation=activation,
            **kwargs,
        )
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.aggregation = aggregation
        self.concat_heads = concat_heads
        self.heads = heads
        self.layer_norm = layer_norm
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        self.mlp_num_layers = mlp_num_layers
        self.mlp_dropout_p = mlp_dropout
        self.mlp_activation = mlp_activation
        self.kwargs = kwargs

        self.conv_layers = nn.ModuleList()
        self.normalization_layers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.activation = STR_TO_ACTIVATION[activation]()
        self.dropout_layer = nn.Dropout(p=dropout)

        current_node_dims = self.node_input_dims
        for _ in range(self.num_layers):
            assert not isinstance(self.node_output_dims, int)
            next_layer_input_dims = self._get_next_layer_input_dims()
            convs_dict = self._get_convs_dict(current_node_dims)
            hetero_conv = HeteroConv(convs_dict, aggr=self.aggregation)
            self.conv_layers.append(hetero_conv)
            current_node_dims = next_layer_input_dims

            mlps = self._get_mlps(input_node_dims=current_node_dims)
            self.mlps.append(mlps)

            current_node_dims = self.node_output_dims

            if self.layer_norm:
                norm_dict = nn.ModuleDict()
                for node_type in self.node_types:
                    norm_dim = current_node_dims[node_type]
                    norm_dict[node_type] = nn.LayerNorm(norm_dim)
                self.normalization_layers.append(norm_dict)

    def _get_mlps(
        self,
        input_node_dims: dict[str, int],
    ) -> nn.Module:
        """Get the MLP for the node types."""
        assert not isinstance(self.node_output_dims, int)
        mlp_dict = nn.ModuleDict()
        for node_type, input_dim in input_node_dims.items():
            output_dim = self.node_output_dims[node_type]
            mlp_dict[node_type] = get_mlp(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=[output_dim] * self.mlp_num_layers,
                dropout=self.mlp_dropout_p,
            )
        return mlp_dict

    def _get_next_layer_input_dims(self) -> dict[str, int]:
        """Get the input dimensions for the next layer."""
        assert not isinstance(self.node_output_dims, int)
        next_layer_input_dims: dict[str, int] = {}
        for node_type, target_dim in self.node_output_dims.items():
            if self.concat_heads:
                next_layer_input_dims[node_type] = target_dim * self.heads
            else:
                next_layer_input_dims[node_type] = target_dim
        return next_layer_input_dims

    def _get_convs_dict(
        self, current_node_dims: dict[str, int]
    ) -> dict[str, GATv2Conv]:
        # Create GATv2Conv for each edge type
        assert not isinstance(self.node_output_dims, int)
        convs_dict: dict[EdgeType, GATv2Conv] = {}
        for edge_type in self.edge_types:
            src_type, _, dst_type = edge_type
            in_channels_src = current_node_dims[src_type]
            in_channels_dst = current_node_dims[dst_type]
            out_channels_dst_per_head = self.node_output_dims[dst_type]

            # Get edge dimension for this edge type
            edge_dim = None
            if self.edge_input_dims:
                edge_dim = self.edge_input_dims.get(edge_type, None)
            convs_dict[edge_type] = GATv2Conv(
                in_channels=(in_channels_src, in_channels_dst),
                out_channels=out_channels_dst_per_head,  # Dim per head
                heads=self.heads,
                concat=self.concat_heads,
                negative_slope=self.negative_slope,
                dropout=self.dropout_p,
                add_self_loops=self.add_self_loops,
                edge_dim=edge_dim,
                **self.kwargs,
            )
        return convs_dict

    @property
    def node_types(self) -> list[str]:
        """Get the node types in the model."""
        return list(self.node_input_dims.keys())

    @staticmethod
    def _get_final_node_output_dims(
        node_input_dims: dict[str, int],
        node_output_dims: dict[str, int] | int,
    ) -> dict[str, int]:
        if isinstance(node_output_dims, int):
            return {
                node_type: node_output_dims for node_type in node_input_dims
            }
        return node_output_dims

    def _get_final_edge_output_dims(
        self,
        edge_input_dims: dict[EdgeType, int] | None,
        edge_output_dims: dict[EdgeType, int] | None | int,
    ) -> dict[EdgeType, int] | None:
        if edge_input_dims is not None and isinstance(edge_output_dims, int):
            return {
                edge_type: edge_output_dims
                for edge_type in edge_input_dims.keys()
            }
        if isinstance(edge_output_dims, int):
            raise ValueError(
                "edge_output_dims must be a dict when edge_input_dims is "
                "provided"
            )
        return edge_output_dims

    def forward(  # type: ignore[override]
        self,
        node_features_dict: dict[str, torch.Tensor],
        edge_indices_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_attributes_dict: (
            dict[tuple[str, str, str], torch.Tensor] | None
        ) = None,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[tuple[str, str, str], torch.Tensor] | None,
    ]:
        pass_kwargs = (
            {"edge_attr_dict": edge_attributes_dict}
            if edge_attributes_dict is not None
            else {}
        )

        for i in range(self.num_layers):
            residual = {
                node_type: features.clone()
                for node_type, features in node_features_dict.items()
            }

            # Convolution
            node_features_dict = self.conv_layers[i](
                node_features_dict, edge_indices_dict, **pass_kwargs
            )

            # MLP
            node_features_dict = {
                node_type: self.mlps[i][node_type](  # type: ignore[index]
                    features
                )
                for node_type, features in node_features_dict.items()
            }

            # Normalization
            if self.layer_norm:
                node_features_dict = {
                    node_type: self.normalization_layers[i][
                        node_type  # type: ignore[index]
                    ](features)
                    for node_type, features in node_features_dict.items()
                }

            # Dropout
            if i < self.num_layers - 1:
                node_features_dict = {
                    node_type: self.dropout_layer(features)
                    for node_type, features in node_features_dict.items()
                }

            # Activation
            node_features_dict = {
                node_type: self.activation(features)
                for node_type, features in node_features_dict.items()
            }

            # Add residual connection if dimensions match
            node_features_dict = {
                node_type: (
                    (features + residual[node_type])
                    if residual[node_type].shape[1] == features.shape[1]
                    else features
                )
                for node_type, features in node_features_dict.items()
            }

        # Check correct output dimensions
        assert not isinstance(self.node_output_dims, int)
        for node_type, features in node_features_dict.items():
            assert features.shape[1] == self.node_output_dims[node_type], (
                f"Output dimension mismatch for {node_type}: "
                f"{features.shape[1]} != {self.node_output_dims[node_type]}"
            )

        return node_features_dict, edge_attributes_dict
