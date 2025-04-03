import torch
from torch import nn
from torch_geometric.nn import (  # type: ignore
    HeteroConv,
    GCNConv,
    GATConv,
    SAGEConv,
    GINConv,
)
from torch_geometric.typing import EdgeType  # type: ignore

from phylo_gnn.model.message_passing import BaseMessagePassing


class HeteroConvMessagePassing(BaseMessagePassing):
    """Message passing module using PyTorch Geometric's HeteroConv.

    This module allows using different convolution operators for different
    edge types, making it flexible for complex heterogeneous graph structures.

    Args:
        node_input_dims: Dictionary mapping node types to input dimensions
        node_output_dims: Dictionary mapping node types to output dimensions,
            or a single integer
        edge_input_dims: Dictionary mapping edge types to input dimensions
            (optional)
        edge_output_dims: Dictionary mapping edge types to output dimensions,
            or a single integer (optional)
        conv_types: Dictionary mapping edge types to convolution types
            ('gcn', 'gat', 'sage', 'gin')
        num_layers: Number of message passing layers
        hidden_dims: Hidden dimensions for intermediate layers
        dropout: Dropout probability
        aggr: Aggregation method ('sum', 'mean', 'max', 'mul')
        **kwargs: Additional arguments for specific convolution types
    """

    def __init__(
        self,
        node_input_dims: dict[str, int],
        node_output_dims: dict[str, int] | int,
        edge_input_dims: dict[EdgeType, int] | None = None,
        edge_output_dims: dict[EdgeType, int] | int | None = None,
        conv_types: dict[EdgeType, str] | None = None,
        num_layers: int = 2,
        hidden_dims: dict[str, int] | int | None = None,
        dropout: float = 0.1,
        aggr: str = "sum",
        **kwargs,
    ):
        super().__init__(
            node_input_dims=node_input_dims,
            node_output_dims=node_output_dims,
            edge_input_dims=edge_input_dims,
            edge_output_dims=edge_output_dims,
            conv_types=conv_types,
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            dropout=dropout,
            aggr=aggr,
            **kwargs,
        )

        if isinstance(node_output_dims, int):
            self.node_output_dims = {
                node_type: node_output_dims for node_type in node_input_dims
            }
        else:
            self.node_output_dims = node_output_dims

        if hidden_dims is None:
            self.hidden_dims = self.node_output_dims
        elif isinstance(hidden_dims, int):
            self.hidden_dims = {
                node_type: hidden_dims for node_type in node_input_dims
            }
        else:
            self.hidden_dims = hidden_dims

        if conv_types is None:
            self.conv_types: dict[EdgeType, str] = {}
        else:
            self.conv_types = conv_types

        self.dropout_p = dropout
        self.aggr = aggr
        self.num_layers = num_layers
        self.kwargs = kwargs

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_norms = nn.ModuleDict()

            out_dims = (
                self.node_output_dims
                if i == num_layers - 1
                else self.hidden_dims
            )
            for node_type, dim in out_dims.items():
                layer_norms[node_type] = nn.LayerNorm(dim)

            layer_dict = {"norms": layer_norms}
            layer = nn.ModuleDict(layer_dict)
            self.layers.append(layer)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def _create_conv(
        self,
        conv_type: str,
        src_dim: int,
        dst_dim: int,
        edge_dim: int | None = None,
    ) -> nn.Module:
        if conv_type.lower() == "gcn":
            return GCNConv(src_dim, dst_dim)

        if conv_type.lower() == "gat":
            heads = self.kwargs.get("heads", 4)
            head_dim = dst_dim // heads
            return GATConv(
                src_dim,
                head_dim,
                heads=heads,
                concat=True,
                edge_dim=edge_dim,
                add_self_loops=True,
            )

        if conv_type.lower() == "sage":
            return SAGEConv((src_dim, -1), dst_dim)

        if conv_type.lower() == "gin":
            nn_core = nn.Sequential(
                nn.Linear(src_dim, dst_dim),
                nn.ReLU(),
                nn.Linear(dst_dim, dst_dim),
            )
            return GINConv(nn_core)

        raise ValueError(f"Unknown convolution type: {conv_type}")

    def _initialize_layer_convs(
        self,
        layer_idx: int,
        edge_indices_dict: dict[EdgeType, torch.Tensor],
        edge_attributes_dict: dict[EdgeType, torch.Tensor] | None = None,
    ) -> None:
        layer = self.layers[layer_idx]

        if "convs" in layer:  # type: ignore[operator]
            return

        in_dims = self.node_input_dims if layer_idx == 0 else self.hidden_dims
        out_dims: dict[str, int] = (  # Mypy bug
            self.node_output_dims  # type: ignore[assignment]
            if layer_idx == self.num_layers - 1
            else self.hidden_dims
        )

        conv_dict: dict[EdgeType, nn.Module] = {}
        for edge_type in edge_indices_dict:
            src, _, dst = edge_type

            if src not in in_dims or dst not in out_dims:
                continue

            conv_type = self.conv_types.get(edge_type, "gcn")

            edge_dim = None
            if edge_attributes_dict and edge_type in edge_attributes_dict:
                edge_attr = edge_attributes_dict[edge_type]
                if edge_attr is not None:
                    edge_dim = edge_attr.size(-1)

            src_dim = in_dims[src]
            dst_dim = out_dims[dst]

            conv = self._create_conv(conv_type, src_dim, dst_dim, edge_dim)
            conv_dict[edge_type] = conv

        layer["convs"] = HeteroConv(  # type: ignore[operator]
            conv_dict,
            aggr=self.aggr,
        )

    def forward(  # type: ignore[override]
        self,
        node_features_dict: dict[str, torch.Tensor],
        edge_indices_dict: dict[EdgeType, torch.Tensor],
        edge_attributes_dict: dict[EdgeType, torch.Tensor] | None = None,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[EdgeType, torch.Tensor] | None,
    ]:
        """Forward pass for heterogeneous message passing.

        Args:
            node_features_dict:
                Dictionary mapping node types to feature tensors
            edge_indices_dict:
                Dictionary mapping edge types to edge index tensors
            edge_attributes_dict:
                Dictionary mapping edge types to edge attribute tensors

        Returns:
            Tuple of (processed node features dict, processed edge attributes
            dict)
        """
        if not self.conv_types:
            self.conv_types = {
                edge_type: "gcn" for edge_type in edge_indices_dict
            }

        for i in range(self.num_layers):
            self._initialize_layer_convs(
                i, edge_indices_dict, edge_attributes_dict
            )

        x_dict = node_features_dict

        for i, layer in enumerate(self.layers):
            convs = layer["convs"]

            x_dict_new = convs(
                x_dict, edge_indices_dict, edge_attr_dict=edge_attributes_dict
            )

            for node_type in x_dict_new:
                norms = layer.get("norms", {})
                if node_type in norms:
                    x_dict_new[node_type] = norms[node_type](
                        x_dict_new[node_type]
                    )

                if i < self.num_layers - 1:
                    x_dict_new[node_type] = self.activation(
                        x_dict_new[node_type]
                    )
                    x_dict_new[node_type] = self.dropout(x_dict_new[node_type])

            x_dict = x_dict_new

        return x_dict, edge_attributes_dict
