from typing import overload
import torch
from torch import nn
from torch_geometric.typing import EdgeType, NodeType  # type: ignore[import]


class BaseEncoder(nn.Module):
    """Base class for node feature encoders"""

    def __init__(
        self,
        node_input_dims: dict[NodeType, int],
        node_output_dims: dict[NodeType, int] | int,
        edge_input_dims: dict[EdgeType, int] | None = None,
        edge_output_dims: dict[EdgeType, int] | int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.node_input_dims = node_input_dims
        self.node_output_dims = node_output_dims
        self.edge_input_dims = edge_input_dims
        self.edge_output_dims = edge_output_dims
        self.hparams = kwargs
        self.hparams.update(
            {
                "node_input_dims": node_input_dims,
                "node_output_dims": node_output_dims,
                "edge_input_dims": edge_input_dims,
                "edge_output_dims": edge_output_dims,
            }
        )

    @overload
    def forward(
        self,
        node_features_dict: dict[NodeType, torch.Tensor],
        edge_attributes_dict: None = None,
    ) -> tuple[
        dict[NodeType, torch.Tensor],
        None,
    ]: ...

    @overload
    def forward(
        self,
        node_features_dict: dict[NodeType, torch.Tensor],
        edge_attributes_dict: dict[EdgeType, torch.Tensor],
    ) -> tuple[
        dict[NodeType, torch.Tensor],
        dict[EdgeType, torch.Tensor],
    ]: ...

    def forward(
        self,
        node_features_dict: dict[NodeType, torch.Tensor],
        edge_attributes_dict: (
            dict[EdgeType, torch.Tensor] | None
        ) = None,
    ) -> tuple[
        dict[NodeType, torch.Tensor],
        dict[EdgeType, torch.Tensor] | None,
    ]:
        raise NotImplementedError(
            "The forward method must be implemented in subclasses."
        )
