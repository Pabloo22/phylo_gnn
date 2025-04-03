import abc
from typing import overload
import torch
from torch import nn
from torch_geometric.typing import EdgeType  # type: ignore


class BaseMessagePassing(nn.Module):
    """Base class for message passing"""

    def __init__(
        self,
        node_input_dims: dict[str, int],
        node_output_dims: dict[str, int] | int,
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

    @overload
    def forward(
        self,
        node_features_dict: dict[str, torch.Tensor],
        edge_indices_dict: dict[EdgeType, torch.Tensor],
        edge_attributes_dict: None = None,
    ) -> tuple[
        dict[str, torch.Tensor],
        None,
    ]: ...

    @overload
    def forward(
        self,
        node_features_dict: dict[str, torch.Tensor],
        edge_indices_dict: dict[EdgeType, torch.Tensor],
        edge_attributes_dict: dict[EdgeType, torch.Tensor],
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[EdgeType, torch.Tensor],
    ]: ...

    @abc.abstractmethod
    def forward(
        self,
        node_features_dict: dict[str, torch.Tensor],
        edge_indices_dict: dict[EdgeType, torch.Tensor],
        edge_attributes_dict: (
            dict[EdgeType, torch.Tensor] | None
        ) = None,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[EdgeType, torch.Tensor] | None,
    ]:
        pass
