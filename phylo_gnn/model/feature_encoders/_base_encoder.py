from typing import overload
import torch
from torch import nn


class BaseEncoder(nn.Module):
    """Base class for node feature encoders"""

    def __init__(
        self,
        node_input_dims: dict[str, int],
        node_output_dims: dict[str, int] | int,
        edge_input_dims: dict[tuple[str, str, str], int] | None = None,
        edge_output_dims: dict[tuple[str, str, str], int] | int | None = None,
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
        edge_attributes_dict: None = None,
    ) -> tuple[
        dict[str, torch.Tensor],
        None,
    ]: ...

    @overload
    def forward(
        self,
        node_features_dict: dict[str, torch.Tensor],
        edge_attributes_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[tuple[str, str, str], torch.Tensor],
    ]: ...

    def forward(
        self,
        node_features_dict: dict[str, torch.Tensor],
        edge_attributes_dict: (
            dict[tuple[str, str, str], torch.Tensor] | None
        ) = None,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[tuple[str, str, str], torch.Tensor] | None,
    ]:
        raise NotImplementedError(
            "The forward method must be implemented in subclasses."
        )
