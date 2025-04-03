import abc
import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    """Base class for node feature encoders"""

    def __init__(
        self,
        node_input_dims: dict[str, int],
        node_output_dims: dict[str, int] | int,
        edge_input_dims: dict[tuple[str, str, str], int] | None = None,
        edge_output_dims: dict[tuple[str, str, str], int] | None = None,
    ):
        super().__init__()
        self.node_input_dims = node_input_dims
        self.node_output_dims = node_output_dims
        self.edge_input_dims = edge_input_dims
        self.edge_output_dims = edge_output_dims

    @abc.abstractmethod
    def forward(
        self,
        node_features_dict: dict[str, torch.Tensor],
        edge_attributes_dict: (
            dict[tuple[str, str, str], torch.Tensor] | None
        ) = None,
    ) -> tuple[
        dict[str, torch.Tensor], dict[tuple[str, str, str], torch.Tensor]
    ]:
        pass
