import abc
import torch
from torch import nn


class BaseReadout(nn.Module):
    """Base class for node feature encoders"""

    def __init__(
        self,
        node_input_dims: dict[str, int],
        node_output_dims: dict[str, int] | int,
        output_dim: int = 6,
        **kwargs,
    ):
        super().__init__()
        self.node_input_dims = node_input_dims
        self.node_output_dims = node_output_dims
        self.output_dim = output_dim

    @abc.abstractmethod
    def forward(
        self,
        node_features_dict: dict[str, torch.Tensor],
        edge_attributes_dict: (
            dict[tuple[str, str, str], torch.Tensor] | None
        ) = None,
    ) -> torch.Tensor:
        pass
