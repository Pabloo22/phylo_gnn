import abc
import torch
from torch import nn
from torch_geometric.typing import EdgeType, NodeType  # type: ignore[import]


class BaseReadout(nn.Module):
    """Base class for node feature encoders"""

    def __init__(
        self,
        node_input_dims: dict[NodeType, int],
        node_output_dims: dict[NodeType, int] | int,
        output_dim: int = 6,
        **kwargs,
    ):
        super().__init__()
        self.node_input_dims = node_input_dims
        self.node_output_dims = node_output_dims
        self.output_dim = output_dim
        self.hparams = kwargs

    @abc.abstractmethod
    def forward(
        self,
        node_features_dict: dict[NodeType, torch.Tensor],
        edge_attributes_dict: (
            dict[EdgeType, torch.Tensor] | None
        ) = None,
    ) -> torch.Tensor:
        pass
