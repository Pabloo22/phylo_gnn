import abc
from typing import overload
import torch
from torch import nn
from torch_geometric.typing import EdgeType  # type: ignore

from phylo_gnn.model import transform_edge_indices_keys_to_str


class BaseMessagePassing(nn.Module):
    """Base class for message passing"""

    def __init__(
        self,
        node_input_dims: dict[str, int],
        node_output_dims: dict[str, int] | int,
        edge_types: list[EdgeType] | None = None,
        edge_input_dims: dict[EdgeType, int] | None = None,
        edge_output_dims: dict[EdgeType, int] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.node_input_dims = node_input_dims
        self.node_output_dims = node_output_dims
        self._edge_types = edge_types
        self.edge_input_dims = edge_input_dims
        self.edge_output_dims = edge_output_dims
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop("nn", None)
        self.hparams = kwargs_copy
        self.hparams.update(
            {
                "node_input_dims": node_input_dims,
                "node_output_dims": node_output_dims,
                "edge_types": edge_types,
                "edge_input_dims": transform_edge_indices_keys_to_str(
                    edge_input_dims
                ),
                "edge_output_dims": transform_edge_indices_keys_to_str(
                    edge_output_dims
                ),
            }
        )

    @property
    def edge_types(self) -> list[EdgeType]:
        """Returns the edge types."""
        if self._edge_types is None:
            if self.edge_input_dims is None:
                raise ValueError(
                    "edge_input_dims must be provided if edge_types is None."
                )
            self._edge_types = list(self.edge_input_dims.keys())
        return self._edge_types

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
