import torch
from numpy.typing import NDArray


def get_graph_classification_target(label_row: NDArray) -> torch.Tensor:
    if label_row.ndim == 2:
        label_row = label_row.flatten()
    elif label_row.ndim > 2:
        raise ValueError("Label row has more than 2 dimensions")
    assert (
        label_row.shape[0] == 1
    ), f"Label row must have shape [1], got {label_row.shape}."
    return torch.tensor(label_row, dtype=torch.long)
