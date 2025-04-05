import torch
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader  # type: ignore

from phylo_gnn.data import PhyloCSVDataset


def create_data_splits(
    dataset: PhyloCSVDataset,
    train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        dataset: The dataset to split
        train_val_test_split: Proportion of data for train/val/test
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    total_size = len(dataset)
    train_size = int(train_val_test_split[0] * total_size)
    val_size = int(train_val_test_split[1] * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = (
        random_split(  # type: ignore[var-annotated]
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )
    )
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(
    train_dataset: Subset,
    val_dataset: Subset,
    test_dataset: Subset,
    batch_size: int = 32,
    num_workers: int = 8,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
