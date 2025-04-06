from typing import Any
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader  # type: ignore
import pytorch_lightning as pl
from phylo_gnn.data import (
    PhyloCSVDataset,
    create_data_splits,
    create_data_loaders,
)
from phylo_gnn.model import PhyloGNNClassifier
from phylo_gnn.training._utils import (
    configure_logger,
    configure_callbacks,
    initialize_trainer,
)
from phylo_gnn.config import TrainingConfig


def train(
    model: PhyloGNNClassifier,
    dataset: PhyloCSVDataset,
    training_config: TrainingConfig,
) -> dict[str, Any]:
    pl.seed_everything(training_config.random_seed, workers=True)
    train_dataset, val_dataset, test_dataset = create_data_splits(
        dataset,
        training_config.train_val_test_split,
        training_config.random_seed,
    )
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        training_config.batch_size,
        training_config.num_workers,
    )
    logger = configure_logger(
        run_name=training_config.run_name,
    )
    callbacks = configure_callbacks(
        early_stopping=training_config.early_stopping,
        patience=training_config.patience,
    )
    trainer = initialize_trainer(
        logger,
        callbacks=callbacks,
        max_epochs=training_config.max_epochs,
    )
    return execute_training(
        model, trainer, train_loader, val_loader, test_loader
    )


def execute_training(
    model: PhyloGNNClassifier,
    trainer: pl.Trainer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> dict[str, Any]:
    """
    Execute model training, testing, and load the best model.

    Args:
        model: The model to train
        trainer: PyTorch Lightning trainer
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        callbacks: List of callbacks (for accessing checkpoint info)

    Returns:
        Tuple of (trained model, results dictionary)
    """
    trainer.fit(model, train_loader, val_loader)

    test_results = trainer.test(model, test_loader, verbose=True)[0]
    checkpoint_callback: ModelCheckpoint = (
        trainer.checkpoint_callback  # type: ignore[assignment]
    )
    best_model_path = None
    best_model_score = None
    if checkpoint_callback is not None:
        best_model_path = checkpoint_callback.best_model_path
        best_model_score = checkpoint_callback.best_model_score
    results = {
        "test_results": test_results,
        "best_model_path": best_model_path,
        "best_model_score": (
            best_model_score.item() if best_model_score else None
        ),
    }
    return results
