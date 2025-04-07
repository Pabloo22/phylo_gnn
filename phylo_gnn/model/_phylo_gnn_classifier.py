from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchmetrics import Accuracy, F1Score, Precision, Recall
import matplotlib.pyplot as plt
from sklearn.metrics import (  # type: ignore
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch_geometric.data import HeteroData  # type: ignore
import wandb

from phylo_gnn.model import (
    get_node_features_dict,
    get_edge_attributes_dict,
    get_edge_indices_dict,
    get_batch_dict,
)
from phylo_gnn.model.feature_encoders import BaseEncoder
from phylo_gnn.model.readouts import BaseReadout
from phylo_gnn.model.message_passing import BaseMessagePassing


class PhyloGNNClassifier(pl.LightningModule):
    """PyTorch Lightning module for Phylogenetic GNN classification.

    Args:
        encoder:
            Module to encode node and edge features
        message_passing:
            Module to perform graph message passing operations
        readout:
            Module to generate final predictions from node features
        learning_rate:
            Base learning rate for the optimizer
        weight_decay:
            Weight decay for AdamW optimizer
        num_classes:
            Number of classes for classification
        class_weights:
            Optional tensor of weights for weighted loss
        scheduler:
            Type of scheduler to use ('cosine', 'plateau', or None)
        scheduler_params:
            Parameters for the scheduler
        eval_interval_steps:
            Number of steps between evaluations (None forepoch-only eval)
        log_confusion_matrix:
            Whether to log confusion matrices during validation/testing
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        message_passing: BaseMessagePassing,
        readout: BaseReadout,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_classes: int = 6,
        class_weights: torch.Tensor | None = None,
        scheduler: str | None = "cosine",
        scheduler_params: dict[str, Any] | None = None,
        eval_interval_steps: int | None = None,
        log_confusion_matrix: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["encoder", "message_passing", "readout"]
        )

        # Model components
        self.encoder = encoder
        self.message_passing = message_passing
        self.readout = readout

        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params or {}
        self.eval_interval_steps = eval_interval_steps
        self.log_confusion_matrix = log_confusion_matrix

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        self.train_precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        self.train_recall = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_recall = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_recall = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        # For per-class metrics
        self.val_f1_per_class = F1Score(
            task="multiclass", num_classes=num_classes, average=None
        )
        self.test_f1_per_class = F1Score(
            task="multiclass", num_classes=num_classes, average=None
        )

        # Counter for step-based evaluation
        self.train_step_count = 0

        # To collect predictions for confusion matrix
        self.val_preds: list[torch.Tensor] = []
        self.val_targets: list[torch.Tensor] = []
        self.test_preds: list[torch.Tensor] = []
        self.test_targets: list[torch.Tensor] = []

    def forward(  # pylint: disable=arguments-differ
        self, hetero_data, *args, **kwargs
    ):
        """Forward pass through the full GNN model.

        Args:
            data: HeteroData object containing the graph

        Returns:
            Tensor: Class logits
        """
        # Extract features from the HeteroData object
        node_features_dict = get_node_features_dict(hetero_data)
        edge_attr_dict = get_edge_attributes_dict(hetero_data)
        edge_indices_dict = get_edge_indices_dict(hetero_data)

        # Extract batch information
        batch_dict, edge_batch_dict = get_batch_dict(hetero_data)

        # Encode node and edge features
        node_features_dict, edge_attr_dict = self.encoder(
            node_features_dict, edge_attr_dict
        )

        # Message passing
        node_features_dict, edge_attr_dict = self.message_passing(
            node_features_dict, edge_indices_dict, edge_attr_dict
        )

        # Readout to get final prediction
        logits = self.readout(
            node_features_dict, edge_attr_dict, batch_dict, edge_batch_dict
        )

        return logits

    def _shared_step(self, batch: HeteroData, unused_batch_idx, stage: str):
        """Shared computation for training, validation, and test steps.

        Args:
            batch: Input batch
            unused_batch_idx: Batch index
            stage: One of 'train', 'val', or 'test'

        Returns:
            Dict: Dictionary containing loss and other metrics
        """
        # Calculate batch size from node data
        batch_size = 0
        for node_type in batch.node_types:
            if hasattr(batch[node_type], "batch"):
                batch_size = int(batch[node_type].batch.max()) + 1
                break

        if batch_size == 0:
            batch_size = 1

        # Get the targets
        if hasattr(batch, "y"):
            targets = batch.y
        else:
            raise ValueError("No target found in batch")

        # Forward pass
        logits = self(batch)

        # Calculate loss
        if self.class_weights is not None and stage == "train":
            loss = F.cross_entropy(
                logits, targets, weight=self.class_weights.to(logits.device)
            )
        else:
            loss = F.cross_entropy(logits, targets)

        # Get predictions
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        if stage == "train":
            self.train_acc(preds, targets)
            self.train_f1(preds, targets)
            self.train_precision(preds, targets)
            self.train_recall(preds, targets)
        elif stage == "val":
            self.val_acc(preds, targets)
            self.val_f1(preds, targets)
            self.val_precision(preds, targets)
            self.val_recall(preds, targets)
            self.val_f1_per_class(preds, targets)

            # Collect predictions for confusion matrix
            if self.log_confusion_matrix:
                self.val_preds.append(preds.detach().cpu())
                self.val_targets.append(targets.detach().cpu())
        elif stage == "test":
            self.test_acc(preds, targets)
            self.test_f1(preds, targets)
            self.test_precision(preds, targets)
            self.test_recall(preds, targets)
            self.test_f1_per_class(preds, targets)

            # Collect predictions for confusion matrix
            if self.log_confusion_matrix:
                self.test_preds.append(preds.detach().cpu())
                self.test_targets.append(targets.detach().cpu())

        return {
            "loss": loss,
            "preds": preds,
            "targets": targets,
            "batch_size": batch_size,
        }

    def training_step(  # pylint: disable=arguments-differ
        self, batch, batch_idx, *args, **kwargs
    ):
        """Training step.

        Args:
            batch: Input batch
            batch_idx: Batch index
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Tensor: Loss value
        """
        self.train_step_count += 1

        # Perform the shared computation
        result = self._shared_step(batch, batch_idx, "train")
        loss = result["loss"]

        # Log metrics
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc",
            self.train_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=result["batch_size"],
        )
        self.log(
            "train/f1",
            self.train_f1,
            on_step=True,
            on_epoch=True,
            batch_size=result["batch_size"],
        )
        self.log(
            "train/precision",
            self.train_precision,
            on_step=False,
            on_epoch=True,
            batch_size=result["batch_size"],
        )
        self.log(
            "train/recall",
            self.train_recall,
            on_step=False,
            on_epoch=True,
            batch_size=result["batch_size"],
        )

        # Step-based evaluation
        if (
            self.eval_interval_steps is not None
            and self.train_step_count % self.eval_interval_steps == 0
        ):
            # Perform validation
            self.trainer.validate(self, self.trainer.val_dataloaders)

            # Log step count for reference
            self.log(
                "train/step",
                float(self.train_step_count),
                on_step=True,
                on_epoch=False,
                batch_size=result["batch_size"],
            )

        return loss

    def validation_step(  # pylint: disable=arguments-differ
        self, batch, batch_idx, *args, **kwargs
    ):
        """Validation step.

        Args:
            batch: Input batch
            batch_idx: Batch index

        Returns:
            Dict: Dictionary containing validation metrics
        """
        result = self._shared_step(batch, batch_idx, "val")
        batch_size = result["batch_size"]

        # Log metrics
        self.log(
            "val/loss",
            result["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "val/acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "val/f1",
            self.val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "val/precision",
            self.val_precision,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val/recall",
            self.val_recall,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return result

    def test_step(  # pylint: disable=arguments-differ
        self, batch, batch_idx, *args, **kwargs
    ):
        """Test step.

        Args:
            batch: Input batch
            batch_idx: Batch index

        Returns:
            Dict: Dictionary containing test metrics
        """
        result = self._shared_step(batch, batch_idx, "test")
        batch_size = result["batch_size"]

        # Log metrics
        self.log(
            "test/loss",
            result["loss"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "test/acc",
            self.test_acc,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "test/f1",
            self.test_f1,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "test/precision",
            self.test_precision,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "test/recall",
            self.test_recall,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return result

    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch."""
        # Log per-class F1 scores
        f1_per_class = self.val_f1_per_class.compute()
        for i, f1 in enumerate(f1_per_class):
            self.log(f"val/f1_class_{i}", f1, on_step=False, on_epoch=True)

        # Log confusion matrix
        if self.log_confusion_matrix and self.val_preds:
            preds = torch.cat(self.val_preds).numpy()
            targets = torch.cat(self.val_targets).numpy()

            # Create and log confusion matrix
            self._log_confusion_matrix(preds, targets, "val")

            # Reset lists
            self.val_preds = []
            self.val_targets = []

    def on_test_epoch_end(self):
        """Called at the end of the test epoch."""
        # Log per-class F1 scores
        f1_per_class = self.test_f1_per_class.compute()
        for i, f1 in enumerate(f1_per_class):
            self.log(f"test/f1_class_{i}", f1, on_step=False, on_epoch=True)

        # Log confusion matrix
        if self.log_confusion_matrix and self.test_preds:
            preds = torch.cat(self.test_preds).numpy()
            targets = torch.cat(self.test_targets).numpy()

            # Create and log confusion matrix
            self._log_confusion_matrix(preds, targets, "test")

            # Reset lists
            self.test_preds = []
            self.test_targets = []

    @rank_zero_only
    def _log_confusion_matrix(self, preds, targets, stage):
        """Generate and log a confusion matrix to wandb.

        Args:
            preds: Model predictions
            targets: Ground truth labels
            stage: Either 'val' or 'test'
        """
        # Generate confusion matrix
        cm = confusion_matrix(targets, preds)

        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        plt.title(f"{stage.capitalize()} Confusion Matrix")

        # Log to wandb
        self.logger.experiment.log(
            {f"{stage}/confusion_matrix": wandb.Image(fig)}
        )
        plt.close(fig)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.

        Returns:
            Dict or List: Optimizer and scheduler configuration
        """
        # Create optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Return just the optimizer if no scheduler is requested
        if self.scheduler is None:
            return optimizer

        # Configure scheduler
        if self.scheduler == "cosine":
            t_max = self.scheduler_params.get("T_max", 10)
            eta_min = self.scheduler_params.get("eta_min", 0)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=eta_min
            )
        elif self.scheduler == "plateau":
            patience = self.scheduler_params.get("patience", 3)
            factor = self.scheduler_params.get("factor", 0.1)
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=factor,
                patience=patience,
                verbose=True,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "monitor": "val/loss",
            "frequency": 1,
        }

        if self.scheduler == "plateau":
            scheduler_config["reduce_on_plateau"] = True

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def on_fit_start(self):
        """Called when fit begins.

        This is where we initialize the wandb run with our hyperparameters.
        """
        if isinstance(self.logger, pl.loggers.WandbLogger):
            # Log model architecture as text
            self.logger.experiment.config.update(
                {
                    "model_summary": str(self),
                    "encoder": str(self.encoder.__class__.__name__),
                    "message_passing": str(
                        self.message_passing.__class__.__name__
                    ),
                    "readout": str(self.readout.__class__.__name__),
                    "num_classes": self.num_classes,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "scheduler": self.scheduler,
                    "scheduler_params": self.scheduler_params,
                    "eval_interval_steps": self.eval_interval_steps,
                }
            )

    def on_save_checkpoint(self, checkpoint):
        """Called when saving a checkpoint.

        This allows us to add extra information to the checkpoint.
        """
        # Add model configuration to checkpoint
        checkpoint["model_config"] = {
            "encoder_config": (
                self.encoder.hparams
                if hasattr(self.encoder, "hparams")
                else {}
            ),
            "message_passing_config": (
                self.message_passing.hparams
                if hasattr(self.message_passing, "hparams")
                else {}
            ),
            "readout_config": (
                self.readout.hparams
                if hasattr(self.readout, "hparams")
                else {}
            ),
        }

    @classmethod
    def load_from_checkpoint(  # pylint: disable=arguments-differ
        cls, checkpoint_path, **kwargs
    ):
        """Load model from checkpoint with custom handling of model components.

        This extends the default load_from_checkpoint to handle our model
        components.

        Args:
            checkpoint_path: Path to the checkpoint
            **kwargs: Additional arguments to pass to the model constructor

        Returns:
            PhyloGNNModule: Loaded model
        """
        # Load model with standard method
        model = super().load_from_checkpoint(checkpoint_path, **kwargs)

        # Log that the model was loaded
        print(f"Loaded model from {checkpoint_path}")

        return model
