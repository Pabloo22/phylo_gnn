import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    Callback,
)
from pytorch_lightning.loggers import WandbLogger

from phylo_gnn import get_project_path


def configure_logger(
    wandb_project: str = "phylo_gnn_classification",
    run_name: str | None = None,
    **kwargs,
) -> WandbLogger:
    return WandbLogger(project=wandb_project, name=run_name, **kwargs)


def configure_callbacks(
    early_stopping: bool = True,
    patience: int = 10,
    monitor: str = "val/acc",
    mode: str = "max",
) -> list[Callback]:
    """Configure callbacks for the PyTorch Lightning Trainer.

    Args:
        checkpoint_dir: Directory to save checkpoints
        early_stopping: Whether to use early stopping
        patience: Patience for early stopping

    Returns:
        List of callbacks
    """
    callbacks: list[Callback] = []

    # get current day and hour for the checkpoint directory
    current_time = time.localtime()
    current_day = time.strftime("%Y-%m-%d", current_time)
    current_hour = time.strftime("%H", current_time)
    checkpoint_dir = f"{current_day}_{current_hour}"
    dirpath = get_project_path() / "models" / checkpoint_dir
    # Create the directory if it doesn't exist
    dirpath.mkdir(parents=True, exist_ok=True)
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename="model-{epoch:02d}-{val_loss:.2f}-{val_f1:.4f}",
        monitor=monitor,
        mode=mode,
        save_top_k=3,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    if early_stopping:
        early_stop_callback = EarlyStopping(
            monitor=monitor, mode=mode, patience=patience, verbose=True
        )
        callbacks.append(early_stop_callback)

    return callbacks


def initialize_trainer(
    logger: WandbLogger,
    callbacks: list[Callback],
    max_epochs: int = 100,
    log_every_n_steps: int = 20,
    detect_anomaly: bool = False,
    gpu_id: int | None = None,
    **kwargs,
) -> pl.Trainer:
    if gpu_id is not None:
        devices_arg: list[int] | str = [gpu_id]
        accelerator_arg = "gpu"
    else:
        # Fallback or default behavior (e.g., use all GPUs or CPU)
        devices_arg = "auto"
        accelerator_arg = "auto"
    return pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=max_epochs,
        detect_anomaly=detect_anomaly,
        log_every_n_steps=log_every_n_steps,
        accelerator=accelerator_arg,
        devices=devices_arg,
        **kwargs,
    )
