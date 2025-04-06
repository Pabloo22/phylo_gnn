from typing import Any
from dataclasses import dataclass, field

from phylo_gnn.model.feature_encoders import BaseEncoder, HeteroPeriodicEncoder
from phylo_gnn.model.message_passing import (
    BaseMessagePassing,
    HeteroConv,
)
from phylo_gnn.model.readouts import BaseReadout, SimpleReadout
from phylo_gnn.config import default_node_features, default_edge_attributes
from phylo_gnn.data.feature_extraction import (
    FeaturePipeline,
)
from phylo_gnn import get_project_path


@dataclass
class TrainingConfig:
    batch_size: int = 256
    max_epochs: int = 100
    random_seed: int = 42
    run_name: str | None = None
    train_val_test_split: tuple[float, float, float] = (0.8, 0.10, 0.10)
    num_workers: int = 8
    early_stopping: bool = True
    patience: int = 10


@dataclass
class ProcessFunctionConfig:
    node_features: dict[str, list[FeaturePipeline]] = field(
        default_factory=default_node_features
    )
    edge_types: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            ("node", "has_parent", "node"),
            ("node", "has_child", "node"),
        ]
    )
    edge_attributes: (
        dict[tuple[str, str, str], list[FeaturePipeline]] | None
    ) = field(default_factory=default_edge_attributes)

    def node_features_dims(self) -> dict[str, int]:
        return {
            node_type: len(pipelines)
            for node_type, pipelines in self.node_features.items()
        }

    def edge_attributes_dims(self) -> dict[tuple[str, str, str], int] | None:
        if self.edge_attributes is None:
            return None
        return {
            edge_type: len(pipelines)
            for edge_type, pipelines in self.edge_attributes.items()
        }


@dataclass
class CSVMetadataConfig:
    dataset_dir: str = str(get_project_path() / "data")
    dataset_name: str = "classification_dataset"
    column_names: list[str] = field(default_factory=lambda: ["nwk", "label"])
    label_names: list[str] | None = None
    csv_filenames: list[str] | str | None = None
    detect_csv_files_pattern: str = "*.csv"
    read_csv_kwargs: dict[str, Any] | None = None


@dataclass
class FeatureEncoderConfig:
    cls: type[BaseEncoder] = HeteroPeriodicEncoder
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class MessagePassingConfig:
    cls: type[BaseMessagePassing] = HeteroConv
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReadoutConfig:
    cls: type[BaseReadout] = SimpleReadout
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class PhyloGNNClassifierConfig:
    encoder: FeatureEncoderConfig
    message_passing: MessagePassingConfig
    readout: ReadoutConfig
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str | None = "cosine"
    scheduler_params: dict[str, Any] = field(default_factory=dict)
    eval_interval_steps: int | None = None


@dataclass
class PhyloCSVDatasetConfig:
    encoding_function_config: ProcessFunctionConfig
    root: str = str(get_project_path() / "data")
    csv_metadata_config: CSVMetadataConfig = field(
        default_factory=CSVMetadataConfig
    )


@dataclass
class Config:
    dataset: PhyloCSVDatasetConfig
    model: PhyloGNNClassifierConfig
    training_config: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        self.model.encoder.parameters["node_input_dims"] = (
            self.dataset.encoding_function_config.node_features_dims()
        )
        self.model.encoder.parameters["edge_input_dims"] = (
            self.dataset.encoding_function_config.edge_attributes_dims()
        )
