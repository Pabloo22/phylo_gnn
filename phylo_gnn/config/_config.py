from collections.abc import Mapping
from typing import Any
from dataclasses import dataclass, field
import pathlib

from torch_geometric.typing import EdgeType  # type: ignore

from phylo_gnn.model.feature_encoders import BaseEncoder, HeteroPeriodicEncoder
from phylo_gnn.model.message_passing import (
    BaseMessagePassing,
    HeteroConvMessagePassing,
)
from phylo_gnn.model import PhyloGNNClassifier
from phylo_gnn.model.readouts import BaseReadout, SimpleReadout
from phylo_gnn.config import default_node_features, default_edge_attributes
from phylo_gnn.data import PhyloCSVDataset, CSVMetadata
from phylo_gnn.data.feature_extraction import (
    FeaturePipeline,
    ProcessFunction,
    get_process_function,
    get_edge_feature_extractor,
    get_edge_indices_extractor,
    get_node_feature_extractor,
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
    node_features: Mapping[str, list[FeaturePipeline]] = field(
        default_factory=default_node_features
    )
    edge_types: list[EdgeType] = field(
        default_factory=lambda: [
            ("node", "has_parent", "node"),
            ("node", "has_child", "node"),
        ]
    )
    edge_attributes: Mapping[EdgeType, list[FeaturePipeline]] | None = field(
        default_factory=default_edge_attributes
    )

    def node_features_dims(self) -> dict[str, int]:
        return {
            node_type: len(pipelines)
            for node_type, pipelines in self.node_features.items()
        }

    def edge_attributes_dims(self) -> dict[EdgeType, int] | None:
        if self.edge_attributes is None:
            return None
        return {
            edge_type: len(pipelines)
            for edge_type, pipelines in self.edge_attributes.items()
        }

    def initialize(self) -> ProcessFunction:
        node_feature_extractor = get_node_feature_extractor(self.node_features)
        edge_indices_extractor = get_edge_indices_extractor(self.edge_types)
        edge_attribute_extractor = (
            get_edge_feature_extractor(self.edge_attributes)
            if self.edge_attributes is not None
            else None
        )

        return get_process_function(
            node_feature_extractor=node_feature_extractor,
            edge_indices_extractor=edge_indices_extractor,
            edge_attribute_extractor=edge_attribute_extractor,
        )


@dataclass
class CSVMetadataConfig:
    dataset_dir: str | pathlib.Path = get_project_path() / "data"
    dataset_name: str = "classification_dataset"
    column_names: list[str] = field(default_factory=lambda: ["nwk", "label"])
    label_names: list[str] | None = None
    csv_filenames: list[str] | str | None = None
    detect_csv_files_pattern: str = "*.csv"
    read_csv_kwargs: dict[str, Any] | None = None

    def initialize(self) -> CSVMetadata:
        return CSVMetadata(
            dataset_dir=pathlib.Path(self.dataset_dir),
            dataset_name=self.dataset_name,
            column_names=self.column_names,
            label_names=self.label_names,
            csv_filenames=self.csv_filenames,
            detect_csv_files_pattern=self.detect_csv_files_pattern,
            read_csv_kwargs=self.read_csv_kwargs,
        )


@dataclass
class FeatureEncoderConfig:
    cls: type[BaseEncoder] = HeteroPeriodicEncoder
    node_output_dims: int | dict[str, int] = 32
    edge_output_dims: int | dict[EdgeType, int] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def node_output_dims_dict(self) -> dict[str, int]:
        if isinstance(self.node_output_dims, int):
            return {
                node_type: self.node_output_dims
                for node_type in self.parameters["node_input_dims"].keys()
            }
        return self.node_output_dims

    @property
    def edge_output_dims_dict(self) -> dict[EdgeType, int] | None:
        if isinstance(self.edge_output_dims, int):
            edge_input_dims = self.parameters["edge_input_dims"]
            if edge_input_dims is None:
                return None
            return {
                edge_type: self.edge_output_dims
                for edge_type in edge_input_dims.keys()
            }
        return self.edge_output_dims

    def initialize(self, **kwargs: Any) -> BaseEncoder:
        return self.cls(
            node_output_dims=self.node_output_dims_dict,
            edge_output_dims=self.edge_output_dims_dict,
            **{**self.parameters, **kwargs},
        )


@dataclass
class MessagePassingConfig:
    cls: type[BaseMessagePassing] = HeteroConvMessagePassing
    node_output_dims: int | dict[str, int] = 32
    edge_output_dims: int | dict[EdgeType, int] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def node_output_dims_dict(self) -> dict[str, int]:
        if isinstance(self.node_output_dims, int):
            return {
                node_type: self.node_output_dims
                for node_type in self.parameters["node_input_dims"].keys()
            }
        return self.node_output_dims

    @property
    def edge_output_dims_dict(self) -> dict[EdgeType, int] | None:
        if isinstance(self.edge_output_dims, int):
            edge_input_dims = self.parameters["edge_input_dims"]
            if edge_input_dims is None:
                return None
            return {
                edge_type: self.edge_output_dims
                for edge_type in edge_input_dims.keys()
            }
        return self.edge_output_dims

    def initialize(self, **kwargs: Any) -> BaseMessagePassing:
        return self.cls(
            node_output_dims=self.node_output_dims_dict,
            edge_output_dims=self.edge_output_dims_dict,
            **{**self.parameters, **kwargs},
        )


@dataclass
class ReadoutConfig:
    cls: type[BaseReadout] = SimpleReadout
    parameters: dict[str, Any] = field(default_factory=dict)

    def initialize(self, **kwargs: Any) -> BaseReadout:
        return self.cls(**{**self.parameters, **kwargs})


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

    def initialize(self) -> PhyloGNNClassifier:
        return PhyloGNNClassifier(
            encoder=self.encoder.initialize(),
            message_passing=self.message_passing.initialize(),
            readout=self.readout.initialize(),
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            scheduler=self.scheduler,
            scheduler_params=self.scheduler_params,
            eval_interval_steps=self.eval_interval_steps,
        )


@dataclass
class PhyloCSVDatasetConfig:
    process_function_config: ProcessFunctionConfig = field(
        default_factory=ProcessFunctionConfig
    )
    root: str = str(get_project_path() / "data")
    csv_metadata_config: CSVMetadataConfig = field(
        default_factory=CSVMetadataConfig
    )
    force_reload: bool = False
    kwargs: dict[str, Any] = field(default_factory=dict)

    def initialize(self) -> PhyloCSVDataset:
        csv_metadata = self.csv_metadata_config.initialize()
        return PhyloCSVDataset(
            root=self.root,
            csv_metadata=csv_metadata,
            process_function=self.process_function_config.initialize(),
            force_reload=self.force_reload,
            **self.kwargs,
        )


@dataclass
class Config:
    model: PhyloGNNClassifierConfig
    dataset: PhyloCSVDatasetConfig = field(
        default_factory=PhyloCSVDatasetConfig
    )
    training_config: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        # Get dimensions from dataset
        node_input_dims = (
            self.dataset.process_function_config.node_features_dims()
        )
        edge_input_dims = (
            self.dataset.process_function_config.edge_attributes_dims()
        )

        # Setup encoder dimensions
        self.model.encoder.parameters["node_input_dims"] = node_input_dims
        self.model.encoder.parameters["edge_input_dims"] = edge_input_dims

        # Get dimensions using properties that handle conversion
        node_output_dims = self.model.encoder.node_output_dims_dict
        edge_output_dims = self.model.encoder.edge_output_dims_dict

        # Connect encoder to message passing
        self.model.message_passing.parameters["node_input_dims"] = (
            node_output_dims
        )
        self.model.message_passing.parameters["edge_input_dims"] = (
            edge_output_dims
        )

        # Get message passing output dimensions
        mp_node_output_dims = self.model.message_passing.node_output_dims_dict
        mp_edge_output_dims = self.model.message_passing.edge_output_dims_dict

        # Connect message passing to readout
        self.model.readout.parameters["node_input_dims"] = mp_node_output_dims
        self.model.readout.parameters["edge_input_dims"] = mp_edge_output_dims

    def initialize_model(self) -> PhyloGNNClassifier:
        return self.model.initialize()

    def initialize_dataset(self) -> PhyloCSVDataset:
        return self.dataset.initialize()
