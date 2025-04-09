from phylo_gnn.config import (
    Config,
    TrainingConfig,
    PhyloCSVDatasetConfig,
    CSVMetadataConfig,
    PhyloGNNClassifierConfig,
    MessagePassingConfig,
    ProcessFunctionConfig,
    ReadoutConfig,
    FeatureEncoderConfig,
)
from phylo_gnn.config import (
    node_features_with_level_nodes,
    edge_attributes_with_level_nodes,
)
from phylo_gnn.data.feature_extraction import NodeNames, EdgeNames

DEBUG = Config(
    training_config=TrainingConfig(run_name="debug"),
    dataset=PhyloCSVDatasetConfig(
        csv_metadata_config=CSVMetadataConfig(
            csv_filenames=["87_10k_nwk_subset_1000.csv"],
            processed_filename="87_10k_nwk_subset_1000",
        ),
        force_reload=False,
    ),
)
DEBUG_2 = Config(
    training_config=TrainingConfig(run_name="debug_2"),
    dataset=PhyloCSVDatasetConfig(
        csv_metadata_config=CSVMetadataConfig(
            csv_filenames=["87_10k_nwk.csv"],
            processed_filename="87_10k_nwk",
        ),
    ),
    model=PhyloGNNClassifierConfig(
        message_passing=MessagePassingConfig(parameters={"layer_norm": False}),
    ),
)
EXPERIMENT_1 = Config(
    training_config=TrainingConfig(
        run_name="exp_1", patience=25, num_workers=1
    ),
    dataset=PhyloCSVDatasetConfig(
        csv_metadata_config=CSVMetadataConfig(
            csv_filenames=[
                "87_10k_nwk.csv",
                "489_10k_nwk.csv",
                "674_10k_nwk.csv",
            ],
            processed_filename="basic_processing_all",
        ),
    ),
    model=PhyloGNNClassifierConfig(
        message_passing=MessagePassingConfig(parameters={"layer_norm": False}),
    ),
)
EXPERIMENT_2 = Config(
    training_config=TrainingConfig(
        run_name="exp_2", patience=40, num_workers=1
    ),
    dataset=PhyloCSVDatasetConfig(
        csv_metadata_config=CSVMetadataConfig(
            csv_filenames=[
                "87_10k_nwk.csv",
                "489_10k_nwk.csv",
                "674_10k_nwk.csv",
            ],
            processed_filename="basic_processing_all",
        ),
    ),
    model=PhyloGNNClassifierConfig(
        message_passing=MessagePassingConfig(
            parameters={
                "layer_norm": False,
                "dropout": 0.0,
            }
        ),
        scheduler=None,
    ),
)


EXPERIMENT_3 = Config(
    training_config=TrainingConfig(
        run_name="exp_3_readout_lr",
        patience=40,
        num_workers=1,
    ),
    dataset=PhyloCSVDatasetConfig(
        csv_metadata_config=CSVMetadataConfig(
            csv_filenames=[
                "87_10k_nwk.csv",
                "489_10k_nwk.csv",
                "674_10k_nwk.csv",
            ],
            processed_filename="basic_processing_all",
        ),
    ),
    model=PhyloGNNClassifierConfig(
        message_passing=MessagePassingConfig(
            parameters={
                "layer_norm": False,
                "dropout": 0.0,
            }
        ),
        scheduler=None,
        learning_rate=0.0003,
        weight_decay=0.00001,
    ),
)
DEBUG_3 = Config(
    training_config=TrainingConfig(
        run_name="debug_level_nodes",
        patience=40,
        num_workers=1,
        max_epochs=100,
    ),
    dataset=PhyloCSVDatasetConfig(
        process_function_config=ProcessFunctionConfig(
            node_features=node_features_with_level_nodes(),
            edge_attributes=edge_attributes_with_level_nodes(),
            edge_types=list(edge_attributes_with_level_nodes().keys()),
        ),
        csv_metadata_config=CSVMetadataConfig(
            csv_filenames=["87_10k_nwk_subset_1000.csv"],
            processed_filename="test_level_nodes",
        ),
    ),
    model=PhyloGNNClassifierConfig(
        encoder=FeatureEncoderConfig(
            node_output_dims={
                NodeNames.LEVEL.value: 96,
                NodeNames.NODE.value: 28,
            },
            edge_output_dims=4,
        ),
        message_passing=MessagePassingConfig(
            parameters={
                "layer_norm": False,
                "dropout": 0.0,
            },
            node_output_dims={
                NodeNames.LEVEL.value: 96,
                NodeNames.NODE.value: 28,
            },
            edge_output_dims=4,
        ),
        readout=ReadoutConfig(
            parameters={
                "node_types_to_use": ["level"],
                "edge_attributes_to_use": [
                    (
                        NodeNames.NODE.value,
                        EdgeNames.HAS_PARENT.value,
                        NodeNames.NODE.value,
                    )
                ],
            },
        ),
        scheduler=None,
        learning_rate=0.0003,
        weight_decay=0.00001,
    ),
)
