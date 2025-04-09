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
EXPERIMENT_4_1 = Config(
    training_config=TrainingConfig(
        run_name="exp_4_levels_87_tips_run_1",
        patience=20,
        num_workers=8,
        max_epochs=1000,
    ),
    dataset=PhyloCSVDatasetConfig(
        process_function_config=ProcessFunctionConfig(
            node_features=node_features_with_level_nodes(),
            edge_attributes=edge_attributes_with_level_nodes(),
            edge_types=list(edge_attributes_with_level_nodes().keys()),
        ),
        csv_metadata_config=CSVMetadataConfig(
            csv_filenames=["87_10k_nwk.csv"],
            processed_filename="87_10k_level_nodes",
        ),
    ),
    model=PhyloGNNClassifierConfig(
        encoder=FeatureEncoderConfig(
            node_output_dims={
                NodeNames.LEVEL.value: 96,
                NodeNames.NODE.value: 28,
            },
            edge_output_dims=8,
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
            edge_output_dims=8,
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
EXPERIMENT_5 = Config(
    training_config=TrainingConfig(
        run_name="exp_5_levels_all_trees",
        patience=20,
        num_workers=8,
        max_epochs=1000,
        gpu_id=1,
    ),
    dataset=PhyloCSVDatasetConfig(
        process_function_config=ProcessFunctionConfig(
            node_features=node_features_with_level_nodes(),
            edge_attributes=edge_attributes_with_level_nodes(),
            edge_types=list(edge_attributes_with_level_nodes().keys()),
        ),
        csv_metadata_config=CSVMetadataConfig(
            csv_filenames=[
                "87_10k_nwk.csv",
                "489_10k_nwk.csv",
                "674_10k_nwk.csv",
            ],
            processed_filename="all_trees_level_nodes",
        ),
    ),
    model=PhyloGNNClassifierConfig(
        encoder=FeatureEncoderConfig(
            node_output_dims={
                NodeNames.LEVEL.value: 96,
                NodeNames.NODE.value: 42,
            },
            edge_output_dims=8,
        ),
        message_passing=MessagePassingConfig(
            parameters={
                "layer_norm": False,
                "dropout": 0.0,
            },
            node_output_dims={
                NodeNames.LEVEL.value: 96,
                NodeNames.NODE.value: 42,
            },
            edge_output_dims=8,
        ),
        readout=ReadoutConfig(
            parameters={
                "node_types_to_use": [
                    NodeNames.LEVEL.value,
                    NodeNames.NODE.value,
                ],
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
        learning_rate=0.0004,
        weight_decay=0.00001,
    ),
)
EXPERIMENT_6 = Config(
    training_config=TrainingConfig(
        run_name="exp_6_levels_87t_more_dim",
        patience=20,
        num_workers=8,
        max_epochs=1000,
        gpu_id=None,
    ),
    dataset=PhyloCSVDatasetConfig(
        process_function_config=ProcessFunctionConfig(
            node_features=node_features_with_level_nodes(),
            edge_attributes=edge_attributes_with_level_nodes(),
            edge_types=list(edge_attributes_with_level_nodes().keys()),
        ),
        csv_metadata_config=CSVMetadataConfig(
            csv_filenames=["87_10k_nwk.csv"],
            processed_filename="87_10k_level_nodes",
        ),
    ),
    model=PhyloGNNClassifierConfig(
        encoder=FeatureEncoderConfig(
            node_output_dims={
                NodeNames.LEVEL.value: 96,
                NodeNames.NODE.value: 42,
            },
            edge_output_dims=8,
        ),
        message_passing=MessagePassingConfig(
            parameters={
                "layer_norm": False,
                "dropout": 0.0,
            },
            node_output_dims={
                NodeNames.LEVEL.value: 96,
                NodeNames.NODE.value: 42,
            },
            edge_output_dims=8,
        ),
        readout=ReadoutConfig(
            parameters={
                "node_types_to_use": [
                    NodeNames.LEVEL.value,
                    NodeNames.NODE.value,
                ],
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
        learning_rate=0.0004,
        weight_decay=0.00001,
    ),
)
EXPERIMENT_7 = Config(
    training_config=TrainingConfig(
        run_name="exp_7_levels_87t",
        patience=10,
        num_workers=8,
        max_epochs=1000,
        gpu_id=None,
    ),
    dataset=PhyloCSVDatasetConfig(
        process_function_config=ProcessFunctionConfig(
            node_features=node_features_with_level_nodes(),
            edge_attributes=edge_attributes_with_level_nodes(
                only_node2level=True
            ),
            edge_types=list(
                edge_attributes_with_level_nodes(only_node2level=True).keys()
            ),
        ),
        csv_metadata_config=CSVMetadataConfig(
            csv_filenames=["87_10k_nwk.csv"],
            processed_filename="87_10k_level_nodes_no_level2node",
        ),
    ),
    model=PhyloGNNClassifierConfig(
        encoder=FeatureEncoderConfig(
            node_output_dims={
                NodeNames.LEVEL.value: 144,
                NodeNames.NODE.value: 42,
            },
            edge_output_dims=8,
        ),
        message_passing=MessagePassingConfig(
            parameters={
                "layer_norm": True,
                "dropout": 0.1,
                "mlp_activation": "relu",
            },
            node_output_dims={
                NodeNames.LEVEL.value: 144,
                NodeNames.NODE.value: 42,
            },
            edge_output_dims=8,
        ),
        readout=ReadoutConfig(
            parameters={
                "node_types_to_use": [
                    NodeNames.LEVEL.value,
                    NodeNames.NODE.value,
                ],
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
        learning_rate=0.0005,
        weight_decay=0.00001,
    ),
)
EXPERIMENT_8 = Config(
    training_config=TrainingConfig(
        run_name="exp_8_levels_87t_more_dim",
        patience=10,
        num_workers=8,
        max_epochs=1000,
        gpu_id=None,
    ),
    dataset=PhyloCSVDatasetConfig(
        process_function_config=ProcessFunctionConfig(
            node_features=node_features_with_level_nodes(),
            edge_attributes=edge_attributes_with_level_nodes(
                only_node2level=True
            ),
            edge_types=list(
                edge_attributes_with_level_nodes(only_node2level=True).keys()
            ),
        ),
        csv_metadata_config=CSVMetadataConfig(
            csv_filenames=["87_10k_nwk.csv"],
            processed_filename="87_10k_level_nodes_no_level2node",
        ),
    ),
    model=PhyloGNNClassifierConfig(
        encoder=FeatureEncoderConfig(
            node_output_dims={
                NodeNames.LEVEL.value: 144,
                NodeNames.NODE.value: 42,
            },
            edge_output_dims=8,
        ),
        message_passing=MessagePassingConfig(
            parameters={
                "layer_norm": True,
                "dropout": 0.,
                "mlp_activation": "elu",
            },
            node_output_dims={
                NodeNames.LEVEL.value: 144,
                NodeNames.NODE.value: 42,
            },
            edge_output_dims=8,
        ),
        readout=ReadoutConfig(
            parameters={
                "node_types_to_use": [
                    NodeNames.LEVEL.value,
                ],
                "edge_attributes_to_use": [
                    (
                        NodeNames.NODE.value,
                        EdgeNames.HAS_PARENT.value,
                        NodeNames.NODE.value,
                    )
                ],
                "aggregator": "max",
                "dropout": 0,
            },
        ),
        scheduler=None,
        learning_rate=0.0002,
        weight_decay=0.00005,
    ),
)
EXPERIMENT_9 = Config(
    training_config=TrainingConfig(
        run_name="exp_489t_levels_more_dim",
        patience=20,
        num_workers=8,
        max_epochs=1000,
        gpu_id=1,
        train_val_test_split=(0.85, 0.05, 0.1),
    ),
    dataset=PhyloCSVDatasetConfig(
        process_function_config=ProcessFunctionConfig(
            node_features=node_features_with_level_nodes(),
            edge_attributes=edge_attributes_with_level_nodes(
                only_node2level=True
            ),
            edge_types=list(
                edge_attributes_with_level_nodes(only_node2level=True).keys()
            ),
        ),
        csv_metadata_config=CSVMetadataConfig(
            csv_filenames=["489_10k_nwk.csv"],
            processed_filename="489_10k_level_nodes_no_level2node",
        ),
    ),
    model=PhyloGNNClassifierConfig(
        encoder=FeatureEncoderConfig(
            node_output_dims={
                NodeNames.LEVEL.value: 144,
                NodeNames.NODE.value: 42,
            },
            edge_output_dims=8,
        ),
        message_passing=MessagePassingConfig(
            parameters={
                "layer_norm": False,
                "dropout": 0.0,
                "mlp_activation": "elu",
            },
            node_output_dims={
                NodeNames.LEVEL.value: 144,
                NodeNames.NODE.value: 42,
            },
            edge_output_dims=8,
        ),
        readout=ReadoutConfig(
            parameters={
                "node_types_to_use": [
                    NodeNames.LEVEL.value,
                    NodeNames.NODE.value,
                ],
                "edge_attributes_to_use": [
                    (
                        NodeNames.NODE.value,
                        EdgeNames.HAS_PARENT.value,
                        NodeNames.NODE.value,
                    )
                ],
                "aggregator": "all",
                "dropout": 0.1,
            },
        ),
        scheduler=None,
        learning_rate=0.0003,
        weight_decay=0.00005,
    ),
)
