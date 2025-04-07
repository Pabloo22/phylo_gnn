from phylo_gnn.config import (
    Config,
    TrainingConfig,
    PhyloCSVDatasetConfig,
    CSVMetadataConfig,
    PhyloGNNClassifierConfig,
    MessagePassingConfig,
)


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
    training_config=TrainingConfig(run_name="exp_1", patience=25),
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
