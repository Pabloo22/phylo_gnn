from phylo_gnn.config import (
    Config,
    TrainingConfig,
    PhyloCSVDatasetConfig,
    CSVMetadataConfig,
)


EXPERIMENT_1 = Config(
    training_config=TrainingConfig(run_name="experiment_1"),
    dataset=PhyloCSVDatasetConfig(
        csv_metadata_config=CSVMetadataConfig(
            csv_filenames=["87_10k_nwk_subset_1000.csv"],
            processed_filename="87_10k_nwk_subset_1000",
        ),
        force_reload=False,
    ),
)
