from phylo_gnn.config.experiment_configs import DEBUG_2
from phylo_gnn.config import Config
from phylo_gnn.training import train


def main(config: Config) -> None:
    dataset = config.initialize_dataset()
    model = config.initialize_model()
    results = train(
        model=model,
        dataset=dataset,
        training_config=config.training_config,
    )
    # Results include test metrics and best model path
    print(f"Test accuracy: {results['test_results']['test/acc']}")
    print(f"Best model saved at: {results['best_model_path']}")


if __name__ == "__main__":
    main(DEBUG_2)
