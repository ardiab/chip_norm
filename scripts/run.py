import hydra
from omegaconf import DictConfig, OmegaConf

from chipvi.utils.path_helper import PathHelper

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for running ChipVI experiments."""
    print("Configuration loaded successfully:")
    print(OmegaConf.to_yaml(cfg))

    # Instantiate the path helper with the loaded config
    paths = PathHelper(cfg)

    # Example of using the new PathHelper
    print("\nResolved raw data path:")
    print(paths.raw_data_dir)
    
    # TODO: The rest of the training logic will be added here in later tasks.

if __name__ == "__main__":
    main()