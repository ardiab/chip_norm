import hydra
from omegaconf import DictConfig

from chipvi.data.preprocessing import run_preprocessing

@hydra.main(config_path="../configs/data", config_name="h3k27me3_preprocess", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Script to run the data pre-processing pipeline.
    """
    print("--- Running Data Pre-processing ---")
    run_preprocessing(cfg)
    print("--- Pre-processing Finished ---")

if __name__ == "__main__":
    main()