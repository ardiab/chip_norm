import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Import our refactored components
from chipvi.data.datasets import MultiReplicateDataset, SingleReplicateDataset
from chipvi.training.trainer import Trainer
from chipvi.training.losses import (
    replicate_concordance_mse_loss,
    concordance_loss_nce_wrapper,
    negative_pearson_loss_wrapper,
    quantile_absolute_loss_wrapper,
    CompositeLoss
)
from chipvi.utils.path_helper import PathHelper

# A simple factory to get the loss function from the config string
LOSS_REGISTRY = {
    "replicate_concordance_mse": replicate_concordance_mse_loss,
    "concordance_loss_nce": concordance_loss_nce_wrapper,
    "negative_pearson_loss": negative_pearson_loss_wrapper, 
    "quantile_absolute_loss": quantile_absolute_loss_wrapper,
    # Add other losses here as they are created
}


def create_composite_loss(loss_config):
    """Factory function to create composite loss from config.
    
    Args:
        loss_config: Either a string (single loss) or dict with 'losses' and 'weights'
        
    Returns:
        Loss function or CompositeLoss instance
    """
    # Handle backward compatibility - single loss string
    if isinstance(loss_config, str):
        return LOSS_REGISTRY[loss_config]
    
    # Handle composite loss configuration
    if isinstance(loss_config, dict) and 'losses' in loss_config:
        loss_names = loss_config['losses']
        weights = loss_config.get('weights', [1.0] * len(loss_names))
        return_components = loss_config.get('return_components', False)
        
        # Get loss functions from registry
        loss_functions = [LOSS_REGISTRY[name] for name in loss_names]
        
        return CompositeLoss(loss_functions, weights, return_components)
    
    # Fallback: assume it's a single loss name
    return LOSS_REGISTRY[loss_config]

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for running ChipVI experiments using Hydra.
    """
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))

    # 1. Setup: Device, Paths, etc.
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    # Get processed data directory path
    paths = PathHelper(cfg)
    processed_data_dir = paths.data_processed
    
    # Construct dataset prefix paths
    train_prefix = processed_data_dir / cfg.output_prefix_train
    val_prefix = processed_data_dir / cfg.output_prefix_val
    
    # Instantiate datasets directly from preprocessed .npy files
    # Check if this is single replicate or multi-replicate data
    if (processed_data_dir / f"{cfg.output_prefix_train}_control_reads_r2.npy").exists():
        # Multi-replicate mode
        train_ds = MultiReplicateDataset(str(train_prefix))
        val_ds = MultiReplicateDataset(str(val_prefix))
    else:
        # Single replicate mode
        train_ds = SingleReplicateDataset(str(train_prefix))
        val_ds = SingleReplicateDataset(str(val_prefix))
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # 3. Model Initialization
    # Hydra instantiates the model for us using the _target_ field
    model = hydra.utils.instantiate(
        cfg.model,
        covariate_dim=train_ds.get_covariate_dim()
    ).to(device)

    # 4. Optimizer and Loss Function
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    loss_fn = create_composite_loss(cfg.loss_fn)

    # 5. Trainer Initialization and Execution
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )
    
    print("\n--- Starting Training ---")
    trainer.fit(num_epochs=cfg.num_epochs)
    print("--- Training Finished ---")

if __name__ == "__main__":
    main()