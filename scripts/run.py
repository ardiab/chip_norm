import hydra
import torch
import numpy as np
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

# Import our refactored components
from chipvi.data.datasets import MultiReplicateDataset, SingleReplicateDataset
from chipvi.training.trainer import Trainer
from chipvi.training.losses import (
    replicate_concordance_mse_loss,
    concordance_loss_nce_wrapper,
    negative_pearson_loss_wrapper,
    quantile_absolute_loss_wrapper,
    CompositeLoss,
    nll_loss
)
from chipvi.utils.path_helper import PathHelper
from chipvi.utils.config_validation import validate_config, ConfigValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)

# A simple factory to get the loss function from the config string
LOSS_REGISTRY = {
    "replicate_concordance_mse": replicate_concordance_mse_loss,
    "concordance_loss_nce": concordance_loss_nce_wrapper,
    "negative_pearson_loss": negative_pearson_loss_wrapper, 
    "quantile_absolute_loss": quantile_absolute_loss_wrapper,
    "nll_loss": nll_loss,
    # Add other losses here as they are created
}


class LogTransformWrapper(torch.utils.data.Dataset):
    """Dataset wrapper that applies log transformation to specified covariate columns."""
    
    def __init__(self, dataset: torch.utils.data.Dataset, columns_to_transform: list[int]):
        """Initialize wrapper with columns to transform.
        
        Args:
            dataset: Original dataset to wrap
            columns_to_transform: List of column indices to apply log transformation
        """
        self.dataset = dataset
        self.columns_to_transform = columns_to_transform
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict:
        """Get item with log transformation applied to specified columns."""
        item = self.dataset[idx]
        
        # Apply log transformation based on dataset structure
        if 'r1' in item and 'r2' in item:
            # MultiReplicateDataset structure
            for replicate in ['r1', 'r2']:
                covariates = item[replicate]['covariates'].clone()
                for col_idx in self.columns_to_transform:
                    if col_idx < len(covariates):
                        # Apply log(1 + x) transformation to avoid log(0)
                        covariates[col_idx] = torch.log1p(covariates[col_idx])
                item[replicate]['covariates'] = covariates
        else:
            # SingleReplicateDataset structure
            if 'covariates' in item:
                covariates = item['covariates'].clone()
                for col_idx in self.columns_to_transform:
                    if col_idx < len(covariates):
                        # Apply log(1 + x) transformation to avoid log(0)
                        covariates[col_idx] = torch.log1p(covariates[col_idx])
                item['covariates'] = covariates
        
        return item
    
    def get_covariate_dim(self) -> int:
        """Forward covariate dimension from wrapped dataset."""
        return self.dataset.get_covariate_dim()


def apply_preprocessing(dataset: torch.utils.data.Dataset, preprocessing_config: DictConfig) -> torch.utils.data.Dataset:
    """Apply preprocessing transformations to dataset based on config.
    
    Args:
        dataset: Dataset to preprocess
        preprocessing_config: Preprocessing configuration
        
    Returns:
        Preprocessed dataset with transformations applied
    """
    if not hasattr(preprocessing_config, 'log_transform'):
        return dataset
        
    log_config = preprocessing_config.log_transform
    if log_config.get('enabled', False):
        columns = log_config.get('columns', [])
        if columns:
            logging.info(f"Applying log transformation to covariate columns: {columns}")
            dataset = LogTransformWrapper(dataset, columns)
        else:
            logging.warning("Log transform enabled but no columns specified")
    
    return dataset


def parse_checkpoint_strategies(checkpoint_config: DictConfig) -> list:
    """Parse checkpoint strategies from configuration.
    
    Args:
        checkpoint_config: Checkpoint configuration
        
    Returns:
        List of checkpoint strategy dictionaries
    """
    if not hasattr(checkpoint_config, 'strategies'):
        return []
    
    strategies = []
    for strategy in checkpoint_config.strategies:
        strategies.append({
            'metric': strategy.metric,
            'mode': strategy.mode,
            'filename': strategy.filename,
            'overwrite': strategy.get('overwrite', True)
        })
    
    return strategies


def setup_wandb_config(wandb_config: DictConfig) -> dict:
    """Setup Weights & Biases configuration.
    
    Args:
        wandb_config: W&B configuration
        
    Returns:
        Dictionary with W&B settings
    """
    config = {
        'enabled': wandb_config.get('enabled', True),
        'project': wandb_config.get('project', 'chipvi'),
        'entity': wandb_config.get('entity', None),
        'tags': wandb_config.get('tags', []),
        'notes': wandb_config.get('notes', None)
    }
    
    logging.info(f"W&B configuration: enabled={config['enabled']}, project={config['project']}")
    return config


def create_scheduler_config(scheduler_config: DictConfig) -> dict:
    """Create scheduler configuration for trainer.
    
    Args:
        scheduler_config: Scheduler configuration
        
    Returns:
        Dictionary with scheduler settings
    """
    if not scheduler_config:
        return {}
        
    config = {
        'warmup_epochs': scheduler_config.get('warmup_epochs', 0),
        'scheduler_type': scheduler_config.get('scheduler_type', 'cosine'),
        'total_epochs': scheduler_config.get('total_epochs', 100)
    }
    
    # Validate configuration
    if config['warmup_epochs'] >= config['total_epochs']:
        raise ValueError(f"Warmup epochs ({config['warmup_epochs']}) must be less than total epochs ({config['total_epochs']})")
    
    logging.info(f"Scheduler configuration: {config}")
    return config


def create_composite_loss(loss_config):
    """Factory function to create composite loss from config.
    
    Args:
        loss_config: Either a string (single loss), dict with 'losses' and 'weights',
                    or Hydra DictConfig with _target_ for instantiation
        
    Returns:
        Loss function or CompositeLoss instance
        
    Raises:
        ValueError: If loss configuration format is invalid or unsupported
        KeyError: If referenced loss functions are not found in registry
    """
    if loss_config is None:
        raise ValueError("Loss configuration cannot be None")
    
    # Handle Hydra configuration with _target_
    if isinstance(loss_config, DictConfig) and '_target_' in loss_config:
        try:
            loss_instance = hydra.utils.instantiate(loss_config)
            logging.info(f"Successfully instantiated loss via Hydra: {loss_config._target_}")
            return loss_instance
        except Exception as e:
            raise ValueError(f"Failed to instantiate loss '{loss_config._target_}' via Hydra. "
                           f"Error: {e}. Please check your loss configuration format and ensure "
                           f"all required parameters are provided.") from e
    
    # Handle single loss string
    if isinstance(loss_config, str):
        if loss_config not in LOSS_REGISTRY:
            available_losses = list(LOSS_REGISTRY.keys())
            raise KeyError(f"Loss function '{loss_config}' not found in registry. "
                          f"Available losses: {available_losses}")
        return LOSS_REGISTRY[loss_config]
    
    # Handle composite loss configuration (old format)
    if isinstance(loss_config, (dict, DictConfig)) and 'losses' in loss_config:
        loss_names = loss_config['losses']
        weights = loss_config.get('weights', [1.0] * len(loss_names))
        
        if not loss_names:
            raise ValueError("Composite loss configuration must specify at least one loss function")
        
        if len(loss_names) != len(weights):
            raise ValueError(f"Number of losses ({len(loss_names)}) must match number of weights ({len(weights)}). "
                           f"Losses: {loss_names}, Weights: {weights}")
        
        # Validate all loss names exist in registry before creating functions
        missing_losses = [name for name in loss_names if name not in LOSS_REGISTRY]
        if missing_losses:
            available_losses = list(LOSS_REGISTRY.keys())
            raise KeyError(f"Loss functions not found in registry: {missing_losses}. "
                          f"Available losses: {available_losses}")
        
        # Get loss functions from registry
        loss_functions = [LOSS_REGISTRY[name] for name in loss_names]
        
        logging.info(f"Created composite loss with functions: {loss_names}, weights: {weights}")
        return CompositeLoss(loss_functions, weights, loss_names)
    
    # Handle DictConfig that might be a single loss reference
    if isinstance(loss_config, DictConfig):
        # Try to extract string representation
        if hasattr(loss_config, '_content') and isinstance(loss_config._content, str):
            loss_name = loss_config._content
        else:
            loss_name = str(loss_config)
        
        if loss_name not in LOSS_REGISTRY:
            available_losses = list(LOSS_REGISTRY.keys())
            raise KeyError(f"Loss function '{loss_name}' not found in registry. "
                          f"Available losses: {available_losses}")
        return LOSS_REGISTRY[loss_name]
    
    # No valid configuration format found
    raise ValueError(f"Unsupported loss configuration format. Expected: "
                    f"string (single loss name), dict with 'losses' key (composite), "
                    f"or DictConfig with '_target_' (Hydra). Got: {type(loss_config)} with value: {loss_config}")

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for running ChipVI experiments using Hydra.
    """
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))

    # 0. Configuration Validation
    try:
        validate_config(cfg, strict=False)  # Use non-strict mode to allow warnings
        logging.info("Configuration validation completed")
    except ConfigValidationError as e:
        logging.error(f"Configuration validation failed: {e}")
        raise

    # 1. Setup: Device, Paths, etc.
    device = torch.device(cfg.get('device', 'cpu') if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 2. Configuration Processing
    # Setup W&B configuration
    wandb_config = setup_wandb_config(cfg.get('wandb', {}))
    
    # Setup scheduler configuration
    scheduler_config = create_scheduler_config(cfg.get('scheduler', {}))
    
    # Parse checkpoint strategies
    checkpoint_strategies = parse_checkpoint_strategies(cfg.get('checkpointing', {}))
    
    # 3. Data Loading
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
    
    # Apply preprocessing based on config
    if hasattr(cfg, 'preprocessing'):
        train_ds = apply_preprocessing(train_ds, cfg.preprocessing)
        val_ds = apply_preprocessing(val_ds, cfg.preprocessing)
    
    # Create data loaders with batch size from training config or fallback to top-level
    batch_size = cfg.get('training', {}).get('batch_size', cfg.get('batch_size', 8192))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 4. Model Initialization
    # Hydra instantiates the model for us using the _target_ field
    model = hydra.utils.instantiate(
        cfg.model,
        covariate_dim=train_ds.get_covariate_dim()
    ).to(device)

    # 5. Optimizer and Loss Function
    # Get training parameters from new structure or fallback to old
    training_cfg = cfg.get('training', cfg)  # Use training section if available, else use root
    learning_rate = training_cfg.get('learning_rate', 0.001)
    weight_decay = training_cfg.get('weight_decay', 0.01)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Handle both old and new loss configuration formats
    loss_config = cfg.get('loss', cfg.get('loss_fn', 'replicate_concordance_mse'))
    loss_fn = create_composite_loss(loss_config)

    # 6. Trainer Initialization and Execution
    trainer_kwargs = {
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'optimizer': optimizer,
        'loss_fn': loss_fn,
        'device': device,
    }
    
    # Add new configuration options if available
    if scheduler_config:
        trainer_kwargs['scheduler_config'] = scheduler_config
        
    if checkpoint_strategies:
        trainer_kwargs['checkpoint_strategies'] = checkpoint_strategies
        
    if wandb_config and wandb_config['enabled']:
        trainer_kwargs['wandb_config'] = wandb_config
        
    # Add early stopping configuration if available
    if hasattr(cfg, 'early_stopping'):
        trainer_kwargs['early_stopping_config'] = {
            'patience': cfg.early_stopping.get('patience', 10),
            'monitor': cfg.early_stopping.get('monitor', 'val_loss'),
            'mode': cfg.early_stopping.get('mode', 'min')
        }
        
    # Add gradient clipping configuration if available
    if hasattr(cfg, 'gradient_clipping'):
        trainer_kwargs['gradient_clipping_config'] = {
            'max_norm': cfg.gradient_clipping.get('max_norm', 1.0)
        }
    
    trainer = Trainer(**trainer_kwargs)
    
    # Get number of epochs from new structure or fallback to old
    num_epochs = training_cfg.get('num_epochs', cfg.get('num_epochs', 100))
    
    print(f"\n--- Starting Training for {num_epochs} epochs ---")
    trainer.fit(num_epochs=num_epochs)
    print("--- Training Finished ---")

if __name__ == "__main__":
    main()