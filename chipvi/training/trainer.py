"""Unified Trainer class for ChipVI training."""

from typing import Callable, Optional, Dict, Any, List
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chipvi.utils.distributions import get_torch_nb_dist, compute_numeric_cdf
from chipvi.training.checkpoint_manager import CheckpointManager

# Check if we're in test mode and use mock wandb if so
if os.environ.get('CHIPVI_TEST_MODE') == '1':
    # Create a mock wandb module for testing
    class MockWandB:
        def init(self, **kwargs):
            return None
        
        def log(self, metrics):
            return None
        
        def finish(self):
            return None
        
        def Image(self, fig):
            return None
    
    wandb = MockWandB()
else:
    import wandb


class Trainer:
    """Unified trainer class that handles both training and validation loops."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: Callable[[dict, dict], torch.Tensor],
        device: torch.device,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: PyTorch optimizer
            loss_fn: Loss function that takes (model_outputs, batch) and returns loss tensor
            device: Device to run training on
            config: Optional configuration dictionary for enhanced features
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        # Parse configuration or set defaults
        self._parse_config(config or {})
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize scheduler (will be set in fit method once we know total epochs)
        self.scheduler = None
        
        # Initialize checkpoint manager if configured
        self.checkpoint_manager = None
        
        # Cache for loss components from the most recent loss computation
        self.val_component_losses = None
    
    def _parse_config(self, config: Dict[str, Any]) -> None:
        """Parse configuration dictionary and set training parameters."""
        # Scheduler configuration
        scheduler_config = config.get('scheduler_config', {})
        self.warmup_epochs = scheduler_config.get('warmup_epochs', 0)
        self.scheduler_type = scheduler_config.get('scheduler_type', 'cosine')
        
        # Early stopping configuration
        early_stopping_config = config.get('early_stopping_config', {})
        self.patience = early_stopping_config.get('patience', None)
        self.monitor_metric = early_stopping_config.get('monitor_metric', 'val_loss')
        
        # Gradient clipping configuration
        self.max_grad_norm = config.get('max_grad_norm', None)
        
        # Weights & Biases configuration
        wandb_config = config.get('wandb_config', {})
        self.wandb_enabled = wandb_config.get('enabled', True)
        self.wandb_project = wandb_config.get('project', 'chipvi_v2')
        self.wandb_entity = wandb_config.get('entity', None)
        self.wandb_name = wandb_config.get('name', None)
        
        # Initialize early stopping state
        if self.patience is not None:
            self.best_val_loss = float('inf')
            self.epochs_without_improvement = 0
            self.best_model_state = None
        
        # Checkpoint configuration
        checkpoint_config = config.get('checkpoint_config', {})
        if checkpoint_config.get('enabled', False):
            output_dir = checkpoint_config['output_dir']
            checkpoint_configs = checkpoint_config.get('strategies', [])
            
            # Initialize checkpoint manager (validation handled internally)
            if checkpoint_configs:
                self.checkpoint_manager = CheckpointManager(output_dir, checkpoint_configs)
            else:
                # Set default checkpoint strategy if none provided
                default_config = [{
                    'metric_name': 'val_loss',
                    'mode': 'min',
                    'filename': 'best_model.pt',
                    'overwrite': True
                }]
                self.checkpoint_manager = CheckpointManager(output_dir, default_config)
    
    def _setup_schedulers(self, total_epochs: int) -> None:
        """Set up learning rate schedulers."""
        if self.warmup_epochs <= 0:
            # No warmup, just cosine annealing
            if self.scheduler_type == 'cosine':
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, 
                    T_max=total_epochs
                )
            return  # No scheduler at all, just optimizer.step()
        
        # Create warmup scheduler: LinearLR with start_factor=1e-3, end_factor=1.0
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=self.warmup_epochs
        )
        
        # Create main scheduler: CosineAnnealingLR
        if self.scheduler_type == 'cosine':
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs - self.warmup_epochs
            )
        else:
            # Default to constant LR after warmup if scheduler_type is not recognized
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=1.0,
                total_iters=total_epochs - self.warmup_epochs
            )
        
        # Combine schedulers using SequentialLR
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.warmup_epochs]
        )
    
    def fit(self, num_epochs: int):
        """Run the training loop for the specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for
        """
        # Set up learning rate schedulers
        self._setup_schedulers(num_epochs)
        
        # Initialize Weights & Biases if enabled
        self._init_wandb()
        
        try:
            for epoch in range(num_epochs):
                train_loss = self._train_one_epoch(epoch)
                val_loss = self._validate_one_epoch(epoch)
                
                # Step the scheduler after each epoch
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Log epoch metrics
                epoch_metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'epoch': epoch,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                self._log_epoch_metrics(epoch_metrics)
                
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {epoch_metrics['learning_rate']:.2e}")
                
                # Check early stopping
                if self._check_early_stopping(val_loss):
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
                    
        finally:
            # Log checkpoint summary if checkpoint manager is active
            if self.checkpoint_manager is not None:
                self._log_checkpoint_summary()
            
            # Clean up W&B
            self._cleanup_wandb()
    
    def _train_one_epoch(self, epoch: int) -> float:
        """Train the model for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            # Move batch tensors to device
            batch = self._move_batch_to_device(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass - call model for each replicate separately
            x_r1 = batch['r1']['covariates']
            x_r2 = batch['r2']['covariates']
            
            # Predict distribution parameters for each replicate
            mu_r1, r_r1 = self.model(x_r1)
            mu_r2, r_r2 = self.model(x_r2)
            
            # Package model outputs for the loss function
            model_outputs = {
                'r1': {'mu': mu_r1, 'r': r_r1},
                'r2': {'mu': mu_r2, 'r': r_r2},
            }
            
            # Calculate loss
            loss_result = self.loss_fn(model_outputs, batch)
            
            # Handle loss components if available
            if isinstance(loss_result, dict) and 'total' in loss_result and 'components' in loss_result:
                loss = loss_result['total']
                self.val_component_losses = loss_result['components']
            else:
                loss = loss_result
                self.val_component_losses = None
            
            # Backward pass
            loss.backward()
            
            # Clip gradients if specified
            grad_norm = self._clip_gradients()
            
            # Update weights
            self.optimizer.step()
            
            # Log batch metrics
            batch_metrics = {
                'train_loss': loss.item(),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'grad_norm': grad_norm
            }
            self._log_batch_metrics(batch_metrics)
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_one_epoch(self, epoch: int) -> float:
        """Validate the model for one epoch.
        
        Returns:
            Average validation loss for the epoch
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Initialize collectors for validation metrics
        residuals_r1 = []
        residuals_r2 = []
        quantiles_r1 = []
        quantiles_r2 = []
        predictions_r1 = []
        predictions_r2 = []
        observations_r1 = []
        observations_r2 = []
        
        # Initialize collectors for loss components
        component_totals = {}
        num_component_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch tensors to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                x_r1 = batch['r1']['covariates']
                x_r2 = batch['r2']['covariates']
                
                # Predict distribution parameters for each replicate
                mu_r1, r_r1 = self.model(x_r1)
                mu_r2, r_r2 = self.model(x_r2)
                
                # Package model outputs for the loss function
                model_outputs = {
                    'r1': {'mu': mu_r1, 'r': r_r1},
                    'r2': {'mu': mu_r2, 'r': r_r2},
                }
                
                # Calculate loss
                loss_result = self.loss_fn(model_outputs, batch)
                
                # Handle loss components if available
                if isinstance(loss_result, dict) and 'total' in loss_result and 'components' in loss_result:
                    loss = loss_result['total']
                    components = loss_result['components']
                    
                    # Initialize component_totals on first batch (now as dict)
                    if len(component_totals) == 0:
                        component_totals = {name: torch.tensor(0.0) for name in components.keys()}
                    
                    # Accumulate components by name
                    for name, component in components.items():
                        component_totals[name] += component.detach()
                    num_component_batches += 1
                else:
                    loss = loss_result
                
                # Collect validation metrics
                y_r1 = batch['r1']['reads']
                y_r2 = batch['r2']['reads']
                sd_ratio = batch['metadata']['sd_ratio']
                
                # Compute residuals
                res_r1 = y_r1 - mu_r1.squeeze()
                res_r2 = y_r2 - mu_r2.squeeze()
                res_r2_scaled = res_r2 * sd_ratio
                
                # Create distributions for quantile computation
                dist_r1 = get_torch_nb_dist(r=r_r1.squeeze(), mu=mu_r1.squeeze())
                dist_r2 = get_torch_nb_dist(r=r_r2.squeeze(), mu=mu_r2.squeeze())
                
                # Compute quantiles (PIT values)
                quant_r1 = compute_numeric_cdf(dist_r1, y_r1)
                quant_r2 = compute_numeric_cdf(dist_r2, y_r2)
                
                # Collect all metrics (move to CPU for storage)
                residuals_r1.extend(res_r1.cpu().numpy())
                residuals_r2.extend(res_r2_scaled.cpu().numpy())
                quantiles_r1.extend(quant_r1.cpu().numpy())
                quantiles_r2.extend(quant_r2.cpu().numpy())
                predictions_r1.extend(mu_r1.squeeze().cpu().numpy())
                predictions_r2.extend(mu_r2.squeeze().cpu().numpy())
                observations_r1.extend(y_r1.cpu().numpy())
                observations_r2.extend(y_r2.cpu().numpy())
                
                total_loss += loss.item()
                num_batches += 1
        
        # Compute averaged loss components if available
        if num_component_batches > 0:
            self.val_component_losses = {name: total / num_component_batches for name, total in component_totals.items()}
        else:
            self.val_component_losses = None
        
        # Compute validation metrics and create visualizations
        if num_batches > 0:
            validation_metrics = self._compute_validation_metrics(
                residuals_r1, residuals_r2, quantiles_r1, quantiles_r2,
                predictions_r1, predictions_r2, observations_r1, observations_r2
            )
            self._log_validation_metrics(validation_metrics, epoch)
            
            # Update checkpoint manager if configured
            if self.checkpoint_manager is not None:
                # Prepare metrics dictionary for checkpoint manager
                val_loss = total_loss / num_batches
                checkpoint_metrics = {
                    'val_loss': val_loss,
                    'val_residual_spearman': validation_metrics['val_residual_spearman'],
                    'val_quantile_spearman': validation_metrics['val_quantile_spearman']
                }
                
                # Add loss components if available
                if self.val_component_losses:
                    checkpoint_metrics['loss_components'] = {
                        name: float(component.item()) if hasattr(component, 'item') else float(component)
                        for name, component in self.val_component_losses.items()
                    }
                
                # Update checkpoints
                self.checkpoint_manager.update(
                    checkpoint_metrics, 
                    self.model.state_dict(), 
                    epoch
                )
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    
    def _log_checkpoint_summary(self) -> None:
        """Log summary of saved checkpoints."""
        if self.checkpoint_manager is None:
            return
        
        print("\n=== Checkpoint Summary ===")
        for config in self.checkpoint_manager.checkpoint_configs:
            metric_name = config['metric_name']
            filename = config['filename']
            best_value = self.checkpoint_manager.best_values.get(metric_name)
            
            if best_value is not None:
                print(f"Best {metric_name}: {best_value:.6f} -> {filename}")
            else:
                print(f"No improvement found for {metric_name}")
        print("==========================\n")
    
    def _move_batch_to_device(self, batch: dict) -> dict:
        """Move all tensors in a batch dictionary to the specified device.
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            Dictionary with all tensors moved to device
        """
        # Move tensors in the specific batch structure we expect
        return {
            'r1': {
                'covariates': batch['r1']['covariates'].to(self.device),
                'reads': batch['r1']['reads'].to(self.device),
            },
            'r2': {
                'covariates': batch['r2']['covariates'].to(self.device),
                'reads': batch['r2']['reads'].to(self.device),
            },
            'metadata': {
                'sd_ratio': batch['metadata']['sd_ratio'].to(self.device),
                'grp_idx': batch['metadata']['grp_idx'].to(self.device),
            }
        }
    
    def _clip_gradients(self) -> float:
        """Clip gradients if max_grad_norm is specified.
        
        Returns:
            Gradient norm before clipping
        """
        if self.max_grad_norm is None:
            # Return gradient norm without clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
            return float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
        
        # Clip gradients and return norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        return float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
    
    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging if enabled."""
        if not self.wandb_enabled:
            return
            
        wandb_init_kwargs = {
            'project': self.wandb_project,
        }
        
        if self.wandb_entity:
            wandb_init_kwargs['entity'] = self.wandb_entity
        if self.wandb_name:
            wandb_init_kwargs['name'] = self.wandb_name
            
        wandb.init(**wandb_init_kwargs)
    
    def _log_batch_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log per-batch metrics to W&B."""
        if not self.wandb_enabled:
            return
            
        wandb.log(metrics)
    
    def _log_epoch_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log per-epoch metrics to W&B."""
        if not self.wandb_enabled:
            return
            
        wandb.log(metrics)
    
    def _cleanup_wandb(self) -> None:
        """Clean up W&B logging."""
        if not self.wandb_enabled:
            return
            
        wandb.finish()
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping should be triggered.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.patience is None:
            return False
        
        # Check if validation loss improved
        if val_loss < self.best_val_loss:
            # Validation loss improved
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            self._save_best_checkpoint()
            return False
        else:
            # Validation loss did not improve
            self.epochs_without_improvement += 1
            return self.epochs_without_improvement >= self.patience
    
    def _save_best_checkpoint(self) -> None:
        """Save the current best model state."""
        self.best_model_state = copy.deepcopy(self.model.state_dict())
    
    def _compute_validation_metrics(
        self, 
        residuals_r1: list, 
        residuals_r2: list, 
        quantiles_r1: list, 
        quantiles_r2: list,
        predictions_r1: list,
        predictions_r2: list, 
        observations_r1: list,
        observations_r2: list
    ) -> Dict[str, Any]:
        """Compute Spearman correlations and prepare metrics for visualization.
        
        Args:
            residuals_r1: List of residuals for replicate 1
            residuals_r2: List of scaled residuals for replicate 2
            quantiles_r1: List of quantiles (PIT values) for replicate 1
            quantiles_r2: List of quantiles (PIT values) for replicate 2
            predictions_r1: List of predicted means for replicate 1
            predictions_r2: List of predicted means for replicate 2
            observations_r1: List of observed values for replicate 1
            observations_r2: List of observed values for replicate 2
            
        Returns:
            Dictionary containing validation metrics and data for plotting
        """
        # Handle empty batches gracefully
        if len(residuals_r1) == 0 or len(residuals_r2) == 0:
            return {
                'val_residual_spearman': 0.0,
                'val_quantile_spearman': 0.0,
                'metrics_dict': None
            }
        
        # Concatenate all batch results into numpy arrays
        residuals_r1 = np.array(residuals_r1)
        residuals_r2 = np.array(residuals_r2)
        quantiles_r1 = np.array(quantiles_r1)
        quantiles_r2 = np.array(quantiles_r2)
        predictions_r1 = np.array(predictions_r1)
        predictions_r2 = np.array(predictions_r2)
        observations_r1 = np.array(observations_r1)
        observations_r2 = np.array(observations_r2)
        
        # Create pandas DataFrame with all metrics
        df = pd.DataFrame({
            'r1_res': residuals_r1,
            'r2_res_scaled': residuals_r2,
            'r1_quant': quantiles_r1,
            'r2_quant': quantiles_r2
        })
        
        # Remove NaN values for correlation computation
        df_clean = df.dropna()
        
        if len(df_clean) == 0:
            return {
                'val_residual_spearman': 0.0,
                'val_quantile_spearman': 0.0,
                'metrics_dict': None
            }
        
        # Use df.corr(method='spearman') to compute correlation matrix
        corr_matrix = df_clean.corr(method='spearman')
        
        # Extract correlations
        residual_corr = corr_matrix.loc['r1_res', 'r2_res_scaled'] if len(df_clean) > 1 else 0.0
        quantile_corr = corr_matrix.loc['r1_quant', 'r2_quant'] if len(df_clean) > 1 else 0.0
        
        # Handle NaN correlations (can happen with constant values)
        residual_corr = 0.0 if pd.isna(residual_corr) else residual_corr
        quantile_corr = 0.0 if pd.isna(quantile_corr) else quantile_corr
        
        # Prepare data for plotting
        metrics_dict = {
            'r1_pred': predictions_r1,
            'r1_obs': observations_r1,
            'r2_pred': predictions_r2,
            'r2_obs': observations_r2,
            'r1_res': residuals_r1,
            'r2_res_scaled': residuals_r2,
            'r1_quant': quantiles_r1,
            'r2_quant': quantiles_r2,
        }
        
        return {
            'val_residual_spearman': residual_corr,
            'val_quantile_spearman': quantile_corr,
            'metrics_dict': metrics_dict
        }
    
    def _log_validation_metrics(self, validation_metrics: Dict[str, Any], epoch: int) -> None:
        """Log validation metrics and create visualization plots.
        
        Args:
            validation_metrics: Dictionary containing computed metrics
            epoch: Current epoch number
        """
        # Log correlation metrics
        corr_metrics = {
            'val_residual_spearman': validation_metrics['val_residual_spearman'],
            'val_quantile_spearman': validation_metrics['val_quantile_spearman'],
        }
        self._log_epoch_metrics(corr_metrics)
        
        # Create and log validation plots if we have data
        metrics_dict = validation_metrics['metrics_dict']
        if metrics_dict is not None and self.wandb_enabled:
            try:
                fig = create_validation_figure(metrics_dict)
                wandb.log({"validation_plots": wandb.Image(fig)})
                plt.close(fig)  # Free memory
            except Exception as e:
                print(f"Warning: Could not create validation plots: {e}")


def create_validation_figure(metrics_dict: Dict[str, np.ndarray]) -> plt.Figure:
    """Create comprehensive validation figure with 9 panels.
    
    Args:
        metrics_dict: Dictionary containing validation metrics with keys:
            - r1_pred, r2_pred: Predicted means
            - r1_obs, r2_obs: Observed values
            - r1_res, r2_res_scaled: Residuals
            - r1_quant, r2_quant: Quantiles (PIT values)
    
    Returns:
        Matplotlib figure with 3x3 subplot grid
    """
    from chipvi.utils.plots import hist2d
    
    # Create 3x3 subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # Convert metrics to DataFrame for plotting
    df = pd.DataFrame(metrics_dict)
    
    # Row 1: Predictions vs Observations
    # (0,0): R1 predicted mean vs observed
    hist2d(df, 'r1_obs', 'r1_pred', axes[0, 0])
    axes[0, 0].set_title('R1: Observed vs Predicted')
    axes[0, 0].set_xlabel('Observed')
    axes[0, 0].set_ylabel('Predicted')
    
    # (0,1): R2 predicted mean vs observed
    hist2d(df, 'r2_obs', 'r2_pred', axes[0, 1])
    axes[0, 1].set_title('R2: Observed vs Predicted')
    axes[0, 1].set_xlabel('Observed')
    axes[0, 1].set_ylabel('Predicted')
    
    # (0,2): R1 vs R2 predicted means consistency
    hist2d(df, 'r1_pred', 'r2_pred', axes[0, 2])
    axes[0, 2].set_title('Predicted Mean Consistency')
    axes[0, 2].set_xlabel('R1 Predicted')
    axes[0, 2].set_ylabel('R2 Predicted')
    
    # Row 2: Residual Analysis
    # (1,0): R1 residuals histogram
    axes[1, 0].hist(df['r1_res'], bins=50, range=(-25, 25), alpha=0.7, density=True)
    axes[1, 0].set_title('R1 Residuals Distribution')
    axes[1, 0].set_xlabel('Residual')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    # (1,1): R2 scaled residuals histogram
    axes[1, 1].hist(df['r2_res_scaled'], bins=50, range=(-25, 25), alpha=0.7, density=True)
    axes[1, 1].set_title('R2 Scaled Residuals Distribution')
    axes[1, 1].set_xlabel('Scaled Residual')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    # (1,2): R1 vs R2 residual scatter plot with correlation
    hist2d(df, 'r1_res', 'r2_res_scaled', axes[1, 2])
    axes[1, 2].set_title('Residual Consistency')
    axes[1, 2].set_xlabel('R1 Residual')
    axes[1, 2].set_ylabel('R2 Scaled Residual')
    
    # Row 3: Quantile Analysis (PIT)
    # (2,0): R1 quantiles histogram (should be uniform [0,1])
    axes[2, 0].hist(df['r1_quant'], bins=20, range=(0, 1), alpha=0.7, density=True)
    axes[2, 0].axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Uniform expectation')
    axes[2, 0].set_title('R1 PIT Distribution')
    axes[2, 0].set_xlabel('Quantile')
    axes[2, 0].set_ylabel('Density')
    axes[2, 0].legend()
    
    # (2,1): R2 quantiles histogram (should be uniform [0,1])
    axes[2, 1].hist(df['r2_quant'], bins=20, range=(0, 1), alpha=0.7, density=True)
    axes[2, 1].axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Uniform expectation')
    axes[2, 1].set_title('R2 PIT Distribution')
    axes[2, 1].set_xlabel('Quantile')
    axes[2, 1].set_ylabel('Density')
    axes[2, 1].legend()
    
    # (2,2): R1 vs R2 quantile consistency plot
    hist2d(df, 'r1_quant', 'r2_quant', axes[2, 2])
    axes[2, 2].set_title('PIT Consistency')
    axes[2, 2].set_xlabel('R1 Quantile')
    axes[2, 2].set_ylabel('R2 Quantile')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig