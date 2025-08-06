"""Unified Trainer class for ChipVI training."""

from typing import Callable, Optional, Dict, Any
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import copy


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
            loss = self.loss_fn(model_outputs, batch)
            
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
                loss = self.loss_fn(model_outputs, batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
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