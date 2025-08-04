"""Unified Trainer class for ChipVI training."""

from typing import Callable
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


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
    ):
        """Initialize the Trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: PyTorch optimizer
            loss_fn: Loss function that takes (model_outputs, batch) and returns loss tensor
            device: Device to run training on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
    
    def fit(self, num_epochs: int):
        """Run the training loop for the specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for
        """
        for epoch in range(num_epochs):
            train_loss = self._train_one_epoch()
            val_loss = self._validate_one_epoch()
            
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    def _train_one_epoch(self) -> float:
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
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_one_epoch(self) -> float:
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