"""Loss functions for ChipVI training."""

import torch
from torch import nn


def compute_residual_mse(
    y_r1: torch.Tensor,
    mu_tech_r1: torch.Tensor,
    y_r2: torch.Tensor,
    mu_tech_r2: torch.Tensor,
    sd_ratio_r1_to_r2: torch.Tensor,
) -> torch.Tensor:
    """Compute the scaled residual MSE between two replicates.
    
    Args:
        y_r1: Observed reads for replicate 1
        mu_tech_r1: Predicted technical mean for replicate 1
        y_r2: Observed reads for replicate 2
        mu_tech_r2: Predicted technical mean for replicate 2
        sd_ratio_r1_to_r2: Sequencing depth ratio from replicate 1 to replicate 2
        
    Returns:
        MSE loss tensor
    """
    # Calculate residuals
    r1_residual = y_r1 - mu_tech_r1
    r2_residual_unscaled = y_r2 - mu_tech_r2
    r2_residual_scaled = sd_ratio_r1_to_r2 * r2_residual_unscaled
    
    # Calculate MSE
    mse = torch.mean((r1_residual - r2_residual_scaled) ** 2)
    
    return mse


def replicate_concordance_mse_loss(model_outputs: dict, batch: dict) -> torch.Tensor:
    """Compute replicate concordance MSE loss from model outputs and batch.
    
    Args:
        model_outputs: Dictionary with structure {'r1': {'mu': ..., 'r': ...}, 'r2': {...}}
        batch: Dictionary from MultiReplicateDataset with replicate data and metadata
        
    Returns:
        MSE loss tensor
    """
    # Extract observed reads from batch
    y_r1 = batch['r1']['reads']
    y_r2 = batch['r2']['reads']
    sd_ratio_r1_to_r2 = batch['metadata']['sd_ratio']
    
    # Extract predicted means from model outputs
    mu_tech_r1 = model_outputs['r1']['mu']
    mu_tech_r2 = model_outputs['r2']['mu']
    
    # Use the compute_residual_mse function
    return compute_residual_mse(y_r1, mu_tech_r1, y_r2, mu_tech_r2, sd_ratio_r1_to_r2)