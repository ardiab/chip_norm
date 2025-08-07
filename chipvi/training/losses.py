"""Loss functions for ChipVI training."""

import torch
import torch.distributions
from torch import nn
from typing import List, Callable, Dict, Union

from chipvi.utils.distributions import compute_numeric_cdf, get_torch_nb_dist


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


def concordance_loss_nce(res1: torch.Tensor, res2: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Compute InfoNCE concordance loss between two residual tensors.
    
    Args:
        res1: Residual tensor from replicate 1, shape (batch_size,) or (batch_size, feature_dim)
        res2: Residual tensor from replicate 2, shape (batch_size,) or (batch_size, feature_dim)
        tau: Temperature parameter for similarity matrix
        
    Returns:
        InfoNCE loss as scalar tensor
    """
    eps = 1e-8  # For numerical stability
    
    # Flatten tensors if they are multi-dimensional
    if res1.dim() > 1:
        res1 = res1.flatten(start_dim=1)  # Shape: (batch_size, -1)
    if res2.dim() > 1:
        res2 = res2.flatten(start_dim=1)  # Shape: (batch_size, -1)
    
    # Ensure both tensors are 1D if they were originally 1D
    if res1.dim() == 1:
        res1 = res1.unsqueeze(-1)  # Shape: (batch_size, 1)
    if res2.dim() == 1:
        res2 = res2.unsqueeze(-1)  # Shape: (batch_size, 1)
    
    # Compute pairwise squared Euclidean distances
    # res1: (batch_size, feature_dim), res2: (batch_size, feature_dim)
    # Use broadcasting to compute all pairwise distances
    res1_expanded = res1.unsqueeze(1)  # (batch_size, 1, feature_dim)
    res2_expanded = res2.unsqueeze(0)  # (1, batch_size, feature_dim)
    
    # Compute squared differences: (batch_size, batch_size, feature_dim)
    squared_diffs = (res1_expanded - res2_expanded) ** 2
    
    # Sum over feature dimension to get distances: (batch_size, batch_size)
    distances = torch.sum(squared_diffs, dim=-1)
    
    # Compute similarity matrix using exp(-distances / tau)
    similarity_matrix = torch.exp(-distances / tau)
    
    # Extract diagonal elements (positive pairs)
    positive_pairs = torch.diag(similarity_matrix)
    
    # Compute sum over each row (denominator for NCE)
    row_sums = torch.sum(similarity_matrix, dim=1)
    
    # Compute NCE loss: -log(positive / row_sum)
    nce_loss = -torch.log(positive_pairs / (row_sums + eps))
    
    # Return mean loss over batch
    return torch.mean(nce_loss)


def negative_pearson_loss(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute negative Pearson correlation as loss function.
    
    Args:
        preds: Predicted values
        targets: Target values
        eps: Epsilon for numerical stability
        
    Returns:
        Negative Pearson correlation coefficient
    """
    # Center both tensors by subtracting their means
    preds_centered = preds - torch.mean(preds)
    targets_centered = targets - torch.mean(targets)
    
    # Compute numerator: sum of element-wise products
    numerator = torch.sum(preds_centered * targets_centered)
    
    # Compute denominator: sqrt of sum of squared deviations
    preds_ss = torch.sum(preds_centered ** 2)
    targets_ss = torch.sum(targets_centered ** 2)
    denominator = torch.sqrt(preds_ss * targets_ss + eps)
    
    # Compute correlation coefficient
    correlation = numerator / denominator
    
    # Return negative correlation (to minimize for optimization)
    return -correlation


def quantile_absolute_loss(
    model_r1: Dict[str, torch.Tensor], 
    model_r2: Dict[str, torch.Tensor],
    y_r1: torch.Tensor, 
    y_r2: torch.Tensor
) -> torch.Tensor:
    """Compute quantile absolute difference loss using PIT values.
    
    Args:
        model_r1: Model outputs for replicate 1 (dict with 'mu' and 'r' keys)
        model_r2: Model outputs for replicate 2 (dict with 'mu' and 'r' keys)
        y_r1: Observed values for replicate 1
        y_r2: Observed values for replicate 2
        
    Returns:
        Mean absolute difference between quantiles
    """
    # Ensure tensors have at least 1 dimension for compute_numeric_cdf
    mu_r1 = torch.atleast_1d(model_r1['mu'])
    r_r1 = torch.atleast_1d(model_r1['r'])
    mu_r2 = torch.atleast_1d(model_r2['mu'])
    r_r2 = torch.atleast_1d(model_r2['r'])
    
    # Ensure y tensors have at least 1 dimension
    y_r1_flat = torch.atleast_1d(y_r1)
    y_r2_flat = torch.atleast_1d(y_r2)
    
    # Create distributions directly to maintain proper dimensions
    # Convert mu, r to p for NegativeBinomial: p = r / (r + mu)
    p_r1 = r_r1 / (r_r1 + mu_r1)
    p_r2 = r_r2 / (r_r2 + mu_r2)
    
    # Create distributions directly without squeezing
    dist_r1 = torch.distributions.NegativeBinomial(total_count=r_r1, probs=p_r1)
    dist_r2 = torch.distributions.NegativeBinomial(total_count=r_r2, probs=p_r2)
    
    # Compute quantiles (PIT values) for both replicates
    quantiles_r1 = compute_numeric_cdf(dist_r1, y_r1_flat)
    quantiles_r2 = compute_numeric_cdf(dist_r2, y_r2_flat)
    
    # Compute mean absolute difference between quantiles
    return torch.mean(torch.abs(quantiles_r1 - quantiles_r2))


# Wrapper functions for loss registry compatibility
def concordance_loss_nce_wrapper(model_outputs: dict, batch: dict) -> torch.Tensor:
    """Wrapper for concordance_loss_nce to match loss registry interface."""
    # Extract observed reads from batch
    y_r1 = batch['r1']['reads']
    y_r2 = batch['r2']['reads']
    
    # Extract predicted means from model outputs to compute residuals
    mu_r1 = model_outputs['r1']['mu']
    mu_r2 = model_outputs['r2']['mu']
    
    # Compute residuals
    res1 = y_r1 - mu_r1
    res2 = y_r2 - mu_r2
    
    # Call the core algorithmic function
    return concordance_loss_nce(res1, res2)


def negative_pearson_loss_wrapper(model_outputs: dict, batch: dict) -> torch.Tensor:
    """Wrapper for negative_pearson_loss to match loss registry interface."""
    # Extract observed reads from batch for both replicates
    targets_r1 = batch['r1']['reads']
    targets_r2 = batch['r2']['reads']
    
    # Extract predictions from model outputs for both replicates
    preds_r1 = model_outputs['r1']['mu']
    preds_r2 = model_outputs['r2']['mu']
    
    # Concatenate predictions and targets from both replicates
    all_preds = torch.cat([preds_r1, preds_r2])
    all_targets = torch.cat([targets_r1, targets_r2])
    
    # Call the core algorithmic function
    return negative_pearson_loss(all_preds, all_targets)


def quantile_absolute_loss_wrapper(model_outputs: dict, batch: dict) -> torch.Tensor:
    """Wrapper for quantile_absolute_loss to match loss registry interface."""
    # Extract observed reads from batch
    y_r1 = batch['r1']['reads']
    y_r2 = batch['r2']['reads']
    
    # Extract model outputs for both replicates
    model_r1 = model_outputs['r1']
    model_r2 = model_outputs['r2']
    
    # Call the core algorithmic function
    return quantile_absolute_loss(model_r1, model_r2, y_r1, y_r2)


class CompositeLoss(nn.Module):
    """Composite loss that combines multiple loss functions with weights and named components."""
    
    def __init__(
        self, 
        loss_functions: List[Callable], 
        weights: List[float],
        component_names: List[str]
    ):
        """Initialize composite loss with mandatory named components.
        
        Args:
            loss_functions: List of loss functions to combine
            weights: Corresponding weights for each loss function
            component_names: Names for each loss component (must be unique)
        """
        super().__init__()
        if len(loss_functions) != len(weights):
            raise ValueError("Number of loss functions must match number of weights")
        
        if len(loss_functions) != len(component_names):
            raise ValueError("Number of loss functions must match number of component names")
        
        if len(set(component_names)) != len(component_names):
            raise ValueError("Component names must be unique")
        
        self.loss_functions = loss_functions
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.component_names = component_names
    
    def forward(
        self, 
        model_outputs: Dict, 
        batch: Dict
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted combination of all loss functions.
        
        Args:
            model_outputs: Model outputs
            batch: Batch data
            
        Returns:
            Dict with 'total' and 'components' keys, where components is a dict of named loss values
        """
        # Compute individual loss components
        loss_components = {}
        component_values = []
        
        for loss_fn, name in zip(self.loss_functions, self.component_names):
            component_loss = loss_fn(model_outputs, batch)
            loss_components[name] = component_loss
            component_values.append(component_loss)
        
        # Stack components and compute weighted sum
        components_tensor = torch.stack(component_values)
        weights_device = self.weights.to(components_tensor.device)
        total_loss = torch.sum(weights_device * components_tensor)
        
        return {
            'total': total_loss,
            'components': loss_components
        }


def nll_loss(model_outputs: dict, batch: dict) -> torch.Tensor:
    """Compute negative log-likelihood loss for observed data given model predictions.
    
    This function computes the negative log-likelihood of the observed read counts
    under the predicted Negative Binomial distribution parameters.
    
    Args:
        model_outputs: Dictionary with structure {'r1': {'mu': ..., 'r': ...}, 'r2': {...}}
                      or single replicate {'mu': ..., 'r': ...}
        batch: Dictionary from dataset with observed read counts
        
    Returns:
        Negative log-likelihood loss as scalar tensor
    """
    total_nll = 0.0
    count = 0
    
    # Handle both single and multi-replicate cases
    if 'r1' in model_outputs:
        # Multi-replicate case
        for replicate in ['r1', 'r2']:
            if replicate in model_outputs and replicate in batch:
                # Get model predictions
                mu = model_outputs[replicate]['mu']
                r = model_outputs[replicate]['r']
                
                # Get observed data
                observed = batch[replicate]['reads']
                
                # Create Negative Binomial distribution
                # Convert from (mu, r) parameterization to (total_count, probs)
                total_count = r
                probs = r / (mu + r)
                
                # Ensure parameters are valid
                total_count = torch.clamp(total_count, min=1e-6)
                probs = torch.clamp(probs, min=1e-6, max=1 - 1e-6)
                
                # Compute negative binomial distribution
                nb_dist = torch.distributions.NegativeBinomial(
                    total_count=total_count, 
                    probs=probs
                )
                
                # Compute negative log-likelihood
                log_prob = nb_dist.log_prob(observed)
                nll = -torch.mean(log_prob)
                
                total_nll += nll
                count += 1
    else:
        # Single replicate case
        mu = model_outputs['mu']
        r = model_outputs['r']
        observed = batch['reads']
        
        # Create Negative Binomial distribution
        total_count = r
        probs = r / (mu + r)
        
        # Ensure parameters are valid
        total_count = torch.clamp(total_count, min=1e-6)
        probs = torch.clamp(probs, min=1e-6, max=1 - 1e-6)
        
        # Compute negative binomial distribution
        nb_dist = torch.distributions.NegativeBinomial(
            total_count=total_count, 
            probs=probs
        )
        
        # Compute negative log-likelihood
        log_prob = nb_dist.log_prob(observed)
        nll = -torch.mean(log_prob)
        
        total_nll = nll
        count = 1
    
    # Return average NLL across replicates
    return total_nll / count if count > 0 else torch.tensor(0.0)