"""Minimal test model for fast CPU-based testing."""

import torch
from torch import nn


class MinimalTestModel(nn.Module):
    """Minimal model that implements the same interface as TechNB_mu_r but with simple linear layers."""
    
    def __init__(self, covariate_dim: int):
        """Initialize the minimal test model.
        
        Args:
            covariate_dim: Dimension of input covariates
        """
        super().__init__()
        self.covariate_dim = covariate_dim
        
        # Single linear layer for mu prediction (log space)
        self.mu_layer = nn.Linear(covariate_dim, 1)
        
        # Single linear layer for r prediction (log space)
        # Takes covariates + mu prediction as input (like original TechNB_mu_r)
        self.r_layer = nn.Linear(covariate_dim + 1, 1)
        
        # Initialize parameters
        self._initialize_parameters()
        
    def _initialize_parameters(self) -> None:
        """Initialize parameters to reasonable values."""
        # Initialize mu layer weights and bias
        nn.init.normal_(self.mu_layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.mu_layer.bias, 0.0)
        
        # Initialize r layer weights and bias  
        nn.init.normal_(self.r_layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.r_layer.bias, 0.0)
        
    def forward(self, x_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to predict mu and r parameters.
        
        Args:
            x_i: Input covariates tensor of shape (batch_size, covariate_dim)
            
        Returns:
            Tuple containing (mu, r) predicted parameters
        """
        # Predict log(mu) using covariates
        log_mu = self.mu_layer(x_i)
        
        # Predict log(r) using covariates + log(mu) (similar to original TechNB_mu_r)
        r_input = torch.cat([log_mu, x_i], dim=1)
        log_r = self.r_layer(r_input)
        
        # Return exponential to ensure positive values
        return (torch.exp(log_mu), torch.exp(log_r))