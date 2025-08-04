"""Tests for the unified Trainer class."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from chipvi.training.trainer import Trainer
from chipvi.training.losses import replicate_concordance_mse_loss


class MockDataset:
    """Mock dataset that returns dictionary-based batch structure."""
    
    def __init__(self, num_samples: int = 32):
        self.num_samples = num_samples
        # Create mock data with the structure expected by MultiReplicateDataset
        self.r1_covariates = torch.randn(num_samples, 5)
        self.r1_reads = torch.randint(0, 100, (num_samples,)).float()
        self.r2_covariates = torch.randn(num_samples, 5)
        self.r2_reads = torch.randint(0, 100, (num_samples,)).float()
        self.sd_ratios = torch.ones(num_samples)
        self.grp_idxs = torch.arange(num_samples)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'r1': {
                'covariates': self.r1_covariates[idx],
                'reads': self.r1_reads[idx]
            },
            'r2': {
                'covariates': self.r2_covariates[idx],
                'reads': self.r2_reads[idx]
            },
            'metadata': {
                'sd_ratio': self.sd_ratios[idx],
                'grp_idx': self.grp_idxs[idx]
            }
        }


class MockModel(nn.Module):
    """Mock model that returns two outputs like TechNB_mu_r."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 2)  # 5 covariates -> 2 outputs
    
    def forward(self, x):
        outputs = self.linear(x)
        mu = torch.exp(outputs[:, 0:1])  # Ensure positive mu
        r = torch.exp(outputs[:, 1:2])   # Ensure positive r
        return mu, r


def test_trainer_completes_one_epoch():
    """Test that the Trainer can complete one epoch without errors."""
    # Create a mock model that returns two outputs like TechNB_mu_r
    model = MockModel()
    
    # Create mock DataLoaders with dictionary-based batch structure
    train_dataset = MockDataset(num_samples=16)
    val_dataset = MockDataset(num_samples=8)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Create mock optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    # Get initial parameter values to check they change after training
    initial_params = [param.clone() for param in model.parameters()]
    
    # Create trainer instance
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=replicate_concordance_mse_loss,
        device=torch.device("cpu"),
    )
    
    # Run training for one epoch
    trainer.fit(num_epochs=1)
    
    # Check that model parameters have been updated
    final_params = list(model.parameters())
    for initial, final in zip(initial_params, final_params):
        # Assert parameters have changed (not exactly equal)
        assert not torch.allclose(initial, final, atol=1e-6), \
            "Model parameters should have been updated after training"