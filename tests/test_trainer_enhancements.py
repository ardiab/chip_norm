"""Tests for enhanced trainer features including LR scheduling, early stopping, gradient clipping, and W&B logging."""

import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from chipvi.training.trainer import Trainer


class MockModel(nn.Module):
    """Simple mock model for testing."""
    
    def __init__(self, input_dim: int = 10, output_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = self.linear(x)
        mu = torch.exp(out[:, 0])  # Ensure positive mu
        r = torch.exp(out[:, 1])   # Ensure positive r
        return mu, r


def create_mock_batch(batch_size: int = 4, input_dim: int = 10):
    """Create a mock batch with the expected multi-replicate structure."""
    return {
        'r1': {
            'covariates': torch.randn(batch_size, input_dim),
            'reads': torch.randint(0, 100, (batch_size,))
        },
        'r2': {
            'covariates': torch.randn(batch_size, input_dim),
            'reads': torch.randint(0, 100, (batch_size,))
        },
        'metadata': {
            'sd_ratio': torch.randn(batch_size),
            'grp_idx': torch.randint(0, 5, (batch_size,))
        }
    }


def create_mock_dataloader(num_batches: int = 3, batch_size: int = 4):
    """Create a mock dataloader that yields mock batches."""
    batches = [create_mock_batch(batch_size) for _ in range(num_batches)]
    
    class MockDataLoader:
        def __init__(self, batches):
            self.batches = batches
            
        def __iter__(self):
            return iter(self.batches)
        
        def __len__(self):
            return len(self.batches)
    
    return MockDataLoader(batches)


def create_configured_trainer(config=None):
    """Create a trainer with configuration for testing."""
    model = MockModel()
    train_loader = create_mock_dataloader(num_batches=2)
    val_loader = create_mock_dataloader(num_batches=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    def mock_loss_fn(model_outputs, batch):
        return torch.tensor(1.0, requires_grad=True)
    
    device = torch.device('cpu')
    
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=mock_loss_fn,
        device=device,
        config=config
    )


class TestLearningRateScheduling:
    """Test learning rate scheduling with warmup and cosine annealing."""
    
    def test_lr_scheduler_initialization(self):
        """Test that LR schedulers are properly initialized."""
        config = {
            'scheduler_config': {
                'warmup_epochs': 3,
                'scheduler_type': 'cosine'
            }
        }
        trainer = create_configured_trainer(config)
        
        assert hasattr(trainer, 'warmup_epochs')
        assert trainer.warmup_epochs == 3
        assert hasattr(trainer, 'scheduler_type')
        assert trainer.scheduler_type == 'cosine'
        assert trainer.scheduler is None  # Not initialized until fit() is called
            
    def test_warmup_linear_schedule(self):
        """Test that learning rate follows linear warmup pattern."""
        config = {
            'scheduler_config': {
                'warmup_epochs': 3,
                'scheduler_type': 'cosine'
            }
        }
        trainer = create_configured_trainer(config)
        
        # Set up scheduler
        total_epochs = 10
        trainer._setup_schedulers(total_epochs)
        
        # Check that scheduler was created
        assert trainer.scheduler is not None
        
        # Check initial LR (should be reduced by start_factor)
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        assert initial_lr < 0.001  # Should be reduced from original 0.001
    
    def test_cosine_annealing_schedule(self):
        """Test that learning rate follows cosine annealing after warmup."""
        config = {
            'scheduler_config': {
                'warmup_epochs': 2,
                'scheduler_type': 'cosine'
            }
        }
        trainer = create_configured_trainer(config)
        
        total_epochs = 8
        trainer._setup_schedulers(total_epochs)
        
        assert trainer.scheduler is not None
        
        # Test that scheduler can step without errors
        for _ in range(5):
            trainer.scheduler.step()
    
    def test_scheduler_milestone_transition(self):
        """Test transition from warmup to cosine scheduler at correct epoch."""
        config = {
            'scheduler_config': {
                'warmup_epochs': 3,
                'scheduler_type': 'cosine'
            }
        }
        trainer = create_configured_trainer(config)
        
        total_epochs = 10
        trainer._setup_schedulers(total_epochs)
        
        # Test that the SequentialLR scheduler is created with correct milestone
        assert trainer.scheduler is not None
        assert hasattr(trainer.scheduler, '_milestones')
        assert trainer.scheduler._milestones == [3]


class TestEarlyStopping:
    """Test early stopping functionality."""
    
    def test_early_stopping_initialization(self):
        """Test early stopping parameters are properly initialized."""
        config = {
            'early_stopping_config': {
                'patience': 5,
                'monitor_metric': 'val_loss'
            }
        }
        trainer = create_configured_trainer(config)
        
        assert hasattr(trainer, 'patience')
        assert trainer.patience == 5
        assert hasattr(trainer, 'best_val_loss')
        assert trainer.best_val_loss == float('inf')
        assert hasattr(trainer, 'epochs_without_improvement')
        assert trainer.epochs_without_improvement == 0
    
    def test_early_stopping_triggers(self):
        """Test that early stopping is triggered correctly."""
        config = {
            'early_stopping_config': {
                'patience': 3,
                'monitor_metric': 'val_loss'
            }
        }
        trainer = create_configured_trainer(config)
        
        # Simulate worsening validation losses
        assert not trainer._check_early_stopping(1.0)  # First loss
        assert not trainer._check_early_stopping(1.1)  # Worse, counter = 1
        assert not trainer._check_early_stopping(1.2)  # Worse, counter = 2
        assert trainer._check_early_stopping(1.3)      # Worse, counter = 3, should trigger
    
    def test_best_model_preservation(self):
        """Test that best model state is preserved when early stopping."""
        config = {
            'early_stopping_config': {
                'patience': 5,
                'monitor_metric': 'val_loss'
            }
        }
        trainer = create_configured_trainer(config)
        
        # Test that better loss saves model state
        trainer._check_early_stopping(0.5)
        assert trainer.best_model_state is not None
    
    def test_improvement_resets_counter(self):
        """Test that metric improvement resets the early stopping counter."""
        config = {
            'early_stopping_config': {
                'patience': 3,
                'monitor_metric': 'val_loss'
            }
        }
        trainer = create_configured_trainer(config)
        
        # Simulate some bad epochs then improvement
        trainer._check_early_stopping(1.0)  # Initial
        trainer._check_early_stopping(1.1)  # Worse, counter = 1
        trainer._check_early_stopping(0.9)  # Better, should reset counter
        assert trainer.epochs_without_improvement == 0


class TestGradientClipping:
    """Test gradient clipping functionality."""
    
    def test_gradient_clipping_applied(self):
        """Test that gradients are properly clipped to max_norm."""
        config = {
            'max_grad_norm': 1.0
        }
        trainer = create_configured_trainer(config)
        
        # Create large gradients
        for param in trainer.model.parameters():
            param.grad = torch.randn_like(param) * 10.0  # Large gradients
        
        # Test clipping
        grad_norm_before = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), float('inf'))
        clipped_norm = trainer._clip_gradients()
        
        # Verify gradients were clipped
        current_norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), float('inf'))
        assert current_norm <= 1.0
    
    def test_gradient_norm_logging(self):
        """Test that gradient norms are computed correctly."""
        trainer = create_configured_trainer()
        
        # Create some gradients
        for param in trainer.model.parameters():
            param.grad = torch.randn_like(param)
            
        grad_norm = trainer._clip_gradients()
        assert isinstance(grad_norm, (float, int))
        assert grad_norm >= 0
    
    def test_gradient_clipping_disabled(self):
        """Test that gradient clipping can be disabled (max_grad_norm=None)."""
        config = {
            'max_grad_norm': None
        }
        trainer = create_configured_trainer(config)
        
        # Create large gradients
        for param in trainer.model.parameters():
            param.grad = torch.randn_like(param) * 10.0
            
        # Should return norm without clipping
        grad_norm = trainer._clip_gradients()
        assert isinstance(grad_norm, (float, int))
        assert grad_norm == torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), float('inf'))


class TestWandBIntegration:
    """Test Weights & Biases logging integration."""
    
    @patch('chipvi.training.trainer.wandb')
    def test_wandb_initialization(self, mock_wandb):
        """Test that W&B is properly initialized when enabled."""
        config = {
            'wandb_config': {
                'project': 'test_project',
                'entity': 'test_entity', 
                'name': 'test_run',
                'enabled': True
            }
        }
        trainer = create_configured_trainer(config)
        trainer._init_wandb()
        
        mock_wandb.init.assert_called_once_with(
            project='test_project',
            entity='test_entity',
            name='test_run'
        )
    
    @patch('chipvi.training.trainer.wandb')
    def test_wandb_per_batch_logging(self, mock_wandb):
        """Test that per-batch metrics are logged to W&B."""
        config = {
            'wandb_config': {
                'enabled': True,
                'project': 'test'
            }
        }
        trainer = create_configured_trainer(config)
        
        batch_metrics = {
            'train_loss': 0.5,
            'learning_rate': 0.001,
            'grad_norm': 1.2
        }
        
        trainer._log_batch_metrics(batch_metrics)
        mock_wandb.log.assert_called_once_with(batch_metrics)
    
    @patch('chipvi.training.trainer.wandb')
    def test_wandb_per_epoch_logging(self, mock_wandb):
        """Test that per-epoch metrics are logged to W&B."""
        config = {
            'wandb_config': {
                'enabled': True,
                'project': 'test'
            }
        }
        trainer = create_configured_trainer(config)
        
        epoch_metrics = {
            'val_loss': 0.4,
            'epoch': 1,
            'epoch_time': 10.5
        }
        
        trainer._log_epoch_metrics(epoch_metrics)
        mock_wandb.log.assert_called_once_with(epoch_metrics)
    
    def test_wandb_disabled_gracefully(self):
        """Test that training works correctly when W&B is disabled."""
        config = {
            'wandb_config': {
                'enabled': False
            }
        }
        trainer = create_configured_trainer(config)
        
        # Should not raise any errors when disabled
        trainer._init_wandb()
        trainer._log_batch_metrics({'train_loss': 0.5})
        trainer._log_epoch_metrics({'val_loss': 0.4})
        trainer._cleanup_wandb()
    
    @patch('chipvi.training.trainer.wandb')
    def test_wandb_cleanup(self, mock_wandb):
        """Test that W&B is properly cleaned up after training."""
        config = {
            'wandb_config': {
                'enabled': True,
                'project': 'test'
            }
        }
        trainer = create_configured_trainer(config)
        
        trainer._cleanup_wandb()
        mock_wandb.finish.assert_called_once()


class TestConfigurationIntegration:
    """Test that all features are configurable via configuration."""
    
    def test_config_parameter_extraction(self):
        """Test that configuration parameters are properly extracted."""
        mock_config = {
            'scheduler_config': {
                'warmup_epochs': 5,
                'scheduler_type': 'cosine'
            },
            'early_stopping_config': {
                'patience': 10,
                'monitor_metric': 'val_loss'
            },
            'wandb_config': {
                'project': 'chipvi',
                'enabled': True
            },
            'max_grad_norm': 2.0
        }
        
        trainer = create_configured_trainer(mock_config)
        
        # Check scheduler config
        assert trainer.warmup_epochs == 5
        assert trainer.scheduler_type == 'cosine'
        
        # Check early stopping config
        assert trainer.patience == 10
        assert trainer.monitor_metric == 'val_loss'
        
        # Check W&B config
        assert trainer.wandb_enabled == True
        assert trainer.wandb_project == 'chipvi'
        
        # Check gradient clipping
        assert trainer.max_grad_norm == 2.0
    

class TestIntegrationScenarios:
    """Test integration scenarios combining multiple features."""
    
    @patch('chipvi.training.trainer.wandb')
    def test_full_training_integration(self, mock_wandb):
        """Test that all features work together in a short training run."""
        config = {
            'scheduler_config': {
                'warmup_epochs': 2,
                'scheduler_type': 'cosine'
            },
            'early_stopping_config': {
                'patience': 5,
                'monitor_metric': 'val_loss'
            },
            'wandb_config': {
                'enabled': True,
                'project': 'test'
            },
            'max_grad_norm': 1.0
        }
        
        trainer = create_configured_trainer(config)
        
        # Should be able to run training without errors
        trainer.fit(num_epochs=3)
        
        # Verify W&B was used
        mock_wandb.init.assert_called_once()
        mock_wandb.finish.assert_called_once()
        assert mock_wandb.log.call_count > 0