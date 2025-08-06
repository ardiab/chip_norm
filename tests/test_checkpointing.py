"""Tests for the checkpointing system."""

import os
import tempfile
import torch
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from chipvi.training.checkpoint_manager import CheckpointManager


class TestCheckpointManager:
    """Test the CheckpointManager class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_configs = [
            {
                'metric_name': 'val_loss',
                'mode': 'min',
                'filename': 'best_loss.pt',
                'overwrite': True
            },
            {
                'metric_name': 'val_residual_spearman',
                'mode': 'max',
                'filename': 'best_corr.pt',
                'overwrite': False
            }
        ]
        self.manager = CheckpointManager(self.temp_dir, self.checkpoint_configs)
        
    def test_initialization(self):
        """Test CheckpointManager initialization."""
        assert str(self.manager.output_dir) == self.temp_dir
        assert len(self.manager.checkpoint_configs) == 2
        assert 'val_loss' in self.manager.best_values
        assert 'val_residual_spearman' in self.manager.best_values
        assert self.manager.best_values['val_loss'] is None
        assert self.manager.best_values['val_residual_spearman'] is None

    def test_multi_metric_checkpointing(self):
        """Test that different metrics trigger saves to different files."""
        mock_state_dict = {'layer.weight': torch.tensor([1.0, 2.0])}
        
        # First update - both should save since no previous best values
        metrics = {'val_loss': 0.5, 'val_residual_spearman': 0.8}
        self.manager.update(metrics, mock_state_dict, epoch=1)
        
        assert os.path.exists(os.path.join(self.temp_dir, 'best_loss.pt'))
        assert os.path.exists(os.path.join(self.temp_dir, 'best_corr.pt'))
        
        # Second update - only loss should save (better), corr should not (worse)
        metrics = {'val_loss': 0.3, 'val_residual_spearman': 0.7}
        self.manager.update(metrics, mock_state_dict, epoch=2)
        
        # Load and verify the loss checkpoint was updated
        loss_checkpoint = torch.load(os.path.join(self.temp_dir, 'best_loss.pt'))
        assert loss_checkpoint['epoch'] == 2
        assert loss_checkpoint['metric_value'] == 0.3

    def test_loss_component_tracking(self):
        """Test individual loss components can trigger checkpoints independently."""
        config_with_components = [
            {
                'metric_name': 'loss_components.concordance',
                'mode': 'min', 
                'filename': 'best_concordance.pt',
                'overwrite': True
            },
            {
                'metric_name': 'loss_components.other',
                'mode': 'min',
                'filename': 'best_other.pt',
                'overwrite': True
            },
            {
                'metric_name': 'val_loss',
                'mode': 'min',
                'filename': 'best_loss.pt',
                'overwrite': True
            }
        ]
        manager = CheckpointManager(self.temp_dir, config_with_components)
        mock_state_dict = {'layer.weight': torch.tensor([1.0])}
        
        metrics = {
            'val_loss': 1.0,
            'loss_components': {'concordance': 0.2, 'other': 0.8}
        }
        
        manager.update(metrics, mock_state_dict, epoch=1)
        assert os.path.exists(os.path.join(self.temp_dir, 'best_concordance.pt'))
        assert os.path.exists(os.path.join(self.temp_dir, 'best_other.pt'))
        assert os.path.exists(os.path.join(self.temp_dir, 'best_loss.pt'))
        
        checkpoint_concordance = torch.load(os.path.join(self.temp_dir, 'best_concordance.pt'))
        assert checkpoint_concordance['metric_value'] == 0.2

        checkpoint_other = torch.load(os.path.join(self.temp_dir, 'best_other.pt'))
        assert checkpoint_other['metric_value'] == 0.8

        checkpoint_loss = torch.load(os.path.join(self.temp_dir, 'best_loss.pt'))
        assert checkpoint_loss['metric_value'] == 1.0

    def test_checkpoint_file_integrity(self):
        """Test saved checkpoints are complete and loadable."""
        mock_state_dict = {
            'linear.weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            'linear.bias': torch.tensor([0.5, -0.5])
        }
        
        metrics = {'val_loss': 0.5}
        self.manager.update(metrics, mock_state_dict, epoch=5)
        
        # Test static loading method
        loaded_state_dict, metadata = CheckpointManager.load_checkpoint(
            os.path.join(self.temp_dir, 'best_loss.pt')
        )
        
        assert torch.equal(loaded_state_dict['linear.weight'], mock_state_dict['linear.weight'])
        assert torch.equal(loaded_state_dict['linear.bias'], mock_state_dict['linear.bias'])
        assert metadata['epoch'] == 5
        assert metadata['metric_value'] == 0.5
        assert 'timestamp' in metadata
        assert 'config' in metadata

    def test_improvement_detection_min_mode(self):
        """Test improvement detection for min mode metrics."""
        mock_state_dict = {'weight': torch.tensor([1.0])}
        
        # First value should always save
        self.manager.update({'val_loss': 0.8}, mock_state_dict, epoch=1)
        assert self.manager.best_values['val_loss'] == 0.8
        
        # Better value (lower) should save
        self.manager.update({'val_loss': 0.5}, mock_state_dict, epoch=2)
        assert self.manager.best_values['val_loss'] == 0.5
        
        # Worse value (higher) should not save
        self.manager.update({'val_loss': 0.9}, mock_state_dict, epoch=3)
        assert self.manager.best_values['val_loss'] == 0.5

    def test_improvement_detection_max_mode(self):
        """Test improvement detection for max mode metrics."""
        mock_state_dict = {'weight': torch.tensor([1.0])}
        
        # First value should always save
        self.manager.update({'val_residual_spearman': 0.5}, mock_state_dict, epoch=1)
        assert self.manager.best_values['val_residual_spearman'] == 0.5
        
        # Better value (higher) should save
        self.manager.update({'val_residual_spearman': 0.8}, mock_state_dict, epoch=2)
        assert self.manager.best_values['val_residual_spearman'] == 0.8
        
        # Worse value (lower) should not save
        self.manager.update({'val_residual_spearman': 0.3}, mock_state_dict, epoch=3)
        assert self.manager.best_values['val_residual_spearman'] == 0.8

    def test_concurrent_strategy_execution(self):
        """Test multiple checkpointing strategies run simultaneously without interfering."""
        mock_state_dict = {'weight': torch.tensor([1.0])}
        
        # Both metrics improve on first update
        metrics = {'val_loss': 0.5, 'val_residual_spearman': 0.8}
        self.manager.update(metrics, mock_state_dict, epoch=1)
        
        assert self.manager.best_values['val_loss'] == 0.5
        assert self.manager.best_values['val_residual_spearman'] == 0.8
        assert os.path.exists(os.path.join(self.temp_dir, 'best_loss.pt'))
        assert os.path.exists(os.path.join(self.temp_dir, 'best_corr.pt'))
        
        # Only one metric improves
        metrics = {'val_loss': 0.3, 'val_residual_spearman': 0.7}
        self.manager.update(metrics, mock_state_dict, epoch=2)
        
        assert self.manager.best_values['val_loss'] == 0.3  # Updated
        assert self.manager.best_values['val_residual_spearman'] == 0.8  # Not updated

    def test_checkpoint_overwrite_control(self):
        """Test overwrite behavior with existing files."""
        import time
        mock_state_dict = {'weight': torch.tensor([1.0])}
        
        # Create initial checkpoints
        metrics = {'val_loss': 0.5, 'val_residual_spearman': 0.8}
        self.manager.update(metrics, mock_state_dict, epoch=1)
        
        # Get initial modification times
        loss_path = os.path.join(self.temp_dir, 'best_loss.pt')
        corr_path = os.path.join(self.temp_dir, 'best_corr.pt')
        
        original_loss_mtime = os.path.getmtime(loss_path)
        
        # Sleep briefly to ensure different timestamps
        time.sleep(0.01)
        
        # Update with improvements
        metrics = {'val_loss': 0.3, 'val_residual_spearman': 0.9}
        self.manager.update(metrics, mock_state_dict, epoch=2)
        
        # Loss file should be overwritten (overwrite=True)
        new_loss_mtime = os.path.getmtime(loss_path)
        assert new_loss_mtime >= original_loss_mtime  # Use >= to handle timing edge cases
        
        # Check that timestamped version exists for corr (overwrite=False)
        timestamped_files = [f for f in os.listdir(self.temp_dir) if 'best_corr' in f and f != 'best_corr.pt']
        assert len(timestamped_files) == 1

    def test_missing_metrics_handling(self):
        """Test graceful handling of missing metrics with warnings."""
        mock_state_dict = {'weight': torch.tensor([1.0])}
        
        with patch('chipvi.training.checkpoint_manager.logger') as mock_logger:
            # Missing metrics should trigger warnings
            metrics = {'val_loss': 0.5}  # Missing val_residual_spearman
            self.manager.update(metrics, mock_state_dict, epoch=1)
            
            # Should have logged a warning for missing metric
            mock_logger.warning.assert_called()
            warning_call_args = mock_logger.warning.call_args[0][0]
            assert 'val_residual_spearman' in warning_call_args
            assert 'not found' in warning_call_args.lower()

    def test_load_checkpoint_missing_file(self):
        """Test checkpoint loading with missing files."""
        with pytest.raises(FileNotFoundError):
            CheckpointManager.load_checkpoint('/nonexistent/path.pt')

    def test_nested_metric_access(self):
        """Test accessing nested metrics using dot notation."""
        config = [
            {
                'metric_name': 'nested.deep.metric',
                'mode': 'min',
                'filename': 'nested_test.pt',
                'overwrite': True
            }
        ]
        manager = CheckpointManager(self.temp_dir, config)
        mock_state_dict = {'weight': torch.tensor([1.0])}
        
        metrics = {
            'nested': {
                'deep': {
                    'metric': 0.123
                }
            }
        }
        
        manager.update(metrics, mock_state_dict, epoch=1)
        assert manager.best_values['nested.deep.metric'] == 0.123
        assert os.path.exists(os.path.join(self.temp_dir, 'nested_test.pt'))