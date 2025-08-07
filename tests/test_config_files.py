"""Tests for test configuration files.

This module tests that the dedicated test configurations include:
- Multi-component loss setup
- Minimal training parameters 
- Disabled wandb
- Paths configured for synthetic test data
"""

import pytest
import os
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
import tempfile
import shutil


class TestConfigurationFiles:
    """Test suite for test configuration files."""
    
    @pytest.fixture
    def config_dir(self) -> Path:
        """Return the absolute path to the configs directory."""
        return Path(__file__).parent.parent / "configs"
    
    @pytest.fixture
    def test_config(self, config_dir: Path) -> DictConfig:
        """Load the test e2e configuration."""
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name="experiment/test_e2e")
            # Access the config under the 'experiment' key
            return cfg.experiment
    
    def test_multi_component_loss_configuration(self, test_config: DictConfig):
        """Test that the configuration properly defines a composite loss with multiple components."""
        assert hasattr(test_config, 'loss_fn'), "Test config must define loss_fn"
        
        # Check that it's a composite loss with multiple components
        assert hasattr(test_config.loss_fn, 'losses'), "Loss configuration must have 'losses' field"
        losses = test_config.loss_fn.losses
        
        # Verify at least 2 components
        assert len(losses) >= 2, f"Must have at least 2 loss components, got {len(losses)}"
        
        # Verify weights match number of components
        if hasattr(test_config.loss_fn, 'weights'):
            assert len(test_config.loss_fn.weights) == len(losses), \
                "Number of weights must match number of loss components"
    
    def test_minimal_training_parameters(self, test_config: DictConfig):
        """Test that training parameters are set to minimal values for fast testing."""
        # Check epochs - should be minimal (≤ 5 for testing)
        assert hasattr(test_config, 'num_epochs'), "Test config must define num_epochs"
        assert test_config.num_epochs <= 5, f"Test epochs should be ≤ 5, got {test_config.num_epochs}"
        
        # Check batch size - should be small (≤ 64 for testing)  
        assert hasattr(test_config, 'batch_size'), "Test config must define batch_size"
        assert test_config.batch_size <= 64, f"Test batch_size should be ≤ 64, got {test_config.batch_size}"
        
        # Check learning rate - should be reasonable
        assert hasattr(test_config, 'learning_rate'), "Test config must define learning_rate"
        assert 0.0001 <= test_config.learning_rate <= 0.1, \
            f"Learning rate should be in reasonable range, got {test_config.learning_rate}"
    
    def test_wandb_disabled(self, test_config: DictConfig):
        """Test that wandb is explicitly disabled in the test configuration."""
        assert hasattr(test_config, 'wandb'), "Test config must define wandb settings"
        assert hasattr(test_config.wandb, 'enabled'), "WandB config must have 'enabled' field"
        assert test_config.wandb.enabled is False, "WandB must be disabled for tests"
    
    def test_model_configuration(self, test_config: DictConfig):
        """Test that the model config points to a minimal/test model."""
        assert hasattr(test_config, 'model'), "Test config must define model"
        assert hasattr(test_config.model, '_target_'), "Model must have _target_ field"
        
        # Check that it's using minimal model parameters
        if hasattr(test_config.model, 'hidden_dims_mu'):
            # For TechNB models, check that hidden dimensions are small
            max_dim = max(test_config.model.hidden_dims_mu)
            assert max_dim <= 64, f"Model hidden dimensions should be small for testing, got max {max_dim}"
    
    def test_data_paths_configurable(self, test_config: DictConfig):
        """Test that data paths can be overridden to point to temporary test data."""
        # Check that paths are defined and can be overridden via Hydra
        assert hasattr(test_config, 'paths'), "Test config must define paths"
        
        # Test that we can override data paths with a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # The configuration should be structured to allow path overrides
            # This tests the structure rather than actual override functionality
            assert hasattr(test_config.paths, 'data_base'), "Paths must define data_base"
            
            # Verify that data_base path can be used to construct other paths
            data_base = test_config.paths.data_base
            assert isinstance(data_base, (str, Path)) or data_base is not None, \
                "data_base must be a valid path"


class TestConfigurationIntegration:
    """Integration tests for configuration loading."""
    
    def test_wandb_config_files_exist(self):
        """Test that wandb test configuration file exists."""
        config_path = Path(__file__).parent.parent / "configs" / "wandb" / "test" / "wandb_test.yaml"
        assert config_path.exists(), f"WandB test config should exist at {config_path}"
    
    def test_model_config_files_exist(self):
        """Test that model test configuration file exists.""" 
        config_path = Path(__file__).parent.parent / "configs" / "model" / "test" / "model_test.yaml"
        assert config_path.exists(), f"Model test config should exist at {config_path}"
    
    def test_experiment_config_loading(self):
        """Test that the main experiment config can be loaded without errors."""
        config_dir = Path(__file__).parent.parent / "configs"
        
        try:
            with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                cfg = compose(config_name="experiment/test_e2e")
                assert cfg is not None, "Configuration should load successfully"
        except Exception as e:
            pytest.fail(f"Configuration failed to load: {e}")