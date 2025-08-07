"""Tests for legacy experiment configuration reproduction."""

import pytest
import subprocess
import tempfile
import shutil
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
import yaml
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

from tests.utils.synthetic_data import SyntheticDataManager, generate_multi_replicate_data


class TestLegacyExperimentConfiguration:
    """Test the legacy_reproduction experiment configuration."""

    def test_legacy_config_loading(self):
        """Test that legacy_reproduction experiment configuration loads successfully."""
        # Try to load the legacy reproduction config
        with initialize(config_path="../configs", version_base=None):
            try:
                cfg = compose(config_name="config", overrides=["experiment=legacy_reproduction"])
                assert cfg is not None
                assert 'experiment' in cfg
            except Exception as e:
                pytest.fail(f"Failed to load legacy_reproduction experiment config: {e}")

    def test_legacy_config_components_instantiation(self):
        """Test that all components in legacy config can be instantiated."""
        with initialize(config_path="../configs", version_base=None):
            try:
                cfg = compose(config_name="config", overrides=["experiment=legacy_reproduction"])
                
                # Check that model config is present and can be instantiated
                assert 'experiment' in cfg
                exp_cfg = cfg.experiment
                assert 'model' in exp_cfg
                model_cfg = exp_cfg.model
                assert model_cfg._target_ == "chipvi.models.technical_model.TechNB_mu_r"
                
                # Check that loss config is present and has composite structure
                assert 'loss_fn' in exp_cfg
                loss_cfg = exp_cfg.loss_fn
                assert hasattr(loss_cfg, 'loss_functions')
                assert hasattr(loss_cfg, 'weights')
                assert len(loss_cfg.loss_functions) == 2
                assert len(loss_cfg.weights) == 2
                
                # Check that optimizer config is present 
                assert 'optimizer' in exp_cfg
                opt_cfg = exp_cfg.optimizer
                assert opt_cfg._target_ == "torch.optim.AdamW"
                assert opt_cfg.weight_decay == 0.01
                
                # Check trainer config exists and has necessary components
                assert 'trainer_config' in exp_cfg
                trainer_cfg = exp_cfg.trainer_config
                
                # Check checkpointing strategies are configured
                assert 'checkpoint_config' in trainer_cfg
                checkpoint_cfg = trainer_cfg.checkpoint_config
                assert 'strategies' in checkpoint_cfg
                assert len(checkpoint_cfg.strategies) == 2  # best_loss and best_corr
                
                # Check W&B integration is configured
                assert 'wandb_config' in trainer_cfg
                wandb_cfg = trainer_cfg.wandb_config
                assert wandb_cfg.enabled is True
                assert wandb_cfg.project == "chipvi"
                
            except Exception as e:
                pytest.fail(f"Failed to instantiate components from legacy config: {e}")

    def test_legacy_config_composite_loss_structure(self):
        """Test that composite loss has NLL + consistency loss with proper weights."""
        with initialize(config_path="../configs", version_base=None):
            try:
                cfg = compose(config_name="config", overrides=["experiment=legacy_reproduction"])
                
                loss_cfg = cfg.experiment.loss_fn
                # Should have two components: nll and consistency
                assert len(loss_cfg.loss_functions) == 2
                
                # Should have weights for both components
                assert len(loss_cfg.weights) == 2
                
                # Component names should match legacy implementation
                expected_components = ["nll", "consistency"]
                for component in expected_components:
                    assert component in loss_cfg.component_names
                
            except Exception as e:
                pytest.fail(f"Failed to verify composite loss structure: {e}")

    def test_legacy_config_scheduler_sequential(self):
        """Test that scheduler is configured for sequential warmup + cosine annealing."""
        with initialize(config_path="../configs", version_base=None):
            try:
                cfg = compose(config_name="config", overrides=["experiment=legacy_reproduction"])
                
                # Check trainer config has the right scheduler settings
                trainer_cfg = cfg.experiment.trainer_config
                assert 'scheduler_config' in trainer_cfg
                assert trainer_cfg.scheduler_config.warmup_epochs == 2
                assert trainer_cfg.scheduler_config.scheduler_type == 'cosine'
                
            except Exception as e:
                pytest.fail(f"Failed to verify scheduler configuration: {e}")

    def test_legacy_config_dual_checkpointing(self):
        """Test that dual checkpointing is configured (best loss + best correlation)."""
        with initialize(config_path="../configs", version_base=None):
            try:
                cfg = compose(config_name="config", overrides=["experiment=legacy_reproduction"])
                
                # Check trainer config has checkpointing configuration
                trainer_cfg = cfg.experiment.trainer_config
                assert 'checkpoint_config' in trainer_cfg
                checkpoint_cfg = trainer_cfg.checkpoint_config
                strategies = checkpoint_cfg.strategies
                
                # Should have exactly 2 strategies
                assert len(strategies) == 2
                
                # Find the loss and correlation strategies
                loss_strategy = None
                corr_strategy = None
                
                for strategy in strategies:
                    if strategy.metric_name == 'val_loss':
                        loss_strategy = strategy
                    elif 'residual' in strategy.metric_name and 'spearman' in strategy.metric_name:
                        corr_strategy = strategy
                
                # Both strategies should exist
                assert loss_strategy is not None
                assert corr_strategy is not None
                
                # Loss strategy should minimize, correlation should maximize
                assert loss_strategy.mode == 'min'
                assert corr_strategy.mode == 'max'
                
                # Filenames should match legacy implementation
                assert loss_strategy.filename == 'best_loss.pt'
                assert corr_strategy.filename == 'best_corr.pt'
                
            except Exception as e:
                pytest.fail(f"Failed to verify dual checkpointing configuration: {e}")

    def test_legacy_config_data_loading(self):
        """Test that data loading is configured to use converted legacy data files."""
        with initialize(config_path="../configs", version_base=None):
            try:
                cfg = compose(config_name="config", overrides=["experiment=legacy_reproduction"])
                
                # Should have output prefix configuration for data loading
                assert 'output_prefix_train' in cfg.experiment
                assert 'output_prefix_val' in cfg.experiment
                
                # Should have data configuration for preprocessing
                assert 'data' in cfg.experiment
                data_cfg = cfg.experiment.data
                
                # Should have log transform configuration
                assert 'preprocessing' in data_cfg
                preprocess_cfg = data_cfg.preprocessing
                assert 'log_transform' in preprocess_cfg
                
                log_transform_cfg = preprocess_cfg.log_transform
                assert 'enabled' in log_transform_cfg
                assert 'columns' in log_transform_cfg
                
                # Columns should be [0, 5] as in legacy implementation
                assert log_transform_cfg.columns == [0, 5]
                assert log_transform_cfg.enabled is False  # Default value
                
            except Exception as e:
                pytest.fail(f"Failed to verify data loading configuration: {e}")

    def test_legacy_config_model_architecture(self):
        """Test that model architecture matches legacy TechNB_mu_r configuration."""
        with initialize(config_path="../configs", version_base=None):
            try:
                cfg = compose(config_name="config", overrides=["experiment=legacy_reproduction"])
                
                model_cfg = cfg.experiment.model
                assert model_cfg._target_ == "chipvi.models.technical_model.TechNB_mu_r"
                
                # Should have configurable hidden dimensions
                assert 'hidden_dims_mu' in model_cfg
                assert 'hidden_dims_r' in model_cfg
                
                # Dimensions should be lists/tuples of integers
                mu_dims = model_cfg.hidden_dims_mu
                r_dims = model_cfg.hidden_dims_r
                
                # Check that dimensions are present and correct
                assert mu_dims is not None
                assert r_dims is not None
                assert len(mu_dims) > 0
                assert len(r_dims) > 0
                
                # Verify the specific values match legacy defaults
                assert mu_dims == [64, 32]
                assert r_dims == [32, 16]
                
            except Exception as e:
                pytest.fail(f"Failed to verify model architecture configuration: {e}")

    def test_legacy_config_training_parameters(self):
        """Test that training parameters match legacy implementation."""
        with initialize(config_path="../configs", version_base=None):
            try:
                cfg = compose(config_name="config", overrides=["experiment=legacy_reproduction"])
                
                # Check optimizer configuration (should be AdamW)
                assert 'optimizer' in cfg.experiment
                opt_cfg = cfg.experiment.optimizer
                assert 'AdamW' in opt_cfg._target_
                assert opt_cfg.weight_decay == 0.01  # As in legacy
                
                # Check early stopping patience in trainer config
                trainer_cfg = cfg.experiment.trainer_config
                assert 'early_stopping_config' in trainer_cfg
                early_stop_cfg = trainer_cfg.early_stopping_config
                assert 'patience' in early_stop_cfg
                assert early_stop_cfg.patience == 10  # As in legacy
                
                # Check gradient clipping in trainer config
                assert 'max_grad_norm' in trainer_cfg
                assert trainer_cfg.max_grad_norm == 1.0  # As in legacy
                
            except Exception as e:
                pytest.fail(f"Failed to verify training parameters: {e}")

    def test_legacy_config_end_to_end_execution(self):
        """Test that a short training session executes successfully with legacy config."""
        with SyntheticDataManager():
            # Create temporary data directory
            data_dir = Path(tempfile.mkdtemp())
            processed_dir = data_dir / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate synthetic data files with legacy naming convention
            target = "test_target"
            train_prefix = processed_dir / f"{target}_train"
            val_prefix = processed_dir / f"{target}_val"
            
            generate_multi_replicate_data(str(train_prefix), n_samples=50, random_seed=42)
            generate_multi_replicate_data(str(val_prefix), n_samples=30, random_seed=43)
            
            try:
                with patch('wandb.init') as mock_init, \
                     patch('wandb.log') as mock_log, \
                     patch('wandb.finish') as mock_finish:
                    
                    mock_init.return_value = MagicMock()
                    
                    # Configure overrides for legacy experiment
                    overrides = [
                        "experiment=legacy_reproduction",
                        f"paths.data_base={data_dir}",
                        f"+target={target}",
                        "+num_epochs=2",
                        "+batch_size=16",
                        "+device=cpu"
                    ]
                    
                    # Set test mode environment
                    env = os.environ.copy()
                    env['CHIPVI_TEST_MODE'] = '1'
                    
                    # Execute training with legacy configuration
                    result = subprocess.run([
                        sys.executable, "-m", "scripts.run",
                        "--config-path", "../configs",
                        "--config-name", "config"
                    ] + overrides,
                    capture_output=True,
                    text=True,
                    cwd=Path.cwd(),
                    env=env,
                    timeout=120
                    )
                    
                    # Training should complete successfully
                    assert result.returncode == 0, f"Legacy config execution failed: {result.stderr}"
                    
                    # Should contain evidence of successful execution
                    assert "Configuration validation completed" in result.stdout or "Using device:" in result.stdout
                    
            except Exception as e:
                pytest.fail(f"End-to-end execution with legacy config failed: {e}")
            finally:
                # Cleanup
                shutil.rmtree(data_dir, ignore_errors=True)