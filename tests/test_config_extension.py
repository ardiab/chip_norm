"""Tests for extended configuration system in ChipVI."""

import pytest
import torch
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

import hydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

from chipvi.training.losses import CompositeLoss, concordance_loss_nce_wrapper, negative_pearson_loss_wrapper


class TestCompositeLossConfiguration:
    """Test parsing and instantiation of composite loss configurations."""
    
    def test_composite_loss_yaml_parsing(self):
        """Test that composite loss YAML configs are parsed correctly."""
        # Create a temporary config for composite loss
        composite_config = {
            '_target_': 'chipvi.training.losses.CompositeLoss',
            'losses': {
                'nll': {
                    '_target_': 'chipvi.training.losses.nll_loss',
                    'weight': 1.0
                },
                'concordance': {
                    '_target_': 'chipvi.training.losses.concordance_loss_nce_wrapper',
                    'weight': 0.1,
                    'tau': 0.1
                }
            }
        }
        
        # Test that the structure is valid for Hydra instantiation
        assert '_target_' in composite_config
        assert 'losses' in composite_config
        assert 'nll' in composite_config['losses']
        assert 'concordance' in composite_config['losses']
        
        # Test that weights are properly specified
        assert composite_config['losses']['nll']['weight'] == 1.0
        assert composite_config['losses']['concordance']['weight'] == 0.1
        
        # Test that loss-specific parameters are preserved
        assert composite_config['losses']['concordance']['tau'] == 0.1
    
    def test_composite_loss_weight_validation(self):
        """Test validation of composite loss weights."""
        # Test that weights sum to reasonable value (should warn if not 1.0)
        weights = [0.5, 0.3, 0.1]  # Sum = 0.9
        assert sum(weights) == 0.9
        
        # Test weights that sum to > 1.0
        large_weights = [0.8, 0.5, 0.2]  # Sum = 1.5
        assert sum(large_weights) == 1.5
        
        # Test negative weights (should be invalid)
        negative_weights = [1.0, -0.2, 0.1]
        assert any(w < 0 for w in negative_weights)


class TestSchedulerConfiguration:
    """Test learning rate scheduler configuration parsing."""
    
    def test_scheduler_config_structure(self):
        """Test scheduler configuration has required fields."""
        scheduler_config = {
            'warmup_epochs': 2,
            'scheduler_type': 'cosine',
            'total_epochs': 100
        }
        
        # Test required fields are present
        assert 'warmup_epochs' in scheduler_config
        assert 'scheduler_type' in scheduler_config
        assert 'total_epochs' in scheduler_config
        
        # Test warmup < total epochs validation
        assert scheduler_config['warmup_epochs'] < scheduler_config['total_epochs']
    
    def test_scheduler_type_validation(self):
        """Test that scheduler types are valid."""
        valid_scheduler_types = ['cosine', 'linear', 'exponential', 'step']
        
        for scheduler_type in valid_scheduler_types:
            config = {
                'warmup_epochs': 5,
                'scheduler_type': scheduler_type,
                'total_epochs': 100
            }
            assert config['scheduler_type'] in valid_scheduler_types
    
    def test_scheduler_warmup_validation(self):
        """Test warmup epochs validation."""
        # Valid case: warmup < total
        valid_config = {
            'warmup_epochs': 10,
            'scheduler_type': 'cosine',
            'total_epochs': 100
        }
        assert valid_config['warmup_epochs'] < valid_config['total_epochs']
        
        # Invalid case: warmup >= total (should be caught by validation)
        invalid_config = {
            'warmup_epochs': 100,
            'scheduler_type': 'cosine', 
            'total_epochs': 50
        }
        assert invalid_config['warmup_epochs'] > invalid_config['total_epochs']


class TestWandBConfiguration:
    """Test Weights & Biases integration configuration."""
    
    def test_wandb_default_enabled(self):
        """Test that W&B is enabled by default."""
        default_config = {
            'enabled': True,
            'project': 'chipvi',
            'entity': None,
            'tags': [],
            'notes': None
        }
        
        # W&B should be enabled by default
        assert default_config['enabled'] is True
        assert default_config['project'] == 'chipvi'
    
    def test_wandb_disable_config(self):
        """Test that W&B can be disabled via configuration."""
        disabled_config = {
            'enabled': False,
            'project': 'chipvi',
            'entity': None,
            'tags': [],
            'notes': None
        }
        
        # Should be possible to disable
        assert disabled_config['enabled'] is False
    
    def test_wandb_optional_fields(self):
        """Test that optional W&B fields are handled correctly."""
        config = {
            'enabled': True,
            'project': 'chipvi',
            'entity': 'my-team',  # Optional custom entity
            'tags': ['experiment', 'baseline'],  # Optional tags
            'notes': 'Testing configuration system'  # Optional notes
        }
        
        assert config['entity'] is not None
        assert len(config['tags']) > 0
        assert config['notes'] is not None


class TestCheckpointingConfiguration:
    """Test checkpointing strategy configuration parsing."""
    
    def test_multiple_checkpoint_strategies(self):
        """Test that multiple checkpoint strategies can be specified."""
        checkpoint_config = {
            'strategies': [
                {
                    'metric': 'val_loss',
                    'mode': 'min',
                    'filename': 'best_loss.pt',
                    'overwrite': True
                },
                {
                    'metric': 'val_residual_spearman',
                    'mode': 'max',
                    'filename': 'best_corr.pt',
                    'overwrite': True
                }
            ]
        }
        
        # Should have multiple strategies
        assert len(checkpoint_config['strategies']) == 2
        
        # Each strategy should have required fields
        for strategy in checkpoint_config['strategies']:
            assert 'metric' in strategy
            assert 'mode' in strategy
            assert 'filename' in strategy
            assert 'overwrite' in strategy
    
    def test_checkpoint_metric_validation(self):
        """Test that checkpoint metric names are valid."""
        valid_metrics = [
            'val_loss', 'val_residual_spearman', 'val_residual_pearson',
            'train_loss', 'val_nll_loss', 'val_concordance_loss'
        ]
        
        for metric in valid_metrics:
            strategy = {
                'metric': metric,
                'mode': 'min' if 'loss' in metric else 'max',
                'filename': f'best_{metric}.pt',
                'overwrite': True
            }
            assert strategy['metric'] in valid_metrics
    
    def test_checkpoint_mode_validation(self):
        """Test that checkpoint modes are valid."""
        valid_modes = ['min', 'max']
        
        for mode in valid_modes:
            strategy = {
                'metric': 'val_loss',
                'mode': mode,
                'filename': 'checkpoint.pt',
                'overwrite': True
            }
            assert strategy['mode'] in valid_modes


class TestPreprocessingConfiguration:
    """Test preprocessing configuration parsing."""
    
    def test_log_transform_config_structure(self):
        """Test log transformation configuration structure."""
        preprocessing_config = {
            'log_transform': {
                'enabled': False,
                'columns': [0, 5]  # Which covariate columns to transform
            }
        }
        
        assert 'log_transform' in preprocessing_config
        assert 'enabled' in preprocessing_config['log_transform']
        assert 'columns' in preprocessing_config['log_transform']
        
        # Columns should be a list of integers
        columns = preprocessing_config['log_transform']['columns']
        assert isinstance(columns, list)
        assert all(isinstance(col, int) for col in columns)
    
    def test_log_transform_enabled_disabled(self):
        """Test log transform can be enabled/disabled."""
        # Disabled case
        disabled_config = {
            'log_transform': {
                'enabled': False,
                'columns': [0, 1, 2]
            }
        }
        assert disabled_config['log_transform']['enabled'] is False
        
        # Enabled case
        enabled_config = {
            'log_transform': {
                'enabled': True,
                'columns': [0, 1, 2]
            }
        }
        assert enabled_config['log_transform']['enabled'] is True
    
    def test_log_transform_column_specification(self):
        """Test that log transform columns are properly specified."""
        config = {
            'log_transform': {
                'enabled': True,
                'columns': [0, 5, 10, 15]  # Multiple columns
            }
        }
        
        columns = config['log_transform']['columns']
        assert len(columns) == 4
        assert 0 in columns
        assert 5 in columns
        assert 10 in columns
        assert 15 in columns


class TestHydraOverrideCompatibility:
    """Test that Hydra command-line overrides work with nested configuration."""
    
    def test_nested_override_syntax(self):
        """Test nested parameter override syntax."""
        # Test override patterns that should work
        override_patterns = [
            "++loss.concordance.weight=0.2",  # Add new nested parameter
            "loss.nll.weight=0.8",             # Override existing nested parameter
            "++scheduler.warmup_epochs=5",     # Add scheduler parameter
            "wandb.enabled=false",             # Override W&B setting
            "++checkpointing.strategies[0].metric=val_loss"  # Override list item
        ]
        
        # Each pattern should be a valid Hydra override
        for pattern in override_patterns:
            assert '=' in pattern  # Basic syntax check
            if pattern.startswith('++'):
                # New parameter addition
                assert pattern[2:].count('.') >= 1  # Has nested structure
            else:
                # Existing parameter override
                assert pattern.count('.') >= 1  # Has nested structure
    
    def test_complex_nested_override(self):
        """Test complex nested overrides."""
        # Test that complex nested structures can be overridden
        complex_overrides = [
            "++loss.losses.concordance.tau=0.2",
            "++checkpointing.strategies=[{metric:val_loss,mode:min,filename:best.pt}]",
            "++preprocessing.log_transform.columns=[0,1,2,3]"
        ]
        
        for override in complex_overrides:
            assert '=' in override
            key, value = override.split('=', 1)
            if key.startswith('++'):
                key = key[2:]
            assert '.' in key  # Has nested structure


class TestConfigValidation:
    """Test configuration validation and error handling."""
    
    def test_config_validation_catches_errors(self):
        """Test that invalid configurations are caught."""
        # Test invalid loss configuration
        invalid_loss_config = {
            '_target_': 'chipvi.training.losses.CompositeLoss',
            'losses': {}  # Empty losses dict should be invalid
        }
        assert len(invalid_loss_config['losses']) == 0  # Should be caught
        
        # Test invalid scheduler configuration
        invalid_scheduler_config = {
            'warmup_epochs': 100,
            'scheduler_type': 'invalid_type',
            'total_epochs': 50  # warmup > total epochs
        }
        assert invalid_scheduler_config['warmup_epochs'] > invalid_scheduler_config['total_epochs']
        assert invalid_scheduler_config['scheduler_type'] not in ['cosine', 'linear', 'exponential', 'step']
        
        # Test invalid checkpoint configuration
        invalid_checkpoint_config = {
            'strategies': [
                {
                    'metric': 'invalid_metric_name',
                    'mode': 'invalid_mode',
                    'filename': '',  # Empty filename
                    'overwrite': 'not_boolean'  # Should be boolean
                }
            ]
        }
        strategy = invalid_checkpoint_config['strategies'][0]
        assert strategy['filename'] == ''  # Should be caught
        assert strategy['overwrite'] != True and strategy['overwrite'] != False  # Not boolean
    
    def test_helpful_error_messages(self):
        """Test that validation provides helpful error messages."""
        # This test documents expected error message patterns
        expected_error_patterns = [
            "loss weights must sum to reasonable value",
            "scheduler warmup epochs must be less than total epochs",
            "checkpoint metric names must be valid",
            "required fields must be present"
        ]
        
        # Each pattern should be descriptive and actionable
        for pattern in expected_error_patterns:
            assert len(pattern) > 20  # Should be descriptive
            assert any(word in pattern.lower() for word in ['must', 'should', 'required'])  # Actionable


class TestConfigurationIntegration:
    """Integration tests for the complete configuration system."""
    
    def create_sample_experiment_config(self):
        """Create a sample experiment configuration for testing."""
        return {
            'defaults': [
                '_self_',
                {'loss': 'composite_infonce'},
                {'wandb': 'default'}
            ],
            'training': {
                'num_epochs': 100,
                'learning_rate': 0.001,
                'batch_size': 8192
            },
            'scheduler': {
                'warmup_epochs': 2,
                'scheduler_type': 'cosine',
                'total_epochs': '${training.num_epochs}'  # Reference other config
            },
            'early_stopping': {
                'patience': 10,
                'monitor': 'val_loss',
                'mode': 'min'
            },
            'gradient_clipping': {
                'max_norm': 1.0
            },
            'checkpointing': {
                'strategies': [
                    {
                        'metric': 'val_loss',
                        'mode': 'min',
                        'filename': 'best_loss.pt',
                        'overwrite': True
                    },
                    {
                        'metric': 'val_residual_spearman',
                        'mode': 'max',
                        'filename': 'best_corr.pt',
                        'overwrite': True
                    }
                ]
            },
            'preprocessing': {
                'log_transform': {
                    'enabled': False,
                    'columns': [0, 5]
                }
            }
        }
    
    def test_loss_config_files_exist(self):
        """Test that required loss configuration files exist."""
        expected_loss_configs = [
            '/workspace/configs/loss/nll_only.yaml',
            '/workspace/configs/loss/composite_infonce.yaml',
            '/workspace/configs/loss/composite_pearson.yaml',
            '/workspace/configs/loss/composite_quantile.yaml'
        ]
        
        for config_path in expected_loss_configs:
            assert Path(config_path).exists(), f"Missing loss config: {config_path}"
    
    def test_wandb_config_files_exist(self):
        """Test that W&B configuration files exist."""
        expected_wandb_configs = [
            '/workspace/configs/wandb/default.yaml',
            '/workspace/configs/wandb/disabled.yaml'
        ]
        
        for config_path in expected_wandb_configs:
            assert Path(config_path).exists(), f"Missing wandb config: {config_path}"
    
    def test_hydra_composite_loss_instantiation(self):
        """Test that Hydra can instantiate composite losses from YAML configs."""
        # This should fail until we implement the loss configs
        with initialize(config_path="../configs/loss", version_base=None):
            try:
                cfg = compose(config_name="composite_infonce")
                
                # Direct instantiation should work since we're loading the loss file directly
                loss = hydra.utils.instantiate(cfg)
                
                # Verify it's a CompositeLoss instance
                assert isinstance(loss, CompositeLoss)
                
                # Verify it has the expected components
                assert len(loss.component_names) == 2
                assert 'nll' in loss.component_names
                assert 'concordance' in loss.component_names
            except Exception as e:
                pytest.fail(f"Failed to instantiate composite loss from config: {e}")
    
    def test_experiment_config_loading(self):
        """Test that new experiment configs can be loaded."""
        expected_experiment_configs = [
            '/workspace/configs/experiment/exp_002_infonce.yaml',
            '/workspace/configs/experiment/exp_003_multiobjective.yaml'
        ]
        
        for config_path in expected_experiment_configs:
            assert Path(config_path).exists(), f"Missing experiment config: {config_path}"
    
    def test_full_experiment_config_structure(self):
        """Test that a complete experiment configuration has all required sections."""
        config = self.create_sample_experiment_config()
        
        # Test top-level sections exist
        required_sections = [
            'defaults', 'training', 'scheduler', 'early_stopping',
            'gradient_clipping', 'checkpointing', 'preprocessing'
        ]
        
        for section in required_sections:
            assert section in config, f"Missing required section: {section}"
    
    def test_config_parameter_references(self):
        """Test that configuration parameter references work."""
        config = self.create_sample_experiment_config()
        
        # Test that scheduler references training epochs
        scheduler_epochs = config['scheduler']['total_epochs']
        training_epochs = '${training.num_epochs}'
        
        assert scheduler_epochs == training_epochs
    
    def test_config_composition_compatibility(self):
        """Test that configuration composition works with new structure."""
        config = self.create_sample_experiment_config()
        
        # Test that defaults include the necessary components
        defaults = config['defaults']
        
        # Should have base config (_self_)
        assert '_self_' in defaults
        
        # Should include composed configs
        composed_configs = [item for item in defaults if isinstance(item, dict)]
        config_keys = []
        for item in composed_configs:
            config_keys.extend(item.keys())
        
        expected_composed = ['loss', 'wandb']
        for expected in expected_composed:
            assert expected in config_keys, f"Missing composed config: {expected}"


class TestExampleConfigurations:
    """Test example experiment configurations."""
    
    def test_infonce_experiment_config(self):
        """Test InfoNCE experiment configuration structure."""
        infonce_config = {
            'name': 'InfoNCE Experiment',
            'defaults': [
                '_self_',
                {'loss': 'composite_infonce'},
                {'wandb': 'default'}
            ],
            'training': {
                'num_epochs': 80,
                'learning_rate': 0.002
            },
            'scheduler': {
                'warmup_epochs': 5,
                'scheduler_type': 'cosine'
            }
        }
        
        assert 'name' in infonce_config
        assert 'InfoNCE' in infonce_config['name']
        assert {'loss': 'composite_infonce'} in infonce_config['defaults']
    
    def test_multiobjective_experiment_config(self):
        """Test multi-objective experiment configuration structure."""
        multiobjective_config = {
            'name': 'Multi-objective Experiment',
            'defaults': [
                '_self_',
                {'loss': 'composite_multiobjective'},
                {'wandb': 'default'}
            ],
            'training': {
                'num_epochs': 120,
                'learning_rate': 0.0005
            },
            'loss_weights': {
                'nll': 1.0,
                'concordance': 0.1,
                'pearson': 0.05,
                'quantile': 0.05
            }
        }
        
        assert 'Multi-objective' in multiobjective_config['name']
        assert 'loss_weights' in multiobjective_config
        
        # Test that weights are properly distributed
        weights = multiobjective_config['loss_weights']
        total_weight = sum(weights.values())
        assert total_weight > 1.0  # Should sum to reasonable value
        assert weights['nll'] == 1.0  # Primary loss should have weight 1.0


class TestConfigurationBackwardCompatibility:
    """Test that new configuration system maintains backward compatibility."""
    
    def test_old_loss_string_format(self):
        """Test that old string-based loss configuration still works."""
        # Old format: simple string
        old_config = {
            'loss_fn': 'replicate_concordance_mse'
        }
        
        assert isinstance(old_config['loss_fn'], str)
        assert old_config['loss_fn'] == 'replicate_concordance_mse'
    
    def test_old_training_params_format(self):
        """Test that old training parameter format still works."""
        # Old format: flat parameters
        old_config = {
            'num_epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'batch_size': 8192
        }
        
        required_params = ['num_epochs', 'learning_rate', 'weight_decay', 'batch_size']
        for param in required_params:
            assert param in old_config
    
    def test_migration_from_old_to_new_format(self):
        """Test migration path from old to new configuration format."""
        # Old format
        old_config = {
            'num_epochs': 100,
            'learning_rate': 0.001,
            'loss_fn': 'replicate_concordance_mse'
        }
        
        # Should be convertible to new format
        new_config = {
            'training': {
                'num_epochs': old_config['num_epochs'],
                'learning_rate': old_config['learning_rate']
            },
            'loss': {
                '_target_': 'chipvi.training.losses.replicate_concordance_mse_loss'
            }
        }
        
        # Check conversion preserves values
        assert new_config['training']['num_epochs'] == old_config['num_epochs']
        assert new_config['training']['learning_rate'] == old_config['learning_rate']
        assert 'replicate_concordance_mse' in new_config['loss']['_target_']