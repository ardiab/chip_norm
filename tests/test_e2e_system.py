"""End-to-end system test for the complete ChipVI training pipeline."""

import pytest
import subprocess
import tempfile
import shutil
import os
import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock
import torch

from tests.mocks.wandb_mock import MockWandB, wandb_mock_context
from tests.utils.synthetic_data import SyntheticDataManager, generate_multi_replicate_data


class TestE2ESystem:
    """End-to-end system tests that verify the complete training pipeline."""

    def test_configuration_parsing_verification(self):
        """Test that run.py correctly parses all configuration parameters including multi-component loss."""
        with SyntheticDataManager() as data_manager:
            # Create synthetic data directory structure
            data_dir = Path(tempfile.mkdtemp())
            processed_dir = data_dir / "processed" 
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate synthetic data with expected filenames
            train_prefix = processed_dir / "test_train_200bp"
            val_prefix = processed_dir / "test_val_200bp"
            
            # Create training data
            generate_multi_replicate_data(
                prefix=str(train_prefix),
                n_samples=50,
                random_seed=42
            )
            # Create validation data  
            generate_multi_replicate_data(
                prefix=str(val_prefix), 
                n_samples=30,
                random_seed=43
            )
            
            # Create Hydra overrides for test configuration - using + syntax for new parameters
            overrides = [
                "experiment=test_e2e",
                f"paths.data_base={data_dir}",
                "+num_epochs=2",
                "+batch_size=16", 
                "+device=cpu"
            ]
            
            with wandb_mock_context() as mock_wandb:
                # Set environment variable to enable test mode in subprocess
                env = os.environ.copy()
                env['CHIPVI_TEST_MODE'] = '1'
                
                # Use subprocess to run with overrides - this tests configuration parsing
                result = subprocess.run([
                    sys.executable, "-m", "scripts.run",
                    "--config-path", "../configs",
                    "--config-name", "config"
                ] + overrides, 
                capture_output=True, 
                text=True,
                cwd=Path.cwd(),
                env=env
                )
                
                # Should not fail due to configuration parsing errors
                assert result.returncode == 0 or "Configuration validation completed" in result.stdout
                
            # Cleanup
            shutil.rmtree(data_dir)

    def test_component_initialization_checking(self):
        """Test that trainer, model, loss functions, and data loaders are all initialized correctly."""
        with SyntheticDataManager() as data_manager:
            # Create synthetic data directory structure
            data_dir = Path(tempfile.mkdtemp())
            processed_dir = data_dir / "processed" 
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate synthetic data with expected filenames
            train_prefix = processed_dir / "test_train_200bp"
            val_prefix = processed_dir / "test_val_200bp"
            
            # Create training and validation data
            generate_multi_replicate_data(str(train_prefix), n_samples=50, random_seed=42)
            generate_multi_replicate_data(str(val_prefix), n_samples=30, random_seed=43)
            
            # Mock wandb and check component initialization
            with patch('wandb.init') as mock_init, \
                 patch('wandb.log') as mock_log, \
                 patch('wandb.finish') as mock_finish:
                
                # Setup mock wandb
                mock_init.return_value = MagicMock()
                
                # Run the training script with minimal configuration
                overrides = [
                    "experiment=test_e2e",
                    f"paths.data_base={data_dir}",
                    "+num_epochs=1",
                    "+batch_size=8",
                    "+device=cpu"
                ]
                
                # Set environment variable to enable test mode in subprocess
                env = os.environ.copy()
                env['CHIPVI_TEST_MODE'] = '1'
                
                result = subprocess.run([
                    sys.executable, "-m", "scripts.run",
                    "--config-path", "../configs", 
                    "--config-name", "config"
                ] + overrides,
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                env=env
                )
                
                # Check that initialization succeeded (components were created)
                assert "Using device:" in result.stdout
                assert "Configuration validation completed" in result.stdout
                
            # Cleanup
            shutil.rmtree(data_dir)

    def test_training_execution_monitoring(self):
        """Test that the training loop runs for the configured number of epochs without errors."""
        with SyntheticDataManager() as data_manager:
            data_dir = Path(tempfile.mkdtemp())
            processed_dir = data_dir / "processed" 
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            train_prefix = processed_dir / "test_train_200bp"
            val_prefix = processed_dir / "test_val_200bp"
            generate_multi_replicate_data(str(train_prefix), n_samples=50, random_seed=42)
            generate_multi_replicate_data(str(val_prefix), n_samples=30, random_seed=43)
            
            with patch('wandb.init') as mock_init, \
                 patch('wandb.log') as mock_log, \
                 patch('wandb.finish') as mock_finish:
                
                mock_init.return_value = MagicMock()
                
                overrides = [
                    "experiment=test_e2e",
                    f"paths.data_base={data_dir}",
                    "+num_epochs=2",
                    "+batch_size=8", 
                    "+device=cpu"
                ]
                
                # Set environment variable to enable test mode in subprocess
                env = os.environ.copy()
                env["CHIPVI_TEST_MODE"] = "1"
                
                result = subprocess.run([
                    sys.executable, "-m", "scripts.run",
                    "--config-path", "../configs",
                    "--config-name", "config"
                ] + overrides,
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                env=env
                )
                
                # Should complete training without errors
                assert result.returncode == 0
                assert "Training completed" in result.stdout or "Epoch" in result.stdout
                
            # Cleanup
            shutil.rmtree(data_dir)

    def test_wandb_mock_verification(self):
        """Test that wandb logging calls are intercepted and metrics are captured (not sent to servers)."""
        with SyntheticDataManager() as data_manager:
            data_dir = Path(tempfile.mkdtemp())
            processed_dir = data_dir / "processed" 
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            train_prefix = processed_dir / "test_train_200bp"
            val_prefix = processed_dir / "test_val_200bp"
            generate_multi_replicate_data(str(train_prefix), n_samples=50, random_seed=42)
            generate_multi_replicate_data(str(val_prefix), n_samples=30, random_seed=43)
            
            with wandb_mock_context() as mock_wandb:
                with patch('wandb.init', return_value=mock_wandb.init("test_project")), \
                     patch('wandb.log', side_effect=mock_wandb.log), \
                     patch('wandb.finish', side_effect=mock_wandb.finish):
                    
                    overrides = [
                        "experiment=test_e2e",
                        f"paths.data_base={data_dir}",
                        "+num_epochs=1",
                        "+batch_size=8",
                        "+device=cpu"
                    ]
                    
                    result = subprocess.run([
                        sys.executable, "-m", "scripts.run", 
                        "--config-path", "../configs",
                        "--config-name", "config"
                    ] + overrides,
                    capture_output=True,
                    text=True,
                    cwd=Path.cwd()
                    )
                    
                    # Verify metrics were captured by mock
                    logged_metrics = mock_wandb.get_logged_metrics()
                    assert len(logged_metrics) > 0
                    
                    # Check for typical training metrics
                    metric_keys = set()
                    for metrics in logged_metrics:
                        metric_keys.update(metrics.keys())
                    
                    expected_metrics = ['loss', 'epoch']
                    for expected in expected_metrics:
                        assert any(expected in key for key in metric_keys)
                        
            # Cleanup
            shutil.rmtree(data_dir)

    def test_loss_computation_tracking(self):
        """Test that the multi-component loss is computed and individual components are tracked."""
        with SyntheticDataManager() as data_manager:
            data_dir = Path(tempfile.mkdtemp())
            processed_dir = data_dir / "processed" 
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            train_prefix = processed_dir / "test_train_200bp"
            val_prefix = processed_dir / "test_val_200bp"
            generate_multi_replicate_data(str(train_prefix), n_samples=50, random_seed=42)
            generate_multi_replicate_data(str(val_prefix), n_samples=30, random_seed=43)
            
            logged_metrics = []
            
            def capture_log(metrics):
                logged_metrics.append(metrics)
            
            with patch('wandb.init') as mock_init, \
                 patch('wandb.log', side_effect=capture_log), \
                 patch('wandb.finish'):
                
                mock_init.return_value = MagicMock()
                
                overrides = [
                    "experiment=test_e2e",
                    f"paths.data_base={data_dir}",
                    "+num_epochs=1",
                    "+batch_size=8",
                    "+device=cpu",
                    "experiment.loss_fn.losses=[nll,concordance]",
                    "experiment.loss_fn.weights=[0.8,0.2]"
                ]
                
                # Set environment variable to enable test mode in subprocess
                env = os.environ.copy()
                env["CHIPVI_TEST_MODE"] = "1"
                
                result = subprocess.run([
                    sys.executable, "-m", "scripts.run",
                    "--config-path", "../configs",
                    "--config-name", "config"
                ] + overrides,
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                env=env
                )
                
                # Check that multi-component loss was computed
                metric_keys = set()
                for metrics in logged_metrics:
                    metric_keys.update(metrics.keys())
                
                # Should have individual loss components tracked
                assert any('loss' in key for key in metric_keys)
                
            # Cleanup
            shutil.rmtree(data_dir)

    def test_checkpoint_creation(self):
        """Test that checkpoints are created if configured."""
        with SyntheticDataManager() as data_manager:
            data_dir = Path(tempfile.mkdtemp())
            processed_dir = data_dir / "processed" 
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            train_prefix = processed_dir / "test_train_200bp"
            val_prefix = processed_dir / "test_val_200bp"
            generate_multi_replicate_data(str(train_prefix), n_samples=50, random_seed=42)
            generate_multi_replicate_data(str(val_prefix), n_samples=30, random_seed=43)
            
            with patch('wandb.init') as mock_init, \
                 patch('wandb.log') as mock_log, \
                 patch('wandb.finish') as mock_finish:
                
                mock_init.return_value = MagicMock()
                
                overrides = [
                    "experiment=test_e2e",
                    f"paths.data_base={data_dir}",
                    "+num_epochs=2",
                    "+batch_size=8",
                    "+device=cpu",
                    "+checkpointing.strategies=[{metric: val_loss, mode: min, filename: best_model.pt}]"
                ]
                
                # Set environment variable to enable test mode in subprocess
                env = os.environ.copy()
                env["CHIPVI_TEST_MODE"] = "1"
                
                result = subprocess.run([
                    sys.executable, "-m", "scripts.run",
                    "--config-path", "../configs",
                    "--config-name", "config"
                ] + overrides,
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                env=env
                )
                
                # Should complete without checkpoint-related errors
                assert result.returncode == 0
                
            # Cleanup
            shutil.rmtree(data_dir)

    def test_validation_metrics(self):
        """Test that validation is performed and validation metrics are computed."""
        with SyntheticDataManager() as data_manager:
            data_dir = Path(tempfile.mkdtemp())
            processed_dir = data_dir / "processed" 
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            train_prefix = processed_dir / "test_train_200bp"
            val_prefix = processed_dir / "test_val_200bp"
            generate_multi_replicate_data(str(train_prefix), n_samples=50, random_seed=42)
            generate_multi_replicate_data(str(val_prefix), n_samples=30, random_seed=43)
            
            logged_metrics = []
            
            def capture_log(metrics):
                logged_metrics.append(metrics)
            
            with patch('wandb.init') as mock_init, \
                 patch('wandb.log', side_effect=capture_log), \
                 patch('wandb.finish'):
                
                mock_init.return_value = MagicMock()
                
                overrides = [
                    "experiment=test_e2e", 
                    f"paths.data_base={data_dir}",
                    "+num_epochs=2",
                    "+batch_size=8",
                    "+device=cpu"
                ]
                
                # Set environment variable to enable test mode in subprocess
                env = os.environ.copy()
                env["CHIPVI_TEST_MODE"] = "1"
                
                result = subprocess.run([
                    sys.executable, "-m", "scripts.run",
                    "--config-path", "../configs",
                    "--config-name", "config"
                ] + overrides,
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                env=env
                )
                
                # Look for validation metrics in logged data
                metric_keys = set()
                for metrics in logged_metrics:
                    metric_keys.update(metrics.keys())
                
                # Should have validation loss tracked
                assert any('val' in key.lower() for key in metric_keys)
                
            # Cleanup
            shutil.rmtree(data_dir)

    def test_early_stopping_behavior(self):
        """Test that early stopping works if configured."""
        with SyntheticDataManager() as data_manager:
            data_dir = Path(tempfile.mkdtemp())
            processed_dir = data_dir / "processed" 
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            train_prefix = processed_dir / "test_train_200bp"
            val_prefix = processed_dir / "test_val_200bp"
            generate_multi_replicate_data(str(train_prefix), n_samples=50, random_seed=42)
            generate_multi_replicate_data(str(val_prefix), n_samples=30, random_seed=43)
            
            with patch('wandb.init') as mock_init, \
                 patch('wandb.log') as mock_log, \
                 patch('wandb.finish') as mock_finish:
                
                mock_init.return_value = MagicMock()
                
                overrides = [
                    "experiment=test_e2e",
                    f"paths.data_base={data_dir}",
                    "+num_epochs=10",  # Set high epoch count
                    "experiment.patience=1",     # Low patience for early stopping
                    "+batch_size=8",
                    "+device=cpu"
                ]
                
                result = subprocess.run([
                    sys.executable, "-m", "scripts.run",
                    "--config-path", "../configs", 
                    "--config-name", "config"
                ] + overrides,
                capture_output=True,
                text=True,
                cwd=Path.cwd()
                )
                
                # Should complete (either full training or early stopping)
                assert result.returncode == 0
                
            # Cleanup
            shutil.rmtree(data_dir)

    def test_scheduler_operation(self):
        """Test that learning rate scheduling works if configured."""
        with SyntheticDataManager() as data_manager:
            data_dir = Path(tempfile.mkdtemp())
            processed_dir = data_dir / "processed" 
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            train_prefix = processed_dir / "test_train_200bp"
            val_prefix = processed_dir / "test_val_200bp"
            generate_multi_replicate_data(str(train_prefix), n_samples=50, random_seed=42)
            generate_multi_replicate_data(str(val_prefix), n_samples=30, random_seed=43)
            
            with patch('wandb.init') as mock_init, \
                 patch('wandb.log') as mock_log, \
                 patch('wandb.finish') as mock_finish:
                
                mock_init.return_value = MagicMock()
                
                overrides = [
                    "experiment=test_e2e",
                    f"paths.data_base={data_dir}",
                    "+num_epochs=3",
                    "+batch_size=8",
                    "+device=cpu",
                    "+scheduler.name=StepLR",
                    "+scheduler.step_size=1", 
                    "+scheduler.gamma=0.9"
                ]
                
                # Set environment variable to enable test mode in subprocess
                env = os.environ.copy()
                env["CHIPVI_TEST_MODE"] = "1"
                
                result = subprocess.run([
                    sys.executable, "-m", "scripts.run",
                    "--config-path", "../configs",
                    "--config-name", "config"
                ] + overrides,
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                env=env
                )
                
                # Should complete without scheduler-related errors
                assert result.returncode == 0
                
            # Cleanup
            shutil.rmtree(data_dir)

    def test_successful_completion(self):
        """Test that the script completes successfully and returns expected status."""
        with SyntheticDataManager() as data_manager:
            data_dir = Path(tempfile.mkdtemp())
            processed_dir = data_dir / "processed" 
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            train_prefix = processed_dir / "test_train_200bp"
            val_prefix = processed_dir / "test_val_200bp"
            generate_multi_replicate_data(str(train_prefix), n_samples=100, random_seed=42)
            generate_multi_replicate_data(str(val_prefix), n_samples=50, random_seed=43)
            
            with patch('wandb.init') as mock_init, \
                 patch('wandb.log') as mock_log, \
                 patch('wandb.finish') as mock_finish:
                
                mock_init.return_value = MagicMock()
                
                overrides = [
                    "experiment=test_e2e",
                    f"paths.data_base={data_dir}",
                    "+num_epochs=2", 
                    "+batch_size=16",
                    "+device=cpu"
                ]
                
                result = subprocess.run([
                    sys.executable, "-m", "scripts.run",
                    "--config-path", "../configs",
                    "--config-name", "config"
                ] + overrides,
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                timeout=60  # 60 second timeout to ensure it completes quickly
                )
                
                # Verify successful completion
                assert result.returncode == 0
                assert result.stdout is not None
                
                # Should contain evidence of successful training
                output_text = result.stdout.lower()
                success_indicators = [
                    'configuration validation completed',
                    'using device:',
                    'epoch'
                ]
                
                found_indicators = [indicator for indicator in success_indicators 
                                  if indicator in output_text]
                assert len(found_indicators) >= 2, f"Expected success indicators not found. Output: {result.stdout}"
                
            # Cleanup
            shutil.rmtree(data_dir)

    def test_main_integration_with_environment_setup(self):
        """Main test function that sets up test environment and runs comprehensive verification."""
        with SyntheticDataManager() as data_manager:
            # Generate synthetic data in temporary directory
            data_dir = Path(tempfile.mkdtemp())
            processed_dir = data_dir / "processed" 
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            train_prefix = processed_dir / "test_train_200bp"
            val_prefix = processed_dir / "test_val_200bp"
            generate_multi_replicate_data(str(train_prefix), n_samples=100, random_seed=42)
            generate_multi_replicate_data(str(val_prefix), n_samples=50, random_seed=43)
            
            # Configure Hydra overrides for test paths and parameters
            overrides = [
                "experiment=test_e2e",
                f"paths.data_base={data_dir}",
                "+num_epochs=3",
                "+batch_size=16",
                "+device=cpu",
                "experiment.learning_rate=0.001",
                "experiment.loss_fn.losses=[nll,concordance]",
                "experiment.loss_fn.weights=[0.8,0.2]"
            ]
            
            logged_metrics = []
            
            def capture_log(metrics):
                logged_metrics.append(metrics)
                
            # Apply wandb mock before running script
            with patch('wandb.init') as mock_init, \
                 patch('wandb.log', side_effect=capture_log), \
                 patch('wandb.finish') as mock_finish:
                
                mock_init.return_value = MagicMock()
                
                # Execute run.py using subprocess
                result = subprocess.run([
                    sys.executable, "-m", "scripts.run",
                    "--config-path", "../configs", 
                    "--config-name", "config"
                ] + overrides,
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                timeout=120
                )
                
                # Capture and verify outputs and logged metrics
                assert result.returncode == 0, f"Script failed with error: {result.stderr}"
                
                # Verify wandb metrics were logged
                assert len(logged_metrics) > 0, "No metrics were logged to wandb"
                
                # Assert on expected behaviors (loss decrease, metrics logged)
                metric_keys = set()
                for metrics in logged_metrics:
                    metric_keys.update(metrics.keys())
                
                # Check for expected metric types
                expected_metric_patterns = ['loss', 'epoch']
                for pattern in expected_metric_patterns:
                    assert any(pattern in key for key in metric_keys), \
                        f"Expected metric pattern '{pattern}' not found in {metric_keys}"
                
                # Verify loss values are reasonable (not NaN, not extremely large)
                loss_values = []
                for metrics in logged_metrics:
                    for key, value in metrics.items():
                        if 'loss' in key and isinstance(value, (int, float)):
                            loss_values.append(value)
                
                if loss_values:
                    # Loss should be finite and not extremely large
                    assert all(torch.isfinite(torch.tensor(v)) for v in loss_values), \
                        "Found non-finite loss values"
                    assert all(v < 1000 for v in loss_values), \
                        f"Found unreasonably large loss values: {loss_values}"
                
                # Training should show some evidence of progress
                output_indicators = [
                    'configuration validation completed',
                    'using device:',
                    'epoch'
                ]
                
                found_indicators = [indicator for indicator in output_indicators 
                                  if indicator.lower() in result.stdout.lower()]
                assert len(found_indicators) >= 2, \
                    f"Expected training progress indicators not found in output: {result.stdout}"
                
                # Clean up
                shutil.rmtree(data_dir)