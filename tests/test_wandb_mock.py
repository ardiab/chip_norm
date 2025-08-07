"""Tests for wandb mock infrastructure."""

import pytest
from unittest.mock import patch
from tests.mocks.wandb_mock import MockWandB, wandb_mock_context


class TestWandBMock:
    """Test cases for wandb mock functionality."""
    
    def test_mock_initialization_basic(self):
        """Test that wandb.init() can be called with basic configuration."""
        mock_wandb = MockWandB()
        
        # Test basic init call
        run = mock_wandb.init(project="test_project")
        
        assert run is not None
        assert mock_wandb.config['project'] == "test_project"
        
    def test_mock_initialization_with_config(self):
        """Test that wandb.init() can be called with various configurations."""
        mock_wandb = MockWandB()
        
        # Test init with multiple parameters
        run = mock_wandb.init(
            project="test_project",
            entity="test_entity", 
            tags=["tag1", "tag2"],
            config={"learning_rate": 0.01, "batch_size": 32}
        )
        
        assert run is not None
        assert mock_wandb.config['project'] == "test_project"
        assert mock_wandb.config['entity'] == "test_entity"
        assert mock_wandb.config['tags'] == ["tag1", "tag2"]
        assert mock_wandb.config['config']['learning_rate'] == 0.01
        assert mock_wandb.config['config']['batch_size'] == 32
        
    def test_log_capture_single_call(self):
        """Test that wandb.log() calls are intercepted and metrics are stored."""
        mock_wandb = MockWandB()
        mock_wandb.init(project="test_project")
        
        # Test single log call
        metrics = {"loss": 0.5, "accuracy": 0.95}
        mock_wandb.log(metrics)
        
        logged_metrics = mock_wandb.get_logged_metrics()
        assert len(logged_metrics) == 1
        assert logged_metrics[0] == metrics
        
    def test_log_capture_multiple_calls(self):
        """Test that multiple wandb.log() calls accumulate metrics properly."""
        mock_wandb = MockWandB()
        mock_wandb.init(project="test_project")
        
        # Test multiple log calls
        metrics1 = {"loss": 0.5, "accuracy": 0.95}
        metrics2 = {"loss": 0.3, "accuracy": 0.97} 
        metrics3 = {"loss": 0.1, "accuracy": 0.99}
        
        mock_wandb.log(metrics1)
        mock_wandb.log(metrics2) 
        mock_wandb.log(metrics3)
        
        logged_metrics = mock_wandb.get_logged_metrics()
        assert len(logged_metrics) == 3
        assert logged_metrics[0] == metrics1
        assert logged_metrics[1] == metrics2
        assert logged_metrics[2] == metrics3
        
    def test_finish_handling(self):
        """Test that wandb.finish() is handled gracefully without errors."""
        mock_wandb = MockWandB()
        mock_wandb.init(project="test_project")
        
        # Should not raise any exceptions
        mock_wandb.finish()
        
        # After finish, should be able to check if finished
        assert mock_wandb.is_finished()
        
    def test_configuration_parameter_capture(self):
        """Test that wandb configuration parameters are captured correctly."""
        mock_wandb = MockWandB()
        
        config_dict = {
            "model_name": "mlp",
            "hidden_layers": [64, 32, 16],
            "dropout": 0.2
        }
        
        mock_wandb.init(
            project="chipvi_test",
            entity="research_team",
            tags=["experiment", "baseline"],
            config=config_dict
        )
        
        # Verify all configuration is captured
        assert mock_wandb.get_project() == "chipvi_test"
        assert mock_wandb.get_entity() == "research_team" 
        assert mock_wandb.get_tags() == ["experiment", "baseline"]
        assert mock_wandb.get_config() == config_dict
        
    def test_context_manager(self):
        """Test the context manager interface for easy use in tests."""
        with wandb_mock_context() as mock_wandb:
            # Inside context, wandb calls should be mocked
            mock_wandb.init(project="test_context")
            mock_wandb.log({"step": 1, "value": 42})
            
            metrics = mock_wandb.get_logged_metrics()
            assert len(metrics) == 1
            assert metrics[0] == {"step": 1, "value": 42}
            
        # After context, mock should be cleaned up
        assert mock_wandb.is_finished()
        
    def test_mock_without_init(self):
        """Test that logging without init raises appropriate error."""
        mock_wandb = MockWandB()
        
        with pytest.raises(RuntimeError, match="wandb.init\\(\\) must be called"):
            mock_wandb.log({"loss": 0.5})
            
    def test_metric_assertion_utilities(self):
        """Test utilities for asserting on logged values."""
        mock_wandb = MockWandB()
        mock_wandb.init(project="test_project")
        
        mock_wandb.log({"loss": 0.5, "accuracy": 0.8})
        mock_wandb.log({"loss": 0.3, "accuracy": 0.9})
        
        # Test utility methods
        assert mock_wandb.get_metric_values("loss") == [0.5, 0.3]
        assert mock_wandb.get_metric_values("accuracy") == [0.8, 0.9]
        assert mock_wandb.get_latest_metric("loss") == 0.3
        assert mock_wandb.get_latest_metric("accuracy") == 0.9