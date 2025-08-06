"""Tests for validation metrics functionality."""

import pytest
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from unittest.mock import MagicMock

from chipvi.utils.distributions import compute_numeric_cdf, get_torch_nb_dist
from chipvi.training.trainer import Trainer


class TestPITComputation:
    """Test Probability Integral Transform (PIT) analysis."""
    
    def test_pit_uniform_for_correct_model(self):
        """Test that PIT produces uniform distribution for correctly specified model."""
        torch.manual_seed(42)
        n_samples = 1000
        
        # True parameters
        mu_true = 10.0
        r_true = 5.0
        
        # Generate data from true distribution
        true_dist = get_torch_nb_dist(r=torch.tensor([r_true]), mu=torch.tensor([mu_true]))
        observations = true_dist.sample((n_samples,)).squeeze()
        
        # Model predictions (same as true parameters - perfect model)
        model_dist = get_torch_nb_dist(
            r=torch.full((n_samples,), r_true),
            mu=torch.full((n_samples,), mu_true)
        )
        
        # Compute PIT values (quantiles)
        pit_values = compute_numeric_cdf(model_dist, observations)
        
        # PIT should be approximately uniform [0,1]
        # Check that mean is close to 0.5 (expected for uniform distribution)
        assert abs(torch.mean(pit_values) - 0.5) < 0.1
        
        # Check that values are reasonably distributed across [0,1] range  
        assert torch.min(pit_values) >= 0.0
        assert torch.max(pit_values) <= 1.0
        
        # Check that standard deviation is close to uniform expectation (1/sqrt(12) ≈ 0.29)
        expected_std = 1.0 / torch.sqrt(torch.tensor(12.0))
        actual_std = torch.std(pit_values)
        assert abs(actual_std - expected_std) < 0.1  # Allow some deviation
        
    def test_pit_non_uniform_for_misspecified_model(self):
        """Test that PIT deviates from uniform for misspecified model."""
        torch.manual_seed(42)
        n_samples = 1000
        
        # True parameters 
        mu_true = 10.0
        r_true = 5.0
        
        # Generate data from true distribution
        true_dist = get_torch_nb_dist(r=torch.tensor([r_true]), mu=torch.tensor([mu_true]))
        observations = true_dist.sample((n_samples,)).squeeze()
        
        # Misspecified model (wrong parameters)
        mu_wrong = 15.0  # Wrong mean
        r_wrong = 2.0    # Wrong dispersion
        model_dist = get_torch_nb_dist(
            r=torch.full((n_samples,), r_wrong),
            mu=torch.full((n_samples,), mu_wrong)
        )
        
        # Compute PIT values
        pit_values = compute_numeric_cdf(model_dist, observations)
        
        # PIT should NOT be uniform
        pit_sorted = torch.sort(pit_values)[0]
        expected_uniform = torch.linspace(0, 1, n_samples + 1)[1:]
        
        # Maximum deviation should be larger for misspecified model
        max_deviation = torch.max(torch.abs(pit_sorted - expected_uniform))
        
        # Should show clear deviation from uniform
        assert max_deviation > 0.1


class TestPlotGeneration:
    """Test validation plot generation."""
    
    def test_create_validation_figure_structure(self):
        """Test that validation figure has correct 3x3 subplot structure."""
        from chipvi.training.trainer import create_validation_figure
        
        # Mock metrics dictionary
        metrics_dict = {
            'r1_pred': np.random.randn(1000),
            'r1_obs': np.random.randn(1000),
            'r2_pred': np.random.randn(1000), 
            'r2_obs': np.random.randn(1000),
            'r1_res': np.random.randn(1000),
            'r2_res_scaled': np.random.randn(1000),
            'r1_quant': np.random.rand(1000),
            'r2_quant': np.random.rand(1000),
        }
        
        # Create validation figure
        fig = create_validation_figure(metrics_dict)
        
        # Check that figure was created
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Clean up
        plt.close(fig)
            
    def test_plot_has_nine_panels(self):
        """Test that validation plot contains 3x3 main subplots (plus colorbars)."""
        from chipvi.training.trainer import create_validation_figure
        
        metrics_dict = {
            'r1_pred': np.random.randn(100),
            'r1_obs': np.random.randn(100),
            'r2_pred': np.random.randn(100),
            'r2_obs': np.random.randn(100), 
            'r1_res': np.random.randn(100),
            'r2_res_scaled': np.random.randn(100),
            'r1_quant': np.random.rand(100),
            'r2_quant': np.random.rand(100),
        }
        
        fig = create_validation_figure(metrics_dict)
        
        # Check figure has reasonable number of axes (9 main + colorbars)
        axes = fig.get_axes()
        assert len(axes) >= 9  # At least 9 main subplots, may have additional colorbars
        
        # Check figure size
        assert fig.get_figwidth() == 18
        assert fig.get_figheight() == 18
        
        plt.close(fig)


class TestMetricComputationEdgeCases:
    """Test validation metrics with edge cases."""
    
    def test_empty_batch_handling(self):
        """Test that metrics computation handles empty batches gracefully."""
        from chipvi.training.trainer import Trainer
        
        # Create a mock trainer to test the method
        trainer = Trainer(
            model=MagicMock(),
            train_loader=MagicMock(),
            val_loader=MagicMock(),
            optimizer=MagicMock(),
            loss_fn=MagicMock(),
            device=torch.device('cpu')
        )
        
        # Empty lists/arrays
        empty_residuals_r1 = []
        empty_residuals_r2 = []
        empty_quantiles_r1 = []
        empty_quantiles_r2 = []
        empty_predictions_r1 = []
        empty_predictions_r2 = []
        empty_observations_r1 = []
        empty_observations_r2 = []
        
        # This should not crash and return appropriate default values
        result = trainer._compute_validation_metrics(
            empty_residuals_r1, 
            empty_residuals_r2,
            empty_quantiles_r1,
            empty_quantiles_r2,
            empty_predictions_r1,
            empty_predictions_r2,
            empty_observations_r1,
            empty_observations_r2
        )
        
        # Should return default values for empty inputs
        assert result['val_residual_spearman'] == 0.0
        assert result['val_quantile_spearman'] == 0.0
        assert result['metrics_dict'] is None
    
    def test_nan_value_handling(self):
        """Test that metrics computation handles NaN values gracefully."""
        from chipvi.training.trainer import Trainer
        
        # Create a mock trainer to test the method
        trainer = Trainer(
            model=MagicMock(),
            train_loader=MagicMock(),
            val_loader=MagicMock(),
            optimizer=MagicMock(),
            loss_fn=MagicMock(),
            device=torch.device('cpu')
        )
        
        # Create data with NaN values
        residuals_r1 = [1.0, 2.0, float('nan'), 4.0]
        residuals_r2 = [1.5, float('nan'), 3.0, 4.5]
        quantiles_r1 = [0.1, 0.2, float('nan'), 0.4]
        quantiles_r2 = [0.15, float('nan'), 0.35, 0.45]
        predictions_r1 = [10.0, 11.0, 12.0, 13.0]
        predictions_r2 = [10.5, 11.5, 12.5, 13.5]
        observations_r1 = [11.0, 13.0, 12.0, 17.0]
        observations_r2 = [12.0, 11.5, 15.5, 18.0]
        
        # This should handle NaN values without crashing
        result = trainer._compute_validation_metrics(
            residuals_r1,
            residuals_r2, 
            quantiles_r1,
            quantiles_r2,
            predictions_r1,
            predictions_r2,
            observations_r1,
            observations_r2
        )
        
        # Should return valid correlation values (NaN handling should work)
        assert isinstance(result['val_residual_spearman'], (float, np.float64))
        assert isinstance(result['val_quantile_spearman'], (float, np.float64))
        assert not np.isnan(result['val_residual_spearman'])
        assert not np.isnan(result['val_quantile_spearman'])


class TestHist2DAccuracy:
    """Test 2D histogram utility function."""
    
    def test_hist2d_basic_functionality(self):
        """Test that hist2d creates appropriate density plots."""
        from chipvi.utils.plots import hist2d
        
        # Create test data
        np.random.seed(42)
        n_points = 1000
        x = np.random.randn(n_points)
        y = x + 0.5 * np.random.randn(n_points)  # Correlated data
        
        df = pd.DataFrame({'x_col': x, 'y_col': y})
        
        # Create plot
        fig, ax = plt.subplots()
        hist2d(df, 'x_col', 'y_col', ax)
        
        # Check that plot was created
        assert ax.get_xlabel() == 'x_col'
        assert ax.get_ylabel() == 'y_col'
        
        # Check that title contains correlation info
        title = ax.get_title()
        assert 'Pearson:' in title
        assert 'Spearman:' in title
        
        plt.close(fig)
        