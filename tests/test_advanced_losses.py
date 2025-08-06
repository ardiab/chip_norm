"""Tests for advanced loss functions in ChipVI training."""

import pytest
import torch
import torch.distributions as D
from scipy.stats import pearsonr

from chipvi.training.losses import (
    concordance_loss_nce,
    negative_pearson_loss, 
    quantile_absolute_loss,
    CompositeLoss,
    concordance_loss_nce_wrapper,
    negative_pearson_loss_wrapper,
    quantile_absolute_loss_wrapper
)


class TestInfoNCELoss:
    """Test InfoNCE concordance loss function."""
    
    def test_concordance_loss_nce_matched_vs_mismatched(self):
        """Test that InfoNCE loss returns lower values for matched pairs than mismatched pairs."""
        # Create simple test data where r1 and r2 are similar (matched case)
        res1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # Shape: (2, 2)
        res2 = torch.tensor([[1.1, 2.1], [3.1, 4.1]])  # Very similar to res1
        
        # Compute InfoNCE loss - should be low for matched pairs
        loss = concordance_loss_nce(res1, res2, tau=0.1)
        
        # Loss should be a scalar tensor
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        
        # Create mismatched case where r1 and r2 are very different
        res1_mismatch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        res2_mismatch = torch.tensor([[10.0, 20.0], [30.0, 40.0]])  # Very different from res1
        
        loss_mismatch = concordance_loss_nce(res1_mismatch, res2_mismatch, tau=0.1)
        
        # Loss should be higher for mismatched pairs
        assert loss < loss_mismatch
    
    def test_concordance_loss_nce_identical_inputs(self):
        """Test InfoNCE loss with identical inputs."""
        res = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        loss = concordance_loss_nce(res, res, tau=0.1)
        
        # Loss should be close to 0 for identical inputs (perfect concordance)
        assert loss < 1e-5
        assert not torch.isnan(loss)
    
    def test_concordance_loss_nce_tau_parameter(self):
        """Test that tau parameter affects the loss magnitude."""
        res1 = torch.tensor([[1.0, 2.0]])
        res2 = torch.tensor([[1.5, 2.5]])
        
        loss_small_tau = concordance_loss_nce(res1, res2, tau=0.1)
        loss_large_tau = concordance_loss_nce(res1, res2, tau=1.0)
        
        # Different tau values should give different losses
        assert not torch.allclose(loss_small_tau, loss_large_tau)


class TestNegativePearsonLoss:
    """Test negative Pearson correlation loss function."""
    
    def test_negative_pearson_perfect_correlation(self):
        """Test that perfectly correlated inputs return -1."""
        preds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])  # Perfect positive correlation
        
        loss = negative_pearson_loss(preds, targets)
        
        # Should return close to -1 for perfect positive correlation
        assert torch.allclose(loss, torch.tensor(-1.0), atol=1e-6)
    
    def test_negative_pearson_uncorrelated_data(self):
        """Test that uncorrelated data approaches 0."""
        torch.manual_seed(42)  # For reproducibility
        preds = torch.randn(1000)
        targets = torch.randn(1000)  # Independent random data
        
        loss = negative_pearson_loss(preds, targets)
        
        # Should be close to 0 for uncorrelated data (large sample)
        assert abs(loss.item()) < 0.1  # Allow some variance for random data
    
    def test_negative_pearson_vs_scipy(self):
        """Test that our implementation matches scipy pearsonr."""
        preds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = torch.tensor([1.1, 1.9, 3.2, 3.8, 5.1])
        
        # Our implementation
        our_loss = negative_pearson_loss(preds, targets)
        
        # Scipy implementation
        scipy_corr, _ = pearsonr(preds.numpy(), targets.numpy())
        expected_loss = -scipy_corr
        
        assert torch.allclose(our_loss, torch.tensor(expected_loss), atol=1e-6)
    
    def test_negative_pearson_identical_inputs(self):
        """Test with identical inputs."""
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        loss = negative_pearson_loss(data, data)
        
        # Perfect correlation should give -1
        assert torch.allclose(loss, torch.tensor(-1.0), atol=1e-6)
    
    def test_negative_pearson_numerical_stability(self):
        """Test numerical stability with edge cases."""
        # Test with zeros
        zeros = torch.zeros(5)
        ones = torch.ones(5)
        loss = negative_pearson_loss(zeros, ones)
        assert not torch.isnan(loss)
        
        # Test with very large values
        large_preds = torch.tensor([1e6, 2e6, 3e6])
        large_targets = torch.tensor([1.1e6, 2.1e6, 3.1e6])
        loss = negative_pearson_loss(large_preds, large_targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestQuantileAbsoluteLoss:
    """Test quantile absolute difference loss function."""
    
    def test_quantile_loss_identical_distributions(self):
        """Test quantile loss with identical model outputs and targets."""
        # Create identical model outputs and targets
        model_r1 = {'mu': torch.tensor([5.0, 10.0]), 'r': torch.tensor([2.0, 3.0])}
        model_r2 = {'mu': torch.tensor([5.0, 10.0]), 'r': torch.tensor([2.0, 3.0])}
        y_r1 = torch.tensor([4.0, 9.0])
        y_r2 = torch.tensor([4.0, 9.0])
        
        loss = quantile_absolute_loss(model_r1, model_r2, y_r1, y_r2)
        
        # Loss should be very small for identical distributions and observations
        assert loss < 1e-5  # Allow for some numerical precision issues
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_quantile_loss_different_distributions(self):
        """Test quantile loss with different model parameters."""
        model_r1 = {'mu': torch.tensor([5.0]), 'r': torch.tensor([2.0])}
        model_r2 = {'mu': torch.tensor([10.0]), 'r': torch.tensor([4.0])}  # Different distribution
        y_r1 = torch.tensor([5.0])
        y_r2 = torch.tensor([5.0])
        
        loss = quantile_absolute_loss(model_r1, model_r2, y_r1, y_r2)
        
        # Loss should be non-zero for different distributions
        assert loss > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_quantile_loss_batch_compatibility(self):
        """Test that quantile loss works with batched data."""
        batch_size = 3
        model_r1 = {
            'mu': torch.tensor([5.0, 10.0, 15.0]), 
            'r': torch.tensor([2.0, 3.0, 4.0])
        }
        model_r2 = {
            'mu': torch.tensor([6.0, 11.0, 16.0]), 
            'r': torch.tensor([2.1, 3.1, 4.1])
        }
        y_r1 = torch.tensor([4.0, 9.0, 14.0])
        y_r2 = torch.tensor([5.0, 10.0, 15.0])
        
        loss = quantile_absolute_loss(model_r1, model_r2, y_r1, y_r2)
        
        # Should return a scalar loss
        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestCompositeLoss:
    """Test composite loss class."""
    
    def test_composite_loss_initialization(self):
        """Test CompositeLoss initialization."""
        # Create simple mock loss functions
        def loss1(model_out, batch):
            return torch.tensor(1.0)
        
        def loss2(model_out, batch):
            return torch.tensor(2.0)
        
        loss_functions = [loss1, loss2]
        weights = [0.5, 0.3]
        
        composite = CompositeLoss(loss_functions, weights)
        
        # Test that initialization works
        assert len(composite.loss_functions) == 2
        assert len(composite.weights) == 2
    
    def test_composite_loss_weighted_sum(self):
        """Test that composite loss returns correct weighted sum."""
        def loss1(model_out, batch):
            return torch.tensor(2.0)
        
        def loss2(model_out, batch):
            return torch.tensor(4.0)
        
        loss_functions = [loss1, loss2]
        weights = [0.5, 0.25]  # Expected total: 0.5*2.0 + 0.25*4.0 = 1.0 + 1.0 = 2.0
        
        composite = CompositeLoss(loss_functions, weights)
        
        # Mock inputs (not used by our simple loss functions)
        model_out = {}
        batch = {}
        
        total_loss = composite(model_out, batch)
        
        # Check that weighted sum is correct
        expected_total = torch.tensor(2.0)
        assert torch.allclose(total_loss, expected_total)
    
    def test_composite_loss_with_components(self):
        """Test composite loss returns individual components when requested."""
        def loss1(model_out, batch):
            return torch.tensor(3.0)
        
        def loss2(model_out, batch):
            return torch.tensor(6.0)
        
        loss_functions = [loss1, loss2]
        weights = [0.4, 0.6]
        
        composite = CompositeLoss(loss_functions, weights, return_components=True)
        
        model_out = {}
        batch = {}
        
        result = composite(model_out, batch)
        
        # Should return a dictionary with total and components
        assert isinstance(result, dict)
        assert 'total' in result
        assert 'components' in result
        assert len(result['components']) == 2
        
        # Check total is correct weighted sum
        expected_total = 0.4 * 3.0 + 0.6 * 6.0  # = 1.2 + 3.6 = 4.8
        assert torch.allclose(result['total'], torch.tensor(expected_total))
    
    def test_composite_loss_gradient_flow(self):
        """Test that gradients flow through composite loss."""
        # Create a simple parameter that requires gradients
        param = torch.tensor([1.0], requires_grad=True)
        
        def loss1(model_out, batch):
            return (param ** 2).sum()
        
        def loss2(model_out, batch):
            return (param * 2).sum()
        
        loss_functions = [loss1, loss2]
        weights = [0.5, 0.5]
        
        composite = CompositeLoss(loss_functions, weights)
        
        model_out = {}
        batch = {}
        
        total_loss = composite(model_out, batch)
        total_loss.backward()
        
        # Check that gradients were computed
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()


class TestWrapperFunctions:
    """Test wrapper functions that interface with MultiReplicateDataset format."""
    
    def create_mock_batch_and_model_outputs(self):
        """Create mock batch and model_outputs dictionaries matching MultiReplicateDataset format."""
        batch = {
            'r1': {'reads': torch.tensor([5.0, 10.0, 15.0])},
            'r2': {'reads': torch.tensor([4.0, 9.0, 14.0])},
            'metadata': {'sd_ratio': torch.tensor([1.0, 1.0, 1.0])}
        }
        
        model_outputs = {
            'r1': {
                'mu': torch.tensor([5.5, 9.5, 14.5]),
                'r': torch.tensor([2.0, 3.0, 4.0])
            },
            'r2': {
                'mu': torch.tensor([4.5, 8.5, 13.5]),
                'r': torch.tensor([2.1, 3.1, 4.1])
            }
        }
        
        return batch, model_outputs
    
    def test_concordance_loss_nce_wrapper(self):
        """Test that concordance NCE wrapper works with MultiReplicateDataset format."""
        batch, model_outputs = self.create_mock_batch_and_model_outputs()
        
        # Call wrapper function
        loss = concordance_loss_nce_wrapper(model_outputs, batch)
        
        # Should return a scalar tensor
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        
        # Verify it matches direct call to core function
        y_r1 = batch['r1']['reads']
        y_r2 = batch['r2']['reads']
        mu_r1 = model_outputs['r1']['mu']
        mu_r2 = model_outputs['r2']['mu']
        res1 = y_r1 - mu_r1
        res2 = y_r2 - mu_r2
        direct_loss = concordance_loss_nce(res1, res2)
        
        assert torch.allclose(loss, direct_loss)
    
    def test_negative_pearson_loss_wrapper(self):
        """Test that negative Pearson wrapper works with MultiReplicateDataset format."""
        batch, model_outputs = self.create_mock_batch_and_model_outputs()
        
        # Call wrapper function
        loss = negative_pearson_loss_wrapper(model_outputs, batch)
        
        # Should return a scalar tensor
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        
        # Verify it matches direct call to core function
        targets_r1 = batch['r1']['reads']
        targets_r2 = batch['r2']['reads']
        preds_r1 = model_outputs['r1']['mu']
        preds_r2 = model_outputs['r2']['mu']
        all_preds = torch.cat([preds_r1, preds_r2])
        all_targets = torch.cat([targets_r1, targets_r2])
        direct_loss = negative_pearson_loss(all_preds, all_targets)
        
        assert torch.allclose(loss, direct_loss)
    
    def test_quantile_absolute_loss_wrapper(self):
        """Test that quantile absolute loss wrapper works with MultiReplicateDataset format."""
        batch, model_outputs = self.create_mock_batch_and_model_outputs()
        
        # Call wrapper function
        loss = quantile_absolute_loss_wrapper(model_outputs, batch)
        
        # Should return a scalar tensor
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        
        # Verify it matches direct call to core function
        y_r1 = batch['r1']['reads']
        y_r2 = batch['r2']['reads']
        model_r1 = model_outputs['r1']
        model_r2 = model_outputs['r2']
        direct_loss = quantile_absolute_loss(model_r1, model_r2, y_r1, y_r2)
        
        assert torch.allclose(loss, direct_loss)
    
    def test_wrapper_functions_with_composite_loss(self):
        """Test that wrapper functions work correctly with CompositeLoss."""
        batch, model_outputs = self.create_mock_batch_and_model_outputs()
        
        # Create composite loss with wrapper functions
        loss_functions = [
            concordance_loss_nce_wrapper,
            negative_pearson_loss_wrapper,
            quantile_absolute_loss_wrapper
        ]
        weights = [0.4, 0.3, 0.3]
        
        composite = CompositeLoss(loss_functions, weights)
        
        # Compute composite loss
        total_loss = composite(model_outputs, batch)
        
        # Should return a scalar tensor
        assert total_loss.dim() == 0
        assert not torch.isnan(total_loss)
        assert not torch.isinf(total_loss)
        
        # Verify it equals the weighted sum of individual components
        loss1 = concordance_loss_nce_wrapper(model_outputs, batch)
        loss2 = negative_pearson_loss_wrapper(model_outputs, batch)
        loss3 = quantile_absolute_loss_wrapper(model_outputs, batch)
        expected_total = 0.4 * loss1 + 0.3 * loss2 + 0.3 * loss3
        
        assert torch.allclose(total_loss, expected_total)


class TestNumericalStability:
    """Test numerical stability of all loss functions."""
    
    def test_all_losses_handle_edge_cases(self):
        """Test that all losses handle edge cases without NaN or inf."""
        # Test with zeros
        zero_tensor = torch.zeros(2)
        
        # InfoNCE loss with zeros
        loss = concordance_loss_nce(zero_tensor, zero_tensor, tau=0.1)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        
        # Pearson loss with constant values (should handle division by zero in std dev)
        constant_tensor = torch.ones(3)
        loss = negative_pearson_loss(constant_tensor, constant_tensor)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        
        # Test with very large values
        large_tensor = torch.tensor([1e6, 1e7])
        loss = concordance_loss_nce(large_tensor, large_tensor + 1, tau=0.1)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)