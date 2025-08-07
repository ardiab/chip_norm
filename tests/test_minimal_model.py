"""Tests for minimal test model."""

import pytest
import torch
import torch.nn as nn
from tests.models.minimal_model import MinimalTestModel


class TestMinimalTestModel:
    """Test cases for MinimalTestModel."""
    
    def test_output_format_tuple(self):
        """Test that the model outputs the correct tuple structure with mu and r values."""
        model = MinimalTestModel(covariate_dim=5)
        x = torch.randn(4, 5)
        
        output = model(x)
        
        # Check output is a tuple with correct structure
        assert isinstance(output, tuple), "Output should be a tuple"
        assert len(output) == 2, "Output should contain exactly 2 elements"
        
        # Check output shapes
        mu, r = output
        assert mu.shape == (4, 1), "mu should have shape (batch_size, 1)"
        assert r.shape == (4, 1), "r should have shape (batch_size, 1)"
        
        # Check output values are positive
        assert torch.all(mu > 0), "mu values should be positive"
        assert torch.all(r > 0), "r values should be positive"
        
    def test_input_dimension_handling(self):
        """Test that the model accepts covariates of the configured dimension."""
        covariate_dim = 10
        model = MinimalTestModel(covariate_dim=covariate_dim)
        
        # Test correct input dimension
        x_correct = torch.randn(3, covariate_dim)
        output = model(x_correct)
        assert isinstance(output, tuple), "Model should handle correct input dimensions"
        
        # Test incorrect input dimension should raise error
        x_wrong = torch.randn(3, covariate_dim + 1)
        with pytest.raises(RuntimeError):
            model(x_wrong)
            
    def test_batch_processing(self):
        """Test that the model can handle batched inputs."""
        model = MinimalTestModel(covariate_dim=3)
        
        # Test different batch sizes
        batch_sizes = [1, 4, 16, 32]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3)
            output = model(x)
            
            assert isinstance(output, tuple), f"Should handle batch size {batch_size}"
            mu, r = output
            assert mu.shape == (batch_size, 1), f"mu shape incorrect for batch size {batch_size}"
            assert r.shape == (batch_size, 1), f"r shape incorrect for batch size {batch_size}"
            
    def test_cpu_execution(self):
        """Test that the model runs efficiently on CPU without GPU requirements."""
        model = MinimalTestModel(covariate_dim=8)
        
        # Explicitly set model to CPU
        model = model.cpu()
        
        # Create CPU tensor
        x = torch.randn(5, 8, device='cpu')
        
        # Model should run on CPU without issues
        output = model(x)
        
        # Check outputs are on CPU
        mu, r = output
        assert mu.device.type == 'cpu', "mu should be on CPU"
        assert r.device.type == 'cpu', "r should be on CPU"
        
        # Check values are reasonable
        assert torch.all(torch.isfinite(mu)), "mu values should be finite"
        assert torch.all(torch.isfinite(r)), "r values should be finite"
        
    def test_parameter_initialization_and_trainability(self):
        """Test that model parameters are initialized and trainable."""
        model = MinimalTestModel(covariate_dim=6)
        
        # Check parameters exist and are trainable
        params = list(model.parameters())
        assert len(params) > 0, "Model should have parameters"
        
        for param in params:
            assert param.requires_grad, "All parameters should require gradients"
            assert torch.all(torch.isfinite(param)), "Parameters should be initialized to finite values"
            
        # Test gradient flow
        x = torch.randn(2, 6, requires_grad=True)
        output = model(x)
        
        # Compute a simple loss to test gradient flow
        mu, r = output
        loss = mu.sum() + r.sum()
        loss.backward()
        
        # Check gradients exist
        for param in params:
            assert param.grad is not None, "Parameters should have gradients after backward pass"
            assert torch.all(torch.isfinite(param.grad)), "Gradients should be finite"
            
    def test_same_interface_as_technb_mu_r(self):
        """Test that minimal model has same interface as TechNB_mu_r."""
        from chipvi.models.technical_model import TechNB_mu_r
        
        covariate_dim = 5
        
        # Create both models
        original_model = TechNB_mu_r(
            covariate_dim=covariate_dim,
            hidden_dims_mu=(32,), 
            hidden_dims_r=(32,)
        )
        minimal_model = MinimalTestModel(covariate_dim=covariate_dim)
        
        x = torch.randn(3, covariate_dim)
        
        # Both should accept same input
        original_output = original_model(x)
        minimal_output = minimal_model(x)
        
        # Both models should return tuples with same structure
        assert isinstance(original_output, tuple), "Original model returns tuple"
        assert len(original_output) == 2, "Original model returns 2 values"
        
        assert isinstance(minimal_output, tuple), "Minimal model returns tuple"
        assert len(minimal_output) == 2, "Minimal model returns 2 values"
        
        # Both outputs should have same shape
        original_mu, original_r = original_output
        minimal_mu, minimal_r = minimal_output
        
        assert original_mu.shape == minimal_mu.shape, "mu shapes should match"
        assert original_r.shape == minimal_r.shape, "r shapes should match"
        
    def test_forward_pass_consistency(self):
        """Test that forward passes are consistent for same input."""
        model = MinimalTestModel(covariate_dim=4)
        x = torch.randn(2, 4)
        
        # Multiple forward passes should give same result
        output1 = model(x)
        output2 = model(x)
        
        mu1, r1 = output1
        mu2, r2 = output2
        
        assert torch.allclose(mu1, mu2), "Forward passes should be consistent for mu"
        assert torch.allclose(r1, r2), "Forward passes should be consistent for r"