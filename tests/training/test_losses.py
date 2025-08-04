import torch

from chipvi.training.losses import compute_residual_mse


def test_residual_mse_loss_calculation():
    """Test that residual MSE loss calculation works correctly."""
    # Create simple torch.Tensor inputs
    y_r1 = torch.tensor([10.0, 20.0])
    mu_tech_r1 = torch.tensor([4.0, 12.0])
    y_r2 = torch.tensor([15.0, 25.0])
    mu_tech_r2 = torch.tensor([5.0, 15.0])
    sd_ratio = torch.tensor([1.0, 1.0])
    
    # Calculate expected residuals
    # r1_residual = y_r1 - mu_tech_r1 = [10.0 - 4.0, 20.0 - 12.0] = [6.0, 8.0]
    # r2_residual = y_r2 - mu_tech_r2 = [15.0 - 5.0, 25.0 - 15.0] = [10.0, 10.0]
    
    # Calculate expected MSE: mean(( [6, 8] - [10, 10] )^2) = mean( [-4, -2]^2 ) = mean([16, 4]) = 10.0
    expected_mse = 10.0
    
    # Call compute_residual_mse function
    actual_mse = compute_residual_mse(y_r1, mu_tech_r1, y_r2, mu_tech_r2, sd_ratio)
    
    # Assert that the function returns a value approximately equal to 10.0
    assert torch.allclose(actual_mse, torch.tensor(expected_mse), atol=1e-6), \
        f"Expected MSE = {expected_mse}, got {actual_mse.item()}"