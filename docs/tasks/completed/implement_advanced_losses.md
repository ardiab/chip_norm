**Reviewed & approved on 2025-08-06**

# Task Document: Implement Advanced Loss Functions

**Task ID:** implement_advanced_losses

**Description:**
Add three advanced loss functions and composite loss system to enable multi-objective training with biological replicate concordance constraints. These losses will allow the model to learn from the concordance between biological replicates, improving the separation of technical and biological signal.

**Blocked By:**
- (None)

**Acceptance Criteria:**
- InfoNCE, Pearson, and quantile losses compute correctly and return scalar tensors
- Composite loss combines multiple terms with configurable weights
- All losses compatible with MultiReplicateDataset format (dict with 'r1', 'r2' keys)
- Gradients flow properly through composite losses for optimization

**Test Scenarios (for the agent to implement):**
1. **InfoNCE Concordance Loss Correctness:** Verify the loss returns lower values for matched replicate pairs (diagonal elements) than mismatched pairs in the similarity matrix.
2. **Negative Pearson Loss Correlation:** Verify the loss returns -1 for perfectly correlated inputs and approaches 0 for uncorrelated data.
3. **Quantile Absolute Loss Calibration:** Verify the loss correctly measures differences in Probability Integral Transform values between replicates.
4. **Composite Loss Weighting:** Verify the composite loss properly combines multiple loss terms with specified weights and that total loss equals weighted sum.
5. **Numerical Stability:** Test all losses handle edge cases including zero values, very large values, and identical inputs without NaN or inf.
6. **Batch Compatibility:** Ensure all losses work with the MultiReplicateDataset tensor dictionary format containing batched data.

**Implementation Todos:**
1. **Implement tests for new loss functions:**
   a. Create `tests/test_advanced_losses.py` with pytest test cases
   b. Test InfoNCE loss: verify it returns lower values for matched pairs (diagonal elements) than mismatched pairs
   c. Test Pearson loss: verify correlation coefficient calculation against scipy.stats.pearsonr
   d. Test quantile loss: verify PIT difference calculation using mock distributions
   e. Test composite loss: verify weighted combination and gradient backpropagation

2. **Ensure tests fail:** Run pytest to confirm tests fail before implementation

3. **Implement InfoNCE concordance loss in `chipvi/training/losses.py`:**
   a. Create function `concordance_loss_nce(res1, res2, tau=0.1)` that takes two residual tensors
   b. Compute pairwise similarity matrix using exp(-(res1 - res2)^2 / tau)
   c. Extract diagonal as positive pairs
   d. Compute NCE loss as -log(positive / sum_over_row)
   e. Add epsilon for numerical stability in log operation

4. **Implement negative Pearson correlation loss:**
   a. Create function `negative_pearson_loss(preds, targets, eps=1e-8)`
   b. Center both tensors by subtracting their means
   c. Compute correlation as sum(centered_preds * centered_targets) / sqrt(sum(centered_preds^2) * sum(centered_targets^2))
   d. Return negative correlation to minimize for optimization

5. **Implement quantile absolute difference loss:**
   a. Create function `quantile_absolute_loss(model_r1, model_r2, y_r1, y_r2)`
   b. Use existing `compute_numeric_cdf` from `chipvi.utils.distributions`
   c. Compute quantiles (PIT) for both replicates
   d. Return mean absolute difference between quantiles

6. **Create CompositeLoss class:**
   a. Initialize with list of loss functions and corresponding weights
   b. In forward method, compute each loss component
   c. Return weighted sum and optionally individual components for logging
   d. Ensure class is compatible with existing loss function interface

7. **Update loss registry in `scripts/run.py`:**
   a. Add new loss functions to LOSS_REGISTRY dictionary
   b. Add composite loss factory that can parse config with multiple losses
   c. Ensure backward compatibility with single loss specifications

8. **Ensure all tests pass:** Run full test suite and verify implementation

9. **(REVIEW) Create wrapper functions for loss registry compatibility:** In `chipvi/training/losses.py`, create wrapper functions that match the expected `loss_fn(model_outputs: dict, batch: dict)` interface for each individual loss function. These wrappers should extract the required tensors from the `model_outputs` and `batch` dictionaries and call the core algorithmic functions.

10. **(REVIEW) Update loss registry with wrapper functions:** In `scripts/run.py`, update the `LOSS_REGISTRY` to use the wrapper functions instead of the core algorithmic functions, ensuring compatibility with the training pipeline interface.

11. **(REVIEW) Create integration tests:** Add tests in `tests/test_advanced_losses.py` that verify the wrapper functions work correctly with the actual `model_outputs` and `batch` dictionary format used by MultiReplicateDataset.

12. **(REVIEW) Verify full integration:** Run a test training loop to ensure the new loss functions can be used in the actual training pipeline without errors.

## Completion Note

Successfully implemented all advanced loss functions with comprehensive test coverage. The implementation includes:

- **InfoNCE concordance loss**: Computes contrastive loss between replicate residuals with proper multi-dimensional tensor support
- **Negative Pearson correlation loss**: Measures correlation between predictions and targets for optimization  
- **Quantile absolute difference loss**: Computes PIT value differences using negative binomial distributions
- **CompositeLoss class**: Enables weighted combination of multiple loss functions with gradient flow support

All 23 tests pass, including 17 new comprehensive tests covering correctness, numerical stability, batch compatibility, and edge cases. The loss registry in `scripts/run.py` has been updated with backward compatibility for single loss configurations.