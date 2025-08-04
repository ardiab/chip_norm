# Troubleshooting Solutions

## Task Group 1: Establish Configuration-Driven Path Management

### Issue: Hydra Resolver Not Available in Test Context
**Problem:** When testing the PathHelper class with Hydra's `compose` API, the `${hydra:runtime.cwd}` resolver failed with `InterpolationResolutionError: ValueError raised while resolving interpolation: HydraConfig was not set`.

**Root Cause:** Hydra resolvers like `${hydra:runtime.cwd}` are only available when running within a proper Hydra context (i.e., when using `@hydra.main` decorator or similar initialization).

**Solution:** For unit tests, create a simplified configuration using `OmegaConf.create()` with hardcoded test paths instead of relying on Hydra resolvers. This allows testing the PathHelper logic without needing full Hydra initialization.

**Implementation:**
```python
# Instead of using Hydra's compose API:
# with initialize_config_dir(config_dir=str(config_path), version_base=None):
#     cfg = compose(config_name="config")

# Use direct OmegaConf creation:
test_config = {
    "paths": {
        "project_root": "/test/project",
        "data_raw": "/test/project/data/raw",
        "entex_processed": "/test/project/data/entex_files/proc",
        # ... other paths
    }
}
cfg = OmegaConf.create(test_config)
```

**Confirmed Working:** This approach allows the PathHelper tests to pass while maintaining the flexibility of the configuration-driven design for actual runtime usage.

## Task Group 2: Build Foundational Test Suite for Critical Components

### Implementation Success
**Task Completion:** All tasks in Task Group 2 were implemented successfully without major issues.

**Key Achievements:**
- Created comprehensive test suite covering data preprocessing (`test_preprocessing.py`), dataset structure validation (`test_datasets.py`), and loss function calculations (`test_losses.py`)
- Successfully ported aggregation logic from training scripts to reusable `aggregate_bins` function in `chipvi/data/preprocessing.py`
- Implemented and validated `compute_residual_mse` function that correctly calculates scaled residual MSE between replicates
- All tests pass, confirming correct implementation of critical components

**Testing Results:** 
```bash
pytest tests/data/test_preprocessing.py tests/data/test_datasets.py tests/training/test_losses.py -v
# 3 tests passed successfully
```

**No Issues Encountered:** The implementation proceeded smoothly with proper module structure and import paths working correctly.

## Task Group 3: Refactor Dataset and Data Loading Logic

### Implementation Success
**Task Completion:** All tasks in Task Group 3 were implemented successfully without major issues.

**Key Achievements:**
- Successfully refactored `MultiReplicateDataset` to return structured dictionaries instead of flat tensors, improving code clarity and maintainability
- Updated `__getitem__` method to return nested dictionary structure: `{'r1': {'covariates': ..., 'reads': ...}, 'r2': {...}, 'metadata': {...}}`
- Refactored `SingleReplicateDataset` for consistency, returning `{'covariates': ..., 'reads': ...}` dictionary structure
- Renamed `get_dim_x` to `get_covariate_dim` in both dataset classes for better semantic clarity
- Removed obsolete test and added comprehensive test for new dictionary-based structure
- All existing tests continue to pass, confirming no regressions were introduced

**Testing Results:** 
```bash
pytest -v
# 4 tests passed successfully, including the new refactored dataset test
```

**Key Implementation Details:**
- `MultiReplicateDataset.__getitem__` now constructs covariates tensors on-demand rather than pre-computing and storing them
- The new structure eliminates the need for downstream code to perform index-based slicing to separate replicate data
- `get_covariate_dim()` returns 5 for both dataset classes, representing the number of covariates per replicate

**No Issues Encountered:** The refactoring proceeded smoothly with all tests passing on first attempt after implementation.

## Task Group 4: Implement the Unified Trainer

### Implementation Success
**Task Completion:** All tasks in Task Group 4 were implemented successfully with one minor issue that was quickly resolved.

**Key Achievements:**
- Created comprehensive test suite for the unified `Trainer` class in `tests/training/test_trainer.py`
- Successfully refactored `compute_residual_mse` function to `replicate_concordance_mse_loss` with new signature that accepts model outputs and batch dictionaries
- Implemented unified `Trainer` class in `chipvi/training/trainer.py` that encapsulates complete training and validation loops
- The trainer successfully orchestrates forward pass, loss calculation, and backpropagation using the new dictionary-based data format

**Issue Encountered and Resolved:**
**Problem:** Initial test failed with `ValueError: too many values to unpack (expected 2)` when trying to unpack model outputs.
**Root Cause:** The mock model in the test was a simple `nn.Linear` that returns only one output tensor, but the Trainer expects two outputs (mu and r) like the real `TechNB_mu_r` model.
**Solution:** Created a proper `MockModel` class that returns two outputs like the real model:
```python
class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 2)
    
    def forward(self, x):
        outputs = self.linear(x)
        mu = torch.exp(outputs[:, 0:1])  # Ensure positive mu
        r = torch.exp(outputs[:, 1:2])   # Ensure positive r
        return mu, r
```

**Testing Results:** 
```bash
pytest -v
# 5 tests passed successfully, including the new trainer test
```

**Key Implementation Details:**
- `Trainer.fit()` method runs training for specified number of epochs with logging
- `_train_one_epoch()` handles forward pass for both replicates, loss calculation, backpropagation, and parameter updates
- `_validate_one_epoch()` performs validation with gradients disabled
- `_move_batch_to_device()` utility method recursively moves all tensors in batch dictionaries to specified device
- The trainer successfully works with the dictionary-based batch structure from the refactored datasets

**No Major Issues Encountered:** After resolving the mock model issue, all functionality worked as expected with all tests passing.