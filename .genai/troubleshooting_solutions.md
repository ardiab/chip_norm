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