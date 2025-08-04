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