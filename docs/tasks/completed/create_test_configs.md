**Reviewed & approved on 2025-01-08**

# Task Document: Create test configuration files

**Task ID:** create_test_configs

**Description:**
Design and create dedicated Hydra configuration files for testing that include multi-component loss setup, minimal training parameters, disabled wandb, and paths configured for synthetic test data. These configurations will be used by the end-to-end system test to run complete training cycles with minimal resource usage.

**Blocked By:**
- (None)

**Acceptance Criteria:**
- Test configuration includes multi-component loss (at least 2 components) ✅
- Training parameters are set to minimal values (few epochs, small batch size) ✅
- WandB is explicitly disabled in configuration ✅
- Paths can be overridden to point to temporary test data ✅
- Configuration is valid and parseable by Hydra ✅

**Test Scenarios (for the agent to implement):**
1. **Multi-component loss config:** Test that the configuration properly defines a composite loss with multiple components ✅
2. **Training parameters:** Test that training parameters (epochs, batch_size, learning_rate) are set to minimal values for fast testing ✅
3. **WandB disabled:** Test that wandb is explicitly disabled in the test configuration ✅
4. **Model configuration:** Test that the model config points to a minimal/test model ✅
5. **Data paths:** Test that data paths can be overridden to point to temporary test data ✅

**Implementation Todos:**
1. **Implement tests:** ✅
   a. Create test file `tests/test_config_files.py` ✅
   b. Write test to verify multi-component loss configuration structure ✅
   c. Write test to verify minimal training parameters ✅
   d. Write test to verify wandb is disabled ✅
   e. Write test to verify model configuration ✅
   f. Write test to verify data path configurability ✅
2. **Ensure tests fail:** Run the tests implemented in step (1) and ensure that they fail (feature not yet implemented) ✅
3. **Implement code:** ✅
   a. Create main test config at `configs/test/test_e2e.yaml` ✅
   b. Configure multi-component loss with nll and concordance components ✅
   c. Set minimal training parameters (2-3 epochs, batch_size=32) ✅
   d. Create test wandb config at `configs/test/wandb_test.yaml` with enabled=false ✅
   e. Create test model config at `configs/test/model_test.yaml` for minimal model ✅
   f. Configure data paths with variables that can be overridden ✅
4. **Ensure all tests are passing:** Run all tests to ensure that code is functioning correctly ✅
5. **(REVIEW) Fix test configuration structure:** Move `configs/test/test_e2e.yaml` to `configs/experiment/test_e2e.yaml` and add a proper defaults section that composes from other configs (model/test/model_test, wandb/test/wandb_test) following the pattern of existing experiment configs. This ensures the config loads at the root level, not nested under 'test'.
6. **(REVIEW) Remove duplicate model configuration:** Delete the duplicate file at `configs/test/model_test.yaml` and keep only the one at `configs/model/test/model_test.yaml`.
7. **(REVIEW) Fix wandb config location:** Move `configs/test/wandb_test.yaml` to `configs/wandb/test/wandb_test.yaml` to match its comment and follow Hydra conventions.
8. **(REVIEW) Update test file to use new config location:** In `tests/test_config_files.py`, update the test_config fixture to load the config from its new location at `experiment/test_e2e` and adjust the test to work with the non-nested structure.
9. **(REVIEW) Run tests:** Run `pytest tests/test_config_files.py -v` to ensure all tests still pass after the restructuring.