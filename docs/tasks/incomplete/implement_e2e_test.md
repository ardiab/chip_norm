# Task Document: Implement end-to-end system test

**Task ID:** implement_e2e_test

**Description:**
Write the main end-to-end test that orchestrates the entire training pipeline by calling scripts/run.py with test configuration. This test will use all previously created components (wandb mock, synthetic data, test configs, minimal model) to verify that the complete system works correctly from configuration parsing through training completion.

**Blocked By:**
- `create_wandb_mock`
- `create_synthetic_data_generator`
- `create_test_configs`
- `create_minimal_model`

**Acceptance Criteria:**
- Test successfully calls scripts/run.py with test configuration
- WandB logging is mocked and no actual network calls are made
- Training completes for configured number of epochs
- All sub-components are initialized and called correctly
- Multi-component loss is computed and tracked properly
- Test runs quickly (under 30 seconds) on CPU

**Test Scenarios (for the agent to implement):**
1. **Configuration parsing:** Test that run.py correctly parses all configuration parameters including multi-component loss
2. **Component initialization:** Test that trainer, model, loss functions, and data loaders are all initialized correctly
3. **Training execution:** Test that the training loop runs for the configured number of epochs without errors
4. **Mock wandb verification:** Test that wandb logging calls are intercepted and metrics are captured (not sent to servers)
5. **Loss computation:** Test that the multi-component loss is computed and individual components are tracked
6. **Checkpoint creation:** Test that checkpoints are created if configured
7. **Validation loop:** Test that validation is performed and validation metrics are computed
8. **Early stopping:** Test that early stopping works if configured
9. **Scheduler operation:** Test that learning rate scheduling works if configured
10. **Completion verification:** Test that the script completes successfully and returns expected status

**Implementation Todos:**
1. **Implement tests:**
   a. Create test file `tests/test_e2e_system.py`
   b. Write test for configuration parsing verification
   c. Write test for component initialization checking
   d. Write test for training execution monitoring
   e. Write test for wandb mock verification
   f. Write test for loss computation tracking
   g. Write test for checkpoint creation
   h. Write test for validation metrics
   i. Write test for early stopping behavior
   j. Write test for scheduler operation
   k. Write test for successful completion
2. **Ensure tests fail:** Run the tests implemented in step (1) and ensure that they fail (feature not yet implemented)
3. **Implement code:**
   a. Create main test function that sets up test environment
   b. Generate synthetic data in temporary directory
   c. Configure Hydra overrides for test paths and parameters
   d. Apply wandb mock before running script
   e. Execute run.py using subprocess or direct import
   f. Capture and verify outputs and logged metrics
   g. Assert on expected behaviors (loss decrease, metrics logged)
   h. Clean up temporary files after test
4. **Ensure all tests are passing:** Run all tests to ensure that code is functioning correctly