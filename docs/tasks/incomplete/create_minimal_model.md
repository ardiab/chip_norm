# Task Document: Create minimal test model

**Task ID:** create_minimal_model

**Description:**
Implement a minimal, lightweight model class that implements the same interface as TechNB_mu_r but uses simple linear layers instead of complex MLPs. This model will be used for CPU-based testing to ensure fast test execution while still exercising the full training pipeline.

**Blocked By:**
- (None)

**Acceptance Criteria:**
- Model implements the same interface as TechNB_mu_r
- Uses single linear layer for simplicity and speed
- Outputs correct dictionary format with 'mu' and 'r' parameters
- Runs efficiently on CPU without GPU requirements
- Parameters are trainable and gradients flow correctly

**Test Scenarios (for the agent to implement):**
1. **Output format:** Test that the model outputs the correct dictionary structure with 'mu' and 'r' keys
2. **Input dimensions:** Test that the model accepts covariates of the configured dimension
3. **Batch processing:** Test that the model can handle batched inputs
4. **CPU compatibility:** Test that the model runs efficiently on CPU without GPU requirements
5. **Parameter initialization:** Test that model parameters are initialized and trainable

**Implementation Todos:**
1. **Implement tests:**
   a. Create test file `tests/test_minimal_model.py`
   b. Write test for output format verification
   c. Write test for input dimension handling
   d. Write test for batch processing capability
   e. Write test for CPU execution
   f. Write test for parameter initialization and trainability
2. **Ensure tests fail:** Run the tests implemented in step (1) and ensure that they fail (feature not yet implemented)
3. **Implement code:**
   a. Create model module in `tests/models/minimal_model.py`
   b. Implement MinimalTestModel class inheriting from nn.Module
   c. Add single linear layer for mu prediction
   d. Add single linear layer for r prediction
   e. Implement forward method returning dict with 'mu' and 'r'
   f. Ensure all operations are CPU-friendly
4. **Ensure all tests are passing:** Run all tests to ensure that code is functioning correctly