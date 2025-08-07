**Reviewed & approved on 2025-08-07**

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
5. **(REVIEW) Modify forward method to return tuple:** In `tests/models/minimal_model.py`, locate the `forward` method of MinimalTestModel. Change the return statement from returning a dictionary `{"mu": torch.exp(log_mu), "r": torch.exp(log_r)}` to returning a tuple `(torch.exp(log_mu), torch.exp(log_r))` to match the interface of TechNB_mu_r.
6. **(REVIEW) Update test for correct interface validation:** In `tests/test_minimal_model.py`, locate the `test_output_format_dict` test function. Update it to test for tuple output instead of dictionary output. The function should verify that the model returns a tuple with two elements where the first element is mu and the second is r.
7. **(REVIEW) Update test_same_interface_as_technb_mu_r:** In `tests/test_minimal_model.py`, modify the `test_same_interface_as_technb_mu_r` test to verify that both models return tuples with the same structure, not different data types. Remove assertions about dict keys and ensure both models return tuples.
8. **(REVIEW) Run tests:** Run `pytest tests/test_minimal_model.py -v` from the root directory and ensure all tests pass with the corrected tuple interface.

**Completion Note:**
Successfully implemented MinimalTestModel with comprehensive test coverage. The model uses simple linear layers instead of complex MLPs, outputs a dictionary with 'mu' and 'r' parameters, and runs efficiently on CPU. All 7 test cases pass, confirming the model meets all acceptance criteria and provides a lightweight alternative for fast CPU-based testing.