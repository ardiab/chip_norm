# Task Document: Create synthetic data generator

**Task ID:** create_synthetic_data_generator

**Description:**
Build utility functions to generate small synthetic datasets in the expected .npy file format for testing purposes. The generator should support both single and multi-replicate dataset formats, creating minimal but valid data that can be used to test the full training pipeline without requiring actual preprocessed data.

**Blocked By:**
- (None)

**Acceptance Criteria:**
- Generator creates valid .npy files matching expected data format
- Supports both single-replicate and multi-replicate modes
- Generated data has consistent dimensions across all files
- Includes cleanup utilities for temporary test data
- Data is minimal in size for fast test execution

**Test Scenarios (for the agent to implement):**
1. **Single replicate generation:** Test creation of valid single replicate dataset files (control_reads.npy, treatment_reads.npy, covariates.npy)
2. **Multi-replicate generation:** Test creation of valid multi-replicate dataset files (with _r1 and _r2 suffixes)
3. **Dimension consistency:** Test that generated arrays have consistent dimensions across control/treatment/covariates
4. **Small dataset size:** Test generation of minimal datasets (e.g., 10-100 samples) for fast test execution
5. **Temporary file cleanup:** Test that temporary data files are properly cleaned up after tests

**Implementation Todos:**
1. **Implement tests:**
   a. Create test file `tests/test_synthetic_data_generator.py`
   b. Write test for single replicate data generation
   c. Write test for multi-replicate data generation
   d. Write test for dimension consistency validation
   e. Write test for small dataset size verification
   f. Write test for cleanup functionality
2. **Ensure tests fail:** Run the tests implemented in step (1) and ensure that they fail (feature not yet implemented)
3. **Implement code:**
   a. Create generator module in `tests/utils/synthetic_data.py`
   b. Implement function to generate single replicate datasets
   c. Implement function to generate multi-replicate datasets
   d. Add dimension validation and consistency checks
   e. Create context manager for temporary data with automatic cleanup
   f. Add configurable parameters for dataset size and covariate dimensions
4. **Ensure all tests are passing:** Run all tests to ensure that code is functioning correctly