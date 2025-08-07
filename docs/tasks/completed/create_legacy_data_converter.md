**Reviewed & approved on 2025-08-07**

# Task Document: Create Legacy Data Converter

**Task ID:** create_legacy_data_converter

**Description:**
Build a standalone script that converts legacy pickle data from `{target}_data_v2` directories to the new `.npy` file format expected by `MultiReplicateDataset`. The script should preserve data integrity, apply optional log transforms to specific covariate columns, and maintain the same train/val splits as the original implementation.

**Blocked By:**
- (None)

**Acceptance Criteria:**
- Script successfully loads pickle files from `{target}_data_v2` directories
- Applies the same log-transform logic to covariate columns 0 and 5 if specified
- Converts legacy data structure to the new `.npy` file format with proper naming conventions
- Preserves train/val splits and data integrity from the original pickle files
- Output files are compatible with the current `MultiReplicateDataset` class

**Test Scenarios (for the agent to implement):**
1. **Data Conversion Accuracy**: Load legacy pickle data, convert to `.npy` format, and verify that data values, shapes, and train/val splits are preserved correctly.
2. **Log Transform Application**: Test that optional log transforms are applied correctly to covariate columns 0 and 5 when specified.

**Implementation Todos:**
1. **Implement tests:**
   a. Create test that loads sample legacy pickle data and verifies conversion accuracy
   b. Create test that verifies log transform application on specific covariate columns
2. **Ensure tests fail:** Run the tests implemented in step (1) and ensure that they fail (script not yet implemented).
3. **Implement code:**
   a. Create `scripts/convert_legacy_data.py` with command-line interface accepting `target`, `log_transform_inputs`, and `output_directory`
   b. Implement legacy pickle loading logic that skips `sd_map` files
   c. Implement data structure conversion to match `MultiReplicateDataset` expectations
   d. Add log transform logic for covariate columns 0 and 5 when specified
   e. Save converted data with proper `.npy` naming conventions
4. **Ensure all tests are passing:** Run all tests to ensure that the conversion script is functioning correctly.

## Completion Note

Successfully implemented the legacy data converter with comprehensive test coverage. The script (`scripts/convert_legacy_data.py`) converts legacy pickle data from `{target}_data_v2` directories to `.npy` format compatible with `MultiReplicateDataset`. Key features include proper train/val data consolidation, optional log transforms for covariate columns 0 and 5 (control reads), and robust error handling. All tests pass, confirming data integrity preservation and correct log transform application.