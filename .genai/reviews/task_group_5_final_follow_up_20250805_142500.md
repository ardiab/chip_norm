# Code Review: Task Group 5 (Final Follow-up)

**Decision:** Request Changes

**Summary:**
The implementation is now very close to completion. The main training script (`scripts/run.py`) and the experiment configuration files have been refactored correctly to use a clean, flattened structure. The `build_datasets` function has been restored with the correct BED-processing logic. However, one critical piece is still missing: the file paths for the input data (`bed_fpath_groups`) are hardcoded within the `build_datasets` function instead of being read from the configuration object. This is the last remaining issue preventing the pipeline from being fully configuration-driven.

---

**Detailed Feedback:**
*   **Finding:** Input File Paths are Hardcoded
    *   **File:** `chipvi/data/datasets.py`
    *   **Issue:** The `bed_fpath_groups` variable, which defines the input experiment and control files, is hardcoded within an `if/else` block inside the `build_datasets` function. The code contains a `TODO` comment acknowledging that this should come from the configuration.
    *   **Suggestion:** Remove the hardcoded `bed_fpath_groups` list. Instead, read this structure from the `cfg` object. The data configuration file (e.g., `configs/data/h3k27me3_200bp.yaml`) should be updated to contain the list of replicate groups and their corresponding experiment and control file names. The `build_datasets` function will then read `cfg.replicate_groups`, iterate through them, and use the `PathHelper` to construct the full, absolute paths to the BED files.
    *   **Reasoning:** Making the input file paths configurable is the entire point of this refactoring effort. This final change will fully decouple the data loading logic from the specific file locations, making the pipeline flexible and portable as intended.
