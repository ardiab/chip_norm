# Code Review: Task Group 5 (Follow-up)

**Decision:** Request Changes

**Summary:**
The updated implementation fails to address the critical issues raised in the previous review. The `build_datasets` function in `chipvi/data/datasets.py` has been rewritten with entirely new logic that deviates from the plan, rather than being refactored as requested. The new implementation now loads data from hardcoded, non-portable pickle file paths and completely omits the original, required logic for processing BED files. This is a more severe deviation from the plan than the previous implementation.

---

**Detailed Feedback:**
*   **Finding:** Data Loading Logic is Incorrect and Breaks the Config-Driven Workflow
    *   **File:** `chipvi/data/datasets.py`
    *   **Issue:** The `build_datasets` function was not refactored as per the corrective task. The task required restoring the original BED-processing logic and adapting it to read parameters from the `cfg` object. Instead, the function was replaced with new logic that attempts to load pre-processed `.pkl` files from a hardcoded absolute path (`/Volumes/abdul/repos/chipvi/h3k27me3_data`). This approach is not portable, ignores the configuration system we are trying to build, and fails to perform the necessary data preparation from the source BED files.
    *   **Suggestion:** **Delete the current implementation of `build_datasets`**. Restore the original logic that was previously commented out (the version that uses `load_and_process_bed`). Modify this restored function to derive its parameters (like `bed_fpath_groups`, `train_chroms`, and `val_chroms`) from the `cfg: DictConfig` object. The file paths for the BED files should be resolved using the `PathHelper` utility, which reads from the `cfg.paths` section.
    *   **Reasoning:** The primary goal of this task group is to create a *configuration-driven* pipeline that processes data from a raw format. The current implementation circumvents this entirely. Restoring and refactoring the original logic is the only way to meet the architectural goals.

*   **Finding:** Inconsistent Configuration Access (Not Yet Addressed)
    *   **File:** `scripts/run.py`
    *   **Issue:** The issue from the previous review regarding inconsistent configuration access (e.g., using `cfg.experiment.training.device`) has not been addressed.
    *   **Suggestion:** First, fix the critical `build_datasets` issue. Once the data loading is functional, this configuration access pattern must be simplified as described in the previous review. The experiment configuration files should be modified to merge their keys into the top-level `cfg` object, and `run.py` should be updated to access them directly (e.g., `cfg.device`, `cfg.batch_size`).
    *   **Reasoning:** A flat, predictable configuration structure is more robust and maintainable. This change is secondary to the data loading issue but is still required for final approval.
