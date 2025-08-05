# Code Review: Task Group 5: Create a Config-Driven Experiment Pipeline

**Decision:** Request Changes

**Summary:**
The implementation for Task Group 5 is incomplete and contains significant logical errors. While the new configuration files have been created and old scripts have been deleted, the core logic for connecting the configuration to the data loading pipeline is broken. The main training script (`scripts/run.py`) relies on a configuration structure that is not clearly defined in the base configuration, and the critical `build_datasets` function has been stubbed out with dummy data instead of being refactored as required by the plan. This prevents any real data from being loaded and used for training.

---

**Detailed Feedback:**
*   **Finding:** Inconsistent and Brittle Configuration Access
    *   **File:** `scripts/run.py`
    *   **Issue:** The script accesses configuration values using a nested structure (e.g., `cfg.experiment.training.device`, `cfg.experiment.data.batch_size`). This `experiment` node is not present in the base `configs/config.yaml` but is expected to be composed by Hydra from a separate experiment file. This makes the configuration difficult to trace and adds a layer of indirection that can easily break if the experiment file structure changes.
    *   **Suggestion:** Modify `scripts/run.py` to access the composed configuration values directly from the `cfg` object. The experiment file should merge its `data` and `training` sections into the top-level `cfg` object. For example, instead of `cfg.experiment.training.learning_rate`, the access should be `cfg.training.learning_rate`. This requires adjusting the experiment configuration files to not nest everything under an `experiment` key.
    *   **Reasoning:** This change simplifies the configuration structure, making it flatter and more intuitive. It removes the unnecessary `experiment` level of nesting, making the `run.py` script more robust and easier to read, as the configuration access path directly matches the keys in the YAML files.

*   **Finding:** Data Loading is Non-Functional due to Incomplete Refactoring
    *   **File:** `chipvi/data/datasets.py`
    *   **Issue:** The `build_datasets` function was not refactored to be configuration-driven as specified in the task. Instead, its entire logic was commented out and replaced with a stub that returns dummy tensors. This is a critical failure, as it prevents the training pipeline from loading and processing any actual data.
    *   **Suggestion:** Restore the original logic of the `build_datasets` function. Modify it to extract all necessary parameters (e.g., file paths, chromosome lists, aggregation factors) directly from the `cfg: DictConfig` object that is passed to it. The function must read the paths and parameters from `cfg` and perform the data loading and processing as it did before.
    *   **Reasoning:** The goal of the task was to make the existing data loading logic configurable, not to disable it. Restoring and refactoring the function is essential for the entire training pipeline to be functional.
