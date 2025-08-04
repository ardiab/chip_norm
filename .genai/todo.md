### Task Group 1: Establish Configuration-Driven Path Management ✅ COMPLETED

This task group addresses the "Hardcoded, Non-Portable File Paths" finding by introducing a configuration system that will manage all file paths, making the project portable and easier to manage.

1.  ✅ **Create new directories:** In the project's root directory, create two new top-level directories:
    *   `configs/`
    *   `outputs/`

2.  ✅ **Add project dependencies:** Add `hydra-core` and `pytest` to your project's dependency management file (e.g., `requirements.txt` or `pyproject.toml`).

3.  ✅ **Create the main configuration file:** In the new `configs/` directory, create a file named `config.yaml` with the following content. This file will define the base structure for all paths.

    ```yaml
    # configs/config.yaml
    
    # Defaults for Hydra
    defaults:
      - _self_
    
    # Define the core directory structure.
    # Paths can be defined relative to the project root or as absolute paths.
    # Using environment variables (e.g., ${oc.env:MY_PROJECT_ROOT}) is encouraged for portability.
    paths:
      # Using Hydra's built-in original working directory resolver
      project_root: ${hydra:runtime.cwd}
      
      # Data directories
      data_base: ${paths.project_root}/data
      data_raw: ${paths.data_base}/raw
      data_processed: ${paths.data_base}/processed
    
      # Large file storage (e.g., ENTEx)
      entex_base: ${paths.data_base}/entex_files
      entex_raw: ${paths.entex_base}/raw
      entex_processed: ${paths.entex_base}/proc
    
      # Hydra will automatically manage the output directory
      outputs: ${hydra:run.dir}
    
    # Placeholder for experiment-specific configs
    experiment:
      name: "default_experiment"
    
    ```

4.  ✅ **Create a test for the new `PathHelper`:** Create a new test file at `tests/utils/test_path_helper.py`.
    *   **Test Case Description:** Write a test named `test_path_helper_resolves_paths_from_config`. This test should:
        *   Use Hydra's `compose` API to load the `configs/config.yaml` into a configuration object (`DictConfig`).
        *   Instantiate the (not-yet-refactored) `PathHelper` class with this configuration object.
        *   Assert that the `raw_data_dir` attribute of the `PathHelper` instance correctly resolves to the absolute path of the `data/raw` directory.
        *   Assert that the `entex_proc_file_dir` attribute resolves to the absolute path of the `data/entex_files/proc` directory.

5.  ✅ **Refactor `PathHelper` to be configuration-driven:** In `chipvi/utils/path_helper.py`, modify the `PathHelper` class.
    *   **Remove all hardcoded `Path` objects** currently defined as class variables.
    *   **Modify the class signature** to accept a configuration object in its constructor. The new signature should be:
        ```python
        from pathlib import Path
        from omegaconf import DictConfig

        class PathHelper:
            def __init__(self, cfg: DictConfig):
        ```
    *   **Implementation Logic:** Inside the `__init__` method, read the path values from the `cfg.paths` object and assign them as instance attributes (e.g., `self.raw_data_dir = Path(cfg.paths.data_raw)`). Create attributes for all the critical paths the application will need.

6.  ✅ **Create a new primary script entry point:** Create a new file at `scripts/run.py`. This script will serve as the new entry point for running experiments and will demonstrate the use of the new configuration system.
    *   **Provide the following function signature and implementation logic:**
        ```python
        import hydra
        from omegaconf import DictConfig, OmegaConf
        
        from chipvi.utils.path_helper import PathHelper
        
        @hydra.main(config_path="../configs", config_name="config", version_base=None)
        def main(cfg: DictConfig) -> None:
            """
            Main entry point for running ChipVI experiments.
            """
            print("Configuration loaded successfully:")
            print(OmegaConf.to_yaml(cfg))
        
            # Instantiate the path helper with the loaded config
            paths = PathHelper(cfg)
        
            # Example of using the new PathHelper
            print("\nResolved raw data path:")
            print(paths.raw_data_dir)
            
            # TODO: The rest of the training logic will be added here in later tasks.
        
        if __name__ == "__main__":
            main()

        ```

7.  ✅ **Run tests:** Run `pytest` from the root directory and ensure the new `test_path_helper_resolves_paths_from_config` test passes. You can also run `python scripts/run.py` to see the configuration-driven path resolution in action.

---

### Task Group 2: Build Foundational Test Suite for Critical Components ✅ COMPLETED

This task group addresses the "Insufficient Test Coverage" finding by creating tests for data aggregation, dataset structure, and custom loss calculations.

1.  ✅ **Create new test file for data processing:** Create a new file at `tests/data/test_preprocessing.py`.

2.  ✅ **Define test for bin aggregation:** In `tests/data/test_preprocessing.py`, write a test case to validate the bin aggregation logic.
    *   **Test Case Description:** Write a test named `test_aggregate_bins_correctly_sums_and_averages`. This test must:
        *   Create a simple `numpy` array representing 16 bins of high-resolution "reads" data (e.g., `np.arange(16)`).
        *   Create a corresponding `numpy` array for "mapq" data.
        *   Call a new (not-yet-created) `aggregate_bins` function with these arrays and an aggregation factor of 8.
        *   Assert that the output "reads" array has a length of 2 and that its values are the correct sums of the original 8-bin windows (i.e., `[56, 72]`).
        *   Assert that the output "mapq" array has a length of 2 and that its values are the correct means of the original 8-bin windows.

3.  ✅ **Create new preprocessing module and function:** Create a new file at `chipvi/data/preprocessing.py`.
    *   **Provide the following function signature:**
        ```python
        import numpy as np
        from typing import Literal

        def aggregate_bins(
            data_array: np.ndarray,
            agg_factor: int,
            agg_method: Literal["sum", "mean"],
        ) -> np.ndarray:
        ```
    *   **Implementation Logic:** Port the aggregation logic currently found inside the `load_data` function in `runs/06_27_bigger_200bp/train.py` into this function. The logic should first trim the array to be divisible by `agg_factor`, then reshape it, and finally perform either a sum or a mean along the appropriate axis.

4.  ✅ **Create new test file for datasets:** Create a new file at `tests/data/test_datasets.py`.

5.  ✅ **Define test for `MultiReplicateDataset` structure:** In `tests/data/test_datasets.py`, write a test case to verify the output of the *existing* `MultiReplicateDataset`.
    *   **Test Case Description:** Write a test named `test_multireplicatedataset_getitem_structure`. This test must:
        *   Create small, distinct `torch.Tensor` objects for each of the 13 required inputs to the `MultiReplicateDataset` constructor (e.g., `control_reads_r1`, `experiment_reads_r1`, etc.). Use, for example, `torch.ones(10)`, `torch.zeros(10)`, `torch.full((10,), 2.0)`, etc., to make each input identifiable.
        *   Instantiate `MultiReplicateDataset` with these tensors.
        *   Get the first item using `dataset[0]`.
        *   Assert that the covariates tensor returned has the expected shape (`11,`).
        *   Assert that the first five values of the covariate tensor correctly correspond to the five `replicate_1` input tensors. For example, assert that `covariates[0]` is `1.0` (from `control_reads_r1`), `covariates[1]` is `...` and so on.
        *   Assert that the experiment reads tensor returned has the expected shape (`3,`) and that its values correspond to the correct inputs (`experiment_reads_r1`, `experiment_reads_r2`, and the seq depth ratio).

6.  ✅ **Create new test file for losses:** Create a new file at `tests/training/test_losses.py`.

7.  ✅ **Define test for biological concordance loss:** In `tests/training/test_losses.py`, write a test case to validate the MSE-based loss term.
    *   **Test Case Description:** Write a test named `test_residual_mse_loss_calculation`. This test must:
        *   Create simple `torch.Tensor` inputs: `y_r1 = torch.tensor([10.0, 20.0])`, `mu_tech_r1 = torch.tensor([4.0, 12.0])`, `y_r2 = torch.tensor([15.0, 25.0])`, `mu_tech_r2 = torch.tensor([5.0, 15.0])`, and `sd_ratio = torch.tensor([1.0, 1.0])`.
        *   Calculate the residuals: `r1_residual` should be `[6.0, 8.0]`. `r2_residual` should be `[10.0, 10.0]`.
        *   Calculate the expected MSE: `mean(( [6, 8] - [10, 10] )^2)` which is `mean([ -4, -2]^2 )` = `mean([16, 4])` = `10.0`.
        *   Call a new (not-yet-created) `compute_residual_mse` function with these tensors.
        *   Assert that the function returns a value approximately equal to `10.0`.

8.  ✅ **Create new loss module and function:** Create a new file at `chipvi/training/losses.py`.
    *   **Provide the following function signature:**
        ```python
        import torch

        def compute_residual_mse(
            y_r1: torch.Tensor,
            mu_tech_r1: torch.Tensor,
            y_r2: torch.Tensor,
            mu_tech_r2: torch.Tensor,
            sd_ratio_r1_to_r2: torch.Tensor,
        ) -> torch.Tensor:
        ```
    *   **Implementation Logic:** Port the logic for calculating the scaled residual MSE from one of the training scripts (e.g., `runs/06_27_bigger_200bp/train.py`, line 211).

9.  ✅ **Run tests:** Run `pytest` from the root directory and ensure all new tests pass.

---

### Task Group 3: Refactor `Dataset` and Data Loading Logic ✅ COMPLETED

This task group will refactor `MultiReplicateDataset` to return a structured dictionary instead of a flat tensor, and adjust the `build_datasets` function accordingly.

1.  ✅ **Define a new test for the refactored `MultiReplicateDataset`:** In `tests/data/test_datasets.py`, add a new test case for the *target* behavior.
    *   **Test Case Description:** Write a test named `test_refactored_multireplicatedataset_returns_structured_dict`. This test must:
        *   Create the same small, identifiable `torch.Tensor` objects for the dataset inputs as in the previous task group.
        *   Instantiate `MultiReplicateDataset` with these tensors. (Note: The test will fail until the class is refactored).
        *   Get the first item: `item = dataset[0]`.
        *   Assert that `item` is a dictionary with top-level keys: `'r1'`, `'r2'`, and `'metadata'`.
        *   Assert that `item['r1']` is a dictionary with keys `'covariates'` and `'reads'`.
        *   Assert that the tensor `item['r1']['covariates']` has the correct shape (5,) and contains the correct values from the five `replicate_1` input tensors.
        *   Assert that the scalar `item['r1']['reads']` contains the correct value from `experiment_reads_r1`.
        *   Perform the same assertions for `item['r2']`.
        *   Assert that `item['metadata']['sd_ratio']` contains the correct value.

2.  ✅ **Refactor `MultiReplicateDataset`:** In `chipvi/data/datasets.py`, modify the `MultiReplicateDataset` class.
    *   **Keep the `__init__` signature and logic the same for now.** It will still accept the long list of tensors.
    *   **Remove the `self.covariates` and `self.experiment_reads` attributes** that stack the tensors.
    *   **Modify the `__getitem__` method:** Change the implementation to construct and return a nested dictionary for the given `idx` as specified in the test above.
    *   **Implementation Logic:**
        ```python
        # Example logic for __getitem__
        def __getitem__(self, idx: int) -> dict:
            r1_covariates = torch.stack([
                self.control_reads_r1[idx],
                self.control_mapq_r1[idx],
                self.control_seq_depth_r1[idx],
                self.experiment_mapq_r1[idx],
                self.experiment_seq_depth_r1[idx],
            ])
            r2_covariates = torch.stack([
                self.control_reads_r2[idx],
                self.control_mapq_r2[idx],
                self.control_seq_depth_r2[idx],
                self.experiment_mapq_r2[idx],
                self.experiment_seq_depth_r2[idx],
            ])
            return {
                'r1': {'covariates': r1_covariates, 'reads': self.experiment_reads_r1[idx]},
                'r2': {'covariates': r2_covariates, 'reads': self.experiment_reads_r2[idx]},
                'metadata': {
                    'sd_ratio': self.exp_sd_ratio[idx],
                    'grp_idx': self.grp_idxs[idx],
                }
            }
        ```
    *   **Modify the `get_dim_x` method:** This method is now ambiguous. Change it to `get_covariate_dim` and have it return the number of covariates for a *single* replicate (i.e., 5).

3.  ✅ **Refactor `SingleReplicateDataset` for consistency:** In `chipvi/data/datasets.py`, modify the `SingleReplicateDataset` class.
    *   **Remove the `self.covariates` attribute.**
    *   **Modify the `__getitem__` method** to return a dictionary: `{'covariates': ..., 'reads': ...}`.
    *   **Rename `get_dim_x` to `get_covariate_dim`**.

4.  ✅ **Clean up old tests:** In `tests/data/test_datasets.py`, **delete** the now-obsolete test `test_multireplicatedataset_getitem_structure`, as it validates the old, brittle design.

5.  ✅ **Run tests:** Run `pytest` from the root directory.
    *   The new test `test_refactored_multireplicatedataset_returns_structured_dict` should now pass.
    *   All other existing tests should continue to pass.

---

### Task Group 4: Implement the Unified Trainer ✅ COMPLETED

This task group will create the new `Trainer` class and refactor the custom loss logic into a standalone, testable function that the trainer will use.

1.  ✅ **Define a test for the new `Trainer` class:** Create a new file at `tests/training/test_trainer.py`.
    *   **Test Case Description:** Write a test named `test_trainer_completes_one_epoch`. This test must:
        *   Create a mock model (e.g., a simple `nn.Linear` model).
        *   Create mock `DataLoaders` for training and validation that yield the new dictionary-based batch structure. The data can be simple random tensors.
        *   Create a mock optimizer (e.g., `optim.Adam(model.parameters(), lr=0.1)`).
        *   Instantiate the new `Trainer` class with these components, along with a reference to the `compute_residual_mse` loss function created in Task Group 2.
        *   Call a `fit()` method on the trainer instance, configured to run for exactly one epoch.
        *   Assert that the `fit()` method completes without errors.
        *   Assert that the model's parameters have been updated after the training step (i.e., they are different from their initial values).

2.  ✅ **Refactor the custom loss function to a more generic interface:** In `chipvi/training/losses.py`, modify the `compute_residual_mse` function.
    *   **Modify the function signature** to accept the model's output and the batch dictionary directly. This decouples the loss from the main training loop.
        ```python
        import torch
        from torch import nn
        
        def replicate_concordance_mse_loss(model_outputs: dict, batch: dict) -> torch.Tensor:
        ```
    *   **Implementation Logic:**
        *   The function will now be responsible for unpacking the necessary data from the `model_outputs` and `batch` dictionaries.
        *   `model_outputs` will be a dictionary like `{'r1': {'dist': ...}, 'r2': {'dist': ...}}`.
        *   `batch` is the dictionary from our `MultiReplicateDataset`.
        *   Extract `y_r1`, `y_r2`, `sd_ratio_r1_to_r2` from the `batch` dict.
        *   Extract the predicted means (`.mean`) from the distribution objects in `model_outputs`.
        *   Calculate the residuals and the final MSE scalar, just as before. Return the MSE tensor.

3.  ✅ **Create the unified `Trainer` class:** Create a new file at `chipvi/training/trainer.py`.
    *   **Provide the following class and method signatures:**
        ```python
        from typing import Callable
        import torch
        from torch import nn, optim
        from torch.utils.data import DataLoader
        
        class Trainer:
            def __init__(
                self,
                model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer: optim.Optimizer,
                loss_fn: Callable[[dict, dict], torch.Tensor],
                device: torch.device,
            ):
        
            def fit(self, num_epochs: int):
        
            def _train_one_epoch(self) -> float:
        
            def _validate_one_epoch(self) -> float:
        ```
    *   **Implementation Logic for `__init__`**: Store all the arguments as instance attributes (`self.model`, `self.train_loader`, etc.).
    *   **Implementation Logic for `fit`**: This is the main public method. It should contain a `for` loop that iterates from `0` to `num_epochs`. Inside the loop, it should call `_train_one_epoch` and `_validate_one_epoch`, and should include logging for the results of each epoch.
    *   **Implementation Logic for `_train_one_epoch`**:
        *   Set the model to train mode: `self.model.train()`.
        *   Iterate over `self.train_loader`. For each batch:
            *   Move the batch's tensors to `self.device`.
            *   Zero the gradients: `self.optimizer.zero_grad()`.
            *   **Perform the forward pass:** Since the model (`TechNB_mu_r`) takes covariates and returns distribution parameters, you'll need to call it for each replicate separately. The structure should be:
                ```python
                # Unpack covariates from the batch dict
                x_r1 = batch['r1']['covariates']
                x_r2 = batch['r2']['covariates']
                
                # Predict distribution for each replicate
                mu_r1, r_r1 = self.model(x_r1)
                mu_r2, r_r2 = self.model(x_r2)

                # Create distribution objects (using get_torch_nb_dist)
                # and package them for the loss function
                model_outputs = {
                    'r1': {'mu': mu_r1, 'r': r_r1}, 
                    'r2': {'mu': mu_r2, 'r': r_r2},
                }
                ```
            *   **Calculate loss:** Call the `self.loss_fn(model_outputs, batch)`.
            *   Perform backpropagation: `loss.backward()`.
            *   Update weights: `self.optimizer.step()`.
        *   Return the average loss for the epoch.
    *   **Implementation Logic for `_validate_one_epoch`**: This is similar to `_train_one_epoch` but wrapped in a `with torch.no_grad():` block, sets the model to `self.model.eval()`, and does not perform backpropagation or optimizer steps.

4.  ✅ **Run tests:** Run `pytest` from the root directory.
    *   The new test `test_trainer_completes_one_epoch` should pass, confirming that the trainer can successfully orchestrate the forward pass, loss calculation, and backpropagation using the new dictionary-based data format.
5.  ✅ **(REVIEW) Modify loss function:** In `chipvi/training/losses.py`, refactor the `replicate_concordance_mse_loss` function to call the `compute_residual_mse` function.
6.  ✅ **(REVIEW) Modify trainer class:** In `chipvi/training/trainer.py`, remove the `.squeeze()` calls from the `model_outputs` dictionary creation in the `_train_one_epoch` and `_validate_one_epoch` methods.
7.  ✅ **(REVIEW) Modify trainer class:** In `chipvi/training/trainer.py`, simplify the `_move_batch_to_device` function to be a non-recursive function.
8.  ✅ **(REVIEW) Run tests:** Run `pytest` from the root directory and ensure all tests pass.

---

### Task Group 5: Create a Config-Driven Experiment Pipeline

This task group will integrate all previously refactored components, allowing us to launch training runs from a single script driven by configuration files.

1.  **Create model and data configuration files:** In the `configs/` directory, create the following subdirectories and files to define the components of our experiments.
    *   Create `configs/model/` directory.
    *   Create `configs/model/nb_mu_r_small.yaml`:
        ```yaml
        # configs/model/nb_mu_r_small.yaml
        _target_: chipvi.models.technical_model.TechNB_mu_r
        covariate_dim: 5 # This will be overridden by data config
        hidden_dims_mu: [32, 32]
        hidden_dims_r: [8, 8]
        ```
    *   Create `configs/data/` directory.
    *   Create `configs/data/h3k27me3_200bp.yaml`:
        ```yaml
        # configs/data/h3k27me3_200bp.yaml
        name: "h3k27me3"
        aggregation_factor: 8 # 25bp -> 200bp
        batch_size: 8192
        # Paths will be resolved at runtime
        ```

2.  **Create a complete experiment configuration:** In `configs/`, create an `experiment/` directory. Inside it, create `configs/experiment/exp_001_concordance.yaml`. This file will compose the pieces into a runnable experiment.
    ```yaml
    # configs/experiment/exp_001_concordance.yaml
    defaults:
      - _self_
      - data: h3k27me3_200bp
      - model: nb_mu_r_small
      - override hydra/job_logging: colorlog
      - override hydra/hydra_logging: colorlog
    
    # Main training parameters
    training:
      num_epochs: 100
      learning_rate: 0.001
      weight_decay: 0.01
      patience: 10
      device: "cuda:0"
      # Name of the loss function to use
      loss_fn: "replicate_concordance_mse"
    
    # Name for this run group in W&B
    wandb:
      group: "refactored_runs"
      project: "chipvi"
    ```

3.  **Update the main script to be a full-fledged runner:** In `scripts/run.py`, replace the existing placeholder `main` function with a complete implementation that uses the configuration to set up and run the training.
    *   **Provide the following function signature and implementation logic:**
        ```python
        import hydra
        import torch
        from omegaconf import DictConfig, OmegaConf
        
        # Import our refactored components
        from chipvi.data.datasets import build_datasets # We will refactor this next
        from chipvi.training.trainer import Trainer
        from chipvi.training.losses import replicate_concordance_mse_loss
        
        # A simple factory to get the loss function from the config string
        LOSS_REGISTRY = {
            "replicate_concordance_mse": replicate_concordance_mse_loss,
            # Add other losses here as they are created
        }
        
        @hydra.main(config_path="../configs", config_name="config", version_base=None)
        def main(cfg: DictConfig) -> None:
            """
            Main entry point for running ChipVI experiments using Hydra.
            """
            print("--- Configuration ---")
            print(OmegaConf.to_yaml(cfg))
        
            # 1. Setup: Device, Paths, etc.
            device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
        
            # 2. Data Loading (Still using old build_datasets for now)
            # NOTE: This part is a temporary bridge and will be refactored next.
            # We pass the whole config to it for now.
            train_ds, val_ds = build_datasets(cfg) # This function will need to be adapted
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.data.batch_size, shuffle=False)
        
            # 3. Model Initialization
            # Hydra instantiates the model for us using the _target_ field
            model = hydra.utils.instantiate(
                cfg.model,
                covariate_dim=train_ds.get_covariate_dim()
            ).to(device)
        
            # 4. Optimizer and Loss Function
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay
            )
            loss_fn = LOSS_REGISTRY[cfg.training.loss_fn]
        
            # 5. Trainer Initialization and Execution
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
            )
            
            print("\n--- Starting Training ---")
            trainer.fit(num_epochs=cfg.training.num_epochs)
            print("--- Training Finished ---")
        
        if __name__ == "__main__":
            main()
        ```

4.  **Adapt `build_datasets` as a temporary bridge:** In `chipvi/data/datasets.py`, modify the `build_datasets` function signature to accept the `DictConfig` object.
    *   **Implementation Logic:** For now, this function will act as a bridge. It will read the necessary parameters (like file paths and chromosome lists) from the `cfg` object instead of from its own function arguments. This is an intermediate step before we fully decompose this monolithic function. *Do not implement the full decomposition yet.* Just make it compatible with the new `scripts/run.py` structure.

5.  **Deprecate old scripts:**
    *   **Delete all files** inside the `runs/` directory (e.g., `rm -rf runs/*`).
    *   **Delete the old trainer files:** `chipvi/training/phase_a.py`, `chipvi/training/tech_biol_rep_model_trainer.py`, `chipvi/training/technical_model_trainer.py`, and `chipvi/training/technical_model_trainer_old.py`.
    *   **Delete `chipvi/utils/tmux.py`**.

6.  **Run an experiment:** From your terminal, execute the new pipeline:
    ```bash
    python scripts/run.py experiment=experiment/exp_001_concordance
    ```    This command should successfully load the configuration, build the components, and start the training process using the new unified `Trainer`.

---

### Task Group 6: Decompose Data Loading and Implement Memory-Efficient Pre-processing

This task group replaces the in-memory data preparation with a script that pre-processes the data and saves it to disk in an efficient format, which the `Dataset` classes will then load directly.

1.  **Define a data pre-processing configuration:** In `configs/data/`, create a new configuration file named `h3k27me3_preprocess.yaml`. This will control the new pre-processing script.
    ```yaml
    # configs/data/h3k27me3_preprocess.yaml
    name: "h3k27me3"
    
    # List of replicate groups. Each group has two replicates (r1, r2).
    # Each replicate has an experiment file and a control file.
    # NOTE: In a real implementation, this would be populated programmatically.
    # For this task, we will use placeholder paths.
    replicate_groups:
      - r1:
          exp: "path/to/exp1.bed"
          ctrl: "path/to/ctrl1.bed"
        r2:
          exp: "path/to/exp2.bed"
          ctrl: "path/to/ctrl2.bed"
    
    # Define chromosome splits for train/validation
    train_chroms: ["chr1", "chr2", ..., "chr7"]
    val_chroms: ["chr8", "chr9"]
    
    # Aggregation settings
    aggregation_factor: 8 # e.g., 25bp -> 200bp
    
    # Output file paths (will be resolved relative to processed_data_dir)
    output_prefix_train: "h3k27me3_train_200bp"
    output_prefix_val: "h3k27me3_val_200bp"
    ```

2.  **Define a test for the new pre-processing pipeline:** In `tests/data/test_preprocessing.py`, add a new test.
    *   **Test Case Description:** Write a test named `test_preprocessing_pipeline_creates_valid_npy_files`.
        *   Use `tmp_path` from `pytest` to create temporary directories for raw data and processed output.
        *   Create dummy raw BED files (e.g., `exp1.bed`) with simple, predictable data.
        *   Create a mock configuration object pointing to these temporary files.
        *   Call a new (not-yet-created) `run_preprocessing` function from `chipvi/data/preprocessing.py` with the mock config.
        *   Assert that the expected `.npy` files (e.g., `h3k27me3_train_200bp_control_reads_r1.npy`) are created in the temporary processed directory.
        *   Load one of the created `.npy` files and assert its contents are correct based on the dummy input data and the aggregation logic.

3.  **Implement the pre-processing pipeline logic:** In `chipvi/data/preprocessing.py`, create the main pipeline function.
    *   **Provide the following function signature:**
        ```python
        from omegaconf import DictConfig
        
        def run_preprocessing(cfg: DictConfig):
        ```
    *   **Implementation Logic:**
        *   This function will orchestrate the entire data preparation process.
        *   It should contain the core logic currently inside `build_datasets`, such as looping through replicate groups, loading BED files into pandas DataFrames using a helper function (`load_and_process_bed`), filtering by chromosome, and applying the `aggregate_bins` function from Task Group 2.
        *   Instead of appending results to in-memory lists, it should collect the processed numpy arrays for each data type (e.g., `train_control_reads_r1`, `val_experiment_mapq_r2`, etc.).
        *   At the end, it should save each of these final arrays to disk as a separate `.npy` file in the configured processed data directory. The filenames should be constructed from the `output_prefix` in the config (e.g., `f"{prefix}_{array_name}.npy"`).

4.  **Create the pre-processing script entry point:** Create a new file at `scripts/preprocess.py`.
    *   **Provide the following implementation:**
        ```python
        import hydra
        from omegaconf import DictConfig
        
        from chipvi.data.preprocessing import run_preprocessing
        
        @hydra.main(config_path="../configs/data", config_name="h3k27me3_preprocess", version_base=None)
        def main(cfg: DictConfig) -> None:
            """
            Script to run the data pre-processing pipeline.
            """
            print("--- Running Data Pre-processing ---")
            run_preprocessing(cfg)
            print("--- Pre-processing Finished ---")
        
        if __name__ == "__main__":
            main()
        ```

5.  **Refactor `Dataset` classes to load from `.npy` files:** In `chipvi/data/datasets.py`:
    *   **DELETE the `build_datasets` function.** It is now obsolete.
    *   **Modify the `MultiReplicateDataset` `__init__` method.** It should now accept a prefix path (e.g., `data/processed/h3k27me3_train_200bp`) and load all necessary arrays from the corresponding `.npy` files using `np.load(f"{prefix_path}_control_reads_r1.npy", mmap_mode='r')`.
    *   Do the same for `SingleReplicateDataset`.

6.  **Update `scripts/run.py` to use the new `Dataset`:**
    *   Remove the call to `build_datasets`.
    *   Instantiate `MultiReplicateDataset` directly, passing the processed data prefix path derived from the experiment configuration.

7.  **Run tests:** Run `pytest` from the root directory. All tests, including the new `test_preprocessing_pipeline_creates_valid_npy_files`, should pass.

---

### Task Group 7: Final Code Cleanup and Robustness Improvements

This task group will address the "Gradient Masking," "Arbitrary Value Replacement," and "Mixing of Library and Executable Logic" findings.

1.  **Define a test for numerical stability in distributions:** In `tests/distributions.py`, add a new test case.
    *   **Test Case Description:** Write a test named `test_get_torch_nb_dist_raises_error_on_invalid_input`.
        *   Use `pytest.raises` to assert that `get_torch_nb_dist` raises a `ValueError` when `mu` contains a `NaN` value.
        *   Use `pytest.raises` to assert that `get_torch_nb_dist` raises a `ValueError` when `r` contains a zero or negative value.

2.  **Improve numerical robustness in distribution utility:** In `chipvi/utils/distributions.py`, refactor the `get_torch_nb_dist` function.
    *   **Remove the `if torch.isnan(mu).any()...` and `if (p == 0).any()...` blocks** that replace invalid values with arbitrary constants.
    *   **Implementation Logic:** Before calculating `p`, add assertions to validate the inputs. If `r` contains non-positive values, or if `mu` contains `NaN` or `inf` values, raise a `ValueError` with a descriptive error message. This enforces that the calling code is responsible for providing valid distribution parameters.

3.  **Remove gradient masking from old trainer:** This task is a cleanup action. The file `chipvi/training/tech_biol_rep_model_trainer.py` should have already been deleted in Task Group 5. **Verify that this file no longer exists.** If it does, delete it now. This resolves the "Gradient Masking Hides Potential Numerical Instability" finding.

4.  **Refactor data downloading script:** In `chipvi/utils/public_data_processing/download_entex_files.py`:
    *   **Remove the `if __name__ == "__main__":` block** and the `main` and `parse_args` functions. This file should now only contain reusable library functions like `download_and_process_entex_files` and `process_bam_file`.

5.  **Create a dedicated script for data downloading:** In the `scripts/` directory, create a new file named `download_data.py`.
    *   **Implementation Logic:**
        *   Move the argument parsing logic (e.g., using `argparse` or `hydra-zen`) into this file.
        *   This script will be the new user-facing entry point for downloading data.
        *   It should import the `download_and_process_entex_files` function from `chipvi.utils.public_data_processing.download_entex_files` and call it with the parsed arguments.
        *   Do the same for `get_entex_metadata.py`, creating a corresponding script in `scripts/` and removing its executable block.

6.  **Update documentation:** In `README.md` or a new `CONTRIBUTING.md`, add a section explaining the new workflow:
    *   How to configure and run the data pre-processing pipeline using `scripts/preprocess.py`.
    *   How to configure and run a new experiment using `scripts/run.py`.
    *   How to download new data using `scripts/download_data.py`.

7.  **Run all tests:** Run `pytest` from the root directory. All tests should continue to pass, confirming that our final cleanup has not introduced any regressions.


---
