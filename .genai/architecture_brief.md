# Architecture Brief: ChipVI

## 1. Directory Structure (Current, pre-refactor)

```
chipvi/
  analysis/
    gene_expression.py
    replicate_concordance.py
  data/
    datasets.py
    experiment_collections.py
  models/
    common.py
    technical_model.py
  training/
    phase_a.py
    tech_biol_rep_model_trainer.py
    technical_model_trainer_old.py
    technical_model_trainer.py
  utils/
    public_data_processing/
      download_entex_files.py
      download_h3k27me3.py
      get_entex_metadata.py
      process_entex_bam.sh
    distributions.py
    path_helper.py
    plots.py
    tmux.py
runs/
  ... (multiple experiment-specific directories)
tests/
  distributions.py
```

## 2. Data Models and Flow (Current, pre-refactor)

The project follows a multi-stage data processing and modeling pipeline:

1.  **Data Acquisition (External)**: The process begins with downloading raw experiment data. `chipvi/utils/public_data_processing/download_entex_files.py` and `get_entex_metadata.py` fetch BAM files and associated metadata JSONs from the ENCODE portal.
2.  **Pre-processing (Shell)**: The `process_entex_bam.sh` script is a critical pre-processing step. It uses tools like `bedtools` and `samtools` to convert each BAM file into a binned BED file. This script takes bin size as a parameter; for instance, the current data is processed at a 25bp resolution. Each row in the resulting BED file represents a genomic bin of the specified size and contains the total read count and the mean MAPQ score for that bin.
3.  **Dataset Construction (Python/PyTorch)**:
    *   The `chipvi/data/datasets.py::build_datasets` function is the primary entry point for creating model-ready datasets. However, some training scripts (e.g., `runs/06_27_bigger_200bp/train.py`) perform an additional on-the-fly aggregation step, reading the high-resolution 25bp data and combining adjacent bins (e.g., summing 8 bins) to create lower-resolution bins (e.g., 200bp) before passing them to the Dataset classes.
    *   For each genomic bin, it assembles a vector of covariates. For a single replicate, this includes control reads, control MAPQ, control sequencing depth, experiment MAPQ, and experiment sequencing depth.
    *   For multiple replicates (`MultiReplicateDataset`), it stacks the covariates from two replicates side-by-side and also includes group indices to identify which samples belong to the same biological source.
    *   The target variable (`y`) is the experimental read count for each bin. For multi-replicate data, `y` is a tensor containing the read counts for both replicates and the ratio of their sequencing depths.
4.  **Model Training**:
    *   The training scripts (e.g., in `runs/`) load the `Dataset` objects into PyTorch `DataLoaders`.
    *   During a training step, a batch of covariate tensors (`x_i`) and target tensors (`y_i`) is passed to the model.
    *   The model (e.g., `TechNB_mu_r`) ingests the covariates and outputs the predicted parameters for a distribution (e.g., `mu_tech` and `r_tech` for a Negative Binomial).
    *   The loss function is computed based on these parameters. The primary component is the negative log-likelihood of the observed read counts (`y_i`) given the predicted distribution.
    *   Some trainers add regularization terms to the loss, such as the Mean Squared Error or Pearson correlation between the biological residuals (`observed_reads - predicted_technical_reads`) of the two replicates.
5.  **Output**: The pipeline's final outputs are the trained model weights (saved as `.pt` files), and logs and visualizations sent to Weights & Biases for experiment tracking.


## 3. Key Components (Current, pre-refactor)

### `chipvi.data.datasets`

*   **`build_datasets`**: This is a monolithic function that orchestrates the entire data preparation process. It reads multiple dataframes, performs filtering, combines data from different replicates, and materializes large Python lists in memory before finally converting them into PyTorch tensors. This design is memory-intensive and tightly couples data loading with pre-processing logic.
*   **`MultiReplicateDataset`**: This class represents paired-replicate data. Its `__getitem__` method returns a single covariate tensor that concatenates features from both replicates, along with a group index. The target tensor packs read counts and the sequencing depth ratio. This structure requires downstream consumers (the model and training loop) to be aware of the specific slicing indices to separate data for each replicate.

### `chipvi.models.technical_model`

*   **`TechNB_mu_r`**: This `nn.Module` implements a Negative Binomial model where the dispersion `r` is predicted based on both the input covariates *and* the predicted mean `mu`. This creates a sequential dependency within the forward pass: `log_mu_tech = f(x_i)` and `log_r_tech = g(log_mu_tech, x_i)`.
*   **`TechNB_r_p`**: An alternative parameterization where dispersion `r` and success probability `p` are predicted independently from the same input covariates.
*   **`MLP` (`common.py`)**: A generic, reusable Multi-Layer Perceptron module that serves as the backbone for all predictive models.

### `chipvi.training` & `runs/`

*   The architecture is characterized by significant code duplication across the `training` and `runs` directories. Multiple files (`technical_model_trainer.py`, `technical_model_trainer_old.py`, `tech_biol_rep_model_trainer.py`, and the various `train.py` scripts in `runs/`) contain very similar training loops.
*   This pattern suggests that new experiments or model variations are created by copying and modifying an existing training script. While effective for rapid iteration in a research context, it leads to poor maintainability and makes it difficult to enforce consistent behavior or apply global improvements.
*   The `runs/**/dispatch.py` scripts use `chipvi.utils.tmux` to programmatically create and manage `tmux` sessions, splitting windows to run multiple training commands in parallel. This is a bespoke, script-based approach to experiment management.

---

## 4. Proposed Architecture (Delta)

This section outlines the planned refactoring. We will transition from the current structure to a more robust, configurable, and maintainable architecture.

### 4.1. New Directory Structure

The project will be reorganized to enforce a clear separation of concerns between library code, configuration, executable scripts, and outputs. The `runs/` directory will be deprecated.

```
chipvi/
  data/
    datasets.py               # Refactored Dataset classes
    preprocessing.py          # NEW: Data aggregation and preparation logic
    ...
  models/
    ... (no major changes)
  training/
    trainer.py                # NEW: Unified, configurable Trainer class
    losses.py                 # NEW: Houses all loss functions
    # OLD trainer files will be DELETED
  utils/
    distributions.py          # Refactored for numerical stability
    path_helper.py            # Refactored to be config-driven
    ... (tmux.py will be deprecated)
scripts/                      # NEW
  run.py                      # NEW: Single entry point for all training runs
  preprocess.py               # NEW: Entry point for data pre-processing pipeline
  download_data.py            # NEW: Entry point for data acquisition
tests/                        # EXPANDED
  test_datasets.py            # NEW
  test_preprocessing.py       # NEW
  test_models.py              # NEW
  test_losses.py              # NEW
  ...
configs/                      # NEW
  data/
    h3k27me3_25bp.yaml
    h3k27me3_200bp.yaml
  model/
    nb_mu_r_small.yaml
  experiment/                 # NEW: Defines complete experiments
    exp_001_baseline.yaml
    exp_002_nce_loss.yaml
  config.yaml                 # NEW: Main config file
outputs/                      # NEW: Structured output directory for models and logs
  YYYY-MM-DD/
    HH-MM-SS/
      .hydra/
      model.pt
      ...
```

### 4.2. New Data and Experiment Flow

1.  **Configuration**: An entire experiment is defined by a single configuration file (e.g., `configs/experiment/exp_001_baseline.yaml`). This file specifies the dataset, model, hyperparameters, and loss function to use. A tool like Hydra will manage loading and composing these configurations.
2.  **Data Pre-processing**: The `scripts/preprocess.py` script will be the single entry point for preparing data. It will orchestrate running the shell scripts and the new Python-based aggregation logic from `chipvi/data/preprocessing.py`. This step produces analysis-ready, memory-mapped numpy arrays.
3.  **Experiment Execution**: The user runs `python scripts/run.py experiment=exp_001_baseline`.
    *   Hydra loads and resolves the configuration.
    *   `run.py` uses the config to instantiate the appropriate `Dataset` from `chipvi/data/datasets.py`. The dataset now loads data from memory-mapped files.
    *   It instantiates the specified model (`chipvi/models/`) and loss function (`chipvi/training/losses.py`).
    *   It instantiates the unified `Trainer` from `chipvi/training/trainer.py`, passing the model, data loaders, loss function, and other configured components.
    *   The `trainer.fit()` method is called.
4.  **Training Loop**:
    *   The `Trainer` pulls a batch from the `DataLoader`. The batch is now a **dictionary** of tensors (e.g., `{'r1': {'covariates': ...}, 'r2': {'covariates': ...}, 'target': ...}`).
    *   The `Trainer` executes the forward pass and passes the model output and batch to the configured loss function.
    *   The `Trainer` handles the backward pass, optimization, logging, and model checkpointing. All logic is contained within this single class.

### 4.3. Key Component Deltas

*   **`chipvi.training.trainer.Trainer` (New)**: A new class that encapsulates the entire training and validation loop. It will be initialized with a model, optimizer, loss function, and data loaders. It will contain the logic for epoch iteration, device placement, gradient calculation, backpropagation, and logging.

*   **`chipvi.data.datasets.MultiReplicateDataset` (Refactored)**:
    *   The `__init__` method will no longer perform complex data processing. It will expect paths to pre-processed, memory-mapped data files.
    *   The `__getitem__` method will be changed to return a dictionary of tensors, eliminating the need for downstream index-based slicing. Example: `{'r1': {'covariates': ...}, 'r2': ...}`.

*   **`chipvi.data.preprocessing` (New Module)**: This new module will contain the Python-based data preparation logic, such as the 25bp-to-200bp aggregation, which is currently hidden in experiment scripts.

*   **`configs/**` (New)**: This directory will contain YAML files defining all configurable aspects of the project, from data paths and model hyperparameters to the composition of different loss functions. This replaces all `runs/**/*.py` scripts.

*   **`tests/**` (Expanded)**: A full `pytest` suite will be developed to ensure the correctness of these refactored components before the old code is removed. This includes unit tests for data aggregation, `Dataset` output structure, and loss calculations.