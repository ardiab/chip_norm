# Project Architecture: ChipVI

## 1. Project Purpose

The ChipVI project is a computational framework for modeling technical noise and biological variation in ChIP-seq data. Its primary goal is to analyze read counts from sequencing experiments, distinguishing the baseline technical signal from the true biological signal. The system is designed to handle single-end experiments as well as paired biological replicates, using the concordance between replicates to help model the biological component of the signal.

The project uses deep learning models, specifically Multi-Layer Perceptrons (MLPs), to predict the parameters of statistical distributions (like Poisson and Negative Binomial) that describe read counts. This allows for a quantitative, data-driven approach to understanding and decomposing the sources of variation in high-throughput sequencing data.

## 2. High-Level Architecture & Experiment Flow

The project is designed around a clear, configuration-driven workflow that separates data preparation, model training, and experiment management. This architecture ensures reproducibility and simplifies experimentation.

1.  **Configuration**: The entire experimental process is defined by a set of hierarchical YAML configuration files managed by the Hydra framework. A main experiment file (e.g., `configs/experiment/exp_001_baseline.yaml`) composes smaller, reusable configuration modules for the dataset, model, training parameters, and loss function. This allows for flexible and clear experiment definition without code changes.

2.  **Data Acquisition & Pre-processing**: A set of executable scripts located in the `scripts/` directory handles the initial data pipeline.
    *   `scripts/download_data.py` manages the download of raw data and metadata from public repositories like ENCODE.
    *   `scripts/preprocess.py` orchestrates the conversion of raw BAM files into binned read count and quality score data. This involves shell scripts using standard bioinformatics tools (`bedtools`, `samtools`) and a dedicated Python module (`chipvi/data/preprocessing.py`) for any necessary aggregation or transformation (e.g., changing genomic bin resolution). The output of this stage is a set of analysis-ready, memory-mapped data files.

3.  **Experiment Execution**: All training runs are launched through a single, unified entry point: `scripts/run.py`. A user can execute an experiment by invoking this script and specifying the experiment configuration:
    `python scripts/run.py experiment=exp_001_baseline`

4.  **Training Pipeline**: The `run.py` script, guided by the Hydra configuration, performs the following steps:
    *   Instantiates the appropriate PyTorch `Dataset` and `DataLoader` from the `chipvi.data` module.
    *   Instantiates the specified neural network model from `chipvi.models`.
    *   Instantiates the corresponding loss function from `chipvi.training.losses`.
    *   Passes these components to a unified `Trainer` object from `chipvi.training.trainer`.
    *   Calls the `trainer.fit()` method, which executes the complete training and validation loop, handling optimization, logging, and checkpointing.

5.  **Outputs**: All artifacts from a run, including trained model weights (`.pt` files), logs, and a snapshot of the configuration, are saved to a structured, timestamped directory under `outputs/`, managed automatically by Hydra.

## 3. Core Components

The project is organized into a modular library (`chipvi/`), configuration files (`configs/`), executable scripts (`scripts/`), and structured outputs (`outputs/`).

### `configs/` - Configuration
This directory is the control center for the entire project. It contains YAML files that define all configurable aspects, including data sources, model architectures, hyperparameters, and optimizer choices. This approach decouples the experimental setup from the core library code, enhancing maintainability and reproducibility.

### `scripts/` - Executable Entry Points
This directory provides the main user-facing interfaces for interacting with the framework. The key script is `run.py`, which serves as the single dispatcher for all model training and evaluation tasks.

### `chipvi/data/` - Data Pipeline
This package handles the loading and preparation of data for the models.
*   **Preprocessing (`preprocessing.py`)**: Contains logic for data aggregation and transformation, such as adjusting the resolution of genomic bins.
*   **Datasets (`datasets.py`)**: Contains PyTorch `Dataset` classes (`SingleReplicateDataset`, `MultiReplicateDataset`). These classes are responsible for loading the pre-processed, memory-mapped data files. A key feature of the `MultiReplicateDataset` is that it yields batches as a **dictionary of tensors**. This provides a clean, explicit interface for the model and training loop, with keys like `'r1'` and `'r2'` to distinguish between replicate data, avoiding ambiguity.

### `chipvi/models/` - Neural Network Models
This package defines the neural network architectures used for prediction.
*   **MLP Backbone**: A common, reusable Multi-Layer Perceptron (MLP) serves as the foundation for the predictive models.
*   **Distribution Models**: The primary models are `nn.Module` classes that use the MLP backbone to predict the parameters of a statistical distribution. For example, `TechNB_mu_r` predicts the mean (`mu`) and dispersion (`r`) of a Negative Binomial distribution from input covariates. Different parameterizations (e.g., predicting `r` and `p` instead) exist as separate models.

### `chipvi/training/` - Model Training
This package contains the unified logic for model training and optimization.
*   **Losses (`losses.py`)**: This module centralizes all loss functions. The primary loss is the negative log-likelihood of the observed read counts given the model's predicted distribution. It also includes more complex loss functions that can regularize the model by, for example, penalizing the divergence between biological signal residuals in paired replicates.
*   **Trainer (`trainer.py`)**: This module contains a single, powerful `Trainer` class that encapsulates the entire training and validation loop. It is initialized with the model, data loaders, optimizer, and loss function. It handles device placement, forward/backward passes, gradient updates, logging to external services (like Weights & Biases), and saving model checkpoints. This centralized design eliminates redundant code and ensures consistent training behavior across all experiments.