# Project Brief: ChipVI

## 1. Project Purpose

The ChipVI project is a computational framework for modeling technical noise and biological variation in ChIP-seq data. Its primary goal is to analyze read counts from sequencing experiments, distinguishing the baseline technical signal from the true biological signal. The system is designed to handle single-end experiments as well as paired biological replicates, using the concordance between replicates to help model the biological component of the signal.

The project uses deep learning models, specifically Multi-Layer Perceptrons (MLPs), to predict the parameters of statistical distributions (Poisson, Negative Binomial) that describe read counts. This allows for a quantitative, data-driven approach to understanding and decomposing the sources of variation in high-throughput sequencing data.

## 2. Core Components

The project is organized into several key modules:

*   **Data Acquisition & Pre-processing (`chipvi/utils/public_data_processing`)**: A set of Python and shell scripts designed to download experiment data and metadata from public repositories (ENCODE/ENTEx), and process raw BAM files into binned read count and quality score (MAPQ) data in BED format.
*   **Data Loading (`chipvi/data`)**: Contains PyTorch `Dataset` classes (`SingleReplicateDataset`, `MultiReplicateDataset`) and a factory function (`build_datasets`) that constructs these datasets from the pre-processed BED files. This module is responsible for preparing covariates (e.g., control reads, sequencing depth, MAPQ) for model consumption.
*   **Modeling (`chipvi/models`)**: Defines the neural network architectures. The core models (`TechNB_mu_r`, `TechNB_r_p`, `TechPoisson`) use MLPs to predict the parameters of count distributions based on the provided covariates.
*   **Training (`chipvi/training`)**: Implements the model training and validation loops. This includes standard log-likelihood-based loss functions as well as more complex, custom loss functions that incorporate biological replicate concordance (e.g., by penalizing divergence in signal residuals between replicates).
*   **Experiment Execution (`runs/`)**: A collection of scripts organized into timestamped or named directories, each representing a specific set of experiments. These scripts handle the configuration and launching of training runs, often in parallel using `tmux`.
*   **Utilities (`chipvi/utils`)**: A suite of helper modules for tasks such as calculating statistical distributions, managing file paths, and generating plots.