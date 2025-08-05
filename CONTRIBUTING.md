# Contributing to ChipVI

This document outlines the new development workflow for the ChipVI project after the recent refactoring.

## Workflow

The new workflow is designed to be configuration-driven and modular, separating data pre-processing, experiment execution, and data downloading into distinct steps.

### 1. Data Downloading

To download new data from public repositories, use the `scripts/download_data.py` script. This script is a command-line tool that takes a file with a list of URLs to download.

**Usage:**

```bash
python scripts/download_data.py --url_list_fpath /path/to/your/url_list.txt
```

This script will download the files and place them in the appropriate directories under `data/`.

### 2. Data Pre-processing

Before training a model, the raw data must be pre-processed into a format suitable for the `Dataset` classes. This is handled by the `scripts/preprocess.py` script. The pre-processing pipeline is configured via a YAML file in the `configs/data/` directory.

**Configuration:**

Create or modify a YAML file (e.g., `configs/data/h3k27me3_preprocess.yaml`) to specify the input files, aggregation factors, and output paths for the pre-processed data.

**Usage:**

```bash
python scripts/preprocess.py --config-name <your_preprocessing_config_name>
```

For example, to run the H3K27me3 pre-processing:

```bash
python scripts/preprocess.py --config-name h3k27me3_preprocess
```

This will generate `.npy` files in the `data/processed` directory.

### 3. Running an Experiment

Experiments are defined and run using Hydra. An experiment is composed of a data configuration, a model configuration, and training parameters, all specified in a single experiment configuration file.

**Configuration:**

1.  **Data Config:** Define your dataset parameters in a file under `configs/data/` (e.g., `h3k27me3_200bp.yaml`).
2.  **Model Config:** Define your model architecture in a file under `configs/model/` (e.g., `nb_mu_r_small.yaml`).
3.  **Experiment Config:** Create an experiment file under `configs/experiment/` (e.g., `exp_001_concordance.yaml`) that composes the data, model, and training configurations.

**Usage:**

To run an experiment, use the `scripts/run.py` script and specify the experiment configuration.

```bash
python scripts/run.py experiment=<your_experiment_name>
```

For example, to run the `exp_001_concordance` experiment:

```bash
python scripts/run.py experiment=exp_001_concordance
```

Hydra will automatically manage the output directory, saving logs and model checkpoints under the `outputs/` directory.
