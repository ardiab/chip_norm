# Data

The `chipvi.data` module is responsible for loading, processing, and preparing the data for training.

## `datasets.py`

This file contains the core data loading and processing logic.

### `build_datasets`

This function is the main entry point for creating the datasets. It takes a list of BED file paths and splits them into training and validation sets based on the provided chromosome lists. It can handle both single-end and replicate experiments.

### `SingleReplicateDataset`

This is a PyTorch `Dataset` class for single-end experiments. It takes as input the control reads, control MAPQ, control sequencing depth, experiment reads, experiment MAPQ, and experiment sequencing depth. It returns a tuple of covariates and experiment reads for each genomic bin.

### `MultiReplicateDataset`

This is a PyTorch `Dataset` class for replicate experiments. It takes the same inputs as `SingleReplicateDataset`, but for two replicates. It also includes a regularization term to encourage consistency between the biological signals of the two replicates.

## `experiment_collections.py`

This file provides a convenient way to manage and access different collections of experiments.

### `CTCFSmallCollection` and `H3K27me3SmallCollection`

These classes provide methods for getting the accessions of the alignments and controls for specific collections of CTCF and H3K27me3 experiments. They read a metadata file (`entex_proc_meta_info.pkl`) to get this information.

### `get_alignment_control_pairs`

This method returns a list of tuples, where each tuple contains the accession of an experiment and its corresponding control. This is useful for setting up the data for training.
