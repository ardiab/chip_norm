# ChipVI Documentation

ChipVI is a project for normalizing ChIP-seq signals to minimize noise from technical covariates. It uses a deep learning approach to model technical noise and separate it from the true biological signal.

## Project Structure

The project is organized into the following main directories:

- `chipvi/`: The core Python package.
  - `models/`: Contains the PyTorch models for the technical noise and biological signal.
  - `training/`: Includes scripts for training the models.
  - `data/`: Provides tools for creating and managing datasets.
  - `utils/`: Contains helper functions for distributions, path management, and plotting.
- `docs/`: This documentation.
- `notebooks/`: Jupyter notebooks for analysis and exploration.
- `runs/`: Scripts for running training experiments.
- `tests/`: Unit tests for the project.

## Core Concepts

The key idea behind ChipVI is to model the observed ChIP-seq signal as a combination of technical noise and biological signal.

- **Technical Noise**: This is the component of the signal that arises from the experimental process itself, such as sequencing depth, GC content, and mappability. ChipVI models this using either a Poisson or a Negative Binomial distribution.
- **Biological Signal**: This is the true signal of interest, which reflects the binding of the target protein to the DNA. ChipVI also models this using a Negative Binomial distribution.

By modeling these two components separately, ChipVI can learn to distinguish between them and provide a normalized signal that is less affected by technical artifacts.

## Further Reading

- [Models](./models.md)
- [Data](./data.md)
- [Training](./training.md)
- [Utils](./utils.md)
