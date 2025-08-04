# Models

The `chipvi.models` module contains the PyTorch models used in the ChipVI project. These models are designed to capture the technical noise and biological signal in ChIP-seq data.

## `common.MLP`

This is a simple Multi-Layer Perceptron (MLP) class that serves as a building block for the other models. It's a standard feed-forward neural network with configurable hidden layers and activation functions.

## Technical Noise Models

These models are responsible for learning the relationship between the technical covariates and the observed signal.

### `technical_model.TechNB_mu_r`

This model uses a Negative Binomial (NB) distribution to model the technical noise. It has two main components:

- A network `f` that predicts the log of the mean (`log_mu_tech`) of the NB distribution from the input covariates.
- A network `g` that predicts the log of the dispersion parameter (`log_r_tech`) of the NB distribution from the input covariates and the predicted `log_mu_tech`.

This model is more flexible than the Poisson model and can capture a wider range of noise profiles.

### `technical_model.TechNB_r_p`

This is an alternative parameterization of the NB technical model. Instead of predicting `mu` and `r`, it predicts `r` (the dispersion) and `p` (the success probability).

### `technical_model.TechPoisson`

This is a simpler technical noise model that uses a Poisson distribution. It has a single network `f` that predicts the log of the mean (`log_mu_tech`) of the Poisson distribution from the input covariates. This model is less flexible than the NB model but can be useful in situations where the noise is well-behaved.

## Biological Signal Model

### `technical_model.BiolNB`

This model is responsible for learning the biological signal. It uses a Negative Binomial distribution and has a single network `h` that predicts the log of the mean (`log_mu_biol`) of the NB distribution from a latent representation of the biological state.
