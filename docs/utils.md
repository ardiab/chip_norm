# Utils

The `chipvi.utils` module contains various helper functions used throughout the project.

## `distributions.py`

This file provides a set of functions for working with probability distributions, primarily from the `torch.distributions` module.

- **`get_torch_nb_dist`**: Creates a Negative Binomial distribution from different parameterizations.
- **`nb_log_prob_mu_r`**, **`nb_log_prob_r_p`**, **`poisson_log_prob`**: Calculate the log probability of a given value for different distributions.
- **`poisson_cdf`**, **`nb_cdf_r_p`**, **`compute_numeric_cdf`**: Compute the cumulative distribution function (CDF) for different distributions.

## `path_helper.py`

This class centralizes all the file paths used in the project. This makes the code more maintainable and easier to configure. It defines paths for data, models, and external scripts.

## `plots.py`

This file contains the `hist2d` function, which is used to create 2D histograms. This is useful for visualizing the relationship between two variables. The function also calculates and displays the Pearson and Spearman correlations on the plot.
