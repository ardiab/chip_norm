"""Data preprocessing utilities for ChipVI."""

import numpy as np
from typing import Literal


def aggregate_bins(
    data_array: np.ndarray,
    agg_factor: int,
    agg_method: Literal["sum", "mean"],
) -> np.ndarray:
    """Aggregate bins by summing or averaging values.
    
    Args:
        data_array: Input array to aggregate
        agg_factor: Factor by which to aggregate (e.g., 8 to aggregate 8 bins into 1)
        agg_method: Method to use for aggregation ("sum" or "mean")
        
    Returns:
        Aggregated array with length reduced by agg_factor
    """
    # Trim the array to be divisible by agg_factor
    old_shape = data_array.shape
    new_shape = old_shape[0] - old_shape[0] % agg_factor
    trimmed_array = data_array[:new_shape]
    
    # Reshape and aggregate
    if agg_method == "sum":
        return trimmed_array.reshape(-1, agg_factor).sum(axis=1)
    elif agg_method == "mean":
        return trimmed_array.reshape(-1, agg_factor).mean(axis=1)
    else:
        raise ValueError(f"Invalid agg_method: {agg_method}. Must be 'sum' or 'mean'")