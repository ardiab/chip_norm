"""Data preprocessing utilities for ChipVI."""

import gc
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, TYPE_CHECKING
from omegaconf import DictConfig

if TYPE_CHECKING:
    from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SD_SCALING_FACTOR = 100_000_000


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


def run_preprocessing(cfg: DictConfig):
    """Run the complete data preprocessing pipeline."""
    from chipvi.utils.path_helper import PathHelper
    
    # Instantiate PathHelper to resolve paths
    paths = PathHelper(cfg)
    processed_data_dir = paths.proc_data_dir
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    train_chroms = cfg.train_chroms
    val_chroms = cfg.val_chroms
    agg_factor = cfg.aggregation_factor
    
    # Initialize data containers dynamically based on available replicates
    # We'll detect which replicates are available from the first group
    first_group = cfg.replicate_groups[0]
    available_replicates = [key for key in first_group.keys() if key.startswith('r')]
    
    # First pass: calculate total sizes needed for pre-allocation
    logger.info("First pass: calculating array sizes...")
    train_sizes = {f"replicate_{i+1}": 0 for i, _ in enumerate(available_replicates)}
    val_sizes = {f"replicate_{i+1}": 0 for i, _ in enumerate(available_replicates)}
    
    for grp_idx, group in enumerate(cfg.replicate_groups):
        for rep_idx, rep_key in enumerate(available_replicates):
            if rep_key not in group:
                continue
            
            rep_data = group[rep_key]
            # Load and check sizes
            exp_train_df, exp_val_df, _ = load_and_process_bed(
                Path(rep_data.exp),
                train_chroms,
                val_chroms,
                agg_factor,
            )
            
            # Count non-zero entries
            nonzero_train = (exp_train_df["reads"] > 0).sum()
            nonzero_val = (exp_val_df["reads"] > 0).sum()
            
            rep_key_name = f"replicate_{rep_idx + 1}"
            train_sizes[rep_key_name] += nonzero_train
            val_sizes[rep_key_name] += nonzero_val
            
            del exp_train_df, exp_val_df
            gc.collect()
    
    # Pre-allocate arrays based on calculated sizes
    logger.info("Allocating arrays...")
    train_ctrl_reads = {}
    val_ctrl_reads = {}
    train_ctrl_mapq = {}
    val_ctrl_mapq = {}
    train_ctrl_seq_depths = {}
    val_ctrl_seq_depths = {}
    train_exp_reads = {}
    val_exp_reads = {}
    train_exp_mapq = {}
    val_exp_mapq = {}
    train_exp_seq_depths = {}
    val_exp_seq_depths = {}
    
    for rep_idx, _ in enumerate(available_replicates):
        rep_key_name = f"replicate_{rep_idx + 1}"
        train_size = train_sizes[rep_key_name]
        val_size = val_sizes[rep_key_name]
        
        # Pre-allocate training arrays
        train_ctrl_reads[rep_key_name] = np.zeros(train_size, dtype=np.float32)
        train_ctrl_mapq[rep_key_name] = np.zeros(train_size, dtype=np.float32)
        train_ctrl_seq_depths[rep_key_name] = np.zeros(train_size, dtype=np.float32)
        train_exp_reads[rep_key_name] = np.zeros(train_size, dtype=np.float32)
        train_exp_mapq[rep_key_name] = np.zeros(train_size, dtype=np.float32)
        train_exp_seq_depths[rep_key_name] = np.zeros(train_size, dtype=np.float32)
        
        # Pre-allocate validation arrays
        val_ctrl_reads[rep_key_name] = np.zeros(val_size, dtype=np.float32)
        val_ctrl_mapq[rep_key_name] = np.zeros(val_size, dtype=np.float32)
        val_ctrl_seq_depths[rep_key_name] = np.zeros(val_size, dtype=np.float32)
        val_exp_reads[rep_key_name] = np.zeros(val_size, dtype=np.float32)
        val_exp_mapq[rep_key_name] = np.zeros(val_size, dtype=np.float32)
        val_exp_seq_depths[rep_key_name] = np.zeros(val_size, dtype=np.float32)
    
    # Pre-allocate group indices if needed
    if len(available_replicates) > 1:
        train_grp_idxs = np.zeros(train_sizes["replicate_2"], dtype=np.int32)
        val_grp_idxs = np.zeros(val_sizes["replicate_2"], dtype=np.int32)
    else:
        train_grp_idxs = None
        val_grp_idxs = None
    
    # Track current position in each array
    train_positions = {f"replicate_{i+1}": 0 for i, _ in enumerate(available_replicates)}
    val_positions = {f"replicate_{i+1}": 0 for i, _ in enumerate(available_replicates)}
    
    # Second pass: fill pre-allocated arrays
    logger.info("Second pass: processing and filling arrays...")
    for grp_idx, group in enumerate(cfg.replicate_groups):
        logger.info(
            "Processing replicate group %d/%d",
            grp_idx + 1,
            len(cfg.replicate_groups),
        )
        
        # Process each replicate in the group
        replicate_data = {}
        
        for rep_idx, rep_key in enumerate(available_replicates):
            if rep_key not in group:
                continue
                
            rep_data = group[rep_key]
            rep_num = rep_idx + 1
            logger.info(
                "Reading Replicate %d data from exp %s and ctrl %s",
                rep_num,
                rep_data.exp,
                rep_data.ctrl,
            )
            
            # Load and process bed files
            exp_train_df, exp_val_df, exp_seq_depth = load_and_process_bed(
                Path(rep_data.exp),
                train_chroms,
                val_chroms,
                agg_factor,
            )
            ctrl_train_df, ctrl_val_df, ctrl_seq_depth = load_and_process_bed(
                Path(rep_data.ctrl),
                train_chroms,
                val_chroms,
                agg_factor,
            )
            
            logger.info("%s (exp) sequencing depth: %s", rep_data.exp, exp_seq_depth)
            logger.info("%s (ctrl) sequencing depth: %s", rep_data.ctrl, ctrl_seq_depth)
            
            # Apply independent nonzero mask for this replicate
            nonzero_mask_train = exp_train_df["reads"] > 0
            nonzero_mask_val = exp_val_df["reads"] > 0
            
            # Filter data and scale sequencing depths
            exp_train_df_filtered = exp_train_df[nonzero_mask_train]
            exp_val_df_filtered = exp_val_df[nonzero_mask_val]
            exp_seq_depth_scaled = exp_seq_depth / SD_SCALING_FACTOR
            ctrl_train_df_filtered = ctrl_train_df[nonzero_mask_train]
            ctrl_val_df_filtered = ctrl_val_df[nonzero_mask_val]
            ctrl_seq_depth_scaled = ctrl_seq_depth / SD_SCALING_FACTOR
            
            # Store processed data for this replicate
            replicate_data[f"replicate_{rep_num}"] = {
                "exp_train_df_filtered": exp_train_df_filtered,
                "exp_val_df_filtered": exp_val_df_filtered,
                "ctrl_train_df_filtered": ctrl_train_df_filtered,
                "ctrl_val_df_filtered": ctrl_val_df_filtered,
                "exp_seq_depth_scaled": exp_seq_depth_scaled,
                "ctrl_seq_depth_scaled": ctrl_seq_depth_scaled,
            }
            
            # Fill pre-allocated arrays
            rep_key_name = f"replicate_{rep_num}"
            
            # Get current positions
            train_pos = train_positions[rep_key_name]
            val_pos = val_positions[rep_key_name]
            
            # Calculate slice lengths
            train_len = len(exp_train_df_filtered)
            val_len = len(exp_val_df_filtered)
            
            # Fill training arrays
            train_ctrl_reads[rep_key_name][train_pos:train_pos + train_len] = ctrl_train_df_filtered["reads"].values
            train_ctrl_mapq[rep_key_name][train_pos:train_pos + train_len] = ctrl_train_df_filtered["mapq"].values
            train_ctrl_seq_depths[rep_key_name][train_pos:train_pos + train_len] = ctrl_seq_depth_scaled
            train_exp_reads[rep_key_name][train_pos:train_pos + train_len] = exp_train_df_filtered["reads"].values
            train_exp_mapq[rep_key_name][train_pos:train_pos + train_len] = exp_train_df_filtered["mapq"].values
            train_exp_seq_depths[rep_key_name][train_pos:train_pos + train_len] = exp_seq_depth_scaled
            
            # Fill validation arrays
            val_ctrl_reads[rep_key_name][val_pos:val_pos + val_len] = ctrl_val_df_filtered["reads"].values
            val_ctrl_mapq[rep_key_name][val_pos:val_pos + val_len] = ctrl_val_df_filtered["mapq"].values
            val_ctrl_seq_depths[rep_key_name][val_pos:val_pos + val_len] = ctrl_seq_depth_scaled
            val_exp_reads[rep_key_name][val_pos:val_pos + val_len] = exp_val_df_filtered["reads"].values
            val_exp_mapq[rep_key_name][val_pos:val_pos + val_len] = exp_val_df_filtered["mapq"].values
            val_exp_seq_depths[rep_key_name][val_pos:val_pos + val_len] = exp_seq_depth_scaled
            
            # Fill group indices for replicate 2 and beyond
            if rep_num > 1 and train_grp_idxs is not None:
                train_grp_idxs[train_pos:train_pos + train_len] = grp_idx
                val_grp_idxs[val_pos:val_pos + val_len] = grp_idx
            
            # Update positions
            train_positions[rep_key_name] += train_len
            val_positions[rep_key_name] += val_len
            
            # Clean up memory for this replicate
            del exp_train_df, exp_val_df, ctrl_train_df, ctrl_val_df
        
        # Clean up replicate_data
        del replicate_data
        gc.collect()
    
    # Save processed arrays to disk
    train_prefix = cfg.output_prefix_train
    val_prefix = cfg.output_prefix_val
    
    # Save data for each replicate
    for rep_idx, rep_key in enumerate(available_replicates):
        rep_num = rep_idx + 1
        rep_key_name = f"replicate_{rep_num}"
        rep_suffix = f"r{rep_num}"
        
        # Save training data (arrays are already numpy arrays)
        np.save(processed_data_dir / f"{train_prefix}_control_reads_{rep_suffix}.npy", 
                train_ctrl_reads[rep_key_name])
        np.save(processed_data_dir / f"{train_prefix}_control_mapq_{rep_suffix}.npy", 
                train_ctrl_mapq[rep_key_name])
        np.save(processed_data_dir / f"{train_prefix}_control_seq_depth_{rep_suffix}.npy", 
                train_ctrl_seq_depths[rep_key_name])
        np.save(processed_data_dir / f"{train_prefix}_experiment_reads_{rep_suffix}.npy", 
                train_exp_reads[rep_key_name])
        np.save(processed_data_dir / f"{train_prefix}_experiment_mapq_{rep_suffix}.npy", 
                train_exp_mapq[rep_key_name])
        np.save(processed_data_dir / f"{train_prefix}_experiment_seq_depth_{rep_suffix}.npy", 
                train_exp_seq_depths[rep_key_name])
        
        # Save validation data (arrays are already numpy arrays)
        np.save(processed_data_dir / f"{val_prefix}_control_reads_{rep_suffix}.npy", 
                val_ctrl_reads[rep_key_name])
        np.save(processed_data_dir / f"{val_prefix}_control_mapq_{rep_suffix}.npy", 
                val_ctrl_mapq[rep_key_name])
        np.save(processed_data_dir / f"{val_prefix}_control_seq_depth_{rep_suffix}.npy", 
                val_ctrl_seq_depths[rep_key_name])
        np.save(processed_data_dir / f"{val_prefix}_experiment_reads_{rep_suffix}.npy", 
                val_exp_reads[rep_key_name])
        np.save(processed_data_dir / f"{val_prefix}_experiment_mapq_{rep_suffix}.npy", 
                val_exp_mapq[rep_key_name])
        np.save(processed_data_dir / f"{val_prefix}_experiment_seq_depth_{rep_suffix}.npy", 
                val_exp_seq_depths[rep_key_name])
    
    # Save group indices if we have multiple replicates
    if len(available_replicates) > 1 and train_grp_idxs is not None:
        np.save(processed_data_dir / f"{train_prefix}_grp_idxs.npy", train_grp_idxs)
        np.save(processed_data_dir / f"{val_prefix}_grp_idxs.npy", val_grp_idxs)
    
    logger.info("Preprocessing completed. Files saved to %s", processed_data_dir)


def load_and_process_bed(bed_fpath, train_chroms, val_chroms, agg_factor):
    """Load and process a BED file with aggregation."""
    columns = ["chr", "start", "end", "reads", "mapq"]
    
    df = pd.read_csv(bed_fpath, sep="\t", header=None, names=columns)
    if not set(train_chroms).issubset(set(df["chr"])):
        raise ValueError(f"Some chromosomes in train_chroms {train_chroms} are missing from {bed_fpath}.")
    if not set(val_chroms).issubset(set(df["chr"])):
        raise ValueError(f"Some chromosomes in val_chroms {val_chroms} are missing from {bed_fpath}.")
    df["reads"] = df["reads"].astype(np.float32)
    if df["reads"].isna().any():  # type: ignore
        raise ValueError(f"{bed_fpath} reads contain NaN values.")
    
    seq_depth = df["reads"].sum()
    
    train_df = df[df["chr"].isin(train_chroms)].copy()
    val_df = df[df["chr"].isin(val_chroms)].copy()
    
    # Convert MAPQ to probability
    train_df["mapq"] = 1 - 10**(train_df["mapq"].replace("NAN", 100).astype(np.float32) / -10)
    val_df["mapq"] = 1 - 10**(val_df["mapq"].replace("NAN", 100).astype(np.float32) / -10)
    
    # Apply aggregation if factor > 1
    if agg_factor > 1:
        train_df = _aggregate_dataframe_bins(train_df, agg_factor)
        val_df = _aggregate_dataframe_bins(val_df, agg_factor)
    
    return train_df, val_df, seq_depth


def _aggregate_dataframe_bins(df, agg_factor):
    """Aggregate bins in a dataframe by summing reads and averaging mapq."""
    # Sort by chromosome and start position
    df_sorted = df.sort_values(['chr', 'start']).copy()
    
    # Process each chromosome separately
    aggregated_dfs = []
    for chrom in df_sorted['chr'].unique():
        chrom_df = df_sorted[df_sorted['chr'] == chrom]
        
        # Calculate the number of complete bins
        n_rows = len(chrom_df)
        n_complete_bins = n_rows // agg_factor
        n_aggregated = n_complete_bins
        
        if n_aggregated == 0:
            continue
        
        # Trim to complete bins
        trimmed_df = chrom_df.iloc[:n_complete_bins * agg_factor]
        
        # Create aggregation indices
        agg_indices = np.repeat(np.arange(n_aggregated), agg_factor)
        
        # Aggregate using groupby on the indices
        aggregated = trimmed_df.groupby(agg_indices).agg({
            'chr': 'first',
            'start': 'first',
            'end': 'last',
            'reads': 'sum',
            'mapq': 'mean'
        }).reset_index(drop=True)
        
        aggregated_dfs.append(aggregated)
    
    # Combine all chromosomes
    if aggregated_dfs:
        result = pd.concat(aggregated_dfs, ignore_index=True)
    else:
        result = pd.DataFrame(columns=['chr', 'start', 'end', 'reads', 'mapq'])
    
    return result