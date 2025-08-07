#!/usr/bin/env python3
"""
Legacy Data Converter

Converts legacy pickle data from {target}_data_v2 directories to the new .npy file format 
expected by MultiReplicateDataset. Preserves data integrity, applies optional log transforms 
to specific covariate columns, and maintains the same train/val splits.
"""

import argparse
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_legacy_pickle_data(legacy_data_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load legacy pickle data from directory, skipping sd_map files.
    
    Args:
        legacy_data_dir: Directory containing pickle files
        
    Returns:
        Dictionary with 'train' and 'val' splits, each containing column arrays
    """
    logger.info(f"Loading legacy data from {legacy_data_dir}")
    
    data = {"train": {}, "val": {}}
    
    for pkl_path in legacy_data_dir.glob("*.pkl"):
        if "sd_map" in pkl_path.stem:
            logger.info(f"Skipping sd_map file: {pkl_path.stem}")
            continue
            
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
            
        if "train" in pkl_path.stem:
            column_name = pkl_path.stem.replace("train_", "")
            data["train"][column_name] = np.array(pkl_data, dtype=np.float32)
            logger.info(f"Loaded {pkl_path.stem} as train column '{column_name}' with shape {data['train'][column_name].shape}")
            
        elif "val" in pkl_path.stem:
            column_name = pkl_path.stem.replace("val_", "")
            data["val"][column_name] = np.array(pkl_data, dtype=np.float32)
            logger.info(f"Loaded {pkl_path.stem} as val column '{column_name}' with shape {data['val'][column_name].shape}")
            
        else:
            logger.info(f"Skipping unrecognized file: {pkl_path.stem}")
            
    return data


def apply_log_transforms(data: Dict[str, Dict[str, np.ndarray]], log_transform: bool) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Apply log transforms to covariate columns 0 and 5 if specified.
    
    Covariate column 0 = control_reads_r1 (for both r1 and r2)
    Covariate column 5 = control_reads_r2 (which is covariate 0 for r2)
    
    Args:
        data: Data dictionary with train/val splits
        log_transform: Whether to apply log transforms
        
    Returns:
        Data with log transforms applied if specified
    """
    if not log_transform:
        logger.info("No log transform requested")
        return data
        
    logger.info("Applying log transforms to covariate columns 0 and 5 (control reads)")
    
    # Apply log(x + 1) transform to control_reads_r1 and control_reads_r2
    for split in ["train", "val"]:
        if "control_reads_r1" in data[split]:
            original_vals = data[split]["control_reads_r1"].copy()
            data[split]["control_reads_r1"] = np.log(data[split]["control_reads_r1"] + 1)
            logger.info(f"Applied log transform to {split} control_reads_r1: {original_vals[:3]} -> {data[split]['control_reads_r1'][:3]}")
            
        if "control_reads_r2" in data[split]:
            original_vals = data[split]["control_reads_r2"].copy()  
            data[split]["control_reads_r2"] = np.log(data[split]["control_reads_r2"] + 1)
            logger.info(f"Applied log transform to {split} control_reads_r2: {original_vals[:3]} -> {data[split]['control_reads_r2'][:3]}")
            
    return data


def create_group_indices(experiment_seq_depth_r1: np.ndarray) -> np.ndarray:
    """
    Create group indices based on unique sequencing depth values.
    
    Args:
        experiment_seq_depth_r1: Sequencing depth values for replicate 1
        
    Returns:
        Group indices array
    """
    unique_depths = np.unique(experiment_seq_depth_r1)
    grp_idxs = np.zeros(len(experiment_seq_depth_r1), dtype=np.int32)
    
    for idx, seq_depth in enumerate(unique_depths):
        grp_idxs[experiment_seq_depth_r1 == seq_depth] = idx
        
    logger.info(f"Created {len(unique_depths)} unique groups from sequencing depths")
    return grp_idxs


def convert_to_multireplicate_format(
    data: Dict[str, Dict[str, np.ndarray]], 
    output_dir: Path, 
    target: str
) -> None:
    """
    Convert legacy data structure to MultiReplicateDataset .npy format.
    
    Args:
        data: Legacy data dictionary with train/val splits
        output_dir: Output directory for .npy files  
        target: Target name for file prefix
    """
    logger.info("Converting data structure to MultiReplicateDataset format")
    
    # Required columns for MultiReplicateDataset
    required_columns = [
        'control_reads_r1', 'control_mapq_r1', 'control_seq_depth_r1',
        'experiment_reads_r1', 'experiment_mapq_r1', 'experiment_seq_depth_r1',
        'control_reads_r2', 'control_mapq_r2', 'control_seq_depth_r2', 
        'experiment_reads_r2', 'experiment_mapq_r2', 'experiment_seq_depth_r2'
    ]
    
    # Verify all required columns exist
    for split in ["train", "val"]:
        missing_columns = [col for col in required_columns if col not in data[split]]
        if missing_columns:
            raise ValueError(f"Missing columns in {split} data: {missing_columns}")
            
    # Combine train and val data
    combined_data = {}
    for column in required_columns:
        combined_data[column] = np.concatenate([
            data["train"][column],
            data["val"][column]
        ], axis=0)
        logger.info(f"Combined {column}: train={len(data['train'][column])}, val={len(data['val'][column])}, total={len(combined_data[column])}")
        
    # Create group indices
    grp_idxs = create_group_indices(combined_data['experiment_seq_depth_r1'])
    
    # Save all arrays to .npy files with proper naming convention
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / target
    
    for column in required_columns:
        output_path = f"{prefix}_{column}.npy"
        np.save(output_path, combined_data[column])
        logger.info(f"Saved {output_path} with shape {combined_data[column].shape}")
        
    # Save group indices
    grp_idxs_path = f"{prefix}_grp_idxs.npy"
    np.save(grp_idxs_path, grp_idxs)
    logger.info(f"Saved {grp_idxs_path} with shape {grp_idxs.shape}")
    
    logger.info(f"Conversion complete! Files saved with prefix: {prefix}")


def main():
    """Main entry point for the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert legacy pickle data to MultiReplicateDataset .npy format"
    )
    parser.add_argument(
        "--target", 
        required=True,
        help="Target name (e.g., 'h3k27me3', 'ctcf')"
    )
    parser.add_argument(
        "--legacy_data_dir",
        type=Path,
        help="Directory containing legacy pickle files (defaults to {target}_data_v2)"
    )
    parser.add_argument(
        "--output_directory",
        type=Path,
        required=True,
        help="Output directory for .npy files"
    )
    parser.add_argument(
        "--log_transform_inputs",
        action="store_true",
        help="Apply log transforms to covariate columns 0 and 5"
    )
    
    args = parser.parse_args()
    
    # Set default legacy data directory if not provided
    if args.legacy_data_dir is None:
        args.legacy_data_dir = Path(f"{args.target}_data_v2")
        
    # Validate inputs
    if not args.legacy_data_dir.exists():
        raise FileNotFoundError(f"Legacy data directory not found: {args.legacy_data_dir}")
        
    logger.info(f"Starting conversion for target '{args.target}'")
    logger.info(f"Legacy data directory: {args.legacy_data_dir}")
    logger.info(f"Output directory: {args.output_directory}")
    logger.info(f"Log transform inputs: {args.log_transform_inputs}")
    
    try:
        # Load legacy pickle data
        data = load_legacy_pickle_data(args.legacy_data_dir)
        
        # Apply log transforms if requested
        data = apply_log_transforms(data, args.log_transform_inputs)
        
        # Convert to MultiReplicateDataset format and save
        convert_to_multireplicate_format(data, args.output_directory, args.target)
        
        logger.info("Conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()