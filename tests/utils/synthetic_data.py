"""Synthetic data generation utilities for testing ChipVI training pipeline."""

import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Optional
from contextlib import contextmanager


def generate_single_replicate_data(
    prefix: str,
    n_samples: int = 100,
    covariate_dim: int = 5,
    random_seed: Optional[int] = 42
) -> None:
    """Generate single replicate synthetic dataset files in .npy format.
    
    Args:
        prefix: File prefix for the generated data files
        n_samples: Number of samples to generate
        covariate_dim: Number of covariates (should be 5 for compatibility)
        random_seed: Random seed for reproducibility
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate realistic read count data (positive integers as float32)
    control_reads = np.random.poisson(lam=50, size=n_samples).astype(np.float32)
    experiment_reads = np.random.poisson(lam=100, size=n_samples).astype(np.float32)
    
    # Generate MAPQ scores (0-60 range, typical for sequencing)
    control_mapq = np.random.uniform(20, 60, size=n_samples).astype(np.float32)
    experiment_mapq = np.random.uniform(20, 60, size=n_samples).astype(np.float32)
    
    # Generate sequence depth values (millions of reads)
    control_seq_depth = np.random.uniform(10_000_000, 50_000_000, size=n_samples).astype(np.float32)
    experiment_seq_depth = np.random.uniform(10_000_000, 50_000_000, size=n_samples).astype(np.float32)
    
    # Save arrays with expected naming convention
    np.save(f"{prefix}_control_reads_r1.npy", control_reads)
    np.save(f"{prefix}_control_mapq_r1.npy", control_mapq)
    np.save(f"{prefix}_control_seq_depth_r1.npy", control_seq_depth)
    np.save(f"{prefix}_experiment_reads_r1.npy", experiment_reads)
    np.save(f"{prefix}_experiment_mapq_r1.npy", experiment_mapq)
    np.save(f"{prefix}_experiment_seq_depth_r1.npy", experiment_seq_depth)


def generate_multi_replicate_data(
    prefix: str,
    n_samples: int = 100, 
    covariate_dim: int = 5,
    random_seed: Optional[int] = 42
) -> None:
    """Generate multi-replicate synthetic dataset files in .npy format.
    
    Args:
        prefix: File prefix for the generated data files  
        n_samples: Number of samples to generate
        covariate_dim: Number of covariates (should be 5 for compatibility)
        random_seed: Random seed for reproducibility
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate replicate 1 data
    control_reads_r1 = np.random.poisson(lam=50, size=n_samples).astype(np.float32)
    control_mapq_r1 = np.random.uniform(20, 60, size=n_samples).astype(np.float32)
    control_seq_depth_r1 = np.random.uniform(10_000_000, 50_000_000, size=n_samples).astype(np.float32)
    experiment_reads_r1 = np.random.poisson(lam=100, size=n_samples).astype(np.float32)
    experiment_mapq_r1 = np.random.uniform(20, 60, size=n_samples).astype(np.float32)
    experiment_seq_depth_r1 = np.random.uniform(10_000_000, 50_000_000, size=n_samples).astype(np.float32)
    
    # Generate replicate 2 data (correlated but with some noise)
    # Keep reads as integers by rounding after scaling
    control_reads_r2 = np.round(control_reads_r1 * np.random.uniform(0.8, 1.2, size=n_samples)).astype(np.float32)
    control_mapq_r2 = np.random.uniform(20, 60, size=n_samples).astype(np.float32)  
    control_seq_depth_r2 = np.random.uniform(10_000_000, 50_000_000, size=n_samples).astype(np.float32)
    experiment_reads_r2 = np.round(experiment_reads_r1 * np.random.uniform(0.8, 1.2, size=n_samples)).astype(np.float32)
    experiment_mapq_r2 = np.random.uniform(20, 60, size=n_samples).astype(np.float32)
    experiment_seq_depth_r2 = np.random.uniform(10_000_000, 50_000_000, size=n_samples).astype(np.float32)
    
    # Generate group indices (int32)
    n_groups = max(1, n_samples // 10)  # Create groups of ~10 samples each
    grp_idxs = np.random.randint(0, n_groups, size=n_samples).astype(np.int32)
    
    # Save replicate 1 files
    np.save(f"{prefix}_control_reads_r1.npy", control_reads_r1)
    np.save(f"{prefix}_control_mapq_r1.npy", control_mapq_r1)
    np.save(f"{prefix}_control_seq_depth_r1.npy", control_seq_depth_r1)
    np.save(f"{prefix}_experiment_reads_r1.npy", experiment_reads_r1)
    np.save(f"{prefix}_experiment_mapq_r1.npy", experiment_mapq_r1)
    np.save(f"{prefix}_experiment_seq_depth_r1.npy", experiment_seq_depth_r1)
    
    # Save replicate 2 files
    np.save(f"{prefix}_control_reads_r2.npy", control_reads_r2)
    np.save(f"{prefix}_control_mapq_r2.npy", control_mapq_r2)
    np.save(f"{prefix}_control_seq_depth_r2.npy", control_seq_depth_r2)
    np.save(f"{prefix}_experiment_reads_r2.npy", experiment_reads_r2)
    np.save(f"{prefix}_experiment_mapq_r2.npy", experiment_mapq_r2)
    np.save(f"{prefix}_experiment_seq_depth_r2.npy", experiment_seq_depth_r2)
    
    # Save group indices
    np.save(f"{prefix}_grp_idxs.npy", grp_idxs)


def validate_data_dimensions(prefix: str, is_multi_replicate: bool = False) -> Tuple[bool, str]:
    """Validate that all data files have consistent dimensions.
    
    Args:
        prefix: File prefix for the data files to validate
        is_multi_replicate: Whether to validate multi-replicate format
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if is_multi_replicate:
            # Check replicate 1 files
            r1_files = [
                f"{prefix}_control_reads_r1.npy",
                f"{prefix}_control_mapq_r1.npy", 
                f"{prefix}_control_seq_depth_r1.npy",
                f"{prefix}_experiment_reads_r1.npy",
                f"{prefix}_experiment_mapq_r1.npy",
                f"{prefix}_experiment_seq_depth_r1.npy"
            ]
            
            # Check replicate 2 files
            r2_files = [
                f"{prefix}_control_reads_r2.npy",
                f"{prefix}_control_mapq_r2.npy",
                f"{prefix}_control_seq_depth_r2.npy", 
                f"{prefix}_experiment_reads_r2.npy",
                f"{prefix}_experiment_mapq_r2.npy",
                f"{prefix}_experiment_seq_depth_r2.npy"
            ]
            
            grp_file = f"{prefix}_grp_idxs.npy"
            all_files = r1_files + r2_files + [grp_file]
            
        else:
            # Single replicate files
            all_files = [
                f"{prefix}_control_reads_r1.npy",
                f"{prefix}_control_mapq_r1.npy",
                f"{prefix}_control_seq_depth_r1.npy",
                f"{prefix}_experiment_reads_r1.npy", 
                f"{prefix}_experiment_mapq_r1.npy",
                f"{prefix}_experiment_seq_depth_r1.npy"
            ]
        
        # Check all files exist
        for file_path in all_files:
            if not Path(file_path).exists():
                return False, f"Missing file: {file_path}"
        
        # Load arrays and check dimensions
        shapes = []
        for file_path in all_files:
            array = np.load(file_path)
            shapes.append(array.shape)
            
            # Check that arrays are 1-dimensional
            if len(array.shape) != 1:
                return False, f"File {file_path} has shape {array.shape}, expected 1D array"
        
        # Check all arrays have same length
        first_shape = shapes[0]
        for i, shape in enumerate(shapes):
            if shape != first_shape:
                return False, f"Inconsistent dimensions: file {all_files[0]} has shape {first_shape}, but file {all_files[i]} has shape {shape}"
        
        return True, "All dimensions are consistent"
        
    except Exception as e:
        return False, f"Error during validation: {str(e)}"


class SyntheticDataManager:
    """Context manager for creating and cleaning up synthetic data files."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize synthetic data manager.
        
        Args:
            base_dir: Base directory for temporary files. If None, uses system temp dir.
        """
        self.base_dir = base_dir
        self.temp_dirs = []
        self.created_files = []
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up all created files and directories."""
        # Remove individual files
        for file_path in self.created_files:
            try:
                if Path(file_path).exists():
                    Path(file_path).unlink()
            except Exception:
                pass  # Ignore cleanup errors
                
        # Remove temporary directories
        for temp_dir in self.temp_dirs:
            try:
                if Path(temp_dir).exists():
                    shutil.rmtree(temp_dir)
            except Exception:
                pass  # Ignore cleanup errors
                
    def create_single_replicate_data(
        self,
        n_samples: int = 100,
        covariate_dim: int = 5,
        random_seed: Optional[int] = 42
    ) -> str:
        """Create single replicate data and track files for cleanup.
        
        Returns:
            File prefix for the generated data files
        """
        if self.base_dir:
            # Create subdirectory in base_dir
            temp_dir = Path(self.base_dir) / f"synthetic_data_{len(self.temp_dirs)}"
            temp_dir.mkdir(exist_ok=True)
            self.temp_dirs.append(str(temp_dir))
            prefix = temp_dir / "data"
        else:
            # Use system temp directory
            temp_dir = Path(tempfile.mkdtemp(prefix="synthetic_data_"))
            self.temp_dirs.append(str(temp_dir))
            prefix = temp_dir / "data"
            
        generate_single_replicate_data(
            prefix=str(prefix),
            n_samples=n_samples,
            covariate_dim=covariate_dim,
            random_seed=random_seed
        )
        
        # Track the created files for cleanup
        file_suffixes = [
            "_control_reads_r1.npy",
            "_control_mapq_r1.npy",
            "_control_seq_depth_r1.npy", 
            "_experiment_reads_r1.npy",
            "_experiment_mapq_r1.npy",
            "_experiment_seq_depth_r1.npy"
        ]
        
        for suffix in file_suffixes:
            self.created_files.append(f"{prefix}{suffix}")
            
        return str(prefix)
        
    def create_multi_replicate_data(
        self,
        n_samples: int = 100,
        covariate_dim: int = 5,
        random_seed: Optional[int] = 42
    ) -> str:
        """Create multi-replicate data and track files for cleanup.
        
        Returns:
            File prefix for the generated data files
        """
        if self.base_dir:
            # Create subdirectory in base_dir
            temp_dir = Path(self.base_dir) / f"synthetic_data_{len(self.temp_dirs)}"
            temp_dir.mkdir(exist_ok=True)
            self.temp_dirs.append(str(temp_dir))
            prefix = temp_dir / "data"
        else:
            # Use system temp directory
            temp_dir = Path(tempfile.mkdtemp(prefix="synthetic_data_"))
            self.temp_dirs.append(str(temp_dir))
            prefix = temp_dir / "data"
            
        generate_multi_replicate_data(
            prefix=str(prefix),
            n_samples=n_samples,
            covariate_dim=covariate_dim,
            random_seed=random_seed
        )
        
        # Track the created files for cleanup
        file_suffixes = [
            "_control_reads_r1.npy", "_control_mapq_r1.npy", "_control_seq_depth_r1.npy",
            "_experiment_reads_r1.npy", "_experiment_mapq_r1.npy", "_experiment_seq_depth_r1.npy",
            "_control_reads_r2.npy", "_control_mapq_r2.npy", "_control_seq_depth_r2.npy",
            "_experiment_reads_r2.npy", "_experiment_mapq_r2.npy", "_experiment_seq_depth_r2.npy",
            "_grp_idxs.npy"
        ]
        
        for suffix in file_suffixes:
            self.created_files.append(f"{prefix}{suffix}")
            
        return str(prefix)