import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from tests.utils.synthetic_data import (
    generate_single_replicate_data,
    generate_multi_replicate_data,
    SyntheticDataManager,
    validate_data_dimensions
)


class TestSyntheticDataGenerator:
    """Test suite for synthetic data generation utilities."""
    
    def test_single_replicate_generation(self, tmp_path):
        """Test creation of valid single replicate dataset files."""
        prefix = tmp_path / "test_single"
        n_samples = 50
        
        generate_single_replicate_data(
            prefix=str(prefix),
            n_samples=n_samples,
            covariate_dim=5
        )
        
        # Check that all required files exist
        expected_files = [
            f"{prefix}_control_reads_r1.npy",
            f"{prefix}_control_mapq_r1.npy", 
            f"{prefix}_control_seq_depth_r1.npy",
            f"{prefix}_experiment_reads_r1.npy",
            f"{prefix}_experiment_mapq_r1.npy",
            f"{prefix}_experiment_seq_depth_r1.npy"
        ]
        
        for file_path in expected_files:
            assert Path(file_path).exists(), f"Expected file {file_path} does not exist"
            
        # Load and verify data
        control_reads = np.load(f"{prefix}_control_reads_r1.npy")
        experiment_reads = np.load(f"{prefix}_experiment_reads_r1.npy")
        
        assert control_reads.shape == (n_samples,), f"Expected shape {(n_samples,)}, got {control_reads.shape}"
        assert experiment_reads.shape == (n_samples,), f"Expected shape {(n_samples,)}, got {experiment_reads.shape}"
        assert control_reads.dtype == np.float32, f"Expected float32, got {control_reads.dtype}"
        assert experiment_reads.dtype == np.float32, f"Expected float32, got {experiment_reads.dtype}"
    
    def test_multi_replicate_generation(self, tmp_path):
        """Test creation of valid multi-replicate dataset files."""
        prefix = tmp_path / "test_multi"
        n_samples = 75
        
        generate_multi_replicate_data(
            prefix=str(prefix),
            n_samples=n_samples,
            covariate_dim=5
        )
        
        # Check that all required files exist (including r1, r2, and grp_idxs)
        expected_files = [
            f"{prefix}_control_reads_r1.npy",
            f"{prefix}_control_mapq_r1.npy",
            f"{prefix}_control_seq_depth_r1.npy", 
            f"{prefix}_experiment_reads_r1.npy",
            f"{prefix}_experiment_mapq_r1.npy",
            f"{prefix}_experiment_seq_depth_r1.npy",
            f"{prefix}_control_reads_r2.npy",
            f"{prefix}_control_mapq_r2.npy",
            f"{prefix}_control_seq_depth_r2.npy",
            f"{prefix}_experiment_reads_r2.npy", 
            f"{prefix}_experiment_mapq_r2.npy",
            f"{prefix}_experiment_seq_depth_r2.npy",
            f"{prefix}_grp_idxs.npy"
        ]
        
        for file_path in expected_files:
            assert Path(file_path).exists(), f"Expected file {file_path} does not exist"
            
        # Load and verify r1 and r2 data have same dimensions
        control_reads_r1 = np.load(f"{prefix}_control_reads_r1.npy")
        control_reads_r2 = np.load(f"{prefix}_control_reads_r2.npy")
        grp_idxs = np.load(f"{prefix}_grp_idxs.npy")
        
        assert control_reads_r1.shape == (n_samples,), f"Expected shape {(n_samples,)}, got {control_reads_r1.shape}"
        assert control_reads_r2.shape == (n_samples,), f"Expected shape {(n_samples,)}, got {control_reads_r2.shape}"
        assert grp_idxs.shape == (n_samples,), f"Expected shape {(n_samples,)}, got {grp_idxs.shape}"
        assert grp_idxs.dtype == np.int32, f"Expected int32, got {grp_idxs.dtype}"
    
    def test_dimension_consistency_validation(self, tmp_path):
        """Test that generated arrays have consistent dimensions across control/treatment/covariates."""
        prefix = tmp_path / "test_consistency"
        n_samples = 100
        
        generate_single_replicate_data(
            prefix=str(prefix),
            n_samples=n_samples,
            covariate_dim=5
        )
        
        # Validate dimensions are consistent
        is_valid, error_msg = validate_data_dimensions(str(prefix), is_multi_replicate=False)
        assert is_valid, f"Dimension validation failed: {error_msg}"
        
        # Load all arrays and verify shapes are consistent
        arrays = {}
        for suffix in ["control_reads_r1", "control_mapq_r1", "control_seq_depth_r1", 
                      "experiment_reads_r1", "experiment_mapq_r1", "experiment_seq_depth_r1"]:
            arrays[suffix] = np.load(f"{prefix}_{suffix}.npy")
            
        shapes = [arr.shape for arr in arrays.values()]
        assert all(shape == (n_samples,) for shape in shapes), f"Inconsistent shapes: {shapes}"
    
    def test_small_dataset_size_verification(self, tmp_path):
        """Test generation of minimal datasets for fast test execution."""
        prefix = tmp_path / "test_small"
        n_samples = 10  # Very small dataset
        
        generate_single_replicate_data(
            prefix=str(prefix),
            n_samples=n_samples,
            covariate_dim=5
        )
        
        # Verify small size
        control_reads = np.load(f"{prefix}_control_reads_r1.npy")
        assert len(control_reads) == n_samples, f"Expected {n_samples} samples, got {len(control_reads)}"
        
        # Verify file sizes are small (each array should be tiny)
        file_size = Path(f"{prefix}_control_reads_r1.npy").stat().st_size
        assert file_size < 1000, f"File size too large: {file_size} bytes"  # Should be much smaller
    
    def test_cleanup_functionality(self, tmp_path):
        """Test that temporary data files are properly cleaned up after tests."""
        temp_dir = tmp_path / "temp_data"
        temp_dir.mkdir()
        
        with SyntheticDataManager(base_dir=str(temp_dir)) as data_manager:
            # Generate some data inside the context manager
            prefix = data_manager.create_single_replicate_data(n_samples=25)
            
            # Verify files exist during context
            assert Path(f"{prefix}_control_reads_r1.npy").exists()
            assert Path(f"{prefix}_experiment_reads_r1.npy").exists()
        
        # After context exits, temporary files should be cleaned up
        # The temp directory structure might remain but the specific data files should be gone
        remaining_npy_files = list(temp_dir.rglob("*.npy"))
        assert len(remaining_npy_files) == 0, f"Found {len(remaining_npy_files)} .npy files that should have been cleaned up"


class TestMultiReplicateDataGeneration:
    """Additional tests specific to multi-replicate data generation."""
    
    def test_multi_replicate_dimension_consistency(self, tmp_path):
        """Test dimension consistency for multi-replicate datasets.""" 
        prefix = tmp_path / "test_multi_consistency"
        n_samples = 50
        
        generate_multi_replicate_data(
            prefix=str(prefix),
            n_samples=n_samples,
            covariate_dim=5
        )
        
        # Validate dimensions are consistent
        is_valid, error_msg = validate_data_dimensions(str(prefix), is_multi_replicate=True)
        assert is_valid, f"Multi-replicate dimension validation failed: {error_msg}"
    
    def test_configurable_parameters(self, tmp_path):
        """Test configurable parameters for dataset size and covariate dimensions."""
        prefix = tmp_path / "test_configurable"
        n_samples = 123  # Non-standard size
        covariate_dim = 7  # Non-standard dimension (though dataset expects 5)
        
        generate_single_replicate_data(
            prefix=str(prefix),
            n_samples=n_samples,
            covariate_dim=covariate_dim
        )
        
        # Verify the custom size was respected
        control_reads = np.load(f"{prefix}_control_reads_r1.npy")
        assert len(control_reads) == n_samples, f"Expected {n_samples} samples, got {len(control_reads)}"