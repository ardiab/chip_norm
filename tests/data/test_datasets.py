import torch
import numpy as np
from pathlib import Path

from chipvi.data.datasets import MultiReplicateDataset



def test_refactored_multireplicatedataset_returns_structured_dict(tmp_path):
    """Test that refactored MultiReplicateDataset returns a structured dictionary."""
    # Create test data arrays
    control_reads_r1 = np.ones(10, dtype=np.float32)
    control_mapq_r1 = np.zeros(10, dtype=np.float32)
    control_seq_depth_r1 = np.full(10, 2.0, dtype=np.float32)
    experiment_reads_r1 = np.full(10, 3.0, dtype=np.float32)
    experiment_mapq_r1 = np.full(10, 4.0, dtype=np.float32)
    experiment_seq_depth_r1 = np.full(10, 5.0, dtype=np.float32)
    
    control_reads_r2 = np.full(10, 6.0, dtype=np.float32)
    control_mapq_r2 = np.full(10, 7.0, dtype=np.float32)
    control_seq_depth_r2 = np.full(10, 8.0, dtype=np.float32)
    experiment_reads_r2 = np.full(10, 9.0, dtype=np.float32)
    experiment_mapq_r2 = np.full(10, 10.0, dtype=np.float32)
    experiment_seq_depth_r2 = np.full(10, 11.0, dtype=np.float32)
    grp_idxs = np.full(10, 12, dtype=np.int32)
    
    # Save arrays to temporary .npy files
    prefix = tmp_path / "test_data"
    np.save(f"{prefix}_control_reads_r1.npy", control_reads_r1)
    np.save(f"{prefix}_control_mapq_r1.npy", control_mapq_r1)
    np.save(f"{prefix}_control_seq_depth_r1.npy", control_seq_depth_r1)
    np.save(f"{prefix}_experiment_reads_r1.npy", experiment_reads_r1)
    np.save(f"{prefix}_experiment_mapq_r1.npy", experiment_mapq_r1)
    np.save(f"{prefix}_experiment_seq_depth_r1.npy", experiment_seq_depth_r1)
    
    np.save(f"{prefix}_control_reads_r2.npy", control_reads_r2)
    np.save(f"{prefix}_control_mapq_r2.npy", control_mapq_r2)
    np.save(f"{prefix}_control_seq_depth_r2.npy", control_seq_depth_r2)
    np.save(f"{prefix}_experiment_reads_r2.npy", experiment_reads_r2)
    np.save(f"{prefix}_experiment_mapq_r2.npy", experiment_mapq_r2)
    np.save(f"{prefix}_experiment_seq_depth_r2.npy", experiment_seq_depth_r2)
    np.save(f"{prefix}_grp_idxs.npy", grp_idxs)
    
    # Instantiate MultiReplicateDataset with prefix path
    dataset = MultiReplicateDataset(str(prefix))
    
    # Get the first item
    item = dataset[0]
    
    # Assert that item is a dictionary with top-level keys: 'r1', 'r2', and 'metadata'
    assert isinstance(item, dict), f"Expected item to be a dictionary, got {type(item)}"
    assert 'r1' in item, "Expected 'r1' key in item dictionary"
    assert 'r2' in item, "Expected 'r2' key in item dictionary"
    assert 'metadata' in item, "Expected 'metadata' key in item dictionary"
    
    # Assert that item['r1'] is a dictionary with keys 'covariates' and 'reads'
    assert isinstance(item['r1'], dict), f"Expected item['r1'] to be a dictionary, got {type(item['r1'])}"
    assert 'covariates' in item['r1'], "Expected 'covariates' key in item['r1'] dictionary"
    assert 'reads' in item['r1'], "Expected 'reads' key in item['r1'] dictionary"
    
    # Assert that the tensor item['r1']['covariates'] has the correct shape (5,) and contains the correct values
    r1_covariates = item['r1']['covariates']
    assert r1_covariates.shape == (5,), f"Expected r1 covariates shape (5,), got {r1_covariates.shape}"
    expected_r1_values = [1.0, 0.0, 2.0, 4.0, 5.0]  # control_reads_r1, control_mapq_r1, control_seq_depth_r1, experiment_mapq_r1, experiment_seq_depth_r1
    for i, expected_val in enumerate(expected_r1_values):
        assert r1_covariates[i].item() == expected_val, f"Expected r1_covariates[{i}] = {expected_val}, got {r1_covariates[i].item()}"
    
    # Assert that the scalar item['r1']['reads'] contains the correct value
    assert item['r1']['reads'].item() == 3.0, f"Expected r1 reads = 3.0, got {item['r1']['reads'].item()}"
    
    # Perform the same assertions for item['r2']
    assert isinstance(item['r2'], dict), f"Expected item['r2'] to be a dictionary, got {type(item['r2'])}"
    assert 'covariates' in item['r2'], "Expected 'covariates' key in item['r2'] dictionary"
    assert 'reads' in item['r2'], "Expected 'reads' key in item['r2'] dictionary"
    
    r2_covariates = item['r2']['covariates']
    assert r2_covariates.shape == (5,), f"Expected r2 covariates shape (5,), got {r2_covariates.shape}"
    expected_r2_values = [6.0, 7.0, 8.0, 10.0, 11.0]  # control_reads_r2, control_mapq_r2, control_seq_depth_r2, experiment_mapq_r2, experiment_seq_depth_r2
    for i, expected_val in enumerate(expected_r2_values):
        assert r2_covariates[i].item() == expected_val, f"Expected r2_covariates[{i}] = {expected_val}, got {r2_covariates[i].item()}"
    
    # Assert that the scalar item['r2']['reads'] contains the correct value
    assert item['r2']['reads'].item() == 9.0, f"Expected r2 reads = 9.0, got {item['r2']['reads'].item()}"
    
    # Assert that item['metadata']['sd_ratio'] contains the correct value
    expected_sd_ratio = 5.0 / 11.0  # experiment_seq_depth_r1 / experiment_seq_depth_r2
    assert 'sd_ratio' in item['metadata'], "Expected 'sd_ratio' key in item['metadata'] dictionary"
    assert abs(item['metadata']['sd_ratio'].item() - expected_sd_ratio) < 1e-6, f"Expected sd_ratio = {expected_sd_ratio}, got {item['metadata']['sd_ratio'].item()}"