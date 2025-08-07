import subprocess
import tempfile
import pickle
import numpy as np
import pytest
from pathlib import Path

from chipvi.data.datasets import MultiReplicateDataset


class TestLegacyDataConverter:
    
    def _create_legacy_data(self, target_dir: Path, n_samples: int = 100) -> dict:
        """Create sample legacy pickle data structure."""
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample data similar to what we saw in the notebook
        data = {
            'train': {},
            'val': {}
        }
        
        # Generate sample data for train split (80% of samples)
        train_size = int(0.8 * n_samples)
        val_size = n_samples - train_size
        
        # Train data columns
        data['train']['control_reads_r1'] = np.random.randint(0, 10, size=train_size).astype(float)
        data['train']['control_mapq_r1'] = np.random.uniform(0.95, 1.0, size=train_size).astype(float)
        data['train']['control_seq_depth_r1'] = np.random.uniform(1.5, 3.0, size=train_size).astype(float)
        data['train']['experiment_reads_r1'] = np.random.randint(0, 20, size=train_size).astype(float)
        data['train']['experiment_mapq_r1'] = np.random.uniform(0.95, 1.0, size=train_size).astype(float)
        data['train']['experiment_seq_depth_r1'] = np.random.uniform(1.5, 3.0, size=train_size).astype(float)
        
        data['train']['control_reads_r2'] = np.random.randint(0, 10, size=train_size).astype(float)
        data['train']['control_mapq_r2'] = np.random.uniform(0.95, 1.0, size=train_size).astype(float)
        data['train']['control_seq_depth_r2'] = np.random.uniform(1.5, 3.0, size=train_size).astype(float)
        data['train']['experiment_reads_r2'] = np.random.randint(0, 20, size=train_size).astype(float)
        data['train']['experiment_mapq_r2'] = np.random.uniform(0.95, 1.0, size=train_size).astype(float)
        data['train']['experiment_seq_depth_r2'] = np.random.uniform(1.5, 3.0, size=train_size).astype(float)
        
        # Val data columns
        data['val']['control_reads_r1'] = np.random.randint(0, 10, size=val_size).astype(float)
        data['val']['control_mapq_r1'] = np.random.uniform(0.95, 1.0, size=val_size).astype(float)
        data['val']['control_seq_depth_r1'] = np.random.uniform(1.5, 3.0, size=val_size).astype(float)
        data['val']['experiment_reads_r1'] = np.random.randint(0, 20, size=val_size).astype(float)
        data['val']['experiment_mapq_r1'] = np.random.uniform(0.95, 1.0, size=val_size).astype(float)
        data['val']['experiment_seq_depth_r1'] = np.random.uniform(1.5, 3.0, size=val_size).astype(float)
        
        data['val']['control_reads_r2'] = np.random.randint(0, 10, size=val_size).astype(float)
        data['val']['control_mapq_r2'] = np.random.uniform(0.95, 1.0, size=val_size).astype(float)
        data['val']['control_seq_depth_r2'] = np.random.uniform(1.5, 3.0, size=val_size).astype(float)
        data['val']['experiment_reads_r2'] = np.random.randint(0, 20, size=val_size).astype(float)
        data['val']['experiment_mapq_r2'] = np.random.uniform(0.95, 1.0, size=val_size).astype(float)
        data['val']['experiment_seq_depth_r2'] = np.random.uniform(1.5, 3.0, size=val_size).astype(float)
        
        # Save train pickle files
        for column, values in data['train'].items():
            pkl_path = target_dir / f"train_{column}.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(values, f)
                
        # Save val pickle files
        for column, values in data['val'].items():
            pkl_path = target_dir / f"val_{column}.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(values, f)
        
        # Create a dummy sd_map file (should be skipped)
        sd_map_path = target_dir / "sd_map_dummy.pkl"
        with open(sd_map_path, 'wb') as f:
            pickle.dump({"dummy": 1.0}, f)
            
        return data
    
    def test_legacy_data_conversion_accuracy(self, tmp_path):
        """Test that legacy pickle data is accurately converted to .npy format."""
        # Create legacy data
        target_dir = tmp_path / "test_target_data_v2"
        original_data = self._create_legacy_data(target_dir)
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Run conversion script
        cmd = [
            'python', '/workspace/scripts/convert_legacy_data.py',
            '--target', 'test_target',
            '--legacy_data_dir', str(target_dir),
            '--output_directory', str(output_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Conversion failed: {result.stderr}"
        
        # Load converted data and verify accuracy
        prefix = output_dir / "test_target"
        dataset = MultiReplicateDataset(str(prefix))
        
        # Verify data shapes and basic properties
        train_len = len(original_data['train']['control_reads_r1'])
        val_len = len(original_data['val']['control_reads_r1'])
        total_len = train_len + val_len
        
        assert len(dataset) == total_len
        
        # Get first item and verify structure
        item = dataset[0]
        assert 'r1' in item
        assert 'r2' in item 
        assert 'metadata' in item
        assert 'covariates' in item['r1']
        assert 'reads' in item['r1']
        assert 'covariates' in item['r2']
        assert 'reads' in item['r2']
        
        # Verify covariate dimensions
        assert item['r1']['covariates'].shape[0] == 5
        assert item['r2']['covariates'].shape[0] == 5
    
    def test_log_transform_application(self, tmp_path):
        """Test that log transforms are correctly applied to covariate columns 0 and 5."""
        # Create legacy data with specific values for testing transforms
        target_dir = tmp_path / "test_target_data_v2"
        n_samples = 50
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Create specific values for columns that should be log-transformed
        # Covariate column 0 = control_reads_r1, column 5 would be control_reads_r2 
        control_reads_r1_train = np.array([1.0, 2.0, 3.0, 10.0] * 10 + [1.0, 2.0])  # 42 values
        control_reads_r2_train = np.array([2.0, 4.0, 6.0, 20.0] * 10 + [2.0, 4.0])  # 42 values
        train_size = len(control_reads_r1_train)
        
        control_reads_r1_val = np.array([5.0, 8.0] * 4)  # 8 values  
        control_reads_r2_val = np.array([10.0, 16.0] * 4)  # 8 values
        val_size = len(control_reads_r1_val)
        
        # Create other required columns with dummy data
        train_data = {
            'control_reads_r1': control_reads_r1_train,
            'control_reads_r2': control_reads_r2_train,
            'control_mapq_r1': np.ones(train_size),
            'control_seq_depth_r1': np.full(train_size, 2.0),
            'experiment_reads_r1': np.ones(train_size),
            'experiment_mapq_r1': np.ones(train_size),
            'experiment_seq_depth_r1': np.full(train_size, 2.5),
            'control_mapq_r2': np.ones(train_size),
            'control_seq_depth_r2': np.full(train_size, 2.0),
            'experiment_reads_r2': np.ones(train_size),
            'experiment_mapq_r2': np.ones(train_size),
            'experiment_seq_depth_r2': np.full(train_size, 2.5),
        }
        
        val_data = {
            'control_reads_r1': control_reads_r1_val,
            'control_reads_r2': control_reads_r2_val,
            'control_mapq_r1': np.ones(val_size),
            'control_seq_depth_r1': np.full(val_size, 2.0),
            'experiment_reads_r1': np.ones(val_size),
            'experiment_mapq_r1': np.ones(val_size),
            'experiment_seq_depth_r1': np.full(val_size, 2.5),
            'control_mapq_r2': np.ones(val_size),
            'control_seq_depth_r2': np.full(val_size, 2.0),
            'experiment_reads_r2': np.ones(val_size),
            'experiment_mapq_r2': np.ones(val_size),
            'experiment_seq_depth_r2': np.full(val_size, 2.5),
        }
        
        # Save pickle files
        for column, values in train_data.items():
            pkl_path = target_dir / f"train_{column}.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(values, f)
                
        for column, values in val_data.items():
            pkl_path = target_dir / f"val_{column}.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(values, f)
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Run conversion script WITH log transform
        cmd = [
            'python', '/workspace/scripts/convert_legacy_data.py',
            '--target', 'test_target',
            '--legacy_data_dir', str(target_dir),
            '--output_directory', str(output_dir),
            '--log_transform_inputs'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Conversion with log transform failed: {result.stderr}"
        
        # Load converted data
        prefix = output_dir / "test_target"
        dataset = MultiReplicateDataset(str(prefix))
        
        # Verify log transforms were applied
        item = dataset[0]
        
        # Check that covariate column 0 (control_reads_r1) was log-transformed
        # Original first value was 1.0, log(1.0 + 1) = log(2.0) ≈ 0.693
        expected_log_val_r1 = np.log(control_reads_r1_train[0] + 1)
        actual_val_r1 = item['r1']['covariates'][0].item()
        assert abs(actual_val_r1 - expected_log_val_r1) < 1e-6, f"Log transform not applied correctly to r1: expected {expected_log_val_r1}, got {actual_val_r1}"
        
        # Check that covariate column 0 for r2 (control_reads_r2) was log-transformed
        expected_log_val_r2 = np.log(control_reads_r2_train[0] + 1)
        actual_val_r2 = item['r2']['covariates'][0].item()
        assert abs(actual_val_r2 - expected_log_val_r2) < 1e-6, f"Log transform not applied correctly to r2: expected {expected_log_val_r2}, got {actual_val_r2}"
        
        # Verify other columns were NOT log-transformed (should be original values)
        # Column 1 should be mapq_r1 (original value = 1.0)
        assert abs(item['r1']['covariates'][1].item() - 1.0) < 1e-6