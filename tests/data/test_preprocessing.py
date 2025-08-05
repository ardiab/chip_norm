import numpy as np
import pytest
from pathlib import Path
from omegaconf import OmegaConf

from chipvi.data.preprocessing import aggregate_bins, run_preprocessing


def test_aggregate_bins_correctly_sums_and_averages():
    """Test that bin aggregation correctly sums and averages data."""
    # Create simple numpy array representing 16 bins of high-resolution "reads" data
    reads_data = np.arange(16)  # [0, 1, 2, ..., 15]
    
    # Create corresponding numpy array for "mapq" data
    mapq_data = np.arange(16, 32)  # [16, 17, 18, ..., 31]
    
    # Call aggregate_bins function with aggregation factor of 8
    aggregated_reads = aggregate_bins(reads_data, agg_factor=8, agg_method="sum")
    aggregated_mapq = aggregate_bins(mapq_data, agg_factor=8, agg_method="mean")
    
    # Assert that the output "reads" array has length of 2
    assert len(aggregated_reads) == 2
    
    # Assert that values are correct sums of the original 8-bin windows
    # First window: sum of [0,1,2,3,4,5,6,7] = 28
    # Second window: sum of [8,9,10,11,12,13,14,15] = 92  
    expected_reads = [28, 92]
    np.testing.assert_array_equal(aggregated_reads, expected_reads)
    
    # Assert that the output "mapq" array has length of 2
    assert len(aggregated_mapq) == 2
    
    # Assert that values are correct means of the original 8-bin windows
    # First window: mean of [16,17,18,19,20,21,22,23] = 19.5
    # Second window: mean of [24,25,26,27,28,29,30,31] = 27.5
    expected_mapq = [19.5, 27.5]
    np.testing.assert_array_equal(aggregated_mapq, expected_mapq)


def test_preprocessing_pipeline_creates_valid_npy_files(tmp_path):
    """Test that the preprocessing pipeline creates valid .npy files."""
    # Create temporary directories for raw data and processed output
    raw_data_dir = tmp_path / "raw"
    processed_data_dir = tmp_path / "processed"
    raw_data_dir.mkdir()
    processed_data_dir.mkdir()
    
    # Create dummy raw BED files with simple, predictable data
    exp1_bed = raw_data_dir / "exp1.bed"
    ctrl1_bed = raw_data_dir / "ctrl1.bed"
    exp2_bed = raw_data_dir / "exp2.bed"
    ctrl2_bed = raw_data_dir / "ctrl2.bed"
    
    # Create simple BED data (chr, start, end, reads, mapq)
    bed_data = [
        "chr1\t0\t25\t10\t30",
        "chr1\t25\t50\t15\t35",
        "chr1\t50\t75\t20\t40",
        "chr1\t75\t100\t25\t45",
        "chr8\t0\t25\t5\t25",
        "chr8\t25\t50\t8\t28"
    ]
    
    for bed_file in [exp1_bed, ctrl1_bed, exp2_bed, ctrl2_bed]:
        bed_file.write_text('\n'.join(bed_data))
    
    # Create a mock configuration object pointing to these temporary files
    mock_config = {
        "name": "h3k27me3",
        "replicate_groups": [
            {
                "r1": {
                    "exp": str(exp1_bed),
                    "ctrl": str(ctrl1_bed)
                },
                "r2": {
                    "exp": str(exp2_bed),
                    "ctrl": str(ctrl2_bed)
                }
            }
        ],
        "train_chroms": ["chr1"],
        "val_chroms": ["chr8"],
        "aggregation_factor": 2,
        "output_prefix_train": "h3k27me3_train_200bp",
        "output_prefix_val": "h3k27me3_val_200bp",
        "paths": {
            "project_root": str(tmp_path),
            "data_base": str(tmp_path / "data"),
            "data_raw": str(raw_data_dir),
            "data_processed": str(processed_data_dir),
            "entex_base": str(tmp_path / "data" / "entex_files"),
            "entex_raw": str(tmp_path / "data" / "entex_files" / "raw"),
            "entex_processed": str(tmp_path / "data" / "entex_files" / "proc"),
            "outputs": str(tmp_path / "outputs")
        }
    }
    cfg = OmegaConf.create(mock_config)
    
    # Call the preprocessing function
    run_preprocessing(cfg)
    
    # Assert that the expected .npy files are created
    expected_train_files = [
        "h3k27me3_train_200bp_control_reads_r1.npy",
        "h3k27me3_train_200bp_control_mapq_r1.npy",
        "h3k27me3_train_200bp_experiment_reads_r1.npy",
        "h3k27me3_train_200bp_experiment_mapq_r1.npy"
    ]
    
    for filename in expected_train_files:
        file_path = processed_data_dir / filename
        assert file_path.exists(), f"Expected file {filename} was not created"
    
    # Load one of the created .npy files and assert its contents are correct
    train_exp_reads_r1 = np.load(processed_data_dir / "h3k27me3_train_200bp_experiment_reads_r1.npy")
    
    # With aggregation factor 2, chr1 data [10, 15, 20, 25] becomes [25, 45]
    expected_aggregated = np.array([25, 45])
    np.testing.assert_array_equal(train_exp_reads_r1, expected_aggregated)