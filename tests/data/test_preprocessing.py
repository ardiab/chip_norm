import numpy as np
import pytest

from chipvi.data.preprocessing import aggregate_bins


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