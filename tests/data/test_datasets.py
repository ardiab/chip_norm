import torch

from chipvi.data.datasets import MultiReplicateDataset


def test_multireplicatedataset_getitem_structure():
    """Test that MultiReplicateDataset returns expected structure."""
    # Create small, distinct torch.Tensor objects for each of the 13 required inputs
    control_reads_r1 = torch.ones(10)
    control_mapq_r1 = torch.zeros(10)
    control_seq_depth_r1 = torch.full((10,), 2.0)
    experiment_reads_r1 = torch.full((10,), 3.0)
    experiment_mapq_r1 = torch.full((10,), 4.0)
    experiment_seq_depth_r1 = torch.full((10,), 5.0)
    
    control_reads_r2 = torch.full((10,), 6.0)
    control_mapq_r2 = torch.full((10,), 7.0)
    control_seq_depth_r2 = torch.full((10,), 8.0)
    experiment_reads_r2 = torch.full((10,), 9.0)
    experiment_mapq_r2 = torch.full((10,), 10.0)
    experiment_seq_depth_r2 = torch.full((10,), 11.0)
    grp_idxs = torch.full((10,), 12)
    
    # Instantiate MultiReplicateDataset with these tensors
    dataset = MultiReplicateDataset(
        control_reads_r1=control_reads_r1,
        control_mapq_r1=control_mapq_r1,
        control_seq_depth_r1=control_seq_depth_r1,
        experiment_reads_r1=experiment_reads_r1,
        experiment_mapq_r1=experiment_mapq_r1,
        experiment_seq_depth_r1=experiment_seq_depth_r1,
        control_reads_r2=control_reads_r2,
        control_mapq_r2=control_mapq_r2,
        control_seq_depth_r2=control_seq_depth_r2,
        experiment_reads_r2=experiment_reads_r2,
        experiment_mapq_r2=experiment_mapq_r2,
        experiment_seq_depth_r2=experiment_seq_depth_r2,
        grp_idxs=grp_idxs,
    )
    
    # Get the first item
    covariates, experiment_reads = dataset[0]
    
    # Assert that the covariates tensor has the expected shape (11,)
    assert covariates.shape == (11,), f"Expected shape (11,), got {covariates.shape}"
    
    # Assert that the first five values correspond to the five replicate_1 input tensors
    expected_r1_covariates = [1.0, 0.0, 2.0, 4.0, 5.0]  # control_reads_r1, control_mapq_r1, control_seq_depth_r1, experiment_mapq_r1, experiment_seq_depth_r1
    for i, expected_val in enumerate(expected_r1_covariates):
        assert covariates[i].item() == expected_val, f"Expected covariates[{i}] = {expected_val}, got {covariates[i].item()}"
    
    # Assert that the next five values correspond to the five replicate_2 input tensors
    expected_r2_covariates = [6.0, 7.0, 8.0, 10.0, 11.0]  # control_reads_r2, control_mapq_r2, control_seq_depth_r2, experiment_mapq_r2, experiment_seq_depth_r2
    for i, expected_val in enumerate(expected_r2_covariates):
        assert covariates[i + 5].item() == expected_val, f"Expected covariates[{i + 5}] = {expected_val}, got {covariates[i + 5].item()}"
    
    # Assert that the last value is the group index
    assert covariates[10].item() == 12.0, f"Expected covariates[10] = 12.0, got {covariates[10].item()}"
    
    # Assert that the experiment reads tensor has the expected shape (3,)
    assert experiment_reads.shape == (3,), f"Expected shape (3,), got {experiment_reads.shape}"
    
    # Assert that its values correspond to the correct inputs (experiment_reads_r1, experiment_reads_r2, and seq depth ratio)
    assert experiment_reads[0].item() == 3.0, f"Expected experiment_reads[0] = 3.0, got {experiment_reads[0].item()}"
    assert experiment_reads[1].item() == 9.0, f"Expected experiment_reads[1] = 9.0, got {experiment_reads[1].item()}"
    # seq depth ratio = experiment_seq_depth_r1 / experiment_seq_depth_r2 = 5.0 / 11.0
    expected_ratio = 5.0 / 11.0
    assert abs(experiment_reads[2].item() - expected_ratio) < 1e-6, f"Expected experiment_reads[2] = {expected_ratio}, got {experiment_reads[2].item()}"