from __future__ import annotations

import gc
import logging
import pickle
from typing import TYPE_CHECKING, Literal
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from itertools import combinations
from omegaconf import DictConfig

if TYPE_CHECKING:
    from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SD_SCALING_FACTOR = 100_000_000




class MultiReplicateDataset(Dataset):
    def __init__(self, prefix_path: str) -> None:
        """Initialize MultiReplicateDataset by loading data from .npy files."""
        prefix = Path(prefix_path)
        
        # Load all arrays using memory mapping for efficiency
        self.control_reads_r1 = torch.from_numpy(np.load(f"{prefix}_control_reads_r1.npy", mmap_mode='r'))
        self.control_mapq_r1 = torch.from_numpy(np.load(f"{prefix}_control_mapq_r1.npy", mmap_mode='r'))
        self.control_seq_depth_r1 = torch.from_numpy(np.load(f"{prefix}_control_seq_depth_r1.npy", mmap_mode='r'))
        self.experiment_reads_r1 = torch.from_numpy(np.load(f"{prefix}_experiment_reads_r1.npy", mmap_mode='r'))
        self.experiment_mapq_r1 = torch.from_numpy(np.load(f"{prefix}_experiment_mapq_r1.npy", mmap_mode='r'))
        self.experiment_seq_depth_r1 = torch.from_numpy(np.load(f"{prefix}_experiment_seq_depth_r1.npy", mmap_mode='r'))
        
        self.control_reads_r2 = torch.from_numpy(np.load(f"{prefix}_control_reads_r2.npy", mmap_mode='r'))
        self.control_mapq_r2 = torch.from_numpy(np.load(f"{prefix}_control_mapq_r2.npy", mmap_mode='r'))
        self.control_seq_depth_r2 = torch.from_numpy(np.load(f"{prefix}_control_seq_depth_r2.npy", mmap_mode='r'))
        self.experiment_reads_r2 = torch.from_numpy(np.load(f"{prefix}_experiment_reads_r2.npy", mmap_mode='r'))
        self.experiment_mapq_r2 = torch.from_numpy(np.load(f"{prefix}_experiment_mapq_r2.npy", mmap_mode='r'))
        self.experiment_seq_depth_r2 = torch.from_numpy(np.load(f"{prefix}_experiment_seq_depth_r2.npy", mmap_mode='r'))
        
        # Load group indices if they exist
        grp_idxs_path = f"{prefix}_grp_idxs.npy"
        if Path(grp_idxs_path).exists():
            self.grp_idxs = torch.from_numpy(np.load(grp_idxs_path, mmap_mode='r'))
        else:
            self.grp_idxs = torch.zeros(len(self.experiment_reads_r1), dtype=torch.int32)
            for idx, seq_depth in enumerate(self.experiment_seq_depth_r1.unique()):
                self.grp_idxs[self.experiment_seq_depth_r1 == seq_depth] = idx
        
        # Calculate sequencing depth ratio
        self.exp_sd_ratio = self.experiment_seq_depth_r1 / self.experiment_seq_depth_r2
        self.n_covariates_per_replicate = 5

    def get_covariate_dim(self) -> int:
        """Return the number of covariates for a single replicate."""
        return self.n_covariates_per_replicate

    def __len__(self) -> int:
        return len(self.experiment_reads_r1)

    def __getitem__(self, idx: int) -> dict:
        """Return a structured dictionary for the given index."""
        r1_covariates = torch.stack([
            self.control_reads_r1[idx],
            self.control_mapq_r1[idx],
            self.control_seq_depth_r1[idx],
            self.experiment_mapq_r1[idx],
            self.experiment_seq_depth_r1[idx],
        ])
        r2_covariates = torch.stack([
            self.control_reads_r2[idx],
            self.control_mapq_r2[idx],
            self.control_seq_depth_r2[idx],
            self.experiment_mapq_r2[idx],
            self.experiment_seq_depth_r2[idx],
        ])
        return {
            'r1': {'covariates': r1_covariates, 'reads': self.experiment_reads_r1[idx]},
            'r2': {'covariates': r2_covariates, 'reads': self.experiment_reads_r2[idx]},
            'metadata': {
                'sd_ratio': self.exp_sd_ratio[idx],
                'grp_idx': self.grp_idxs[idx],
            }
        }


class SingleReplicateDataset(Dataset):
    def __init__(self, prefix_path: str) -> None:
        """Initialize SingleReplicateDataset by loading data from .npy files."""
        prefix = Path(prefix_path)
        
        # Load all arrays using memory mapping for efficiency
        self.control_reads = torch.from_numpy(np.load(f"{prefix}_control_reads_r1.npy", mmap_mode='r'))
        self.control_mapq = torch.from_numpy(np.load(f"{prefix}_control_mapq_r1.npy", mmap_mode='r'))
        self.control_seq_depth = torch.from_numpy(np.load(f"{prefix}_control_seq_depth_r1.npy", mmap_mode='r'))
        self.experiment_reads = torch.from_numpy(np.load(f"{prefix}_experiment_reads_r1.npy", mmap_mode='r'))
        self.experiment_mapq = torch.from_numpy(np.load(f"{prefix}_experiment_mapq_r1.npy", mmap_mode='r'))
        self.experiment_seq_depth = torch.from_numpy(np.load(f"{prefix}_experiment_seq_depth_r1.npy", mmap_mode='r'))

    def get_covariate_dim(self) -> int:
        """Return the number of covariates."""
        return 5

    def __len__(self) -> int:
        return len(self.experiment_reads)

    def __getitem__(self, idx: int) -> dict:
        """Return a dictionary with covariates and reads."""
        covariates = torch.stack([
            self.control_reads[idx],
            self.control_mapq[idx],
            self.control_seq_depth[idx],
            self.experiment_mapq[idx],
            self.experiment_seq_depth[idx]
        ])
        return {'covariates': covariates, 'reads': self.experiment_reads[idx]}

