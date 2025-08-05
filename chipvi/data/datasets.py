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


def build_datasets(cfg: DictConfig):
    """
    Build datasets from configuration by processing BED files.
    """
    from chipvi.utils.path_helper import PathHelper
    
    # Initialize path helper to resolve file paths
    paths = PathHelper(cfg)
    
    # Extract BED file paths from configuration
    if cfg.name == "h3k27me3":
        # Read replicate groups from configuration
        bed_fpath_groups = []
        for group in cfg.replicate_groups:
            group_paths = []
            for replicate in group:
                group_paths.append({
                    "exp": paths.entex_proc_file_dir / replicate.exp,
                    "ctrl": paths.entex_proc_file_dir / replicate.ctrl
                })
            bed_fpath_groups.append(group_paths)
        
        # Read chromosome splits from configuration
        train_chroms = cfg.train_chroms
        val_chroms = cfg.val_chroms
    else:
        raise ValueError(f"Unknown dataset name: {cfg.name}")
    
    # Each sample will have one experimental reads file and one control reads file.
    replicate_mode = len(bed_fpath_groups[0]) > 1

    if any(len(path_group) > 1 for path_group in bed_fpath_groups) and \
        any(len(path_group) != 2 for path_group in bed_fpath_groups):
        raise ValueError("Either all experiments or none must have exactly 2 replicates.")

    n_replicates = 2 if replicate_mode else 1
    train_ctrl_reads = {
            f"replicate_{i + 1}": [] for i in range(n_replicates)
        }
    val_ctrl_reads = {
            f"replicate_{i + 1}": [] for i in range(n_replicates)
        }
    train_ctrl_mapq = {
            f"replicate_{i + 1}": [] for i in range(n_replicates)
        }
    val_ctrl_mapq = {
            f"replicate_{i + 1}": [] for i in range(n_replicates)
        }
    train_ctrl_seq_depths = {
        f"replicate_{i + 1}": [] for i in range(n_replicates)
    }
    val_ctrl_seq_depths = {
        f"replicate_{i + 1}": [] for i in range(n_replicates)
    }

    train_exp_reads = {
        f"replicate_{i + 1}": [] for i in range(n_replicates)
    }
    val_exp_reads = {
        f"replicate_{i + 1}": [] for i in range(n_replicates)
    }
    train_exp_mapq = {
            f"replicate_{i + 1}": [] for i in range(n_replicates)
        }
    val_exp_mapq = {
            f"replicate_{i + 1}": [] for i in range(n_replicates)
        }
    train_exp_seq_depths = {
        f"replicate_{i + 1}": [] for i in range(n_replicates)
    }
    val_exp_seq_depths = {
        f"replicate_{i + 1}": [] for i in range(n_replicates)
    }
    train_grp_idxs = []
    val_grp_idxs = []


    for grp_idx, path_group in enumerate(bed_fpath_groups):
        if len(path_group) not in [1, 2]:
            raise ValueError(
                f"Each replicate group must have either 1 or 2 replicates, "
                f"but {len(path_group)} were found."
            )
        logger.info(
                    "Processing replicate group %d/%d",
                    grp_idx + 1,
                    len(bed_fpath_groups),
                    )
        rep_group_data = {}
        for rep_idx, rep_dict in enumerate(path_group):
            logger.info(
                "Reading Replicate %s data from exp %s and ctrl %s",
                rep_idx + 1,
                rep_dict["exp"],
                rep_dict["ctrl"],
            )
            exp_train_df, exp_val_df, exp_seq_depth = load_and_process_bed(
                rep_dict["exp"],
                train_chroms,
                val_chroms,
            )
            ctrl_train_df, ctrl_val_df, ctrl_seq_depth = load_and_process_bed(
                rep_dict["ctrl"],
                train_chroms,
                val_chroms,
            )
            rep_group_data[rep_idx] = {
                    "exp_train_df": exp_train_df,
                    "exp_val_df": exp_val_df,
                    "exp_seq_depth": exp_seq_depth,
                    "ctrl_train_df": ctrl_train_df,
                    "ctrl_val_df": ctrl_val_df,
                    "ctrl_seq_depth": ctrl_seq_depth,
                }
            logger.info(
                    "%s (exp) sequencing depth: %s",
                    rep_dict["exp"],
                    exp_seq_depth,
                    )
            logger.info(
                    "%s (ctrl) sequencing depth: %s",
                    rep_dict["ctrl"],
                    ctrl_seq_depth,
                    )

        if not replicate_mode:
            replicate_pairs = [(0, None)]
        else:
            replicate_pairs = list(combinations(rep_group_data.keys(), 2))
            if len(replicate_pairs) != 1:
                raise ValueError(f"Expected 1 replicate pair, but {len(replicate_pairs)} were found.")

        for rep_idx, (rep_1_idx, rep_2_idx) in enumerate(replicate_pairs):
            logger.info(
                    "Processing replicate combination %d/%d",
                    rep_idx + 1,
                    len(replicate_pairs),
                    )
            rep_dict_1 = rep_group_data[rep_1_idx]
            if not replicate_mode:
                nonzero_mask_train = rep_dict_1["exp_train_df"]["reads"] > 0
                nonzero_mask_val = rep_dict_1["exp_val_df"]["reads"] > 0
            else:
                rep_dict_2 = rep_group_data[rep_2_idx]
                nonzero_mask_train = (rep_dict_1["exp_train_df"]["reads"] > 0) | (rep_dict_2["exp_train_df"]["reads"] > 0)
                nonzero_mask_val = (rep_dict_1["exp_val_df"]["reads"] > 0) | (rep_dict_2["exp_val_df"]["reads"] > 0)

            exp_train_df_r1 = rep_dict_1["exp_train_df"][nonzero_mask_train]
            exp_val_df_r1 = rep_dict_1["exp_val_df"][nonzero_mask_val]
            exp_seq_depth_r1 = rep_dict_1["exp_seq_depth"] / SD_SCALING_FACTOR
            ctrl_train_df_r1 = rep_dict_1["ctrl_train_df"][nonzero_mask_train]
            ctrl_val_df_r1 = rep_dict_1["ctrl_val_df"][nonzero_mask_val]
            ctrl_seq_depth_r1 = rep_dict_1["ctrl_seq_depth"] / SD_SCALING_FACTOR


            train_ctrl_reads["replicate_1"].extend(ctrl_train_df_r1["reads"].values)
            train_ctrl_mapq["replicate_1"].extend(ctrl_train_df_r1["mapq"].values)
            train_ctrl_seq_depths["replicate_1"].extend([ctrl_seq_depth_r1] * len(exp_train_df_r1))
            train_exp_reads["replicate_1"].extend(exp_train_df_r1["reads"].values)
            train_exp_mapq["replicate_1"].extend(exp_train_df_r1["mapq"].values)
            train_exp_seq_depths["replicate_1"].extend([exp_seq_depth_r1] * len(exp_train_df_r1))

            val_ctrl_reads["replicate_1"].extend(ctrl_val_df_r1["reads"].values)
            val_ctrl_mapq["replicate_1"].extend(ctrl_val_df_r1["mapq"].values)
            val_ctrl_seq_depths["replicate_1"].extend([ctrl_seq_depth_r1] * len(exp_val_df_r1))
            val_exp_reads["replicate_1"].extend(exp_val_df_r1["reads"].values)
            val_exp_mapq["replicate_1"].extend(exp_val_df_r1["mapq"].values)
            val_exp_seq_depths["replicate_1"].extend([exp_seq_depth_r1] * len(exp_val_df_r1))

            if rep_2_idx is not None:
                rep_dict_2 = rep_group_data[rep_2_idx]
                exp_train_df_r2 = rep_dict_2["exp_train_df"][nonzero_mask_train]
                exp_val_df_r2 = rep_dict_2["exp_val_df"][nonzero_mask_val]
                exp_seq_depth_r2 = rep_dict_2["exp_seq_depth"] / SD_SCALING_FACTOR
                ctrl_train_df_r2 = rep_dict_2["ctrl_train_df"][nonzero_mask_train]
                ctrl_val_df_r2 = rep_dict_2["ctrl_val_df"][nonzero_mask_val]
                ctrl_seq_depth_r2 = rep_dict_2["ctrl_seq_depth"] / SD_SCALING_FACTOR


                train_ctrl_reads["replicate_2"].extend(ctrl_train_df_r2["reads"].values)
                train_ctrl_mapq["replicate_2"].extend(ctrl_train_df_r2["mapq"].values)
                train_ctrl_seq_depths["replicate_2"].extend([ctrl_seq_depth_r2] * len(exp_train_df_r2))
                train_exp_reads["replicate_2"].extend(exp_train_df_r2["reads"].values)
                train_exp_mapq["replicate_2"].extend(exp_train_df_r2["mapq"].values)
                train_exp_seq_depths["replicate_2"].extend([exp_seq_depth_r2] * len(exp_train_df_r2))
                train_grp_idxs.extend([grp_idx] * len(exp_train_df_r2))

                val_ctrl_reads["replicate_2"].extend(ctrl_val_df_r2["reads"].values)
                val_ctrl_mapq["replicate_2"].extend(ctrl_val_df_r2["mapq"].values)
                val_ctrl_seq_depths["replicate_2"].extend([ctrl_seq_depth_r2] * len(exp_val_df_r2))
                val_exp_reads["replicate_2"].extend(exp_val_df_r2["reads"].values)
                val_exp_mapq["replicate_2"].extend(exp_val_df_r2["mapq"].values)
                val_exp_seq_depths["replicate_2"].extend([exp_seq_depth_r2] * len(exp_val_df_r2))
                val_grp_idxs.extend([grp_idx] * len(exp_val_df_r2))


        del exp_train_df_r1, exp_val_df_r1, ctrl_train_df_r1, ctrl_val_df_r1  # type: ignore
        if replicate_mode:
            del exp_train_df_r2, exp_val_df_r2, ctrl_train_df_r2, ctrl_val_df_r2  # type: ignore
        del rep_group_data
        gc.collect()

    if not replicate_mode:
        return SingleReplicateDataset(
            control_reads=torch.tensor(train_ctrl_reads["replicate_1"], dtype=torch.float32),
            control_mapq=torch.tensor(train_ctrl_mapq["replicate_1"], dtype=torch.float32),
            control_seq_depth=torch.tensor(train_ctrl_seq_depths["replicate_1"], dtype=torch.float32),
            experiment_reads=torch.tensor(train_exp_reads["replicate_1"], dtype=torch.float32),
            experiment_mapq=torch.tensor(train_exp_mapq["replicate_1"], dtype=torch.float32),
            experiment_seq_depth=torch.tensor(train_exp_seq_depths["replicate_1"], dtype=torch.float32,),
        ), SingleReplicateDataset(
            control_reads=torch.tensor(val_ctrl_reads["replicate_1"], dtype=torch.float32),
            control_mapq=torch.tensor(val_ctrl_mapq["replicate_1"], dtype=torch.float32),
            control_seq_depth=torch.tensor(val_ctrl_seq_depths["replicate_1"], dtype=torch.float32),
            experiment_reads=torch.tensor(val_exp_reads["replicate_1"], dtype=torch.float32),
            experiment_mapq=torch.tensor(val_exp_mapq["replicate_1"], dtype=torch.float32),
            experiment_seq_depth=torch.tensor(val_exp_seq_depths["replicate_1"], dtype=torch.float32,),
        )
    else:
        return MultiReplicateDataset(
                control_reads_r1=torch.tensor(train_ctrl_reads["replicate_1"], dtype=torch.float32),
                control_mapq_r1=torch.tensor(train_ctrl_mapq["replicate_1"], dtype=torch.float32),
                control_seq_depth_r1=torch.tensor(train_ctrl_seq_depths["replicate_1"], dtype=torch.float32),
                experiment_reads_r1=torch.tensor(train_exp_reads["replicate_1"], dtype=torch.float32),
                experiment_mapq_r1=torch.tensor(train_exp_mapq["replicate_1"], dtype=torch.float32),
                experiment_seq_depth_r1=torch.tensor(train_exp_seq_depths["replicate_1"], dtype=torch.float32),
                control_reads_r2=torch.tensor(train_ctrl_reads["replicate_2"], dtype=torch.float32),
                control_mapq_r2=torch.tensor(train_ctrl_mapq["replicate_2"], dtype=torch.float32),
                control_seq_depth_r2=torch.tensor(train_ctrl_seq_depths["replicate_2"], dtype=torch.float32),
                experiment_reads_r2=torch.tensor(train_exp_reads["replicate_2"], dtype=torch.float32),
                experiment_mapq_r2=torch.tensor(train_exp_mapq["replicate_2"], dtype=torch.float32),
                experiment_seq_depth_r2=torch.tensor(train_exp_seq_depths["replicate_2"], dtype=torch.float32),
                grp_idxs=torch.tensor(train_grp_idxs, dtype=torch.int32),
        ), MultiReplicateDataset(
                control_reads_r1=torch.tensor(val_ctrl_reads["replicate_1"], dtype=torch.float32),
                control_mapq_r1=torch.tensor(val_ctrl_mapq["replicate_1"], dtype=torch.float32),
                control_seq_depth_r1=torch.tensor(val_ctrl_seq_depths["replicate_1"], dtype=torch.float32),
                experiment_reads_r1=torch.tensor(val_exp_reads["replicate_1"], dtype=torch.float32),
                experiment_mapq_r1=torch.tensor(val_exp_mapq["replicate_1"], dtype=torch.float32),
                experiment_seq_depth_r1=torch.tensor(val_exp_seq_depths["replicate_1"], dtype=torch.float32),
                control_reads_r2=torch.tensor(val_ctrl_reads["replicate_2"], dtype=torch.float32),
                control_mapq_r2=torch.tensor(val_ctrl_mapq["replicate_2"], dtype=torch.float32),
                control_seq_depth_r2=torch.tensor(val_ctrl_seq_depths["replicate_2"], dtype=torch.float32),
                experiment_reads_r2=torch.tensor(val_exp_reads["replicate_2"], dtype=torch.float32),
                experiment_mapq_r2=torch.tensor(val_exp_mapq["replicate_2"], dtype=torch.float32),
                experiment_seq_depth_r2=torch.tensor(val_exp_seq_depths["replicate_2"], dtype=torch.float32),
                grp_idxs=torch.tensor(val_grp_idxs, dtype=torch.int32),
        )


def load_and_process_bed(bed_fpath, train_chroms, val_chroms):
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

    train_df = df[df["chr"].isin(train_chroms)]
    val_df = df[df["chr"].isin(val_chroms)]

    # Based on examination, basically every bin with a null MAPQ has 0 reads. Since we
    # are converting from MAPQ to probability of the signal being correct, we can treat
    # no reads as having probability 1 (since no sequences mapped to the bin). We drop
    # 0 exp reads anyway, so this only matters for control reads.
    train_df["mapq"] = 1 - 10**(df["mapq"].replace("NAN", 100).astype(np.float32) / -10)
    val_df["mapq"] = 1 - 10**(val_df["mapq"].replace("NAN", 100).astype(np.float32) / -10)

    return train_df, val_df, seq_depth


class MultiReplicateDataset(Dataset):
    def __init__(
        self,
        control_reads_r1: torch.Tensor,
        control_mapq_r1: torch.Tensor,
        control_seq_depth_r1: torch.Tensor,
        experiment_reads_r1: torch.Tensor,
        experiment_mapq_r1: torch.Tensor,
        experiment_seq_depth_r1: torch.Tensor,
        control_reads_r2: torch.Tensor,
        control_mapq_r2: torch.Tensor,
        control_seq_depth_r2: torch.Tensor,
        experiment_reads_r2: torch.Tensor,
        experiment_mapq_r2: torch.Tensor,
        experiment_seq_depth_r2: torch.Tensor,
        grp_idxs: torch.Tensor | None = None,
    ) -> None:
        self.control_reads_r1 = control_reads_r1
        self.control_mapq_r1 = control_mapq_r1
        self.control_seq_depth_r1 = control_seq_depth_r1
        self.experiment_reads_r1 = experiment_reads_r1
        self.experiment_mapq_r1 = experiment_mapq_r1
        self.experiment_seq_depth_r1 = experiment_seq_depth_r1
        self.control_reads_r2 = control_reads_r2
        self.control_mapq_r2 = control_mapq_r2
        self.control_seq_depth_r2 = control_seq_depth_r2
        self.experiment_reads_r2 = experiment_reads_r2
        self.experiment_mapq_r2 = experiment_mapq_r2
        self.experiment_seq_depth_r2 = experiment_seq_depth_r2
        self.exp_sd_ratio = experiment_seq_depth_r1 / experiment_seq_depth_r2
        if grp_idxs is not None:
            self.grp_idxs = grp_idxs
        else:
            self.grp_idxs = torch.zeros(len(experiment_reads_r1), dtype=torch.int32)
            for idx, seq_depth in enumerate(experiment_seq_depth_r1.unique()):
                self.grp_idxs[experiment_seq_depth_r1 == seq_depth] = idx

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
    def __init__(
        self,
        control_reads: torch.Tensor,
        control_mapq: torch.Tensor,
        control_seq_depth: torch.Tensor,
        experiment_reads: torch.Tensor,
        experiment_mapq: torch.Tensor,
        experiment_seq_depth: torch.Tensor,
    ) -> None:
        self.control_reads = control_reads
        self.control_mapq = control_mapq
        self.control_seq_depth = control_seq_depth
        self.experiment_reads = experiment_reads
        self.experiment_mapq = experiment_mapq
        self.experiment_seq_depth = experiment_seq_depth

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

