from __future__ import annotations

import argparse
import copy
import datetime
import logging
import pickle as pkl
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from chipvi.data.datasets import MultiReplicateDataset
from chipvi.models.technical_model import TechNB_mu_r
from chipvi.utils.distributions import compute_numeric_cdf
from chipvi.utils.plots import hist2d

if TYPE_CHECKING:
    from pathlib import Path
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
DATA_DIR = Path("/lotterlab/users/abdul/repos/chipvi/")

# --- New Loss Functions ---
def concordance_loss_NCE(res1, res2, tau=0.1):
    res1 = res1.unsqueeze(1)
    res2 = res2.unsqueeze(0)
    sim = torch.exp(-(res1 - res2).pow(2) / tau)
    pos = sim.diag()
    eps = 1e-8
    loss = -torch.log((pos + eps) / (sim.sum(dim=1) + eps))
    return loss.mean()

def negative_pearson_loss(preds, targets, eps=1e-8):
    preds_centered = preds - preds.mean()
    targets_centered = targets - targets.mean()
    numerator = (preds_centered * targets_centered).sum()
    denominator = torch.sqrt((preds_centered**2).sum() * (targets_centered**2).sum() + eps)
    return -numerator / denominator

class TechTrainer_Mu_R:
    def __init__(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: torch.device,
            consistency_loss: str,
            consistency_weight: float,
            ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.consistency_loss = consistency_loss
        self.consistency_weight = consistency_weight
        self.model: TechNB_mu_r = None
        self.optimizer: optim.Optimizer = None
        self.scheduler: SequentialLR = None
        self.best_val_loss = float("inf")
        self.best_val_corr = float("-inf")
        self.epochs_no_improve = 0

    def build_and_train(
        self,
        covariate_dim: int,
        hidden_dims_mu: tuple,
        hidden_dims_r: tuple,
        weight_decay: float,
        num_epochs: int,
        base_lr: float,
        patience: int,
        warmup_epochs: int,
        run_name: str,
        run_group: str,
        model_save_dir: Path,
        ) -> None:

        wandb.init(
            project="chipvi",
            name=run_name,
            group=run_group,
            config={
                "covariate_dim": covariate_dim,
                "hidden_dims_mu": hidden_dims_mu,
                "hidden_dims_r": hidden_dims_r,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
                "base_lr": base_lr,
                "patience": patience,
                "warmup_epochs": warmup_epochs,
                "consistency_loss": self.consistency_loss,
                "consistency_weight": self.consistency_weight,
            }
        )

        self.model = TechNB_mu_r(
            covariate_dim=covariate_dim,
            hidden_dims_mu=hidden_dims_mu,
            hidden_dims_r=hidden_dims_r
            ).to(self.device)
        
        wandb.watch(self.model, log="all")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=base_lr, weight_decay=weight_decay)
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs - warmup_epochs)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

        logger.info("Starting training...")
        for epoch in range(num_epochs):
            self.loader_forward(self.train_loader, train=True, log_freq=1_000)
            avg_val_loss, _, avg_val_res_corr, avg_val_quant_corr = self.loader_forward(self.val_loader, train=False)
            self.scheduler.step()

            wandb.summary["best_residual_spearman"] = avg_val_res_corr
            wandb.summary["best_quantile_spearman"] = avg_val_quant_corr

            improve = False
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), model_save_dir / "best_loss.pt")
                logger.info(f"New best validation loss: {self.best_val_loss:.4f}. Model saved.")
                improve = True
            if avg_val_res_corr > self.best_val_corr:
                self.best_val_corr = avg_val_res_corr
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), model_save_dir / "best_corr.pt")
                logger.info(f"New best validation residual correlation: {self.best_val_corr:.4f}. Model saved.")
                improve = True

            if not improve:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping after {epoch + 1} epochs.")
                break

    def loader_forward(self, loader: DataLoader, train: bool, log_freq: int = 1_000):
        total_loss, total_mse, n_batches = 0, 0, 0
        self.model.train(train)
        
        if not train:
            all_r1_res, all_r2_res_scaled, all_r1_quant, all_r2_quant = [], [], [], []
            all_y_r1, all_r1_pred_mean, all_y_r2, all_r2_pred_mean = [], [], [], []

        cov_dim = loader.dataset.get_dim_x()
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            if train: self.optimizer.zero_grad()

            x_r1, x_r2 = x[:, :cov_dim], x[:, cov_dim:2*cov_dim]
            y_r1, y_r2, sd_ratio_r1_r2 = y[:, 0], y[:, 1], y[:, 2]

            with torch.set_grad_enabled(train):
                r1_dist = self.model.predict_dist(x_r1)
                r2_dist = self.model.predict_dist(x_r2)
                r1_ll = r1_dist.log_prob(y_r1).mean()
                r2_ll = r2_dist.log_prob(y_r2).mean()

                r1_residual = y_r1 - r1_dist.mean
                r2_residual_scaled = sd_ratio_r1_r2 * (y_r2 - r2_dist.mean)

                # Compute quantiles for loss if needed, or just for validation plotting
                if not train or self.consistency_loss == 'quantile_abs':
                    r1_quant = compute_numeric_cdf(r1_dist, y_r1)
                    r2_quant = compute_numeric_cdf(r2_dist, y_r2)

                if self.consistency_loss == 'quantile_abs':
                    consistency_loss = torch.mean(torch.abs(r1_quant - r2_quant))
                elif self.consistency_loss == 'infonce':
                    consistency_loss = concordance_loss_NCE(r1_residual, r2_residual_scaled, tau=0.1)
                elif self.consistency_loss == 'pearson':
                    consistency_loss = negative_pearson_loss(r1_residual, r2_residual_scaled)
                else:
                    consistency_loss = 0.0

                loss = -(r1_ll + r2_ll) + self.consistency_weight * consistency_loss

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if not train:
                all_r1_res.append(r1_residual.cpu().numpy())
                all_r2_res_scaled.append(r2_residual_scaled.cpu().numpy())
                all_r1_quant.append(r1_quant.cpu().numpy())
                all_r2_quant.append(r2_quant.cpu().numpy())
                all_y_r1.append(y_r1.cpu().numpy()); all_r1_pred_mean.append(r1_dist.mean.cpu().numpy())
                all_y_r2.append(y_r2.cpu().numpy()); all_r2_pred_mean.append(r2_dist.mean.cpu().numpy())

        if train:
            wandb.log({"train_loss": total_loss / n_batches})
            return total_loss / n_batches, np.nan, np.nan, np.nan
        else:
            _df = pd.DataFrame({
                "r1_res": np.concatenate(all_r1_res), "r2_res_scaled": np.concatenate(all_r2_res_scaled),
                "r1_quant": np.concatenate(all_r1_quant), "r2_quant": np.concatenate(all_r2_quant),
                "y_r1": np.concatenate(all_y_r1), "r1_pred_mean": np.concatenate(all_r1_pred_mean),
                "y_r2": np.concatenate(all_y_r2), "r2_pred_mean": np.concatenate(all_r2_pred_mean),
            })
            
            res_corr = _df.corr(method='spearman').loc['r1_res', 'r2_res_scaled']
            quant_corr = _df.corr(method='spearman').loc['r1_quant', 'r2_quant']

            wandb.log({"val_loss": total_loss / n_batches, "val_residual_spearman": res_corr, "val_quantile_spearman": quant_corr})
            
            fig, axes = plt.subplots(3, 3, figsize=(18, 18)); fig.tight_layout(pad=5.0)
            hist2d(_df, "y_r1", "r1_pred_mean", axes[0, 0], discrete_bins=True); axes[0, 0].set_title(f"R1: Pred vs Obs\n{axes[0, 0].get_title()}")
            hist2d(_df, "y_r2", "r2_pred_mean", axes[0, 1], discrete_bins=True); axes[0, 1].set_title(f"R2: Pred vs Obs\n{axes[0, 1].get_title()}")
            hist2d(_df, "r1_pred_mean", "r2_pred_mean", axes[0, 2], discrete_bins=True); axes[0, 2].set_title(f"Pred Mean Consistency\n{axes[0, 2].get_title()}")
            axes[1, 0].hist(_df["r1_res"], bins=100, range=(-25, 25)); axes[1, 0].set_title("R1 Residuals (y - mu)")
            axes[1, 1].hist(_df["r2_res_scaled"], bins=100, range=(-25, 25)); axes[1, 1].set_title("R2 Residuals (scaled)")
            hist2d(_df, "r1_res", "r2_res_scaled", axes[1, 2], discrete_bins=True); axes[1, 2].set_title(f"Residual Consistency\n{axes[1, 2].get_title()}")
            
            axes[2, 0].hist(_df["r1_quant"], bins=50, range=(0, 1)); axes[2, 0].set_title("R1 Quantile (PIT)")
            axes[2, 1].hist(_df["r2_quant"], bins=50, range=(0, 1)); axes[2, 1].set_title("R2 Quantile (PIT)")
            hist2d(_df, "r1_quant", "r2_quant", axes[2, 2], discrete_bins=False, bins=50); axes[2, 2].set_title(f"Quantile Consistency\n{axes[2, 2].get_title()}"); axes[2, 2].set_xlim(0, 1); axes[2, 2].set_ylim(0, 1)

            wandb.log({"validation_plots": wandb.Image(fig)}); plt.close(fig)
            
            return total_loss / n_batches, total_mse / n_batches, res_corr, quant_corr

def load_data(target, log_transform_inputs):
    target_data_dir = DATA_DIR / f"{target}_data_v2"
    data = {"train": {}, "val": {}}
    for fpath in target_data_dir.glob("*.pkl"):
        if "sd_map" in fpath.stem: continue
        with open(fpath, "rb") as f: arr = pkl.load(f).flatten()
        old_shape = arr.shape
        new_shape = old_shape[0] - old_shape[0] % 8
        if "reads" in fpath.stem: arr = arr[:new_shape].reshape(-1, 8).sum(axis=1).flatten()
        else: arr = arr[:new_shape].reshape(-1, 8).mean(axis=1).flatten()
        if "train" in fpath.stem: data["train"][fpath.stem.replace("train_", "")] = arr
        else: data["val"][fpath.stem.replace("val_", "")] = arr

    train_ds = MultiReplicateDataset(**data["train"])
    val_ds = MultiReplicateDataset(**data["val"])

    if log_transform_inputs:
        train_ds.covariates[:, 0] = torch.log1p(train_ds.covariates[:, 0])
        train_ds.covariates[:, 5] = torch.log1p(train_ds.covariates[:, 5])
        val_ds.covariates[:, 0] = torch.log1p(val_ds.covariates[:, 0])
        val_ds.covariates[:, 5] = torch.log1p(val_ds.covariates[:, 5])
        
    return train_ds, val_ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mu_dims", type=int, nargs="+", required=True)
    parser.add_argument("--r_dims", type=int, nargs="+", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--run_group", type=str, required=True)
    parser.add_argument("--consistency_loss", type=str, required=True, choices=["infonce", "pearson", "quantile_abs"])
    parser.add_argument("--consistency_weight", type=float, required=True)
    parser.add_argument("--log_transform_inputs", action="store_true")
    args = parser.parse_args()

    train_ds, val_ds = load_data(args.target, args.log_transform_inputs)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    trainer = TechTrainer_Mu_R(
        train_loader=train_loader, val_loader=val_loader, device=torch.device(args.device),
        consistency_loss=args.consistency_loss, consistency_weight=args.consistency_weight
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.target}_loss_{args.consistency_loss}_log_{args.log_transform_inputs}"
    save_dir = Path(__file__).parent / args.target / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"SAVING TO: {save_dir}")

    trainer.build_and_train(
        covariate_dim=train_ds.get_dim_x(), hidden_dims_mu=tuple(args.mu_dims), hidden_dims_r=tuple(args.r_dims),
        weight_decay=0.01, num_epochs=args.num_epochs, base_lr=args.lr, patience=args.patience,
        warmup_epochs=2, run_name=run_name, run_group=args.run_group, model_save_dir=save_dir
    )