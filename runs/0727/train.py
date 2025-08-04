from __future__ import annotations

import argparse
import copy
import datetime
import logging
import pickle as pkl
import time
from pathlib import Path
from typing import TYPE_CHECKING

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


class TechTrainer_Mu_R:
    def __init__(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: torch.device,
            ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.model: TechNB_mu_r = None  # type: ignore
        self.optimizer: optim.Optimizer = None  # type: ignore
        self.scheduler: SequentialLR = None  # type: ignore
        self.best_val_loss = float("inf")
        self.best_val_mse = float("inf")
        self.best_val_corr = float("-inf")
        self.epochs_no_improve = 0

    def build_and_train(
        self,
        covariate_dim: int,
        hidden_dims_mu: tuple,
        hidden_dims_r: tuple,
        weight_decay: float,
        num_epochs: int,
        device: torch.device,
        model_save_dir: Path,
        base_lr: float,
        patience: int,
        warmup_epochs: int,
        run_name: str,
        run_group: str,
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
            }
        )

        self.model = TechNB_mu_r(
            covariate_dim=covariate_dim,
            hidden_dims_mu=hidden_dims_mu,
            hidden_dims_r=hidden_dims_r
            ).to(device)
        
        wandb.watch(self.model, log="all")

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=weight_decay
        )

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs - warmup_epochs,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        start_time = time.time()
        logger.info("Starting training...")
        for epoch in range(num_epochs):
            self.loader_forward(self.train_loader, train=True, log_freq=1_000)
            
            avg_val_loss, avg_val_mse, avg_val_res_corr, avg_val_quant_corr = self.loader_forward(
                self.val_loader, train=False
            )

            self.scheduler.step()

            # Update wandb summary statistics with the latest validation correlations
            wandb.summary["best_residual_spearman"] = avg_val_res_corr
            wandb.summary["best_quantile_spearman"] = avg_val_quant_corr

            improve = False
            # --- Early Stopping & Model Checkpointing ---
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
                model_save_path = model_save_dir / "best_loss.pt"
                torch.save(best_model_state, model_save_path)
                logger.info(
                    "New best validation loss: %s. Model saved to %s",
                    self.best_val_loss,
                    model_save_path,
                )
                improve = True

            if avg_val_res_corr > self.best_val_corr:
                self.best_val_corr = avg_val_res_corr
                epochs_no_improve = 0
                best_corr_model_state = copy.deepcopy(self.model.state_dict())
                model_save_path = model_save_dir / "best_corr.pt"
                torch.save(best_corr_model_state, model_save_path)
                logger.info(
                    "New best validation residual correlation: %s. Model saved to %s",
                    self.best_val_corr,
                    model_save_path,
                )
                improve = True

            if not improve:
                epochs_no_improve += 1
                logger.info(
                    "Validation performance did not improve for %s epoch(s).",
                    epochs_no_improve,
                )

            if epochs_no_improve >= patience:
                logger.info(
                    "Early stopping triggered after %s epochs due to no improvement for %s epochs.",
                    epoch + 1,
                    patience,
                )
                break

        logger.info(
            "Training finished. Total time: %s seconds",
            time.time() - start_time,
        )

    def loader_forward(
            self,
            loader: DataLoader,
            train: bool,
            log_freq: int = 1_000,
        ):
        total_loss = 0
        total_mse = 0
        n_batches = 0

        if train:
            self.model.train()
        else:
            self.model.eval()
            # Lists to store results from all validation batches
            all_r1_res, all_r2_res_scaled, all_r1_quant, all_r2_quant = [], [], [], []
            all_y_r1, all_r1_pred_mean, all_y_r2, all_r2_pred_mean = [], [], [], []
            all_r1_pred_mean_val, all_r2_pred_mean_val = [], []

        cov_dim = loader.dataset.get_dim_x()
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            if train:
                self.optimizer.zero_grad()

            x_r1 = x[:, :cov_dim]
            x_r2 = x[:, cov_dim:2*cov_dim]
            y_r1 = y[:, 0]
            y_r2 = y[:, 1]
            sd_ratio_r1_r2 = y[:, 2]

            with torch.set_grad_enabled(train):
                r1_dist = self.model.predict_dist(x_r1)
                r2_dist = self.model.predict_dist(x_r2)
                r1_quant = compute_numeric_cdf(r1_dist, y_r1)
                r2_quant = compute_numeric_cdf(r2_dist, y_r2)
                r1_ll = r1_dist.log_prob(y_r1).mean()
                r2_ll = r2_dist.log_prob(y_r2).mean()

                r1_residual = y_r1 - r1_dist.mean
                r2_residual_unscaled = y_r2 - r2_dist.mean
                r2_residual_scaled = sd_ratio_r1_r2 * r2_residual_unscaled

                mse = torch.mean((r1_residual - r2_residual_scaled) ** 2)
                quant_loss = torch.mean(torch.abs(r1_quant - r2_quant))
                
                loss = -(r1_ll + r2_ll) + 10 * quant_loss

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            total_mse += mse.item()
            n_batches += 1

            if train and (batch_idx % log_freq == 0):
                logger.info(
                    "Batch %s/%s, Train Loss: %s, LL R1: %s, LL R2: %s, Quant Loss: %s",
                    batch_idx, len(loader), loss.item(), r1_ll.item(), r2_ll.item(), quant_loss.item()
                )
            
            if not train:
                all_r1_res.append(r1_residual.detach().cpu().numpy())
                all_r2_res_scaled.append(r2_residual_scaled.detach().cpu().numpy())
                all_r1_quant.append(r1_quant.detach().cpu().numpy())
                all_r2_quant.append(r2_quant.detach().cpu().numpy())
                all_y_r1.append(y_r1.detach().cpu().numpy())
                all_r1_pred_mean.append(r1_dist.mean.detach().cpu().numpy())
                all_y_r2.append(y_r2.detach().cpu().numpy())
                all_r2_pred_mean.append(r2_dist.mean.detach().cpu().numpy())
                all_r1_pred_mean_val.append(r1_dist.mean.detach().cpu().numpy())
                all_r2_pred_mean_val.append(r2_dist.mean.detach().cpu().numpy())

        if train:
            wandb.log({
                "train_loss": total_loss / n_batches,
                "train_mse": total_mse / n_batches,
            })
            return total_loss / n_batches, total_mse / n_batches, np.nan, np.nan
        else:
            _df = pd.DataFrame({
                "r1_res": np.concatenate(all_r1_res),
                "r2_res_scaled": np.concatenate(all_r2_res_scaled),
                "r1_quant": np.concatenate(all_r1_quant),
                "r2_quant": np.concatenate(all_r2_quant),
                "y_r1": np.concatenate(all_y_r1),
                "r1_pred_mean": np.concatenate(all_r1_pred_mean),
                "y_r2": np.concatenate(all_y_r2),
                "r2_pred_mean": np.concatenate(all_r2_pred_mean),
                "r1_pred_mean_val": np.concatenate(all_r1_pred_mean_val),
                "r2_pred_mean_val": np.concatenate(all_r2_pred_mean_val),
            })
            
            overall_residual_spearman = _df.corr(method='spearman').loc['r1_res', 'r2_res_scaled']
            overall_quantile_spearman = _df.corr(method='spearman').loc['r1_quant', 'r2_quant']

            wandb.log({
                "val_loss": total_loss / n_batches,
                "val_mse": total_mse / n_batches,
                "val_residual_spearman": overall_residual_spearman,
                "val_quantile_spearman": overall_quantile_spearman,
            })

            fig, axes = plt.subplots(3, 3, figsize=(18, 18))
            fig.tight_layout(pad=5.0)

            # Row 0: Model Fit
            hist2d(_df, "y_r1", "r1_pred_mean", axes[0, 0], discrete_bins=True)
            axes[0, 0].set_title(f"R1: Pred vs Obs\n{axes[0, 0].get_title()}")
            hist2d(_df, "y_r2", "r2_pred_mean", axes[0, 1], discrete_bins=True)
            axes[0, 1].set_title(f"R2: Pred vs Obs\n{axes[0, 1].get_title()}")
            hist2d(_df, "r1_pred_mean_val", "r2_pred_mean_val", axes[0, 2], discrete_bins=True)
            axes[0, 2].set_title(f"Pred Mean Consistency\n{axes[0, 2].get_title()}")

            # Row 1: Residual Analysis
            axes[1, 0].hist(_df["r1_res"], bins=100, range=(-25, 25))
            axes[1, 0].set_title("R1 Residuals (y - mu)")
            axes[1, 1].hist(_df["r2_res_scaled"], bins=100, range=(-25, 25))
            axes[1, 1].set_title("R2 Residuals (scaled)")
            hist2d(_df, "r1_res", "r2_res_scaled", axes[1, 2], discrete_bins=True)
            axes[1, 2].set_title(f"Residual Consistency\n{axes[1, 2].get_title()}")

            # Row 2: Quantile (PIT) Analysis
            axes[2, 0].hist(_df["r1_quant"], bins=50, range=(0, 1))
            axes[2, 0].set_title("R1 Quantile (PIT)")
            axes[2, 1].hist(_df["r2_quant"], bins=50, range=(0, 1))
            axes[2, 1].set_title("R2 Quantile (PIT)")
            hist2d(_df, "r1_quant", "r2_quant", axes[2, 2], discrete_bins=False, bins=50)
            axes[2, 2].set_title(f"Quantile Consistency\n{axes[2, 2].get_title()}")
            axes[2, 2].set_xlim(0, 1)
            axes[2, 2].set_ylim(0, 1)

            wandb.log({"validation_plots": wandb.Image(fig)})
            plt.close(fig)
            
            return total_loss / n_batches, total_mse / n_batches, overall_residual_spearman, overall_quantile_spearman


def load_data(target):
    target_data_dir = DATA_DIR / f"{target}_data_v2"
    data = {"train": {}, "val": {}}
    for fpath in target_data_dir.glob("*.pkl"):
        if "sd_map" in fpath.stem:
            continue
        with open(fpath, "rb") as f:
            arr = pkl.load(f).flatten()
            
        old_shape = arr.shape
        new_shape = old_shape[0] - old_shape[0] % 8
        print(f"FPATH: {fpath.stem}, {old_shape} -> {new_shape}")
        if "reads" in fpath.stem:
            arr = arr[:new_shape].reshape(-1, 8).sum(axis=1).flatten()
        else:
            arr = arr[:new_shape].reshape(-1, 8).mean(axis=1).flatten()
        if "train" in fpath.stem:
            data["train"][fpath.stem.replace("train_", "")] = arr
        else:
            data["val"][fpath.stem.replace("val_", "")] = arr

    return MultiReplicateDataset(**data["train"]), MultiReplicateDataset(**data["val"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--mu_dims", type=int, nargs="+", required=True)
    parser.add_argument("--r_dims", type=int, nargs="+", required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--warmup_epochs", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--run_group", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)

    args = parser.parse_args()
    train_ds, val_ds = load_data(args.target)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    trainer = TechTrainer_Mu_R(
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device(args.device),
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.target}_mu_{'_'.join(map(str, args.mu_dims))}_r_{'_'.join(map(str, args.r_dims))}_wd_{args.weight_decay}_lr_{args.lr}_warmup_{args.warmup_epochs}"

    save_dir = Path(__file__).parent / args.target / run_name
    save_dir.mkdir(parents=True)
    print(f"SAVING TO: {save_dir}")


    trainer.build_and_train(
            covariate_dim=train_ds.get_dim_x(),
            hidden_dims_mu=args.mu_dims,
            hidden_dims_r=args.r_dims,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            device=torch.device(args.device),
            model_save_dir=save_dir,
            base_lr=args.lr,
            patience=args.patience,
            warmup_epochs=args.warmup_epochs,
            run_name=run_name,
            run_group=args.run_group,
            )
