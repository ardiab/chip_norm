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
            avg_train_loss, avg_train_mse, avg_train_corr = self.loader_forward(
                self.train_loader, train=True, log_freq=1_000
            )
            avg_val_loss, avg_val_mse, avg_val_corr = self.loader_forward(
                self.val_loader, train=False, log_freq=1_000
            )

            self.scheduler.step()
            epoch_duration = time.time() - start_time

            improve = False
            # --- Early Stopping & Model Checkpointing ---
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # Save the best model state
                best_model_state = copy.deepcopy(self.model.state_dict())
                model_save_path = model_save_dir / "best_loss.pt"
                torch.save(best_model_state, model_save_path)
                logger.info(
                    "New best validation loss: %s. Model saved to %s",
                    self.best_val_loss,
                    model_save_path,
                )
                improve = True
            if avg_val_mse < self.best_val_mse:
                self.best_val_mse = avg_val_mse
                epochs_no_improve = 0
                # Save the best model state
                best_mse_model_state = copy.deepcopy(self.model.state_dict())
                model_save_path = model_save_dir / "best_mse.pt"
                torch.save(best_mse_model_state, model_save_path)
                logger.info(
                    "New best validation MSE: %s. Model saved to %s",
                    self.best_val_mse,
                    model_save_path,
                )
                improve = True
            if avg_val_corr > self.best_val_corr:
                self.best_val_corr = avg_val_corr
                epochs_no_improve = 0
                # Save the best model state
                best_corr_model_state = copy.deepcopy(self.model.state_dict())
                model_save_path = model_save_dir / "best_corr.pt"
                torch.save(best_corr_model_state, model_save_path)
                logger.info(
                    "New best validation corr: %s. Model saved to %s",
                    self.best_val_corr,
                    model_save_path,
                )
                improve = True

            if not improve:
                epochs_no_improve += 1
                logger.info(
                    "Validation log-likelihood did not improve for %s epoch(s).",
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
            "Phase A training finished. Total time: %s seconds",
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
        all_mse_corrs = []
        all_quant_corrs = []
        n_batches = 0

        if train:
            self.model.train()
        else:
            self.model.eval()

        for batch_idx, (x, y) in enumerate(loader):
            batch_corrs = []
            x, y = x.to(self.device), y.to(self.device)
            if train:
                self.optimizer.zero_grad()

            x_r1 = x[:, :5]
            x_r2 = x[:, 5:10]
            grp_idx = x[:, 10]
            y_r1 = y[:, 0]
            y_r2 = y[:, 1]
            sd_ratio_r1_r2 = y[:, 2]

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
            quant_loss = - torch.corrcoef(torch.stack([r1_quant, r2_quant]))[0, 1]
            total_mse += mse.item()
            for grp_idx_val in grp_idx.unique():
                try:
                    grp_idx_mask = grp_idx == grp_idx_val
                    r1_sub = r1_residual[grp_idx_mask].squeeze()
                    r2_sub = r2_residual_scaled[grp_idx_mask].squeeze()
                    corr = torch.corrcoef(torch.stack([r1_sub, r2_sub]))
                    all_mse_corrs.append(corr[0, 1].item())
                    all_quant_corrs.append(quant_loss.item())
                    batch_corrs.append(corr[0, 1].item())
                except IndexError:  # Sometimes get 1 sample from a group by chance
                    continue

            loss = -(r1_ll + r2_ll) + 5 * quant_loss
            loss.backward()
            total_loss += loss.item()

            if train:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            n_batches += 1

            if (batch_idx % log_freq == 0) and train:
                logger.info(
                    "Batch %s/%s, Loss: %s, LL R1: %s, LL R2: %s, MSE: %s, Res corr: %s, Res corr std: %s, Quant corr: %s, Quant corr std: %s",
                    batch_idx,
                    len(loader),
                    loss.item(),
                    r1_ll.item(),
                    r2_ll.item(),
                    mse.item(),
                    np.mean(batch_corrs),
                    np.std(batch_corrs),
                    np.mean(all_quant_corrs),
                    np.std(all_quant_corrs),
                )

        if not train:
            logger.info(
                "Validation loss: %s, Validation MSE: %s, Validation corr: %s",
                total_loss / n_batches,
                total_mse / n_batches,
                np.mean(all_mse_corrs),
            )
            wandb.log({
                "val_loss": total_loss / n_batches,
                "val_mse": total_mse / n_batches,
                "val_corr": np.mean(all_mse_corrs),
            })
            r1_res_np = r1_residual.detach().cpu().numpy()
            r2_res_np = r2_residual_unscaled.detach().cpu().numpy()
            y_r1_np = y_r1.detach().cpu().numpy()
            y_r2_np = y_r2.detach().cpu().numpy()
            x_r1_np = x_r1.detach().cpu().numpy()
            x_r2_np = x_r2.detach().cpu().numpy()
            r1_dist_np = r1_dist.mean.detach().cpu().numpy()
            r2_dist_np = r2_dist.mean.detach().cpu().numpy()
            r1_std_np = r1_dist.stddev.detach().cpu().numpy()
            r2_std_np = r2_dist.stddev.detach().cpu().numpy()
            r1_quant = compute_numeric_cdf(r1_dist, torch.tensor(y_r1_np, device=self.device)).detach().cpu().numpy()
            r2_quant = compute_numeric_cdf(r2_dist, torch.tensor(y_r2_np, device=self.device)).detach().cpu().numpy()

            fig, axes = plt.subplots(4, 4, figsize=(20, 20))

            _df = pd.DataFrame({
                "r1_res": r1_res_np,
                "r2_res": r2_res_np,
                "y_r1": y_r1_np,
                "y_r2": y_r2_np,
                "ctrl_r1": x_r1_np[:, 0],
                "ctrl_r2": x_r2_np[:, 0],
                "r1_pred_mean": r1_dist_np,
                "r1_pred_std": r1_std_np,
                "r2_pred_mean": r2_dist_np,
                "r2_pred_std": r2_std_np,
                "r1_quant": r1_quant,
                "r2_quant": r2_quant,
            })
            _df["r1_res_clipped"] = _df["r1_res"].clip(lower=0)
            _df["r2_res_clipped"] = _df["r2_res"].clip(lower=0)
            _df = _df.sort_values(by="r1_pred_mean", ascending=False)

            axes[0, 0].hist(r1_res_np, bins=100)
            axes[0, 0].set_title("R1 Residuals")
            axes[0, 1].hist(r2_res_np, bins=100)
            axes[0, 1].set_title("R2 Residuals")
            hist2d(_df, "r1_res", "r2_res", axes[0, 2])
            hist2d(_df, "r1_res_clipped", "r2_res_clipped", axes[0, 3])

            axes[1, 0].hist(y_r1_np, bins=100)
            axes[1, 0].set_title("R1 (exp reads)")
            axes[1, 1].hist(_df["ctrl_r1"].values, bins=100)
            axes[1, 1].set_title("Ctrl R1")
            hist2d(_df, "y_r1", "ctrl_r1", axes[1, 2])
            hist2d(_df, "y_r1", "r1_pred_mean", axes[1, 3])

            axes[2, 0].hist(y_r2_np, bins=100)
            axes[2, 0].set_title("R2 (exp reads)")
            axes[2, 1].hist(_df["ctrl_r2"].values, bins=100)
            axes[2, 1].set_title("Ctrl R2")
            hist2d(_df, "y_r2", "ctrl_r2", axes[2, 2])
            hist2d(_df, "y_r2", "r2_pred_mean", axes[2, 3])

            hist2d(_df, "r1_quant", "r2_quant", axes[3, 0], discrete_bins=False)
            axes[3, 1].hist(_df["r1_quant"].values, bins=100)
            axes[3, 1].set_title("R1 Quant")
            axes[3, 2].hist(_df["r2_quant"].values, bins=100)
            axes[3, 2].set_title("R2 Quant")
            axes[3, 3].plot(np.arange(_df.shape[0]), _df["r1_pred_mean"].values, label="R1")
            axes[3, 3].fill_between(
                np.arange(_df.shape[0]),
                _df["r1_pred_mean"].values - _df["r1_pred_std"].values,  # type: ignore
                _df["r1_pred_mean"].values + _df["r1_pred_std"].values,  # type: ignore
                alpha=0.5,
                label="R1 std",
            )
            axes[3, 3].plot(np.arange(_df.shape[0]), _df["r2_pred_mean"].values, label="R2")
            axes[3, 3].fill_between(
                np.arange(_df.shape[0]),
                _df["r2_pred_mean"].values - _df["r2_pred_std"].values,  # type: ignore
                _df["r2_pred_mean"].values + _df["r2_pred_std"].values,  # type: ignore
                alpha=0.5,
                label="R2 std",
            )
            axes[3, 3].legend()
            axes[3, 3].set_title("R1 vs R2")

            wandb.log({
                "transform_vis": wandb.Image(fig),
            })
            plt.close(fig)

            wandb.log({
                "df": wandb.Table(dataframe=_df),
            })
        else:
            wandb.log({
                "train_loss": total_loss / n_batches,
                "train_mse": total_mse / n_batches,
                "train_corr": np.mean(all_mse_corrs),
            })

        return total_loss / n_batches, total_mse / n_batches, np.mean(all_mse_corrs)


def load_data(target):
    target_data_dir = DATA_DIR / f"{target}_data_v2"
    data = {"train": {}, "val": {}}
    for fpath in target_data_dir.glob("*.pkl"):
        if "sd_map" in fpath.stem:
            continue
        with open(fpath, "rb") as f:
            arr = pkl.load(f).flatten()
            # data["train"][fpath.stem.replace("train_", "")] = pkl.load(f)
        #     print(f"Loaded {fpath.stem} as train")
        # elif "val" in fpath.stem:
        #     with open(fpath, "rb") as f:
        #         arr = pkl.load(f).flatten()
        #         # data["val"][fpath.stem.replace("val_", "")] = pkl.load(f)
            
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
