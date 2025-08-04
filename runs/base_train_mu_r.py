from __future__ import annotations

import argparse
import copy
import logging
import pickle as pkl
import time
from pathlib import Path
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from chipvi.data.datasets import MultiReplicateDataset
from chipvi.models.technical_model import TechNB_mu_r
import wandb

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
        self.best_val_biol_mse = float("inf")
        self.best_val_biol_corr = float("-inf")
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
        all_corrs = []
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
            r1_ll = r1_dist.log_prob(y_r1).mean()
            r2_ll = r2_dist.log_prob(y_r2).mean()

            r1_residual = y_r1 - r1_dist.mean
            r2_residual_unscaled = y_r2 - r2_dist.mean
            r2_residual_scaled = sd_ratio_r1_r2 * r2_residual_unscaled

            mse = torch.mean((r1_residual - r2_residual_scaled) ** 2)
            total_mse += mse.item()
            for grp_idx_val in grp_idx.unique():
                grp_idx_mask = grp_idx == grp_idx_val
                r1_sub = r1_residual[grp_idx_mask].squeeze()
                r2_sub = r2_residual_scaled[grp_idx_mask].squeeze()
                print(r1_sub.shape, r2_sub.shape)
                stack = torch.stack([r1_sub, r2_sub])
                print(stack.shape)
                corr = torch.corrcoef(
                    stack,
                )
                all_corrs.append(corr[0, 1].item())
                batch_corrs.append(corr[0, 1].item())

            loss = -(r1_ll + r2_ll)
            loss.backward()
            total_loss += loss.item()

            if train:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            n_batches += 1

            if (batch_idx % log_freq == 0) and train:
                logger.info(
                    "Batch %s/%s, Loss: %s, LL R1: %s, LL R2: %s, Res delta (unscaled): %s, Res delta (scaled): %s, Res corr: %s",
                    batch_idx,
                    len(loader),
                    loss.item(),
                    r1_ll.item(),
                    r2_ll.item(),
                    mse.item(),
                    torch.mean(torch.stack(batch_corrs)),
                )

        if not train:
            logger.info(
                "Validation loss: %s, Validation MSE: %s, Validation corr: %s",
                total_loss / n_batches,
                total_mse / n_batches,
                np.mean(all_corrs),
            )
            wandb.log({
                "val_loss": total_loss / n_batches,
                "val_mse": total_mse / n_batches,
                "val_corr": np.mean(all_corrs),
            })
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes[0, 0].hist(r1_residual.cpu().numpy(), bins=100)
            axes[0, 0].set_title("R1 Residuals")
            axes[0, 1].hist(r2_residual_unscaled.cpu().numpy(), bins=100)
            axes[0, 1].set_title("R2 Residuals (unscaled)")
            axes[0, 2].hist(r2_residual_scaled.cpu().numpy(), bins=100)
            axes[0, 2].set_title("R2 Residuals (scaled)")
            axes[0, 3].scatter(r1_residual.cpu().numpy(), r2_residual_scaled.cpu().numpy())
            axes[0, 3].set_title("R1 Residuals vs R2 Residuals (scaled)")
            axes[1, 0].scatter(y_r1.cpu().numpy(), y_r2.cpu().numpy())
            axes[1, 0].set_title("R1 (obs) vs R1 (obs)")
            axes[1, 1].scatter(y_r1.cpu().numpy(), r1_dist.mean.cpu().numpy())
            axes[1, 1].set_title("R1 (obs) vs R1 (pred)")
            axes[1, 2].scatter(y_r1.cpu().numpy(), x_r1[:, 0].cpu().numpy())
            axes[1, 2].set_title("R1 (obs) vs R1 (ctrl)")
            axes[1, 3].scatter(r1_dist.mean.cpu().numpy(), x_r1[:, 0].cpu().numpy())
            axes[1, 3].set_title("R1 (pred) vs R1 (ctrl)")
            wandb.log({
                "residuals": wandb.Image(fig),
            })
            val_table = wandb.Table(columns=[
               "R1 (ctrl reads)",
               "R1 (ctrl mapq)",
               "R1 (ctrl seq depth)",
               "R1 (exp mapq)",
               "R1 (exp seq depth)",
               "R2 (ctrl reads)",
               "R2 (ctrl mapq)",
               "R2 (ctrl seq depth)",
               "R2 (exp mapq)",
               "R2 (exp seq depth)",
               "R2 (exp reads)",
               "R2 (exp reads) / R1 (exp reads)",
               "R1 (pred mean)",
               "R2 (pred mean)",
               "R1 (pred std)",
               "R2 (pred std)",
            ])
            val_table.add_data(
                x_r1[:, 0].cpu().numpy(), x_r1[:, 1].cpu().numpy(), x_r1[:, 2].cpu().numpy(), x_r1[:, 3].cpu().numpy(), x_r1[:, 4].cpu().numpy(),
                x_r2[:, 0].cpu().numpy(), x_r2[:, 1].cpu().numpy(), x_r2[:, 2].cpu().numpy(), x_r2[:, 3].cpu().numpy(), x_r2[:, 4].cpu().numpy(),
                y_r2.cpu().numpy(), y_r2.cpu().numpy() / y_r1.cpu().numpy(),
                r1_dist.mean.cpu().numpy(), r2_dist.mean.cpu().numpy(),
                r1_dist.stddev.cpu().numpy(), r2_dist.stddev.cpu().numpy(),
            )
            wandb.log({
                "val_table": val_table,
            })
        else:
            wandb.log({
                "train_loss": total_loss / n_batches,
                "train_mse": total_mse / n_batches,
                "train_corr": np.mean(all_corrs),
            })

        return total_loss / n_batches, total_mse / n_batches, np.mean(all_corrs)


def load_data(target):
    target_data_dir = DATA_DIR / f"{target}_data_v2"
    data = {"train": {}, "val": {}}
    for fpath in target_data_dir.glob("*.pkl"):
        if "sd_map" in fpath.stem:
            continue
        if "train" in fpath.stem:
            with open(fpath, "rb") as f:
                data["train"][fpath.stem.replace("train_", "")] = pkl.load(f)
            print(f"Loaded {fpath.stem} as train")
        elif "val" in fpath.stem:
            with open(fpath, "rb") as f:
                data["val"][fpath.stem.replace("val_", "")] = pkl.load(f)
            print(f"Loaded {fpath.stem} as val")

    return MultiReplicateDataset(**data["train"]), MultiReplicateDataset(**data["val"])


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--target", type=str, required=True)
#     parser.add_argument("--device", type=str, required=True)
#     args = parser.parse_args()
#     train_ds, val_ds = load_data(args.target)

#     train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8_192, shuffle=False)
#     val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8_192, shuffle=False)

#     trainer = TechTrainer_Mu_R(
#         train_loader=train_loader,
#         val_loader=val_loader,
#         device=torch.device(args.device),
#     )

#     save_dir = DATA_DIR / f"runs/{}/{args.target}"
#     save_dir.mkdir(parents=False)
#     print(f"SAVING TO: {save_dir}")


#     trainer.build_and_train(
#             covariate_dim=train_ds.get_dim_x(),
#             hidden_dims_mu=(32, 32),
#             hidden_dims_r=(8, 8),
#             weight_decay=0.01,
#             num_epochs=100,
#             device=torch.device("cuda:0"),
#             model_save_dir=save_dir,
#             base_lr=0.001,
#             patience=10,
#             warmup_epochs=2,
#             run_name="",
#             run_group="",
#             )
