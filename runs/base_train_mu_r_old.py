from __future__ import annotations

import argparse
import copy
import logging
import pickle as pkl
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from chipvi.data.datasets import MultiReplicateDataset
from chipvi.models.technical_model import TechNB_mu_r, TechNB_r_p

if TYPE_CHECKING:
    from pathlib import Path

    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


DATA_DIR = Path("/lotterlab/users/abdul/repos/chipvi/")


def build_and_train(
    dim_x: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    hidden_dims_r: tuple,
    weight_decay: float,  # TODO: Adjust?
    num_epochs: int,
    device: torch.device,
    model_save_dir: Path,
    base_lr: float,
    nb_form: Literal["mu_r", "r_p"],
    hidden_dims_mu: tuple | None = None,
    hidden_dims_p: tuple | None = None,
    patience: int = 10,
    warmup_epochs: int = 2,
    ) -> None:

    if nb_form == "mu_r":
        if hidden_dims_mu is None:
            raise ValueError("hidden_dims_mu must be provided for mu_r")
        model = TechNB_mu_r(
            dim_x=dim_x,
            hidden_dims_mu=hidden_dims_mu,
            hidden_dims_r=hidden_dims_r
            ).to(device)
    elif nb_form == "r_p":
        if hidden_dims_p is None:
            raise ValueError("hidden_dims_p must be provided for r_p")
        model = TechNB_r_p(
            dim_x=dim_x,
            hidden_dims_r=hidden_dims_r,
            hidden_dims_p=hidden_dims_p,
            ).to(device)
    else:
        raise ValueError(f"Invalid nb_form: {nb_form}")
    
    # TODO: AdamW?
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)


    warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs,
            )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs])

    best_val_log_likelihood = float("-inf")
    best_val_biol_mse = float("inf")
    best_val_biol_corr = float("-inf")
    epochs_no_improve = 0

    start_time = time.time()
    logger.info("Starting training...")
    for epoch in range(num_epochs):

        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        train_batches = 0

        for batch_idx, (x_i, y_i) in enumerate(train_loader):
            x_i, y_i = x_i.to(device), y_i.to(device).float()  # noqa: PLW2901

            optimizer.zero_grad()

            train_x_r1 = x_i[:, :5]
            train_x_r2 = x_i[:, 5:]
            train_y_r1 = y_i[:, 0]
            train_y_r2 = y_i[:, 1]
            train_sd_ratio_r1_r2 = y_i[:, 2]

            # TODO: Make sure computational graph is getting these
            train_r1_dist = model.predict_dist(train_x_r1)
            train_r2_dist = model.predict_dist(train_x_r2)

            train_ll_r1 = train_r1_dist.log_prob(train_y_r1).mean()
            train_ll_r2 = train_r2_dist.log_prob(train_y_r2).mean()

            # If replicate 1 has 2x the seduencing depth, we assume it will have ~2x the biological signal.
            train_r1_res = train_y_r1 - train_r1_dist.mean
            train_r2_res_unscaled = train_y_r2 - train_r2_dist.mean
            train_r2_res_scaled = train_sd_ratio_r1_r2 * train_r2_res_unscaled
            
            train_mse_unscaled = torch.mean((train_r1_res - train_r2_res_unscaled) ** 2)
            train_mse_scaled = torch.mean((train_r1_res - train_r2_res_scaled) ** 2)
            # TODO: Use torchmetrics, group by SD?
            neg_corr = negative_pearson_loss(train_r1_res, train_r2_res)
            loss = -(train_ll_r1 + train_ll_r2)

            loss.backward()

            # TODO: Where are NaN vals coming from?
            with torch.no_grad():
                grad_mins = []
                grad_maxs = []
                grad_means = []
                grad_norms = []

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        g = param.grad.detach()
                        grad_mins.append(g.min().item())
                        grad_maxs.append(g.max().item())
                        grad_means.append(g.mean().item())
                        grad_norms.append(g.norm().item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

            if batch_idx % 1_000 == 0:  # Log every 1,000 batches
                logger.info(
                    "Epoch %s/%s, Batch %s/%s, Train Loss: %s. LL R1: %s, LL R2: %s, "
                    "Res delta (unscaled): %s, Res delta (scaled): %s, Res corr: %s",
                    epoch + 1,
                    num_epochs,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                    train_ll_r1.item(),
                    train_ll_r2.item(),
                    train_mse_unscaled.item(),
                    train_mse_scaled.item(),
                    -neg_corr.item(),
                )
                logger.info(
                    "[Grad Stats Before Clipping] Min: %s, Max: %s, Mean: %s, Total Norm: %s",
                    min(grad_mins),
                    max(grad_maxs),
                    sum(grad_means)/len(grad_means),
                    sum(grad_norms),
                )

        avg_train_loss = total_train_loss / train_batches

        model.eval()
        total_val_log_likelihood = 0
        total_val_biol_mse = 0
        total_val_biol_corr = 0
        val_batches = 0

        with torch.no_grad():
            for batch_idx, (x_i_val, y_i_val) in enumerate(val_loader):
                x_i_val, y_i_val = x_i_val.to(device), y_i_val.to(device).float()  # noqa: PLW2901
                x_i_val_r1 = x_i_val[:, :5]
                x_i_val_r2 = x_i_val[:, 5:]
                y_i_val_r1 = y_i_val[:, 0]
                y_i_val_r2 = y_i_val[:, 1]
                train_sd_ratio_r1_r2 = y_i_val[:, 2]

                val_r1_dist = model.predict_dist(x_i_val_r1)
                val_r2_dist = model.predict_dist(x_i_val_r2)

                val_ll_r1 = val_r1_dist.log_prob(y_i_val_r1).mean()
                val_ll_r2 = val_r2_dist.log_prob(y_i_val_r2).mean()

                val_r1_res = y_i_val_r1 - val_r1_dist.mean
                val_r2_res = train_sd_ratio_r1_r2 * (y_i_val_r2 - val_r2_dist.mean)

                # TODO
                mse = torch.mean((val_r1_res - val_r2_res) ** 2)
                neg_corr = negative_pearson_loss(val_r1_res, val_r2_res)

                total_val_log_likelihood += val_ll_r1.item()
                total_val_log_likelihood += val_ll_r2.item()
                total_val_biol_mse += mse.item()
                total_val_biol_corr += -neg_corr.item()

                val_batches += 1

            avg_val_log_likelihood = total_val_log_likelihood / val_batches
            avg_val_biol_mse = total_val_biol_mse / val_batches
            avg_val_biol_corr = total_val_biol_corr / val_batches

        scheduler.step()
        epoch_duration = time.time() - start_time

        logger.info(
            "Epoch %s/%s Summary: Duration: %s, Avg Train Loss: %s, Avg Val Log-Likelihood: %s, Avg Val Biol MSE: %s, Avg Val Biol Corr: %s",
            epoch + 1,
            num_epochs,
            epoch_duration,
            avg_train_loss,
            avg_val_log_likelihood,
            avg_val_biol_mse,
            avg_val_biol_corr,
        )

        improve = False
        # --- Early Stopping & Model Checkpointing ---
        if avg_val_log_likelihood > best_val_log_likelihood:
            best_val_log_likelihood = avg_val_log_likelihood
            epochs_no_improve = 0
            # Save the best model state
            best_ll_model_state = copy.deepcopy(model.state_dict())
            model_save_path = model_save_dir / "best_likelihood.pt"
            torch.save(best_ll_model_state, model_save_path)
            logger.info(
                "New best validation log-likelihood: %s. Model saved to %s",
                best_val_log_likelihood,
                model_save_path,
            )
            improve = True
        if avg_val_biol_mse < best_val_biol_mse:
            best_val_biol_mse = avg_val_biol_mse
            epochs_no_improve = 0
            # Save the best model state
            best_mse_model_state = copy.deepcopy(model.state_dict())
            model_save_path = model_save_dir / "best_mse.pt"
            torch.save(best_mse_model_state, model_save_path)
            logger.info(
                "New best validation biol MSE: %s. Model saved to %s",
                best_val_biol_mse,
                model_save_path,
            )
            improve = True
            if avg_val_biol_corr > best_val_biol_corr:
                best_val_biol_corr = avg_val_biol_corr
            epochs_no_improve = 0
            # Save the best model state
            best_corr_model_state = copy.deepcopy(model.state_dict())
            model_save_path = model_save_dir / "best_corr.pt"
            torch.save(best_corr_model_state, model_save_path)
            logger.info(
                "New best validation biol corr: %s. Model saved to %s",
                best_val_biol_corr,
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


def negative_pearson_loss(preds, targets, eps=1e-8):
    preds = preds - preds.mean()
    targets = targets - targets.mean()
    
    numerator = (preds * targets).sum()
    denominator = torch.sqrt((preds**2).sum() * (targets**2).sum() + eps)
    
    return -numerator / denominator


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    args = parser.parse_args()

    train_ds, val_ds = load_data(args.target)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8_192, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8_192, shuffle=False)

    save_dir = DATA_DIR / f"runs/{}/{args.target}"
    save_dir.mkdir(parents=False)
    print(f"SAVING TO: {save_dir}")


    build_and_train(
            dim_x=train_ds.get_dim_x(),
            train_loader=train_loader,
            val_loader=val_loader,
            hidden_dims_mu=(32, 32),
            hidden_dims_r=(8, 8),
            weight_decay=0.01,
            num_epochs=100,
            device=torch.device("cuda:0"),
            model_save_dir=save_dir,
            base_lr=0.001,
            max_lr=0.01,
            patience=10,
            )
