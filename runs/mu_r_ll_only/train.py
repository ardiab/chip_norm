from __future__ import annotations

import argparse
import copy  # For saving best model state
import logging
import pickle as pkl
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import wandb
from torch import optim
from torch.optim.lr_scheduler import CyclicLR

from chipvi.data.datasets import MultiReplicateDataset
from chipvi.models.technical_model import TechNB_mu_r
from chipvi.utils.distributions import nb_log_prob_mu_r

if TYPE_CHECKING:
    from pathlib import Path

    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


DATA_DIR = Path("/lotterlab/users/abdul/repos/chipvi/")


def build_and_train(
    dim_x: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    hidden_dims_mean: tuple,
    hidden_dims_disp: tuple,
    weight_decay: float,
    num_epochs: int,
    device: torch.device,
    model_save_dir: Path,
    base_lr: float,
    max_lr: float,
    patience: int = 10,
    wandb_name: str | None = None,
    ) -> None:

    model = TechNB_mu_r(
            dim_x=dim_x,
            hidden_dims_mu=hidden_dims_mean,
            hidden_dims_r=hidden_dims_disp
            ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    scheduler = CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        mode="triangular2",
        cycle_momentum=False,
        step_size_up=int(len(train_loader) / 2),
    )

    best_val_log_likelihood = float("-inf")
    best_val_biol_mse = float("inf")
    best_val_biol_corr = float("-inf")
    epochs_no_improve = 0

    start_time = time.time()
    logger.info("Starting Phase A training...")
    for epoch in range(num_epochs):

        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        train_batches = 0

        for batch_idx, (x_i, y_i) in enumerate(train_loader):
            x_i, y_i = x_i.to(device), y_i.to(device).float()  # noqa: PLW2901

            optimizer.zero_grad()

            x_i_r1 = x_i[:, :5]
            x_i_r2 = x_i[:, 5:]
            y_i_r1 = y_i[:, 0]
            y_i_r2 = y_i[:, 1]
            sd_ratio_r1_to_r2 = y_i[:, 2]

            log_mu_tech_r1, log_r_tech_r1 = model(x_i_r1)
            log_mu_tech_r1 = log_mu_tech_r1.squeeze()
            log_r_tech_r1 = log_r_tech_r1.squeeze()
            log_mu_tech_r2, log_r_tech_r2 = model(x_i_r2)
            log_mu_tech_r2 = log_mu_tech_r2.squeeze()
            log_r_tech_r2 = log_r_tech_r2.squeeze()
            r_tech_r1 = torch.exp(log_r_tech_r1)
            r_tech_r2 = torch.exp(log_r_tech_r2)

            mu_tech_r1 = torch.exp(log_mu_tech_r1)
            mu_tech_r2 = torch.exp(log_mu_tech_r2)

            log_likelihood_r1 = nb_log_prob_mu_r(y_i_r1, mu_tech_r1, r_tech_r1).mean()  # type: ignore
            log_likelihood_r2 = nb_log_prob_mu_r(y_i_r2, mu_tech_r2, r_tech_r2).mean()  # type: ignore
            # If replicate 1 has 2x the seduencing depth, we assume it will have ~2x the biological signal.
            # mu1_r1_diff = torch.clamp(y_i_r1 - mu_tech_r1, min=0)
            # mu2_r2_diff_scaled = torch.clamp(sd_ratio_r1_to_r2 * (y_i_r2 - mu_tech_r2), min=0)
            mu1_r1_diff = y_i_r1 - mu_tech_r1
            # mu2_r2_diff_scaled = sd_ratio_r1_to_r2 * (y_i_r2 - mu_tech_r2)
            mu2_r2_diff= sd_ratio_r1_to_r2 * (y_i_r2 - mu_tech_r2)
            mse = torch.mean((mu1_r1_diff - mu2_r2_diff) ** 2)
            neg_corr = negative_pearson_loss(mu1_r1_diff, mu2_r2_diff)
            loss = -(log_likelihood_r1 + log_likelihood_r2)

            loss.backward()
            # TODO: Look into reason for NaN vals (I'm assuming it's gradients)
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.copy_(torch.nan_to_num(param.grad, nan=0.0))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            train_batches += 1

            if batch_idx % 1_000 == 0:  # Log every 1,000 batches
                logger.info(
                        "Epoch %s/%s, Batch %s/%s, Train Loss: %s. Log likelihood R1: %s, Log likelihood R2: %s, Biol signal MSE: %s, Biol signal corr: %s",
                    epoch + 1,
                    num_epochs,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                    log_likelihood_r1.item(),
                    log_likelihood_r2.item(),
                    mse.item(),
                    -neg_corr.item(),
                )

        avg_train_loss = total_train_loss / train_batches

        # --- Validation Phase ---
        model.eval()
        total_val_log_likelihood = 0
        total_val_biol_mse = 0
        total_val_biol_corr = 0
        val_batches = 0

        with torch.no_grad():
            epoch_mu_tech_val_r1_list = []
            epoch_y_i_val_r1_list = []
            epoch_mu_tech_val_r2_list = []
            epoch_y_i_val_r2_list = []
            epoch_mu1_r1_diff_list = []
            epoch_mu2_r2_diff_scaled_list = []
            epocch_r1_tech_val_r1_list = []
            epocch_r2_tech_val_r2_list = []
            epoch_xi_val_r1_list = []
            epoch_xi_val_r2_list = []
            for batch_idx, (x_i_val, y_i_val) in enumerate(val_loader):
                x_i_val, y_i_val = x_i_val.to(device), y_i_val.to(device).float()  # noqa: PLW2901
                x_i_val_r1 = x_i_val[:, :5]
                x_i_val_r2 = x_i_val[:, 5:]
                y_i_val_r1 = y_i_val[:, 0]
                y_i_val_r2 = y_i_val[:, 1]
                sd_ratio_r1_to_r2 = y_i_val[:, 2]

                log_mu_tech_val_r1, log_r_tech_val_r1 = model(x_i_val_r1)
                log_mu_tech_val_r1 = log_mu_tech_val_r1.squeeze()
                log_r_tech_val_r1 = log_r_tech_val_r1.squeeze()
                log_mu_tech_val_r2, log_r_tech_val_r2 = model(x_i_val_r2)
                log_mu_tech_val_r2 = log_mu_tech_val_r2.squeeze()
                log_r_tech_val_r2 = log_r_tech_val_r2.squeeze()
                r_tech_val_r1 = torch.exp(log_r_tech_val_r1)
                r_tech_val_r2 = torch.exp(log_r_tech_val_r2)

                mu_tech_val_r1 = torch.exp(log_mu_tech_val_r1)
                mu_tech_val_r2 = torch.exp(log_mu_tech_val_r2)

                val_log_likelihood_r1 = nb_log_prob_mu_r(y_i_val_r1, mu_tech_val_r1, r_tech_val_r1).mean()  # type: ignore
                val_log_likelihood_r2 = nb_log_prob_mu_r(y_i_val_r2, mu_tech_val_r2, r_tech_val_r2).mean()  # type: ignore
                mu1_r1_diff = y_i_val_r1 - mu_tech_val_r1
                mu2_r2_diff = sd_ratio_r1_to_r2 * (y_i_val_r2 - mu_tech_val_r2)
                mse = torch.mean((mu1_r1_diff - mu2_r2_diff) ** 2)
                neg_corr = negative_pearson_loss(mu1_r1_diff, mu2_r2_diff)

                total_val_log_likelihood += val_log_likelihood_r1.item()
                total_val_log_likelihood += val_log_likelihood_r2.item()
                total_val_biol_mse += mse.item()
                total_val_biol_corr += -neg_corr.item()

                epoch_mu_tech_val_r1_list.append(mu_tech_val_r1.detach().cpu().numpy().squeeze())
                epoch_y_i_val_r1_list.append(y_i_val_r1.detach().cpu().numpy().squeeze()) # y_i_val_r1 is already 1D from slicing
                epoch_mu_tech_val_r2_list.append(mu_tech_val_r2.detach().cpu().numpy().squeeze())
                epoch_y_i_val_r2_list.append(y_i_val_r2.detach().cpu().numpy().squeeze()) # y_i_val_r2 is already 1D
                epoch_mu1_r1_diff_list.append(mu1_r1_diff.detach().cpu().numpy().squeeze())
                epoch_mu2_r2_diff_scaled_list.append(mu2_r2_diff.detach().cpu().numpy().squeeze())
                if hidden_dims_disp is not None:
                    epocch_r1_tech_val_r1_list.append(r_tech_val_r1.detach().cpu().numpy().squeeze())  # type: ignore
                    epocch_r2_tech_val_r2_list.append(r_tech_val_r2.detach().cpu().numpy().squeeze())  # type: ignore

                if epoch == 0:
                    epoch_xi_val_r1_list.append(x_i_val_r1.detach().cpu().numpy())
                    epoch_xi_val_r2_list.append(x_i_val_r2.detach().cpu().numpy())

                val_batches += 1

            avg_val_log_likelihood = total_val_log_likelihood / val_batches
            avg_val_biol_mse = total_val_biol_mse / val_batches
            avg_val_biol_corr = total_val_biol_corr / val_batches
            # Batches can have different sizes, so skip the last batch.
            all_mu_tech_val_r1 = np.concatenate(epoch_mu_tech_val_r1_list[:-1]) if epoch_mu_tech_val_r1_list else np.array([])
            all_y_i_val_r1 = np.concatenate(epoch_y_i_val_r1_list[:-1]) if epoch_y_i_val_r1_list else np.array([])
            all_mu_tech_val_r2 = np.concatenate(epoch_mu_tech_val_r2_list[:-1]) if epoch_mu_tech_val_r2_list else np.array([])
            all_y_i_val_r2 = np.concatenate(epoch_y_i_val_r2_list[:-1]) if epoch_y_i_val_r2_list else np.array([])
            all_mu1_r1_diff = np.concatenate(epoch_mu1_r1_diff_list[:-1]) if epoch_mu1_r1_diff_list else np.array([])
            all_mu2_r2_diff_scaled = np.concatenate(epoch_mu2_r2_diff_scaled_list[:-1]) if epoch_mu2_r2_diff_scaled_list else np.array([])
            if hidden_dims_disp is not None:
                all_r1_tech_val_r1 = np.concatenate(epocch_r1_tech_val_r1_list[:-1]) if epocch_r1_tech_val_r1_list else np.array([])
                all_r2_tech_val_r2 = np.concatenate(epocch_r2_tech_val_r2_list[:-1]) if epocch_r2_tech_val_r2_list else np.array([])
                with open(model_save_dir / f'val_r1_tech_val_r1_{epoch}.npy', 'wb') as f:
                    np.save(f, all_r1_tech_val_r1)
                with open(model_save_dir / f'val_r2_tech_val_r2_{epoch}.npy', 'wb') as f:
                    np.save(f, all_r2_tech_val_r2)
            if epoch == 0:
                all_xi_val_r1 = np.concatenate(epoch_xi_val_r1_list[:-1]) if epoch_xi_val_r1_list else np.array([])
                all_xi_val_r2 = np.concatenate(epoch_xi_val_r2_list[:-1]) if epoch_xi_val_r2_list else np.array([])
                with open(model_save_dir / f'val_xi_val_r1.npy', 'wb') as f:
                    np.save(f, all_xi_val_r1)
                with open(model_save_dir / f'val_xi_val_r2.npy', 'wb') as f:
                    np.save(f, all_xi_val_r2)
            
            for arr, arr_name in zip([all_mu_tech_val_r1, all_y_i_val_r1, all_mu_tech_val_r2, all_y_i_val_r2, all_mu1_r1_diff, all_mu2_r2_diff_scaled], ['mu_tech_val_r1', 'y_i_val_r1', 'mu_tech_val_r2', 'y_i_val_r2', 'mu1_r1_diff', 'mu2_r2_diff_scaled']):
                with open(model_save_dir / f'val_{arr_name}_{epoch}.npy', 'wb') as f:
                    np.save(f, arr)

            if wandb_name is not None:
                wandb.log(
                    {
                        "epoch_val_likelihood": avg_val_log_likelihood,
                        "epoch_val_biol_mse": avg_val_biol_mse,
                        "epoch_val_biol_corr": avg_val_biol_corr,
                        "epoch": epoch,
                    },
                )
                # Batches can have different sizes, so skip the last batch.
                all_mu_tech_val_r1 = np.concatenate(epoch_mu_tech_val_r1_list[:-1]) if epoch_mu_tech_val_r1_list else np.array([])
                all_y_i_val_r1 = np.concatenate(epoch_y_i_val_r1_list[:-1]) if epoch_y_i_val_r1_list else np.array([])
                all_mu_tech_val_r2 = np.concatenate(epoch_mu_tech_val_r2_list[:-1]) if epoch_mu_tech_val_r2_list else np.array([])
                all_y_i_val_r2 = np.concatenate(epoch_y_i_val_r2_list[:-1]) if epoch_y_i_val_r2_list else np.array([])
                all_mu1_r1_diff = np.concatenate(epoch_mu1_r1_diff_list[:-1]) if epoch_mu1_r1_diff_list else np.array([])
                all_mu2_r2_diff_scaled = np.concatenate(epoch_mu2_r2_diff_scaled_list[:-1]) if epoch_mu2_r2_diff_scaled_list else np.array([])
                
                for arr, arr_name in zip([all_mu_tech_val_r1, all_y_i_val_r1, all_mu_tech_val_r2, all_y_i_val_r2, all_mu1_r1_diff, all_mu2_r2_diff_scaled], ['mu_tech_val_r1', 'y_i_val_r1', 'mu_tech_val_r2', 'y_i_val_r2', 'mu1_r1_diff', 'mu2_r2_diff_scaled']):
                    with open(model_save_dir / f'val_{arr_name}.npy', 'wb') as f:
                        np.save(f, arr)

#             # Convert aggregated numpy arrays back to torch.Tensors for the helper function
#             # The helper function expects torch.Tensors
            
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

    save_dir = DATA_DIR / f"runs/mu_r_ll_only/{args.target}"
    save_dir.mkdir(parents=False)
    print(f"SAVING TO: {save_dir}")


    build_and_train(
            dim_x=train_ds.get_dim_x(),
            train_loader=train_loader,
            val_loader=val_loader,
            hidden_dims_mean=(32, 32),
            hidden_dims_disp=(8, 8),
            weight_decay=0.01,
            num_epochs=100,
            device=torch.device("cuda:0"),
            model_save_dir=save_dir,
            base_lr=0.001,
            max_lr=0.01,
            patience=10,
            )

