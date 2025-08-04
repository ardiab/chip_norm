from __future__ import annotations

import copy  # For saving best model state
import logging
import time
from typing import TYPE_CHECKING, Literal
import matplotlib.pyplot as plt

import torch
import wandb
from torch import optim
from torch.optim.lr_scheduler import CyclicLR
import numpy as np

from chipvi.models.technical_model import TechNB, TechPoisson
from chipvi.utils.distributions import nb_log_prob, poisson_log_prob

if TYPE_CHECKING:
    from pathlib import Path

    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def build_and_train(
    dim_x: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    hidden_dims_mean: tuple,
    weight_decay: float,
    num_epochs: int,
    device: torch.device,
    model_save_dir: Path,
    base_lr: float,
    max_lr: float,
    patience: int = 10,
    wandb_name: str | None = None,
    hidden_dims_disp: tuple | None = None,
) -> tuple[TechPoisson, TechPoisson]:
    if wandb_name is not None:
        wandb.init(project="chipvi", name=wandb_name)
        wandb.config.update(
            {
                "method": "tech_biol_rep_joint",
                "tech_dist": "poisson",
                "hidden_dims_mean": hidden_dims_mean,
                "base_lr": base_lr,
                "max_lr": max_lr,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
                "patience": patience,
            },
        )

    if hidden_dims_disp is None:
        logger.info("Training Poisson")
        model = TechPoisson(input_dim=dim_x, hidden_dims=hidden_dims_mean).to(device)
    else:
        logger.info("Training NB")
        model = TechNB(
                dim_x=dim_x,
                hidden_dims_mean=hidden_dims_mean,
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

            if hidden_dims_disp is None:
                log_mu_tech_r1 = model(x_i_r1).squeeze()
                log_mu_tech_r2 = model(x_i_r2).squeeze()
            else:
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

            if hidden_dims_disp is None:
                log_likelihood_r1 = poisson_log_prob(y_i_r1, mu_tech_r1).mean()
                log_likelihood_r2 = poisson_log_prob(y_i_r2, mu_tech_r2).mean()
            else:
                log_likelihood_r1 = nb_log_prob(y_i_r1, mu_tech_r1, r_tech_r1).mean()  # type: ignore
                log_likelihood_r2 = nb_log_prob(y_i_r2, mu_tech_r2, r_tech_r2).mean()  # type: ignore
            # If replicate 1 has 2x the seduencing depth, we assume it will have ~2x the biological signal.
            # mu1_r1_diff = torch.clamp(y_i_r1 - mu_tech_r1, min=0)
            # mu2_r2_diff_scaled = torch.clamp(sd_ratio_r1_to_r2 * (y_i_r2 - mu_tech_r2), min=0)
            mu1_r1_diff = y_i_r1 - mu_tech_r1
            # mu2_r2_diff_scaled = sd_ratio_r1_to_r2 * (y_i_r2 - mu_tech_r2)
            mu2_r2_diff= sd_ratio_r1_to_r2 * (y_i_r2 - mu_tech_r2)
            mse = torch.mean((mu1_r1_diff - mu2_r2_diff) ** 2)
            neg_corr = negative_pearson_loss(mu1_r1_diff, mu2_r2_diff)
            if epoch == 0:
                loss = (0.7 * -(log_likelihood_r1 + log_likelihood_r2)) + (0.3 * (mse + 3 * neg_corr))
                # loss = (0.7 * -(log_likelihood_r1 + log_likelihood_r2)) + (0.3 * mse)
                # loss = (0.2 * -(log_likelihood_r1 + log_likelihood_r2)) + (0.8 * neg_corr)
                # loss = -(log_likelihood_r1 + log_likelihood_r2)
            else:
                loss = (0.7 * -(log_likelihood_r1 + log_likelihood_r2)) + (0.3 * (mse + 3 * neg_corr))
                # loss = (0.3 * -(log_likelihood_r1 + log_likelihood_r2)) + (0.7 * mse)
                # loss = (0.2 * -(log_likelihood_r1 + log_likelihood_r2)) + (0.8 * neg_corr)
                # loss = -(log_likelihood_r1 + log_likelihood_r2)
            # loss = -(log_likelihood_r1 + log_likelihood_r2) + mse + 2 * neg_corr

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
                if wandb_name is not None:
                    wandb.log(
                        {
                            "batch_train_loss": loss.item(),
                            "batch_train_log_likelihood_r1": log_likelihood_r1.item(),
                            "batch_train_log_likelihood_r2": log_likelihood_r2.item(),
                            "batch_train_biol_mse": mse.item(),
                            "batch_train_biol_corr": -neg_corr.item(),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "batch": epoch * len(train_loader) + batch_idx,
                        },
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

                if hidden_dims_disp is None:
                    log_mu_tech_val_r1 = model(x_i_val_r1).squeeze()
                    log_mu_tech_val_r2 = model(x_i_val_r2).squeeze()
                else:
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

                if hidden_dims_disp is None:
                    val_log_likelihood_r1 = poisson_log_prob(y_i_val_r1, mu_tech_val_r1).mean()
                    val_log_likelihood_r2 = poisson_log_prob(y_i_val_r2, mu_tech_val_r2).mean()
                else:
                    val_log_likelihood_r1 = nb_log_prob(y_i_val_r1, mu_tech_val_r1, r_tech_val_r1).mean()  # type: ignore
                    val_log_likelihood_r2 = nb_log_prob(y_i_val_r2, mu_tech_val_r2, r_tech_val_r2).mean()  # type: ignore
                # mu1_r1_diff = torch.clamp(y_i_val_r1 - mu_tech_val_r1 , min=0)
                # mu2_r2_diff_scaled = torch.clamp(sd_ratio_r1_to_r2 * (y_i_val_r2 - mu_tech_val_r2), min=0)
                mu1_r1_diff = y_i_val_r1 - mu_tech_val_r1
                # mu2_r2_diff_scaled = sd_ratio_r1_to_r2 * (y_i_val_r2 - mu_tech_val_r2)
                mu2_r2_diff = sd_ratio_r1_to_r2 * (y_i_val_r2 - mu_tech_val_r2)
                mse = torch.mean((mu1_r1_diff - mu2_r2_diff) ** 2)
                neg_corr = negative_pearson_loss(mu1_r1_diff, mu2_r2_diff)

                total_val_log_likelihood += val_log_likelihood_r1.item()
                total_val_log_likelihood += val_log_likelihood_r2.item()
                total_val_biol_mse += mse.item()
                total_val_biol_corr += -neg_corr.item()

                # if wandb_name is not None:
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
            
#             # Plot 1: mu_tech_val_r1 vs y_i_val_r1
                log_2d_histogram_wandb(
                    x_data=torch.from_numpy(all_mu_tech_val_r1), 
                    y_data=torch.from_numpy(all_y_i_val_r1),
                    title=f'Epoch {epoch} Val: Predicted mu_r1 vs True y_r1',
                    xlabel='Predicted mu_tech_val_r1', ylabel='True y_i_val_r1',
                    wandb_key=f'val_histograms_epoch/epoch_{epoch}/mu_tech_r1_vs_y_r1'
                )

#             # Plot 2: mu_tech_val_r2 vs y_i_val_r2
                log_2d_histogram_wandb(
                    x_data=torch.from_numpy(all_mu_tech_val_r2), 
                    y_data=torch.from_numpy(all_y_i_val_r2),
                    title=f'Epoch {epoch} Val: Predicted mu_r2 vs True y_r2',
                    xlabel='Predicted mu_tech_val_r2', ylabel='True y_i_val_r2',
                    wandb_key=f'val_histograms_epoch/epoch_{epoch}/mu_tech_r2_vs_y_r2'
                )

#             # Plot 3: mu_tech_val_r1 vs mu_tech_val_r2
                log_2d_histogram_wandb(
                    x_data=torch.from_numpy(all_mu_tech_val_r1), 
                    y_data=torch.from_numpy(all_mu_tech_val_r2),
                    title=f'Epoch {epoch} Val: Predicted mu_r1 vs Predicted mu_r2',
                    xlabel='Predicted mu_tech_val_r1', ylabel='Predicted mu_tech_val_r2',
                    wandb_key=f'val_histograms_epoch/epoch_{epoch}/mu_tech_r1_vs_mu_tech_r2'
                )

#             # Plot 4: mu1_r1_diff vs mu2_r2_diff_scaled
                log_2d_histogram_wandb(
                    x_data=torch.from_numpy(all_mu1_r1_diff), 
                    y_data=torch.from_numpy(all_mu2_r2_diff_scaled),
                    title=f'Epoch {epoch} Val: mu1_r1_diff vs mu2_r2_diff_scaled',
                    xlabel='mu_tech_r1 - y_r1', ylabel='sd_ratio * (mu_tech_r2 - y_r2)',
                    wandb_key=f'val_histograms_epoch/epoch_{epoch}/diff_r1_vs_diff_scaled_r2'
                )

#             # Plot 5: mu1_r1_diff vs y_i_val_r1
                log_2d_histogram_wandb(
                    x_data=torch.from_numpy(all_mu1_r1_diff), 
                    y_data=torch.from_numpy(all_y_i_val_r1), # Reusing all_y_i_val_r1
                    title=f'Epoch {epoch} Val: mu1_r1_diff vs True y_r1',
                    xlabel='mu_tech_r1 - y_r1', ylabel='True y_i_val_r1',
                    wandb_key=f'val_histograms_epoch/epoch_{epoch}/diff_r1_vs_y_r1'
                )

#             # Plot 6: mu2_r2_diff_scaled vs y_i_val_r2
                log_2d_histogram_wandb(
                    x_data=torch.from_numpy(all_mu2_r2_diff_scaled), 
                    y_data=torch.from_numpy(all_y_i_val_r2), # Reusing all_y_i_val_r2
                    title=f'Epoch {epoch} Val: mu2_r2_diff_scaled vs True y_r2',
                    xlabel='sd_ratio * (mu_tech_r2 - y_r2)', ylabel='True y_i_val_r2',
                    wandb_key=f'val_histograms_epoch/epoch_{epoch}/diff_scaled_r2_vs_y_r2'
                )

                log_1d_histogram_wandb(
                    data=all_mu_tech_val_r1,
                    title=f'Epoch {epoch} Val: Distribution of Predicted mu_r1 (flat)',
                    xlabel='Predicted mu_tech_val_r1 (flattened features)',
                    wandb_key=f'val_histograms_1d_epoch/epoch_{epoch}/predicted_mu_r1_dist'
                )
                log_1d_histogram_wandb(
                    data=all_mu_tech_val_r2,
                    title=f'Epoch {epoch} Val: Distribution of Predicted mu_r2 (flat)',
                    xlabel='Predicted mu_tech_val_r2 (flattened features)',
                    wandb_key=f'val_histograms_1d_epoch/epoch_{epoch}/predicted_mu_r2_dist'
                )
                log_1d_histogram_wandb(
                    data=all_mu1_r1_diff,
                    title=f'Epoch {epoch} Val: Distribution of mu1_r1_diff (flat)',
                    xlabel='(y_r1 - mu_tech_r1) (flattened)',
                    wandb_key=f'val_histograms_1d_epoch/epoch_{epoch}/mu1_r1_diff_dist'
                )
                log_1d_histogram_wandb(
                    data=all_mu2_r2_diff_scaled,
                    title=f'Epoch {epoch} Val: Distribution of mu2_r2_diff_scaled (flat)',
                    xlabel='sd_ratio * (y_r2 - mu_tech_r2) (flattened)',
                    wandb_key=f'val_histograms_1d_epoch/epoch_{epoch}/mu2_r2_diff_scaled_dist'
                )
                log_1d_histogram_wandb(
                    data=all_y_i_val_r1,
                    title=f'Epoch {epoch} Val: Distribution of true y_r1 (flat)',
                    xlabel='True y_i_val_r1 (flattened)',
                    wandb_key=f'val_histograms_1d_epoch/epoch_{epoch}/true_y_r1_dist'
                )
                log_1d_histogram_wandb(
                    data=all_y_i_val_r2,
                    title=f'Epoch {epoch} Val: Distribution of true y_r2 (flat)',
                    xlabel='True y_i_val_r2 (flattened)',
                    wandb_key=f'val_histograms_1d_epoch/epoch_{epoch}/true_y_r2_dist'
                )

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
    best_ll_model = TechPoisson(input_dim=dim_x, hidden_dims=hidden_dims_mean)
    best_ll_model.load_state_dict(best_ll_model_state)  # type: ignore
    best_mse_model = TechPoisson(input_dim=dim_x, hidden_dims=hidden_dims_mean)
    best_mse_model.load_state_dict(best_mse_model_state)  # type: ignore
    # if best_model_state:
    #     model.load_state_dict(best_model_state)

    return best_ll_model, best_mse_model  # Returns the model with the best validation weights


def negative_pearson_loss(preds, targets, eps=1e-8):
    preds = preds - preds.mean()
    targets = targets - targets.mean()
    
    numerator = (preds * targets).sum()
    denominator = torch.sqrt((preds**2).sum() * (targets**2).sum() + eps)
    
    return -numerator / denominator


def log_2d_histogram_wandb(x_data, y_data, title, xlabel, ylabel, wandb_key, bins=50):
    """
    Creates a 2D histogram from x_data and y_data, and logs it to Weights & Biases.
    Expects x_data and y_data to be torch.Tensors containing all data for the histogram.

    Args:
        x_data (torch.Tensor): Data for the x-axis.
        y_data (torch.Tensor): Data for the y-axis.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        wandb_key (str): Key under which to log the histogram in W&B.
        bins (int, optional): Number of bins for the histogram. Defaults to 50.
    """
    # Ensure data is on CPU, detached from graph, and converted to numpy
    # .squeeze() handles cases where tensors might be (N, 1) instead of (N,)
    x_np = x_data.detach().cpu().numpy().squeeze()
    y_np = y_data.detach().cpu().numpy().squeeze()

    # Ensure x_np and y_np are 1D arrays
    if x_np.ndim > 1 or y_np.ndim > 1:
        print(f"Warning: Data for histogram '{title}' is not 1D after squeezing. Skipping plot.")
        print(f"x_data shape: {x_data.shape}, y_data shape: {y_data.shape}")
        print(f"x_np shape: {x_np.shape}, y_np shape: {y_np.shape}")
        return
    
    if len(x_np) == 0 or len(y_np) == 0:
        print(f"Warning: Data for histogram '{title}' is empty. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6)) # Create a new figure and axes
    try:
        counts, xedges, yedges, im = ax.hist2d(x_np, y_np, bins=bins, cmap='viridis')
        fig.colorbar(im, ax=ax) # Add a colorbar to show density
    except Exception as e:
        print(f"Error creating hist2d for {title}: {e}")
        print(f"x_np length: {len(x_np)}, y_np length: {len(y_np)}")
        if len(x_np) > 0: print(f"x_np (first 10): {x_np[:10]}")
        if len(y_np) > 0: print(f"y_np (first 10): {y_np[:10]}")
        plt.close(fig)
        return

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Log the plot to W&B
    wandb.log({wandb_key: wandb.Image(fig)})
    
    plt.close(fig) # Important to close the figure to free memory


def log_1d_histogram_wandb(data, title, xlabel, wandb_key, bins=50):
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy().squeeze()
    else:
        data_np = data.squeeze()

    if data_np.ndim > 1:
        print(f"Warning: Data for 1D histogram '{title}' is not 1D after processing. Skipping plot.")
        print(f"data_np shape: {data_np.shape}")
        return

    if len(data_np) == 0:
        print(f"Warning: Data for 1D histogram '{title}' is empty. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        ax.hist(data_np, bins=bins, color='skyblue', edgecolor='black')
    except Exception as e:
        print(f"Error creating 1D hist for {title}: {e}")
        print(f"data_np length: {len(data_np)}")
        if len(data_np) > 0: print(f"data_np (first 10): {data_np[:10]}")
        plt.close(fig)
        return
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    
    wandb.log({wandb_key: wandb.Image(fig)})
    plt.close(fig)

