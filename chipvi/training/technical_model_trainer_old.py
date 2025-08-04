from __future__ import annotations

import copy  # For saving best model state
import logging
import time
from typing import TYPE_CHECKING, Literal

import torch
import wandb
from torch import optim
from torch.optim.lr_scheduler import CyclicLR

from chipvi.models.technical_model import TechNB, TechPoisson
from chipvi.utils.distributions import nb_log_prob, poisson_log_prob

if TYPE_CHECKING:
    from pathlib import Path

    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


LOG_MU_CLAMP_MIN = -10
LOG_MU_CLAMP_MAX = 10
LOG_R_CLAMP_MIN = -10
LOG_R_CLAMP_MAX = 10


def train_technical_model(
    dim_x: int,
    train_mode: Literal["replicate", "single"],
    train_loader: DataLoader,
    val_loader: DataLoader,
    hidden_dims_mean: tuple,
    base_lr: float,
    max_lr: float,
    weight_decay: float,
    num_epochs: int,
    device: torch.device,
    model_save_path: Path,
    log_wandb: bool,
    hidden_dims_r: tuple | None = None,
    patience: int = 10,
    wandb_name: str | None = None,
) -> TechNB | TechPoisson:
    if log_wandb:
        wandb.init(project="chipvi", name=wandb_name)
        wandb.config.update(
            {
                "tech_dist": "poisson",
                "hidden_dims_mean": hidden_dims_mean,
                "hidden_dims_r": hidden_dims_r,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
                "patience": patience,
                "base_lr": base_lr,
                "max_lr": max_lr,
            },
        )

    model = TechPoisson(input_dim=dim_x, hidden_dims=hidden_dims_mean).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    scheduler = CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        mode="triangular2",
        cycle_momentum=False,
        step_size_up=int(len(train_loader) / 2),
    )

    best_val_log_likelihood = float("-inf")
    epochs_no_improve = 0
    best_model_state = None

    start_time = time.time()
    logger.info("Starting technical model training.")
    for epoch in range(num_epochs):
        start_time = time.time()

        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        train_batches = 0

        for batch_idx, (x_i, y_i) in enumerate(train_loader):
            x_i, y_i = x_i.to(device), y_i.to(device).float()  # noqa: PLW2901

            optimizer.zero_grad()

            if train_mode == "single":
                loss = forward_single(model, x_i, y_i)

            elif train_mode == "replicate":
                loss = forward_replicate(model, x_i, y_i)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            train_batches += 1

            if batch_idx % 1_000 == 0:
                logger.info(
                    "Epoch %s/%s, Batch %s/%s, Train Loss: %s",
                    epoch + 1,
                    num_epochs,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                )
                if log_wandb:
                    wandb.log(
                        {
                            "batch_train_loss": loss.item(),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "batch": epoch * len(train_loader) + batch_idx,
                        },
                    )

        avg_train_loss = total_train_loss / train_batches

        # --- Validation Phase ---
        model.eval()
        total_val_log_likelihood = 0
        val_batches = 0

        with torch.no_grad():
            for batch_idx, (x_i_val, y_i_val) in enumerate(val_loader):
                x_i_val, y_i_val = x_i_val.to(device), y_i_val.to(device).float()  # noqa: PLW2901
                if train_mode == "single":
                    loss = forward_single(model, x_i_val, y_i_val, tech_dist)
                elif train_mode == "replicate":
                    loss = forward_replicate(model, x_i_val, y_i_val, tech_dist)

                val_batches += 1

            avg_val_log_likelihood = total_val_log_likelihood / val_batches

        epoch_duration = time.time() - start_time

        logger.info(
            "Epoch %s/%s Summary: Duration: %s, Avg Train Loss: %s, Avg Val Log-Likelihood: %s",
            epoch + 1,
            num_epochs,
            epoch_duration,
            avg_train_loss,
            avg_val_log_likelihood,
        )

        # --- Early Stopping & Model Checkpointing ---
        if avg_val_log_likelihood > best_val_log_likelihood:
            best_val_log_likelihood = avg_val_log_likelihood
            epochs_no_improve = 0
            # Save the best model state
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, model_save_path)
            logger.info(
                "New best validation log-likelihood: %s. Model saved to %s",
                best_val_log_likelihood,
                model_save_path,
            )
        else:
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
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model  # Returns the model with the best validation weights
