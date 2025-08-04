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


def train_technical_model(
    dim_x: int,
    tech_dist: Literal["poisson", "nb"],
    train_mode: Literal["replicate", "single"],
    train_loader: DataLoader,
    val_loader: DataLoader,
    hidden_dims_mean: tuple,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    device: torch.device,
    model_save_path: Path,
    hidden_dims_r: tuple | None = None,
    patience: int = 10,
    log_r_clamp_min: float = -10.0,
    log_r_clamp_max: float = 10.0,
    log_mu_clamp_min: float = -10.0,
    log_mu_clamp_max: float = 10.0,
    wandb_project: str | None = "chipvi",
    wandb_name: str | None = None,
    base_lr: float | None = None,
    max_lr: float | None = None,
    lambda_reg: float = 0.5,
) -> TechNB | TechPoisson:
    """Train the Phase A technical baseline model using a standard PyTorch loop.

    Args:
        dim_x (int): Dimensionality of the input covariates x_i.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        hidden_dims_mean (tuple): Hidden layer dimensions for network f.
        hidden_dims_r (tuple): Hidden layer dimensions for network g.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        num_epochs (int): Number of epochs to train for.
        patience (int): Number of epochs to wait before early stopping.
        log_r_clamp_min (float): Minimum value for log_r_tech.
        log_r_clamp_max (float): Maximum value for log_r_tech.
        model_save_path (str | Path): Path to save the best model state.
        device (torch.device): Device to train on.
        wandb_project (str | None): Weights & Biases project name.
        wandb_name (str | None): Weights & Biases run name.
        base_lr (float | None): Base learning rate for cyclic LR.
        max_lr (float | None): Maximum learning rate for cyclic LR.

    Returns:
        TechNB | TechPoisson: The trained model.

    """
    # Initialize wandb if project name is provided
    if wandb_project is not None:
        wandb.init(project=wandb_project, name=wandb_name)
        wandb.config.update(
            {
                "tech_dist": tech_dist,
                "hidden_dims_mean": hidden_dims_mean,
                "hidden_dims_r": hidden_dims_r,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
                "patience": patience,
            },
        )

    if tech_dist == "poisson":
        model = TechPoisson(input_dim=dim_x, hidden_dims=hidden_dims_mean).to(device)
    elif tech_dist == "nb":
        if hidden_dims_r is None:
            msg = "hidden_dims_r must be provided for NB model"
            raise ValueError(msg)
        model = TechNB(
            dim_x=dim_x,
            hidden_dims_mean=hidden_dims_mean,
            hidden_dims_r=hidden_dims_r,
        ).to(device)
    else:
        msg = f"Got unexpected choice for technical distribution: {tech_dist}"
        logger.exception(msg)
        raise ValueError(msg)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize cyclic learning rate scheduler if base_lr and max_lr are provided
    if base_lr is not None and max_lr is not None:
        scheduler = CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            mode="triangular2",
            cycle_momentum=False,
            step_size_up=int(len(train_loader) / 2),
        )
    else:
        scheduler = None

    best_val_log_likelihood = float("-inf")
    epochs_no_improve = 0
    best_model_state = None

    start_time = time.time()
    logger.info("Starting Phase A training...")
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
                if tech_dist == "poisson":
                    log_mu_tech = model(x_i)
                    log_r_tech = None
                elif tech_dist == "nb":
                    log_mu_tech, log_r_tech = model(x_i)

                    log_r_tech_clamped = torch.clamp(
                        log_r_tech,
                        log_r_clamp_min,
                        log_r_clamp_max,
                    )

                log_mu_tech_clamped = torch.clamp(
                    log_mu_tech,
                    log_mu_clamp_min,
                    log_mu_clamp_max,
                )

                mu_tech = torch.exp(log_mu_tech_clamped)
                if tech_dist == "nb":
                    r_tech = torch.exp(log_r_tech_clamped)

                # Squeeze if model outputs (batch, 1) and y_i is (batch,)
                if mu_tech.ndim == y_i.ndim + 1 and mu_tech.shape[-1] == 1:
                    mu_tech = mu_tech.squeeze(-1)
                if tech_dist == "nb" and r_tech.ndim == y_i.ndim + 1 and r_tech.shape[-1] == 1:
                    r_tech = r_tech.squeeze(-1)

                if tech_dist == "nb":
                    log_likelihoods = nb_log_prob(y_i, mu_tech, r_tech)
                elif tech_dist == "poisson":
                    log_likelihoods = poisson_log_prob(y_i, mu_tech)
                loss = -log_likelihoods.mean()

            elif train_mode == "replicate":
                x_i_r1 = x_i[:, :4]
                x_i_r2 = x_i[:, 4:]
                y_i_r1 = y_i[:, 0]
                y_i_r2 = y_i[:, 1]

                if tech_dist == "poisson":
                    log_mu_tech_r1 = model(x_i_r1)
                    log_mu_tech_r2 = model(x_i_r2)
                    log_r_tech_r1 = None
                    log_r_tech_r2 = None
                elif tech_dist == "nb":
                    log_mu_tech_r1, log_r_tech_r1 = model(x_i_r1)
                    log_mu_tech_r2, log_r_tech_r2 = model(x_i_r2)
                    log_r_tech_r1_clamped = torch.clamp(
                        log_r_tech_r1,
                        log_r_clamp_min,
                        log_r_clamp_max,
                    )
                    log_r_tech_r2_clamped = torch.clamp(
                        log_r_tech_r2,
                        log_r_clamp_min,
                        log_r_clamp_max,
                    )
                    r_tech_r1 = torch.exp(log_r_tech_r1_clamped)
                    r_tech_r2 = torch.exp(log_r_tech_r2_clamped)

                mu_tech_r1 = torch.exp(log_mu_tech_r1)
                mu_tech_r2 = torch.exp(log_mu_tech_r2)

                if tech_dist == "nb":
                    log_likelihoods_r1 = nb_log_prob(y_i_r1, mu_tech_r1, r_tech_r1)
                    log_likelihoods_r2 = nb_log_prob(y_i_r2, mu_tech_r2, r_tech_r2)
                elif tech_dist == "poisson":
                    log_likelihoods_r1 = poisson_log_prob(y_i_r1, mu_tech_r1)
                    log_likelihoods_r2 = poisson_log_prob(y_i_r2, mu_tech_r2)
                delta = ((mu_tech_r1 - y_i_r1) - (mu_tech_r2 - y_i_r2)).abs().mean()
                loss = -log_likelihoods_r1.mean() - log_likelihoods_r2.mean() + lambda_reg * delta

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_train_loss += loss.item()
            train_batches += 1

            if batch_idx % 1_000 == 0:  # Log every 1,000 batches
                logger.info(
                    "Epoch %s/%s, Batch %s/%s, Train Loss: %s",
                    epoch + 1,
                    num_epochs,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                )
                if wandb_project is not None:
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
                    if tech_dist == "poisson":
                        log_mu_tech_val = model(x_i_val)
                        log_r_tech_val = None
                    elif tech_dist == "nb":
                        log_mu_tech_val, log_r_tech_val = model(x_i_val)

                    if log_r_tech_val is not None:
                        log_r_tech_clamped_val = torch.clamp(
                            log_r_tech_val,
                            log_r_clamp_min,
                            log_r_clamp_max,
                        )

                    mu_tech_val = torch.exp(log_mu_tech_val)
                    if tech_dist == "nb":
                        r_tech_val = torch.exp(log_r_tech_clamped_val)

                    if mu_tech_val.ndim == y_i_val.ndim + 1 and mu_tech_val.shape[-1] == 1:
                        mu_tech_val = mu_tech_val.squeeze(-1)
                    if (
                        tech_dist == "nb"
                        and r_tech_val.ndim == y_i_val.ndim + 1
                        and r_tech_val.shape[-1] == 1
                    ):
                        r_tech_val = r_tech_val.squeeze(-1)

                    if tech_dist == "nb":
                        val_log_likelihoods = nb_log_prob(y_i_val, mu_tech_val, r_tech_val)
                    elif tech_dist == "poisson":
                        val_log_likelihoods = poisson_log_prob(y_i_val, mu_tech_val)
                    total_val_log_likelihood += val_log_likelihoods.mean().item()
                elif train_mode == "replicate":
                    x_i_val_r1 = x_i_val[:, :4]
                    x_i_val_r2 = x_i_val[:, 4:]
                    y_i_val_r1 = y_i_val[:, 0]
                    y_i_val_r2 = y_i_val[:, 1]

                    if tech_dist == "poisson":
                        log_mu_tech_val_r1 = model(x_i_val_r1)
                        log_mu_tech_val_r2 = model(x_i_val_r2)
                        log_r_tech_val_r1 = None
                        log_r_tech_val_r2 = None
                    elif tech_dist == "nb":
                        log_mu_tech_val_r1, log_r_tech_val_r1 = model(x_i_val_r1)
                        log_mu_tech_val_r2, log_r_tech_val_r2 = model(x_i_val_r2)

                        log_r_tech_val_r1_clamped = torch.clamp(
                            log_r_tech_val_r1,
                            log_r_clamp_min,
                            log_r_clamp_max,
                        )
                        log_r_tech_val_r2_clamped = torch.clamp(
                            log_r_tech_val_r2,
                            log_r_clamp_min,
                            log_r_clamp_max,
                        )
                        r_tech_val_r1 = torch.exp(log_r_tech_val_r1_clamped)
                        r_tech_val_r2 = torch.exp(log_r_tech_val_r2_clamped)

                    mu_tech_val_r1 = torch.exp(log_mu_tech_val_r1)
                    mu_tech_val_r2 = torch.exp(log_mu_tech_val_r2)

                    if tech_dist == "nb":
                        val_log_likelihoods_r1 = nb_log_prob(
                            y_i_val_r1, mu_tech_val_r1, r_tech_val_r1
                        )
                        val_log_likelihoods_r2 = nb_log_prob(
                            y_i_val_r2, mu_tech_val_r2, r_tech_val_r2
                        )
                    elif tech_dist == "poisson":
                        val_log_likelihoods_r1 = poisson_log_prob(y_i_val_r1, mu_tech_val_r1)
                        val_log_likelihoods_r2 = poisson_log_prob(y_i_val_r2, mu_tech_val_r2)
                    delta = (
                        ((mu_tech_val_r1 - y_i_val_r1) - (mu_tech_val_r2 - y_i_val_r2)).abs().mean()
                    )
                    loss = (
                        -val_log_likelihoods_r1.mean()
                        - val_log_likelihoods_r2.mean()
                        + lambda_reg * delta
                    )

                    total_val_log_likelihood += val_log_likelihoods_r1.mean().item()
                    total_val_log_likelihood += val_log_likelihoods_r2.mean().item()

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


def forward_single(model, x_i, y_i, tech_dist):
    if tech_dist == "poisson":
        log_mu_tech = model(x_i)
        r_tech = None

    elif tech_dist == "nb":
        log_mu_tech, log_r_tech = model(x_i)

        log_r_tech_clamped = torch.clamp(
            log_r_tech,
            min=-10,
            max=10,
        )
        r_tech = torch.exp(log_r_tech_clamped)

    log_mu_tech_clamped = torch.clamp(
        log_mu_tech,
        min=-10,
        max=10,
    )

    mu_tech = torch.exp(log_mu_tech_clamped)

    if tech_dist == "nb":
        log_likelihoods = nb_log_prob(y_i, mu_tech, r_tech)
    elif tech_dist == "poisson":
        log_likelihoods = poisson_log_prob(y_i, mu_tech)

    loss = -log_likelihoods.mean()

    return loss


def forward_replicate(model, x_i, y_i, tech_dist):
    # TODO: This will break if number of covariates changes
    x_i_r1 = x_i[:, :4]
    x_i_r2 = x_i[:, 4:]
    y_i_r1 = y_i[:, 0]
    y_i_r2 = y_i[:, 1]

    if tech_dist == "poisson":
        log_mu_tech_r1 = model(x_i_r1)
        log_mu_tech_r2 = model(x_i_r2)
        log_r_tech_r1 = None
        log_r_tech_r2 = None

    elif tech_dist == "nb":
        log_mu_tech_r1, log_r_tech_r1 = model(x_i_r1)
        log_mu_tech_r2, log_r_tech_r2 = model(x_i_r2)
        log_r_tech_r1_clamped = torch.clamp(
            log_r_tech_r1,
            min=LOG_R_CLAMP_MIN,
            max=LOG_R_CLAMP_MAX,
        )
        log_r_tech_r2_clamped = torch.clamp(
            log_r_tech_r2,
            min=LOG_R_CLAMP_MIN,
            max=LOG_R_CLAMP_MAX,
        )
        r_tech_r1 = torch.exp(log_r_tech_r1_clamped)
        r_tech_r2 = torch.exp(log_r_tech_r2_clamped)

    mu_tech_r1 = torch.exp(
            torch.clamp(
                log_mu_tech_r1,
                min=LOG_MU_CLAMP_MIN,
                max=LOG_MU_CLAMP_MAX,
            )
                )
    mu_tech_r2 = torch.exp(
            torch.clamp(
                log_mu_tech_r2,
                min=LOG_MU_CLAMP_MIN,
                max=LOG_MU_CLAMP_MAX,
            )
                )

    if tech_dist == "nb":
        log_likelihoods_r1 = nb_log_prob(y_i_r1, mu_tech_r1, r_tech_r1)
        log_likelihoods_r2 = nb_log_prob(y_i_r2, mu_tech_r2, r_tech_r2)
    elif tech_dist == "poisson":
        log_likelihoods_r1 = poisson_log_prob(y_i_r1, mu_tech_r1)
        log_likelihoods_r2 = poisson_log_prob(y_i_r2, mu_tech_r2)

    delta = ((mu_tech_r1 - y_i_r1) - (mu_tech_r2 - y_i_r2)).abs().mean()
    loss = -log_likelihoods_r1.mean() - log_likelihoods_r2.mean() + lambda_reg * delta

    return loss

