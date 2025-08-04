from chipvi.data.datasets import MultiReplicateDataset
from pathlib import Path
import pickle as pkl
import argparse
import torch

import copy  # For saving best model state
import logging
import time
from typing import TYPE_CHECKING, Literal
import matplotlib.pyplot as plt

import torch
from torch import optim
import numpy as np

from chipvi.models.technical_model import TechNB_r_p
from chipvi.utils.distributions import nb_log_prob_r_p
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)
DATA_DIR = Path("/lotterlab/users/abdul/repos/chipvi/")


def build_and_train(
    dim_x: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    hidden_dims_r: tuple,
    hidden_dims_p: tuple,
    weight_decay: float,
    device: torch.device,
    model_save_dir: Path,
    base_lr: float,
    patience: int = 10,
    ) -> None:
    model = TechNB_r_p(
            dim_x=dim_x,
            hidden_dims_r=hidden_dims_r,
            hidden_dims_p=hidden_dims_p
            ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    warmup_epochs = 2
    total_epochs = 50
    cosine_epochs = total_epochs - warmup_epochs

    warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs,
            )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs])


    best_val_log_likelihood = float("-inf")
    best_val_conc_loss = float("inf")
    epochs_no_improve = 0

    start_time = time.time()
    logger.info("Starting training...")

    for epoch in range(total_epochs):
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

            r_1, p_1 = model(x_i_r1)
            r_2, p_2 = model(x_i_r2)
            ll_1 = nb_log_prob_r_p(
                    y=y_i_r1,
                    r=r_1,
                    p=p_1,
                    ).mean()
            ll_2 = nb_log_prob_r_p(
                    y=y_i_r2,
                    r=r_2,
                    p=p_2,
                    ).mean()

            mu_1 = torch.distributions.NegativeBinomial(
                    total_count=r_1,
                    probs=p_1
                    ).mean.squeeze()
            mu_2 = torch.distributions.NegativeBinomial(
                    total_count=r_2,
                    probs=p_2
                    ).mean.squeeze()
            residual_1 = y_i_r1 - mu_1
            residual_2 = sd_ratio_r1_to_r2 * (y_i_r2 - mu_2)
            # TODO: What should tau be?
            conc_loss = concordance_loss_NCE(residual_1, residual_2, tau=0.1)
            ll_loss = (-ll_1 - ll_2).mean()


            if epoch == 0:
                loss = ll_loss
            else:
                # Concordance loss is on order of ~18 for H3K27me3
                loss = ll_loss + 0.2 * conc_loss


            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

            if batch_idx % 1_000 == 0:  # Log every 1,000 batches
                # print("---Train---")
                # print("xi", x_i.shape, "yi", y_i.shape)
                # print("r1", r_1.shape, "p1", p_1.shape)
                # print("r2", r_2.shape, "p2", p_2.shape)
                # print("ll1", ll_1.shape, "ll2", ll_2.shape)
                # print("mu1", mu_1.shape, "mu2", mu_2.shape)
                # print("res_1", residual_1.shape, "res2", residual_2.shape)
                # print("sd_ratio", sd_ratio_r1_to_r2.shape)
                # print("-----------")
                logger.info(
                        "Epoch %s/%s, Batch %s/%s, Train Loss: %s. Log likelihood R1: %s, Log likelihood R2: %s, Concordance loss: %s, R1 residuals: Min=%s, 25Q=%s, 50Q=%s, 75Q=%s, Max=%s, R2 residuals: Min=%s, 25Q=%s, 50Q=%s, 75Q=%s, Max=%s",
                    epoch + 1,
                    total_epochs,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                    ll_1.item(),
                    ll_2.item(),
                    conc_loss.item(),
                    residual_1.min().item(),
                    residual_1.quantile(0.25).item(),
                    residual_1.median().item(),
                    residual_1.quantile(0.75).item(),
                    residual_1.max().item(),
                    residual_2.min().item(),
                    residual_2.quantile(0.25).item(),
                    residual_2.median().item(),
                    residual_2.quantile(0.75).item(),
                    residual_2.max().item(),
                )

        avg_train_loss = total_train_loss / train_batches

        model.eval()
        total_val_ll_loss = 0
        total_val_conc_loss = 0
        val_batches = 0

        with torch.no_grad():
            all_val_x_1 = []
            all_val_x_2 = []
            all_val_y_1 = []
            all_val_y_2 = []
            all_val_r_1 = []
            all_val_p_1 = []
            all_val_r_2 = []
            all_val_p_2 = []
            all_val_ll_1 = []
            all_val_ll_2 = []
            all_val_mu_1 = []
            all_val_mu_2 = []
            all_val_residual_1 = []
            all_val_residual_2 = []
            all_sd_ratio_r1_to_r2 = []
            for batch_idx, (x_i_val, y_i_val) in enumerate(val_loader):
                x_i_val, y_i_val = x_i_val.to(device), y_i_val.to(device).float()  # noqa: PLW2901
                x_i_val_r1 = x_i_val[:, :5]
                x_i_val_r2 = x_i_val[:, 5:]
                y_i_val_r1 = y_i_val[:, 0]
                y_i_val_r2 = y_i_val[:, 1]
                val_sd_ratio_r1_to_r2 = y_i_val[:, 2]

                val_r_1, val_p_1 = model(x_i_val_r1)
                val_r_2, val_p_2 = model(x_i_val_r2)
                val_ll_1 = nb_log_prob_r_p(
                        y=y_i_val_r1.squeeze(),
                        r=val_r_1.squeeze(),
                        p=val_p_1.squeeze(),
                        )
                val_ll_2 = nb_log_prob_r_p(
                        y=y_i_val_r2.squeeze(),
                        r=val_r_2.squeeze(),
                        p=val_p_2.squeeze(),
                        )
                val_mu_1 = torch.distributions.NegativeBinomial(
                        total_count=val_r_1.squeeze(),
                        probs=val_p_1.squeeze(),
                        ).mean.squeeze()
                val_mu_2 = torch.distributions.NegativeBinomial(
                        total_count=val_r_2.squeeze(),
                        probs=val_p_2.squeeze(),
                        ).mean.squeeze()
                val_residual_1 = y_i_val_r1.squeeze() - val_mu_1.squeeze()
                val_residual_2 = val_sd_ratio_r1_to_r2.squeeze() * (y_i_val_r2.squeeze() - val_mu_2.squeeze())

                val_ll_loss = (- val_ll_1.mean() - val_ll_2.mean()).mean()
                val_conc_loss = concordance_loss_NCE(
                        val_residual_1,
                        val_residual_2,
                        tau=0.1
                        )
                # print("---Val---")
                # print("xi", x_i_val.shape, "yi", y_i_val.shape)
                # print("r1", val_r_1.shape, "p1", val_p_1.shape)
                # print("r2", val_r_2.shape, "p2", val_p_2.shape)
                # print("ll1", val_ll_1.shape, "ll2", val_ll_2.shape)
                # print("mu1", val_mu_1.shape, "mu2", val_mu_2.shape)
                # print("res_1", val_residual_1.shape, "res2", val_residual_2.shape)
                # print("sd_ratio", val_sd_ratio_r1_to_r2.shape)
                # print("-----------")
                # 1/0

                total_val_ll_loss += val_ll_loss.item()
                total_val_conc_loss += val_conc_loss.item()

                all_val_x_1.append(x_i_val_r1.detach().cpu().numpy())
                all_val_x_2.append(x_i_val_r2.detach().cpu().numpy())
                all_val_y_1.append(y_i_val_r1.detach().cpu().numpy())
                all_val_y_2.append(y_i_val_r2.detach().cpu().numpy())
                all_val_r_1.append(val_r_1.detach().cpu().numpy())
                all_val_p_1.append(val_p_1.detach().cpu().numpy())
                all_val_r_2.append(val_r_2.detach().cpu().numpy())
                all_val_p_2.append(val_p_2.detach().cpu().numpy())
                all_val_ll_1.append(val_ll_1.detach().cpu().numpy())
                all_val_ll_2.append(val_ll_2.detach().cpu().numpy())
                all_val_mu_1.append(val_mu_1.detach().cpu().numpy())
                all_val_mu_2.append(val_mu_2.detach().cpu().numpy())
                all_val_residual_1.append(val_residual_1.detach().cpu().numpy())
                all_val_residual_2.append(val_residual_2.detach().cpu().numpy())
                all_sd_ratio_r1_to_r2.append(val_sd_ratio_r1_to_r2.detach().cpu().numpy())


                val_batches += 1

            avg_val_log_likelihood = total_val_ll_loss / val_batches
            avg_val_conc_loss = total_val_conc_loss / val_batches

            if epoch == 0:
                all_val_x_1 = np.concatenate(all_val_x_1[:-1])
                all_val_x_2 = np.concatenate(all_val_x_2[:-1])
                all_val_y_1 = np.concatenate(all_val_y_1[:-1])
                all_val_y_2 = np.concatenate(all_val_y_2[:-1])
                for arr, arr_name in zip([all_val_x_1, all_val_x_2, all_val_y_1, all_val_y_2], ['x_1', 'x_2', 'y_1', 'y_2']):
                    with open(model_save_dir / f'val_{arr_name}.npy', 'wb') as f:
                        np.save(f, arr)

            all_val_r_1 = np.concatenate(all_val_r_1[:-1])
            all_val_p_1 = np.concatenate(all_val_p_1[:-1])
            all_val_r_2 = np.concatenate(all_val_r_2[:-1])
            all_val_p_2 = np.concatenate(all_val_p_2[:-1])
            all_val_ll_1 = np.concatenate(all_val_ll_1[:-1])
            all_val_ll_2 = np.concatenate(all_val_ll_2[:-1])
            all_val_mu_1 = np.concatenate(all_val_mu_1[:-1])
            all_val_mu_2 = np.concatenate(all_val_mu_2[:-1])
            all_val_residual_1 = np.concatenate(all_val_residual_1[:-1])
            all_val_residual_2 = np.concatenate(all_val_residual_2[:-1])
            all_sd_ratio_r1_to_r2 = np.concatenate(all_sd_ratio_r1_to_r2[:-1])
            for arr_name, arr in zip(['r_1', 'p_1', 'r_2', 'p_2', 'll_1', 'll_2', 'mu_1', 'mu_2', 'residual_1', 'residual_2', 'sd_ratio_r1_to_r2'], [all_val_r_1, all_val_p_1, all_val_r_2, all_val_p_2, all_val_ll_1, all_val_ll_2, all_val_mu_1, all_val_mu_2, all_val_residual_1, all_val_residual_2, all_sd_ratio_r1_to_r2]):
                with open(model_save_dir / f'val_{arr_name}_{epoch}.npy', 'wb') as f:
                    np.save(f, arr)


        scheduler.step()
        epoch_duration = time.time() - start_time

        logger.info(
                "Epoch %s/%s Summary: Duration: %s, Avg Train Loss: %s, Avg Val Log-Likelihood: %s, Avg Val Concordance Loss: %s, R1 residuals (last val batch): Min=%s, 25Q=%s, 50Q=%s, 75Q=%s, Max=%s, R2 residuals (last val batch): Min=%s, 25Q=%s, 50Q=%s, 75Q=%s, Max=%s",
            epoch + 1,
            total_epochs,
            epoch_duration,
            avg_train_loss,
            avg_val_log_likelihood,
            avg_val_conc_loss,
            val_residual_1.min().item(),  # type: ignore
            val_residual_1.quantile(0.25).item(),  # type: ignore
            val_residual_1.median().item(),  # type: ignore
            val_residual_1.quantile(0.75).item(),  # type: ignore
            val_residual_1.max().item(),  # type: ignore
            val_residual_2.min().item(),  # type: ignore
            val_residual_2.quantile(0.25).item(),  # type: ignore
            val_residual_2.median().item(),  # type: ignore
            val_residual_2.quantile(0.75).item(),  # type: ignore
            val_residual_2.max().item(),  # type: ignore
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
            if avg_val_conc_loss < best_val_conc_loss:
                best_val_conc_loss = avg_val_conc_loss
                epochs_no_improve = 0
                # Save the best model state
                best_conc_model_state = copy.deepcopy(model.state_dict())
                model_save_path = model_save_dir / "best_concordance.pt"
                torch.save(best_conc_model_state, model_save_path)
                logger.info(
                    "New best validation concordance: %s. Model saved to %s",
                    best_val_conc_loss,
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


def concordance_loss_NCE(res1, res2, tau=0.1):
    """
    res1, res2:  (B,) tensors of residuals from two replicates.
    """
    z1 = res1.unsqueeze(1)          # (B,1)
    z2 = res2.unsqueeze(0)          # (1,B)
    sim = torch.exp(-(z1 - z2).pow(2) / tau)  # Gaussian similarity, (B,B)
    pos = sim.diag()                # (B,)
    eps = 1e-8
    loss = -torch.log((pos + eps) / (sim.sum(dim=1) + eps))
    return loss.mean()



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
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()

    # data = load_data(args.target)
    # n_train = int(0.8 * len(data["control_mapq_r1"]))
    # train_ds = MultiReplicateDataset(**{k: v[:n_train] for k, v in data.items()})
    # val_ds = MultiReplicateDataset(**{k: v[n_train:] for k, v in data.items()})
    train_ds, val_ds = load_data(args.target)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8_192, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8_192, shuffle=False)

    base_save_dir = DATA_DIR / f"runs/rp_info_loss/{args.target}"
    save_dir = base_save_dir / f"run_{args.run_name}"
    save_dir.mkdir(parents=True)
    print(f"SAVING TO: {save_dir}")


    build_and_train(
            dim_x=train_ds.get_dim_x(),
            train_loader=train_loader,
            val_loader=val_loader,
            hidden_dims_r=(32, 32),
            hidden_dims_p=(8, 8),
            weight_decay=0.01,
            device=torch.device("cuda:0"),
            model_save_dir=save_dir,
            base_lr=0.001,
            patience=10,
            )




