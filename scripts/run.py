import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Import our refactored components
from chipvi.data.datasets import build_datasets # We will refactor this next
from chipvi.training.trainer import Trainer
from chipvi.training.losses import replicate_concordance_mse_loss

# A simple factory to get the loss function from the config string
LOSS_REGISTRY = {
    "replicate_concordance_mse": replicate_concordance_mse_loss,
    # Add other losses here as they are created
}

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for running ChipVI experiments using Hydra.
    """
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))

    # 1. Setup: Device, Paths, etc.
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    train_ds, val_ds = build_datasets(cfg)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # 3. Model Initialization
    # Hydra instantiates the model for us using the _target_ field
    model = hydra.utils.instantiate(
        cfg.model,
        covariate_dim=train_ds.get_covariate_dim()
    ).to(device)

    # 4. Optimizer and Loss Function
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    loss_fn = LOSS_REGISTRY[cfg.loss_fn]

    # 5. Trainer Initialization and Execution
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )
    
    print("\n--- Starting Training ---")
    trainer.fit(num_epochs=cfg.num_epochs)
    print("--- Training Finished ---")

if __name__ == "__main__":
    main()