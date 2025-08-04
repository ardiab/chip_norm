from chipvi.utils.tmux import run_in_tmux
from pathlib import Path
import itertools

TARGET = "h3k27me3"
DEVICE = "cuda:1"
SESSION_NAME = f"0728_loss_comparison_{TARGET}"

if __name__ == "__main__":
    commands = []

    # --- Hyperparameter Grid ---
    consistency_losses = ["infonce", "pearson", "quantile_abs"]
    log_transforms = [True, False]
    mu_dims_options = [(64, 64), (128, 128)]
    r_dims_options = [(16, 16), (32, 32)]
    lrs = [0.0005]
    # Use different weights for different losses as they have different scales
    # This is a starting point; these weights are hyperparameters to tune
    consistency_weights = {
        "infonce": 1.0,
        "pearson": 5.0,
        "quantile_abs": 10.0,
    }

    # --- Base command ---
    base_cmd = (
        f"python {Path(__file__).parent / 'train.py'} "
        f"--target {TARGET} --device {DEVICE} "
        f"--num_epochs 100 --patience 10 --batch_size 8192 "
        f"--run_group {SESSION_NAME}"
    )

    for loss, log_trans, mu_dims, r_dims, lr in itertools.product(
        consistency_losses, log_transforms, mu_dims_options, r_dims_options, lrs
    ):
        
        weight = consistency_weights[loss]

        mu_dim_str = " ".join(map(str, mu_dims))
        r_dim_str = " ".join(map(str, r_dims))
        
        cmd = (
            f"{base_cmd} "
            f"--mu_dims {mu_dim_str} --r_dims {r_dim_str} --lr {lr} "
            f"--consistency_loss {loss} "
            f"--consistency_weight {weight} "
        )
        if log_trans:
            cmd += "--log_transform_inputs "
        
        commands.append(cmd)

    print(f"Dispatching {len(commands)} commands to tmux session '{SESSION_NAME}'...")
    run_in_tmux(SESSION_NAME, commands)