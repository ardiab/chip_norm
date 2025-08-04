from chipvi.utils.tmux import run_in_tmux
from pathlib import Path
import itertools

TARGET = "h3k27me3"
DEVICE = "cuda:0"
SESSION_NAME = f"0727_optimize_cdf_diffs_{TARGET}"


if __name__ == "__main__":
    commands = []

    mu_dims = [(32, 32), (64, 64)]
    r_dims = [(8, 8), (16, 16)]
    weight_decays = [0, 0.01]
    lrs = [0.001, 0.0001]
    warmup_epochs = [2]
    num_epochs = 100
    patience = 10
    batch_size = 8_192
    for mu_dim, r_dim, weight_decay, lr, warmup_epoch in itertools.product(mu_dims, r_dims, weight_decays, lrs, warmup_epochs):
        mu_dim_str = " ".join(map(str, mu_dim))
        r_dim_str = " ".join(map(str, r_dim))
        commands.append(
            f"python {Path(__file__).parent / 'train.py'} --target {TARGET} --device {DEVICE} --mu_dims {mu_dim_str} --r_dims {r_dim_str} --weight_decay {weight_decay} --lr {lr} --warmup_epochs {warmup_epoch} --num_epochs {num_epochs} --patience {patience} --run_group {SESSION_NAME} --batch_size {batch_size}"
        )
    run_in_tmux(SESSION_NAME, commands)
