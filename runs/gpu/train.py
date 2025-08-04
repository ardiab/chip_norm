from chipvi.data.datasets import MultiReplicateDataset
from chipvi.training.tech_biol_rep_model_trainer import build_and_train
from pathlib import Path
import pickle as pkl
import argparse
import torch


DATA_DIR = Path("/lotterlab/users/abdul/repos/chipvi/")


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
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=512, shuffle=False)

    base_save_dir = DATA_DIR / f"runs/gpu/{args.target}"
    # run_num = len(list(base_save_dir.glob("*"))) + 1
    save_dir = base_save_dir / f"run_{args.run_name}"
    save_dir.mkdir(parents=True)
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
            # wandb_name=f"gpu_{args.target}",
            )




