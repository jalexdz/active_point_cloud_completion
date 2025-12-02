import os
import torch

from apcc.cfg import load_cfg
from apcc.data.dataset import MVPSequenceDataset
from apcc.models.apcc_model import APCCModel
from apcc.training.training import train_model


def main():
    cfg = load_cfg("configs/train.yaml")

    os.makedirs(cfg.logging.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.logging.logdir, exist_ok=True)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # dataset
    train_dataset = MVPSequenceDataset(
        prefix=cfg.data.split,           
        num_views=cfg.data.num_views,    
        data_root=cfg.data.root,
        views_per_object=cfg.data.views_per_object,
    )

    val_dataset = MVPSequenceDataset(
        prefix="val",
        num_views=cfg.data.num_views,
        data_root=cfg.data.root,
        views_per_object=cfg.data.views_per_object,
    )
    # model
    model = APCCModel(cfg.model)

    # train
    train_model(model, train_dataset, val_dataset, cfg, device)


if __name__ == "__main__":
    main()