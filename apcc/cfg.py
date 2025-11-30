from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

@dataclass
class ModelCfg:
    enc_feat_dim: int = 256
    gru_hidden_dim: int = 512
    gru_layers: int = 1
    gru_dropout: float = 0.0
    dec_mlp_hidden_dim: int = 512
    dec_num_layers: int = 3

@dataclass
class DataCfg:
    root: str = "./data/"
    split: str = "train"
    num_views: int = 4
    num_points_partial: int = 2048
    num_points_complete: int = 16384
    normalize: bool = True # unit volume
    views_per_object: int = 26
    num_queries: int = 512

@dataclass
class TrainCfg:
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    amp: bool = False            
    grad_clip_norm: Optional[float] = None

@dataclass
class LoggingCfg:
    logdir: str = "./runs/apcc"
    ckpt_dir: str = "./checkpoints"
    ckpt_every: int = 5           # epochs
    print_every: int = 10         # batches
    save_last: bool = True

@dataclass
class ExpCfg:
    """Top-level config object passed around the codebase."""
    model: ModelCfg
    data: DataCfg
    train: TrainCfg
    logging: LoggingCfg
    seed: int = 42
    exp_name: str = "apcc_default"


def _update_dataclass(dc_cls, values: dict):
    """Helper: fill a dataclass from a dict, keeping defaults where keys are missing."""
    fields = {f.name for f in dc_cls.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in values.items() if k in fields}
    return dc_cls(**kwargs)


def load_cfg(path: str | Path) -> ExpCfg:
    """
    Load a YAML config file and return an ExpCfg instance.

    Expected YAML structure (example):

    model:
      enc_feat_dim: 256
      gru_hidden_dim: 512
      ...

    data:
      root: ./data
      num_views: 4
      ...

    train:
      batch_size: 8
      epochs: 50
      ...

    logging:
      logdir: ./runs/apcc
      ...

    seed: 123
    exp_name: my_experiment
    """
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    model_cfg = _update_dataclass(ModelCfg, raw.get("model", {}))
    data_cfg = _update_dataclass(DataCfg, raw.get("data", {}))
    train_cfg = _update_dataclass(TrainCfg, raw.get("train", {}))
    log_cfg = _update_dataclass(LoggingCfg, raw.get("logging", {}))

    seed = raw.get("seed", 42)
    exp_name = raw.get("exp_name", path.stem)

    return ExpCfg(
        model=model_cfg,
        data=data_cfg,
        train=train_cfg,
        logging=log_cfg,
        seed=seed,
        exp_name=exp_name,
    )