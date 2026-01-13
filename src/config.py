import os
from dataclasses import dataclass
from typing import Any, Dict

import torch
import yaml

@dataclass
class Config:
    DATA_PATH: str
    LABEL_PATH: str
    RESULTS_DIR: str
    IMG_SIZE: int
    NUM_BANDS: int
    SEED: int
    THRESH: float
    NUM_EPOCHS: int
    BATCH_SIZE: int
    LEARNING_RATE: float
    WGAN_GP_LAMBDA: float
    L1_LOSS_WEIGHT: float
    GRAD_CLIP_VALUE: float
    DEVICE: str = "auto"

    def resolve_device(self) -> torch.device:
        if self.DEVICE == "cpu":
            return torch.device("cpu")
        if self.DEVICE == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        d: Dict[str, Any] = yaml.safe_load(f)
    cfg = Config(**d)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    for subdir in ["architecture", "loss_function", "training_strategy", "comparison"]:
        os.makedirs(os.path.join(cfg.RESULTS_DIR, subdir), exist_ok=True)
    return cfg
