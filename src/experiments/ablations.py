import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from ..config import load_config
from ..seed import set_seed
from ..data import WildfireDataset, build_splits
from ..models import (
    UNetChebyKANGenerator, Pix2PixGenerator, CycleGANGenerator, AttentionUNetGenerator, UNetPlusPlusGenerator,
    PatchGANDiscriminator, SpectralNormDiscriminator
)
from ..train import train_wgan_gp, evaluate
from ..viz import save_loss_curve, save_results_table
from ..io_utils import save_json

def _make_loaders(cfg, train_files, test_files):
    (tr_d, tr_l) = train_files
    (te_d, te_l) = test_files
    train_ds = WildfireDataset(cfg.DATA_PATH, cfg.LABEL_PATH, tr_d, tr_l, cfg.NUM_BANDS, cfg.IMG_SIZE)
    test_ds  = WildfireDataset(cfg.DATA_PATH, cfg.LABEL_PATH, te_d, te_l, cfg.NUM_BANDS, cfg.IMG_SIZE)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader

def run_all(config_path: str = "configs/default.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.SEED)
    device = cfg.resolve_device()

    train_files, test_files = build_splits(cfg.DATA_PATH, cfg.LABEL_PATH, train_ratio=0.7, seed=cfg.SEED)
    train_loader, test_loader = _make_loaders(cfg, train_files, test_files)

    # ---- Experiment grid (compact starter set) ----
    experiments = [
        ("unet_chebykan_encoder", lambda: UNetChebyKANGenerator(cfg.NUM_BANDS, 1, "encoder"), lambda: PatchGANDiscriminator(cfg.NUM_BANDS+1)),
        ("unet_chebykan_bottleneck", lambda: UNetChebyKANGenerator(cfg.NUM_BANDS, 1, "bottleneck"), lambda: PatchGANDiscriminator(cfg.NUM_BANDS+1)),
        ("pix2pix", lambda: Pix2PixGenerator(cfg.NUM_BANDS, 1), lambda: PatchGANDiscriminator(cfg.NUM_BANDS+1)),
        ("cyclegan_like", lambda: CycleGANGenerator(cfg.NUM_BANDS, 1), lambda: PatchGANDiscriminator(cfg.NUM_BANDS+1)),
        ("attention_unet", lambda: AttentionUNetGenerator(cfg.NUM_BANDS, 1), lambda: PatchGANDiscriminator(cfg.NUM_BANDS+1)),
        ("unetpp", lambda: UNetPlusPlusGenerator(cfg.NUM_BANDS, 1), lambda: PatchGANDiscriminator(cfg.NUM_BANDS+1)),
        ("unet_chebykan_sn_disc", lambda: UNetChebyKANGenerator(cfg.NUM_BANDS, 1, "encoder"), lambda: SpectralNormDiscriminator(cfg.NUM_BANDS+1)),
    ]

    rows: List[Dict] = []
    for name, make_g, make_d in experiments:
        out_dir = os.path.join(cfg.RESULTS_DIR, "comparison", name)
        os.makedirs(out_dir, exist_ok=True)

        G = make_g()
        D = make_d()

        hist = train_wgan_gp(
            G, D, train_loader, device,
            num_epochs=cfg.NUM_EPOCHS,
            lr=cfg.LEARNING_RATE,
            l1_weight=cfg.L1_LOSS_WEIGHT,
            gp_lambda=cfg.WGAN_GP_LAMBDA,
            grad_clip=cfg.GRAD_CLIP_VALUE,
        )
        metrics = evaluate(G, test_loader, device, thr=cfg.THRESH)

        save_loss_curve(hist, os.path.join(out_dir, "loss_curves.png"), title=name)
        save_json(os.path.join(out_dir, "history.json"), hist)
        save_json(os.path.join(out_dir, "metrics.json"), metrics)

        row = {"experiment": name, **metrics}
        rows.append(row)

    save_results_table(rows, os.path.join(cfg.RESULTS_DIR, "comparison", "summary.xlsx"))
    print("Done. Results written to:", cfg.RESULTS_DIR)
