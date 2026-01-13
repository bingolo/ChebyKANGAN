import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def gradient_penalty(D, real_x, real_y, fake_y, device):
    # interpolate between real and fake masks
    alpha = torch.rand(real_x.size(0), 1, 1, 1, device=device)
    interp_y = alpha * real_y + (1 - alpha) * fake_y
    interp_y.requires_grad_(True)

    d_out = D(real_x, interp_y)
    grad = torch.autograd.grad(
        outputs=d_out,
        inputs=interp_y,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(grad.size(0), -1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def train_wgan_gp(
    G,
    D,
    train_loader,
    device,
    num_epochs: int,
    lr: float,
    l1_weight: float,
    gp_lambda: float,
    grad_clip: float,
) -> Dict[str, List[float]]:
    G.to(device); D.to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    l1 = nn.L1Loss()

    history = {"G_total": [], "D_total": [], "G_l1": [], "G_adv": []}

    for ep in range(1, num_epochs + 1):
        G.train(); D.train()
        g_total = d_total = g_l1 = g_adv = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{num_epochs}", leave=False)
        for x, y, _ in pbar:
            x = x.to(device)
            y = y.to(device)

            # ---- Train D ----
            with torch.no_grad():
                fake = G(x)
            d_real = D(x, y).mean()
            d_fake = D(x, fake).mean()
            gp = gradient_penalty(D, x, y, fake, device)
            loss_d = -(d_real - d_fake) + gp_lambda * gp

            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), grad_clip)
            opt_d.step()

            # ---- Train G ----
            fake = G(x)
            adv = -D(x, fake).mean()
            l1loss = l1(fake, y)
            loss_g = adv + l1_weight * l1loss

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), grad_clip)
            opt_g.step()

            n_batches += 1
            g_total += loss_g.item()
            d_total += loss_d.item()
            g_l1 += l1loss.item()
            g_adv += adv.item()

            pbar.set_postfix({"G": loss_g.item(), "D": loss_d.item()})

        history["G_total"].append(g_total / max(1, n_batches))
        history["D_total"].append(d_total / max(1, n_batches))
        history["G_l1"].append(g_l1 / max(1, n_batches))
        history["G_adv"].append(g_adv / max(1, n_batches))

    return history
