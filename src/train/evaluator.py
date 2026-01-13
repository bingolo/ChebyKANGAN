import torch
from tqdm import tqdm

from ..metrics.segmentation_metrics import compute_metrics

@torch.no_grad()
def evaluate(G, test_loader, device, thr: float = 0.5):
    G.to(device)
    G.eval()

    agg = {"IoU": 0.0, "Dice": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0, "Accuracy": 0.0}
    n = 0

    for x, y, _ in tqdm(test_loader, desc="Eval", leave=False):
        x = x.to(device); y = y.to(device)
        p = G(x)
        pb = (p > thr).float()
        m = compute_metrics(pb, y)
        for k in agg:
            agg[k] += m[k]
        n += 1

    for k in agg:
        agg[k] /= max(1, n)
    return agg
