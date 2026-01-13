import torch

def compute_metrics(pred_bin: torch.Tensor, gt_bin: torch.Tensor, eps: float = 1e-7):
    # pred_bin/gt_bin: [B,1,H,W] in {0,1}
    pred = pred_bin.float()
    gt = gt_bin.float()

    tp = (pred * gt).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()
    tn = ((1 - pred) * (1 - gt)).sum().item()

    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = (2 * prec * rec) / (prec + rec + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "Accuracy": float(acc),
    }
