import os
import matplotlib.pyplot as plt

def save_loss_curve(history: dict, out_path: str, title: str = "Training Curves"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
