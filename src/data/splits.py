import os
import numpy as np
from tqdm import tqdm

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except Exception:
    RASTERIO_AVAILABLE = False

def _list_files(p):
    return sorted([f for f in os.listdir(p) if f.lower().endswith((".tif", ".tiff", ".npy"))])

def check_nan_files(data_path: str, label_path: str):
    data_files = _list_files(data_path)
    label_files = _list_files(label_path)
    if len(data_files) == 0 or len(label_files) == 0:
        raise ValueError(f"No files! Data:{len(data_files)} Label:{len(label_files)}")

    n = min(len(data_files), len(label_files))
    valid_data, valid_label, corrupted = [], [], []

    print("Scanning for NaN/Inf...")
    for i in tqdm(range(n), desc="File check"):
        df, lf = data_files[i], label_files[i]
        try:
            dp = os.path.join(data_path, df)
            lp = os.path.join(label_path, lf)

            if df.endswith(".npy"):
                d = np.load(dp).astype(np.float32)
            else:
                if not RASTERIO_AVAILABLE:
                    corrupted.append((df, lf, "rasterio missing")); continue
                with rasterio.open(dp) as src:
                    d = src.read().astype(np.float32)

            if np.isnan(d).any() or np.isinf(d).any():
                corrupted.append((df, lf, "data NaN/Inf")); continue

            if lf.endswith(".npy"):
                lab = np.load(lp).astype(np.float32)
                if lab.ndim == 3:
                    lab = lab[0]
            else:
                if not RASTERIO_AVAILABLE:
                    corrupted.append((df, lf, "rasterio missing")); continue
                with rasterio.open(lp) as src:
                    lab = src.read(1).astype(np.float32)

            if np.isnan(lab).any() or np.isinf(lab).any():
                corrupted.append((df, lf, "label NaN/Inf")); continue

            valid_data.append(df); valid_label.append(lf)
        except Exception as e:
            corrupted.append((df, lf, f"read err: {str(e)[:120]}"))

    print(f"Valid: {len(valid_data)} | Corrupted: {len(corrupted)}")
    return valid_data, valid_label, corrupted

def build_splits(data_path: str, label_path: str, train_ratio=0.7, seed=42):
    data_files, label_files, _ = check_nan_files(data_path, label_path)
    n = len(data_files)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    split = int(n * train_ratio)
    tr, te = idx[:split], idx[split:]
    return ([data_files[i] for i in tr], [label_files[i] for i in tr]), \
           ([data_files[i] for i in te], [label_files[i] for i in te])
