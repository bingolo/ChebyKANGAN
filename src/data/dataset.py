import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except Exception:
    RASTERIO_AVAILABLE = False

class WildfireDataset(Dataset):
    def __init__(self, data_path, label_path, data_files, label_files, num_bands: int, img_size: int):
        self.dp, self.lp = data_path, label_path
        self.dfs, self.lfs = data_files, label_files
        self.num_bands = num_bands
        self.img_size = img_size

    def __len__(self):
        return len(self.dfs)

    def _read_arr(self, fp, is_label=False):
        if fp.endswith(".npy"):
            arr = np.load(fp).astype(np.float32)
            if is_label and arr.ndim == 3:
                arr = arr[0]
        else:
            if not RASTERIO_AVAILABLE:
                raise RuntimeError("rasterio not available but .tif requested.")
            with rasterio.open(fp) as src:
                arr = src.read(1).astype(np.float32) if is_label else src.read().astype(np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def __getitem__(self, i):
        dp = os.path.join(self.dp, self.dfs[i])
        lp = os.path.join(self.lp, self.lfs[i])

        img = self._read_arr(dp, is_label=False)
        lab_raw = self._read_arr(lp, is_label=True)

        if img.ndim == 2:
            img = np.expand_dims(img, 0)
        if img.ndim == 3 and img.shape[0] != self.num_bands and img.shape[-1] == self.num_bands:
            img = np.transpose(img, (2, 0, 1))

        if img.shape[0] < self.num_bands:
            pad = np.zeros((self.num_bands - img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
            img = np.concatenate([img, pad], axis=0)
        elif img.shape[0] > self.num_bands:
            img = img[:self.num_bands]

        # per-band robust scaling
        for b in range(self.num_bands):
            band = img[b]
            p1, p99 = np.percentile(band, 1), np.percentile(band, 99)
            if p99 > p1:
                band = np.clip((band - p1) / (p99 - p1), 0, 1)
            else:
                band = np.zeros_like(band)
            img[b] = band

        lscale = 255.0 if lab_raw.max() > 1.5 else 1.0
        lab_01 = (lab_raw / lscale).astype(np.float32)
        lab_bin = (lab_01 > 0.5).astype(np.float32)

        img_t = torch.from_numpy(img).float().unsqueeze(0)
        lab_t = torch.from_numpy(lab_bin).float().unsqueeze(0).unsqueeze(0)

        img_t = F.interpolate(img_t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        lab_t = F.interpolate(lab_t, size=(self.img_size, self.img_size), mode="nearest")

        lsc_t = torch.tensor([lscale], dtype=torch.float32)
        return img_t.squeeze(0), lab_t.squeeze(0), lsc_t
