# 🔥 ChebyKANGAN: A Polynomially Adaptive GAN for Wildfire Burned Area Segmentation  
### Comprehensive Ablation Study & Literature Comparison

This repository provides a **comprehensive experimental analysis** of **ChebyKANGAN**, a GAN-based architecture developed for **automatic burned area segmentation from multispectral satellite imagery**, enhanced with **Chebyshev Kolmogorov–Arnold Network (ChebyKAN)** integration.

This repo is prepared to support the following paper:

> **“ChebyKANGAN: A Polynomially Adaptive GAN for Wildfire Burned Area Segmentation”**  
> *(Manuscript under review)*

The study is designed to deliver **quantitative and qualitative comparisons** across architectural choices, loss functions, training strategies, and widely-used baseline methods from the literature.

---

## 📌 Objectives

The main objectives of this work are:

- To investigate **where ChebyKAN modules are most effective** within UNet-based segmentation architectures  
- To analyze the impact of different **pixel-level loss functions** (L1, Dice, Focal, etc.) on segmentation performance  
- To compare different optimization and learning strategies in terms of **convergence, stability, and efficiency**  
- To fairly benchmark the proposed best configuration against **commonly used GAN-based methods** in the literature  

---

## 🧠 Study Scope

This repository executes the following experimental stages in a **single end-to-end pipeline**:

### 1️⃣ Architecture Ablation (ChebyKAN Integration)
ChebyKAN layers are integrated into a UNet-like generator at different locations:

- Encoder
- Decoder
- Encoder + Decoder
- Bottleneck
- Encoder + Bottleneck
- Bottleneck + Decoder  

A total of **6 architectural configurations** are evaluated.

---

### 2️⃣ Loss Function Ablation
Using the best architecture, multiple pixel-level loss functions are tested:

- L1 Loss  
- Dice Loss  
- Focal Loss  

---

### 3️⃣ Training Strategy Ablation
Using the best architecture + best loss function, multiple training strategies are compared:

- Adam (standard)
- Adam (higher learning rate)
- RMSProp
- SGD + Cosine Annealing  

---

### 4️⃣ Literature Comparison
The best proposed model is compared with the following baseline methods:

- Pix2Pix GAN  
- CycleGAN  
- Attention UNet GAN  
- UNet++ GAN  
- WGAN (Spectral Normalization)  

---

## 📦 Dataset & Data Release Policy

This repository provides a **small sample dataset** for reproducibility and quick testing.

### ✅ Sample dataset included in this repo (GitHub)
Only **27 image-mask pairs** are included in this repository:

- `data/sample/images/` → 27 multispectral images  
- `data/sample/masks/`  → 27 corresponding segmentation masks  

This sample is provided to:
- validate that the pipeline runs correctly,
- demonstrate the required folder structure,
- enable fast demo/testing scenarios.

### 🌍 Full dataset (Kaggle)
Due to GitHub size limitations, the full dataset is **not** included in this repository.  
The complete dataset will be published via Kaggle:

- Kaggle Dataset Link: **https://www.kaggle.com/datasets/bingolo/wildfire-burned-area-segmentation-dataset/data**

After downloading the full dataset, it should follow this structure:

```text
data/
  images/
  masks/
```

Then, you only need to update the dataset paths in the configuration file.

---

## 📂 Project Structure

```text
ChebyKANGAN/
│
├── data/
│   ├── sample/
│   │   ├── images/               # 27 sample multispectral images
│   │   └── masks/                # 27 corresponding masks
│
├── configs/
│   └── default.yaml              # Paths + experiment hyperparameters
│
├── src/
│   ├── config.py                 # Config loader
│   ├── seed.py                   # Reproducibility (seed control)
│   ├── io_utils.py               # JSON / file utilities
│   │
│   ├── data/
│   │   ├── dataset.py            # WildfireDataset class
│   │   └── splits.py             # Train/test split + NaN checks
│   │
│   ├── losses/
│   │   └── segmentation_losses.py
│   │
│   ├── models/
│   │   ├── blocks.py             # ChebyKAN, Attention, Residual blocks
│   │   ├── generators.py         # UNet, UNet++, Pix2Pix, CycleGAN, etc.
│   │   └── discriminators.py
│   │
│   ├── train/
│   │   ├── gan_trainer.py        # Training loop
│   │   └── evaluator.py          # Evaluation logic
│   │
│   ├── metrics/
│   │   └── segmentation_metrics.py
│   │
│   ├── vis/
│   │   ├── plots.py              # Curves + correlation plots
│   │   └── tables.py             # Excel/CSV export
│   │
│   └── experiments/
│       └── ablations.py          # Full experiment pipeline
│
├── run.py                        # Main entry point
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Outputs Produced

For each ablation study and comparison, the pipeline automatically produces:

### 📈 Figures
- Training loss curves (Generator & Discriminator)
- Convergence analysis (combined loss, G/D ratio)
- Performance correlations (F1 vs IoU, Precision vs Recall)
- Metric comparison bar charts and heatmaps
- Qualitative sample predictions (visual outputs)
- Radar charts for top-5 models

### 📄 Tables
- Excel (`.xlsx`) exports including:
  - Test metrics
  - Model rankings (Overall Rank)
  - Training time statistics
  - Parameter count statistics

### 🧾 Reports
- `summary_report.json`  
- `COMPREHENSIVE_RESULTS.xlsx`

---

## ⚙️ Installation

```bash
git clone <repo_url>
cd ChebyKANGAN

python -m venv .venv
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

> **Note:** `rasterio` is required for `.tif/.tiff` files.  
> If you use `.npy` format only, `rasterio` is not mandatory.

---

## 🚀 Running Experiments

### 1) Run with the sample dataset (27 pairs)
Update `configs/default.yaml` as follows:

```yaml
DATA_PATH: "data/sample/images"
LABEL_PATH: "data/sample/masks"
```

Then run:

```bash
python run.py
```

---

### 2) Run with the full dataset (Kaggle)
Download the full dataset and place it as:

```text
data/images/
data/masks/
```

Then update the config:

```yaml
DATA_PATH: "data/images"
LABEL_PATH: "data/masks"
```

Run:

```bash
python run.py
```


---

## 🔁 Reproducibility

This repository is designed with reproducibility in mind:

- fixed random `SEED` usage  
- deterministic train/test split  
- automatic logging of all experiments  
- automated metric reporting and exports  

This setup is suitable for **academic publications**, **thesis work**, and **benchmarking**.

---

## 📜 License

### Code License
The source code in this repository is released under the **MIT License**.  

### Dataset License
The wildfire burned area segmentation dataset is released under the  
**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

- Kaggle release: CC BY-NC 4.0  
- GitHub sample data (`data/sample/`): CC BY-NC 4.0  


## 📚 Academic Use & Citation

This work is intended to serve as a reference for research in:
**wildfire segmentation**, **remote sensing**, **deep learning-based segmentation**, and **KAN-based architectures**.

If you use the dataset and codes, please cite the following paper:

### BibTeX (Paper under review)
```bibtex
@article{chebykangan2026,
  title   = {ChebyKANGAN: A Polynomially Adaptive GAN for Wildfire Burned Area Segmentation},
  author  = {Under Review (Double Blind) },
  journal = {Under Review},
  year    = {2026}
}
```

---

## ⚠️ Notes & Limitations

- The pipeline assumes the multispectral input has a number of bands consistent with `NUM_BANDS`.
- Some datasets store masks in 0/255 format; the code automatically normalizes masks into 0/1 format.
- Training time may significantly vary depending on GPU hardware and dataset size.

---

## 📬 Contact

For questions, improvement suggestions, or academic collaboration:

📧 **email : Under Review (Double Blind)**
