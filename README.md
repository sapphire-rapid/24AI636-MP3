# 🚗 Mini Project 3 — Autoencoder + WGAN-GP on DVM-Car Dataset

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?style=flat-square&logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-RTX%203060-76B900?style=flat-square&logo=nvidia)
![Course](https://img.shields.io/badge/Course-24AI636-blueviolet?style=flat-square)

> A deep learning project that trains a **Convolutional Autoencoder** and a **WGAN-GP (Wasserstein GAN with Gradient Penalty)** on the DVM-Car dataset to learn compact latent representations and generate realistic car front-view images.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Setup & Installation](#-setup--installation)
- [Configuration](#-configuration)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Results](#-results)
- [Dependencies](#-dependencies)

---

## 🧠 Overview

This project implements a two-stage generative pipeline:

1. **Stage 1 — Autoencoder**: A convolutional encoder-decoder network trained to compress 64×64 RGB car images into a compact 128-dimensional latent space using MSE reconstruction loss.

2. **Stage 2 — WGAN-GP**: A Wasserstein GAN with Gradient Penalty trained to generate photorealistic car images from random noise vectors, using the DCGAN-style architecture.

3. **Evaluation**: Trained models are evaluated using:
   - **FID Score** (Fréchet Inception Distance) — image quality metric
   - **PCA** — 2D projection of the latent space
   - **t-SNE** — non-linear latent space visualization

---

## 📁 Project Structure

```
SCAFFOLD_R3/
│
├── config.py           # Central config: paths, hyperparameters, device
├── dataset.py          # DVMCarDataset class + DataLoader factory
├── models.py           # Encoder, Decoder, Autoencoder, Generator, Critic
├── train_ae.py         # Autoencoder training loop + visualization
├── train_gan.py        # WGAN-GP training loop with gradient penalty
├── evaluate.py         # FID score, PCA, t-SNE evaluation pipeline
├── main.ipynb          # Main Jupyter notebook (end-to-end orchestration)
├── requirements.txt    # Python dependencies
│
├── Dataset/
│   └── car_fronts/     # Images organized as: brand → year → image.jpg
│       ├── Audi/
│       │   ├── 2015/
│       │   └── ...
│       └── ...
│
└── DL_R3_Git/          # Git tracking folder
```

---

## 🏗️ Architecture

### Autoencoder

```
Input [3, 64, 64]
    │
    ▼  Conv2d blocks (64 → 128 → 256 → 512 channels)
    │  + BatchNorm + ReLU
    ▼
Flatten → Linear → Latent z [128d]
    │
    ▼  Linear → Unflatten
    │  ConvTranspose2d blocks (512 → 256 → 128 → 64 → 3)
    │  + BatchNorm + ReLU → Tanh
    ▼
Reconstruction [3, 64, 64]
```

**Loss**: MSE (Mean Squared Error) between input and reconstruction.

---

### WGAN-GP

```
Noise z [128d]
    │
    ▼  Generator (ConvTranspose2d: 512→256→128→64→3 + Tanh)
    ▼
Fake Image [3, 64, 64]
    │
    ▼  Critic (Conv2d: 64→128→256→512→1 + LeakyReLU, NO Sigmoid)
    ▼
Wasserstein Score
```

**Loss**:
- Critic: `-(E[real] - E[fake]) + λ * GradientPenalty`
- Generator: `-E[critic(fake)]`
- λ (GP penalty weight) = **10**
- Critic updated **5×** per generator step

---

## 📦 Dataset

The project uses the **DVM-Car Dataset** — a large-scale car image dataset organized by brand and model year.

- **Path**: `Dataset/car_fronts/<brand>/<year>/*.jpg`
- **Image Format**: JPG, JPEG, PNG, WebP (auto-detected)
- **Preprocessing**:
  - Resize to 64×64
  - Normalize to `[-1, 1]` (mean=0.5, std=0.5 per channel)
- **Full dataset** used locally (no image cap) on RTX 3060

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd SCAFFOLD_R3
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv DLR3
# Windows
DLR3\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ Make sure you have **CUDA-compatible PyTorch** installed for GPU support. Visit [pytorch.org](https://pytorch.org/get-started/locally/) if needed.

### 4. Configure Paths

Edit `config.py` and update the dataset paths:

```python
BASE_DIR  = r"path\to\your\SCAFFOLD_R3\Dataset\R3_DL_DVMCar"
DATA_ROOT = r"path\to\your\SCAFFOLD_R3\Dataset\car_fronts"
```

---

## 🔧 Configuration

All hyperparameters are centralized in `config.py`:

| Parameter | Value | Description |
|---|---|---|
| `IMAGE_SIZE` | 64 | Input/output image resolution |
| `AE_LATENT_DIM` | 128 | Autoencoder bottleneck size |
| `AE_EPOCHS` | 5 | Autoencoder training epochs |
| `AE_LR` | 1e-3 | Autoencoder learning rate (Adam) |
| `AE_BATCH_SIZE` | 64 | Autoencoder batch size |
| `GAN_NOISE_DIM` | 128 | Generator input noise dimension |
| `GAN_EPOCHS` | 5 | GAN training epochs |
| `GAN_LR_G` | 2e-4 | Generator learning rate |
| `GAN_LR_C` | 2e-4 | Critic learning rate |
| `GAN_CRITIC_ITERS` | 5 | Critic steps per generator step |
| `GAN_LAMBDA_GP` | 10 | Gradient penalty coefficient |
| `FID_NUM_SAMPLES` | 1000 | Samples for FID computation |
| `SEED` | 42 | Global random seed |

---

## 🚀 Training

### Option A: Run via Jupyter Notebook (Recommended)

Open `main.ipynb` and run cells sequentially — the notebook orchestrates the full pipeline.

### Option B: Run Scripts Directly

**Train the Autoencoder:**
```bash
python train_ae.py
```

**Train the WGAN-GP:**
```bash
python train_gan.py
```

**Checkpoints** are saved to `Dataset/R3_DL_DVMCar/checkpoints/`:
- `autoencoder.pth`
- `generator.pth`
- `critic.pth`

---

## 📊 Evaluation

After training, run the evaluation pipeline:

```bash
python evaluate.py
```

This will:
1. Load trained `autoencoder.pth` and `generator.pth`
2. Extract encoder latent vectors from the dataset
3. Generate and save **PCA** visualization (`outputs/pca.png`)
4. Generate and save **t-SNE** visualization (`outputs/tsne.png`)
5. Generate 1000 fake images and compute **FID score**

**Output files** are saved to `Dataset/R3_DL_DVMCar/outputs/`:

| File | Description |
|---|---|
| `ae_loss.png` | Autoencoder MSE loss curve |
| `ae_reconstruction.png` | Original vs. reconstructed images |
| `gan_loss.png` | Generator & Critic loss curves |
| `gan_samples.png` | Grid of GAN-generated images |
| `pca.png` | PCA projection of latent space |
| `tsne.png` | t-SNE projection of latent space |

---

## 📈 Results

| Metric | Description |
|---|---|
| AE Reconstruction Loss | MSE on held-out images |
| FID Score | Lower = better image quality (vs real images) |
| PCA / t-SNE | Visual clustering of latent representations by brand |

---

## 📦 Dependencies

| Package | Version |
|---|---|
| `torch` | 2.3.0 |
| `torchvision` | 0.18.0 |
| `numpy` | 1.26.4 |
| `Pillow` | 10.3.0 |
| `matplotlib` | 3.8.4 |
| `scikit-learn` | 1.4.2 |
| `tqdm` | 4.66.4 |
| `pytorch-fid` | 0.3.0 |
| `scipy` | 1.13.0 |
| `seaborn` | 0.13.2 |

Install all with:
```bash
pip install -r requirements.txt
```

---

## 👩‍💻 Author

**Amritha**
Course: 24AI636 — Deep Learning
Hardware: NVIDIA RTX 3060 Mobile (local training)

---

*Mini Project 3 | SCAFFOLD_R3*
