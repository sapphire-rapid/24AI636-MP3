# ============================================================
# 24AI636 - Mini Project 3: Autoencoder + WGAN
# evaluate.py — FID + PCA + t-SNE
# ============================================================

import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import config
from dataset import get_dataloader
from models import Autoencoder, Generator

from pytorch_fid import fid_score


# ============================================================
# 1. LOAD TRAINED MODELS
# ============================================================
def load_models():
    device = config.DEVICE

    # Load full Autoencoder and extract the encoder
    ae = Autoencoder().to(device)
    ae.load_state_dict(torch.load(
        os.path.join(config.CHECKPOINT_DIR, config.AE_CKPT_NAME),
        map_location=device
    ))
    encoder = ae.encoder

    generator = Generator().to(device)
    generator.load_state_dict(torch.load(
        os.path.join(config.CHECKPOINT_DIR, config.GEN_CKPT_NAME),
        map_location=device
    ))

    encoder.eval()
    generator.eval()

    print("Models loaded successfully.")

    return encoder, generator


# ============================================================
# 2. LATENT SPACE EXTRACTION
# ============================================================
def extract_latents(encoder, max_samples=2000):
    device = config.DEVICE

    loader, _ = get_dataloader(
        batch_size=64,
        max_images=max_samples
    )

    latents = []
    labels = []

    with torch.no_grad():
        for images, meta in loader:
            images = images.to(device)

            z = encoder(images)
            latents.append(z.cpu().numpy())
            labels.extend(meta['brand'])

    latents = np.concatenate(latents, axis=0)

    return latents, labels


# ============================================================
# 3. PCA VISUALIZATION
# ============================================================
def plot_pca(latents, labels):
    print("Running PCA...")

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(latents)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=5)

    plt.title("PCA of Latent Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    save_path = os.path.join(config.OUTPUT_DIR, "pca.png")
    plt.savefig(save_path)
    plt.show()

    print(f"PCA saved at: {save_path}")


# ============================================================
# 4. t-SNE VISUALIZATION
# ============================================================
def plot_tsne(latents, labels):
    print("Running t-SNE (this may take time)...")

    tsne = TSNE(
        n_components=2,
        perplexity=config.TSNE_PERPLEXITY,
        random_state=config.SEED
    )

    reduced = tsne.fit_transform(latents[:config.TSNE_MAX_SAMPLES])

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=5)

    plt.title("t-SNE of Latent Space")

    save_path = os.path.join(config.OUTPUT_DIR, "tsne.png")
    plt.savefig(save_path)
    plt.show()

    print(f"t-SNE saved at: {save_path}")


# ============================================================
# 5. GENERATE IMAGES FOR FID
# ============================================================
def generate_fid_images(generator, save_dir, num_images=1000):
    device = config.DEVICE

    os.makedirs(save_dir, exist_ok=True)

    generator.eval()

    for i in range(num_images):
        noise = torch.randn(1, config.GAN_NOISE_DIM).to(device)

        with torch.no_grad():
            fake = generator(noise).cpu()

        img = (fake.squeeze(0) + 1) / 2  # [-1,1] → [0,1]

        save_path = os.path.join(save_dir, f"{i}.png")
        plt.imsave(save_path, img.permute(1, 2, 0).numpy())


# ============================================================
# 6. COMPUTE FID SCORE
# ============================================================
def compute_fid(real_dir, fake_dir):
    print("Computing FID score...")

    fid = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=32,
        device=config.DEVICE,
        dims=2048
    )

    print(f"FID Score: {fid:.2f}")
    return fid


# ============================================================
# 7. MAIN
# ============================================================
if __name__ == "__main__":

    encoder, generator = load_models()

    # Latent space analysis
    latents, labels = extract_latents(encoder)
    plot_pca(latents, labels)
    plot_tsne(latents, labels)

    # FID evaluation
    real_dir = os.path.join(config.OUTPUT_DIR, "real_images")
    fake_dir = os.path.join(config.OUTPUT_DIR, "fake_images")

    print("Preparing images for FID...")

    # Save real images
    loader, _ = get_dataloader(batch_size=1, max_images=config.FID_NUM_SAMPLES)
    os.makedirs(real_dir, exist_ok=True)

    for i, (img, _) in enumerate(loader):
        img = (img.squeeze(0) + 1) / 2
        plt.imsave(os.path.join(real_dir, f"{i}.png"),
                   img.permute(1, 2, 0).numpy())

    # Generate fake images
    generate_fid_images(generator, fake_dir, config.FID_NUM_SAMPLES)

    # Compute FID
    compute_fid(real_dir, fake_dir)