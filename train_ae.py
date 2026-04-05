# ============================================================
# 24AI636 - Mini Project 3: Autoencoder + WGAN
# train_ae.py — Autoencoder Training
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import config
from dataset import get_dataloader
from models import Autoencoder


# ============================================================
# 1. TRAIN FUNCTION
# ============================================================
def train_autoencoder():

    device = config.DEVICE

    # Load data
    loader, _ = get_dataloader(batch_size=config.AE_BATCH_SIZE)

    # Model
    model = Autoencoder().to(device)

    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.AE_LR,
        betas=(config.AE_BETA1, config.AE_BETA2)
    )

    losses = []

    print("\nStarting Autoencoder Training...\n")

    for epoch in range(config.AE_EPOCHS):
        model.train()
        epoch_loss = 0

        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{config.AE_EPOCHS}]")

        for images, _ in loop:
            images = images.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, images)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)

        print(f"Epoch [{epoch+1}] Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0:
            ckpt_path = os.path.join(
                config.CHECKPOINT_DIR,
                f"ae_epoch_{epoch+1}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    print("\nTraining Complete.")

    # Save final model
    final_path = os.path.join(config.CHECKPOINT_DIR, config.AE_CKPT_NAME)
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")

    return model, losses


# ============================================================
# 2. PLOT LOSS
# ============================================================
def plot_losses(losses):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="AE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training Loss")
    plt.legend()
    plt.grid()

    save_path = os.path.join(config.OUTPUT_DIR, "ae_loss.png")
    plt.savefig(save_path)
    plt.show()

    print(f"Loss plot saved at: {save_path}")


# ============================================================
# 3. RECONSTRUCTION VISUALIZATION
# ============================================================
def show_reconstructions(model, num_images=8):
    import torchvision.utils as vutils

    device = config.DEVICE
    loader, _ = get_dataloader(batch_size=num_images, max_images=100)

    model.eval()

    images, _ = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        recon = model(images)

    images = images.cpu()
    recon = recon.cpu()

    # Create comparison grid
    combined = torch.cat([images, recon], dim=0)

    grid = vutils.make_grid(combined, nrow=num_images, normalize=True)

    plt.figure(figsize=(12, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Top: Original | Bottom: Reconstruction")

    save_path = os.path.join(config.OUTPUT_DIR, "ae_reconstruction.png")
    plt.savefig(save_path)
    plt.show()

    print(f"Reconstruction saved at: {save_path}")


# ============================================================
# 4. MAIN (FOR DIRECT RUN)
# ============================================================
if __name__ == "__main__":

    model, losses = train_autoencoder()

    plot_losses(losses)

    show_reconstructions(model)