# ============================================================
# 24AI636 - Mini Project 3: Autoencoder + WGAN
# train_gan.py — WGAN-GP Training
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

import config
from dataset import get_dataloader
from models import Generator, Critic, weights_init


# ============================================================
# 1. GRADIENT PENALTY (CORE OF WGAN-GP)
# ============================================================
def gradient_penalty(critic, real, fake, device):
    batch_size, C, H, W = real.shape

    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = epsilon * real + (1 - epsilon) * fake

    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated)

    gradients = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp


# ============================================================
# 2. TRAIN FUNCTION
# ============================================================
def train_gan():

    device = config.DEVICE

    # Data
    loader, _ = get_dataloader(batch_size=config.GAN_BATCH_SIZE)

    # Models
    gen = Generator().to(device)
    critic = Critic().to(device)

    gen.apply(weights_init)
    critic.apply(weights_init)

    # Optimizers
    opt_gen = optim.Adam(
        gen.parameters(),
        lr=config.GAN_LR_G,
        betas=(config.GAN_BETA1, config.GAN_BETA2)
    )

    opt_critic = optim.Adam(
        critic.parameters(),
        lr=config.GAN_LR_C,
        betas=(config.GAN_BETA1, config.GAN_BETA2)
    )

    gen_losses = []
    critic_losses = []

    print("\nStarting WGAN-GP Training...\n")

    for epoch in range(config.GAN_EPOCHS):

        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{config.GAN_EPOCHS}]")

        for batch_idx, (real, _) in enumerate(loop):
            real = real.to(device)
            cur_batch_size = real.shape[0]

            # =========================
            # TRAIN CRITIC
            # =========================
            for _ in range(config.GAN_CRITIC_ITERS):
                noise = torch.randn(cur_batch_size, config.GAN_NOISE_DIM).to(device)
                fake = gen(noise)

                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake.detach()).reshape(-1)

                gp = gradient_penalty(critic, real, fake.detach(), device)

                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + config.GAN_LAMBDA_GP * gp
                )

                opt_critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

            # =========================
            # TRAIN GENERATOR
            # =========================
            noise = torch.randn(cur_batch_size, config.GAN_NOISE_DIM).to(device)
            fake = gen(noise)

            loss_gen = -torch.mean(critic(fake))

            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            gen_losses.append(loss_gen.item())
            critic_losses.append(loss_critic.item())

            loop.set_postfix(
                gen_loss=loss_gen.item(),
                critic_loss=loss_critic.item()
            )

        print(f"Epoch [{epoch+1}] G Loss: {loss_gen:.4f}, C Loss: {loss_critic:.4f}")

        # Save checkpoints
        if (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0:
            torch.save(gen.state_dict(),
                       os.path.join(config.CHECKPOINT_DIR, f"gen_epoch_{epoch+1}.pth"))
            torch.save(critic.state_dict(),
                       os.path.join(config.CHECKPOINT_DIR, f"critic_epoch_{epoch+1}.pth"))

    print("\nGAN Training Complete.")

    # Save final models
    torch.save(gen.state_dict(),
               os.path.join(config.CHECKPOINT_DIR, config.GEN_CKPT_NAME))
    torch.save(critic.state_dict(),
               os.path.join(config.CHECKPOINT_DIR, config.CRITIC_CKPT_NAME))

    return gen, critic, gen_losses, critic_losses


# ============================================================
# 3. PLOT LOSSES
# ============================================================
def plot_gan_losses(gen_losses, critic_losses):

    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label="Generator Loss")
    plt.plot(critic_losses, label="Critic Loss")

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("WGAN-GP Training Loss")
    plt.legend()
    plt.grid()

    save_path = os.path.join(config.OUTPUT_DIR, "gan_loss.png")
    plt.savefig(save_path)
    plt.show()

    print(f"Loss plot saved at: {save_path}")


# ============================================================
# 4. GENERATE SAMPLE IMAGES
# ============================================================
def generate_samples(gen, num_images=16):
    import torchvision.utils as vutils

    device = config.DEVICE

    gen.eval()

    noise = torch.randn(num_images, config.GAN_NOISE_DIM).to(device)

    with torch.no_grad():
        fake = gen(noise).cpu()

    grid = vutils.make_grid(fake, nrow=4, normalize=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Generated Images")

    save_path = os.path.join(config.OUTPUT_DIR, "gan_samples.png")
    plt.savefig(save_path)
    plt.show()

    print(f"Generated images saved at: {save_path}")


# ============================================================
# 5. MAIN
# ============================================================
if __name__ == "__main__":

    gen, critic, gen_losses, critic_losses = train_gan()

    plot_gan_losses(gen_losses, critic_losses)

    generate_samples(gen)