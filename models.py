# ============================================================
# 24AI636 - Mini Project 3: Autoencoder + WGAN
# models.py — AE + WGAN (DCGAN-based)
# ============================================================

import torch
import torch.nn as nn
import config


# ============================================================
# 1. AUTOENCODER
# ============================================================

# -------------------------
# Encoder
# -------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=config.AE_LATENT_DIM):
        super().__init__()

        self.model = nn.Sequential(
            # [3, 64, 64] → [64, 32, 32]
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # → [128, 16, 16]
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # → [256, 8, 8]
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # → [512, 4, 4]
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z


# -------------------------
# Decoder
# -------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim=config.AE_LATENT_DIM):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        self.model = nn.Sequential(
            # [512, 4, 4]
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # → [128, 16, 16]
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # → [64, 32, 32]
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # → [3, 64, 64]
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()   # output in [-1,1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)
        x = self.model(x)
        return x


# -------------------------
# Full Autoencoder
# -------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


# ============================================================
# 2. WGAN (DCGAN STYLE)
# ============================================================

# -------------------------
# Generator
# -------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim=config.GAN_NOISE_DIM):
        super().__init__()

        self.model = nn.Sequential(
            # [Z,1,1] → [512,4,4]
            nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # → [256,8,8]
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # → [128,16,16]
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # → [64,32,32]
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # → [3,64,64]
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        img = self.model(z)
        return img


# -------------------------
# Critic (NO SIGMOID)
# -------------------------
class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # [3,64,64] → [64,32,32]
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # → [128,16,16]
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # → [256,8,8]
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # → [512,4,4]
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # → [1,1,1]
            nn.Conv2d(512, 1, 4, 1, 0)
        )

    def forward(self, x):
        out = self.model(x)
        return out.view(-1)


# ============================================================
# 3. WEIGHT INITIALIZATION (DCGAN STANDARD)
# ============================================================
def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ============================================================
# 4. SANITY CHECK
# ============================================================
if __name__ == "__main__":
    print("Running models sanity check...")

    device = config.DEVICE

    # Autoencoder
    ae = Autoencoder().to(device)
    x = torch.randn(4, 3, 64, 64).to(device)
    out = ae(x)
    print("AE Output Shape:", out.shape)

    # Generator
    gen = Generator().to(device)
    z = torch.randn(4, config.GAN_NOISE_DIM).to(device)
    fake = gen(z)
    print("Generator Output:", fake.shape)

    # Critic
    critic = Critic().to(device)
    score = critic(fake)
    print("Critic Output:", score.shape)

    print("Models check complete.")