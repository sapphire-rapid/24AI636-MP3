import os
import torch

# ─────────────────────────────────────────
# 1. ENVIRONMENT
# ─────────────────────────────────────────
ENVIRONMENT = "local"

# ─────────────────────────────────────────
# 2. PATHS (UPDATE THESE!)
# ─────────────────────────────────────────
BASE_DIR  = r"D:\Amritha\Deep Learning - 4\SCAFFOLD_R3\Dataset\R3_DL_DVMCar"
DATA_ROOT = r"D:\Amritha\Deep Learning - 4\SCAFFOLD_R3\Dataset\car_fronts"


# Create directories
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")
LOG_DIR        = os.path.join(BASE_DIR, "logs")

for _dir in [CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ─────────────────────────────────────────
# 3. DEVICE
# ─────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────
# 4. DATA SETTINGS
# ─────────────────────────────────────────
IMAGE_SIZE   = 64
NUM_CHANNELS = 3

# IMPORTANT CHANGE for local:
MAX_IMAGES = None   # use full dataset on RTX 3060

# ─────────────────────────────────────────
# 5. AUTOENCODER
# ─────────────────────────────────────────
AE_LATENT_DIM = 128
AE_BATCH_SIZE = 64
AE_EPOCHS     = 5
AE_LR         = 1e-3
AE_BETA1      = 0.9
AE_BETA2      = 0.999

# ─────────────────────────────────────────
# 6. WGAN-GP
# ─────────────────────────────────────────
GAN_NOISE_DIM    = 128
GAN_BATCH_SIZE   = 64
GAN_EPOCHS       = 5
GAN_LR_G         = 2e-4
GAN_LR_C         = 2e-4
GAN_BETA1        = 0.0
GAN_BETA2        = 0.9
GAN_CRITIC_ITERS = 5
GAN_LAMBDA_GP    = 10

# ─────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────
FID_NUM_SAMPLES  = 1000
TSNE_PERPLEXITY  = 30
TSNE_MAX_SAMPLES = 2000
PCA_MAX_SAMPLES  = 5000

# ─────────────────────────────────────────
# 8. CHECKPOINTS
# ─────────────────────────────────────────
AE_CKPT_NAME        = "autoencoder.pth"
GEN_CKPT_NAME       = "generator.pth"
CRITIC_CKPT_NAME    = "critic.pth"
SAVE_EVERY_N_EPOCHS = 10

# ─────────────────────────────────────────
# 9. SEED
# ─────────────────────────────────────────
SEED = 42

# ─────────────────────────────────────────
# 10. PRINT CONFIG
# ─────────────────────────────────────────
def print_config():
    print("=" * 50)
    print("  Mini Project 3 Configuration (LOCAL)")
    print("=" * 50)
    print(f"Device       : {DEVICE}")
    print(f"Data Root    : {DATA_ROOT}")
    print(f"Image Size   : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"AE Latent    : {AE_LATENT_DIM}d | Epochs: {AE_EPOCHS}")
    print(f"GAN Noise    : {GAN_NOISE_DIM}d | Epochs: {GAN_EPOCHS}")
    print(f"Checkpoints  : {CHECKPOINT_DIR}")
    print(f"Outputs      : {OUTPUT_DIR}")
    print("=" * 50)

print_config()