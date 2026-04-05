# ============================================================
# 24AI636 - Mini Project 3: Autoencoder + WGAN
# dataset.py — Kaggle + Local (RTX 3060) Compatible
#
# Supports TWO dataset structures:
#   Local  : Brand/Year/image.jpg
#   Kaggle : validation/Brand/Brand$$Model$$Year$$Colour$$...jpg
#
# Structure is auto-detected based on config.ENVIRONMENT
# ============================================================

import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import config


# ─────────────────────────────────────────
# 1. REPRODUCIBILITY
# ─────────────────────────────────────────
random.seed(config.SEED)
torch.manual_seed(config.SEED)

if config.DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(config.SEED)


# ─────────────────────────────────────────
# 2. IMAGE TRANSFORMS
# ─────────────────────────────────────────
def get_transforms():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


# ─────────────────────────────────────────
# 3. DATASET CLASS
# ─────────────────────────────────────────
class DVMCarDataset(Dataset):
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

    def __init__(self, root=config.DATA_ROOT,
                 max_images=config.MAX_IMAGES,
                 transform=None):

        self.root      = root
        self.transform = transform if transform else get_transforms()
        self.samples   = []

        # Auto-detect structure based on environment
        if config.ENVIRONMENT in ("kaggle",):
            self._scan_dataset_flat(max_images)
        else:
            self._scan_dataset_nested(max_images)

    # ── LOCAL structure: Brand/Year/image.jpg ──────────────
    def _scan_dataset_nested(self, max_images):
        all_samples = []

        if not os.path.exists(self.root):
            raise FileNotFoundError(
                f"DATA_ROOT not found: {self.root}"
            )

        for brand in sorted(os.listdir(self.root)):
            brand_path = os.path.join(self.root, brand)
            if not os.path.isdir(brand_path):
                continue

            for year in sorted(os.listdir(brand_path)):
                year_path = os.path.join(brand_path, year)
                if not os.path.isdir(year_path):
                    continue

                for fname in os.listdir(year_path):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in self.VALID_EXTENSIONS:
                        full_path = os.path.join(year_path, fname)
                        all_samples.append((full_path, brand, year))

        random.shuffle(all_samples)
        if max_images is not None:
            all_samples = all_samples[:max_images]
        self.samples = all_samples
        print(f"[Dataset] Nested structure | "
              f"Loaded {len(self.samples)} images from {self.root}")

    # ── KAGGLE structure: Brand/Brand$$Model$$Year$$Colour$$...jpg ──
    def _scan_dataset_flat(self, max_images):
        """
        Kaggle DVM-Car dataset has all images flat inside Brand folder.
        Filename format: Brand$$Model$$Year$$Colour$$ID$$AdvID$$image_N.jpg
        Year is extracted from filename for latent space labelling.
        """
        all_samples = []

        if not os.path.exists(self.root):
            raise FileNotFoundError(
                f"DATA_ROOT not found: {self.root}\n"
                f"Make sure the dataset is attached to your Kaggle notebook."
            )

        for brand in sorted(os.listdir(self.root)):
            brand_path = os.path.join(self.root, brand)
            if not os.path.isdir(brand_path):
                continue

            for fname in os.listdir(brand_path):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self.VALID_EXTENSIONS:
                    continue

                full_path = os.path.join(brand_path, fname)

                # Extract year from filename: Brand$$Model$$Year$$...
                try:
                    year = fname.split('$$')[2]
                except IndexError:
                    year = 'unknown'

                all_samples.append((full_path, brand, year))

        random.shuffle(all_samples)
        if max_images is not None:
            all_samples = all_samples[:max_images]
        self.samples = all_samples
        print(f"[Dataset] Flat structure | "
              f"Loaded {len(self.samples)} images from {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, brand, year = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[Warning] Corrupt image: {img_path}")
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE))

        if self.transform:
            image = self.transform(image)

        label = {'brand': brand, 'year': year}
        return image, label


# ─────────────────────────────────────────
# 4. DATALOADER
# ─────────────────────────────────────────
def get_dataloader(batch_size=config.AE_BATCH_SIZE,
                   max_images=config.MAX_IMAGES,
                   shuffle=True):

    dataset = DVMCarDataset(max_images=max_images)

    # Kaggle needs num_workers=2, local can handle 4
    num_workers = 2 if config.ENVIRONMENT == "kaggle" else 4

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    print(f"[DataLoader] Batches: {len(loader)} | "
          f"Batch size: {batch_size} | "
          f"Total images: {len(dataset)}")

    return loader, dataset


# ─────────────────────────────────────────
# 5. SANITY CHECK (RUN: python dataset.py)
# ─────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    print("Running dataset sanity check...")

    loader, dataset = get_dataloader(batch_size=16, max_images=200)
    images, labels  = next(iter(loader))

    print(f"Batch shape  : {images.shape}")
    print(f"Pixel range  : [{images.min():.2f}, {images.max():.2f}]")
    print(f"Sample brands: {set(labels['brand'])}")
    print(f"Sample years : {set(labels['year'])}")

    grid = vutils.make_grid(images, nrow=4, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Dataset Sanity Check — DVM-Car")
    plt.savefig("dataset_check.png", bbox_inches='tight')
    plt.show()

    print("Done.")
