# ============================================================
# 24AI636 - Mini Project 3: Autoencoder + WGAN
# dataset.py — Local (RTX 3060 Optimized)
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

        self.root = root
        self.transform = transform if transform else get_transforms()
        self.samples = []

        self._scan_dataset(max_images)

    def _scan_dataset(self, max_images):
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

        # Shuffle before limiting (important for diversity)
        random.shuffle(all_samples)

        if max_images is not None:
            all_samples = all_samples[:max_images]

        self.samples = all_samples

        print(f"[Dataset] Loaded {len(self.samples)} images from {self.root}")

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
# 4. DATALOADER (LOCAL OPTIMIZED)
# ─────────────────────────────────────────
def get_dataloader(batch_size=config.AE_BATCH_SIZE,
                   max_images=config.MAX_IMAGES,
                   shuffle=True):

    dataset = DVMCarDataset(max_images=max_images)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,

        # 🔥 Local GPU optimizations
        num_workers=4,               # change to 0 if Windows error
        pin_memory=True,
        persistent_workers=True,

        drop_last=True
    )

    print(f"[DataLoader] Batches: {len(loader)} | Batch size: {batch_size}")

    return loader, dataset


# ─────────────────────────────────────────
# 5. SANITY CHECK (RUN: python dataset.py)
# ─────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    print("Running dataset sanity check...")

    loader, dataset = get_dataloader(batch_size=16, max_images=200)

    images, labels = next(iter(loader))

    print(f"Batch shape : {images.shape}")
    print(f"Pixel range : [{images.min():.2f}, {images.max():.2f}]")
    print(f"Brands sample: {set(labels['brand'])}")

    grid = vutils.make_grid(images, nrow=4, normalize=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Dataset Check")
    plt.savefig("dataset_check.png", bbox_inches='tight')
    plt.show()

    print("Dataset sanity check complete.")