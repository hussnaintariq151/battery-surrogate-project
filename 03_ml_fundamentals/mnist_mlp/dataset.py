"""
dataset.py
----------
Yahan hum MNIST data ko load karte hain aur PyTorch ke
Dataset + DataLoader ka use karte hain.

MNIST kya hai?
  - 70,000 handwritten digit images (0-9)
  - Har image: 28x28 pixels, grayscale (black & white)
  - 60,000 training + 10,000 testing images
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# ─────────────────────────────────────────────
# STEP 1: Transform define karo
# ─────────────────────────────────────────────
# Transform ka matlab: raw image ko tensor mein badlo
# aur normalize karo
#
# Normalize kyun?
#   - Raw pixel values: 0 to 255
#   - Neural network ke liye behtar: -1 to 1 ya 0 to 1
#   - mean=0.1307, std=0.3081 → yeh MNIST ke actual values hain
#     (puri dataset ka average aur spread)
#
# transforms.Compose → multiple transforms ek saath lagao
# transforms.ToTensor() → image ko [0,1] range mein tensor banao
# transforms.Normalize() → (value - mean) / std

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])


# ─────────────────────────────────────────────
# STEP 2: Custom Dataset class banao
# ─────────────────────────────────────────────
# PyTorch ka Dataset class inherit karte hain
# Hume sirf 3 cheezein likhni hain:
#   1. __init__   → data load karo
#   2. __len__    → kitne samples hain?
#   3. __getitem__ → ek sample do (index se)

class MNISTDataset(Dataset):
    def __init__(self, train=True, transform=None):
        """
        train=True  → 60,000 training images load karo
        train=False → 10,000 test images load karo
        """
        # torchvision se MNIST download karo (pehli baar sirf)
        # root='./data' → yahan save hoga
        self.data = datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )
        self.transform = transform

    def __len__(self):
        # DataLoader ko batao → total kitne samples hain
        return len(self.data)

    def __getitem__(self, index):
        # Ek image aur uska label do
        # image shape: [1, 28, 28] → 1 channel, 28x28 pixels
        # label: 0 to 9 (kaunsa digit hai)
        image, label = self.data[index]
        return image, label


# ─────────────────────────────────────────────
# STEP 3: DataLoader banao
# ─────────────────────────────────────────────
# DataLoader kya karta hai?
#   - Dataset se batch by batch data deta hai
#   - shuffle=True → training mein data randomly mix karo
#     (taake model ek order yaad na kar le)
#   - batch_size=32 → ek baar mein 32 images process karo

def get_dataloaders(batch_size=32):
    """
    Training aur test DataLoader return karta hai.

    Args:
        batch_size: ek baar mein kitni images (default: 32)

    Returns:
        train_loader, test_loader
    """
    # Dataset objects banao
    train_dataset = MNISTDataset(train=True,  transform=transform)
    test_dataset  = MNISTDataset(train=False, transform=transform)

    # DataLoader wrap karo
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True       # training mein shuffle zaroori hai
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False      # test mein shuffle ki zaroorat nahi
    )

    return train_loader, test_loader


# ─────────────────────────────────────────────
# STEP 4: Quick sanity check
# ─────────────────────────────────────────────
# Yeh tab chalega jab tum seedha
# python dataset.py karoge

if __name__ == '__main__':
    print("=== Dataset Check ===\n")

    train_loader, test_loader = get_dataloaders(batch_size=32)

    # Ek batch lo aur uski details dekho
    images, labels = next(iter(train_loader))

    print(f"Training batches  : {len(train_loader)}")
    print(f"Test batches      : {len(test_loader)}")
    print(f"Batch image shape : {images.shape}")
    # [32, 1, 28, 28] → 32 images, 1 channel, 28x28
    print(f"Batch label shape : {labels.shape}")
    # [32] → 32 labels (0-9)
    print(f"Label examples    : {labels[:8].tolist()}")
    print(f"Pixel min/max     : {images.min():.3f} / {images.max():.3f}")
    # Normalize ke baad roughly -0.4 to 2.8 hoga
    print("\nDataset ready hai!")