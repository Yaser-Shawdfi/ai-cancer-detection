import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

TRAIN_MEAN = [0.7008, 0.5384, 0.6916]
TRAIN_STD  = [0.2350, 0.2774, 0.2129]

def get_transforms(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),
    ])

class PCamDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = str(h5_path)
        self.transform = transform
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['y'])
            self.labels = f['y'][:].astype(np.int64).squeeze()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            img = f['x'][idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def get_dataloaders(batch_size=64, num_workers=0):
    train_ds = PCamDataset(DATA_DIR / "train_subset.h5", transform=get_transforms("train"))
    valid_ds = PCamDataset(DATA_DIR / "valid_subset.h5", transform=get_transforms("valid"))
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False)
    print(f"Train: {len(train_ds):,} | Valid: {len(valid_ds):,}")
    pos = train_ds.labels.sum()
    print(f"Class balance (train): {pos/len(train_ds)*100:.1f}% malignant")
    return train_loader, valid_loader
