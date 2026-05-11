import urllib.request
import os
import h5py
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

PCAM_FILES = {
    "train_x": ("https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz",
                 "camelyonpatch_level_2_split_train_x.h5.gz"),
    "train_y": ("https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz",
                 "camelyonpatch_level_2_split_train_y.h5.gz"),
    "valid_x": ("https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz",
                 "camelyonpatch_level_2_split_valid_x.h5.gz"),
    "valid_y": ("https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz",
                 "camelyonpatch_level_2_split_valid_y.h5.gz"),
}

def reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
    mb = downloaded / 1e6
    print(f"\r  {pct:.1f}% ({mb:.1f} MB)", end="", flush=True)

def download_and_decompress(url, filename):
    gz_path = DATA_DIR / filename
    h5_path = DATA_DIR / filename.replace(".gz", "")
    if h5_path.exists():
        print(f"  Already exists: {h5_path.name}")
        return h5_path
    print(f"\nDownloading {filename}...")
    urllib.request.urlretrieve(url, gz_path, reporthook)
    print(f"\n  Decompressing...")
    import gzip, shutil
    with gzip.open(gz_path, 'rb') as f_in:
        with open(h5_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
    print(f"  Done: {h5_path.name}")
    return h5_path

def extract_subset(x_file, y_file, out_name, n_samples=50000, seed=42):
    out_path = DATA_DIR / out_name
    if out_path.exists():
        print(f"  Subset already exists: {out_name}")
        return
    print(f"\nExtracting {n_samples:,} samples from {x_file.name}...")
    with h5py.File(x_file, 'r') as fx, h5py.File(y_file, 'r') as fy:
        total = fx['x'].shape[0]
        rng = np.random.default_rng(seed)
        idx = rng.choice(total, min(n_samples, total), replace=False)
        idx.sort()
        X = fx['x'][idx]
        Y = fy['y'][idx].squeeze()
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('x', data=X, compression='gzip')
        f.create_dataset('y', data=Y)
    pos = Y.sum()
    print(f"  Saved {len(Y):,} samples -> {int(pos):,} malignant ({pos/len(Y)*100:.1f}%), {len(Y)-int(pos):,} normal")

if __name__ == "__main__":
    print("=" * 60)
    print("PatchCamelyon (PCam) Data Downloader")
    print("Real H&E-stained lymph node histopathology images")
    print("Source: Veeling et al. (2018) — MICCAI / Zenodo #2546921")
    print("=" * 60)

    tx = download_and_decompress(*PCAM_FILES["train_x"])
    ty = download_and_decompress(*PCAM_FILES["train_y"])
    vx = download_and_decompress(*PCAM_FILES["valid_x"])
    vy = download_and_decompress(*PCAM_FILES["valid_y"])

    extract_subset(tx, ty, "train_subset.h5", n_samples=40000)
    extract_subset(vx, vy, "valid_subset.h5",  n_samples=10000)

    print("\n✓ Dataset ready in data/")
