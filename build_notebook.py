import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: Markdown Setup
text_1 = """# AI Cancer Detection — Google Colab Training
**Inspired by PathAI & Paige.AI**

This notebook will run the full AI model training pipeline on Google Colab's Free T4 GPU. 

### 🚀 Before you start:
1. Go to **Runtime > Change runtime type** at the top menu.
2. Under "Hardware accelerator", select **T4 GPU** and click Save.
3. Click **Runtime > Run all**.

The entire process (downloading data, training for 10 epochs, evaluating, and exporting) will take about **5 to 10 minutes**!
"""

# Cell 2: Setup
code_2 = """!pip install -q torch torchvision h5py tqdm tensorboard scikit-learn matplotlib seaborn"""

# Cell 3: Data Download
code_3 = """import urllib.request
import os, h5py, gzip, shutil
import numpy as np
from pathlib import Path

# Setup directories
for d in ['data', 'models', 'results', 'results/logs']:
    Path(d).mkdir(parents=True, exist_ok=True)

PCAM_FILES = {
    "train_x": ("https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz", "camelyonpatch_level_2_split_train_x.h5.gz"),
    "train_y": ("https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz", "camelyonpatch_level_2_split_train_y.h5.gz"),
    "valid_x": ("https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz", "camelyonpatch_level_2_split_valid_x.h5.gz"),
    "valid_y": ("https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz", "camelyonpatch_level_2_split_valid_y.h5.gz"),
}

def download_pcam(url, filename):
    gz_path = Path('data') / filename
    h5_path = Path('data') / filename.replace(".gz", "")
    if h5_path.exists(): return h5_path
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, gz_path)
    with gzip.open(gz_path, 'rb') as f_in, open(h5_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
    return h5_path

def extract_subset(x_file, y_file, out_name, n_samples):
    out_path = Path('data') / out_name
    if out_path.exists(): return
    print(f"Extracting {n_samples} samples into {out_name}...")
    with h5py.File(x_file, 'r') as fx, h5py.File(y_file, 'r') as fy:
        total = fx['x'].shape[0]
        rng = np.random.default_rng(42)
        idx = rng.choice(total, min(n_samples, total), replace=False)
        idx.sort()
        X, Y = fx['x'][idx], fy['y'][idx].squeeze()
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('x', data=X, compression='gzip')
        f.create_dataset('y', data=Y)

print("Downloading PatchCamelyon Dataset...")
tx = download_pcam(*PCAM_FILES['train_x'])
ty = download_pcam(*PCAM_FILES['train_y'])
vx = download_pcam(*PCAM_FILES['valid_x'])
vy = download_pcam(*PCAM_FILES['valid_y'])

print("Full dataset downloaded and ready!")"""

# Cell 4: Model & Dataset
code_4 = """import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

class CancerDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, 1))
    def forward(self, x): return self.backbone(x)

class PCamDataset(Dataset):
    def __init__(self, x_path, y_path, transform=None):
        self.x_path = str(x_path)
        self.y_path = str(y_path)
        self.transform = transform
        with h5py.File(self.y_path, 'r') as f:
            self.len = len(f['y'])
            self.labels = f['y'][:].astype(np.int64).squeeze()
    def __len__(self): return self.len
    def __getitem__(self, idx):
        with h5py.File(self.x_path, 'r') as fx: img = fx['x'][idx]
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

norm = transforms.Normalize([0.7008, 0.5384, 0.6916], [0.2350, 0.2774, 0.2129])
train_tf = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor(), norm])
val_tf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), norm])

train_loader = DataLoader(PCamDataset("data/camelyonpatch_level_2_split_train_x.h5", "data/camelyonpatch_level_2_split_train_y.h5", train_tf), batch_size=64, shuffle=True)
valid_loader = DataLoader(PCamDataset("data/camelyonpatch_level_2_split_valid_x.h5", "data/camelyonpatch_level_2_split_valid_y.h5", val_tf), batch_size=64, shuffle=False)
"""

# Cell 5: Training
code_5 = """import time, json
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")

model = CancerDetector().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10)
writer = SummaryWriter(log_dir="results/logs")

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}
best_auc = 0.0

for epoch in range(1, 11):
    model.train()
    tr_loss, tr_correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.float().to(DEVICE)
        optimizer.zero_grad()
        outs = model(imgs).squeeze(1)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * imgs.size(0)
        tr_correct += ((torch.sigmoid(outs) >= 0.5).long() == labels.long()).sum().item()
        total += imgs.size(0)
    
    tr_loss /= total; tr_acc = tr_correct / total

    model.eval()
    va_loss, va_correct, v_total = 0, 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(DEVICE), labels.float().to(DEVICE)
            outs = model(imgs).squeeze(1)
            loss = criterion(outs, labels)
            probs = torch.sigmoid(outs)
            va_loss += loss.item() * imgs.size(0)
            va_correct += ((probs >= 0.5).long() == labels.long()).sum().item()
            v_total += imgs.size(0)
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().long().tolist())
            
    va_loss /= v_total; va_acc = va_correct / v_total
    va_auc = roc_auc_score(all_labels, all_probs)
    
    writer.add_scalar("AUC/Validation", va_auc, epoch)
    scheduler.step()
    
    print(f"Epoch {epoch:02d} | Train Acc: {tr_acc*100:.2f}% | Val Acc: {va_acc*100:.2f}% | Val AUC: {va_auc:.4f}")
    
    history["train_acc"].append(round(tr_acc, 4)); history["val_acc"].append(round(va_acc, 4))
    history["val_auc"].append(round(va_auc, 4))
    
    if va_auc > best_auc:
        best_auc = va_auc
        torch.save(model.state_dict(), "models/best_model.pth")

summary = {"final_auc": round(va_auc, 4), "final_acc": round(va_acc, 4), "history": history}
with open("results/training_summary.json", "w") as f: json.dump(summary, f)
print("Training Complete! Saved models/best_model.pth")"""

# Cell 6: Export Results
code_6 = """!zip -r model_and_results.zip results/ models/
from google.colab import files
files.download('model_and_results.zip')
print("✅ Download started! Place the contents of this zip into your local 'ai-cancer-detection' folder.")"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_code_cell(code_6)
]

with open("notebooks/AI_Cancer_Detection_Colab.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Colab Notebook created successfully!")
