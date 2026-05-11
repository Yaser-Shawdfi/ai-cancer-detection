import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import json, time
from dataset import get_dataloaders
from model import CancerDetector

MODELS_DIR = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

EPOCHS     = 10
BATCH_SIZE = 64
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(imgs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            outputs = model(imgs).squeeze(1)
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)
            total_loss += loss.item() * imgs.size(0)
            preds = (probs >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += imgs.size(0)
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().long().tolist())
    return total_loss / total, correct / total, all_probs, all_labels

def compute_auc(labels, probs):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, probs)

def main():
    print("=" * 60)
    print("AI Cancer Detection — Training Pipeline")
    print(f"Model : EfficientNetB0 (ImageNet pretrained → fine-tuned)")
    print(f"Data  : PatchCamelyon (PCam) — Real H&E lymph node patches")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    train_loader, valid_loader = get_dataloaders(batch_size=BATCH_SIZE)

    model = CancerDetector(pretrained=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}
    best_auc = 0.0
    start = time.time()

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(RESULTS_DIR / "logs"))

    for epoch in range(1, EPOCHS + 1):
        ep_start = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        va_loss, va_acc, probs, labels = eval_epoch(model, valid_loader, criterion, DEVICE)
        va_auc = compute_auc(labels, probs)
        
        # Log to TensorBoard
        writer.add_scalar("Loss/Train", tr_loss, epoch)
        writer.add_scalar("Loss/Validation", va_loss, epoch)
        writer.add_scalar("Accuracy/Train", tr_acc, epoch)
        writer.add_scalar("Accuracy/Validation", va_acc, epoch)
        writer.add_scalar("AUC/Validation", va_auc, epoch)
        writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)
        
        scheduler.step()

        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["val_loss"].append(round(va_loss, 4))
        history["val_acc"].append(round(va_acc, 4))
        history["val_auc"].append(round(va_auc, 4))

        ep_time = time.time() - ep_start
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Loss {tr_loss:.4f}/{va_loss:.4f} | "
              f"Acc {tr_acc*100:.2f}%/{va_acc*100:.2f}% | "
              f"AUC {va_auc:.4f} | {ep_time:.0f}s")

        if va_auc > best_auc:
            print(f"  ✓ Validation AUC improved from {best_auc:.4f} to {va_auc:.4f} — saving model weights")
            best_auc = va_auc
            torch.save(model.state_dict(), MODELS_DIR / "best_model.pth")
            
    writer.close()
    total_time = (time.time() - start) / 60
    print(f"\nTraining complete in {total_time:.1f} min | Best AUC: {best_auc:.4f}")

    summary = {
        "best_val_auc": round(best_auc, 4),
        "best_val_acc": round(max(history["val_acc"]), 4),
        "epochs": EPOCHS,
        "model": "EfficientNetB0 (ImageNet pretrained)",
        "dataset": "PatchCamelyon (PCam) — Veeling et al. 2018",
        "train_samples": 40000,
        "valid_samples": 10000,
        "history": history,
        "probs_sample": probs[:200],
        "labels_sample": labels[:200],
    }
    with open(RESULTS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to results/training_summary.json")

if __name__ == "__main__":
    main()
