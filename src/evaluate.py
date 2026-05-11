import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                              classification_report, accuracy_score)
from dataset import get_dataloaders
from model import CancerDetector

MODELS_DIR  = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_gradcam(model, img_tensor, device):
    model.eval()
    features, grads = [], []
    def fwd_hook(m, i, o): features.append(o)
    def bwd_hook(m, i, o): grads.append(o[0])

    target_layer = model.backbone.features[-1]
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    img = img_tensor.unsqueeze(0).to(device)
    img.requires_grad_(True)
    out = model(img)
    model.zero_grad()
    out[0, 0].backward()

    h1.remove(); h2.remove()
    f = features[0].squeeze().detach().cpu().numpy()
    g = grads[0].squeeze().detach().cpu().numpy()
    weights = g.mean(axis=(1, 2))
    cam = np.zeros(f.shape[1:])
    for i, w in enumerate(weights):
        cam += w * f[i]
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def run_full_evaluation():
    print("=" * 60)
    print("AI Cancer Detection — Full Evaluation")
    print("=" * 60)

    _, valid_loader = get_dataloaders(batch_size=64)

    model = CancerDetector(pretrained=False).to(DEVICE)
    checkpoint = MODELS_DIR / "best_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError("Run train.py first to generate best_model.pth")
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from {checkpoint}")

    all_probs, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().tolist()
            preds = [1 if p >= 0.5 else 0 for p in probs]
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds)

    auc  = roc_auc_score(all_labels, all_probs)
    acc  = accuracy_score(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    cm   = confusion_matrix(all_labels, all_preds)
    rep  = classification_report(all_labels, all_preds,
                                  target_names=["Normal", "Malignant"])
    print(f"\nAccuracy : {acc*100:.2f}%")
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"\n{rep}")

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0d1117')

    axes[0].plot(fpr, tpr, color='#38bdf8', lw=2, label=f'AUC = {auc:.3f}')
    axes[0].plot([0,1],[0,1], color='gray', linestyle='--', lw=1)
    axes[0].fill_between(fpr, tpr, alpha=0.15, color='#38bdf8')
    axes[0].set_xlim([0,1]); axes[0].set_ylim([0,1.02])
    axes[0].set_xlabel('False Positive Rate', color='white')
    axes[0].set_ylabel('True Positive Rate', color='white')
    axes[0].set_title('ROC Curve — EfficientNetB0 on PCam', color='white', fontsize=13)
    axes[0].legend(loc='lower right', facecolor='#1e293b')
    axes[0].set_facecolor('#0d1117')
    axes[0].tick_params(colors='white')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Normal','Malignant'],
                yticklabels=['Normal','Malignant'],
                annot_kws={'size': 14})
    axes[1].set_xlabel('Predicted', color='white')
    axes[1].set_ylabel('Actual', color='white')
    axes[1].set_title('Confusion Matrix', color='white', fontsize=13)
    axes[1].set_facecolor('#0d1117')
    axes[1].tick_params(colors='white')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "evaluation_plots.png", dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    print(f"\nSaved evaluation_plots.png")

    with open(RESULTS_DIR / "training_summary.json") as f:
        summary = json.load(f)

    summary["final_auc"] = round(auc, 4)
    summary["final_acc"] = round(acc, 4)
    summary["confusion_matrix"] = cm.tolist()
    summary["roc_fpr"] = [round(v, 4) for v in fpr[::10].tolist()]
    summary["roc_tpr"] = [round(v, 4) for v in tpr[::10].tolist()]

    with open(RESULTS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Updated training_summary.json with evaluation results")
    print(f"\n✓ Final AUC: {auc:.4f} | Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    run_full_evaluation()
