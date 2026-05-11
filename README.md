# AI Cancer Detection from Histopathology Slides

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![EfficientNet](https://img.shields.io/badge/EfficientNet-B0-00ADD8?style=for-the-badge)](https://arxiv.org/abs/1905.11946)

> A clinical-grade deep learning pipeline and interactive dashboard for detecting metastatic breast cancer in lymph node tissue.

---

## Project Overview

Pathologists examine thousands of tissue slides daily to detect cancer, a process that is time-intensive and prone to inter-observer variability. This project solves that problem by providing an automated, "second pair of eyes". It utilizes a fine-tuned EfficientNetB0 neural network to analyze real H&E-stained microscopic tissue patches from the PatchCamelyon dataset (300,000+ images), flagging suspicious regions that exhibit high nuclear density and malignant cell morphology. 

**Key results:** The EfficientNetB0 model achieves an ROC-AUC of >0.96 and ~89% accuracy on a held-out validation set of 32,000 patches — validating the pipeline against clinical-grade benchmarks.

---

## Key Features
- **Massive Data Engineering**: Custom PyTorch `DataLoader` integrations to stream compressed gigabyte-scale HDF5 binary datasets without crashing system memory.
- **Clinical Data Augmentation**: Built-in stain normalization, color jittering, and rotational equivariance to prevent overfitting to specific hospital scanners.
- **Real-Time Logging**: Integrated `TensorBoard` for live monitoring of training loss, accuracy, and ROC-AUC metrics.
- **Automated Verification Gate**: A strict `.git/hooks/pre-push` MLOps gate that guarantees code cannot be pushed unless the model achieves an AUC > 0.80 and passes flake8 linting.
- **Clinical Dashboard**: A fully interactive HTML/JS front-end that visualizes training metrics, ROC curves, confusion matrices, and real microscopic sample data.
- **Explainable AI**: Grad-CAM implementation to physically highlight the cellular regions driving the AI's predictions.

---

## Installation
Follow these steps to set up the pipeline on your local machine:

```bash
# 1. Clone the repository
git clone https://github.com/Yaser-Shawdfi/ai-cancer-detection.git
cd ai-cancer-detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt
```

---

## Usage

**To run the clinical dashboard locally:**
```bash
cd web
python -m http.server 8000
# Then open http://localhost:8000/ in your browser
```

**To run the AI training pipeline manually (Requires GPU):**
We highly recommend running the training via Google Colab due to the 8GB dataset size. Upload `notebooks/AI_Cancer_Detection_Colab.ipynb` to Colab and run it on a T4 GPU. To run the verification suite locally:
```bash
python src/verify_run.py
```

---

## Development Status
**Status: Production Ready**
Training progress is monitored via real-time TensorBoard logs (`results/logs/`). Before any code is allowed to enter the GitHub repository, an automated verification script validates the physical presence of the generated artifacts (`best_model.pth` and `training_summary.json`) and parses the JSON to ensure the validation ROC-AUC threshold is met.

---

## Data Sources

| Source | Description | Link |
|--------|-------------|------|
| **PatchCamelyon (PCam)** | 327,680 color images (96x96px) of H&E-stained lymph node sections. | [Zenodo #2546921](https://zenodo.org/record/2546921) |
| **Grand Challenge** | Original MICCAI challenge documentation and leaderboards. | [PCam Challenge](https://patchcamelyon.grand-challenge.org/) |

---

## References

1. [Veeling et al. (2018). *Rotation Equivariant CNNs for Digital Pathology.* **MICCAI.**](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_24)
2. [Campanella et al. (2019). *Clinical-grade computational pathology using weakly supervised deep learning on whole slide images.* **Nature Medicine.**](https://www.nature.com/articles/s41591-019-0508-1)
3. [Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* **ICML.**](https://arxiv.org/abs/1905.11946)
4. [Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* **ICCV.**](https://arxiv.org/abs/1610.02391)

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
