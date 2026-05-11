import torch
import torch.nn as nn
from torchvision import models

def build_model(pretrained=True):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 1),
    )
    return model

class CancerDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = build_model(pretrained)

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    model = CancerDetector(pretrained=True)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    dummy = torch.randn(2, 3, 96, 96)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
