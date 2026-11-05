"""MobileNetV3 Small Vision Classifier for Math Misconception Diagnosis.

Compact ImageNet-pretrained vision backbone with custom classification head
for 76 taxonomy classes (75 root misconceptions + NONE).
Optimized for NVIDIA GeForce GTX 1050 (4 GB VRAM) with memory caching.
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights


class MobileNetV3Classifier(nn.Module):
    """MobileNetV3 Small backbone with replacement 76-class diagnostic head."""

    def __init__(
        self,
        num_classes: int = 76,
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # In mobilenet_v3_small, features output 576 channels
        in_features = 576
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def unfreeze_backbone(self, lr_ratio: float = 0.1):
        """Unfreeze entire feature extractor for full fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True

    def unfreeze_top_blocks(self, num_blocks: int = 2):
        """Unfreeze the final num_blocks of the feature extractor while keeping earlier blocks frozen."""
        total_layers = len(self.features)
        split_point = max(0, total_layers - num_blocks)
        for i, layer in enumerate(self.features):
            requires_grad = (i >= split_point)
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def get_parameter_groups(
        self,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4
    ) -> List[dict]:
        """Return separate parameter groups for optimizer with differential learning rates."""
        backbone_params = [p for p in self.features.parameters() if p.requires_grad]
        head_params = list(self.classifier.parameters())
        return [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay}
        ]
