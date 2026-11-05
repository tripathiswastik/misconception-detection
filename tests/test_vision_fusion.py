"""Unit tests for MobileNetV3Classifier and MultimodalLateFusion."""
import numpy as np
import torch
from src.models.fusion import MultimodalLateFusion
from src.models.vision import MobileNetV3Classifier


def test_mobilenet_v3_classifier_forward():
    model = MobileNetV3Classifier(num_classes=76, pretrained=False, freeze_backbone=True)
    model.eval()

    # Pass dummy batch of (B=2, C=3, H=224, W=224)
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 76)
    assert not torch.isnan(output).any()


def test_multimodal_late_fusion_alignment():
    classes = ["FRM-01", "INT-04", "NONE"]
    text_scores = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1]
    ])
    vision_scores = np.array([
        [0.6, 0.2, 0.2],
        [0.2, 0.7, 0.1]
    ])
    y_true = ["FRM-01", "INT-04"]

    res = MultimodalLateFusion.tune_fusion_weight(
        y_true=y_true,
        text_scores=text_scores,
        vision_scores=vision_scores,
        classes=classes,
        candidate_weights=[0.0, 0.1, 0.2, 0.3]
    )

    assert "best_oof_map3" in res
    assert "optimal_vision_weight" in res
    assert res["best_oof_map3"] >= 0.5
