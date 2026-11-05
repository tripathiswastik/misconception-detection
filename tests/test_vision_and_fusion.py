"""Unit and integration tests for MobileNetV3 vision classifier and late fusion."""
import json
import numpy as np
import pytest
import torch
from pathlib import Path
from PIL import Image

from src.models.vision import MobileNetV3Classifier
from src.models.train_vision import MathImageDataset
from src.models.fusion import MultimodalLateFusion


def test_mobilenet_v3_classifier_forward():
    model = MobileNetV3Classifier(num_classes=76, pretrained=False, freeze_backbone=True)
    # Check frozen backbone
    for param in model.features.parameters():
        assert not param.requires_grad
    # Check trainable classifier
    for param in model.classifier.parameters():
        assert param.requires_grad

    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 76)

    # Test unfreeze
    model.unfreeze_backbone()
    for param in model.features.parameters():
        assert param.requires_grad


def test_math_image_dataset(tmp_path):
    img_dir = tmp_path / "test_imgs"
    img_dir.mkdir()
    img_path1 = img_dir / "sample1.png"
    img_path2 = img_dir / "sample2.png"

    Image.new("RGB", (100, 100), color="blue").save(img_path1)
    # img_path2 doesn't exist to test fallback

    dataset = MathImageDataset(
        image_paths=[img_path1, img_path2],
        targets=[5, 12]
    )

    assert len(dataset) == 2

    # Sample 1 (valid)
    t1, y1 = dataset[0]
    assert t1.shape == (3, 224, 224)
    assert y1 == 5

    # Sample 2 (missing fallback)
    t2, y2 = dataset[1]
    assert t2.shape == (3, 224, 224)
    assert torch.equal(t2, torch.zeros(3, 224, 224))
    assert y2 == 12


def test_multimodal_late_fusion_alignment():
    source_probs = np.array([
        [0.8, 0.2],
        [0.1, 0.9]
    ], dtype=np.float32)
    source_classes = ["CODE_A", "CODE_B"]
    target_classes = ["CODE_B", "CODE_A", "NONE"]

    aligned = MultimodalLateFusion.align_predictions(
        source_probs=source_probs,
        source_classes=source_classes,
        target_classes=target_classes
    )

    assert aligned.shape == (2, 3)
    # CODE_B is at target index 0, CODE_A at target index 1
    np.testing.assert_allclose(aligned[0], [0.2, 0.8, 0.0])
    np.testing.assert_allclose(aligned[1], [0.9, 0.1, 0.0])


def test_multimodal_late_fusion_tuning():
    y_true = ["A", "B", "A", "C"]
    classes = ["A", "B", "C"]

    # Text scores: perfectly predicts A and C, struggles on B
    text_scores = np.array([
        [0.9, 0.05, 0.05],  # A (correct)
        [0.4, 0.3, 0.3],    # B (top is A, mistake)
        [0.8, 0.1, 0.1],    # A (correct)
        [0.1, 0.1, 0.8],    # C (correct)
    ], dtype=np.float32)

    # Vision scores: strongly confident on sample 1 being B
    vision_scores = np.array([
        [0.5, 0.3, 0.2],
        [0.1, 0.9, 0.0],    # B (strong)
        [0.5, 0.3, 0.2],
        [0.1, 0.2, 0.7],
    ], dtype=np.float32)

    results = MultimodalLateFusion.tune_fusion_weight(
        y_true=y_true,
        text_scores=text_scores,
        vision_scores=vision_scores,
        classes=classes,
        candidate_weights=[0.0, 0.2, 0.5]
    )

    assert "best_oof_map3" in results
    assert "optimal_vision_weight" in results
    assert "baseline_text_map3" in results
    assert results["best_oof_map3"] >= results["baseline_text_map3"]


def test_apply_fusion():
    text = np.array([[0.8, 0.2]], dtype=np.float32)
    vision = np.array([[0.2, 0.8]], dtype=np.float32)
    fused = MultimodalLateFusion.apply_fusion(text, vision, vision_weight=0.5)
    np.testing.assert_allclose(fused, [[0.5, 0.5]])
