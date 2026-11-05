"""Train 5-Fold MobileNetV3 Vision Classifier on cuda:0 with Feature Caching.

Extracts 576-dim feature vectors from ImageNet-pretrained MobileNetV3 backbone once,
caches embeddings to disk, and trains 5-fold diagnostic heads in memory on cuda:0.

Outputs:
- artifacts/train_vision_features.npy: (1500, 576) pre-extracted train embeddings.
- artifacts/test_vision_features.npy: (3498, 576) pre-extracted test embeddings.
- artifacts/checkpoints/vision_fold{0..4}.pt: Trained head weights for each fold.
- artifacts/oof_vision.npy: (1500, 76) OOF predicted probability matrix.
- artifacts/test_vision.npy: (3498, 76) Ensemble test prediction probability matrix.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Small_Weights

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, PATHS
from src.data.dataset import MathMisconceptionData
from src.data.splits import FoldSplitter
from src.data.taxonomy import TaxonomyManager
from src.metrics.map3 import calculate_map3
from src.models.vision import MobileNetV3Classifier


class MathImageDataset(Dataset):
    """PyTorch Dataset for math handwriting images with robust error handling."""

    def __init__(
        self,
        image_paths: List[Path],
        targets: Optional[List[int]] = None,
        transform: Optional[transforms.Compose] = None
    ):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[index]
        try:
            if img_path.exists():
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    tensor = self.transform(img)
            else:
                tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
        except Exception:
            tensor = torch.zeros((3, 224, 224), dtype=torch.float32)

        target = self.targets[index] if self.targets is not None else -1
        return tensor, target


def extract_features(
    image_paths: List[Path],
    device: torch.device,
    batch_size: int = 32,
    cache_path: Optional[Path] = None
) -> np.ndarray:
    """Extract and cache 576-dim MobileNetV3 feature vectors."""
    if cache_path and cache_path.exists():
        print(f"[VisionFeatures] Loading cached features from {cache_path}...", flush=True)
        return np.load(cache_path)

    print(f"[VisionFeatures] Extracting features for {len(image_paths)} images on {device}...", flush=True)
    t0 = time.time()
    
    backbone = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    extractor = nn.Sequential(backbone.features, backbone.avgpool)
    extractor.to(device)
    extractor.eval()

    dataset = MathImageDataset(image_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    feature_list = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and CONFIG.fp16_amp)):
                feats = extractor(batch_x)
                feats = torch.flatten(feats, 1)
            feature_list.append(feats.cpu().numpy().astype(np.float32))

    features = np.concatenate(feature_list, axis=0)
    print(f"[VisionFeatures] Extraction complete in {time.time() - t0:.2f}s. Shape: {features.shape}", flush=True)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, features)
        print(f"[VisionFeatures] Cached features to {cache_path}", flush=True)

    del backbone, extractor, loader, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return features


def train_vision_ensemble(epochs: int = 15, batch_size: int = 32, lr: float = 1e-3):
    device = torch.device("cuda:0" if torch.cuda.is_available() and CONFIG.prefer_gpu else "cpu")
    print(f"[VisionTrain] Initializing on device: {device} (FP16 AMP={CONFIG.fp16_amp})", flush=True)

    tax = TaxonomyManager(PATHS.taxonomy_csv)
    all_classes = tax.all_codes(include_none=True)
    class_to_idx = {c: idx for idx, c in enumerate(all_classes)}
    n_classes = len(all_classes)
    assert n_classes == 76, f"Expected 76 classes, found {n_classes}"

    # Load splits
    splitter = FoldSplitter(n_splits=CONFIG.n_splits, random_state=CONFIG.random_state)
    folds = splitter.load_splits(PATHS.splits_cache)

    # Load data
    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    test_df = data_mgr.load_test()

    n_train = len(train_df)
    n_test = len(test_df)

    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = artifacts_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Collect image paths and targets
    train_img_paths = [PATHS.data_dir / "images" / "train" / str(f) for f in train_df["image_file"].values]
    train_targets = np.array([class_to_idx.get(str(r), class_to_idx["NONE"]) for r in train_df["root_code"].values], dtype=np.int64)
    test_img_paths = [PATHS.data_dir / "images" / "test" / str(f) for f in test_df["image_file"].values]

    # Pre-extract or load cached features
    train_feats = extract_features(
        train_img_paths, device, batch_size=32, cache_path=artifacts_dir / "train_vision_features.npy"
    )
    test_feats = extract_features(
        test_img_paths, device, batch_size=32, cache_path=artifacts_dir / "test_vision_features.npy"
    )

    oof_probs = np.zeros((n_train, n_classes), dtype=np.float32)
    test_probs = np.zeros((n_test, n_classes), dtype=np.float32)

    X_test_tensor = torch.tensor(test_feats, dtype=torch.float32).to(device)

    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        print(f"\n--- Fold {fold_idx}/{len(folds)} (Train: {len(train_idx)}, Val: {len(val_idx)}) ---", flush=True)

        X_tr = torch.tensor(train_feats[train_idx], dtype=torch.float32)
        y_tr = torch.tensor(train_targets[train_idx], dtype=torch.int64)
        X_va = torch.tensor(train_feats[val_idx], dtype=torch.float32).to(device)

        tr_dataset = TensorDataset(X_tr, y_tr)
        tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)

        # 2-layer MLP head matching MobileNetV3Classifier head
        head = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, n_classes)
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

        for epoch in range(1, epochs + 1):
            head.train()
            running_loss = 0.0
            for bx, by in tr_loader:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                optimizer.zero_grad()
                logits = head(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(by)

        # Validation eval
        head.eval()
        with torch.no_grad():
            va_logits = head(X_va)
            va_probs = torch.softmax(va_logits, dim=1).cpu().numpy()
            oof_probs[val_idx] = va_probs

            # Test eval
            te_logits = head(X_test_tensor)
            te_probs = torch.softmax(te_logits, dim=1).cpu().numpy()
            test_probs += te_probs / len(folds)

        top_indices = np.argsort(-va_probs, axis=1, kind="mergesort")[:, :3]
        fold_preds = [[all_classes[idx] for idx in row] for row in top_indices]
        y_val_str = [all_classes[train_targets[idx]] for idx in val_idx]
        fold_map = calculate_map3(y_val_str, fold_preds)
        print(f"  Fold {fold_idx} Vision MAP@3: {fold_map:.4f}", flush=True)

        torch.save(head.state_dict(), checkpoints_dir / f"vision_fold{fold_idx-1}.pt")
        del head, optimizer, tr_loader, tr_dataset

    overall_top = np.argsort(-oof_probs, axis=1, kind="mergesort")[:, :3]
    overall_preds = [[all_classes[idx] for idx in row] for row in overall_top]
    y_true_str = [all_classes[idx] for idx in train_targets]
    overall_vision_map3 = calculate_map3(y_true_str, overall_preds)
    print(f"\n[VisionTrain] Overall 5-Fold Vision OOF MAP@3: {overall_vision_map3:.4f}", flush=True)

    np.save(artifacts_dir / "oof_vision.npy", oof_probs)
    np.save(artifacts_dir / "test_vision.npy", test_probs)
    print(f"[VisionTrain] Saved artifacts/oof_vision.npy & test_vision.npy", flush=True)

    return oof_probs, test_probs, overall_vision_map3


if __name__ == "__main__":
    train_vision_ensemble(epochs=15, batch_size=32, lr=1e-3)
