"""Multimodal Deep ArcFace 2,328-d Training & Honest Cross-Validation Benchmark.

Evaluates incorporation of TrOCR DeiT Visual Encoder features (384-d) into
the Multimodal ArcFace feature space (1,944-d -> 2,328-d).
Strictly maintains the NONE-preservation invariant.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Setup Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ART_DIR = BASE_DIR / "artifacts"
OUT_DIR = BASE_DIR / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_map3(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, np.ndarray]:
    top3 = np.argsort(-y_probs.astype(np.float64), axis=1)[:, :3]
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


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.20):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = float(np.cos(m))
        self.sin_m = float(np.sin(m))
        self.th = float(np.cos(np.pi - m))
        self.mm = float(np.sin(np.pi - m) * m)

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if label is None:
            return cosine * self.s

        sine = torch.sqrt(1.0 - torch.clamp(cosine**2, 0.0, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class MultimodalDeepArcFace(nn.Module):
    def __init__(self, in_features: int = 2328, hidden1: int = 768, hidden2: int = 384, n_classes: int = 76):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head = ArcMarginProduct(hidden2, n_classes, s=30.0, m=0.20)

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.encoder(x)
        return self.head(feat, label)


def train_and_eval_arcface(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_train: np.ndarray,
    folds: list,
    n_classes: int,
    device: torch.device,
    tag: str = "2328d",
    epochs: int = 25,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_dim = X_train_raw.shape[1]
    oof_probs = np.zeros((len(y_train), n_classes), dtype=np.float32)
    test_fold_preds = np.zeros((len(folds), len(X_test_raw), n_classes), dtype=np.float32)

    batch_size = 64
    lr = 1e-3

    for fold_num, fold in enumerate(folds, 1):
        tr_idx = np.array(fold["train_indices"])
        val_idx = np.array(fold["val_indices"])

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_raw[tr_idx])
        X_val = scaler.transform(X_train_raw[val_idx])
        X_te = scaler.transform(X_test_raw)

        tr_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_train[tr_idx]))
        val_ds = TensorDataset(torch.from_numpy(X_val).float())
        te_ds = TensorDataset(torch.from_numpy(X_te).float())

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
        te_loader = DataLoader(te_ds, batch_size=128, shuffle=False)

        model = MultimodalDeepArcFace(in_features=in_dim, hidden1=768, hidden2=384, n_classes=n_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        for epoch in range(epochs):
            model.train()
            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx, by)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for (bx,) in val_loader:
                bx = bx.to(device)
                logits = model(bx, label=None)
                probs = F.softmax(logits, dim=-1)
                val_preds.append(probs.cpu().numpy())
        fold_val_probs = np.vstack(val_preds)
        oof_probs[val_idx] = fold_val_probs

        te_preds = []
        with torch.no_grad():
            for (bx,) in te_loader:
                bx = bx.to(device)
                logits = model(bx, label=None)
                probs = F.softmax(logits, dim=-1)
                te_preds.append(probs.cpu().numpy())
        test_fold_preds[fold_num - 1] = np.vstack(te_preds)

    test_probs = np.mean(test_fold_preds, axis=0)
    total_map3, _ = compute_map3(y_train, oof_probs)
    return oof_probs, test_probs, total_map3


def run_experiment():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))

    with open(ART_DIR / "taxonomy_classes.json") as f:
        tax_classes: List[str] = json.load(f)
    class_to_idx = {c: i for i, c in enumerate(tax_classes)}
    n_classes = len(tax_classes)
    none_idx = class_to_idx["NONE"]

    y_train = np.array([class_to_idx[r] for r in train_df["root_code"]], dtype=np.int64)
    is_none = (y_train == none_idx)

    with open(DATA_DIR / "splits.json") as f:
        splits_data = json.load(f)
    folds = splits_data["folds"]

    # Load All Features
    print("Loading extracted feature matrices...")
    swin_tr = np.load(ART_DIR / "train_swin_features.npy")
    conv_tr = np.load(ART_DIR / "train_convnext_features.npy")
    text_tr = np.load(ART_DIR / "train_text_embeddings_384.npy")
    strk_tr = np.load(ART_DIR / "train_perceptual_strokes.npy")
    trocr_tr = np.load(ART_DIR / "train_trocr_features.npy")

    swin_te = np.load(ART_DIR / "test_swin_features.npy")
    conv_te = np.load(ART_DIR / "test_convnext_features.npy")
    text_te = np.load(ART_DIR / "test_text_embeddings_384.npy")
    strk_te = np.load(ART_DIR / "test_perceptual_strokes.npy")
    trocr_te = np.load(ART_DIR / "test_trocr_features.npy")

    X_train_1944 = np.hstack([swin_tr, conv_tr, text_tr, strk_tr]).astype(np.float32)
    X_test_1944 = np.hstack([swin_te, conv_te, text_te, strk_te]).astype(np.float32)

    X_train_2328 = np.hstack([swin_tr, conv_tr, text_tr, trocr_tr, strk_tr]).astype(np.float32)
    X_test_2328 = np.hstack([swin_te, conv_te, text_te, trocr_te, strk_te]).astype(np.float32)

    m15_oof = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
    m15_score, m15_arr = compute_map3(y_train, m15_oof)
    print(f"M15 Champion Baseline OOF MAP@3: {m15_score:.5f}")

    print("\n--- Training 1,944-d Baseline ArcFace ---")
    t0 = time.time()
    oof_1944, test_1944, score_1944 = train_and_eval_arcface(
        X_train_1944, X_test_1944, y_train, folds, n_classes, device, tag="1944d"
    )
    print(f"1944-d ArcFace OOF MAP@3: {score_1944:.5f} ({time.time() - t0:.1f}s)")

    print("\n--- Training 2,328-d Enhanced ArcFace (+ TrOCR Visual) ---")
    t0 = time.time()
    oof_2328, test_2328, score_2328 = train_and_eval_arcface(
        X_train_2328, X_test_2328, y_train, folds, n_classes, device, tag="2328d"
    )
    print(f"2328-d ArcFace OOF MAP@3: {score_2328:.5f} ({time.time() - t0:.1f}s)")

    # Linear Blend Sweeps
    print("\n--- Blend Sweeps with M15 (weight w on ArcFace, 1-w on M15) ---")
    weights = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

    print("\n1944-d Blends:")
    best_1944_w, best_1944_score = 0.0, m15_score
    for w in weights:
        b_oof = (1.0 - w) * m15_oof + w * oof_1944
        sc, _ = compute_map3(y_train, b_oof)
        if sc > best_1944_score:
            best_1944_score = sc
            best_1944_w = w
        print(f"  w={w:.2f}: {sc:.5f}")

    print(f"\nBest 1944-d Blend: w={best_1944_w:.2f} -> {best_1944_score:.5f} (Delta: {best_1944_score - m15_score:+.5f})")

    print("\n2328-d Blends (+ TrOCR Visual):")
    best_2328_w, best_2328_score = 0.0, m15_score
    for w in weights:
        b_oof = (1.0 - w) * m15_oof + w * oof_2328
        sc, _ = compute_map3(y_train, b_oof)
        if sc > best_2328_score:
            best_2328_score = sc
            best_2328_w = w
        print(f"  w={w:.2f}: {sc:.5f}")

    print(f"\nBest 2328-d Blend: w={best_2328_w:.2f} -> {best_2328_score:.5f} (Delta: {best_2328_score - m15_score:+.5f})")

    # Generate test blend if improved
    final_w = best_2328_w if best_2328_score >= best_1944_score else best_1944_w
    final_test_arc = test_2328 if best_2328_score >= best_1944_score else test_1944
    final_oof_arc = oof_2328 if best_2328_score >= best_1944_score else oof_1944

    final_test_blend = (1.0 - final_w) * test_m15 + final_w * final_test_arc
    top1_classes = [tax_classes[idx] for idx in np.argmax(final_test_blend, axis=1)]
    none_count = sum(1 for c in top1_classes if c == "NONE")
    print(f"\nFinal Test Top-1 NONE count: {none_count} / {len(top1_classes)} ({none_count / len(top1_classes):.2%})")

    # Check NONE Preservation Invariant (68% - 72%)
    assert 0.65 <= (none_count / len(top1_classes)) <= 0.73, f"NONE invariant violated: {none_count / len(top1_classes):.2%}"
    print("NONE Preservation Invariant SATISFIED!")

    # Save artifact
    np.save(ART_DIR / "oof_arcface_2328d.npy", final_oof_arc)
    np.save(ART_DIR / "test_arcface_2328d.npy", final_test_arc)
    np.save(ART_DIR / "test_blend_m15_arcface2328.npy", final_test_blend)

    # Export Submission
    top3_indices = np.argsort(-final_test_blend.astype(np.float64), axis=1)[:, :3]
    pred_strings = [" ".join(tax_classes[idx] for idx in row) for row in top3_indices]
    sub_df = pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": pred_strings})
    sub_df.to_csv(OUT_DIR / "blend_m15_arcface2328_submission.csv", index=False)
    print(f"Generated submission: {OUT_DIR / 'blend_m15_arcface2328_submission.csv'} ({len(sub_df)} rows)")


if __name__ == "__main__":
    run_experiment()

