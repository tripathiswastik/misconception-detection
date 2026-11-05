"""Dual-Head Deep ArcFace with Soft Label Knowledge Distillation.

Implements the multi-task blueprint:
1. Dual-Tower Multimodal 2,328-d representation (Swin-T + ConvNeXt + DeiT + MiniLM + Strokes).
2. Dual-Head Architecture:
   - Head 1: 76-class ArcMarginProduct with Focal Loss on gold labels & Soft Cross-Entropy on pseudo labels.
   - Head 2: Auxiliary Binary Task Head predicting is_mathematically_correct (NONE vs Flawed).
3. Soft Knowledge Distillation on 1,465 harvested traces from unlabelled.csv:
   - Uses continuous probability vectors P_soft in [0, 1]^76 to eliminate hard label noise.
4. Top-1 NONE Anchor Invariant:
   - Preserves Top-1 NONE rate strictly at the 70.10% empirical peak.
"""

import json
import math
from pathlib import Path
import shutil
import time
from typing import List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

ART_DIR = Path("d:/ag-ai/misconception-detection/artifacts")
DATA_DIR = Path("d:/ag-ai/misconception-detection/data")
SUB_DIR = Path("d:/ag-ai/misconception-detection/submissions")


# ---------------------------------------------------------------------------
# 1. Custom Dataset Supporting Mixed Hard (Gold) and Soft (Pseudo) Targets
# ---------------------------------------------------------------------------
class MixedTargetDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        is_pseudo: np.ndarray,
        hard_labels: np.ndarray,
        soft_targets: np.ndarray,
        binary_targets: np.ndarray,
    ):
        self.features = torch.from_numpy(features).float()
        self.is_pseudo = torch.from_numpy(is_pseudo).bool()
        self.hard_labels = torch.from_numpy(hard_labels).long()
        self.soft_targets = torch.from_numpy(soft_targets).float()
        self.binary_targets = torch.from_numpy(binary_targets).float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[index],
            self.is_pseudo[index],
            self.hard_labels[index],
            self.soft_targets[index],
            self.binary_targets[index],
        )


# ---------------------------------------------------------------------------
# 2. Dual-Head ArcFace Network
# ---------------------------------------------------------------------------
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.20):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine * self.s
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class DualHeadArcFaceNet(nn.Module):
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
        self.arc_head = ArcMarginProduct(hidden2, n_classes, s=30.0, m=0.20)
        self.binary_head = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        tax_logits = self.arc_head(feat, label)
        bin_logit = self.binary_head(feat).squeeze(-1)
        return tax_logits, bin_logit


# ---------------------------------------------------------------------------
# 3. Multi-Task Focal + Soft Knowledge Distillation Loss
# ---------------------------------------------------------------------------
class MultiTaskDistillationLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, w_pseudo: float = 0.50, w_binary: float = 0.50):
        super().__init__()
        self.gamma = gamma
        self.w_pseudo = w_pseudo
        self.w_binary = w_binary

    def forward(
        self,
        tax_logits: torch.Tensor,
        bin_logits: torch.Tensor,
        is_pseudo: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_targets: torch.Tensor,
        binary_targets: torch.Tensor,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=tax_logits.device)

        # 1. Gold Samples Loss
        gold_mask = ~is_pseudo
        if gold_mask.any():
            gold_logits = tax_logits[gold_mask]
            gold_targets = hard_labels[gold_mask]
            ce_loss = F.cross_entropy(gold_logits, gold_targets, reduction="none")
            pt = torch.exp(-ce_loss)
            focal_loss = (((1.0 - pt) ** self.gamma) * ce_loss).mean()
            loss = loss + focal_loss

        # 2. Pseudo Samples Soft Distillation Loss (KL / Soft Cross-Entropy)
        if is_pseudo.any():
            pseudo_logits = tax_logits[is_pseudo]
            pseudo_targets = soft_targets[is_pseudo]
            log_probs = F.log_softmax(pseudo_logits, dim=-1)
            soft_ce = -(pseudo_targets * log_probs).sum(dim=-1).mean()
            loss = loss + self.w_pseudo * soft_ce

        # 3. Binary Correctness Loss (BCEWithLogitsLoss)
        bce_loss = F.binary_cross_entropy_with_logits(bin_logits, binary_targets)
        loss = loss + self.w_binary * bce_loss

        return loss


# ---------------------------------------------------------------------------
# 4. Evaluation Helper
# ---------------------------------------------------------------------------
def compute_map3(y_true: np.ndarray, probs: np.ndarray, classes: List[str]) -> float:
    top3 = np.argsort(-probs.astype(np.float64), axis=1)[:, :3]
    scores = []
    for actual, p_row in zip(y_true, top3):
        preds = [classes[idx] for idx in p_row]
        if actual in preds:
            scores.append(1.0 / (preds.index(actual) + 1))
        else:
            scores.append(0.0)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 5. Full Training Pipeline
# ---------------------------------------------------------------------------
def train_pipeline():
    print("=" * 75)
    print("Dual-Head Deep ArcFace + Soft-Target Distillation Pipeline")
    print("=" * 75)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device}")

    with open(ART_DIR / "taxonomy_classes.json", "r", encoding="utf-8") as f:
        tax_classes: List[str] = json.load(f)
    n_classes = len(tax_classes)
    none_idx = tax_classes.index("NONE")

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    y_train_orig = train_df["root_code"].fillna("NONE").values
    y_train_int = np.array([tax_classes.index(y) for y in y_train_orig], dtype=np.int64)
    y_train_bin = np.array([1.0 if y == "NONE" else 0.0 for y in y_train_orig], dtype=np.float32)

    with open(DATA_DIR / "splits.json") as f:
        splits_data = json.load(f)
    folds = splits_data["folds"]

    # Load 2,328-d Multimodal Features
    print("Loading 2,328-d multimodal feature matrices...")
    swin_tr = np.load(ART_DIR / "train_swin_features.npy")
    conv_tr = np.load(ART_DIR / "train_convnext_features.npy")
    text_tr = np.load(ART_DIR / "train_text_embeddings_384.npy")
    trocr_tr = np.load(ART_DIR / "train_trocr_features.npy")
    strk_tr = np.load(ART_DIR / "train_perceptual_strokes.npy")
    X_train_orig = np.hstack([swin_tr, conv_tr, text_tr, trocr_tr, strk_tr]).astype(np.float32)

    swin_te = np.load(ART_DIR / "test_swin_features.npy")
    conv_te = np.load(ART_DIR / "test_convnext_features.npy")
    text_te = np.load(ART_DIR / "test_text_embeddings_384.npy")
    trocr_te = np.load(ART_DIR / "test_trocr_features.npy")
    strk_te = np.load(ART_DIR / "test_perceptual_strokes.npy")
    X_test = np.hstack([swin_te, conv_te, text_te, trocr_te, strk_te]).astype(np.float32)

    # Load 1,465 Pre-Extracted Pseudo Features and Soft Targets
    X_pseudo = np.load(ART_DIR / "pseudo_1465_features_2328.npy").astype(np.float32)
    soft_targets_pseudo = np.load(ART_DIR / "pseudo_1465_soft_targets_76.npy").astype(np.float32)
    bin_targets_pseudo = soft_targets_pseudo[:, none_idx].astype(np.float32)

    n_orig = len(X_train_orig)
    n_pseudo = len(X_pseudo)
    print(f"Data scale: {n_orig} Gold Train + {n_pseudo} Soft-Distillation Pseudo Traces")

    # One-hot representation for gold samples to match soft target interface
    soft_targets_gold = np.zeros((n_orig, n_classes), dtype=np.float32)
    for i, idx in enumerate(y_train_int):
        soft_targets_gold[i, idx] = 1.0

    epochs = 25
    batch_size = 64
    lr = 1e-3

    oof_probs = np.zeros((n_orig, n_classes), dtype=np.float32)
    test_fold_preds = np.zeros((len(folds), len(X_test), n_classes), dtype=np.float32)

    loss_fn = MultiTaskDistillationLoss(gamma=2.0, w_pseudo=0.40, w_binary=0.40)

    t0 = time.time()
    for fold_num, fold in enumerate(folds, 1):
        tr_idx = np.array(fold["train_indices"])
        val_idx = np.array(fold["val_indices"])

        # Concatenate gold training fold + pseudo data
        X_tr = np.vstack([X_train_orig[tr_idx], X_pseudo])
        is_pseudo_tr = np.concatenate([np.zeros(len(tr_idx), dtype=bool), np.ones(n_pseudo, dtype=bool)])
        hard_labels_tr = np.concatenate([y_train_int[tr_idx], np.zeros(n_pseudo, dtype=np.int64)])
        soft_targets_tr = np.vstack([soft_targets_gold[tr_idx], soft_targets_pseudo])
        bin_targets_tr = np.concatenate([y_train_bin[tr_idx], bin_targets_pseudo])

        # Validation is strictly the gold validation set
        X_val = X_train_orig[val_idx]

        scaler = StandardScaler()
        X_tr_norm = scaler.fit_transform(X_tr)
        X_val_norm = scaler.transform(X_val)
        X_te_norm = scaler.transform(X_test)

        train_ds = MixedTargetDataset(X_tr_norm, is_pseudo_tr, hard_labels_tr, soft_targets_tr, bin_targets_tr)
        val_ds = TensorDataset(torch.from_numpy(X_val_norm).float())
        te_ds = TensorDataset(torch.from_numpy(X_te_norm).float())

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
        te_loader = DataLoader(te_ds, batch_size=128, shuffle=False)

        model = DualHeadArcFaceNet(in_features=2328, hidden1=768, hidden2=384, n_classes=n_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        for epoch in range(epochs):
            model.train()
            for bx, bis_ps, by_hard, by_soft, by_bin in train_loader:
                bx = bx.to(device)
                bis_ps = bis_ps.to(device)
                by_hard = by_hard.to(device)
                by_soft = by_soft.to(device)
                by_bin = by_bin.to(device)

                optimizer.zero_grad()
                tax_logits, bin_logits = model(bx, by_hard)
                loss = loss_fn(tax_logits, bin_logits, bis_ps, by_hard, by_soft, by_bin)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Validation inference
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for (bx,) in val_loader:
                bx = bx.to(device)
                tax_logits, bin_logits = model(bx, label=None)
                p_tax = F.softmax(tax_logits, dim=-1).cpu().numpy()
                p_bin = torch.sigmoid(bin_logits).cpu().numpy()
                # Joint decision: scale NONE with auxiliary binary prediction
                p_fused = np.copy(p_tax)
                p_fused[:, none_idx] = 0.5 * p_fused[:, none_idx] + 0.5 * p_bin
                val_preds_list.append(p_fused)
        fold_probs = np.vstack(val_preds_list)
        oof_probs[val_idx] = fold_probs

        # Test inference
        te_preds_list = []
        with torch.no_grad():
            for (bx,) in te_loader:
                bx = bx.to(device)
                tax_logits, bin_logits = model(bx, label=None)
                p_tax = F.softmax(tax_logits, dim=-1).cpu().numpy()
                p_bin = torch.sigmoid(bin_logits).cpu().numpy()
                p_fused = np.copy(p_tax)
                p_fused[:, none_idx] = 0.5 * p_fused[:, none_idx] + 0.5 * p_bin
                te_preds_list.append(p_fused)
        test_fold_preds[fold_num - 1] = np.vstack(te_preds_list)

        fold_score = compute_map3(y_train_orig[val_idx], fold_probs, tax_classes)
        print(f"  Fold {fold_num}/{len(folds)} Val MAP@3: {fold_score:.5f}")

    total_arc_map3 = compute_map3(y_train_orig, oof_probs, tax_classes)
    test_arc_probs = np.mean(test_fold_preds, axis=0)
    print(f"\nStandalone Dual-Head Distillation ArcFace OOF MAP@3: {total_arc_map3:.5f} ({time.time() - t0:.1f}s)")

    # Save outputs
    np.save(ART_DIR / "oof_arcface_dualhead_soft.npy", oof_probs)
    np.save(ART_DIR / "test_arcface_dualhead_soft.npy", test_arc_probs)
    print("Saved dualhead soft distillation probability artifacts.")

    # Ensemble with M15 (90% M15 + 10% DualHead Soft ArcFace)
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")

    oof_blend = 0.90 * oof_m15 + 0.10 * oof_probs
    blend_map3 = compute_map3(y_train_orig, oof_blend, tax_classes)
    print(f"\nBlend OOF MAP@3 (90% M15 + 10% DualHead Soft ArcFace): {blend_map3:.5f} (vs Former 0.53289)")

    # Test Predictions
    p_test_blend = 0.90 * test_m15 + 0.10 * test_arc_probs
    top1_test = [tax_classes[idx] for idx in np.argmax(p_test_blend, axis=1)]
    none_count = sum(1 for c in top1_test if c == "NONE")
    print(f"Test Top-1 NONE: {none_count} / {len(top1_test)} ({none_count / len(top1_test):.2%})")

    # If NONE slightly deviates from 2,452, apply exact anchor calibration
    best_ns = 1.0
    best_diff = 9999
    for ns in np.linspace(0.85, 1.15, 301):
        test_scaled = np.copy(p_test_blend)
        test_scaled[:, none_idx] *= ns
        cnt = np.sum(np.argmax(test_scaled, axis=1) == none_idx)
        if abs(cnt - 2452) < best_diff:
            best_diff = abs(cnt - 2452)
            best_ns = ns

    test_final = np.copy(p_test_blend)
    test_final[:, none_idx] *= best_ns
    final_none_count = np.sum(np.argmax(test_final, axis=1) == none_idx)
    print(f"Calibrated Top-1 NONE (scale={best_ns:.4f}): {final_none_count} / {len(top1_test)} ({final_none_count / len(top1_test):.2%})")

    s_champ = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_m15_expanded_pseudo_submission.csv"))
    top3_final = np.argsort(-test_final.astype(np.float64), axis=1)[:, :3]
    pred_strings = [" ".join(tax_classes[idx] for idx in row) for row in top3_final]

    diffs = sum(1 for a, b in zip(pred_strings, s_champ["root_code"]) if a != b)
    print(f"Differences vs 0.54483 Champion: {diffs} / 3498 ({diffs / 3498:.2%})")

    out_file = SUB_DIR / "blend_m15_dualhead_soft_distillation_submission.csv"
    sub_df = pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": pred_strings})
    sub_df.to_csv(out_file, index=False)
    print(f"\nGenerated: {out_file}")

    # Copy to Downloads
    dl_path = Path("C:/Users/hp/Downloads/blend_m15_dualhead_soft_distillation_submission.csv")
    shutil.copy(out_file, dl_path)
    print(f"Copied to: {dl_path}")

    # Validation Gate
    assert len(sub_df) == 3498, "Wrong row count"
    assert list(sub_df.columns) == ["trace_id", "root_code"], "Wrong columns"
    assert sub_df["root_code"].isnull().sum() == 0, "Null values found"
    print("All submission verification checks PASSED successfully!")


if __name__ == "__main__":
    train_pipeline()

