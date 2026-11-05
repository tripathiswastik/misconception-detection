"""Composite Objective Optimization with 448x448 Letterbox Multimodal ArcFace.

Implements the complete CoLearn MAP@3 optimization formula:
1. Class-Weighted Focal Loss with dynamic inverse-frequency alpha weights:
   alpha[c] = total_samples / (num_classes * class_count[c]), clamped to [0.2, 5.0].
2. Direct MAP@3 Soft-Ranking Proxy Loss (Multi-Class Margin Ranking):
   Hinge loss over top-5 hardest negative distractors: max(0, margin - (s_y - s_neg)).
3. Soft-Target Cross-Entropy for 1,465 pseudo-labeled traces from unlabelled.csv.
4. Auxiliary Binary Task Head (NONE vs Flawed) with BCE loss.
5. 448x448 Letterbox ConvNeXt (784 visual tokens, 4x pixel clarity, 0% aspect distortion).
6. Preserves the strict 70.10% Top-1 NONE anchor invariant.
"""

import json
import math
from pathlib import Path
import shutil
import time
from typing import Dict, List, Optional, Tuple, cast

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
SUB_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Dataset Supporting Mixed Hard (Gold) and Soft (Pseudo) Targets
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
# 2. Dual-Head ArcFace Network (LayerNorm immune to BatchNorm collapse)
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
# 3. Composite MAP@3 Optimization Loss Function
# ---------------------------------------------------------------------------
class CoLearnCompositeLoss(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: float = 2.0,
        margin: float = 1.0,
        top_k: int = 5,
        lambda_rank: float = 0.20,
        w_pseudo: float = 0.40,
        w_binary: float = 0.40,
    ):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.margin = margin
        self.top_k = top_k
        self.lambda_rank = lambda_rank
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

        # 1. Gold Samples Loss: Class-Weighted Focal + MAP@3 Margin Ranking Proxy
        gold_mask = ~is_pseudo
        if gold_mask.any():
            g_logits = tax_logits[gold_mask]
            g_targets = hard_labels[gold_mask]
            batch_sz = g_logits.shape[0]

            # A. Class-Weighted Focal Loss
            ce_loss = F.cross_entropy(g_logits, g_targets, reduction="none")
            pt = torch.exp(-ce_loss)
            at = self.alpha.to(g_targets.device)[g_targets]
            focal_loss = (at * ((1.0 - pt) ** self.gamma) * ce_loss).mean()
            loss = loss + focal_loss

            # B. Direct MAP@3 Soft-Ranking Margin Proxy Loss
            target_scores = g_logits[torch.arange(batch_sz, device=g_logits.device), g_targets].unsqueeze(1)
            mask = torch.ones_like(g_logits, dtype=torch.bool)
            mask[torch.arange(batch_sz, device=g_logits.device), g_targets] = False
            neg_scores = g_logits[mask].view(batch_sz, -1)
            top_negatives, _ = torch.topk(neg_scores, min(self.top_k, neg_scores.shape[1]), dim=-1)
            rank_loss = F.relu(self.margin - (target_scores - top_negatives)).mean()
            loss = loss + self.lambda_rank * rank_loss

        # 2. Pseudo Samples: Soft-Target Cross-Entropy Knowledge Distillation
        if is_pseudo.any():
            ps_logits = tax_logits[is_pseudo]
            ps_targets = soft_targets[is_pseudo]
            log_probs = F.log_softmax(ps_logits, dim=-1)
            soft_loss = -(ps_targets * log_probs).sum(dim=-1).mean()
            loss = loss + self.w_pseudo * soft_loss

        # 3. Auxiliary Binary Correctness Task Loss
        bce_loss = F.binary_cross_entropy_with_logits(bin_logits, binary_targets)
        loss = loss + self.w_binary * bce_loss

        return loss


# ---------------------------------------------------------------------------
# 4. MAP@3 Metric Function
# ---------------------------------------------------------------------------
def compute_map3(y_true: np.ndarray, y_probs: np.ndarray, classes: List[str]) -> float:
    top3_indices = np.argsort(-y_probs.astype(np.float64), axis=1)[:, :3]
    scores = []
    for true_label, top3 in zip(y_true, top3_indices):
        preds = [classes[idx] for idx in top3]
        if true_label == preds[0]:
            scores.append(1.0)
        elif true_label == preds[1]:
            scores.append(0.5)
        elif true_label == preds[2]:
            scores.append(1.0 / 3.0)
        else:
            scores.append(0.0)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 5. Full Composite Training Pipeline
# ---------------------------------------------------------------------------
def train_composite_pipeline():
    print("=" * 80)
    print("CoLearn Composite Objective Optimization (Focal + Margin + Soft Distill + 448)")
    print("=" * 80)

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

    # Calculate Class Inverse-Frequency Alpha Weights
    class_counts = np.zeros(n_classes, dtype=np.int64)
    for val in y_train_int:
        class_counts[val] += 1
    raw_alpha = len(y_train_int) / (n_classes * np.maximum(class_counts, 1.0))
    alpha_clamped = np.clip(raw_alpha, 0.20, 5.0).astype(np.float32)
    # Normalize alpha so mean is 1.0
    alpha_norm = (alpha_clamped / alpha_clamped.mean()).astype(np.float32)
    alpha_tensor = torch.from_numpy(alpha_norm).float()
    print(f"Alpha Weights: min={alpha_norm.min():.3f}, max={alpha_norm.max():.3f}, NONE_alpha={alpha_norm[none_idx]:.3f}")

    # Load 448x448 Multimodal Features
    print("Loading 448x448 multimodal feature matrices...")
    swin_tr = np.load(ART_DIR / "train_swin_features.npy")
    conv448_tr = np.load(ART_DIR / "train_convnext_448_letterbox.npy")
    text_tr = np.load(ART_DIR / "train_text_embeddings_384.npy")
    trocr_tr = np.load(ART_DIR / "train_trocr_features.npy")
    strk_tr = np.load(ART_DIR / "train_perceptual_strokes.npy")

    swin_te = np.load(ART_DIR / "test_swin_features.npy")
    conv448_te = np.load(ART_DIR / "test_convnext_448_letterbox.npy")
    text_te = np.load(ART_DIR / "test_text_embeddings_384.npy")
    trocr_te = np.load(ART_DIR / "test_trocr_features.npy")
    strk_te = np.load(ART_DIR / "test_perceptual_strokes.npy")

    X_train_orig = np.hstack([swin_tr, conv448_tr, text_tr, trocr_tr, strk_tr]).astype(np.float32)
    X_test = np.hstack([swin_te, conv448_te, text_te, trocr_te, strk_te]).astype(np.float32)

    pseudo_old = np.load(ART_DIR / "pseudo_1465_features_2328.npy")
    pseudo_conv448 = np.load(ART_DIR / "pseudo_convnext_448_letterbox.npy")
    X_pseudo = np.hstack([pseudo_old[:, :768], pseudo_conv448, pseudo_old[:, 1536:]]).astype(np.float32)
    soft_targets_pseudo = np.load(ART_DIR / "pseudo_1465_soft_targets_76.npy").astype(np.float32)
    bin_targets_pseudo = soft_targets_pseudo[:, none_idx].astype(np.float32)

    n_orig = len(X_train_orig)
    n_pseudo = len(X_pseudo)
    print(f"Data scale: {n_orig} Gold Train + {n_pseudo} Soft-Distillation Pseudo Traces (Dim: 2,328)")

    soft_targets_gold = np.zeros((n_orig, n_classes), dtype=np.float32)
    for i, idx in enumerate(y_train_int):
        soft_targets_gold[i, idx] = 1.0

    epochs = 25
    batch_size = 64
    lr = 1e-3

    oof_probs = np.zeros((n_orig, n_classes), dtype=np.float32)
    test_fold_preds = np.zeros((len(folds), len(X_test), n_classes), dtype=np.float32)

    # Composite Loss: Focal (alpha-weighted) + Margin Ranking (lambda=0.20, margin=1.0, top-5) + Soft CE + BCE
    loss_fn = CoLearnCompositeLoss(
        alpha=alpha_tensor,
        gamma=2.0,
        margin=1.0,
        top_k=5,
        lambda_rank=0.20,
        w_pseudo=0.40,
        w_binary=0.40,
    ).to(device)

    t0 = time.time()
    for fold_num, fold in enumerate(folds, 1):
        tr_idx = np.array(fold["train_indices"])
        val_idx = np.array(fold["val_indices"])

        X_tr = np.vstack([X_train_orig[tr_idx], X_pseudo])
        is_pseudo_tr = np.concatenate([np.zeros(len(tr_idx), dtype=bool), np.ones(n_pseudo, dtype=bool)])
        hard_labels_tr = np.concatenate([y_train_int[tr_idx], np.zeros(n_pseudo, dtype=np.int64)])
        soft_targets_tr = np.vstack([soft_targets_gold[tr_idx], soft_targets_pseudo])
        bin_targets_tr = np.concatenate([y_train_bin[tr_idx], bin_targets_pseudo])

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
        print(f"  Fold {fold_num}/{len(folds)} Val MAP@3: {fold_score:.5f}", flush=True)

    total_arc_map3 = compute_map3(y_train_orig, oof_probs, tax_classes)
    test_arc_probs = np.mean(test_fold_preds, axis=0)
    print(f"\nStandalone Composite ArcFace OOF MAP@3: {total_arc_map3:.5f} ({time.time() - t0:.1f}s)", flush=True)

    # Save outputs
    np.save(ART_DIR / "oof_arcface_composite_448.npy", oof_probs)
    np.save(ART_DIR / "test_arcface_composite_448.npy", test_arc_probs)
    print("Saved composite loss 448 probability artifacts.", flush=True)

    # Ensemble with M15 (Evaluate grid of blend weights)
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")

    best_w = 0.10
    best_blend_score = 0.0
    for w in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        score = compute_map3(y_train_orig, (1.0 - w) * oof_m15 + w * oof_probs, tax_classes)
        print(f"  Blend Weight {int(w*100)}% ArcFace + {int((1-w)*100)}% M15: OOF MAP@3 = {score:.5f}", flush=True)
        if score > best_blend_score:
            best_blend_score = score
            best_w = w

    print(f"\nOptimal Blend Weight: {int(best_w*100)}% ArcFace (OOF MAP@3: {best_blend_score:.5f})", flush=True)
    p_test_blend = (1.0 - best_w) * test_m15 + best_w * test_arc_probs

    # Calibrate Top-1 NONE strictly to 2,452
    top1_raw = [tax_classes[idx] for idx in np.argmax(p_test_blend, axis=1)]
    raw_none = sum(1 for c in top1_raw if c == "NONE")
    print(f"Raw Test Top-1 NONE: {raw_none} / {len(top1_raw)} ({raw_none / len(top1_raw):.2%})", flush=True)

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
    calib_none = np.sum(np.argmax(test_final, axis=1) == none_idx)
    print(f"Calibrated Top-1 NONE (scale={best_ns:.4f}): {calib_none} / {len(top1_raw)} ({calib_none / len(top1_raw):.2%})", flush=True)

    s_champ = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_m15_expanded_pseudo_submission.csv"))
    top3_final = np.argsort(-test_final.astype(np.float64), axis=1)[:, :3]
    pred_strings = [" ".join(tax_classes[idx] for idx in row) for row in top3_final]

    diffs = sum(1 for a, b in zip(pred_strings, s_champ["root_code"]) if a != b)
    print(f"Differences vs 0.54483 Champion: {diffs} / 3498 ({diffs / 3498:.2%})", flush=True)

    out_file = SUB_DIR / "blend_m15_composite_448_submission.csv"
    sub_df = pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": pred_strings})
    sub_df.to_csv(out_file, index=False)
    print(f"\nGenerated: {out_file}", flush=True)

    # Copy to Downloads
    dl_path = Path("C:/Users/hp/Downloads/blend_m15_composite_448_submission.csv")
    shutil.copy(out_file, dl_path)
    print(f"Copied to: {dl_path}", flush=True)

    # Verification Gate
    assert len(sub_df) == 3498, "Wrong row count"
    assert list(sub_df.columns) == ["trace_id", "root_code"], "Wrong columns"
    assert sub_df["root_code"].isnull().sum() == 0, "Null values found"
    assert all(len(s.split()) == 3 for s in sub_df["root_code"]), "Invalid triplet"
    assert all(len(set(s.split())) == 3 for s in sub_df["root_code"]), "Duplicate codes in row"
    print("All submission verification checks PASSED successfully!", flush=True)


if __name__ == "__main__":
    train_composite_pipeline()
