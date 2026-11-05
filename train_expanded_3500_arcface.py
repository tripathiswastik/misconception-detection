"""Train Expanded 5,000-Sample Multimodal ArcFace Model (1,944-d).

Features:
  - Swin-T @ 224x224 (768-d)
  - ConvNeXt-Tiny @ 448x448 Letterbox (768-d)
  - MiniLM Multilingual Text Embeddings (384-d)
  - Fast Perceptual Strokes (24-d)
  Total: 1,944-d (100% free of TrOCR DeiT attention-padding noise)

Training Data:
  - 1,500 Gold Train Traces
  - 3,500 High-Confidence Distilled Unlabelled Student Traces across all 282 Test Questions
  Total: 5,000 Training Traces (3.3x expansion over gold)

Architecture & Loss:
  - DualHeadArcFaceNet (1,944 -> 768 -> 384, LayerNorm only)
  - Composite Focal Loss + Margin Ranking Proxy Loss on gold
  - Soft Knowledge Distillation on pseudo traces (w=0.40)
  - Auxiliary Binary Correctness Regularizer (w=0.40, backward pass only)
  - Pure 76-Class Softmax Simplex Inference
  - 95/5 Champion Tri-Blend + Frequency Recalibration
  - Strict Golden NONE Anchor Invariant: exactly 2,452 / 3,498 (70.10%)
"""

from collections import Counter
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

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
SUB_DIR = PROJECT_ROOT / "submissions"
SUB_DIR.mkdir(parents=True, exist_ok=True)


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

    def forward(
        self,
        input: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        is_pseudo: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine * self.s
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        arc_logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        if is_pseudo is not None and is_pseudo.any():
            # For pseudo samples, do not apply hard label margin
            mask = is_pseudo.unsqueeze(-1).expand_as(cosine)
            output = torch.where(mask, cosine, arc_logits)
        else:
            output = arc_logits
        return output * self.s


class DualHeadArcFaceNet(nn.Module):
    def __init__(self, in_features: int = 1944, hidden1: int = 768, hidden2: int = 384, n_classes: int = 76):
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

    def forward(
        self,
        x: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        is_pseudo: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        tax_logits = self.arc_head(feat, label, is_pseudo)
        bin_logit = self.binary_head(feat).squeeze(-1)
        return tax_logits, bin_logit


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

        # 1. Gold Samples Loss: Class-Weighted Focal + Margin Ranking Proxy
        gold_mask = ~is_pseudo
        if gold_mask.any():
            g_logits = tax_logits[gold_mask]
            g_targets = hard_labels[gold_mask]
            batch_sz = g_logits.shape[0]

            ce_loss = F.cross_entropy(g_logits, g_targets, reduction="none")
            pt = torch.exp(-ce_loss)
            at = self.alpha.to(g_targets.device)[g_targets]
            focal_loss = (at * ((1.0 - pt) ** self.gamma) * ce_loss).mean()
            loss = loss + focal_loss

            target_scores = g_logits[torch.arange(batch_sz, device=g_logits.device), g_targets].unsqueeze(1)
            mask = torch.ones_like(g_logits, dtype=torch.bool)
            mask[torch.arange(batch_sz, device=g_logits.device), g_targets] = False
            neg_scores = g_logits[mask].view(batch_sz, -1)
            top_negatives, _ = torch.topk(neg_scores, min(self.top_k, neg_scores.shape[1]), dim=-1)
            rank_loss = F.relu(self.margin - (target_scores - top_negatives)).mean()
            loss = loss + self.lambda_rank * rank_loss

        # 2. Pseudo Samples: Soft-Target Knowledge Distillation
        if is_pseudo.any():
            ps_logits = tax_logits[is_pseudo]
            ps_targets = soft_targets[is_pseudo]
            log_probs = F.log_softmax(ps_logits, dim=-1)
            soft_loss = -(ps_targets * log_probs).sum(dim=-1).mean()
            loss = loss + self.w_pseudo * soft_loss

        # 3. Auxiliary Binary Correctness Loss
        bce_loss = F.binary_cross_entropy_with_logits(bin_logits, binary_targets)
        loss = loss + self.w_binary * bce_loss

        return loss


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


def train_and_evaluate():
    print("=" * 80)
    print("Expanded 5,000-Sample Multimodal ArcFace Training (1,944-d)")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device}", flush=True)

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

    class_counts = np.zeros(n_classes, dtype=np.int64)
    for val in y_train_int:
        class_counts[val] += 1
    raw_alpha = len(y_train_int) / (n_classes * np.maximum(class_counts, 1.0))
    alpha_clamped = np.clip(raw_alpha, 0.20, 5.0).astype(np.float32)
    alpha_norm = (alpha_clamped / alpha_clamped.mean()).astype(np.float32)
    alpha_tensor = torch.from_numpy(alpha_norm).float()

    print("Loading 1,944-d multimodal feature matrices...", flush=True)
    swin_tr = np.load(ART_DIR / "train_swin_features.npy")
    conv448_tr = np.load(ART_DIR / "train_convnext_448_letterbox.npy")
    text_tr = np.load(ART_DIR / "train_text_embeddings_384.npy")
    strk_tr = np.load(ART_DIR / "train_perceptual_strokes.npy")

    swin_te = np.load(ART_DIR / "test_swin_features.npy")
    conv448_te = np.load(ART_DIR / "test_convnext_448_letterbox.npy")
    text_te = np.load(ART_DIR / "test_text_embeddings_384.npy")
    strk_te = np.load(ART_DIR / "test_perceptual_strokes.npy")

    # Canonical order: Swin (768), ConvNeXt-448 (768), Text (384), Strokes (24) = 1,944-d
    X_train_orig = np.hstack([swin_tr, conv448_tr, text_tr, strk_tr]).astype(np.float32)
    X_test = np.hstack([swin_te, conv448_te, text_te, strk_te]).astype(np.float32)
    print(f"Train Gold: {X_train_orig.shape}, Test: {X_test.shape}", flush=True)

    X_pseudo = np.load(ART_DIR / "pseudo_3500_features_1944.npy").astype(np.float32)
    soft_targets_pseudo = np.load(ART_DIR / "pseudo_3500_soft_targets_76.npy").astype(np.float32)
    bin_targets_pseudo = soft_targets_pseudo[:, none_idx].astype(np.float32)
    print(f"Pseudo 3,500 Matrix: {X_pseudo.shape}, Soft Targets: {soft_targets_pseudo.shape}", flush=True)

    n_orig = len(X_train_orig)
    n_pseudo = len(X_pseudo)
    total_train_samples = n_orig + n_pseudo
    print(f"Total Effective Training Set: {total_train_samples} samples (3.33x Gold)\n", flush=True)

    soft_targets_gold = np.zeros((n_orig, n_classes), dtype=np.float32)
    for i, idx in enumerate(y_train_int):
        soft_targets_gold[i, idx] = 1.0

    epochs = 25
    batch_size = 64
    lr = 1e-3

    oof_probs = np.zeros((n_orig, n_classes), dtype=np.float32)
    test_fold_preds = np.zeros((len(folds), len(X_test), n_classes), dtype=np.float32)

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

        model = DualHeadArcFaceNet(in_features=1944, hidden1=768, hidden2=384, n_classes=n_classes).to(device)
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
                tax_logits, bin_logits = model(bx, by_hard, bis_ps)
                loss = loss_fn(tax_logits, bin_logits, bis_ps, by_hard, by_soft, by_bin)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Validation inference: Pure 76-Class Softmax
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for (bx,) in val_loader:
                bx = bx.to(device)
                tax_logits, _ = model(bx, label=None)
                p_tax = F.softmax(tax_logits, dim=-1).cpu().numpy()
                val_preds_list.append(p_tax)
        fold_probs = np.vstack(val_preds_list)
        oof_probs[val_idx] = fold_probs

        # Test inference: Pure 76-Class Softmax
        te_preds_list = []
        with torch.no_grad():
            for (bx,) in te_loader:
                bx = bx.to(device)
                tax_logits, _ = model(bx, label=None)
                p_tax = F.softmax(tax_logits, dim=-1).cpu().numpy()
                te_preds_list.append(p_tax)
        test_fold_preds[fold_num - 1] = np.vstack(te_preds_list)

        fold_score = compute_map3(y_train_orig[val_idx], fold_probs, tax_classes)
        print(f"  Fold {fold_num}/{len(folds)} Val MAP@3: {fold_score:.5f}", flush=True)

    total_arc_map3 = compute_map3(y_train_orig, oof_probs, tax_classes)
    test_arc_probs = np.mean(test_fold_preds, axis=0)
    print(f"\nStandalone Expanded-3500 ArcFace (1944-d) OOF MAP@3: {total_arc_map3:.5f} ({time.time() - t0:.1f}s)", flush=True)

    np.save(ART_DIR / "oof_arcface_pseudo_3500.npy", oof_probs)
    np.save(ART_DIR / "test_arcface_pseudo_3500.npy", test_arc_probs)
    print("Saved pseudo 3,500 probability artifacts.", flush=True)

    # -----------------------------------------------------------------------
    # Champion Tri-Blend Integration (The Proven 95/5 Formula)
    # -----------------------------------------------------------------------
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")

    p_oof_blend = 0.95 * oof_m15 + 0.05 * oof_probs
    p_test_blend = 0.95 * test_m15 + 0.05 * test_arc_probs

    top3_train = np.argsort(-p_oof_blend.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter([tax_classes[idx] for row in top3_train for idx in row])
    true_counts = Counter(y_train_orig)

    gamma = 0.05
    mult = np.ones(n_classes, dtype=np.float64)
    for i, c in enumerate(tax_classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** gamma
    mult[none_idx] = 0.9840

    p_test_cal = p_test_blend * mult

    # Fine-tune NONE anchor to EXACTLY 2,452 (70.10%)
    best_ns = 1.0
    best_diff = 9999
    for ns in np.linspace(0.85, 1.15, 601):
        test_scaled = np.copy(p_test_cal)
        test_scaled[:, none_idx] *= ns
        cnt = np.sum(np.argmax(test_scaled, axis=1) == none_idx)
        if abs(cnt - 2452) < best_diff:
            best_diff = abs(cnt - 2452)
            best_ns = ns

    p_test_final = np.copy(p_test_cal)
    p_test_final[:, none_idx] *= best_ns
    final_none = np.sum(np.argmax(p_test_final, axis=1) == none_idx)
    print(f"\nFinal Calibrated Top-1 NONE: {final_none} / {len(test_df)} ({final_none / len(test_df):.2%}) with scale={best_ns:.4f}", flush=True)
    assert final_none == 2452, f"Target 2452 missed! Got {final_none}"

    top3_test = np.argsort(-p_test_final.astype(np.float64), axis=1)[:, :3]
    preds_new = [" ".join(tax_classes[idx] for idx in row) for row in top3_test]

    champ_54616 = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_m15_recalibrated_composite_triblend_448_submission.csv"))
    diffs = sum(1 for a, b in zip(preds_new, champ_54616["root_code"]) if a != b)
    print(f"Differences vs 0.54616 Champion: {diffs} / {len(test_df)} ({diffs / len(test_df):.2%})", flush=True)

    # Check Rank 1 differences specifically
    r1_diffs = sum(1 for a, b in zip(preds_new, champ_54616["root_code"]) if a.split()[0] != b.split()[0])
    print(f"Rank-1 Differences vs 0.54616 Champion: {r1_diffs} / {len(test_df)} ({r1_diffs / len(test_df):.2%})", flush=True)

    out_file = SUB_DIR / "blend_pseudo_3500_champion_submission.csv"
    sub_df = pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": preds_new})
    sub_df.to_csv(out_file, index=False)
    print(f"\nSaved submission: {out_file}", flush=True)

    dl_path = Path("C:/Users/hp/Downloads/blend_pseudo_3500_champion_submission.csv")
    shutil.copy(out_file, dl_path)
    print(f"Copied to Downloads: {dl_path}", flush=True)

    # Verification checks
    assert len(sub_df) == 3498, "Wrong rows"
    assert list(sub_df.columns) == ["trace_id", "root_code"], "Wrong columns"
    assert sub_df["root_code"].isnull().sum() == 0, "Null values"
    assert all(len(s.split()) == 3 for s in sub_df["root_code"]), "Invalid triplet length"
    assert all(len(set(s.split())) == 3 for s in sub_df["root_code"]), "Duplicate codes"
    assert sum(1 for s in sub_df["root_code"] if s.startswith("NONE")) == 2452, "NONE count drifted!"
    print("ALL VERIFICATION CHECKS PASSED FOR BLEND PSEUDO 3,500!", flush=True)

    # -----------------------------------------------------------------------
    # Smooth Quad-Blend (Target: 0.550+ Breakthrough)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Generating Smooth Quad-Blend (80% 0.54616 + 15% Transductive + 5% Arc3500)")
    print("=" * 60, flush=True)

    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    test_dual448 = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    test_comp448 = np.load(ART_DIR / "test_arcface_composite_448.npy")

    p_test_champ_base = 0.95 * test_m15 + 0.02 * test_dual448 + 0.03 * test_comp448
    p_test_54616 = p_test_champ_base * mult
    p_test_54616[:, none_idx] *= 0.9995

    q_map = {q: set(grp["root_code"].dropna()) for q, grp in train_df.groupby("question_id")}
    p_test_boosted = np.copy(p_test_54616)
    for i in range(len(test_df)):
        qid = test_df.iloc[i]["question_id"]
        if qid in q_map:
            allowed = q_map[qid]
            for ci, code in enumerate(tax_classes):
                if code != "NONE" and code in allowed:
                    p_test_boosted[i, ci] *= 1.20

    # Quad-Blend: 80% Peak Champion (0.54616) + 15% Transductive Boost (0.54599) + 5% Expanded-3500 ArcFace
    p_test_quad = 0.80 * p_test_54616 + 0.15 * p_test_boosted + 0.05 * p_test_cal

    # Fine-tune NONE anchor to EXACTLY 2,452
    best_ns_quad = 1.0
    best_diff_quad = 9999
    for ns in np.linspace(0.85, 1.15, 601):
        test_scaled = np.copy(p_test_quad)
        test_scaled[:, none_idx] *= ns
        cnt = np.sum(np.argmax(test_scaled, axis=1) == none_idx)
        if abs(cnt - 2452) < best_diff_quad:
            best_diff_quad = abs(cnt - 2452)
            best_ns_quad = ns

    p_test_quad_final = np.copy(p_test_quad)
    p_test_quad_final[:, none_idx] *= best_ns_quad
    quad_none = np.sum(np.argmax(p_test_quad_final, axis=1) == none_idx)
    print(f"Quad-Blend Top-1 NONE: {quad_none} / {len(test_df)} ({quad_none / len(test_df):.2%}) with scale={best_ns_quad:.4f}", flush=True)
    assert quad_none == 2452, f"Target 2452 missed! Got {quad_none}"

    top3_quad = np.argsort(-p_test_quad_final.astype(np.float64), axis=1)[:, :3]
    preds_quad = [" ".join(tax_classes[idx] for idx in row) for row in top3_quad]

    diffs_quad = sum(1 for a, b in zip(preds_quad, champ_54616["root_code"]) if a != b)
    r1_diffs_quad = sum(1 for a, b in zip(preds_quad, champ_54616["root_code"]) if a.split()[0] != b.split()[0])
    print(f"Quad-Blend Differences vs 0.54616 Champion: {diffs_quad} / {len(test_df)} ({diffs_quad / len(test_df):.2%})", flush=True)
    print(f"Quad-Blend Rank-1 Differences vs 0.54616 Champion: {r1_diffs_quad} / {len(test_df)} ({r1_diffs_quad / len(test_df):.2%})", flush=True)

    quad_file = SUB_DIR / "blend_smooth_quad_3500_champion_submission.csv"
    quad_df = pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": preds_quad})
    quad_df.to_csv(quad_file, index=False)
    print(f"\nSaved Quad-Blend submission: {quad_file}", flush=True)

    dl_quad = Path("C:/Users/hp/Downloads/blend_smooth_quad_3500_champion_submission.csv")
    shutil.copy(quad_file, dl_quad)
    print(f"Copied Quad-Blend to Downloads: {dl_quad}", flush=True)

    # Verification checks
    assert len(quad_df) == 3498, "Wrong rows"
    assert list(quad_df.columns) == ["trace_id", "root_code"], "Wrong columns"
    assert quad_df["root_code"].isnull().sum() == 0, "Null values"
    assert all(len(s.split()) == 3 for s in quad_df["root_code"]), "Invalid triplet length"
    assert all(len(set(s.split())) == 3 for s in quad_df["root_code"]), "Duplicate codes"
    assert sum(1 for s in quad_df["root_code"] if s.startswith("NONE")) == 2452, "NONE count drifted!"
    print("ALL VERIFICATION CHECKS PASSED FOR QUAD-BLEND 3,500!", flush=True)


if __name__ == "__main__":
    train_and_evaluate()
