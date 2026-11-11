"""Controlled, Gold-Only ArcFace Margin Sweep across Canonical 5 Folds.

Investigates whether increasing ArcFace margin m and scale s from the baseline
(m=0.20, s=30.0) up to (m=0.35, s=32.0) improves rare-label detection on handwriting
traces without inducing false promotions on NONE rows.

Evaluated on:
1. Standalone Multimodal ArcFace (2,328-d features):
   - ConvNeXt-448 (768-d)
   - Swin-T (768-d)
   - TrOCR (384-d)
   - Question Text (384-d)
   - Perceptual Strokes (24-d)
2. Rare-Label Recall@10, Recall@5, and Recall@3 on all 298 rare traces (<15 samples).
3. False Promotion Rate on True NONE rows (where root_code is NaN / NONE).
4. End-to-End Champion S550 Integration:
   - In-fold substitution into S550's clean multimodal branch.
   - Evaluates Seen (1,335 traces), Unseen (165 traces), and Composite MAP@3 (0.957*Seen + 0.043*Unseen).
   - Strict promotion gate vs S550 reference (0.54214).
"""

from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRATCH_DIR = PROJECT_ROOT / "scratch"
if str(SCRATCH_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_DIR))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.domain_mask import DomainConstraintEngine
from src.metrics.map3 import calculate_map3
from audit_rare_class_s550_oof_error_breakdown import S550AuditEngine, sharp


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GoldDataset(Dataset):
    def __init__(self, features: np.ndarray, hard_labels: np.ndarray, binary_targets: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.hard_labels = torch.from_numpy(hard_labels).long()
        self.binary_targets = torch.from_numpy(binary_targets).float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[index], self.hard_labels[index], self.binary_targets[index]


class ArcFaceClassifier(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale: float = 30.0, margin: float = 0.20):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1))
        if labels is None:
            return cosine * self.scale
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.scale


class CleanDualHeadArcFaceNet(nn.Module):
    def __init__(
        self,
        in_dim: int = 2328,
        num_classes: int = 76,
        embed_dim: int = 384,
        scale: float = 30.0,
        margin: float = 0.20
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(768, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.arcface_head = ArcFaceClassifier(embed_dim, num_classes, scale=scale, margin=margin)
        self.binary_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.backbone(x)
        tax_logits = self.arcface_head(emb, labels)
        bin_logits = self.binary_head(emb).squeeze(-1)
        return tax_logits, bin_logits


class CompositeRankingLoss(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: float = 2.0,
        margin: float = 1.0,
        top_k: int = 5,
        lambda_rank: float = 0.20
    ):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.margin = margin
        self.top_k = top_k
        self.lambda_rank = lambda_rank

    def forward(
        self,
        tax_logits: torch.Tensor,
        bin_logits: torch.Tensor,
        hard_labels: torch.Tensor,
        bin_targets: torch.Tensor
    ) -> torch.Tensor:
        batch_sz = tax_logits.shape[0]
        # 1. Focal Loss
        ce = F.cross_entropy(tax_logits, hard_labels, reduction="none")
        pt = torch.exp(-ce)
        at = self.alpha[hard_labels]  # type: ignore[index]
        focal_loss = (at * ((1.0 - pt) ** self.gamma) * ce).mean()

        # 2. Soft-Ranking Proxy Margin Loss
        target_scores = tax_logits[torch.arange(batch_sz, device=tax_logits.device), hard_labels].unsqueeze(1)
        mask = torch.ones_like(tax_logits, dtype=torch.bool)
        mask[torch.arange(batch_sz, device=tax_logits.device), hard_labels] = False
        neg_scores = tax_logits[mask].view(batch_sz, -1)
        top_negatives, _ = torch.topk(neg_scores, min(self.top_k, neg_scores.shape[1]), dim=-1)
        rank_loss = F.relu(self.margin - (target_scores - top_negatives)).mean()

        # 3. Binary Head Loss
        bin_loss = F.binary_cross_entropy_with_logits(bin_logits, bin_targets)

        return focal_loss + self.lambda_rank * rank_loss + 0.30 * bin_loss


def train_single_fold(
    X_train_orig: np.ndarray,
    y_true_int: np.ndarray,
    y_bin: np.ndarray,
    trn_idx: np.ndarray,
    val_idx: np.ndarray,
    classes: List[str],
    train_df: pd.DataFrame,
    tax: TaxonomyManager,
    margin: float,
    scale: float,
    device: torch.device,
    fold_idx: int,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3
) -> np.ndarray:
    set_seed(42 + fold_idx)
    n_classes = len(classes)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_orig[trn_idx])
    X_val = scaler.transform(X_train_orig[val_idx])

    counts = np.zeros(n_classes, dtype=np.int64)
    for v in y_true_int[trn_idx]:
        counts[v] += 1
    raw_a = len(trn_idx) / (n_classes * np.maximum(counts, 1.0))
    alpha = np.clip(raw_a, 0.20, 5.0).astype(np.float32)
    alpha_t = torch.from_numpy(alpha / alpha.mean()).float().to(device)

    train_ds = GoldDataset(X_tr, y_true_int[trn_idx], y_bin[trn_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = CleanDualHeadArcFaceNet(
        in_dim=2328,
        num_classes=n_classes,
        embed_dim=384,
        scale=scale,
        margin=margin
    ).to(device)

    criterion = CompositeRankingLoss(alpha=alpha_t, margin=1.0, top_k=5, lambda_rank=0.20)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y, batch_bin in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_bin = batch_bin.to(device)

            optimizer.zero_grad()
            tax_logits, bin_logits = model(batch_x, batch_y)
            loss = criterion(tax_logits, bin_logits, batch_y, batch_bin)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        v_x = torch.from_numpy(X_val).float().to(device)
        v_logits, _ = model(v_x, labels=None)
        val_p = F.softmax(v_logits, dim=-1).cpu().numpy()

    masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df.iloc[trn_idx])
    v_topics = [str(t) for t in train_df.iloc[val_idx]["topic_name"].values]
    v_grades = [int(float(g)) for g in train_df.iloc[val_idx]["grade"].values]
    val_calibrated = masker.apply_mask_batch(val_p, classes, v_topics, v_grades, damping=0.05)

    return val_calibrated


def main():
    print("=" * 115)
    print("CONTROLLED GOLD-ONLY ARCFACE MARGIN SWEEP ACROSS CANONICAL 5 FOLDS")
    print("=" * 115)
    start_total = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_true_str = train_df["root_code"].fillna("NONE").astype(str).tolist()
    y_true_int = np.array([class_to_idx.get(r, none_idx) for r in y_true_str], dtype=np.int64)
    y_bin = np.array([1.0 if y == "NONE" else 0.0 for y in y_true_str], dtype=np.float32)
    n_samples = len(train_df)

    # Class frequencies & Rare Classes (<15)
    class_counts = Counter(y_true_str)
    rare_classes = {c for c, cnt in class_counts.items() if c != "NONE" and cnt < 15}
    rare_indices = [i for i in range(n_samples) if y_true_str[i] in rare_classes]
    none_indices = [i for i in range(n_samples) if y_true_str[i] == "NONE"]

    print(f"Total Traces: {n_samples} | Rare Traces (<15 samples): {len(rare_indices)} ({len(rare_classes)} classes)")
    print(f"Total True NONE Traces: {len(none_indices)}")

    # Load 2,328-d features
    swin_tr = np.load(ART_DIR / "train_swin_features.npy")
    conv448_tr = np.load(ART_DIR / "train_convnext_448_letterbox.npy")
    text_tr = np.load(ART_DIR / "train_text_embeddings_384.npy")
    trocr_tr = np.load(ART_DIR / "train_trocr_features.npy")
    strk_tr = np.load(ART_DIR / "train_perceptual_strokes.npy")

    X_train_orig = np.hstack([swin_tr, conv448_tr, text_tr, trocr_tr, strk_tr]).astype(np.float32)
    print(f"Concatenated 2,328-d Feature Matrix Shape: {X_train_orig.shape}")

    # Canonical splits
    with open(DATA_DIR / "splits.json") as f:
        splits = json.load(f)["folds"]

    # Seen question mask
    seen_mask = np.zeros(n_samples, dtype=bool)
    for fold in splits:
        tr_qids = set(train_df.iloc[fold["train_indices"]]["question_id"].astype(str))
        for i in fold["val_indices"]:
            if str(train_df.iloc[i]["question_id"]) in tr_qids:
                seen_mask[i] = True

    n_seen = int(np.sum(seen_mask))
    n_unseen = n_samples - n_seen
    print(f"Canonical Split Balance: {n_seen} Seen Traces, {n_unseen} Unseen Traces")

    # Initialize S550 Audit Engine for baseline reference and in-fold substitution
    engine = S550AuditEngine()

    # Regenerate canonical S550 baseline reference
    print("\nRegenerating Canonical S550 Baseline Reference...")
    oof_p_s550_ref = np.zeros((n_samples, n_classes), dtype=np.float64)
    q_dists_per_fold: Dict[int, Dict[str, Counter]] = {}
    s550_ref_fold_scores: List[float] = []

    for fold_idx, fold in enumerate(splits, start=1):
        tr_idx = np.array(fold["train_indices"], dtype=np.int64)
        val_idx = np.array(fold["val_indices"], dtype=np.int64)

        p_s550_val, q_dist = engine.generate_s550_probs(tr_idx, val_idx)
        oof_p_s550_ref[val_idx] = p_s550_val
        q_dists_per_fold[fold_idx] = q_dist

        y_val_gold = [y_true_str[i] for i in val_idx]
        preds_s550_val = engine.rank_and_gate(p_s550_val, val_idx, q_dist)
        fold_sc = calculate_map3(y_val_gold, preds_s550_val)
        s550_ref_fold_scores.append(fold_sc)

    preds_s550_ref_all = [[] for _ in range(n_samples)]
    for fold_idx, fold in enumerate(splits, start=1):
        val_idx = np.array(fold["val_indices"], dtype=np.int64)
        preds_val = engine.rank_and_gate(oof_p_s550_ref[val_idx], val_idx, q_dists_per_fold[fold_idx])
        for i, idx in enumerate(val_idx):
            preds_s550_ref_all[idx] = preds_val[i]

    s550_seen_map3 = calculate_map3(
        [y_true_str[i] for i in range(n_samples) if seen_mask[i]],
        [preds_s550_ref_all[i] for i in range(n_samples) if seen_mask[i]]
    )
    s550_unseen_map3 = calculate_map3(
        [y_true_str[i] for i in range(n_samples) if not seen_mask[i]],
        [preds_s550_ref_all[i] for i in range(n_samples) if not seen_mask[i]]
    )
    s550_comp_map3 = 0.957 * s550_seen_map3 + 0.043 * s550_unseen_map3

    # Reference S550 Rare Recall & False NONE Promotions
    s550_rare_r10_count = sum(1 for i in rare_indices if y_true_str[i] in [classes[c] for c in np.argsort(-oof_p_s550_ref[i])[:10]])
    s550_rare_r3_count = sum(1 for i in rare_indices if y_true_str[i] in preds_s550_ref_all[i])
    s550_false_none_promotions = sum(1 for i in none_indices if preds_s550_ref_all[i][0] != "NONE")

    print(f"S550 Reference:")
    print(f"  Seen MAP@3: {s550_seen_map3:.5f} | Unseen MAP@3: {s550_unseen_map3:.5f} | Composite MAP@3: {s550_comp_map3:.5f}")
    print(f"  Folds: {[round(s, 5) for s in s550_ref_fold_scores]} (Mean: {np.mean(s550_ref_fold_scores):.5f})")
    print(f"  Rare Recall@10 (Prob Matrix): {s550_rare_r10_count} / {len(rare_indices)} ({s550_rare_r10_count / len(rare_indices) * 100:.2f}%)")
    print(f"  Rare Top-3 Hit Rate:          {s550_rare_r3_count} / {len(rare_indices)} ({s550_rare_r3_count / len(rare_indices) * 100:.2f}%)")
    print(f"  False NONE Promotions:        {s550_false_none_promotions} / {len(none_indices)} ({s550_false_none_promotions / len(none_indices) * 100:.2f}%)")

    # Define Parameter Grid
    sweep_configs = [
        {"name": "ArcFace_m020_s30 (Baseline)", "margin": 0.20, "scale": 30.0},
        {"name": "ArcFace_m025_s30",            "margin": 0.25, "scale": 30.0},
        {"name": "ArcFace_m025_s32",            "margin": 0.25, "scale": 32.0},
        {"name": "ArcFace_m030_s30",            "margin": 0.30, "scale": 30.0},
        {"name": "ArcFace_m030_s32",            "margin": 0.30, "scale": 32.0},
        {"name": "ArcFace_m035_s30",            "margin": 0.35, "scale": 30.0},
        {"name": "ArcFace_m035_s32",            "margin": 0.35, "scale": 32.0},
    ]

    sweep_results: List[Dict[str, Any]] = []

    for cfg_idx, cfg in enumerate(sweep_configs, start=1):
        m = cfg["margin"]
        s = cfg["scale"]
        name = cfg["name"]
        print("\n" + "=" * 95)
        print(f"CONFIG {cfg_idx}/{len(sweep_configs)}: {name} (margin={m:.2f}, scale={s:.1f})")
        print("=" * 95)
        t_cfg_start = time.time()

        oof_probs_cand = np.zeros((n_samples, n_classes), dtype=np.float32)
        cand_fold_map3: List[float] = []

        # 5 Canonical Folds
        for fold_idx, fold in enumerate(splits, start=1):
            tr_idx = np.array(fold["train_indices"], dtype=np.int64)
            val_idx = np.array(fold["val_indices"], dtype=np.int64)

            val_calibrated = train_single_fold(
                X_train_orig=X_train_orig,
                y_true_int=y_true_int,
                y_bin=y_bin,
                trn_idx=tr_idx,
                val_idx=val_idx,
                classes=classes,
                train_df=train_df,
                tax=tax,
                margin=m,
                scale=s,
                device=device,
                fold_idx=fold_idx,
                epochs=25,
                batch_size=64,
                lr=1e-3
            )
            oof_probs_cand[val_idx] = val_calibrated

            # Standalone fold score
            y_val_gold = [y_true_str[i] for i in val_idx]
            top3_val = [[classes[c] for c in np.argsort(-val_calibrated[r])[:3]] for r in range(len(val_idx))]
            f_sc = calculate_map3(y_val_gold, top3_val)
            cand_fold_map3.append(f_sc)
            print(f"  Fold {fold_idx}/5 Standalone MAP@3: {f_sc:.5f}", flush=True)

        # Standalone OOF Metrics
        cand_top10 = [np.argsort(-oof_probs_cand[i])[:10] for i in range(n_samples)]
        cand_top3 = [[classes[c] for c in row[:3]] for row in cand_top10]

        standalone_seen_map3 = calculate_map3(
            [y_true_str[i] for i in range(n_samples) if seen_mask[i]],
            [cand_top3[i] for i in range(n_samples) if seen_mask[i]]
        )
        standalone_unseen_map3 = calculate_map3(
            [y_true_str[i] for i in range(n_samples) if not seen_mask[i]],
            [cand_top3[i] for i in range(n_samples) if not seen_mask[i]]
        )
        standalone_comp_map3 = 0.957 * standalone_seen_map3 + 0.043 * standalone_unseen_map3

        # Rare Recall Metrics on Standalone ArcFace
        cand_rare_r10_count = sum(1 for i in rare_indices if y_true_int[i] in cand_top10[i])
        cand_rare_r5_count = sum(1 for i in rare_indices if y_true_int[i] in cand_top10[i][:5])
        cand_rare_r3_count = sum(1 for i in rare_indices if y_true_int[i] in cand_top10[i][:3])

        cand_rare_r10 = cand_rare_r10_count / len(rare_indices)
        cand_rare_r5 = cand_rare_r5_count / len(rare_indices)
        cand_rare_r3 = cand_rare_r3_count / len(rare_indices)

        # Standalone false promotions on true NONE
        cand_false_none_promotions = sum(1 for i in none_indices if cand_top10[i][0] != none_idx)
        cand_false_none_rate = cand_false_none_promotions / len(none_indices)

        # In-Fold S550 Integration
        oof_p_s550_with_cand = np.zeros((n_samples, n_classes), dtype=np.float64)
        s550_cand_fold_scores: List[float] = []

        for fold_idx, fold in enumerate(splits, start=1):
            tr_idx = np.array(fold["train_indices"], dtype=np.int64)
            val_idx = np.array(fold["val_indices"], dtype=np.int64)
            q_dist = q_dists_per_fold[fold_idx]

            # In-fold clean multimodal blend with candidate ArcFace
            trn_df_fold = train_df.iloc[tr_idx]
            val_df_fold = train_df.iloc[val_idx]
            from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
            prior_eng = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df_fold)
            val_prior = prior_eng.predict_proba(val_df_fold).astype(np.float64)

            p_clean_val = (
                0.6448 * val_prior +
                0.0839 * sharp(engine.honest_text[val_idx], 0.6) +
                0.0401 * sharp(engine.oof_sem_text[val_idx], 0.6) +
                0.0458 * engine.oof_convnext[val_idx] +
                0.0454 * engine.oof_swin[val_idx] +
                0.0483 * engine.oof_finetuned[val_idx] +
                0.0455 * engine.oof_trocr[val_idx] +
                0.0462 * sharp(oof_probs_cand[val_idx], 0.8)
            )

            # Reconstitute S550 with p_clean_val
            p_base_v1_trn = 0.95 * engine.oof_m15[tr_idx] + 0.02 * engine.oof_dual[tr_idx] + 0.03 * engine.oof_comp[tr_idx]
            top3_trn = np.argsort(-p_base_v1_trn.astype(np.float64), axis=1)[:, :3]
            pred_counts = Counter(classes[idx] for row in top3_trn for idx in row)
            true_counts = Counter(trn_df_fold["root_code"].fillna("NONE").values)
            mult = np.ones(n_classes, dtype=np.float64)
            for i, c in enumerate(classes):
                if c != "NONE":
                    mult[i] = (max(true_counts.get(c, 1), 1) / max(pred_counts.get(c, 1), 1)) ** 0.05
            mult[none_idx] = 0.9840

            p_base_v1_val = (0.95 * engine.oof_m15[val_idx] + 0.02 * engine.oof_dual[val_idx] + 0.03 * engine.oof_comp[val_idx]) * mult
            p_base_v1_val[:, none_idx] *= 0.9995

            p_boosted_v1 = np.copy(p_base_v1_val)
            for i in range(len(val_idx)):
                qid = str(val_df_fold.iloc[i]["question_id"])
                if qid in q_dist:
                    allowed = {c for c, cnt in q_dist[qid].items() if c != "NONE"}
                    for ci, c in enumerate(classes):
                        if c in allowed:
                            p_boosted_v1[i, ci] *= 1.20

            p_arc_v1 = (0.95 * engine.oof_m15[val_idx] + 0.05 * engine.oof_v1[val_idx]) * mult
            p_champ_v1 = 0.70 * p_base_v1_val + 0.25 * p_boosted_v1 + 0.05 * p_arc_v1
            p_s550_cand = 0.96 * p_champ_v1 + 0.04 * p_clean_val

            p_s550_cand = np.maximum(p_s550_cand, 0.0)
            p_s550_cand = p_s550_cand / p_s550_cand.sum(axis=1, keepdims=True)
            oof_p_s550_with_cand[val_idx] = p_s550_cand

            preds_val = engine.rank_and_gate(p_s550_cand, val_idx, q_dist)
            f_sc_s550 = calculate_map3([y_true_str[i] for i in val_idx], preds_val)
            s550_cand_fold_scores.append(f_sc_s550)

        # Full S550 Gated Predictions
        preds_s550_cand_all = [[] for _ in range(n_samples)]
        for fold_idx, fold in enumerate(splits, start=1):
            val_idx = np.array(fold["val_indices"], dtype=np.int64)
            preds_val = engine.rank_and_gate(oof_p_s550_with_cand[val_idx], val_idx, q_dists_per_fold[fold_idx])
            for i, idx in enumerate(val_idx):
                preds_s550_cand_all[idx] = preds_val[i]

        s550_cand_seen = calculate_map3(
            [y_true_str[i] for i in range(n_samples) if seen_mask[i]],
            [preds_s550_cand_all[i] for i in range(n_samples) if seen_mask[i]]
        )
        s550_cand_unseen = calculate_map3(
            [y_true_str[i] for i in range(n_samples) if not seen_mask[i]],
            [preds_s550_cand_all[i] for i in range(n_samples) if not seen_mask[i]]
        )
        s550_cand_comp = 0.957 * s550_cand_seen + 0.043 * s550_cand_unseen
        delta_vs_s550 = s550_cand_comp - s550_comp_map3

        # S550 Integrated Rare Recall & False NONE Promotions
        s550_cand_rare_r10_count = sum(1 for i in rare_indices if y_true_str[i] in [classes[c] for c in np.argsort(-oof_p_s550_with_cand[i])[:10]])
        s550_cand_rare_r3_count = sum(1 for i in rare_indices if y_true_str[i] in preds_s550_cand_all[i])
        s550_cand_false_none = sum(1 for i in none_indices if preds_s550_cand_all[i][0] != "NONE")

        cfg_elapsed = time.time() - t_cfg_start
        print(f"\nResults for {name} ({cfg_elapsed:.1f}s):")
        print(f"  Standalone ArcFace: Seen={standalone_seen_map3:.5f} | Unseen={standalone_unseen_map3:.5f} | Comp={standalone_comp_map3:.5f}")
        print(f"  Standalone Rare Recall: R@10={cand_rare_r10*100:.1f}%, R@5={cand_rare_r5*100:.1f}%, R@3={cand_rare_r3*100:.1f}%")
        print(f"  Standalone False NONE Promotions: {cand_false_none_promotions} / {len(none_indices)} ({cand_false_none_rate*100:.2f}%)")
        print(f"  S550 Integrated:    Seen={s550_cand_seen:.5f} | Unseen={s550_cand_unseen:.5f} | Comp={s550_cand_comp:.5f} (Delta: {delta_vs_s550:+.5f})")
        print(f"  S550 Fold Scores:   {[round(s, 5) for s in s550_cand_fold_scores]}")
        print(f"  S550 Rare R@10:     {s550_cand_rare_r10_count} (Delta vs S550: {s550_cand_rare_r10_count - s550_rare_r10_count:+d})")
        print(f"  S550 False NONE:    {s550_cand_false_none} (Delta vs S550: {s550_cand_false_none - s550_false_none_promotions:+d})")

        sweep_results.append({
            "name": name,
            "margin": m,
            "scale": s,
            "standalone_map3": {
                "seen": round(standalone_seen_map3, 5),
                "unseen": round(standalone_unseen_map3, 5),
                "composite": round(standalone_comp_map3, 5),
                "folds": [round(s, 5) for s in cand_fold_map3],
                "fold_mean": round(float(np.mean(cand_fold_map3)), 5),
                "fold_std": round(float(np.std(cand_fold_map3)), 5)
            },
            "standalone_rare_recall": {
                "recall_10": round(cand_rare_r10, 4),
                "recall_5": round(cand_rare_r5, 4),
                "recall_3": round(cand_rare_r3, 4),
                "hits_10": cand_rare_r10_count,
                "hits_5": cand_rare_r5_count,
                "hits_3": cand_rare_r3_count,
                "total_rare": len(rare_indices)
            },
            "standalone_false_none": {
                "count": cand_false_none_promotions,
                "rate": round(cand_false_none_rate, 4),
                "total_none": len(none_indices)
            },
            "s550_integrated_map3": {
                "seen": round(s550_cand_seen, 5),
                "unseen": round(s550_cand_unseen, 5),
                "composite": round(s550_cand_comp, 5),
                "delta_vs_s550": round(delta_vs_s550, 5),
                "folds": [round(s, 5) for s in s550_cand_fold_scores],
                "fold_mean": round(float(np.mean(s550_cand_fold_scores)), 5),
                "fold_std": round(float(np.std(s550_cand_fold_scores)), 5)
            },
            "s550_integrated_rare_recall": {
                "hits_10": s550_cand_rare_r10_count,
                "hits_3": s550_cand_rare_r3_count,
                "delta_hits_10": s550_cand_rare_r10_count - s550_rare_r10_count,
                "delta_hits_3": s550_cand_rare_r3_count - s550_rare_r3_count
            },
            "s550_integrated_false_none": {
                "count": s550_cand_false_none,
                "delta_count": s550_cand_false_none - s550_false_none_promotions
            }
        })

    # Summary Comparison Table
    print("\n" + "=" * 125)
    print("CONTROLLED ARCFACE MARGIN SWEEP: COMPARATIVE AUDIT SUMMARY")
    print("=" * 125)
    print(
        f"{'Configuration':32s} | {'Standalone Comp':15s} | {'Standalone R@10':15s} | {'Standalone False NONE':21s} | "
        f"{'S550 Comp':9s} | {'Delta S550':10s} | {'S550 R@10':9s}"
    )
    print("-" * 125)

    for r in sweep_results:
        st_c = r["standalone_map3"]["composite"]
        st_r10 = r["standalone_rare_recall"]["recall_10"] * 100
        st_fn = r["standalone_false_none"]["count"]
        st_fn_r = r["standalone_false_none"]["rate"] * 100
        s550_c = r["s550_integrated_map3"]["composite"]
        delta_s550 = r["s550_integrated_map3"]["delta_vs_s550"]
        s550_r10 = r["s550_integrated_rare_recall"]["hits_10"]
        print(
            f"{r['name']:32s} | {st_c:15.5f} | {st_r10:14.1f}% | {st_fn:4d} ({st_fn_r:5.1f}%)         | "
            f"{s550_c:9.5f} | {delta_s550:+10.5f} | {s550_r10:4d} / 298"
        )
    print("=" * 125)

    # Decision Gate Verification
    best_candidate = max(sweep_results, key=lambda x: x["s550_integrated_map3"]["composite"])
    best_delta = float(best_candidate["s550_integrated_map3"]["delta_vs_s550"])
    best_name = best_candidate["name"]

    print("\n" + "=" * 95)
    print("DECISION GATE & PROMOTION VERIFICATION:")
    print("=" * 95)
    print(f"  Reference S550 Composite MAP@3: {s550_comp_map3:.5f}")
    print(f"  Best Sweep Candidate:           {best_name} ({best_candidate['s550_integrated_map3']['composite']:.5f})")
    print(f"  Composite Delta:                {best_delta:+.5f}")

    gate_cleared = (
        best_delta > 0.00010 and
        best_candidate["s550_integrated_false_none"]["delta_count"] <= 0 and
        best_candidate["s550_integrated_rare_recall"]["delta_hits_10"] >= 0
    )

    if gate_cleared:
        print(f"  GATE STATUS: [PASSED] -> Candidate {best_name} clears held-out promotion gate!")
    else:
        print(f"  GATE STATUS: [FAILED] -> No margin candidate cleared promotion criteria.")
        print(f"  Reason: Delta ({best_delta:+.5f}) fails to show consistent held-out gain over S550 reference.")
        print(f"  ACTION: S550 remains active champion (0.54214 CV / 0.55049 Public LB). Do not create submission.")
    print("=" * 95)

    # Save comprehensive report
    report_dict = {
        "execution_time_seconds": round(time.time() - start_total, 2),
        "s550_reference": {
            "seen_map3": round(s550_seen_map3, 5),
            "unseen_map3": round(s550_unseen_map3, 5),
            "composite_map3": round(s550_comp_map3, 5),
            "folds": [round(s, 5) for s in s550_ref_fold_scores],
            "rare_r10_hits": s550_rare_r10_count,
            "rare_r3_hits": s550_rare_r3_count,
            "false_none_promotions": s550_false_none_promotions
        },
        "sweep_candidates": sweep_results,
        "best_candidate": best_candidate["name"],
        "gate_cleared": gate_cleared
    }

    out_path = ART_DIR / "arcface_margin_sweep_report.json"
    with open(out_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[Saved Audit Report] {out_path.name}")


if __name__ == "__main__":
    main()

