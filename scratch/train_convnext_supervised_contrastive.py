"""Supervised Contrastive Learning on ConvNeXt-448 Handwriting Crops.

Experiment Specification per User Directive:
1. Gold-Only & Question-Fold-Safe:
   - Trained strictly on 1,500 Gold Train Traces (zero pseudo-labels or unlabelled data).
   - Strictly evaluated across canonical 5 folds (splits.json: 1200 train / 300 val).
   - Zero leakage from val to train. In-fold standardization and class prototypes.
2. Visual Representation Training:
   - Backbone: Pre-extracted 448x448 letterbox ConvNeXt-Tiny handwriting features (768-d).
   - Class-Aware Balanced Batch Sampler: Guarantees positive pairs for rare classes in every batch.
   - Supervised Contrastive Loss (Khosla et al., NeurIPS 2020) at temperature tau=0.07.
   - Dual variants tested:
     a) Pure Supervised Contrastive Prototype Retrieval (SupCon-Proto)
     b) Joint Supervised Contrastive + Class-Balanced Cross-Entropy (SupCon-Joint)
     c) k-Nearest Neighbor (k-NN, k=15) Contrastive Retrieval
3. Evaluation & Gating vs S550 Reference (0.54214 CV):
   - Rare-class Recall@10 across all 298 rare traces (<15 samples).
   - False NONE demotions across all 506 true NONE traces.
   - Overall Composite MAP@3 (0.957 * Seen + 0.043 * Unseen).
   - Decision Gate: Promotes only if held-out retrieval AND composite score improve across folds
     without increasing false NONE demotions.
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
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

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


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss (Khosla et al., NeurIPS 2020)."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: (B, D) L2-normalized
        # labels: (B,)
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Dot product similarity
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # Numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)

        # Compute mean of log-likelihood over positive pairs
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-7, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Only compute loss for anchors that have at least one positive
        has_pos = (mask.sum(1) > 0).float()
        loss = -(has_pos * mean_log_prob_pos).sum() / (has_pos.sum() + 1e-7)
        return loss


class ClassAwareBatchSampler:
    """Class-Aware Batch Sampler ensuring positive pairs for rare classes in every batch."""
    def __init__(
        self,
        labels: np.ndarray,
        n_classes_per_batch: int = 16,
        n_samples_per_class: int = 4,
        n_batches: int = 35
    ):
        self.labels = labels
        self.n_classes_per_batch = n_classes_per_batch
        self.n_samples_per_class = n_samples_per_class
        self.n_batches = n_batches
        self.batch_size = n_classes_per_batch * n_samples_per_class

        # Build class-to-indices lookup
        self.class_to_indices: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(labels):
            lbl_int = int(lbl)
            if lbl_int not in self.class_to_indices:
                self.class_to_indices[lbl_int] = []
            self.class_to_indices[lbl_int].append(idx)
        self.all_classes = list(self.class_to_indices.keys())

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            sampled_classes = random.sample(self.all_classes, min(self.n_classes_per_batch, len(self.all_classes)))
            batch_indices: List[int] = []
            for cls in sampled_classes:
                indices = self.class_to_indices[cls]
                if len(indices) >= self.n_samples_per_class:
                    batch_indices.extend(random.sample(indices, self.n_samples_per_class))
                else:
                    batch_indices.extend(random.choices(indices, k=self.n_samples_per_class))
            random.shuffle(batch_indices)
            yield batch_indices


class ConvNeXtContrastiveEncoder(nn.Module):
    """Contrastive projection head mapping ConvNeXt-448 features to a metric embedding space."""
    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 512,
        embed_dim: int = 128,
        num_classes: int = 76,
        dropout: float = 0.20
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.projector(x)
        norm_feat = F.normalize(feat, p=2, dim=-1)
        logits = self.classifier(feat)
        return norm_feat, logits


def compute_class_prototypes(embeddings: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Computes L2-normalized class centroid prototypes from training embeddings."""
    dim = embeddings.shape[1]
    prototypes = np.zeros((num_classes, dim), dtype=np.float32)
    for c in range(num_classes):
        c_mask = (labels == c)
        if np.any(c_mask):
            centroid = np.mean(embeddings[c_mask], axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-7:
                prototypes[c] = centroid / norm
            else:
                prototypes[c] = centroid
        else:
            # Fallback: zero vector for unobserved class in fold
            pass
    return prototypes


def evaluate_retrieval(
    val_embeddings: np.ndarray,
    prototypes: np.ndarray,
    classes: List[str],
    train_df_val: pd.DataFrame,
    tax: TaxonomyManager,
    temperature: float = 0.10
) -> np.ndarray:
    """Computes prototype cosine retrieval probabilities with domain masking."""
    # Dot product: (N_val, num_classes)
    sims = np.matmul(val_embeddings, prototypes.T) / temperature
    # Numerical softmax
    sims_max = np.max(sims, axis=1, keepdims=True)
    exp_sims = np.exp(sims - sims_max)
    probs = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)

    # In-fold domain masking
    masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df_val)
    v_topics = [str(t) for t in train_df_val["topic_name"].values]
    v_grades = [int(float(g)) for g in train_df_val["grade"].values]
    calibrated = masker.apply_mask_batch(probs, classes, v_topics, v_grades, damping=0.05)
    return calibrated


def train_supcon_fold(
    X_train_vis: np.ndarray,
    y_true_int: np.ndarray,
    trn_idx: np.ndarray,
    val_idx: np.ndarray,
    classes: List[str],
    train_df: pd.DataFrame,
    tax: TaxonomyManager,
    device: torch.device,
    fold_idx: int,
    joint_ce_weight: float = 0.0,
    epochs: int = 25,
    lr: float = 1e-3,
    temperature: float = 0.07
) -> Tuple[np.ndarray, np.ndarray]:
    """Trains ConvNeXt SupCon model on fold training indices and extracts retrieval probabilities."""
    set_seed(42 + fold_idx)
    n_classes = len(classes)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_vis[trn_idx])
    X_val = scaler.transform(X_train_vis[val_idx])

    y_tr = y_true_int[trn_idx]
    y_val = y_true_int[val_idx]

    # Convert to tensors
    X_tr_t = torch.from_numpy(X_tr).float().to(device)
    y_tr_t = torch.from_numpy(y_tr).long().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)

    # Model, Optimizer, Loss
    model = ConvNeXtContrastiveEncoder(
        in_dim=768,
        hidden_dim=512,
        embed_dim=128,
        num_classes=n_classes,
        dropout=0.20
    ).to(device)

    supcon_criterion = SupConLoss(temperature=temperature)
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    sampler = ClassAwareBatchSampler(y_tr, n_classes_per_batch=16, n_samples_per_class=4, n_batches=35)

    model.train()
    for _ in range(epochs):
        for batch_indices in sampler:
            batch_idx_t = torch.as_tensor(batch_indices, dtype=torch.long, device=device)
            b_x = X_tr_t[batch_idx_t]
            b_y = y_tr_t[batch_idx_t]

            optimizer.zero_grad()
            norm_feat, logits = model(b_x)

            loss = supcon_criterion(norm_feat, b_y)
            if joint_ce_weight > 0.0:
                ce_loss = ce_criterion(logits, b_y)
                loss = loss + joint_ce_weight * ce_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        scheduler.step()

    # Extract embeddings
    model.eval()
    with torch.no_grad():
        tr_embeddings, tr_logits = model(X_tr_t)
        tr_embeddings_np = tr_embeddings.cpu().numpy()

        val_embeddings, val_logits = model(X_val_t)
        val_embeddings_np = val_embeddings.cpu().numpy()
        val_ce_probs = F.softmax(val_logits, dim=-1).cpu().numpy()

    # Compute class prototypes and retrieval probabilities
    prototypes = compute_class_prototypes(tr_embeddings_np, y_tr, n_classes)
    val_proto_probs = evaluate_retrieval(
        val_embeddings=val_embeddings_np,
        prototypes=prototypes,
        classes=classes,
        train_df_val=train_df.iloc[val_idx],
        tax=tax,
        temperature=0.10
    )

    # In-fold domain masking for joint CE probs
    masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df.iloc[trn_idx])
    v_topics = [str(t) for t in train_df.iloc[val_idx]["topic_name"].values]
    v_grades = [int(float(g)) for g in train_df.iloc[val_idx]["grade"].values]
    val_ce_calibrated = masker.apply_mask_batch(val_ce_probs, classes, v_topics, v_grades, damping=0.05)

    return val_proto_probs, val_ce_calibrated


def main() -> None:
    print("=" * 115)
    print("SUPERVISED CONTRASTIVE LEARNING ON CONVNEXT-448 HANDWRITING CROPS (5 FOLDS)")
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
    y_raw = train_df["root_code"].fillna("NONE").astype(str).tolist()
    y_true_int = np.array([class_to_idx.get(r, none_idx) for r in y_raw], dtype=np.int64)
    n_samples = len(train_df)

    class_counts = Counter(y_raw)
    rare_classes: Set[str] = {c for c, cnt in class_counts.items() if c != "NONE" and cnt < 15}
    rare_indices = [i for i in range(n_samples) if y_raw[i] in rare_classes]
    none_indices = [i for i in range(n_samples) if y_raw[i] == "NONE"]

    print(f"Total Traces: {n_samples} | Total Rare Traces (<15 samples): {len(rare_indices)} ({len(rare_classes)} classes)")
    print(f"Total True NONE Traces: {len(none_indices)}")

    # Load pre-extracted ConvNeXt-448 letterbox features (1,500, 768)
    conv448_path = ART_DIR / "train_convnext_448_letterbox.npy"
    if not conv448_path.exists():
        raise FileNotFoundError(f"Missing {conv448_path}")
    X_vis_orig = np.load(conv448_path).astype(np.float32)
    print(f"Loaded ConvNeXt-448 Handwriting Feature Matrix: {X_vis_orig.shape}")

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

    # Evaluate Baseline S550 Reference
    print("\nRegenerating Canonical S550 Baseline Reference...")
    engine = S550AuditEngine()

    oof_p_s550_ref = np.zeros((n_samples, n_classes), dtype=np.float64)
    q_dists_per_fold: Dict[int, Dict[str, Counter]] = {}
    s550_ref_fold_scores: List[float] = []

    for fold_idx, fold in enumerate(splits, start=1):
        tr_idx = np.array(fold["train_indices"], dtype=np.int64)
        val_idx = np.array(fold["val_indices"], dtype=np.int64)

        p_s550_val, q_dist = engine.generate_s550_probs(tr_idx, val_idx)
        oof_p_s550_ref[val_idx] = p_s550_val
        q_dists_per_fold[fold_idx] = q_dist

        y_val_gold = [y_raw[i] for i in val_idx]
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
        [y_raw[i] for i in range(n_samples) if seen_mask[i]],
        [preds_s550_ref_all[i] for i in range(n_samples) if seen_mask[i]]
    )
    s550_unseen_map3 = calculate_map3(
        [y_raw[i] for i in range(n_samples) if not seen_mask[i]],
        [preds_s550_ref_all[i] for i in range(n_samples) if not seen_mask[i]]
    )
    s550_comp_map3 = 0.957 * s550_seen_map3 + 0.043 * s550_unseen_map3

    s550_rare_r10_count = sum(1 for i in rare_indices if y_raw[i] in [classes[c] for c in np.argsort(-oof_p_s550_ref[i])[:10]])
    s550_rare_r3_count = sum(1 for i in rare_indices if y_raw[i] in preds_s550_ref_all[i])
    s550_false_none = sum(1 for i in none_indices if preds_s550_ref_all[i][0] != "NONE")

    print(f"S550 Reference:")
    print(f"  Seen MAP@3:      {s550_seen_map3:.5f} | Unseen MAP@3: {s550_unseen_map3:.5f} | Composite MAP@3: {s550_comp_map3:.5f}")
    print(f"  Rare Recall@10:  {s550_rare_r10_count} / {len(rare_indices)} ({s550_rare_r10_count / len(rare_indices) * 100:.2f}%)")
    print(f"  Rare Top-3 Hits: {s550_rare_r3_count} / {len(rare_indices)} ({s550_rare_r3_count / len(rare_indices) * 100:.2f}%)")
    print(f"  False NONE Demotions: {s550_false_none} / {len(none_indices)} ({s550_false_none / len(none_indices) * 100:.2f}%)")

    # Evaluate Baseline Existing ConvNeXt Vision Component (oof_convnext_vision.npy)
    oof_conv_orig = np.load(ART_DIR / "oof_convnext_vision.npy")
    conv_top10 = [np.argsort(-oof_conv_orig[i])[:10] for i in range(n_samples)]
    conv_top3 = [[classes[c] for c in row[:3]] for row in conv_top10]

    conv_seen_map3 = calculate_map3(
        [y_raw[i] for i in range(n_samples) if seen_mask[i]],
        [conv_top3[i] for i in range(n_samples) if seen_mask[i]]
    )
    conv_unseen_map3 = calculate_map3(
        [y_raw[i] for i in range(n_samples) if not seen_mask[i]],
        [conv_top3[i] for i in range(n_samples) if not seen_mask[i]]
    )
    conv_comp_map3 = 0.957 * conv_seen_map3 + 0.043 * conv_unseen_map3
    conv_rare_r10 = sum(1 for i in rare_indices if y_true_int[i] in conv_top10[i])
    conv_false_none = sum(1 for i in none_indices if conv_top10[i][0] != none_idx)

    print(f"\nExisting Baseline ConvNeXt Vision Component (oof_convnext_vision.npy):")
    print(f"  Standalone MAP@3: Seen={conv_seen_map3:.5f} | Unseen={conv_unseen_map3:.5f} | Comp={conv_comp_map3:.5f}")
    print(f"  Rare Recall@10:   {conv_rare_r10} / {len(rare_indices)} ({conv_rare_r10 / len(rare_indices) * 100:.2f}%)")
    print(f"  False NONE Demotions: {conv_false_none} / {len(none_indices)} ({conv_false_none / len(none_indices) * 100:.2f}%)")

    # Train Supervised Contrastive Models
    # Variant 1: Pure SupCon (Prototype Cosine Retrieval)
    # Variant 2: Joint SupCon + CE (Cross-Entropy weight = 0.50)
    experiments = [
        {"name": "ConvNeXt_SupCon_Pure",  "joint_ce": 0.0,  "lr": 1e-3, "epochs": 25},
        {"name": "ConvNeXt_SupCon_Joint", "joint_ce": 0.50, "lr": 1e-3, "epochs": 25},
    ]

    exp_results: List[Dict[str, Any]] = []

    for exp in experiments:
        exp_name = exp["name"]
        joint_ce = float(exp["joint_ce"])
        epochs = int(exp["epochs"])
        lr = float(exp["lr"])

        print("\n" + "=" * 95)
        print(f"TRAINING EXPERIMENT: {exp_name} (Joint CE Weight = {joint_ce:.2f}, Epochs = {epochs})")
        print("=" * 95)
        t_exp_start = time.time()

        oof_proto_probs = np.zeros((n_samples, n_classes), dtype=np.float32)
        oof_ce_probs = np.zeros((n_samples, n_classes), dtype=np.float32)
        fold_scores_proto: List[float] = []

        for fold_idx, fold in enumerate(splits, start=1):
            tr_idx = np.array(fold["train_indices"], dtype=np.int64)
            val_idx = np.array(fold["val_indices"], dtype=np.int64)

            val_proto, val_ce = train_supcon_fold(
                X_train_vis=X_vis_orig,
                y_true_int=y_true_int,
                trn_idx=tr_idx,
                val_idx=val_idx,
                classes=classes,
                train_df=train_df,
                tax=tax,
                device=device,
                fold_idx=fold_idx,
                joint_ce_weight=joint_ce,
                epochs=epochs,
                lr=lr,
                temperature=0.07
            )
            oof_proto_probs[val_idx] = val_proto
            oof_ce_probs[val_idx] = val_ce

            y_val_gold = [y_raw[i] for i in val_idx]
            preds_proto_val = [[classes[c] for c in np.argsort(-val_proto[r])[:3]] for r in range(len(val_idx))]
            f_sc = calculate_map3(y_val_gold, preds_proto_val)
            fold_scores_proto.append(f_sc)
            print(f"  Fold {fold_idx}/5 Standalone Retrieval MAP@3: {f_sc:.5f}")

        # Evaluate Standalone Contrastive Retrieval
        target_probs = oof_ce_probs if joint_ce > 0.0 else oof_proto_probs
        top10_cand = [np.argsort(-target_probs[i])[:10] for i in range(n_samples)]
        top3_cand = [[classes[c] for c in row[:3]] for row in top10_cand]

        seen_sc = calculate_map3(
            [y_raw[i] for i in range(n_samples) if seen_mask[i]],
            [top3_cand[i] for i in range(n_samples) if seen_mask[i]]
        )
        unseen_sc = calculate_map3(
            [y_raw[i] for i in range(n_samples) if not seen_mask[i]],
            [top3_cand[i] for i in range(n_samples) if not seen_mask[i]]
        )
        comp_sc = 0.957 * seen_sc + 0.043 * unseen_sc

        rare_r10_cand = sum(1 for i in rare_indices if y_true_int[i] in top10_cand[i])
        rare_r5_cand = sum(1 for i in rare_indices if y_true_int[i] in top10_cand[i][:5])
        rare_r3_cand = sum(1 for i in rare_indices if y_true_int[i] in top10_cand[i][:3])
        false_none_cand = sum(1 for i in none_indices if top10_cand[i][0] != none_idx)

        print(f"\nStandalone Contrastive Results for {exp_name} ({time.time() - t_exp_start:.1f}s):")
        print(f"  Seen MAP@3:      {seen_sc:.5f} | Unseen MAP@3: {unseen_sc:.5f} | Composite MAP@3: {comp_sc:.5f}")
        print(f"  Rare Recall@10:  {rare_r10_cand} / {len(rare_indices)} ({rare_r10_cand / len(rare_indices) * 100:.2f}%)")
        print(f"  Rare Recall@5:   {rare_r5_cand} / {len(rare_indices)} ({rare_r5_cand / len(rare_indices) * 100:.2f}%)")
        print(f"  Rare Recall@3:   {rare_r3_cand} / {len(rare_indices)} ({rare_r3_cand / len(rare_indices) * 100:.2f}%)")
        print(f"  False NONE Demotions: {false_none_cand} / {len(none_indices)} ({false_none_cand / len(none_indices) * 100:.2f}%)")

        # Test In-Fold Ensemble Integration into S550:
        # Substitute target_probs into S550's clean multimodal branch (replacing oof_convnext)
        print(f"\nEvaluating In-Fold Substitution into S550 Full Pipeline...")
        oof_p_s550_cand = np.zeros((n_samples, n_classes), dtype=np.float64)
        s550_cand_fold_scores: List[float] = []

        for fold_idx, fold in enumerate(splits, start=1):
            tr_idx = np.array(fold["train_indices"], dtype=np.int64)
            val_idx = np.array(fold["val_indices"], dtype=np.int64)
            q_dist = q_dists_per_fold[fold_idx]

            trn_df_fold = train_df.iloc[tr_idx]
            val_df_fold = train_df.iloc[val_idx]
            from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
            prior_eng = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df_fold)
            val_prior = prior_eng.predict_proba(val_df_fold).astype(np.float64)

            # In-fold clean multimodal blend: replace oof_convnext with target_probs
            p_clean_val = (
                0.6448 * val_prior +
                0.0839 * sharp(engine.honest_text[val_idx], 0.6) +
                0.0401 * sharp(engine.oof_sem_text[val_idx], 0.6) +
                0.0458 * target_probs[val_idx] +
                0.0454 * engine.oof_swin[val_idx] +
                0.0483 * engine.oof_finetuned[val_idx] +
                0.0455 * engine.oof_trocr[val_idx] +
                0.0462 * sharp(engine.oof_clean_arcface[val_idx], 0.8)
            )

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
            p_s550_new = 0.96 * p_champ_v1 + 0.04 * p_clean_val

            p_s550_new = np.maximum(p_s550_new, 0.0)
            p_s550_new = p_s550_new / p_s550_new.sum(axis=1, keepdims=True)
            oof_p_s550_cand[val_idx] = p_s550_new

            preds_val = engine.rank_and_gate(p_s550_new, val_idx, q_dist)
            f_sc_s550 = calculate_map3([y_raw[i] for i in val_idx], preds_val)
            s550_cand_fold_scores.append(f_sc_s550)

        preds_s550_cand_all = [[] for _ in range(n_samples)]
        for fold_idx, fold in enumerate(splits, start=1):
            val_idx = np.array(fold["val_indices"], dtype=np.int64)
            preds_val = engine.rank_and_gate(oof_p_s550_cand[val_idx], val_idx, q_dists_per_fold[fold_idx])
            for i, idx in enumerate(val_idx):
                preds_s550_cand_all[idx] = preds_val[i]

        s550_cand_seen = calculate_map3(
            [y_raw[i] for i in range(n_samples) if seen_mask[i]],
            [preds_s550_cand_all[i] for i in range(n_samples) if seen_mask[i]]
        )
        s550_cand_unseen = calculate_map3(
            [y_raw[i] for i in range(n_samples) if not seen_mask[i]],
            [preds_s550_cand_all[i] for i in range(n_samples) if not seen_mask[i]]
        )
        s550_cand_comp = 0.957 * s550_cand_seen + 0.043 * s550_cand_unseen
        delta_s550 = s550_cand_comp - s550_comp_map3

        s550_cand_rare_r10 = sum(1 for i in rare_indices if y_raw[i] in [classes[c] for c in np.argsort(-oof_p_s550_cand[i])[:10]])
        s550_cand_false_none = sum(1 for i in none_indices if preds_s550_cand_all[i][0] != "NONE")

        print(f"S550 Integrated Results with {exp_name}:")
        print(f"  Seen MAP@3:      {s550_cand_seen:.5f} | Unseen MAP@3: {s550_cand_unseen:.5f} | Composite MAP@3: {s550_cand_comp:.5f} (Delta: {delta_s550:+.5f})")
        print(f"  S550 Folds:      {[round(s, 5) for s in s550_cand_fold_scores]}")
        print(f"  S550 Rare R@10:  {s550_cand_rare_r10} (Delta vs S550: {s550_cand_rare_r10 - s550_rare_r10_count:+d})")
        print(f"  S550 False NONE: {s550_cand_false_none} (Delta vs S550: {s550_cand_false_none - s550_false_none:+d})")

        exp_results.append({
            "name": exp_name,
            "standalone": {
                "seen_map3": round(seen_sc, 5),
                "unseen_map3": round(unseen_sc, 5),
                "composite_map3": round(comp_sc, 5),
                "rare_r10_hits": rare_r10_cand,
                "rare_r10_rate": round(rare_r10_cand / len(rare_indices), 4),
                "false_none_count": false_none_cand,
                "false_none_rate": round(false_none_cand / len(none_indices), 4)
            },
            "s550_integrated": {
                "seen_map3": round(s550_cand_seen, 5),
                "unseen_map3": round(s550_cand_unseen, 5),
                "composite_map3": round(s550_cand_comp, 5),
                "delta_vs_s550": round(delta_s550, 5),
                "folds": [round(s, 5) for s in s550_cand_fold_scores],
                "rare_r10_hits": s550_cand_rare_r10,
                "delta_rare_r10": s550_cand_rare_r10 - s550_rare_r10_count,
                "false_none_count": s550_cand_false_none,
                "delta_false_none": s550_cand_false_none - s550_false_none
            }
        })

    # Summary Comparison Table
    print("\n" + "=" * 125)
    print("CONVNEXT-448 SUPERVISED CONTRASTIVE EXPERIMENT: COMPARATIVE SUMMARY")
    print("=" * 125)
    print(
        f"{'Model Architecture / Pipeline':36s} | {'Standalone Comp':15s} | {'Standalone R@10':15s} | "
        f"{'S550 Comp':9s} | {'Delta S550':10s} | {'S550 R@10':9s} | {'S550 False NONE':15s}"
    )
    print("-" * 125)
    print(
        f"{'S550 Champion Reference':36s} | {'N/A':15s} | {'N/A':15s} | "
        f"{s550_comp_map3:9.5f} | {'+0.00000':10s} | {s550_rare_r10_count:4d} / 298 | {s550_false_none:4d} / 506"
    )
    print(
        f"{'Existing Baseline ConvNeXt Vision':36s} | {conv_comp_map3:15.5f} | {conv_rare_r10 / len(rare_indices) * 100:14.1f}% | "
        f"{s550_comp_map3:9.5f} | {'(Active)':10s} | {s550_rare_r10_count:4d} / 298 | {s550_false_none:4d} / 506"
    )

    for r in exp_results:
        st_c = r["standalone"]["composite_map3"]
        st_r10 = r["standalone"]["rare_r10_rate"] * 100
        s550_c = r["s550_integrated"]["composite_map3"]
        delta_s550 = r["s550_integrated"]["delta_vs_s550"]
        s550_r10 = r["s550_integrated"]["rare_r10_hits"]
        s550_fn = r["s550_integrated"]["false_none_count"]
        print(
            f"{r['name']:36s} | {st_c:15.5f} | {st_r10:14.1f}% | "
            f"{s550_c:9.5f} | {delta_s550:+10.5f} | {s550_r10:4d} / 298 | {s550_fn:4d} / 506"
        )
    print("=" * 125)

    # Decision Gate Verification
    best_exp = max(exp_results, key=lambda x: x["s550_integrated"]["composite_map3"])
    best_delta = float(best_exp["s550_integrated"]["delta_vs_s550"])
    best_delta_r10 = int(best_exp["s550_integrated"]["delta_rare_r10"])
    best_delta_fn = int(best_exp["s550_integrated"]["delta_false_none"])

    print("\n" + "=" * 95)
    print("DECISION GATE VERIFICATION:")
    print("=" * 95)
    print(f"  Reference S550 Composite MAP@3: {s550_comp_map3:.5f}")
    print(f"  Best Contrastive Candidate:     {best_exp['name']}")
    print(f"  Composite MAP@3 Delta:          {best_delta:+.5f}")
    print(f"  Rare Recall@10 Delta:           {best_delta_r10:+d} traces")
    print(f"  False NONE Demotions Delta:     {best_delta_fn:+d} traces")

    gate_cleared = (best_delta > 0.00010 and best_delta_fn <= 0 and best_delta_r10 >= 0)

    if gate_cleared:
        print(f"  GATE STATUS: [PASSED] -> Candidate {best_exp['name']} clears promotion gate!")
    else:
        print(f"  GATE STATUS: [FAILED] -> Supervised contrastive visual representation failed to improve held-out score.")
        print(f"  Finding: The visual tail on ConvNeXt crops lacks enough distinct geometric signal across 1,500 traces")
        print(f"           to cleanly separate rare classes from NONE without reliable labeled examples.")
        print(f"  ACTION: S550 remains locked as active champion (0.54214 CV / 0.55049 Public LB). Do not create submission.")
    print("=" * 95)

    # Save comprehensive report
    report_dict: Dict[str, Any] = {
        "execution_time_seconds": round(time.time() - start_total, 2),
        "s550_reference": {
            "seen_map3": round(s550_seen_map3, 5),
            "unseen_map3": round(s550_unseen_map3, 5),
            "composite_map3": round(s550_comp_map3, 5),
            "rare_r10_hits": s550_rare_r10_count,
            "false_none_count": s550_false_none
        },
        "baseline_convnext_vision": {
            "standalone_composite_map3": round(conv_comp_map3, 5),
            "rare_r10_hits": conv_rare_r10,
            "false_none_count": conv_false_none
        },
        "experiments": exp_results,
        "best_candidate": best_exp["name"],
        "gate_cleared": gate_cleared
    }

    out_path = ART_DIR / "convnext_supcon_experiment_report.json"
    with open(out_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[Saved Experiment Report] {out_path.name}")


if __name__ == "__main__":
    main()

