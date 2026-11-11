"""Nested Cross-Validation and Paired Uncertainty Analysis for VLM Integration.

Protocol:
1. Outer Cross-Validation: 5 canonical question-grouped folds (data/splits.json).
2. Inner Cross-Validation (Per Outer Fold):
   - 4-fold question-grouped cross-validation strictly on the 1,200 training traces (using GroupKFold).
   - Evaluate candidate blend configurations w in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10] (linear & protected).
   - Include w=0.0 (Pure S550 Champion) in candidate pool so inner folds can reject BLIP entirely.
   - Select configuration C_k* maximizing inner OOF Composite MAP@3 (tie-breaker: w=0.0).
3. Outer Fold Scoring:
   - Train VLM classification head on all 1,200 outer train traces.
   - Evaluate single inner-selected configuration C_k* on the untouched outer validation fold (300 traces).
   - Apply precision gating (S550AuditEngine.rank_and_gate) and record row-level predictions.
4. Paired Uncertainty Analysis by Question Group:
   - Group row-level predictions by question_id (~90-100 questions).
   - Compute question-level paired delta Delta_q = MAP@3_nested(q) - MAP@3_s550(q).
   - Run paired t-test, Wilcoxon signed-rank test, and 10,000-replicate cluster bootstrap 95% CI.
   - Save row-level and question-level CSVs and final validation JSON report.
"""

from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from audit_rare_class_s550_oof_error_breakdown import S550AuditEngine


class VLMClassificationHead(nn.Module):
    """Cosine classification head for VLM feature vectors."""
    def __init__(self, in_dim: int = 768, num_classes: int = 76):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = nn.Parameter(torch.tensor(16.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_norm = F.normalize(self.weight, p=2, dim=1)
        x_norm = F.normalize(x, p=2, dim=1)
        cosine = F.linear(x_norm, w_norm)
        return self.scale * cosine


def train_vlm_head(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    num_classes: int = 76,
    epochs: int = 40,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """Train classification head on VLM embeddings in-fold."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    head = VLMClassificationHead(in_dim=X_train.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_train_d = X_train.to(device)
    y_train_d = y_train.to(device)
    n = len(X_train)
    batch_size = 128

    head.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            bx = X_train_d[idx]
            by = y_train_d[idx]

            logits = head(bx)
            loss = F.cross_entropy(logits, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    head.eval()
    with torch.no_grad():
        val_logits = head(X_val.to(device))
        val_probs = F.softmax(val_logits, dim=1).cpu().numpy().astype(np.float64)

    return val_probs


def compute_sample_map3(gold: str, pred_top3: List[str]) -> float:
    """Compute MAP@3 score for a single sample."""
    for rank, p in enumerate(pred_top3[:3]):
        if p == gold:
            return 1.0 / (rank + 1)
    return 0.0


def blend_probabilities(
    p_s550: np.ndarray,
    p_vlm: np.ndarray,
    cfg: Dict[str, Any],
    none_idx: int,
    n_classes: int
) -> np.ndarray:
    """Blend S550 and VLM probability matrices according to configuration."""
    mode = cfg["type"]
    w = cfg["w"]

    if mode == "pure_s550" or w == 0.0:
        return p_s550.copy()

    if mode == "linear":
        p_blend = (1.0 - w) * p_s550 + w * p_vlm
        return p_blend / np.maximum(p_blend.sum(axis=1, keepdims=True), 1e-12)

    if mode == "protected":
        p_s550_none = p_s550[:, none_idx : none_idx + 1]
        p_s550_misc = np.delete(p_s550, none_idx, axis=1)
        sum_s550_misc = p_s550_misc.sum(axis=1, keepdims=True)
        p_s550_misc_norm = np.where(sum_s550_misc > 1e-12, p_s550_misc / sum_s550_misc, 1.0 / (n_classes - 1))

        p_vlm_misc = np.delete(p_vlm, none_idx, axis=1)
        sum_vlm_misc = p_vlm_misc.sum(axis=1, keepdims=True)
        p_vlm_misc_norm = np.where(sum_vlm_misc > 1e-12, p_vlm_misc / sum_vlm_misc, 1.0 / (n_classes - 1))

        p_cond = (1.0 - w) * p_s550_misc_norm + w * p_vlm_misc_norm
        p_prot = (1.0 - p_s550_none) * p_cond
        return np.insert(p_prot, none_idx, p_s550_none[:, 0], axis=1)

    raise ValueError(f"Unknown blend mode: {mode}")


def main():
    print("=" * 115)
    print("NESTED CROSS-VALIDATION & QUESTION-PAIRED UNCERTAINTY ANALYSIS FOR VLM")
    print("=" * 115)
    start_time = time.time()

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_raw = train_df["root_code"].fillna("NONE").astype(str).tolist()
    y_true_int = np.array([class_to_idx.get(r, none_idx) for r in y_raw], dtype=np.int64)
    n_samples = len(train_df)

    vlm_cache_path = ART_DIR / "train_blip_vlm_features_768.npy"
    if not vlm_cache_path.exists():
        raise FileNotFoundError(f"Missing cached VLM features at {vlm_cache_path}")
    print(f"Loading pre-computed BLIP VLM 768-dim features from {vlm_cache_path}...")
    vlm_features = np.load(vlm_cache_path)
    print(f"VLM features loaded: {vlm_features.shape}")

    engine = S550AuditEngine()

    with open(DATA_DIR / "splits.json") as f:
        splits = json.load(f)["folds"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Candidate grid for inner-fold selection
    candidate_configs: List[Dict[str, Any]] = [
        {"name": "pure_s550", "type": "pure_s550", "w": 0.0},
        {"name": "linear_w0.01", "type": "linear", "w": 0.01},
        {"name": "linear_w0.02", "type": "linear", "w": 0.02},
        {"name": "linear_w0.03", "type": "linear", "w": 0.03},
        {"name": "linear_w0.05", "type": "linear", "w": 0.05},
        {"name": "linear_w0.08", "type": "linear", "w": 0.08},
        {"name": "linear_w0.10", "type": "linear", "w": 0.10},
        {"name": "protected_w0.02", "type": "protected", "w": 0.02},
        {"name": "protected_w0.05", "type": "protected", "w": 0.05},
        {"name": "protected_w0.10", "type": "protected", "w": 0.10},
        {"name": "protected_w0.15", "type": "protected", "w": 0.15},
        {"name": "protected_w0.20", "type": "protected", "w": 0.20},
    ]

    # Pre-compute question-level seen masks for outer folds
    seen_mask_outer = np.zeros(n_samples, dtype=bool)
    for fold in splits:
        tr_qids = set(train_df.iloc[fold["train_indices"]]["question_id"].astype(str))
        for i in fold["val_indices"]:
            if str(train_df.iloc[i]["question_id"]) in tr_qids:
                seen_mask_outer[i] = True

    # Containers for outer-fold row-level records
    row_records: List[Dict[str, Any]] = []
    outer_selected_configs: List[Dict[str, Any]] = []

    print("\n" + "=" * 115)
    print("STEP 1: NESTED 5-FOLD EXECUTION (INNER SELECTION -> UNTOUCHED OUTER EVALUATION)")
    print("=" * 115)

    for outer_fold_idx, outer_fold in enumerate(splits, start=1):
        tr_idx_outer = np.array(outer_fold["train_indices"], dtype=np.int64)
        val_idx_outer = np.array(outer_fold["val_indices"], dtype=np.int64)
        n_tr = len(tr_idx_outer)
        n_val = len(val_idx_outer)

        print(f"\n>>> [Outer Fold {outer_fold_idx}/5] Outer Train: {n_tr} | Outer Val: {n_val}")

        # --- INNER CROSS-VALIDATION ON OUTER TRAIN SET (1,200 samples) ---
        print(f"  [Inner CV] Setting up 4 question-grouped inner folds on {n_tr} samples...")
        inner_gkf = GroupKFold(n_splits=4)
        inner_groups = train_df.iloc[tr_idx_outer]["question_id"].astype(str).values

        # Storage for inner OOF predictions
        inner_oof_vlm = np.zeros((n_tr, n_classes), dtype=np.float64)
        inner_oof_s550 = np.zeros((n_tr, n_classes), dtype=np.float64)
        inner_q_dists: Dict[int, Dict[str, Counter]] = {}
        inner_val_subidx_map: Dict[int, np.ndarray] = {}

        # Seen mask for inner folds
        inner_seen_mask = np.zeros(n_tr, dtype=bool)

        for inner_f_idx, (sub_tr_rel, sub_val_rel) in enumerate(inner_gkf.split(tr_idx_outer, groups=inner_groups), start=1):
            sub_tr_abs = tr_idx_outer[sub_tr_rel]
            sub_val_abs = tr_idx_outer[sub_val_rel]

            # Mark inner seen status
            inner_tr_qids = set(train_df.iloc[sub_tr_abs]["question_id"].astype(str))
            for i_sub in sub_val_rel:
                if str(train_df.iloc[tr_idx_outer[i_sub]]["question_id"]) in inner_tr_qids:
                    inner_seen_mask[i_sub] = True

            # Train VLM head in inner fold
            X_in_tr = torch.from_numpy(vlm_features[sub_tr_abs]).float()
            y_in_tr = torch.from_numpy(y_true_int[sub_tr_abs]).long()
            X_in_val = torch.from_numpy(vlm_features[sub_val_abs]).float()

            p_in_vlm_raw = train_vlm_head(
                X_train=X_in_tr,
                y_train=y_in_tr,
                X_val=X_in_val,
                num_classes=n_classes,
                epochs=35,
                lr=1e-2,
                weight_decay=1e-4,
                device=device
            )

            # In-fold domain constraint masking
            masker_in = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df.iloc[sub_tr_abs])
            in_v_topics = [str(t) for t in train_df.iloc[sub_val_abs]["topic_name"].values]
            in_v_grades = [int(float(g)) for g in train_df.iloc[sub_val_abs]["grade"].values]
            p_in_vlm_cal = masker_in.apply_mask_batch(p_in_vlm_raw, classes, in_v_topics, in_v_grades, damping=0.05)
            inner_oof_vlm[sub_val_rel] = p_in_vlm_cal

            # In-fold S550 generation
            p_in_s550, q_dist_in = engine.generate_s550_probs(sub_tr_abs, sub_val_abs)
            inner_oof_s550[sub_val_rel] = p_in_s550
            inner_q_dists[inner_f_idx] = q_dist_in
            inner_val_subidx_map[inner_f_idx] = sub_val_rel

        # Evaluate candidate configurations on inner OOF predictions
        print("  [Inner Selection] Evaluating 12 blend candidates on inner 4-fold OOF...")
        gold_inner = [y_raw[idx] for idx in tr_idx_outer]

        best_cfg = candidate_configs[0]
        best_inner_comp = -1.0
        s550_inner_comp = -1.0
        inner_eval_records = []

        for cfg in candidate_configs:
            # Blend inner OOF
            p_in_blend = blend_probabilities(inner_oof_s550, inner_oof_vlm, cfg, none_idx, n_classes)

            # Apply inner precision gating per inner fold
            preds_in_all = [[] for _ in range(n_tr)]
            for inner_f_idx in range(1, 5):
                sub_val_rel = inner_val_subidx_map[inner_f_idx]
                sub_val_abs = tr_idx_outer[sub_val_rel]
                p_sub_blend = p_in_blend[sub_val_rel]
                preds_sub = engine.rank_and_gate(p_sub_blend, sub_val_abs, inner_q_dists[inner_f_idx])
                for rel_i, pred in zip(sub_val_rel, preds_sub):
                    preds_in_all[rel_i] = pred

            if inner_seen_mask.any() and (~inner_seen_mask).any():
                in_seen = calculate_map3([gold_inner[i] for i in range(n_tr) if inner_seen_mask[i]], [preds_in_all[i] for i in range(n_tr) if inner_seen_mask[i]])
                in_unseen = calculate_map3([gold_inner[i] for i in range(n_tr) if not inner_seen_mask[i]], [preds_in_all[i] for i in range(n_tr) if not inner_seen_mask[i]])
                in_comp = 0.957 * in_seen + 0.043 * in_unseen
            else:
                in_comp = calculate_map3(gold_inner, preds_in_all)

            if cfg["name"] == "pure_s550":
                s550_inner_comp = in_comp

            inner_eval_records.append({
                "config": cfg["name"],
                "composite_map3": round(in_comp, 5),
                "delta_vs_s550": round(in_comp - s550_inner_comp, 5)
            })

            # Selection criterion: must strictly beat best with tie-breaker preferring pure S550
            if in_comp > best_inner_comp + 1e-6:
                best_inner_comp = in_comp
                best_cfg = cfg

        print(f"  [Inner Selected] Configuration: '{best_cfg['name']}' (Inner MAP@3: {best_inner_comp:.5f}, Delta vs S550: {best_inner_comp - s550_inner_comp:+.5f})")
        outer_selected_configs.append({
            "outer_fold": outer_fold_idx,
            "selected_config": best_cfg["name"],
            "inner_composite_map3": round(best_inner_comp, 5),
            "inner_delta_vs_s550": round(best_inner_comp - s550_inner_comp, 5),
            "inner_evaluations": inner_eval_records
        })

        # --- EVALUATE ON UNTOUCHED OUTER VALIDATION SET (300 samples) ---
        print(f"  [Outer Evaluation] Scoring '{best_cfg['name']}' on untouched outer val (Fold {outer_fold_idx})...")
        X_out_tr = torch.from_numpy(vlm_features[tr_idx_outer]).float()
        y_out_tr = torch.from_numpy(y_true_int[tr_idx_outer]).long()
        X_out_val = torch.from_numpy(vlm_features[val_idx_outer]).float()

        p_out_vlm_raw = train_vlm_head(
            X_train=X_out_tr,
            y_train=y_out_tr,
            X_val=X_out_val,
            num_classes=n_classes,
            epochs=40,
            lr=1e-2,
            weight_decay=1e-4,
            device=device
        )

        masker_out = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df.iloc[tr_idx_outer])
        out_v_topics = [str(t) for t in train_df.iloc[val_idx_outer]["topic_name"].values]
        out_v_grades = [int(float(g)) for g in train_df.iloc[val_idx_outer]["grade"].values]
        p_out_vlm_cal = masker_out.apply_mask_batch(p_out_vlm_raw, classes, out_v_topics, out_v_grades, damping=0.05)

        # Outer S550 generation
        p_out_s550, q_dist_out = engine.generate_s550_probs(tr_idx_outer, val_idx_outer)

        # Baseline S550 predictions on outer val
        preds_s550_val = engine.rank_and_gate(p_out_s550, val_idx_outer, q_dist_out)

        # Nested Model predictions on outer val (using best_cfg selected strictly from inner folds)
        p_out_nested_blend = blend_probabilities(p_out_s550, p_out_vlm_cal, best_cfg, none_idx, n_classes)
        preds_nested_val = engine.rank_and_gate(p_out_nested_blend, val_idx_outer, q_dist_out)

        # Compute row-level scores
        gold_out = [y_raw[idx] for idx in val_idx_outer]
        fold_s550_map3 = calculate_map3(gold_out, preds_s550_val)
        fold_nested_map3 = calculate_map3(gold_out, preds_nested_val)
        print(f"  [Outer Result] S550 MAP@3: {fold_s550_map3:.5f} | Nested MAP@3: {fold_nested_map3:.5f} (Delta: {fold_nested_map3 - fold_s550_map3:+.5f})")

        for i, val_global_idx in enumerate(val_idx_outer):
            row_data = train_df.iloc[val_global_idx]
            g = gold_out[i]
            p_s550 = preds_s550_val[i]
            p_nest = preds_nested_val[i]
            score_s550 = compute_sample_map3(g, p_s550)
            score_nest = compute_sample_map3(g, p_nest)

            row_records.append({
                "trace_id": str(row_data["trace_id"]),
                "global_index": int(val_global_idx),
                "outer_fold": outer_fold_idx,
                "question_id": str(row_data["question_id"]),
                "topic_name": str(row_data["topic_name"]),
                "grade": int(float(row_data["grade"])),
                "is_seen": bool(seen_mask_outer[val_global_idx]),
                "gold_code": g,
                "s550_pred_1": p_s550[0],
                "s550_pred_2": p_s550[1],
                "s550_pred_3": p_s550[2],
                "nested_pred_1": p_nest[0],
                "nested_pred_2": p_nest[1],
                "nested_pred_3": p_nest[2],
                "s550_map3": round(score_s550, 5),
                "nested_map3": round(score_nest, 5),
                "delta_map3": round(score_nest - score_s550, 5),
                "chosen_config": best_cfg["name"]
            })

    # Save Row-Level Predictions
    row_df = pd.DataFrame(row_records).sort_values("global_index").reset_index(drop=True)
    row_csv_path = ART_DIR / "nested_vlm_row_level_predictions.csv"
    row_df.to_csv(row_csv_path, index=False)
    print(f"\nSaved {len(row_df)} row-level predictions to {row_csv_path}")

    # --- STEP 2: AGGREGATE NESTED OUTCOMES ---
    print("\n" + "=" * 115)
    print("STEP 2: FULL-DATASET AGGREGATE PERFORMANCE COMPARISON")
    print("=" * 115)

    all_gold = row_df["gold_code"].tolist()
    all_s550_preds = row_df[["s550_pred_1", "s550_pred_2", "s550_pred_3"]].values.tolist()
    all_nested_preds = row_df[["nested_pred_1", "nested_pred_2", "nested_pred_3"]].values.tolist()
    is_seen_arr = row_df["is_seen"].values

    total_s550_seen = calculate_map3([all_gold[i] for i in range(n_samples) if is_seen_arr[i]], [all_s550_preds[i] for i in range(n_samples) if is_seen_arr[i]])
    total_s550_unseen = calculate_map3([all_gold[i] for i in range(n_samples) if not is_seen_arr[i]], [all_s550_preds[i] for i in range(n_samples) if not is_seen_arr[i]])
    total_s550_comp = 0.957 * total_s550_seen + 0.043 * total_s550_unseen

    total_nest_seen = calculate_map3([all_gold[i] for i in range(n_samples) if is_seen_arr[i]], [all_nested_preds[i] for i in range(n_samples) if is_seen_arr[i]])
    total_nest_unseen = calculate_map3([all_gold[i] for i in range(n_samples) if not is_seen_arr[i]], [all_nested_preds[i] for i in range(n_samples) if not is_seen_arr[i]])
    total_nest_comp = 0.957 * total_nest_seen + 0.043 * total_nest_unseen

    print(f"  S550 Baseline (Champion):  Composite MAP@3: {total_s550_comp:.5f} (Seen: {total_s550_seen:.5f}, Unseen: {total_s550_unseen:.5f})")
    print(f"  Nested VLM (Outer Validated): Composite MAP@3: {total_nest_comp:.5f} (Seen: {total_nest_seen:.5f}, Unseen: {total_nest_unseen:.5f})")
    print(f"  True Generalization Delta:    {total_nest_comp - total_s550_comp:+.5f} Composite MAP@3")

    # --- STEP 3: PAIRED UNCERTAINTY ANALYSIS BY QUESTION GROUP ---
    print("\n" + "=" * 115)
    print("STEP 3: PAIRED UNCERTAINTY ANALYSIS BY QUESTION GROUP")
    print("=" * 115)

    q_groups = row_df.groupby("question_id")
    q_records: List[Dict[str, Any]] = []

    for qid, grp in q_groups:
        n_q = len(grp)
        s550_q_mean = float(grp["s550_map3"].mean())
        nest_q_mean = float(grp["nested_map3"].mean())
        delta_q = nest_q_mean - s550_q_mean

        q_records.append({
            "question_id": str(qid),
            "outer_fold": int(grp["outer_fold"].iloc[0]),
            "topic_name": str(grp["topic_name"].iloc[0]),
            "grade": int(grp["grade"].iloc[0]),
            "is_seen": bool(grp["is_seen"].iloc[0]),
            "n_traces": n_q,
            "s550_mean_map3": round(s550_q_mean, 5),
            "nested_mean_map3": round(nest_q_mean, 5),
            "delta_map3": round(delta_q, 5),
            "n_improved": int((grp["delta_map3"] > 0).sum()),
            "n_degraded": int((grp["delta_map3"] < 0).sum()),
            "n_unchanged": int((grp["delta_map3"] == 0).sum()),
        })

    q_df = pd.DataFrame(q_records).sort_values("question_id").reset_index(drop=True)
    q_csv_path = ART_DIR / "nested_vlm_question_paired_analysis.csv"
    q_df.to_csv(q_csv_path, index=False)
    print(f"Saved {len(q_df)} question-group paired records to {q_csv_path}")

    # Statistical Tests across question groups
    deltas = q_df["delta_map3"].values
    n_q_total = len(deltas)
    n_improved_q = int((deltas > 1e-6).sum())
    n_degraded_q = int((deltas < -1e-6).sum())
    n_neutral_q = int((np.abs(deltas) <= 1e-6).sum())

    mean_delta_q = float(np.mean(deltas))
    std_delta_q = float(np.std(deltas, ddof=1))
    sem_delta_q = std_delta_q / np.sqrt(n_q_total)

    # Paired t-test
    t_stat, t_p_value = stats.ttest_1samp(deltas, 0.0)

    # Wilcoxon signed-rank test (on non-zero differences)
    non_zero_deltas = deltas[np.abs(deltas) > 1e-6]
    if len(non_zero_deltas) > 0:
        w_res = stats.wilcoxon(non_zero_deltas, alternative="two-sided")
        wilcoxon_stat = float(w_res.statistic)
        wilcoxon_p_value = float(w_res.pvalue)
    else:
        wilcoxon_stat = 0.0
        wilcoxon_p_value = 1.0

    # 10,000-replicate cluster bootstrap 95% Confidence Interval
    np.random.seed(42)
    n_boot = 10000
    boot_means = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample_q = np.random.choice(deltas, size=n_q_total, replace=True)
        boot_means[b] = np.mean(sample_q)

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    ci_contains_zero = ci_lower <= 0.0 <= ci_upper

    print(f"\n--- Question-Group Paired Statistical Test Results (N = {n_q_total} questions) ---")
    print(f"  Improved Questions: {n_improved_q}/{n_q_total} ({n_improved_q / n_q_total:.1%})")
    print(f"  Degraded Questions: {n_degraded_q}/{n_q_total} ({n_degraded_q / n_q_total:.1%})")
    print(f"  Neutral Questions:  {n_neutral_q}/{n_q_total} ({n_neutral_q / n_q_total:.1%})")
    print(f"  Mean Paired Question Delta:  {mean_delta_q:+.6f} +/- {sem_delta_q:.6f} (SEM)")
    print(f"  Paired t-test:               t = {t_stat:.4f}, p = {t_p_value:.4f} (df = {n_q_total - 1})")
    print(f"  Wilcoxon Signed-Rank Test:   stat = {wilcoxon_stat:.1f}, p = {wilcoxon_p_value:.4f}")
    print(f"  Cluster Bootstrap 95% CI:    [{ci_lower:+.6f}, {ci_upper:+.6f}] (Contains Zero: {ci_contains_zero})")

    # Seen vs Unseen breakdown
    seen_deltas = q_df[q_df["is_seen"]]["delta_map3"].values
    unseen_deltas = q_df[~q_df["is_seen"]]["delta_map3"].values
    t_seen, p_seen = stats.ttest_1samp(seen_deltas, 0.0) if len(seen_deltas) > 1 else (0.0, 1.0)
    t_unseen, p_unseen = stats.ttest_1samp(unseen_deltas, 0.0) if len(unseen_deltas) > 1 else (0.0, 1.0)

    print(f"\n--- Subgroup Breakdown ---")
    print(f"  Seen Questions (N={len(seen_deltas)}):   Mean Delta = {float(np.mean(seen_deltas)):+.6f} | t = {t_seen:.4f}, p = {p_seen:.4f}")
    print(f"  Unseen Questions (N={len(unseen_deltas)}): Mean Delta = {float(np.mean(unseen_deltas)):+.6f} | t = {t_unseen:.4f}, p = {p_unseen:.4f}")

    # Final Gate Verdict
    gate_verdict = "FAIL_MAINTAIN_S550"
    if total_nest_comp > total_s550_comp and not ci_contains_zero and ci_lower > 0:
        gate_verdict = "PASS_PROCEED"

    print("\n" + "=" * 115)
    print("FINAL NESTED VALIDATION VERDICT")
    print("=" * 115)
    print(f"Gate Verdict: {gate_verdict}")
    if gate_verdict == "FAIL_MAINTAIN_S550":
        print("Conclusion: Under nested cross-validation, the BLIP VLM does not achieve a statistically significant positive gain.")
        print("S550 remains the validated, frozen champion submission.")

    # Save validation report
    report = {
        "execution_time_seconds": round(time.time() - start_time, 2),
        "protocol": "Nested 5-Fold Question-Grouped CV (Inner 4-Fold GroupKFold Selection)",
        "gate_verdict": gate_verdict,
        "outer_evaluations": outer_selected_configs,
        "dataset_aggregate_metrics": {
            "s550_composite_map3": round(total_s550_comp, 5),
            "s550_seen_map3": round(total_s550_seen, 5),
            "s550_unseen_map3": round(total_s550_unseen, 5),
            "nested_composite_map3": round(total_nest_comp, 5),
            "nested_seen_map3": round(total_nest_seen, 5),
            "nested_unseen_map3": round(total_nest_unseen, 5),
            "generalization_delta_composite_map3": round(total_nest_comp - total_s550_comp, 5)
        },
        "question_paired_uncertainty_analysis": {
            "n_question_groups": n_q_total,
            "improved_questions": n_improved_q,
            "degraded_questions": n_degraded_q,
            "neutral_questions": n_neutral_q,
            "mean_question_delta": round(mean_delta_q, 6),
            "std_question_delta": round(std_delta_q, 6),
            "sem_question_delta": round(sem_delta_q, 6),
            "paired_t_test": {
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(t_p_value), 4),
                "df": n_q_total - 1
            },
            "wilcoxon_test": {
                "statistic": round(wilcoxon_stat, 2),
                "p_value": round(wilcoxon_p_value, 4)
            },
            "cluster_bootstrap_95_ci": {
                "ci_lower": round(ci_lower, 6),
                "ci_upper": round(ci_upper, 6),
                "contains_zero": ci_contains_zero
            },
            "subgroups": {
                "seen_mean_delta": round(float(np.mean(seen_deltas)), 6),
                "seen_t_p": round(float(p_seen), 4),
                "unseen_mean_delta": round(float(np.mean(unseen_deltas)), 6),
                "unseen_t_p": round(float(p_unseen), 4)
            }
        }
    }

    report_path = ART_DIR / "nested_vlm_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved master validation report to {report_path}")


if __name__ == "__main__":
    main()

