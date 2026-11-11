"""Pilot: Privileged Information Text Distillation (LUPI) with Strict Inner Cross-Fitting.

Architecture & Protocol:
1. Outer Cross-Validation: 5 canonical folds from data/splits.json (1,200 train / 300 val).
2. Privileged Information (Teacher): Sees Question Context + Metadata + Human Gold Transcription Steps.
3. Test-Available Information (Student): Sees Question Context + Metadata ONLY (transcription steps stripped).
4. Strict Inner Cross-Fitting on Training Split (Leak-Free Targets):
   - Inside each outer train set (1,200 rows), split into 4 question-grouped inner folds.
   - Train 4 inner Teacher models on 3/4 train rows; predict on the held-out 1/4 train rows.
   - Reassemble un-leaked out-of-fold soft teacher targets P_teacher for all 1,200 outer train rows.
5. Student Training on Outer Train Split:
   - Input: Test-available features only (TF-IDF word/char + symbolic tokens).
   - Target: Gold one-hot targets + soft teacher targets (via KL-divergence distillation with temperature tau).
   - Baseline Comparison: Student trained with pure gold targets (alpha=0.0) vs Distilled Student (alpha=0.35).
6. Held-Out Evaluation on Outer Validation Split (300 rows):
   - Evaluated strictly using test-available features.
   - Measures Standalone Seen, Unseen, and Composite MAP@3.
   - Measures Rare Recall@10 (298 rare traces) and False NONE Demotions (506 true NONE traces).
   - Evaluates integration with S550 reference benchmark.
"""

from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, cast
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
from src.diagnoser.math_parser import MathFeatureExtractor
from src.metrics.map3 import calculate_map3
from audit_rare_class_s550_oof_error_breakdown import S550AuditEngine, sharp


def build_teacher_payload(row: pd.Series) -> str:
    """Build privileged text payload: Question + Metadata + Working Steps."""
    grade = str(row.get("grade", ""))
    topic = str(row.get("topic_name", ""))
    q_text = str(row.get("question_text", ""))
    key = str(row.get("correct_answer", ""))
    steps = str(row.get("transcription_steps", ""))

    math_tokens = MathFeatureExtractor.extract_feature_tokens(
        question_text=q_text,
        transcription_steps=steps,
        correct_answer=key,
        topic_name=topic,
        grade=grade
    )
    tokens_str = " ".join(math_tokens)
    return (
        f"GRADE_{grade} TOPIC_{topic} KEY_{key} {tokens_str} "
        f"QUESTION: {q_text} "
        f"WORKING: {steps}"
    )


def build_student_payload(row: pd.Series) -> str:
    """Build test-available text payload: Question + Metadata ONLY (zero working steps)."""
    grade = str(row.get("grade", ""))
    topic = str(row.get("topic_name", ""))
    q_text = str(row.get("question_text", ""))
    key = str(row.get("correct_answer", ""))

    math_tokens = MathFeatureExtractor.extract_feature_tokens(
        question_text=q_text,
        transcription_steps="",  # Strictly empty at test time
        correct_answer=key,
        topic_name=topic,
        grade=grade
    )
    tokens_str = " ".join(math_tokens)
    return (
        f"GRADE_{grade} TOPIC_{topic} KEY_{key} {tokens_str} "
        f"QUESTION: {q_text}"
    )


class TextClassifierMLP(nn.Module):
    """Fast, regularized text classification head with support for soft target distillation."""
    def __init__(self, in_features: int, num_classes: int = 76, hidden_dim: int = 256, dropout: float = 0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_student_head(
    X_train: torch.Tensor,
    y_hard: torch.Tensor,
    y_soft: Optional[torch.Tensor],
    X_val: torch.Tensor,
    num_classes: int = 76,
    alpha: float = 0.0,
    temperature: float = 1.5,
    epochs: int = 20,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """Train student model with hard labels and optional soft teacher distillation."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_train_d = X_train.to(device)
    y_hard_d = y_hard.to(device)
    y_soft_d = y_soft.to(device) if y_soft is not None else torch.zeros((len(X_train), num_classes), device=device)

    in_dim = X_train.shape[1]
    model = TextClassifierMLP(in_features=in_dim, num_classes=num_classes, hidden_dim=256, dropout=0.30).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    batch_size = 128
    n = len(X_train)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            bx = X_train_d[idx]
            by_hard = y_hard_d[idx]
            by_soft = y_soft_d[idx]

            logits = model(bx)
            loss_hard = F.cross_entropy(logits, by_hard)

            if alpha > 0.0 and y_soft is not None:
                log_prob_s = F.log_softmax(logits / temperature, dim=1)
                prob_t = F.softmax(by_soft / temperature, dim=1)
                loss_soft = F.kl_div(log_prob_s, prob_t, reduction="batchmean") * (temperature ** 2)
                loss = (1.0 - alpha) * loss_hard + alpha * loss_soft
            else:
                loss = loss_hard

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

    # Evaluation on val
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val.to(device))
        val_probs = F.softmax(val_logits, dim=1).cpu().numpy()

    return val_probs.astype(np.float64)


def main():
    print("=" * 110)
    print("PILOT: PRIVILEGED INFORMATION TEXT DISTILLATION (LUPI) WITH INNER CROSS-FITTING")
    print("=" * 110)
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

    class_counts = Counter(y_raw)
    rare_classes: Set[str] = {c for c, cnt in class_counts.items() if c != "NONE" and cnt < 15}
    rare_indices = [i for i in range(n_samples) if y_raw[i] in rare_classes]
    none_indices = [i for i in range(n_samples) if y_raw[i] == "NONE"]

    with open(DATA_DIR / "splits.json") as f:
        splits = json.load(f)["folds"]

    seen_mask = np.zeros(n_samples, dtype=bool)
    for fold in splits:
        tr_qids = set(train_df.iloc[fold["train_indices"]]["question_id"].astype(str))
        for i in fold["val_indices"]:
            if str(train_df.iloc[i]["question_id"]) in tr_qids:
                seen_mask[i] = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Total Samples: {n_samples} | Rare Samples: {len(rare_indices)} | NONE Samples: {len(none_indices)}")

    # 1. Precompute Structured Texts
    print("\n[Step 1] Precomputing Privileged (Teacher) and Test-Available (Student) Payloads...")
    teacher_texts = [build_teacher_payload(row) for _, row in train_df.iterrows()]
    student_texts = [build_student_payload(row) for _, row in train_df.iterrows()]
    print(f"  Sample Teacher Payload (first 120 chars): {teacher_texts[0][:120]}...")
    print(f"  Sample Student Payload (first 120 chars): {student_texts[0][:120]}...")

    # Containers for out-of-fold predictions
    oof_teacher_oracle = np.zeros((n_samples, n_classes), dtype=np.float64)
    oof_student_baseline = np.zeros((n_samples, n_classes), dtype=np.float64)
    oof_student_distilled = np.zeros((n_samples, n_classes), dtype=np.float64)

    # 2. Outer 5-Fold Loop
    print("\n[Step 2] Executing 5-Fold Question-Grouped Cross-Validation...")

    for fold_idx, fold in enumerate(splits, start=1):
        tr_idx = np.array(fold["train_indices"], dtype=np.int64)
        val_idx = np.array(fold["val_indices"], dtype=np.int64)
        n_tr = len(tr_idx)
        n_v = len(val_idx)

        print(f"\n--- Outer Fold {fold_idx}/5 (Train: {n_tr}, Val: {n_v}) ---")

        # Feature Vectorization inside fold (strict fold isolation)
        teacher_vec = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 3), max_features=8000, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b|[+\-*/=^><\\]")),
            ("char", TfidfVectorizer(ngram_range=(3, 5), max_features=12000, sublinear_tf=True, analyzer="char"))
        ])
        student_vec = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 3), max_features=8000, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b|[+\-*/=^><\\]")),
            ("char", TfidfVectorizer(ngram_range=(3, 5), max_features=12000, sublinear_tf=True, analyzer="char"))
        ])

        # Fit vectorizers on training split only
        tr_teacher_texts = [teacher_texts[i] for i in tr_idx]
        val_teacher_texts = [teacher_texts[i] for i in val_idx]
        tr_student_texts = [student_texts[i] for i in tr_idx]
        val_student_texts = [student_texts[i] for i in val_idx]

        X_tr_teach = torch.from_numpy(teacher_vec.fit_transform(tr_teacher_texts).toarray()).float()
        X_val_teach = torch.from_numpy(teacher_vec.transform(val_teacher_texts).toarray()).float()

        X_tr_stud = torch.from_numpy(student_vec.fit_transform(tr_student_texts).toarray()).float()
        X_val_stud = torch.from_numpy(student_vec.transform(val_student_texts).toarray()).float()

        y_tr_t = torch.from_numpy(y_true_int[tr_idx]).long()
        tr_qids = train_df.iloc[tr_idx]["question_id"].values

        # -----------------------------------------------------------------
        # A. Inner Cross-Fitting for Leak-Free Teacher Targets on tr_idx
        # -----------------------------------------------------------------
        inner_gkf = GroupKFold(n_splits=4)
        inner_teacher_targets = np.zeros((n_tr, n_classes), dtype=np.float32)

        for inner_tr, inner_val in inner_gkf.split(X_tr_teach, y_tr_t, groups=tr_qids):
            in_X_tr = X_tr_teach[inner_tr]
            in_y_tr = y_tr_t[inner_tr]
            in_X_val = X_tr_teach[inner_val]

            # Train inner teacher on inner_tr
            in_teach_probs = train_student_head(
                X_train=in_X_tr,
                y_hard=in_y_tr,
                y_soft=None,
                X_val=in_X_val,
                num_classes=n_classes,
                alpha=0.0,
                epochs=20,
                lr=2e-3,
                device=device
            )
            inner_teacher_targets[inner_val] = in_teach_probs

        # Outer teacher oracle on val_idx (for upper-bound ceiling audit)
        val_teach_probs = train_student_head(
            X_train=X_tr_teach,
            y_hard=y_tr_t,
            y_soft=None,
            X_val=X_val_teach,
            num_classes=n_classes,
            alpha=0.0,
            epochs=20,
            lr=2e-3,
            device=device
        )
        oof_teacher_oracle[val_idx] = val_teach_probs

        # -----------------------------------------------------------------
        # B. Train Baseline Student (alpha=0.0, gold labels only)
        # -----------------------------------------------------------------
        val_base_stud_probs = train_student_head(
            X_train=X_tr_stud,
            y_hard=y_tr_t,
            y_soft=None,
            X_val=X_val_stud,
            num_classes=n_classes,
            alpha=0.0,
            epochs=25,
            lr=2e-3,
            device=device
        )
        oof_student_baseline[val_idx] = val_base_stud_probs

        # -----------------------------------------------------------------
        # C. Train Distilled Student (LUPI: soft cross-fitted teacher targets)
        # -----------------------------------------------------------------
        inner_t_tensor = torch.from_numpy(inner_teacher_targets).float()
        val_dist_stud_probs = train_student_head(
            X_train=X_tr_stud,
            y_hard=y_tr_t,
            y_soft=inner_t_tensor,
            X_val=X_val_stud,
            num_classes=n_classes,
            alpha=0.40,
            temperature=1.5,
            epochs=25,
            lr=2e-3,
            device=device
        )
        oof_student_distilled[val_idx] = val_dist_stud_probs

        # In-fold domain masking for evaluation
        masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df.iloc[tr_idx])
        v_topics = [str(t) for t in train_df.iloc[val_idx]["topic_name"].values]
        v_grades = [int(float(g)) for g in train_df.iloc[val_idx]["grade"].values]

        p_teach_cal = masker.apply_mask_batch(val_teach_probs, classes, v_topics, v_grades, damping=0.05)
        p_base_cal = masker.apply_mask_batch(val_base_stud_probs, classes, v_topics, v_grades, damping=0.05)
        p_dist_cal = masker.apply_mask_batch(val_dist_stud_probs, classes, v_topics, v_grades, damping=0.05)

        oof_teacher_oracle[val_idx] = p_teach_cal
        oof_student_baseline[val_idx] = p_base_cal
        oof_student_distilled[val_idx] = p_dist_cal

        gold_v = [y_raw[i] for i in val_idx]
        preds_teach_v = [[classes[c] for c in row[:3]] for row in np.argsort(-p_teach_cal, axis=1)]
        preds_base_v = [[classes[c] for c in row[:3]] for row in np.argsort(-p_base_cal, axis=1)]
        preds_dist_v = [[classes[c] for c in row[:3]] for row in np.argsort(-p_dist_cal, axis=1)]

        sc_teach = calculate_map3(gold_v, preds_teach_v)
        sc_base = calculate_map3(gold_v, preds_base_v)
        sc_dist = calculate_map3(gold_v, preds_dist_v)
        print(f"  Fold {fold_idx} MAP@3: Teacher (Privileged): {sc_teach:.4f} | Student Baseline: {sc_base:.4f} | Student Distilled: {sc_dist:.4f} (Delta: {sc_dist - sc_base:+.4f})", flush=True)

    # 3. Overall 5-Fold Evaluation
    print("\n" + "=" * 110)
    print("5-FOLD FULL DATASET BENCHMARK EVALUATION")
    print("=" * 110)

    def eval_model(p_matrix: np.ndarray, model_name: str) -> Dict[str, Any]:
        preds_all = [[classes[c] for c in row[:3]] for row in np.argsort(-p_matrix, axis=1)]
        sc_seen = calculate_map3([y_raw[i] for i in range(n_samples) if seen_mask[i]], [preds_all[i] for i in range(n_samples) if seen_mask[i]])
        sc_unseen = calculate_map3([y_raw[i] for i in range(n_samples) if not seen_mask[i]], [preds_all[i] for i in range(n_samples) if not seen_mask[i]])
        sc_comp = 0.957 * sc_seen + 0.043 * sc_unseen

        top10_all = [np.argsort(-p_matrix[i])[:10] for i in range(n_samples)]
        r10_hits = sum(1 for i in rare_indices if y_true_int[i] in top10_all[i])
        r3_hits = sum(1 for i in rare_indices if y_true_int[i] in top10_all[i][:3])
        fn_count = sum(1 for i in none_indices if preds_all[i][0] != "NONE")

        return {
            "name": model_name,
            "composite_map3": round(sc_comp, 5),
            "seen_map3": round(sc_seen, 5),
            "unseen_map3": round(sc_unseen, 5),
            "rare_r10_hits": r10_hits,
            "rare_r10_rate": round(r10_hits / len(rare_indices), 4),
            "rare_r3_hits": r3_hits,
            "false_none_count": fn_count,
            "false_none_rate": round(fn_count / len(none_indices), 4)
        }

    teach_res = eval_model(oof_teacher_oracle, "Teacher (Privileged Oracle: Question + Working Steps)")
    base_res = eval_model(oof_student_baseline, "Student Baseline (Gold-only: Question Context Only)")
    dist_res = eval_model(oof_student_distilled, "Student Distilled (LUPI: Question Context + Soft Targets)")

    print(f"\nModel Benchmark Results:")
    for r in [teach_res, base_res, dist_res]:
        print(f"  {r['name']:65s} | Comp MAP@3: {r['composite_map3']:.5f} (Seen: {r['seen_map3']:.5f}, Unseen: {r['unseen_map3']:.5f}) | "
              f"Rare R@10: {r['rare_r10_hits']}/{len(rare_indices)} | False NONE: {r['false_none_count']}/{len(none_indices)}")

    # 4. Integrate Student Distillation into Champion S550 Reference
    print("\n[Step 4] Evaluating Integration with Active Champion S550 Reference...")
    engine = S550AuditEngine()

    oof_p_s550 = np.zeros((n_samples, n_classes), dtype=np.float64)
    q_dists_per_fold: Dict[int, Dict[str, Counter]] = {}
    for fold_idx, fold in enumerate(splits, start=1):
        tr_idx = np.array(fold["train_indices"], dtype=np.int64)
        val_idx = np.array(fold["val_indices"], dtype=np.int64)
        p_val, q_dist = engine.generate_s550_probs(tr_idx, val_idx)
        oof_p_s550[val_idx] = p_val
        q_dists_per_fold[fold_idx] = q_dist

    preds_s550_all = [[] for _ in range(n_samples)]
    for fold_idx, fold in enumerate(splits, start=1):
        val_idx = np.array(fold["val_indices"], dtype=np.int64)
        preds_val = engine.rank_and_gate(oof_p_s550[val_idx], val_idx, q_dists_per_fold[fold_idx])
        for i, idx in enumerate(val_idx):
            preds_s550_all[idx] = preds_val[i]

    s550_seen = calculate_map3([y_raw[i] for i in range(n_samples) if seen_mask[i]], [preds_s550_all[i] for i in range(n_samples) if seen_mask[i]])
    s550_unseen = calculate_map3([y_raw[i] for i in range(n_samples) if not seen_mask[i]], [preds_s550_all[i] for i in range(n_samples) if not seen_mask[i]])
    s550_comp = 0.957 * s550_seen + 0.043 * s550_unseen
    s550_rare_r10 = sum(1 for i in rare_indices if y_raw[i] in [classes[c] for c in np.argsort(-oof_p_s550[i])[:10]])
    s550_fn = sum(1 for i in none_indices if preds_s550_all[i][0] != "NONE")

    print(f"\nS550 Reference Benchmark:")
    print(f"  Composite MAP@3: {s550_comp:.5f} | Rare R@10: {s550_rare_r10}/{len(rare_indices)} | False NONE: {s550_fn}/{len(none_indices)}")

    # Blend Distilled Student at w=0.03 and w=0.05
    for w in [0.02, 0.04, 0.06]:
        p_blend = (1.0 - w) * oof_p_s550 + w * oof_student_distilled
        p_blend = p_blend / p_blend.sum(axis=1, keepdims=True)

        preds_blend = [[] for _ in range(n_samples)]
        for fold_idx, fold in enumerate(splits, start=1):
            val_idx = np.array(fold["val_indices"], dtype=np.int64)
            preds_val = engine.rank_and_gate(p_blend[val_idx], val_idx, q_dists_per_fold[fold_idx])
            for i, idx in enumerate(val_idx):
                preds_blend[idx] = preds_val[i]

        bl_seen = calculate_map3([y_raw[i] for i in range(n_samples) if seen_mask[i]], [preds_blend[i] for i in range(n_samples) if seen_mask[i]])
        bl_unseen = calculate_map3([y_raw[i] for i in range(n_samples) if not seen_mask[i]], [preds_blend[i] for i in range(n_samples) if not seen_mask[i]])
        bl_comp = 0.957 * bl_seen + 0.043 * bl_unseen
        bl_r10 = sum(1 for i in rare_indices if y_raw[i] in [classes[c] for c in np.argsort(-p_blend[i])[:10]])
        bl_fn = sum(1 for i in none_indices if preds_blend[i][0] != "NONE")

        print(f"  S550 + Distilled Student (w={w:.2f}) -> Comp MAP@3: {bl_comp:.5f} (Delta: {bl_comp - s550_comp:+.5f}) | "
              f"Rare R@10: {bl_r10} (Delta: {bl_r10 - s550_rare_r10:+d}) | False NONE: {bl_fn} (Delta: {bl_fn - s550_fn:+d})", flush=True)

    # Gate check
    # Also save report artifact
    report = {
        "execution_time_seconds": round(time.time() - start_time, 2),
        "models": {
            "teacher_privileged_oracle": teach_res,
            "student_baseline_gold_only": base_res,
            "student_distilled_lupi": dist_res
        },
        "student_distillation_gain": {
            "standalone_delta_comp_map3": round(dist_res["composite_map3"] - base_res["composite_map3"], 5),
            "standalone_delta_rare_r10": dist_res["rare_r10_hits"] - base_res["rare_r10_hits"],
            "standalone_delta_false_none": dist_res["false_none_count"] - base_res["false_none_count"],
        },
        "s550_reference": {
            "composite_map3": round(s550_comp, 5),
            "rare_r10": s550_rare_r10,
            "false_none": s550_fn
        }
    }

    report_path = ART_DIR / "privileged_information_pilot_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved privileged information pilot report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
