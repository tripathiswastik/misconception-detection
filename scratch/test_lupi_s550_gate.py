"""Comprehensive Fold-Safe Evaluation of LUPI Distilled Student Blended with S550.

Gate Directives:
1. S550 Champion Baseline: 0.54214 Composite MAP@3, False NONE Demotions <= 104 / 506.
2. Evaluate Naive Linear Blend Sweep: w in [0.001, 0.050].
3. Evaluate Strict NONE-Protected Blend: P(NONE) locked to S550, student blends only over 75 misconception classes.
4. Evaluate Non-NONE Misconception Slice: test ranking on the 994 true misconception traces.
5. Save all OOF matrices (oof_student_distilled.npy, oof_student_baseline.npy, oof_p_s550.npy).
6. Binary Decision: If no blend configuration clears >0.54214 with <= 104 false NONE demotions, definitively close LUPI.
"""

from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion
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
from src.diagnoser.math_parser import MathFeatureExtractor
from src.metrics.map3 import calculate_map3
from audit_rare_class_s550_oof_error_breakdown import S550AuditEngine


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


def to_dense(matrix: Any) -> np.ndarray:
    """Safely convert sparse or dense matrix to dense ndarray."""
    if hasattr(matrix, "toarray"):
        return cast(np.ndarray, matrix.toarray())
    return np.asarray(matrix)


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

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val.to(device))
        val_probs = F.softmax(val_logits, dim=1).cpu().numpy()

    return val_probs.astype(np.float64)


def main():
    print("=" * 115)
    print("LUPI DISTILLED STUDENT x S550 CHAMPION: DEFINITIVE FOLD-SAFE GATE EVALUATION")
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

    class_counts = Counter(y_raw)
    rare_classes: Set[str] = {c for c, cnt in class_counts.items() if c != "NONE" and cnt < 15}
    rare_indices = [i for i in range(n_samples) if y_raw[i] in rare_classes]
    none_indices = [i for i in range(n_samples) if y_raw[i] == "NONE"]
    non_none_indices = [i for i in range(n_samples) if y_raw[i] != "NONE"]

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
    print(f"Total Samples: {n_samples} | True NONE: {len(none_indices)} | True Misconceptions: {len(non_none_indices)} | Rare: {len(rare_indices)}")

    # Step 1: Precompute Structured Texts
    print("\n[Step 1] Precomputing Privileged (Teacher) and Test-Available (Student) Payloads...")
    teacher_texts = [build_teacher_payload(row) for _, row in train_df.iterrows()]
    student_texts = [build_student_payload(row) for _, row in train_df.iterrows()]

    oof_student_baseline = np.zeros((n_samples, n_classes), dtype=np.float64)
    oof_student_distilled = np.zeros((n_samples, n_classes), dtype=np.float64)
    oof_p_s550 = np.zeros((n_samples, n_classes), dtype=np.float64)

    # Step 2: Outer 5-Fold Loop
    print("\n[Step 2] Executing 5-Fold Question-Grouped Cross-Validation with Inner Cross-Fitting...")
    engine = S550AuditEngine()
    q_dists_per_fold: Dict[int, Dict[str, Counter]] = {}

    for fold_idx, fold in enumerate(splits, start=1):
        tr_idx = np.array(fold["train_indices"], dtype=np.int64)
        val_idx = np.array(fold["val_indices"], dtype=np.int64)
        n_tr = len(tr_idx)
        n_v = len(val_idx)

        print(f"\n--- Fold {fold_idx}/5 (Train: {n_tr}, Val: {n_v}) ---")

        # 1. Generate S550 probabilities for this fold
        p_val_s550, q_dist = engine.generate_s550_probs(tr_idx, val_idx)
        oof_p_s550[val_idx] = p_val_s550
        q_dists_per_fold[fold_idx] = q_dist

        # 2. Vectorize texts
        teacher_vec = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 3), max_features=8000, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b|[+\-*/=^><\\]")),
            ("char", TfidfVectorizer(ngram_range=(3, 5), max_features=12000, sublinear_tf=True, analyzer="char"))
        ])
        student_vec = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 3), max_features=8000, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b|[+\-*/=^><\\]")),
            ("char", TfidfVectorizer(ngram_range=(3, 5), max_features=12000, sublinear_tf=True, analyzer="char"))
        ])

        tr_teacher_texts = [teacher_texts[i] for i in tr_idx]
        val_teacher_texts = [teacher_texts[i] for i in val_idx]
        tr_student_texts = [student_texts[i] for i in tr_idx]
        val_student_texts = [student_texts[i] for i in val_idx]

        X_tr_teach = torch.from_numpy(to_dense(teacher_vec.fit_transform(tr_teacher_texts))).float()
        X_tr_stud = torch.from_numpy(to_dense(student_vec.fit_transform(tr_student_texts))).float()
        X_val_stud = torch.from_numpy(to_dense(student_vec.transform(val_student_texts))).float()

        y_tr_t = torch.from_numpy(y_true_int[tr_idx]).long()
        tr_qids = train_df.iloc[tr_idx]["question_id"].values

        # Inner cross-fitting for leak-free teacher targets
        inner_gkf = GroupKFold(n_splits=4)
        inner_teacher_targets = np.zeros((n_tr, n_classes), dtype=np.float32)

        for inner_tr, inner_val in inner_gkf.split(X_tr_teach, y_tr_t, groups=tr_qids):
            in_X_tr = X_tr_teach[inner_tr]
            in_y_tr = y_tr_t[inner_tr]
            in_X_val = X_tr_teach[inner_val]

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

        # Train Baseline Student (alpha=0.0)
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

        # Train Distilled Student (alpha=0.40, tau=1.5)
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

        # Apply domain masking calibrated strictly on training split
        masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df.iloc[tr_idx])
        v_topics = [str(t) for t in train_df.iloc[val_idx]["topic_name"].values]
        v_grades = [int(float(g)) for g in train_df.iloc[val_idx]["grade"].values]

        p_base_cal = masker.apply_mask_batch(val_base_stud_probs, classes, v_topics, v_grades, damping=0.05)
        p_dist_cal = masker.apply_mask_batch(val_dist_stud_probs, classes, v_topics, v_grades, damping=0.05)

        oof_student_baseline[val_idx] = p_base_cal
        oof_student_distilled[val_idx] = p_dist_cal

        gold_v = [y_raw[i] for i in val_idx]
        preds_base_v = [[classes[c] for c in row[:3]] for row in np.argsort(-p_base_cal, axis=1)]
        preds_dist_v = [[classes[c] for c in row[:3]] for row in np.argsort(-p_dist_cal, axis=1)]
        print(f"  Fold {fold_idx} MAP@3: Baseline: {calculate_map3(gold_v, preds_base_v):.4f} | Distilled: {calculate_map3(gold_v, preds_dist_v):.4f}", flush=True)

    # Save OOF Matrices
    print("\n[Step 3] Saving OOF Probability Matrices to Disk...")
    np.save(ART_DIR / "oof_student_baseline.npy", oof_student_baseline)
    np.save(ART_DIR / "oof_student_distilled.npy", oof_student_distilled)
    np.save(ART_DIR / "oof_p_s550.npy", oof_p_s550)
    print("  Saved: oof_student_baseline.npy, oof_student_distilled.npy, oof_p_s550.npy")

    # Step 4: S550 Reference Benchmark
    print("\n[Step 4] Benchmarking S550 Reference Champion...")
    preds_s550_all = [[] for _ in range(n_samples)]
    for fold_idx, fold in enumerate(splits, start=1):
        val_idx = np.array(fold["val_indices"], dtype=np.int64)
        preds_val = engine.rank_and_gate(oof_p_s550[val_idx], val_idx, q_dists_per_fold[fold_idx])
        for i, idx in enumerate(val_idx):
            preds_s550_all[idx] = preds_val[i]

    def evaluate_predictions(preds: List[List[str]], probs: np.ndarray, name: str) -> Dict[str, Any]:
        sc_seen = calculate_map3([y_raw[i] for i in range(n_samples) if seen_mask[i]], [preds[i] for i in range(n_samples) if seen_mask[i]])
        sc_unseen = calculate_map3([y_raw[i] for i in range(n_samples) if not seen_mask[i]], [preds[i] for i in range(n_samples) if not seen_mask[i]])
        sc_comp = 0.957 * sc_seen + 0.043 * sc_unseen

        # Non-NONE slice MAP@3 (ranking among misconceptions)
        gold_miscon = [y_raw[i] for i in non_none_indices]
        preds_miscon = [preds[i] for i in non_none_indices]
        sc_miscon = calculate_map3(gold_miscon, preds_miscon)

        top10_all = [np.argsort(-probs[i])[:10] for i in range(n_samples)]
        r10_hits = sum(1 for i in rare_indices if y_true_int[i] in top10_all[i])
        r3_hits = sum(1 for i in rare_indices if y_raw[i] in preds[i][:3])
        fn_count = sum(1 for i in none_indices if preds[i][0] != "NONE")

        return {
            "name": name,
            "composite_map3": round(sc_comp, 5),
            "seen_map3": round(sc_seen, 5),
            "unseen_map3": round(sc_unseen, 5),
            "non_none_map3": round(sc_miscon, 5),
            "rare_r10_hits": r10_hits,
            "rare_r3_hits": r3_hits,
            "false_none_count": fn_count,
            "false_none_rate": round(fn_count / len(none_indices), 4)
        }

    s550_eval = evaluate_predictions(preds_s550_all, oof_p_s550, "S550 Champion Baseline")
    print(f"  S550 Benchmark: Comp MAP@3: {s550_eval['composite_map3']:.5f} | Non-NONE MAP@3: {s550_eval['non_none_map3']:.5f} | "
          f"Rare R@10: {s550_eval['rare_r10_hits']}/{len(rare_indices)} | False NONE: {s550_eval['false_none_count']}/{len(none_indices)}")

    # Step 5: Test All Blend Hypotheses
    print("\n" + "=" * 115)
    print("HYPOTHESIS 1: FINE-GRAINED LINEAR BLEND SWEEP (w in [0.001, 0.050])")
    print("=" * 115)

    linear_results: List[Dict[str, Any]] = []
    w_linear_list = [0.001, 0.002, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050]

    for w in w_linear_list:
        p_blend = (1.0 - w) * oof_p_s550 + w * oof_student_distilled
        p_blend = p_blend / p_blend.sum(axis=1, keepdims=True)

        preds_blend = [[] for _ in range(n_samples)]
        for fold_idx, fold in enumerate(splits, start=1):
            val_idx = np.array(fold["val_indices"], dtype=np.int64)
            preds_val = engine.rank_and_gate(p_blend[val_idx], val_idx, q_dists_per_fold[fold_idx])
            for i, idx in enumerate(val_idx):
                preds_blend[idx] = preds_val[i]

        res = evaluate_predictions(preds_blend, p_blend, f"Linear Blend (w={w:.3f})")
        res["delta_comp"] = round(res["composite_map3"] - s550_eval["composite_map3"], 5)
        res["delta_fn"] = res["false_none_count"] - s550_eval["false_none_count"]
        res["gate_pass"] = bool(res["composite_map3"] > s550_eval["composite_map3"] and res["false_none_count"] <= s550_eval["false_none_count"])
        linear_results.append(res)

        print(f"  w={w:0.3f} | Comp MAP@3: {res['composite_map3']:.5f} (Delta: {res['delta_comp']:+.5f}) | "
              f"Non-NONE: {res['non_none_map3']:.5f} | Rare R@10: {res['rare_r10_hits']} | False NONE: {res['false_none_count']} (Delta: {res['delta_fn']:+d}) | Gate: {'PASS' if res['gate_pass'] else 'FAIL'}")

    print("\n" + "=" * 115)
    print("HYPOTHESIS 2: STRICT NONE-PROTECTED BLEND (P(NONE) LOCKED TO S550, ZERO FALSE DEMOTIONS)")
    print("=" * 115)
    print("  Construction: Student only redistributes probability among the 75 misconception classes.")
    print("  P_cond(c | c != NONE) = (1 - w) * P_s550_cond + w * P_stud_cond")
    print("  P_blend(c) = (1 - P_s550(NONE)) * P_cond(c), P_blend(NONE) = P_s550(NONE)")

    protected_results: List[Dict[str, Any]] = []
    w_prot_list = [0.005, 0.010, 0.020, 0.030, 0.050, 0.080, 0.100, 0.150, 0.200]

    # Precompute conditional distributions over non-NONE classes
    p_s550_none = oof_p_s550[:, none_idx : none_idx + 1]
    p_s550_misc = np.delete(oof_p_s550, none_idx, axis=1)
    sum_s550_misc = p_s550_misc.sum(axis=1, keepdims=True)
    p_s550_misc_norm = np.where(sum_s550_misc > 1e-12, p_s550_misc / sum_s550_misc, 1.0 / (n_classes - 1))

    p_stud_misc = np.delete(oof_student_distilled, none_idx, axis=1)
    sum_stud_misc = p_stud_misc.sum(axis=1, keepdims=True)
    p_stud_misc_norm = np.where(sum_stud_misc > 1e-12, p_stud_misc / sum_stud_misc, 1.0 / (n_classes - 1))

    for w in w_prot_list:
        p_cond_blend = (1.0 - w) * p_s550_misc_norm + w * p_stud_misc_norm
        # Scale back to non-NONE total mass
        p_blend_misc = (1.0 - p_s550_none) * p_cond_blend
        # Reinsert NONE probability exactly
        p_prot_blend = np.insert(p_blend_misc, none_idx, p_s550_none[:, 0], axis=1)

        preds_blend = [[] for _ in range(n_samples)]
        for fold_idx, fold in enumerate(splits, start=1):
            val_idx = np.array(fold["val_indices"], dtype=np.int64)
            preds_val = engine.rank_and_gate(p_prot_blend[val_idx], val_idx, q_dists_per_fold[fold_idx])
            for i, idx in enumerate(val_idx):
                preds_blend[idx] = preds_val[i]

        res = evaluate_predictions(preds_blend, p_prot_blend, f"NONE-Protected Blend (w={w:.3f})")
        res["delta_comp"] = round(res["composite_map3"] - s550_eval["composite_map3"], 5)
        res["delta_fn"] = res["false_none_count"] - s550_eval["false_none_count"]
        res["gate_pass"] = bool(res["composite_map3"] > s550_eval["composite_map3"] and res["false_none_count"] <= s550_eval["false_none_count"])
        protected_results.append(res)

        print(f"  w={w:0.3f} | Comp MAP@3: {res['composite_map3']:.5f} (Delta: {res['delta_comp']:+.5f}) | "
              f"Non-NONE: {res['non_none_map3']:.5f} | Rare R@10: {res['rare_r10_hits']} | False NONE: {res['false_none_count']} (Delta: {res['delta_fn']:+d}) | Gate: {'PASS' if res['gate_pass'] else 'FAIL'}")

    print("\n" + "=" * 115)
    print("HYPOTHESIS 3: UNCERTAINTY-GATED BLEND (ONLY BLEND WHEN S550 IS UNCERTAIN)")
    print("=" * 115)

    gated_results: List[Dict[str, Any]] = []
    # Measure S550 top-1 to top-2 margin
    sorted_s550 = np.sort(oof_p_s550, axis=1)[:, ::-1]
    margins = sorted_s550[:, 0] - sorted_s550[:, 1]

    for margin_thresh in [0.05, 0.10, 0.15, 0.20]:
        for w in [0.02, 0.05, 0.10]:
            uncertain_mask = (margins < margin_thresh)
            p_gated = oof_p_s550.copy()
            # In uncertain rows, blend student
            p_gated[uncertain_mask] = (1.0 - w) * oof_p_s550[uncertain_mask] + w * oof_student_distilled[uncertain_mask]
            p_gated = p_gated / p_gated.sum(axis=1, keepdims=True)

            preds_blend = [[] for _ in range(n_samples)]
            for fold_idx, fold in enumerate(splits, start=1):
                val_idx = np.array(fold["val_indices"], dtype=np.int64)
                preds_val = engine.rank_and_gate(p_gated[val_idx], val_idx, q_dists_per_fold[fold_idx])
                for i, idx in enumerate(val_idx):
                    preds_blend[idx] = preds_val[i]

            res = evaluate_predictions(preds_blend, p_gated, f"Gated (margin<{margin_thresh:.2f}, w={w:.2f})")
            res["delta_comp"] = round(res["composite_map3"] - s550_eval["composite_map3"], 5)
            res["delta_fn"] = res["false_none_count"] - s550_eval["false_none_count"]
            res["gate_pass"] = bool(res["composite_map3"] > s550_eval["composite_map3"] and res["false_none_count"] <= s550_eval["false_none_count"])
            gated_results.append(res)

            print(f"  margin<{margin_thresh:.2f}, w={w:.2f} | Comp MAP@3: {res['composite_map3']:.5f} (Delta: {res['delta_comp']:+.5f}) | "
                  f"Non-NONE: {res['non_none_map3']:.5f} | Rare R@10: {res['rare_r10_hits']} | False NONE: {res['false_none_count']} (Delta: {res['delta_fn']:+d}) | Gate: {'PASS' if res['gate_pass'] else 'FAIL'}")

    # Summary and Gate Decision
    all_experiments = linear_results + protected_results + gated_results
    passing_configs = [e for e in all_experiments if e["gate_pass"]]

    gate_decision = "PASS" if len(passing_configs) > 0 else "FAIL_CLOSE_LUPI"
    print("\n" + "=" * 115)
    print(f"FINAL GATE DECISION: {gate_decision}")
    print("=" * 115)
    if gate_decision == "PASS":
        best_cfg = max(passing_configs, key=lambda x: x["composite_map3"])
        print(f"  Winning Configuration: {best_cfg['name']}")
        print(f"  Composite MAP@3: {best_cfg['composite_map3']} (Delta: {best_cfg['delta_comp']:+.5f})")
        print(f"  False NONE Demotions: {best_cfg['false_none_count']} (Delta: {best_cfg['delta_fn']:+d})")
        print("  Recommendation: Proceed to transformer upgrade.")
    else:
        print("  Result: Zero blend configurations cleared the gate (Composite MAP@3 > 0.54214 with False NONE <= 104).")
        print("  Recommendation: Definitively CLOSE the Privileged Information (LUPI) track per user directive.")
        print("  S550 remains locked as the final champion submission.")

    # Save complete report
    final_report = {
        "execution_time_seconds": round(time.time() - start_time, 2),
        "gate_decision": gate_decision,
        "s550_reference": s550_eval,
        "linear_blend_sweep": linear_results,
        "none_protected_blend_sweep": protected_results,
        "uncertainty_gated_sweep": gated_results
    }
    report_path = ART_DIR / "lupi_s550_gate_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nSaved definitive gate report to {report_path}")


if __name__ == "__main__":
    main()

