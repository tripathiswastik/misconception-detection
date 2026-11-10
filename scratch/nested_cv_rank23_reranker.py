"""Nested Question-Held-Out CV Benchmark for Rank-2/3 Candidate Reranker.

Protocol & Invariants:
1. Outer 5-Fold GroupKFold strictly on question_id (0.0% question leakage).
2. Rank 1 is strictly FIXED across all operations.
3. Within each outer fold k:
   - 4-Fold Inner GroupKFold on question_id within the 1,200 training rows.
   - In inner folds, evaluate alpha in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00].
   - Select alpha*_k that maximizes inner held-out MAP@3.
4. Apply alpha*_k ONCE to untouched outer holdout fold k.
5. Report true held-out MAP@3 delta vs. untouched baseline.
"""
from collections import Counter
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.metrics.map3 import calculate_map3

def rerank_rank23(
    base_probs: np.ndarray,
    tg_prior: np.ndarray,
    base_top3: List[List[str]],
    classes: List[str],
    class_to_idx: Dict[str, int],
    alpha: float
) -> Tuple[List[List[str]], int]:
    """Rerank Rank 2 and Rank 3 for rows where Rank 1 is NONE, keeping Rank 1 strictly fixed."""
    new_top3 = [list(row) for row in base_top3]
    swaps_count = 0
    if alpha >= 0.999:
        return new_top3, 0

    n_samples = len(base_top3)
    for i in range(n_samples):
        if base_top3[i][0] == "NONE":
            r1 = base_top3[i][0]  # FIXED
            r2 = base_top3[i][1]
            r3 = base_top3[i][2]

            c2_idx = class_to_idx[r2]
            c3_idx = class_to_idx[r3]

            p_mod2 = max(base_probs[i, c2_idx], 1e-12)
            p_mod3 = max(base_probs[i, c3_idx], 1e-12)

            p_tg2 = max(tg_prior[i, c2_idx], 1e-12)
            p_tg3 = max(tg_prior[i, c3_idx], 1e-12)

            # Combined score in log-space: S(c) = P_model^alpha * P_tg^(1-alpha)
            score2 = (p_mod2 ** alpha) * (p_tg2 ** (1.0 - alpha))
            score3 = (p_mod3 ** alpha) * (p_tg3 ** (1.0 - alpha))

            if score3 > score2:
                new_top3[i][1] = r3
                new_top3[i][2] = r2
                swaps_count += 1

    return new_top3, swaps_count


def main():
    print("=" * 80)
    print("NESTED QUESTION-HELD-OUT CV: RANK-2/3 CANDIDATE RERANKER")
    print("Strict outer holdouts: alpha selected in inner folds, scored once on outer folds")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_raw = train_df["root_code"].fillna("NONE").astype(str).values
    y_indices = np.array([class_to_idx[c] for c in y_raw], dtype=np.int64)
    groups = train_df["question_id"].values
    n_samples = len(train_df)

    # 1. Load Multimodal Features (2,880-d)
    cn_feat = np.load(ART_DIR / "train_convnext_features.npy")
    sw_feat = np.load(ART_DIR / "train_swin_features.npy")
    tr_feat = np.load(ART_DIR / "train_trocr_features.npy")
    tx_feat = np.load(ART_DIR / "train_text_embeddings_384.npy")
    vi_feat = np.load(ART_DIR / "train_vision_features.npy")

    X_all = np.hstack([
        normalize(cn_feat), normalize(sw_feat), normalize(tr_feat), normalize(tx_feat), normalize(vi_feat)
    ])
    print(f"Concatenated feature matrix: {X_all.shape}")

    # 2. Setup Outer 5-Fold GroupKFold
    outer_gkf = GroupKFold(n_splits=5)
    candidate_alphas = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]

    outer_baseline_preds: List[List[str]] = [[] for _ in range(n_samples)]
    outer_nested_preds: List[List[str]] = [[] for _ in range(n_samples)]
    
    selected_alphas: List[float] = []
    outer_baseline_scores: List[float] = []
    outer_nested_scores: List[float] = []
    outer_swaps_list: List[int] = []

    for outer_fold, (trn_idx, val_idx) in enumerate(outer_gkf.split(train_df, y_indices, groups=groups), start=1):
        print(f"\n" + "=" * 60)
        print(f"OUTER FOLD {outer_fold} / 5 ({len(trn_idx)} trn rows, {len(val_idx)} untouched val rows)")
        print("=" * 60)

        trn_df = train_df.iloc[trn_idx]
        val_df = train_df.iloc[val_idx]
        val_y = y_raw[val_idx].tolist()

        # ---------------------------------------------------------------------
        # INNER 4-FOLD CV ON trn_df TO TUNE alpha
        # ---------------------------------------------------------------------
        print("  Running Inner 4-Fold CV on training partition to select alpha*...")
        inner_gkf = GroupKFold(n_splits=4)
        inner_groups = trn_df["question_id"].values
        inner_y = y_indices[trn_idx]
        inner_X = X_all[trn_idx]
        n_inner = len(trn_idx)

        # Precompute inner OOF predictions & inner topic-grade priors
        inner_base_probs = np.zeros((n_inner, n_classes), dtype=np.float64)
        inner_tg_priors = np.zeros((n_inner, n_classes), dtype=np.float64)

        for in_fold, (in_trn, in_val) in enumerate(inner_gkf.split(trn_df, inner_y, groups=inner_groups), start=1):
            in_trn_df = trn_df.iloc[in_trn]
            in_val_df = trn_df.iloc[in_val]

            # In-fold prior strictly on in_trn_df
            prior_eng = AdaptiveQuestionPriorEngine(classes=classes).fit(in_trn_df)
            in_val_prior = prior_eng.predict_proba(in_val_df).astype(np.float64)
            inner_tg_priors[in_val] = in_val_prior

            # In-fold classifier
            clf_in = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=42)
            clf_in.fit(inner_X[in_trn], inner_y[in_trn])

            in_val_raw = clf_in.predict_proba(inner_X[in_val])
            in_val_clf = np.zeros((len(in_val), n_classes), dtype=np.float64)
            for col_i, cls_i in enumerate(clf_in.classes_):
                in_val_clf[:, cls_i] = in_val_raw[:, col_i]

            inner_base_probs[in_val] = 0.55 * in_val_clf + 0.45 * in_val_prior

        inner_base_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-inner_base_probs, axis=1)]
        inner_true_y = y_raw[trn_idx].tolist()
        inner_base_map3 = calculate_map3(inner_true_y, inner_base_top3)

        # Evaluate candidate alphas on inner CV
        best_alpha = 1.00
        best_inner_score = inner_base_map3

        print(f"  Inner Baseline MAP@3 (alpha=1.00): {inner_base_map3:.5f}")
        for alpha in candidate_alphas:
            if alpha == 1.00:
                continue
            in_reranked, in_swaps = rerank_rank23(
                inner_base_probs, inner_tg_priors, inner_base_top3, classes, class_to_idx, alpha
            )
            in_score = calculate_map3(inner_true_y, in_reranked)
            print(f"    alpha = {alpha:.2f} | Inner Swaps: {in_swaps:3d} | Inner MAP@3: {in_score:.5f} ({in_score - inner_base_map3:+.5f})")
            if in_score > best_inner_score:
                best_inner_score = in_score
                best_alpha = alpha

        print(f"  >> Selected alpha* for Outer Fold {outer_fold}: {best_alpha:.2f} (Inner gain: {best_inner_score - inner_base_map3:+.5f})")
        selected_alphas.append(best_alpha)

        # ---------------------------------------------------------------------
        # OUTER MODEL TRAINING & EVALUATION ON UNTOUCHED val_df
        # ---------------------------------------------------------------------
        # Train outer prior & classifier on all 1,200 rows in trn_df
        outer_prior_eng = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df)
        outer_val_prior = outer_prior_eng.predict_proba(val_df).astype(np.float64)

        outer_clf = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=42)
        outer_clf.fit(X_all[trn_idx], y_indices[trn_idx])

        outer_val_raw = outer_clf.predict_proba(X_all[val_idx])
        outer_val_clf = np.zeros((len(val_idx), n_classes), dtype=np.float64)
        for col_i, cls_i in enumerate(outer_clf.classes_):
            outer_val_clf[:, cls_i] = outer_val_raw[:, col_i]

        outer_val_probs = 0.55 * outer_val_clf + 0.45 * outer_val_prior

        # Outer Baseline predictions (no reranking)
        outer_val_base_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-outer_val_probs, axis=1)]
        f_base_score = calculate_map3(val_y, outer_val_base_top3)
        outer_baseline_scores.append(f_base_score)

        # Apply inner-selected alpha* to outer validation fold
        outer_val_nested_top3, outer_swaps = rerank_rank23(
            outer_val_probs, outer_val_prior, outer_val_base_top3, classes, class_to_idx, best_alpha
        )
        f_nested_score = calculate_map3(val_y, outer_val_nested_top3)
        outer_nested_scores.append(f_nested_score)
        outer_swaps_list.append(outer_swaps)

        print(f"  Outer Fold {outer_fold} Result (Untouched Holdout):")
        print(f"    Baseline MAP@3: {f_base_score:.5f}")
        print(f"    Nested MAP@3  : {f_nested_score:.5f} ({f_nested_score - f_base_score:+.5f}) | Swaps: {outer_swaps}")

        for loc_i, glob_i in enumerate(val_idx):
            outer_baseline_preds[glob_i] = outer_val_base_top3[loc_i]
            outer_nested_preds[glob_i] = outer_val_nested_top3[loc_i]

    # 3. Overall Outer Evaluation
    print("\n" + "=" * 80)
    print("FINAL NESTED QUESTION-HELD-OUT RESULTS (OVERALL 1,500 ROWS)")
    print("=" * 80)
    total_base_map3 = calculate_map3(y_raw.tolist(), outer_baseline_preds)
    total_nested_map3 = calculate_map3(y_raw.tolist(), outer_nested_preds)
    delta_total = total_nested_map3 - total_base_map3

    print(f"Selected alphas across 5 folds: {selected_alphas}")
    print(f"Total swaps on untouched outer holdouts: {sum(outer_swaps_list)}")
    print(f"\nOuter Baseline MAP@3 (Unmodified): {total_base_map3:.5f}")
    print(f"Outer Nested CV MAP@3 (Reranked) : {total_nested_map3:.5f}")
    print(f"Net Generalization Delta        : {delta_total:+.5f}")

    # Subgroup breakdown on untouched outer holdouts
    rare_counts = Counter(y_raw)
    rare_classes = set(c for c, cnt in rare_counts.items() if cnt <= 10)
    rare_mask = np.array([c in rare_classes for c in y_raw])

    rare_y = [y_raw[i] for i in range(n_samples) if rare_mask[i]]
    rare_base_p = [outer_baseline_preds[i] for i in range(n_samples) if rare_mask[i]]
    rare_nest_p = [outer_nested_preds[i] for i in range(n_samples) if rare_mask[i]]
    rare_base_score = calculate_map3(rare_y, rare_base_p)
    rare_nest_score = calculate_map3(rare_y, rare_nest_p)

    common_y = [y_raw[i] for i in range(n_samples) if not rare_mask[i]]
    common_base_p = [outer_baseline_preds[i] for i in range(n_samples) if not rare_mask[i]]
    common_nest_p = [outer_nested_preds[i] for i in range(n_samples) if not rare_mask[i]]
    common_base_score = calculate_map3(common_y, common_base_p)
    common_nest_score = calculate_map3(common_y, common_nest_p)

    print(f"\nSubgroup Performance on Untouched Outer Holdouts:")
    print(f"  Common Classes ({sum(~rare_mask)} rows): Baseline {common_base_score:.5f} -> Nested {common_nest_score:.5f} ({common_nest_score - common_base_score:+.5f})")
    print(f"  Rare Classes   ({sum(rare_mask)} rows): Baseline {rare_base_score:.5f} -> Nested {rare_nest_score:.5f} ({rare_nest_score - rare_base_score:+.5f})")

    # Gate decision
    print("\n" + "=" * 80)
    if delta_total > 0.00010:
        print(f"GATE STATUS: PASSED (+{delta_total:.5f} net gain across untouched outer folds)")
    else:
        print(f"GATE STATUS: FLAT / UNVALIDATED ({delta_total:+.5f} net gain across untouched outer folds)")
    print("=" * 80)

if __name__ == "__main__":
    main()

