"""100% Leak-Free Question-Held-Out Reranker Pipeline.

Enforces Complete Methodological Hygiene:
1. Outer 5-Fold GroupKFold strictly on question_id (0.0% question leakage).
2. For each outer fold k:
   - Outer-training only: Fit priors and calibration strictly on T_k.
   - Inner cross-fitting (4-fold inner GroupKFold within T_k) to generate T_k's reranker features.
     No row in T_k is scored by a model or prior that saw its own question.
   - Outer validation candidates (V_k) generated strictly by a model that never saw V_k's questions.
3. Honest Recall@K audit on truly unseen questions.
4. Candidate reranker trained on inner cross-fitted features and evaluated on untouched outer holdouts.
5. Invariant: Rank 1 is 100% strictly fixed; 0.55049 champion remains untouched.
"""
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, normalize

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.metrics.map3 import calculate_map3

def main():
    print("=" * 80)
    print("REBUILDING 100% LEAK-FREE QUESTION-HELDOUT RERANKER PIPELINE")
    print("Outer GroupKFold + Inner Cross-Fitting + Untouched Outer Evaluation")
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

    # 1. Load Multimodal Embeddings
    print("Loading multimodal features...")
    cn_feat = np.load(ART_DIR / "train_convnext_features.npy")
    sw_feat = np.load(ART_DIR / "train_swin_features.npy")
    tr_feat = np.load(ART_DIR / "train_trocr_features.npy")
    tx_feat = np.load(ART_DIR / "train_text_embeddings_384.npy")
    vi_feat = np.load(ART_DIR / "train_vision_features.npy")
    tax_feats = np.load(ART_DIR / "taxonomy_embeddings_384.npy")

    # Normalized features
    X_all = np.hstack([
        normalize(cn_feat), normalize(sw_feat), normalize(tr_feat), normalize(tx_feat), normalize(vi_feat)
    ])
    trocr_norm = normalize(tr_feat)
    q_norm = normalize(tx_feat)
    tax_norm = normalize(tax_feats)

    # 2. Outer 5-Fold GroupKFold Setup
    outer_gkf = GroupKFold(n_splits=5)
    
    # Global arrays for honest outer predictions
    outer_base_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    outer_cand_indices = np.zeros((n_samples, 5), dtype=np.int64)
    outer_baseline_top3: List[List[str]] = [[] for _ in range(n_samples)]
    outer_reranked_top3: List[List[str]] = [[] for _ in range(n_samples)]

    fold_deltas = []
    total_outer_swaps = 0

    for outer_fold, (trn_idx, val_idx) in enumerate(outer_gkf.split(train_df, y_indices, groups=groups), start=1):
        print(f"\n" + "=" * 60)
        print(f"OUTER FOLD {outer_fold} / 5 ({len(trn_idx)} trn rows, {len(val_idx)} untouched val rows)")
        print("=" * 60)

        trn_df = train_df.iloc[trn_idx]
        val_df = train_df.iloc[val_idx]
        trn_y = y_indices[trn_idx]
        trn_y_raw = y_raw[trn_idx]
        val_y_raw = y_raw[val_idx].tolist()
        trn_groups = trn_df["question_id"].values
        n_trn = len(trn_idx)

        # ---------------------------------------------------------------------
        # STEP A: Inner Cross-Fitting on trn_df to generate leak-free training features
        # ---------------------------------------------------------------------
        print("  Generating training reranker features via 4-fold inner cross-fitting...")
        inner_gkf = GroupKFold(n_splits=4)
        inner_oof_probs = np.zeros((n_trn, n_classes), dtype=np.float64)
        inner_tg_priors = np.zeros((n_trn, n_classes), dtype=np.float64)

        for in_fold, (in_trn, in_val) in enumerate(inner_gkf.split(trn_df, trn_y, groups=trn_groups), start=1):
            in_trn_df = trn_df.iloc[in_trn]
            in_val_df = trn_df.iloc[in_val]

            # In-fold prior strictly on in_trn_df
            prior_in = AdaptiveQuestionPriorEngine(classes=classes).fit(in_trn_df)
            val_prior_in = prior_in.predict_proba(in_val_df).astype(np.float64)
            inner_tg_priors[in_val] = val_prior_in

            # In-fold classifier on in_trn
            clf_in = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=42)
            clf_in.fit(X_all[trn_idx[in_trn]], trn_y[in_trn])

            raw_val_in = clf_in.predict_proba(X_all[trn_idx[in_val]])
            clf_val_in = np.zeros((len(in_val), n_classes), dtype=np.float64)
            for col_i, cls_i in enumerate(clf_in.classes_):
                clf_val_in[:, cls_i] = raw_val_in[:, col_i]

            blend_in = 0.55 * clf_val_in + 0.45 * val_prior_in

            # In-fold calibration on in_trn
            pred_in_trn = clf_in.predict(X_all[trn_idx[in_trn]])
            c_pred = Counter(pred_in_trn)
            c_true = Counter(trn_y[in_trn])
            mult_in = np.ones(n_classes, dtype=np.float64)
            for c_i in range(n_classes):
                if c_i != none_idx:
                    mult_in[c_i] = (max(c_true.get(c_i, 1), 1) / max(c_pred.get(c_i, 1), 1)) ** 0.05
            mult_in[none_idx] = 0.9850

            inner_oof_probs[in_val] = blend_in * mult_in

        # Extract inner candidate lists & features for trn_df
        inner_cand_matrix = inner_oof_probs.copy()
        inner_cand_matrix[:, none_idx] = -1.0
        inner_cand_indices = np.argsort(-inner_cand_matrix, axis=1)[:, :5]

        inner_r1_is_none = (np.argmax(inner_oof_probs, axis=1) == none_idx)

        trn_features = []
        trn_targets = []

        for loc_i in range(n_trn):
            if inner_r1_is_none[loc_i]:
                glob_i = trn_idx[loc_i]
                true_c = trn_y_raw[loc_i]
                cands = inner_cand_indices[loc_i]
                p_cands = [inner_oof_probs[loc_i, c] for c in cands]
                p_best = p_cands[0]

                for rank_pos, c_idx in enumerate(cands, start=1):
                    code = classes[c_idx]
                    p_val = inner_oof_probs[loc_i, c_idx]
                    gap = p_val - p_best
                    cos_ocr = float(np.dot(trocr_norm[glob_i], tax_norm[c_idx]))
                    cos_q = float(np.dot(q_norm[glob_i], tax_norm[c_idx]))
                    tg_p = float(inner_tg_priors[loc_i, c_idx])

                    trn_features.append([p_val, rank_pos, gap, cos_ocr, cos_q, tg_p, cos_ocr * cos_q])
                    trn_targets.append(1 if code == true_c else 0)

        X_trn_rerank = np.array(trn_features, dtype=np.float32)
        y_trn_rerank = np.array(trn_targets, dtype=np.int32)
        print(f"  Constructed leak-free training pairs: {len(X_trn_rerank)} candidate examples")

        # Train shallow ranker on inner cross-fitted features
        scaler = StandardScaler()
        X_trn_scaled = scaler.fit_transform(X_trn_rerank)
        ranker = LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", class_weight="balanced", random_state=42)
        ranker.fit(X_trn_scaled, y_trn_rerank)

        # ---------------------------------------------------------------------
        # STEP B: Outer Model Training & Generating Candidates for UNTOUCHED val_df
        # ---------------------------------------------------------------------
        print("  Generating predictions on untouched validation questions (V_k)...")
        # Outer prior & classifier strictly on full T_k
        outer_prior_eng = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df)
        val_prior_outer = outer_prior_eng.predict_proba(val_df).astype(np.float64)

        outer_clf = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=42)
        outer_clf.fit(X_all[trn_idx], trn_y)

        val_raw_outer = outer_clf.predict_proba(X_all[val_idx])
        val_clf_outer = np.zeros((len(val_idx), n_classes), dtype=np.float64)
        for col_i, cls_i in enumerate(outer_clf.classes_):
            val_clf_outer[:, cls_i] = val_raw_outer[:, col_i]

        val_blend_outer = 0.55 * val_clf_outer + 0.45 * val_prior_outer

        # Outer calibration strictly from T_k
        pred_trn_outer = outer_clf.predict(X_all[trn_idx])
        c_pred_out = Counter(pred_trn_outer)
        c_true_out = Counter(trn_y)
        mult_outer = np.ones(n_classes, dtype=np.float64)
        for c_i in range(n_classes):
            if c_i != none_idx:
                mult_outer[c_i] = (max(c_true_out.get(c_i, 1), 1) / max(c_pred_out.get(c_i, 1), 1)) ** 0.05
        mult_outer[none_idx] = 0.9850

        val_probs_outer = val_blend_outer * mult_outer
        outer_base_probs[val_idx] = val_probs_outer

        # Extract outer candidates
        val_cand_mat = val_probs_outer.copy()
        val_cand_mat[:, none_idx] = -1.0
        val_cands_out = np.argsort(-val_cand_mat, axis=1)[:, :5]
        outer_cand_indices[val_idx] = val_cands_out

        # Baseline outer predictions
        val_base_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-val_probs_outer, axis=1)]
        f_base_score = calculate_map3(val_y_raw, val_base_top3)

        # ---------------------------------------------------------------------
        # STEP C: Reranking on UNTOUCHED val_df (Rank 1 strictly fixed)
        # ---------------------------------------------------------------------
        val_r1_is_none = (np.argmax(val_probs_outer, axis=1) == none_idx)
        val_reranked_top3 = [list(row) for row in val_base_top3]
        f_swaps = 0

        for loc_i in range(len(val_idx)):
            if val_r1_is_none[loc_i]:
                glob_i = val_idx[loc_i]
                cands = val_cands_out[loc_i]
                p_cands = [val_probs_outer[loc_i, c] for c in cands]
                p_best = p_cands[0]

                val_cand_feats = []
                for rank_pos, c_idx in enumerate(cands, start=1):
                    p_val = val_probs_outer[loc_i, c_idx]
                    gap = p_val - p_best
                    cos_ocr = float(np.dot(trocr_norm[glob_i], tax_norm[c_idx]))
                    cos_q = float(np.dot(q_norm[glob_i], tax_norm[c_idx]))
                    tg_p = float(val_prior_outer[loc_i, c_idx])
                    val_cand_feats.append([p_val, rank_pos, gap, cos_ocr, cos_q, tg_p, cos_ocr * cos_q])

                val_cand_scaled = scaler.transform(np.array(val_cand_feats))
                cand_prob_correct = ranker.predict_proba(val_cand_scaled)[:, 1]

                # Constrained swap condition: only swap if model gap between C1 and C2 is tight (< 0.03)
                # and ranker strictly prefers C2 over C1 by >= 0.15
                c1_idx = cands[0]
                c2_idx = cands[1]
                gap_12 = p_cands[0] - p_cands[1]
                ranker_margin = cand_prob_correct[1] - cand_prob_correct[0]

                if 0 < gap_12 < 0.030 and ranker_margin >= 0.15:
                    val_reranked_top3[loc_i][1] = classes[c2_idx]
                    val_reranked_top3[loc_i][2] = classes[c1_idx]
                    f_swaps += 1

        f_new_score = calculate_map3(val_y_raw, val_reranked_top3)
        delta_f = f_new_score - f_base_score
        fold_deltas.append(delta_f)
        total_outer_swaps += f_swaps

        print(f"  Outer Fold {outer_fold} Result (Untouched Holdout):")
        print(f"    Baseline MAP@3: {f_base_score:.5f}")
        print(f"    Reranked MAP@3: {f_new_score:.5f} ({delta_f:+.5f}) | Swaps: {f_swaps}")

        for loc_i, glob_i in enumerate(val_idx):
            outer_baseline_top3[glob_i] = val_base_top3[loc_i]
            outer_reranked_top3[glob_i] = val_reranked_top3[loc_i]

    # -------------------------------------------------------------------------
    # 3. Honest Recall@K Audit on True Held-Out Questions (Across All 1,500 Rows)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("HONEST CANDIDATE RETRIEVAL RECALL@K (100% UNSEEN QUESTIONS)")
    print("Zero leakage: candidates generated solely by models that never saw these questions")
    print("=" * 80)
    misc_rows = [i for i in range(n_samples) if y_raw[i] != "NONE"]
    print(f"Total labeled student traces: {n_samples}")
    print(f"Traces where student has a misconception: {len(misc_rows)}")

    # Check Recall@K on unseen questions
    outer_cand_matrix = outer_base_probs.copy()
    outer_cand_matrix[:, none_idx] = -1.0
    sorted_outer_cands = np.argsort(-outer_cand_matrix, axis=1)

    for k in [1, 2, 3, 5, 8, 10]:
        hits = 0
        for i in misc_rows:
            true_label = y_raw[i]
            top_k_codes = [classes[idx] for idx in sorted_outer_cands[i, :k]]
            if true_label in top_k_codes:
                hits += 1
        rec = hits / len(misc_rows)
        print(f"  Honest Candidate Recall@{k:2d}: {hits:4d} / {len(misc_rows)} ({rec:.1%})")

    # Subgroup where model predicted NONE at Rank 1
    r1_is_none_outer = [i for i in misc_rows if np.argmax(outer_base_probs[i]) == none_idx]
    print(f"\nSubgroup: NONE at Rank 1 on unseen questions while true label is a misconception ({len(r1_is_none_outer)} rows):")
    for k in [1, 2, 3, 5, 8, 10]:
        hits = 0
        for i in r1_is_none_outer:
            true_label = y_raw[i]
            top_k_codes = [classes[idx] for idx in sorted_outer_cands[i, :k]]
            if true_label in top_k_codes:
                hits += 1
        rec = hits / len(r1_is_none_outer)
        print(f"  Honest Recall@{k:2d} on 'NONE-at-Rank-1' rows: {hits:4d} / {len(r1_is_none_outer)} ({rec:.1%})")

    # -------------------------------------------------------------------------
    # 4. Overall Reranker Evaluation on Untouched Outer Holdouts
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL 100% LEAK-FREE CANDIDATE RERANKER RESULTS (OVERALL 1,500 ROWS)")
    print("=" * 80)
    total_base_map3 = calculate_map3(y_raw.tolist(), outer_baseline_top3)
    total_reranked_map3 = calculate_map3(y_raw.tolist(), outer_reranked_top3)
    net_delta = total_reranked_map3 - total_base_map3

    print(f"Baseline MAP@3 (Held-Out Questions) : {total_base_map3:.5f}")
    print(f"Reranked MAP@3 (Held-Out Questions) : {total_reranked_map3:.5f}")
    print(f"Net Generalization Delta           : {net_delta:+.5f}")
    print(f"Total swaps on untouched holdouts   : {total_outer_swaps}")
    print(f"Per-fold deltas                     : {[round(float(d), 5) for d in fold_deltas]}")

    if net_delta > 0.00010 and all(d >= 0.0 for d in fold_deltas):
        print("\nGATE STATUS: PASSED (Consistent net positive generalization across untouched folds)")
    else:
        print("\nGATE STATUS: UNVALIDATED / MARGINAL (Does not meet strict promotion threshold)")
    print("=" * 80)

if __name__ == "__main__":
    main()

