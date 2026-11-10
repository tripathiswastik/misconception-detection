"""100% Leak-Free Question-Grouped Candidate Reranker Benchmark.

Enforces Complete Methodological Hygiene (Resolving all user-flagged leakage paths):
1. Zero Precomputed OOF Leakage:
   - Does NOT use historical oof_m15, oof_arcface, or splits.json (which had ~90% question overlap).
   - Generates all candidate probabilities from scratch across 5-fold outer GroupKFold on question_id.
   - For every fold k, all classifier heads, priors, and calibrations are fitted ONLY on T_k.
   - Questions in V_k have NEVER been seen by any component model.
2. Zero In-Sample Prior / Calibration Leakage:
   - To train the ranker on T_k, all candidate probabilities, priors, and calibrations are generated
     strictly via inner 4-fold GroupKFold cross-fitting within T_k.
   - No row in T_k is scored by a prior, classifier, or calibration that saw its own question.
3. Strict Rank-1 Invariant:
   - Rank 1 predictions from the champion recipe remain 100% locked.
   - Only Rank 2 (Candidate 1) and Rank 3 (Candidate 2) are reordered for NONE-at-Rank-1 rows.
4. Transparent Audit & Result Artifact:
   - Full swap breakdown: Helped, Hurt, Neutral across all outer folds.
   - Exports results to artifacts/honest_question_grouped_reranker_report.json.
"""
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression, RidgeClassifier
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

def extract_tokens(text: str) -> Set[str]:
    clean = re.sub(r"[^a-zA-Z0-9\+\-\*\/\=\^\_]", " ", str(text).lower())
    return set(w for w in clean.split() if len(w) > 1)

def jaccard(s1: Set[str], s2: Set[str]) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def main():
    print("=" * 80)
    print("100% LEAK-FREE QUESTION-GROUPED CANDIDATE RERANKER BENCHMARK")
    print("Zero Precomputed OOFs | In-Fold Component Training | Inner Cross-Fitting")
    print("=" * 80)

    start_time = time.time()

    # 1. Load Data & Taxonomy
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    tax_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "taxonomy_v1.csv"))
    tax_info = {str(r["id"]): (str(r["name"]), str(r["description"])) for _, r in tax_df.iterrows()}

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_raw = train_df["root_code"].fillna("NONE").astype(str).values
    y_indices = np.array([class_to_idx[c] for c in y_raw], dtype=np.int64)
    groups = train_df["question_id"].values
    n_samples = len(train_df)

    print(f"Loaded {n_samples} samples across {len(np.unique(groups))} unique questions.")

    # 2. Load Normalized Multimodal Embeddings
    print("Loading normalized multimodal features (ConvNeXt, Swin, TrOCR, Text, Vision)...")
    cn_norm = normalize(np.load(ART_DIR / "train_convnext_features.npy"), norm="l2")
    sw_norm = normalize(np.load(ART_DIR / "train_swin_features.npy"), norm="l2")
    tr_norm = normalize(np.load(ART_DIR / "train_trocr_features.npy"), norm="l2")
    tx_norm = normalize(np.load(ART_DIR / "train_text_embeddings_384.npy"), norm="l2")
    vi_norm = normalize(np.load(ART_DIR / "train_vision_features.npy"), norm="l2")

    X_all = np.hstack([cn_norm, sw_norm, tr_norm, tx_norm, vi_norm]).astype(np.float32)
    print(f"Multimodal feature matrix: {X_all.shape}")

    # 3. Load MiniLM Text Embeddings for Candidate Reranking
    print("Loading cached MiniLM embeddings...")
    npz = np.load(ART_DIR / "minilm_reranker_text_embeddings.npz")
    emb_q = npz["emb_q"]
    emb_ocr = npz["emb_ocr"]
    emb_comb = npz["emb_comb"]
    emb_tax = npz["emb_tax"]

    ocr_token_sets = [extract_tokens(t) for t in train_df["transcription_steps"]]
    tax_token_sets = {c: extract_tokens(f"{tax_info.get(c, ('', ''))[0]} {tax_info.get(c, ('', ''))[1]}") for c in classes}

    def build_candidate_features(sample_idx: int, cand_c_idx: int, cand_rank: int, p_c: float, p_best: float, prior_p: float) -> List[float]:
        code = classes[cand_c_idx]
        gap = p_c - p_best
        ratio = p_c / max(p_best, 1e-6)

        c_q = float(np.dot(emb_q[sample_idx], emb_tax[cand_c_idx]))
        c_ocr = float(np.dot(emb_ocr[sample_idx], emb_tax[cand_c_idx]))
        c_comb = float(np.dot(emb_comb[sample_idx], emb_tax[cand_c_idx]))
        c_diff = c_ocr - c_q
        j_sim = jaccard(ocr_token_sets[sample_idx], tax_token_sets[code])

        return [p_c, float(cand_rank), gap, ratio, c_q, c_ocr, c_comb, c_diff, j_sim, prior_p]

    # 4. Setup 5-Fold Outer GroupKFold
    outer_gkf = GroupKFold(n_splits=5)
    K_CANDS = 5

    outer_baseline_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    outer_baseline_top3: List[List[str]] = [[] for _ in range(n_samples)]
    outer_reranked_top3: List[List[str]] = [[] for _ in range(n_samples)]

    fold_reports: List[Dict[str, Any]] = []

    print("\nExecuting 5-Fold Outer GroupKFold with Inner Cross-Fitting...")

    for outer_fold, (trn_idx, val_idx) in enumerate(outer_gkf.split(train_df, y_indices, groups=groups), start=1):
        fold_t0 = time.time()
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
        n_val = len(val_idx)

        # Verify zero question leakage
        val_questions = set(val_df["question_id"])
        trn_questions = set(trn_df["question_id"])
        assert len(val_questions & trn_questions) == 0, f"Leakage detected in outer fold {outer_fold}!"

        # ---------------------------------------------------------------------
        # STEP A: Inner 4-Fold Cross-Fitting on trn_df to generate leak-free training features
        # ---------------------------------------------------------------------
        print("  [1/3] Generating training reranker features via 4-fold inner cross-fitting...")
        inner_gkf = GroupKFold(n_splits=4)
        inner_oof_probs = np.zeros((n_trn, n_classes), dtype=np.float64)
        inner_oof_priors = np.zeros((n_trn, n_classes), dtype=np.float64)

        for in_fold, (in_trn, in_val) in enumerate(inner_gkf.split(trn_df, trn_y, groups=trn_groups), start=1):
            in_trn_df = trn_df.iloc[in_trn]
            in_val_df = trn_df.iloc[in_val]

            # In-fold prior strictly on in_trn_df
            prior_in = AdaptiveQuestionPriorEngine(classes=classes).fit(in_trn_df)
            val_prior_in = prior_in.predict_proba(in_val_df).astype(np.float64)
            inner_oof_priors[in_val] = val_prior_in

            # In-fold classifier on in_trn
            clf_in = RidgeClassifier(alpha=10.0, random_state=42)
            clf_in.fit(X_all[trn_idx[in_trn]], trn_y[in_trn])

            dec_val_in = clf_in.decision_function(X_all[trn_idx[in_val]])
            raw_val_in = softmax(dec_val_in, axis=1)

            # Map to full classes if some classes are missing in in_trn
            clf_val_in = np.zeros((len(in_val), n_classes), dtype=np.float64)
            for local_col, global_cls_idx in enumerate(clf_in.classes_):
                clf_val_in[:, global_cls_idx] = raw_val_in[:, local_col]

            blend_in = 0.55 * clf_val_in + 0.45 * val_prior_in

            # In-fold calibration strictly on in_trn
            dec_trn_in = clf_in.decision_function(X_all[trn_idx[in_trn]])
            pred_trn_in = [clf_in.classes_[i] for i in np.argmax(dec_trn_in, axis=1)]
            c_pred = Counter(pred_trn_in)
            c_true = Counter(trn_y[in_trn])

            mult_in = np.ones(n_classes, dtype=np.float64)
            for c_i in range(n_classes):
                if c_i != none_idx:
                    mult_in[c_i] = (max(c_true.get(c_i, 1), 1) / max(c_pred.get(c_i, 1), 1)) ** 0.05
            mult_in[none_idx] = 0.9850

            inner_oof_probs[in_val] = blend_in * mult_in

        # Build clean training candidate pairs from inner cross-fitted predictions
        inner_cand_mat = inner_oof_probs.copy()
        inner_cand_mat[:, none_idx] = -1.0
        inner_cand_indices = np.argsort(-inner_cand_mat, axis=1)[:, :K_CANDS]
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
                    prior_val = float(inner_oof_priors[loc_i, c_idx])
                    feat = build_candidate_features(glob_i, c_idx, rank_pos, p_val, p_best, prior_val)
                    trn_features.append(feat)
                    trn_targets.append(1 if code == true_c else 0)

        X_trn_rerank = np.array(trn_features, dtype=np.float32)
        y_trn_rerank = np.array(trn_targets, dtype=np.int32)
        print(f"  Constructed {len(X_trn_rerank)} clean training candidate examples (Zero prior/calibration leakage)")

        # Train Ranker on clean cross-fitted features
        scaler = StandardScaler()
        X_trn_scaled = scaler.fit_transform(X_trn_rerank)
        ranker = LogisticRegression(C=0.10, penalty="l2", class_weight="balanced", solver="lbfgs", random_state=42)
        ranker.fit(X_trn_scaled, y_trn_rerank)

        # ---------------------------------------------------------------------
        # STEP B: Outer Model Training & Generating Candidates for UNTOUCHED val_df
        # ---------------------------------------------------------------------
        print("  [2/3] Generating predictions on untouched validation questions (V_k)...")
        # Outer prior & classifier strictly on full T_k
        outer_prior_eng = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df)
        val_prior_outer = outer_prior_eng.predict_proba(val_df).astype(np.float64)

        outer_clf = RidgeClassifier(alpha=10.0, random_state=42)
        outer_clf.fit(X_all[trn_idx], trn_y)

        dec_val_out = outer_clf.decision_function(X_all[val_idx])
        raw_val_out = softmax(dec_val_out, axis=1)

        val_clf_outer = np.zeros((len(val_idx), n_classes), dtype=np.float64)
        for local_col, global_cls_idx in enumerate(outer_clf.classes_):
            val_clf_outer[:, global_cls_idx] = raw_val_out[:, local_col]

        val_blend_outer = 0.55 * val_clf_outer + 0.45 * val_prior_outer

        # Outer calibration strictly from T_k
        dec_trn_out = outer_clf.decision_function(X_all[trn_idx])
        pred_trn_outer = [outer_clf.classes_[i] for i in np.argmax(dec_trn_out, axis=1)]
        c_pred_out = Counter(pred_trn_outer)
        c_true_out = Counter(trn_y)

        mult_outer = np.ones(n_classes, dtype=np.float64)
        for c_i in range(n_classes):
            if c_i != none_idx:
                mult_outer[c_i] = (max(c_true_out.get(c_i, 1), 1) / max(c_pred_out.get(c_i, 1), 1)) ** 0.05
        mult_outer[none_idx] = 0.9850

        val_probs_outer = val_blend_outer * mult_outer
        outer_baseline_probs[val_idx] = val_probs_outer

        # Extract baseline top-3 predictions
        val_cand_mat = val_probs_outer.copy()
        val_cand_mat[:, none_idx] = -1.0
        val_cands_out = np.argsort(-val_cand_mat, axis=1)[:, :K_CANDS]

        val_base_top3 = []
        for loc_i in range(n_val):
            r1_is_none = (val_probs_outer[loc_i, none_idx] >= np.max(val_cand_mat[loc_i]))
            c1_code = classes[val_cands_out[loc_i, 0]]
            c2_code = classes[val_cands_out[loc_i, 1]]
            if r1_is_none:
                val_base_top3.append(["NONE", c1_code, c2_code])
            else:
                val_base_top3.append([c1_code, "NONE", c2_code])

        f_base_score = calculate_map3(val_y_raw, val_base_top3)

        # ---------------------------------------------------------------------
        # STEP C: Reranking on UNTOUCHED val_df (Rank 1 strictly fixed)
        # ---------------------------------------------------------------------
        print("  [3/3] Evaluating candidate reranker on untouched holdout V_k...")
        val_reranked_top3 = [list(r) for r in val_base_top3]
        f_swaps = 0
        f_helped = 0
        f_hurt = 0
        f_neutral = 0

        # Swap thresholds: tight probability gap (< 0.025) and confident ranker margin (>= 0.10)
        EPS_SWAP = 0.025
        MARGIN_SWAP = 0.10

        for loc_i in range(n_val):
            glob_i = val_idx[loc_i]
            true_label = val_y_raw[loc_i]
            r1_is_none = (val_base_top3[loc_i][0] == "NONE")

            if r1_is_none:
                cands = val_cands_out[loc_i]
                p_c1 = float(val_probs_outer[loc_i, cands[0]])
                p_c2 = float(val_probs_outer[loc_i, cands[1]])
                gap_12 = p_c1 - p_c2

                cand_feats = []
                for rank_pos, c_idx in enumerate(cands, start=1):
                    p_val = float(val_probs_outer[loc_i, c_idx])
                    prior_val = float(val_prior_outer[loc_i, c_idx])
                    feat = build_candidate_features(glob_i, c_idx, rank_pos, p_val, p_c1, prior_val)
                    cand_feats.append(feat)

                cand_scaled = scaler.transform(np.array(cand_feats))
                cand_scores = ranker.predict_proba(cand_scaled)[:, 1]
                ranker_margin = float(cand_scores[1] - cand_scores[0])

                if 0 < gap_12 < EPS_SWAP and ranker_margin >= MARGIN_SWAP:
                    c1_code = classes[cands[0]]
                    c2_code = classes[cands[1]]
                    val_reranked_top3[loc_i][1] = c2_code
                    val_reranked_top3[loc_i][2] = c1_code
                    f_swaps += 1

                    if true_label == c2_code:
                        f_helped += 1
                    elif true_label == c1_code:
                        f_hurt += 1
                    else:
                        f_neutral += 1

        f_new_score = calculate_map3(val_y_raw, val_reranked_top3)
        delta_f = f_new_score - f_base_score

        for loc_i, glob_i in enumerate(val_idx):
            outer_baseline_top3[glob_i] = val_base_top3[loc_i]
            outer_reranked_top3[glob_i] = val_reranked_top3[loc_i]

        fold_duration = time.time() - fold_t0
        print(f"  Outer Fold {outer_fold} Result (Untouched Holdout):")
        print(f"    Baseline MAP@3: {f_base_score:.5f}")
        print(f"    Reranked MAP@3: {f_new_score:.5f} ({delta_f:+.5f})")
        print(f"    Swaps: {f_swaps} (Helped: {f_helped}, Hurt: {f_hurt}, Neutral: {f_neutral}) | Time: {fold_duration:.1f}s")

        fold_reports.append({
            "fold": outer_fold,
            "val_samples": n_val,
            "baseline_map3": round(f_base_score, 5),
            "reranked_map3": round(f_new_score, 5),
            "delta": round(delta_f, 5),
            "swaps": f_swaps,
            "helped": f_helped,
            "hurt": f_hurt,
            "neutral": f_neutral,
        })

    # 5. Global Out-of-Fold Evaluation
    total_base = calculate_map3(y_raw.tolist(), outer_baseline_top3)
    total_rerank = calculate_map3(y_raw.tolist(), outer_reranked_top3)
    net_delta = total_rerank - total_base

    total_swaps = sum(f["swaps"] for f in fold_reports)
    total_helped = sum(f["helped"] for f in fold_reports)
    total_hurt = sum(f["hurt"] for f in fold_reports)
    total_neutral = sum(f["neutral"] for f in fold_reports)
    per_fold_deltas = [f["delta"] for f in fold_reports]

    # Candidate Retrieval Recall Audit on genuine held-out questions
    misc_rows = [i for i in range(n_samples) if y_raw[i] != "NONE"]
    outer_cand_mat = outer_baseline_probs.copy()
    outer_cand_mat[:, none_idx] = -1.0
    sorted_outer_cands = np.argsort(-outer_cand_mat, axis=1)

    rec_table = {}
    for k in [1, 2, 3, 5, 8, 10]:
        hits = sum(1 for i in misc_rows if y_raw[i] in [classes[idx] for idx in sorted_outer_cands[i, :k]])
        rec_table[f"Recall@{k}"] = {"hits": hits, "total": len(misc_rows), "rate": round(hits / len(misc_rows), 4)}

    print("\n" + "=" * 80)
    print("FINAL 100% LEAK-FREE GENERALIZATION REPORT (1,500 ROWS)")
    print("=" * 80)
    print(f"Held-Out Baseline MAP@3  : {total_base:.5f}")
    print(f"Held-Out Reranked MAP@3  : {total_rerank:.5f}")
    print(f"Net Generalization Delta : {net_delta:+.5f}")
    print(f"Per-Fold Deltas          : {per_fold_deltas}")
    print(f"Total Swaps across Folds : {total_swaps} (Helped: {total_helped}, Hurt: {total_hurt}, Neutral: {total_neutral})")
    if total_helped + total_hurt > 0:
        print(f"Active Swap Precision    : {total_helped / (total_helped + total_hurt):.1%}")

    print("\nHonest Candidate Retrieval Recall (100% Unseen Questions):")
    for k, v in rec_table.items():
        print(f"  {k:10s}: {v['hits']:4d} / {v['total']} ({v['rate']:.1%})")

    # Gate Evaluation
    all_non_negative = all(d >= -1e-7 for d in per_fold_deltas)
    gate_passed = (net_delta >= 0.00020) and all_non_negative

    print("\nGATE STATUS: " + ("PASSED" if gate_passed else "UNVALIDATED / REJECTED"))
    print("=" * 80)

    # 6. Save Persistent Result Artifact
    report_artifact = {
        "benchmark_name": "honest_question_grouped_candidate_reranker",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "leakage_hygiene": {
            "outer_splits": "5-Fold GroupKFold strictly on question_id (0.0% question overlap)",
            "inner_cross_fitting": "4-Fold GroupKFold within T_k for prior, calibration, and base probs",
            "precomputed_oofs_used": False,
            "rank1_invariant": "Strictly locked (0 Rank-1 mutations)"
        },
        "overall_results": {
            "baseline_map3": round(total_base, 5),
            "reranked_map3": round(total_rerank, 5),
            "net_delta": round(net_delta, 5),
            "per_fold_deltas": per_fold_deltas,
            "total_swaps": total_swaps,
            "helped_swaps": total_helped,
            "hurt_swaps": total_hurt,
            "neutral_swaps": total_neutral,
            "swap_precision": round(total_helped / max(total_helped + total_hurt, 1), 4),
            "gate_passed": gate_passed
        },
        "candidate_retrieval_recall": rec_table,
        "fold_details": fold_reports,
        "total_runtime_seconds": round(time.time() - start_time, 2)
    }

    report_path = ART_DIR / "honest_question_grouped_reranker_report.json"
    with open(report_path, "w") as f:
        json.dump(report_artifact, f, indent=2)
    print(f"\n[Saved Persistent Result Artifact] {report_path}")

if __name__ == "__main__":
    main()

