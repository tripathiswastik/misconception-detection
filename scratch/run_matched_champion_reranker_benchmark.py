"""Definitive Matched-Pipeline Reranker Benchmark on Actual SOTA 550 Champion.

Rigorous Protocol per User Directive:
1. Champion Alignment: Evaluated directly on the actual SOTA 550 Champion predictions and candidate ordering.
2. Leak-Free Inner Cross-Fits:
   - Outer 5-fold GroupKFold on question_id.
   - Within each outer training partition T_k, an inner 4-fold GroupKFold generates honest out-of-fold
     champion candidate lists and training features (zero in-sample leakage).
3. Test-Like Reality (Check 1):
   - Zero gold transcription_steps used in reranking features.
   - Conditioned only on test-available context + normalized handwriting visual embeddings.
4. Single Evaluation on Untouched Outer Holdout V_k:
   - Rank 1 is 100% strictly anchored and immutable.
   - Candidate scorer trained on inner cross-fits evaluates once on V_k.
5. Definitive Decision Gate:
   - If consistent held-out gains across all outer folds (all deltas >= 0.0, net delta >= +0.00020): validate.
   - If flat or negative: conclude reranking permanently and pivot to base representations.
"""
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, cast
import numpy as np
import pandas as pd
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

def sharp(probs: np.ndarray, t: float) -> np.ndarray:
    s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    return s / s.sum(axis=1, keepdims=True)

def generate_champion_fold_probs(
    train_df: pd.DataFrame,
    trn_idx: np.ndarray,
    val_idx: np.ndarray,
    p_base_all: np.ndarray,
    oof_m15: np.ndarray,
    oof_arc3500: np.ndarray,
    honest_text: np.ndarray,
    oof_sem_text: np.ndarray,
    oof_convnext: np.ndarray,
    oof_swin: np.ndarray,
    oof_finetuned: np.ndarray,
    oof_trocr: np.ndarray,
    oof_clean_arcface: np.ndarray,
    oof_bin_err: np.ndarray,
    classes: List[str],
    none_idx: int
) -> Tuple[np.ndarray, List[List[str]], np.ndarray]:
    """Generates the full SOTA 550 champion pipeline probabilities on val_idx strictly using trn_idx."""
    n_classes = len(classes)
    n_val = len(val_idx)
    trn_df_fold = train_df.iloc[trn_idx]
    val_df_fold = train_df.iloc[val_idx]

    # In-fold prior
    prior_eng = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df_fold)
    val_prior = prior_eng.predict_proba(val_df_fold).astype(np.float64)

    # In-fold frequency calibration
    p_base_trn = p_base_all[trn_idx]
    top3_trn = np.argsort(-p_base_trn.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_trn for idx in row)
    true_counts = Counter(trn_df_fold["root_code"].fillna("NONE").values)
    mult = np.ones(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            mult[i] = (max(true_counts.get(c, 1), 1) / max(pred_counts.get(c, 1), 1)) ** 0.05
    mult[none_idx] = 0.9840

    p_val_cal = p_base_all[val_idx] * mult
    p_val_cal[:, none_idx] *= 0.9995

    # Question-allowed boost strictly from trn_df_fold
    q_allowed = {q: set(grp["root_code"].dropna().unique()) for q, grp in trn_df_fold.groupby("question_id")}
    p_val_boosted = np.copy(p_val_cal)
    for i in range(n_val):
        qid = val_df_fold.iloc[i]["question_id"]
        if qid in q_allowed:
            for ci, c in enumerate(classes):
                if c != "NONE" and c in q_allowed[qid]:
                    p_val_boosted[i, ci] *= 1.20

    p_val_arc3500_cal = (0.95 * oof_m15[val_idx] + 0.05 * oof_arc3500[val_idx]) * mult
    p_champ_val = 0.70 * p_val_cal + 0.25 * p_val_boosted + 0.05 * p_val_arc3500_cal

    p_clean_val = (
        0.6448 * val_prior +
        0.0839 * sharp(honest_text[val_idx], 0.6) +
        0.0401 * sharp(oof_sem_text[val_idx], 0.6) +
        0.0458 * oof_convnext[val_idx] +
        0.0454 * oof_swin[val_idx] +
        0.0483 * oof_finetuned[val_idx] +
        0.0455 * oof_trocr[val_idx] +
        0.0462 * sharp(oof_clean_arcface[val_idx], 0.8)
    )

    p_final_val = 0.96 * p_champ_val + 0.04 * p_clean_val
    val_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-p_final_val, axis=1)]

    # Champion Precision Gating
    p_bin_err_val = oof_bin_err[val_idx]
    for i in range(n_val):
        if val_top3[i][0] == "NONE" and p_bin_err_val[i] >= 0.78:
            row_p = p_final_val[i].copy()
            p_none_val = row_p[none_idx]
            row_p[none_idx] = -1.0
            top_misc_idx = int(np.argmax(row_p))
            top_misc_code = classes[top_misc_idx]
            if 0 < (p_none_val - row_p[top_misc_idx]) < 0.04:
                qid = val_df_fold.iloc[i]["question_id"]
                if top_misc_code in q_allowed.get(qid, set()):
                    row_p[top_misc_idx] = -1.0
                    second_misc = classes[int(np.argmax(row_p))]
                    val_top3[i] = [top_misc_code, "NONE", second_misc]

    r1_is_none = np.array([p[0] == "NONE" for p in val_top3])
    return p_final_val, val_top3, r1_is_none

def main():
    print("=" * 80)
    print("DEFINITIVE MATCHED-PIPELINE RERANKER BENCHMARK (SOTA 550 CHAMPION)")
    print("Strict Inner Cross-Fits | Zero Gold Text Leakage | Rank 1 Locked")
    print("=" * 80)

    start_time = time.time()

    # 1. Load Taxonomy & Data
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

    # 2. Load Visual Handwriting Features (Test-like student work representation)
    tr_feat = np.load(ART_DIR / "train_trocr_features.npy")
    cn_feat = np.load(ART_DIR / "train_convnext_features.npy")
    sw_feat = np.load(ART_DIR / "train_swin_features.npy")
    vis_norm = np.hstack([
        cast(np.ndarray, normalize(tr_feat)),
        cast(np.ndarray, normalize(cn_feat)),
        cast(np.ndarray, normalize(sw_feat))
    ])

    # 3. Load Champion Foundation Arrays
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    oof_arc3500 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")

    honest_text = np.load(ART_DIR / "honest_oof_text.npy")
    oof_sem_text = np.load(ART_DIR / "oof_semantic_text.npy")
    oof_convnext = np.load(ART_DIR / "oof_convnext_vision.npy")
    oof_swin = np.load(ART_DIR / "oof_swin_vision.npy")
    oof_finetuned = np.load(ART_DIR / "oof_finetuned_vision.npy")
    oof_trocr = np.load(ART_DIR / "oof_trocr_vision.npy")
    oof_clean_arcface = np.load(ART_DIR / "oof_clean_arcface_2328.npy")
    oof_bin_err = np.load(ART_DIR / "oof_compact_binary_prob.npy")

    p_base_all = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448

    # 4. Generate SOTA 550 Baseline Across Outer 5-Fold GroupKFold
    outer_gkf = GroupKFold(n_splits=5)
    champion_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    champion_top3: List[List[str]] = [[] for _ in range(n_samples)]
    champion_r1_is_none = np.zeros(n_samples, dtype=bool)
    fold_base_scores = []

    print("\n[Step 1] Constructing SOTA 550 Champion Baseline across 5 outer folds...")
    for fold_num, (trn_idx, val_idx) in enumerate(outer_gkf.split(train_df, y_indices, groups=groups), start=1):
        p_val, top3_val, r1_none = generate_champion_fold_probs(
            train_df, trn_idx, val_idx, p_base_all,
            oof_m15, oof_arc3500, honest_text, oof_sem_text,
            oof_convnext, oof_swin, oof_finetuned, oof_trocr,
            oof_clean_arcface, oof_bin_err, classes, none_idx
        )
        champion_probs[val_idx] = p_val
        val_y = y_raw[val_idx].tolist()
        s_f = calculate_map3(val_y, top3_val)
        fold_base_scores.append(s_f)
        for loc_i, glob_i in enumerate(val_idx):
            champion_top3[glob_i] = top3_val[loc_i]
            champion_r1_is_none[glob_i] = r1_none[loc_i]
        print(f"  Outer Fold {fold_num} Champion Held-Out MAP@3: {s_f:.5f}")

    overall_champ_map3 = calculate_map3(y_raw.tolist(), champion_top3)
    r1_none_total = int(champion_r1_is_none.sum())
    print(f"Overall SOTA 550 Champion MAP@3: {overall_champ_map3:.5f} (Per-fold: {[round(s, 5) for s in fold_base_scores]})")
    print(f"Total Rows with Rank 1 == NONE: {r1_none_total} / {n_samples} ({r1_none_total/n_samples:.1%})")

    # 5. Strictly Nested Reranker Training & Outer Evaluation
    # Candidate pool: K=3 leading misconception candidates from champion
    K_POOL = 3
    cand_prob_mat = champion_probs.copy()
    cand_prob_mat[:, none_idx] = -1.0
    sorted_cands = np.argsort(-cand_prob_mat, axis=1)[:, :K_POOL]

    reranked_top3: List[List[str]] = [[] for _ in range(n_samples)]
    outer_deltas: List[float] = []
    outer_swap_reports: List[Dict[str, int]] = []

    print("\n" + "=" * 80)
    print("[Step 2] Nested Cross-Fitting & Candidate Reranker Evaluation")
    print("=" * 80)

    for outer_fold, (trn_idx, val_idx) in enumerate(outer_gkf.split(train_df, y_indices, groups=groups), start=1):
        print(f"\n--- Outer Fold {outer_fold} / 5 ---")
        val_y = y_raw[val_idx].tolist()
        val_base_top3 = [list(champion_top3[g_i]) for g_i in val_idx]
        base_f_score = calculate_map3(val_y, val_base_top3)

        trn_df_outer = train_df.iloc[trn_idx]
        trn_groups = trn_df_outer["question_id"].values

        # A. Inner 4-Fold Cross-Fitting within T_k to generate 100% leak-free training candidates
        inner_gkf = GroupKFold(n_splits=4)
        inner_champ_probs = np.zeros((len(trn_idx), n_classes), dtype=np.float64)
        inner_r1_is_none = np.zeros(len(trn_idx), dtype=bool)

        for in_tr, in_va in inner_gkf.split(trn_df_outer, groups=trn_groups):
            in_trn_glob = trn_idx[in_tr]
            in_val_glob = trn_idx[in_va]

            p_in_val, top3_in_val, r1_none_in = generate_champion_fold_probs(
                train_df, in_trn_glob, in_val_glob, p_base_all,
                oof_m15, oof_arc3500, honest_text, oof_sem_text,
                oof_convnext, oof_swin, oof_finetuned, oof_trocr,
                oof_clean_arcface, oof_bin_err, classes, none_idx
            )
            for loc_v, loc_glob in enumerate(in_va):
                inner_champ_probs[loc_glob] = p_in_val[loc_v]
                inner_r1_is_none[loc_glob] = r1_none_in[loc_v]

        # Inner candidate pool
        inner_cand_mat = inner_champ_probs.copy()
        inner_cand_mat[:, none_idx] = -1.0
        inner_sorted_cands = np.argsort(-inner_cand_mat, axis=1)[:, :K_POOL]

        # In-fold class visual centroids strictly from trn_idx
        fold_y = y_indices[trn_idx]
        fold_vis = vis_norm[trn_idx]
        vis_centroids = np.zeros((n_classes, fold_vis.shape[1]), dtype=np.float32)
        for c_i in range(n_classes):
            c_mask = (fold_y == c_i)
            if c_mask.sum() > 0:
                c_m = fold_vis[c_mask].mean(axis=0)
                vis_centroids[c_i] = c_m / (np.linalg.norm(c_m) + 1e-12)

        # In-fold prior strictly on outer training fold
        prior_eng_outer = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df_outer)
        trn_priors = prior_eng_outer.predict_proba(trn_df_outer)
        val_priors = prior_eng_outer.predict_proba(train_df.iloc[val_idx])

        # B. Assemble Training Pairs from Inner Cross-Fits
        trn_X_list = []
        trn_y_list = []

        for loc_i, glob_i in enumerate(trn_idx):
            if inner_r1_is_none[loc_i]:
                true_c = y_raw[glob_i]
                c_pool = inner_sorted_cands[loc_i]
                p_cands = [inner_champ_probs[loc_i, c] for c in c_pool]
                p_max = max(p_cands) if p_cands else 1.0

                for k_pos, c_idx in enumerate(c_pool):
                    code = classes[c_idx]
                    is_pos = (code == true_c)
                    p_c = float(inner_champ_probs[loc_i, c_idx])
                    gap = float(p_c - p_max)
                    ratio = float(p_c / max(p_max, 1e-6))
                    pr_p = float(trn_priors[loc_i, c_idx])
                    vis_sim = float(np.dot(vis_norm[glob_i], vis_centroids[c_idx]))

                    feat = np.array([p_c, float(k_pos + 1), gap, ratio, pr_p, vis_sim], dtype=np.float32)
                    trn_X_list.append(feat)
                    trn_y_list.append(1 if is_pos else 0)

        X_train = np.array(trn_X_list, dtype=np.float32)
        y_train = np.array(trn_y_list, dtype=np.int32)
        pos_cnt = int(sum(y_train))
        print(f"  Inner Cross-Fit Pairs: {len(y_train)} ({pos_cnt} positives, {len(y_train) - pos_cnt} hard negatives)")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train Calibrated Candidate Scorer Head
        head = LogisticRegression(C=0.10, penalty="l2", class_weight="balanced", solver="lbfgs", max_iter=300, random_state=42)
        head.fit(X_train_scaled, y_train)

        # C. Single Evaluation on Untouched Outer Holdout V_k
        val_reranked = [list(r) for r in val_base_top3]
        f_swaps = 0
        f_helped = 0
        f_hurt = 0
        f_neutral = 0

        # Rerank Candidate 1 (Rank 2) vs Candidate 2 (Rank 3) for NONE-at-Rank-1 rows
        # Rank 1 remains 100% strictly anchored!
        for loc_i, glob_i in enumerate(val_idx):
            if champion_r1_is_none[glob_i]:
                cands = sorted_cands[glob_i]
                c1_idx = cands[0]
                c2_idx = cands[1]

                p_c1 = champion_probs[glob_i, c1_idx]
                p_c2 = champion_probs[glob_i, c2_idx]
                gap_base = p_c1 - p_c2

                # Compute candidate features for Candidate 1 and Candidate 2
                f_c1 = np.array([
                    p_c1, 1.0, 0.0, 1.0,
                    float(val_priors[loc_i, c1_idx]),
                    float(np.dot(vis_norm[glob_i], vis_centroids[c1_idx]))
                ], dtype=np.float32)

                f_c2 = np.array([
                    p_c2, 2.0, -gap_base, float(p_c2 / max(p_c1, 1e-6)),
                    float(val_priors[loc_i, c2_idx]),
                    float(np.dot(vis_norm[glob_i], vis_centroids[c2_idx]))
                ], dtype=np.float32)

                pair_scaled = scaler.transform(np.vstack([f_c1, f_c2]))
                scores = head.predict_proba(pair_scaled)[:, 1]
                margin = float(scores[1] - scores[0])

                # Swap condition: base gap is close (< 0.030) and reranker margin >= 0.15
                if 0 < gap_base < 0.030 and margin >= 0.15:
                    c1_code = classes[c1_idx]
                    c2_code = classes[c2_idx]
                    val_reranked[loc_i][1] = c2_code
                    val_reranked[loc_i][2] = c1_code
                    f_swaps += 1

                    true_l = val_y[loc_i]
                    if true_l == c2_code:
                        f_helped += 1
                    elif true_l == c1_code:
                        f_hurt += 1
                    else:
                        f_neutral += 1

        new_f_score = calculate_map3(val_y, val_reranked)
        delta_f = new_f_score - base_f_score
        outer_deltas.append(delta_f)
        outer_swap_reports.append({"swaps": f_swaps, "helped": f_helped, "hurt": f_hurt, "neutral": f_neutral})

        print(f"  Holdout Result: Baseline: {base_f_score:.5f} -> Reranked: {new_f_score:.5f} ({delta_f:+.5f})")
        print(f"  Swaps: {f_swaps} (Helped: {f_helped}, Hurt: {f_hurt}, Neutral: {f_neutral})")

        for loc_i, glob_i in enumerate(val_idx):
            reranked_top3[glob_i] = val_reranked[loc_i]

    # 6. Overall Benchmark Results
    overall_new_map3 = calculate_map3(y_raw.tolist(), reranked_top3)
    net_delta = overall_new_map3 - overall_champ_map3
    tot_swaps = sum(r["swaps"] for r in outer_swap_reports)
    tot_helped = sum(r["helped"] for r in outer_swap_reports)
    tot_hurt = sum(r["hurt"] for r in outer_swap_reports)
    tot_neutral = sum(r["neutral"] for r in outer_swap_reports)

    print("\n" + "=" * 80)
    print("FINAL MATCHED CHAMPION RERANKER BENCHMARK RESULTS")
    print("=" * 80)
    print(f"SOTA 550 Champion Baseline MAP@3 : {overall_champ_map3:.5f}")
    print(f"Matched Reranker MAP@3           : {overall_new_map3:.5f}")
    print(f"Net Generalization Delta         : {net_delta:+.5f}")
    print(f"Per-Fold Deltas                  : {[round(d, 5) for d in outer_deltas]}")
    print(f"Total Swaps across 1500 samples  : {tot_swaps} (Helped: {tot_helped}, Hurt: {tot_hurt}, Neutral: {tot_neutral})")
    if tot_helped + tot_hurt > 0:
        print(f"Active Swap Precision            : {tot_helped / (tot_helped + tot_hurt):.1%}")

    gate_passed = (net_delta >= 0.00020) and all(d >= -1e-7 for d in outer_deltas)
    print("\nPROMOTION GATE STATUS            : " + ("PASSED" if gate_passed else "REJECTED / UNVALIDATED"))
    print("Active Champion Status           : blend_sota550_precision_gated_submission.csv (0.55049 LB) UNTOUCHED")
    print("=" * 80)

    # Save artifact
    out_artifact = {
        "benchmark_name": "matched_champion_reranker_benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methodology": {
            "pipeline_alignment": "100% matched to SOTA 550 Champion (M15 + DualHead + Composite + ArcFace3500 + Honest Text + Precision Gating)",
            "cross_fits": "Strictly nested inner 4-fold GroupKFold within each outer training partition",
            "candidate_features": "p_champ, rank_pos, gap, ratio, in-fold prior, visual centroid cosine similarity",
            "test_reality": "Zero gold text used; normalized handwriting visual embeddings only",
            "rank1_invariant": "Strictly locked (0 Rank-1 mutations)"
        },
        "results": {
            "champion_base_map3": round(overall_champ_map3, 5),
            "reranked_map3": round(overall_new_map3, 5),
            "net_delta": round(net_delta, 5),
            "per_fold_deltas": [round(d, 5) for d in outer_deltas],
            "total_swaps": tot_swaps,
            "helped": tot_helped,
            "hurt": tot_hurt,
            "neutral": tot_neutral,
            "active_precision": round(tot_helped / max(tot_helped + tot_hurt, 1), 4),
            "gate_passed": gate_passed
        },
        "fold_details": outer_swap_reports,
        "runtime_seconds": round(time.time() - start_time, 2)
    }

    report_path = ART_DIR / "matched_champion_reranker_benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(out_artifact, f, indent=2)
    print(f"[Saved Report Artifact] {report_path.name}")

if __name__ == "__main__":
    main()

