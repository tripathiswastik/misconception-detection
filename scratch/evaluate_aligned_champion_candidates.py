"""Aligned SOTA 550 Champion Pipeline on Question-Held-Out Folds.

Faithfully mirrors the exact test-time champion recipe (generate_precision_gated_submission.py):
1. Outer 5-fold GroupKFold strictly on question_id (0.0% question leakage).
2. For each outer fold:
   - Fits frequency calibration strictly on outer-training rows (T_k).
   - Fits question-allowed sets strictly on T_k.
   - Fits AdaptiveQuestionPrior strictly on T_k.
   - Evaluates M15 + DualHead + Composite + ArcFace3500 + Clean blend on untouched V_k.
   - Applies precision gating with compact binary detector on V_k.
3. Computes the honest champion question-held-out baseline MAP@3.
4. Measures honest Candidate Retrieval Recall@K (K = 1, 2, 3, 5, 8, 10) on the actual champion candidates.
5. Saves the champion candidates and probabilities for the candidate reranker.
"""
from collections import Counter
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

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

def main():
    print("=" * 80)
    print("ALIGNED SOTA 550 CHAMPION EVALUATION ON QUESTION-HELDOUT FOLDS")
    print("Exact Champion Recipe (M15 + DualHead + Composite + Prior + Gating)")
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

    # 1. Load All Champion OOF Arrays
    print("Loading champion component OOF arrays...")
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    oof_arc3500 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")

    oof_text = np.load(ART_DIR / "oof_text.npy")
    oof_semantic_text = np.load(ART_DIR / "oof_semantic_text.npy")
    oof_convnext = np.load(ART_DIR / "oof_convnext_vision.npy")
    oof_swin = np.load(ART_DIR / "oof_swin_vision.npy")
    oof_finetuned = np.load(ART_DIR / "oof_finetuned_vision.npy")
    oof_trocr = np.load(ART_DIR / "oof_trocr_vision.npy")
    oof_clean_arcface = np.load(ART_DIR / "oof_clean_arcface_2328.npy")
    oof_bin_err = np.load(ART_DIR / "oof_compact_binary_prob.npy")

    # Outer 5-Fold GroupKFold
    outer_gkf = GroupKFold(n_splits=5)

    champion_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    champion_top3: List[List[str]] = [[] for _ in range(n_samples)]
    champion_r1_is_none = np.zeros(n_samples, dtype=bool)
    fold_scores = []
    gated_flips_total = 0

    for fold, (trn_idx, val_idx) in enumerate(outer_gkf.split(train_df, y_indices, groups=groups), start=1):
        trn_df = train_df.iloc[trn_idx]
        val_df = train_df.iloc[val_idx]
        trn_y_raw = y_raw[trn_idx]
        val_y_raw = y_raw[val_idx].tolist()
        n_val = len(val_idx)

        # A. Base probabilities (M15 + DualHead + Composite)
        p_base_trn = 0.95 * oof_m15[trn_idx] + 0.02 * oof_dual448[trn_idx] + 0.03 * oof_comp448[trn_idx]
        p_base_val = 0.95 * oof_m15[val_idx] + 0.02 * oof_dual448[val_idx] + 0.03 * oof_comp448[val_idx]

        # B. Frequency calibration strictly from trn_idx
        top3_trn = np.argsort(-p_base_trn.astype(np.float64), axis=1)[:, :3]
        pred_counts = Counter(classes[idx] for row in top3_trn for idx in row)
        true_counts = Counter(trn_df["root_code"].fillna("NONE").values)
        mult = np.ones(n_classes, dtype=np.float64)
        for i, c in enumerate(classes):
            if c != "NONE":
                pc = max(pred_counts.get(c, 1), 1)
                tc = max(true_counts.get(c, 1), 1)
                mult[i] = (tc / pc) ** 0.05
        mult[none_idx] = 0.9840

        p_val_cal = p_base_val * mult
        p_val_cal[:, none_idx] *= 0.9995

        # C. Question-allowed boost strictly from trn_df
        q_allowed = {q: set(grp["root_code"].dropna().unique()) for q, grp in trn_df.groupby("question_id")}
        p_val_boosted = np.copy(p_val_cal)
        for i in range(n_val):
            qid = val_df.iloc[i]["question_id"]
            if qid in q_allowed:
                for ci, c in enumerate(classes):
                    if c != "NONE" and c in q_allowed[qid]:
                        p_val_boosted[i, ci] *= 1.20

        # D. ArcFace 3500 component
        p_val_arc3500_cal = (0.95 * oof_m15[val_idx] + 0.05 * oof_arc3500[val_idx]) * mult
        p_champ_val = 0.70 * p_val_cal + 0.25 * p_val_boosted + 0.05 * p_val_arc3500_cal

        # E. In-fold Adaptive Prior strictly on trn_df
        prior_engine = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df)
        val_prior_probs = prior_engine.predict_proba(val_df)

        p_clean_val = (
            0.6448 * val_prior_probs +
            0.0839 * sharp(oof_text[val_idx], 0.6) +
            0.0401 * sharp(oof_semantic_text[val_idx], 0.6) +
            0.0458 * oof_convnext[val_idx] +
            0.0454 * oof_swin[val_idx] +
            0.0483 * oof_finetuned[val_idx] +
            0.0455 * oof_trocr[val_idx] +
            0.0462 * sharp(oof_clean_arcface[val_idx], 0.8)
        )

        p_final_val = 0.96 * p_champ_val + 0.04 * p_clean_val
        champion_probs[val_idx] = p_final_val

        # F. Initial top-3 predictions
        val_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-p_final_val, axis=1)]

        # G. Champion Precision Gating (tau=0.78, eps=0.04)
        p_bin_err_val = oof_bin_err[val_idx]
        f_flips = 0
        for i in range(n_val):
            orig_r1 = val_top3[i][0]
            if orig_r1 == "NONE" and p_bin_err_val[i] >= 0.78:
                row_p = p_final_val[i].copy()
                p_none_val = row_p[none_idx]
                row_p[none_idx] = -1.0
                top_misc_idx = int(np.argmax(row_p))
                top_misc_code = classes[top_misc_idx]
                p_misc_val = row_p[top_misc_idx]
                gap = p_none_val - p_misc_val

                if 0 < gap < 0.04:
                    qid = val_df.iloc[i]["question_id"]
                    if top_misc_code in q_allowed.get(qid, set()):
                        row_p[top_misc_idx] = -1.0
                        second_misc_idx = int(np.argmax(row_p))
                        second_misc_code = classes[second_misc_idx]
                        val_top3[i] = [top_misc_code, "NONE", second_misc_code]
                        f_flips += 1

        score_f = calculate_map3(val_y_raw, val_top3)
        fold_scores.append(score_f)
        gated_flips_total += f_flips

        for loc_i, glob_i in enumerate(val_idx):
            champion_top3[glob_i] = val_top3[loc_i]
            champion_r1_is_none[glob_i] = (val_top3[loc_i][0] == "NONE")

        print(f"Fold {fold} Held-Out MAP@3: {score_f:.5f} (Precision-gated flips: {f_flips})")

    overall_map3 = calculate_map3(y_raw.tolist(), champion_top3)
    print("\n" + "=" * 80)
    print(f"ALIGNED CHAMPION HELD-OUT MAP@3 : {overall_map3:.5f}")
    print(f"Per-Fold Scores                  : {[round(s, 5) for s in fold_scores]}")
    print(f"Total Precision Gating Flips     : {gated_flips_total}")
    print("=" * 80)

    # Save aligned champion arrays for reranker
    np.save(ART_DIR / "oof_aligned_champion_probs.npy", champion_probs)
    print(f"Saved aligned champion probabilities: {ART_DIR / 'oof_aligned_champion_probs.npy'}")

    # -------------------------------------------------------------------------
    # 2. Honest Candidate Retrieval Recall@K on ACTUAL Champion Candidates
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("HONEST CANDIDATE RETRIEVAL RECALL@K (ACTUAL CHAMPION PIPELINE)")
    print("Evaluated strictly on held-out questions across all 1,500 rows")
    print("=" * 80)

    misc_rows = [i for i in range(n_samples) if y_raw[i] != "NONE"]
    print(f"Total labeled student traces: {n_samples}")
    print(f"Traces where student has a misconception: {len(misc_rows)}")

    # Extract champion candidate matrix (excluding NONE)
    champ_cand_matrix = champion_probs.copy()
    champ_cand_matrix[:, none_idx] = -1.0
    sorted_champ_cands = np.argsort(-champ_cand_matrix, axis=1)

    # Save top-8 candidates for reranking
    champ_top8_cands = sorted_champ_cands[:, :8]
    np.save(ART_DIR / "oof_aligned_champion_top8_cands.npy", champ_top8_cands)

    for k in [1, 2, 3, 5, 8, 10]:
        hits = 0
        for i in misc_rows:
            true_label = y_raw[i]
            top_k_codes = [classes[idx] for idx in sorted_champ_cands[i, :k]]
            if true_label in top_k_codes:
                hits += 1
        rec = hits / len(misc_rows)
        print(f"  Champion Candidate Recall@{k:2d}: {hits:4d} / {len(misc_rows)} ({rec:.1%})")

    # Position breakdown for true misconception
    print("\nExact position of true misconception in Champion's candidate ranking:")
    pos_counts = Counter()
    for i in misc_rows:
        true_label = y_raw[i]
        cands_all = [classes[idx] for idx in sorted_champ_cands[i]]
        if true_label in cands_all:
            rank_pos = cands_all.index(true_label) + 1
            pos_counts[rank_pos] += 1
        else:
            pos_counts[-1] += 1

    for pos in range(1, 9):
        cnt = pos_counts[pos]
        print(f"  Candidate Pos {pos}: {cnt:4d} ({cnt / len(misc_rows):.1%})")
    outside_8 = sum(pos_counts[p] for p in pos_counts if p > 8 or p == -1)
    print(f"  Outside Top 8  : {outside_8:4d} ({outside_8 / len(misc_rows):.1%})")

    # Subgroup where Champion predicted NONE at Rank 1
    r1_none_rows = [i for i in misc_rows if champion_r1_is_none[i]]
    print(f"\nSubgroup: Champion predicted NONE at Rank 1 while true label is a misconception ({len(r1_none_rows)} rows):")
    for k in [1, 2, 3, 5, 8, 10]:
        hits = 0
        for i in r1_none_rows:
            true_label = y_raw[i]
            top_k_codes = [classes[idx] for idx in sorted_champ_cands[i, :k]]
            if true_label in top_k_codes:
                hits += 1
        rec = hits / len(r1_none_rows)
        print(f"  Champion Recall@{k:2d} on 'NONE-at-Rank-1' rows: {hits:4d} / {len(r1_none_rows)} ({rec:.1%})")

    print("\nPosition of true misconception on 'NONE-at-Rank-1' rows:")
    for pos in range(1, 9):
        cnt = sum(1 for i in r1_none_rows if classes[sorted_champ_cands[i, pos - 1]] == y_raw[i])
        print(f"  Candidate Pos {pos}: {cnt:4d} ({cnt / len(r1_none_rows):.1%})")

    print("=" * 80)

if __name__ == "__main__":
    main()

