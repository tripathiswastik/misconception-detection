"""100% Strictly Nested Question-Held-Out Candidate Reranker.

Enforces zero leakage threshold selection:
1. Outer 5-fold GroupKFold on question_id.
2. For each outer fold k:
   - Within T_k (outer training):
     * Run inner 4-fold GroupKFold on questions.
     * Inner ranker cross-fitting to generate inner candidate scores.
     * Inner grid search over (eps, margin) evaluating exclusively on inner held-out folds.
     * Select (eps*, margin*) that maximizes inner held-out MAP@3.
   - Fit ranker on all of T_k.
   - Evaluate untouched outer holdout V_k using ONLY (eps*, margin*).
3. Report final out-of-fold generalization MAP@3 and outer fold deltas.
"""
from collections import Counter
from pathlib import Path
import re
import sys
from typing import Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

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
    print("STRICTLY NESTED CANDIDATE RERANKER (ZERO LEAKAGE SELECTION)")
    print("Inner CV selects (eps*, margin*); Evaluated on untouched outer holdouts")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")

    tax_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "taxonomy_v1.csv"))
    tax_info = {str(r["id"]): (str(r["name"]), str(r["description"])) for _, r in tax_df.iterrows()}

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_raw = train_df["root_code"].fillna("NONE").astype(str).values
    groups = train_df["question_id"].values
    n_samples = len(train_df)

    champ_probs = np.load(ART_DIR / "oof_aligned_champion_probs.npy")
    champ_cand_mat = champ_probs.copy()
    champ_cand_mat[:, none_idx] = -1.0
    sorted_champ_cands = np.argsort(-champ_cand_mat, axis=1)

    npz = np.load(ART_DIR / "minilm_reranker_text_embeddings.npz")
    emb_q = npz["emb_q"]
    emb_ocr = npz["emb_ocr"]
    emb_comb = npz["emb_comb"]
    emb_tax = npz["emb_tax"]

    ocr_token_sets = [extract_tokens(t) for t in train_df["transcription_steps"]]
    tax_token_sets = {c: extract_tokens(f"{tax_info.get(c, ('', ''))[0]} {tax_info.get(c, ('', ''))[1]}") for c in classes}

    def build_candidate_features(sample_idx: int, cand_c_idx: int, cand_rank: int, p_best: float, prior_p: float) -> List[float]:
        code = classes[cand_c_idx]
        p_c = float(champ_probs[sample_idx, cand_c_idx])
        gap = p_c - p_best
        ratio = p_c / max(p_best, 1e-6)

        c_q = float(np.dot(emb_q[sample_idx], emb_tax[cand_c_idx]))
        c_ocr = float(np.dot(emb_ocr[sample_idx], emb_tax[cand_c_idx]))
        c_comb = float(np.dot(emb_comb[sample_idx], emb_tax[cand_c_idx]))
        c_diff = c_ocr - c_q
        j_sim = jaccard(ocr_token_sets[sample_idx], tax_token_sets[code])

        return [p_c, float(cand_rank), gap, ratio, c_q, c_ocr, c_comb, c_diff, j_sim, prior_p]

    outer_gkf = GroupKFold(n_splits=5)
    K_CANDS = 5

    outer_baseline_top3: List[List[str]] = [[] for _ in range(n_samples)]
    outer_reranked_top3: List[List[str]] = [[] for _ in range(n_samples)]
    outer_deltas = []
    outer_swaps = 0
    selected_configs = []

    EPS_GRID = [0.020, 0.025, 0.030, 0.035, 0.040]
    MARGIN_GRID = [0.10, 0.15, 0.20]

    for outer_fold, (trn_idx, val_idx) in enumerate(outer_gkf.split(train_df, groups=groups), start=1):
        print(f"\n--- OUTER FOLD {outer_fold} / 5 ---")
        trn_df = train_df.iloc[trn_idx]
        val_df = train_df.iloc[val_idx]
        trn_y_raw = y_raw[trn_idx]
        val_y_raw = y_raw[val_idx].tolist()
        trn_groups = trn_df["question_id"].values
        n_trn = len(trn_idx)
        n_val = len(val_idx)

        # Baseline on val_df
        val_base_top3 = []
        for glob_i in val_idx:
            r1_pred = "NONE" if champ_probs[glob_i, none_idx] >= np.max(champ_cand_mat[glob_i]) else classes[sorted_champ_cands[glob_i, 0]]
            c1_code = classes[sorted_champ_cands[glob_i, 0]]
            c2_code = classes[sorted_champ_cands[glob_i, 1]]
            if r1_pred == "NONE":
                val_base_top3.append(["NONE", c1_code, c2_code])
            else:
                val_base_top3.append([c1_code, "NONE", c2_code])

        f_base = calculate_map3(val_y_raw, val_base_top3)

        # ---------------------------------------------------------------------
        # Inner 4-fold Cross-Fitting within T_k to select (eps*, margin*)
        # ---------------------------------------------------------------------
        inner_gkf = GroupKFold(n_splits=4)
        inner_oof_scores = np.zeros(n_trn, dtype=np.float32)
        inner_oof_margins = np.zeros(n_trn, dtype=np.float32)
        inner_gaps = np.zeros(n_trn, dtype=np.float32)

        # Precompute in-fold training prior
        prior_trn_eng = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df)
        trn_priors = prior_trn_eng.predict_proba(trn_df)

        for in_fold, (in_trn, in_val) in enumerate(inner_gkf.split(trn_df, groups=trn_groups), start=1):
            in_trn_df = trn_df.iloc[in_trn]
            in_val_df = trn_df.iloc[in_val]
            prior_in = AdaptiveQuestionPriorEngine(classes=classes).fit(in_trn_df)
            in_val_priors = prior_in.predict_proba(in_val_df)
            in_trn_priors = prior_in.predict_proba(in_trn_df)

            # Build features on in_trn
            in_feats = []
            in_targs = []
            for loc_i in in_trn:
                glob_i = trn_idx[loc_i]
                if champ_probs[glob_i, none_idx] >= np.max(champ_cand_mat[glob_i]):
                    true_c = trn_y_raw[loc_i]
                    cands = sorted_champ_cands[glob_i, :K_CANDS]
                    p_best = float(champ_probs[glob_i, cands[0]])
                    for rank_pos, c_idx in enumerate(cands, start=1):
                        code = classes[c_idx]
                        prior_val = float(in_trn_priors[loc_i, c_idx]) if loc_i < len(in_trn_priors) else 0.0
                        in_feats.append(build_candidate_features(glob_i, c_idx, rank_pos, p_best, prior_val))
                        in_targs.append(1 if code == true_c else 0)

            scaler_in = StandardScaler()
            X_in_scaled = scaler_in.fit_transform(np.array(in_feats, dtype=np.float32))
            clf_in = LogisticRegression(C=0.10, penalty="l2", class_weight="balanced", solver="lbfgs", random_state=42)
            clf_in.fit(X_in_scaled, np.array(in_targs, dtype=np.int32))

            # Predict on in_val
            for loc_i in in_val:
                glob_i = trn_idx[loc_i]
                if champ_probs[glob_i, none_idx] >= np.max(champ_cand_mat[glob_i]):
                    cands = sorted_champ_cands[glob_i, :K_CANDS]
                    p_c1 = float(champ_probs[glob_i, cands[0]])
                    p_c2 = float(champ_probs[glob_i, cands[1]])
                    gap_12 = p_c1 - p_c2
                    inner_gaps[loc_i] = gap_12

                    cand_feats = []
                    val_loc_in = list(in_val).index(loc_i)
                    for rank_pos, c_idx in enumerate(cands, start=1):
                        prior_val = float(in_val_priors[val_loc_in, c_idx])
                        cand_feats.append(build_candidate_features(glob_i, c_idx, rank_pos, p_c1, prior_val))

                    cand_scaled = scaler_in.transform(np.array(cand_feats))
                    scores = clf_in.predict_proba(cand_scaled)[:, 1]
                    inner_oof_margins[loc_i] = float(scores[1] - scores[0])

        # Evaluate inner grid to find best (eps*, margin*)
        best_cfg = (0.035, 0.15)
        best_inner_gain = -1.0

        for eps in EPS_GRID:
            for margin in MARGIN_GRID:
                # Calculate inner MAP@3 gain
                inner_helped = 0
                inner_hurt = 0
                for loc_i in range(n_trn):
                    glob_i = trn_idx[loc_i]
                    if champ_probs[glob_i, none_idx] >= np.max(champ_cand_mat[glob_i]):
                        if 0 < inner_gaps[loc_i] < eps and inner_oof_margins[loc_i] >= margin:
                            c1_code = classes[sorted_champ_cands[glob_i, 0]]
                            c2_code = classes[sorted_champ_cands[glob_i, 1]]
                            true_label = trn_y_raw[loc_i]
                            if true_label == c2_code:
                                inner_helped += 1
                            elif true_label == c1_code:
                                inner_hurt += 1

                inner_gain = inner_helped - inner_hurt
                if inner_gain > best_inner_gain:
                    best_inner_gain = inner_gain
                    best_cfg = (eps, margin)

        selected_configs.append(best_cfg)
        print(f"  Inner CV selected config: eps*={best_cfg[0]:.3f}, margin*={best_cfg[1]:.2f} (Inner Net Gain: +{best_inner_gain})")

        # ---------------------------------------------------------------------
        # Train Outer Ranker on ALL of T_k and Evaluate Untouched V_k
        # ---------------------------------------------------------------------
        out_trn_feats = []
        out_trn_targs = []
        for loc_i in range(n_trn):
            glob_i = trn_idx[loc_i]
            if champ_probs[glob_i, none_idx] >= np.max(champ_cand_mat[glob_i]):
                true_c = trn_y_raw[loc_i]
                cands = sorted_champ_cands[glob_i, :K_CANDS]
                p_best = float(champ_probs[glob_i, cands[0]])
                for rank_pos, c_idx in enumerate(cands, start=1):
                    code = classes[c_idx]
                    prior_val = float(trn_priors[loc_i, c_idx])
                    out_trn_feats.append(build_candidate_features(glob_i, c_idx, rank_pos, p_best, prior_val))
                    out_trn_targs.append(1 if code == true_c else 0)

        scaler_out = StandardScaler()
        X_out_scaled = scaler_out.fit_transform(np.array(out_trn_feats, dtype=np.float32))
        ranker_out = LogisticRegression(C=0.10, penalty="l2", class_weight="balanced", solver="lbfgs", random_state=42)
        ranker_out.fit(X_out_scaled, np.array(out_trn_targs, dtype=np.int32))

        # Evaluate V_k
        val_priors = prior_trn_eng.predict_proba(val_df)
        val_reranked = [list(r) for r in val_base_top3]
        f_swaps = 0
        eps_star, margin_star = best_cfg

        for loc_i in range(n_val):
            glob_i = val_idx[loc_i]
            if val_base_top3[loc_i][0] == "NONE":
                cands = sorted_champ_cands[glob_i, :K_CANDS]
                p_c1 = float(champ_probs[glob_i, cands[0]])
                p_c2 = float(champ_probs[glob_i, cands[1]])
                gap_12 = p_c1 - p_c2

                cand_feats = []
                for rank_pos, c_idx in enumerate(cands, start=1):
                    prior_val = float(val_priors[loc_i, c_idx])
                    cand_feats.append(build_candidate_features(glob_i, c_idx, rank_pos, p_c1, prior_val))

                cand_scaled = scaler_out.transform(np.array(cand_feats))
                cand_scores = ranker_out.predict_proba(cand_scaled)[:, 1]
                ranker_margin = cand_scores[1] - cand_scores[0]

                if 0 < gap_12 < eps_star and ranker_margin >= margin_star:
                    val_reranked[loc_i][1] = classes[cands[1]]
                    val_reranked[loc_i][2] = classes[cands[0]]
                    f_swaps += 1

        f_rerank = calculate_map3(val_y_raw, val_reranked)
        delta_f = f_rerank - f_base
        outer_deltas.append(delta_f)
        outer_swaps += f_swaps

        print(f"  Outer Holdout Baseline : {f_base:.5f}")
        print(f"  Outer Holdout Reranked : {f_rerank:.5f} ({delta_f:+.5f}) | Swaps: {f_swaps}")

        for loc_i, glob_i in enumerate(val_idx):
            outer_baseline_top3[glob_i] = val_base_top3[loc_i]
            outer_reranked_top3[glob_i] = val_reranked[loc_i]

    total_base = calculate_map3(y_raw.tolist(), outer_baseline_top3)
    total_rerank = calculate_map3(y_raw.tolist(), outer_reranked_top3)
    net_delta = total_rerank - total_base

    print("\n" + "=" * 80)
    print("FINAL STRICTLY NESTED EVALUATION REPORT")
    print("=" * 80)
    print(f"Champion Baseline MAP@3 (Held-Out) : {total_base:.5f}")
    print(f"Nested Reranked MAP@3 (Held-Out)   : {total_rerank:.5f}")
    print(f"Net Generalization Delta           : {net_delta:+.5f}")
    print(f"Total Swaps across Holdouts        : {outer_swaps}")
    print(f"Selected Inner Configs             : {selected_configs}")
    print(f"Per-Fold Deltas                    : {[round(float(d), 5) for d in outer_deltas]}")

    if net_delta >= 0.00020 and all(d >= 0.0 for d in outer_deltas):
        print("\nGATE STATUS: PASSED (Stable positive generalization across all outer folds)")
    else:
        print("\nGATE STATUS: UNVALIDATED / REJECTED")
    print("=" * 80)

if __name__ == "__main__":
    main()

