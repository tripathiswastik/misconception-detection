"""Inspect swap dynamics of aligned candidate reranker across all folds."""
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

    print(f"{'Fold':<6} | {'Base MAP@3':<10} | {'Rerank MAP@3':<12} | {'Exact Delta':<14} | {'Swaps':<6} | {'Helped':<6} | {'Hurt':<6} | {'Neutral':<7}")
    print("-" * 80)

    for outer_fold, (trn_idx, val_idx) in enumerate(outer_gkf.split(train_df, groups=groups), start=1):
        trn_df = train_df.iloc[trn_idx]
        val_df = train_df.iloc[val_idx]
        trn_y_raw = y_raw[trn_idx]
        val_y_raw = y_raw[val_idx].tolist()
        n_trn = len(trn_idx)
        n_val = len(val_idx)

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

        prior_trn_eng = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df)
        trn_priors = prior_trn_eng.predict_proba(trn_df)

        trn_feats = []
        trn_targs = []
        for loc_i in range(n_trn):
            glob_i = trn_idx[loc_i]
            if champ_probs[glob_i, none_idx] >= np.max(champ_cand_mat[glob_i]):
                true_c = trn_y_raw[loc_i]
                cands = sorted_champ_cands[glob_i, :K_CANDS]
                p_best = float(champ_probs[glob_i, cands[0]])
                for rank_pos, c_idx in enumerate(cands, start=1):
                    code = classes[c_idx]
                    prior_val = float(trn_priors[loc_i, c_idx])
                    trn_feats.append(build_candidate_features(glob_i, c_idx, rank_pos, p_best, prior_val))
                    trn_targs.append(1 if code == true_c else 0)

        scaler = StandardScaler()
        X_trn_scaled = scaler.fit_transform(np.array(trn_feats, dtype=np.float32))
        ranker = LogisticRegression(C=0.10, penalty="l2", class_weight="balanced", solver="lbfgs", random_state=42)
        ranker.fit(X_trn_scaled, np.array(trn_targs, dtype=np.int32))

        val_priors = prior_trn_eng.predict_proba(val_df)
        val_reranked = [list(r) for r in val_base_top3]

        swaps = 0
        helped = 0
        hurt = 0
        neutral = 0

        for loc_i in range(n_val):
            glob_i = val_idx[loc_i]
            true_label = val_y_raw[loc_i]
            if val_base_top3[loc_i][0] == "NONE":
                cands = sorted_champ_cands[glob_i, :K_CANDS]
                p_c1 = float(champ_probs[glob_i, cands[0]])
                p_c2 = float(champ_probs[glob_i, cands[1]])
                gap_12 = p_c1 - p_c2

                cand_feats = []
                for rank_pos, c_idx in enumerate(cands, start=1):
                    prior_val = float(val_priors[loc_i, c_idx])
                    cand_feats.append(build_candidate_features(glob_i, c_idx, rank_pos, p_c1, prior_val))

                cand_scaled = scaler.transform(np.array(cand_feats))
                cand_scores = ranker.predict_proba(cand_scaled)[:, 1]

                ranker_margin = cand_scores[1] - cand_scores[0]
                if 0 < gap_12 < 0.030 and ranker_margin >= 0.15:
                    c1_code = classes[cands[0]]
                    c2_code = classes[cands[1]]

                    val_reranked[loc_i][1] = c2_code
                    val_reranked[loc_i][2] = c1_code
                    swaps += 1

                    # Check impact on this specific trace
                    # Old order: NONE, C1, C2. New order: NONE, C2, C1
                    if true_label == c2_code:
                        helped += 1  # Promoted true label from Rank 3 (1/3) to Rank 2 (1/2): +1/6 (+0.1667)
                    elif true_label == c1_code:
                        hurt += 1    # Demoted true label from Rank 2 (1/2) to Rank 3 (1/3): -1/6 (-0.1667)
                    else:
                        neutral += 1 # True label was neither C1 nor C2 (e.g. true label was NONE or another misconception)

        f_rerank = calculate_map3(val_y_raw, val_reranked)
        exact_delta = f_rerank - f_base
        print(f"Fold {outer_fold:<2} | {f_base:<10.5f} | {f_rerank:<12.5f} | {exact_delta:<+14.7f} | {swaps:<6} | {helped:<6} | {hurt:<6} | {neutral:<7}")

if __name__ == "__main__":
    main()

