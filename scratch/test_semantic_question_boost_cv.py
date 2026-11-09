"""Evaluate Semantic Question Boost Backoff for Unseen Questions in 5-Fold CV.

Tests whether expanding question-conditioned priors via semantic question similarity (MiniLM-384)
improves MAP@3 on unseen questions without leaking labels.
"""
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.metrics.map3 import calculate_map3

def sharp(probs: np.ndarray, t: float) -> np.ndarray:
    s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    return s / s.sum(axis=1, keepdims=True)

def main():
    print("=" * 80, flush=True)
    print("EVALUATING SEMANTIC QUESTION BOOST BACKOFF IN 5-FOLD CV", flush=True)
    print("=" * 80, flush=True)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    oof_v1 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")

    p_base_v1 = 0.95 * oof_m15 + 0.02 * oof_dual + 0.03 * oof_comp

    oof_clean = (
        0.6448 * np.load(ART_DIR / "best_adaptive_prior_oof.npy") +
        0.0839 * sharp(np.load(ART_DIR / "honest_oof_text.npy"), 0.6) +
        0.0401 * sharp(np.load(ART_DIR / "oof_semantic_text.npy"), 0.6) +
        0.0458 * np.load(ART_DIR / "oof_convnext_vision.npy") +
        0.0454 * np.load(ART_DIR / "oof_swin_vision.npy") +
        0.0483 * np.load(ART_DIR / "oof_finetuned_vision.npy") +
        0.0455 * np.load(ART_DIR / "oof_trocr_vision.npy") +
        0.0462 * sharp(np.load(ART_DIR / "oof_clean_arcface_2328.npy"), 0.8)
    )

    oof_bin = np.load(ART_DIR / "oof_compact_binary_prob.npy")
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]

    # Load 384-d text embeddings
    tr_emb = np.load(ART_DIR / "train_text_embeddings_384.npy")
    # Normalize
    tr_emb_norm = tr_emb / np.linalg.norm(tr_emb, axis=1, keepdims=True)

    # Unique question embeddings map
    q_to_emb = {}
    for i, qid in enumerate(train_df["question_id"]):
        if qid not in q_to_emb:
            q_to_emb[qid] = tr_emb_norm[i]

    # Test similarity thresholds for backoff
    sim_thresholds = [None, 0.92, 0.88, 0.85, 0.80]

    for thresh in sim_thresholds:
        outer_preds = [[] for _ in range(n_samples)]
        unseen_boost_count = 0

        for fold in splits:
            tr_idx = np.array(fold["train_indices"], dtype=np.int32)
            va_idx = np.array(fold["val_indices"], dtype=np.int32)
            n_eval = len(va_idx)

            # In-fold Multiplier
            top3_tr = np.argsort(-p_base_v1[tr_idx].astype(np.float64), axis=1)[:, :3]
            pred_counts = Counter(classes[idx] for row in top3_tr for idx in row)
            true_labels_tr = train_df.iloc[tr_idx]["root_code"].fillna("NONE").astype(str).tolist()
            true_counts = Counter(true_labels_tr)

            mult = np.ones(n_classes, dtype=np.float64)
            for i, c in enumerate(classes):
                if c != "NONE":
                    pc = max(pred_counts.get(c, 1), 1)
                    tc = max(true_counts.get(c, 1), 1)
                    mult[i] = (tc / pc) ** 0.05
            mult[none_idx] = 0.9840

            p_eval_cal = p_base_v1[va_idx] * mult
            p_eval_cal[:, none_idx] *= 0.9995

            # Question Allowed Map from Training Split
            tr_sub_df = train_df.iloc[tr_idx]
            q_map: Dict[str, Set[str]] = {q: set(grp["root_code"].dropna()) for q, grp in tr_sub_df.groupby("question_id")}
            tr_qids = list(q_map.keys())
            tr_q_matrix = np.array([q_to_emb[q] for q in tr_qids])

            # Backoff for unseen questions
            q_map_effective = dict(q_map)
            if thresh is not None:
                va_qids = set(train_df.iloc[va_idx]["question_id"]) - set(tr_qids)
                for u_qid in va_qids:
                    u_vec = q_to_emb[u_qid]
                    cos_sims = tr_q_matrix @ u_vec
                    best_i = int(np.argmax(cos_sims))
                    best_sim = cos_sims[best_i]
                    if best_sim >= thresh:
                        best_tr_qid = tr_qids[best_i]
                        q_map_effective[u_qid] = q_map[best_tr_qid]
                        unseen_boost_count += 1

            p_eval_boosted = np.copy(p_eval_cal)
            eval_df = train_df.iloc[va_idx].reset_index(drop=True)
            for i in range(n_eval):
                qid = eval_df.iloc[i]["question_id"]
                if qid in q_map_effective:
                    allowed = q_map_effective[qid]
                    for ci, code in enumerate(classes):
                        if code != "NONE" and code in allowed:
                            p_eval_boosted[i, ci] *= 1.20

            p_eval_arc_cal = (0.95 * oof_m15[va_idx] + 0.05 * oof_v1[va_idx]) * mult
            p_eval_champ = 0.70 * p_eval_cal + 0.25 * p_eval_boosted + 0.05 * p_eval_arc_cal
            p_eval_final = 0.96 * p_eval_champ + 0.04 * oof_clean[va_idx]

            # Decision Engine
            cand = p_eval_final.copy()
            row_idx = np.arange(n_eval)
            r1 = np.argmax(cand, axis=1)
            cand[row_idx, r1] = -1.0
            cand[r1 != none_idx, none_idx] *= 0.82
            top2 = np.argsort(-cand, axis=1)[:, :2]
            preds = [[classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(n_eval)]

            # Gating
            for i in range(n_eval):
                if preds[i][0] == "NONE" and oof_bin[va_idx[i]] >= 0.780:
                    row_p = p_eval_final[i].copy()
                    p_none_val = row_p[none_idx]
                    row_p[none_idx] = -1.0
                    top_misc_idx = int(np.argmax(row_p))
                    top_misc_code = classes[top_misc_idx]
                    gap = p_none_val - row_p[top_misc_idx]

                    if 0 < gap < 0.040:
                        qid = eval_df.iloc[i]["question_id"]
                        if top_misc_code in q_map_effective.get(qid, set()):
                            row_p[top_misc_idx] = -1.0
                            second_misc_idx = int(np.argmax(row_p))
                            second_misc_code = classes[second_misc_idx]
                            preds[i] = [top_misc_code, "NONE", second_misc_code]

            for local_i, global_i in enumerate(va_idx):
                outer_preds[global_i] = preds[local_i]

        score = calculate_map3(y_true, outer_preds)
        name = f"Thresh >= {thresh:.2f}" if thresh is not None else "No Backoff (Exact Match Only)"
        print(f"[{name:35s}] Leak-Free OOF MAP@3 = {score:.5f} | Unseen Questions Boosted: {unseen_boost_count}", flush=True)

    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()

