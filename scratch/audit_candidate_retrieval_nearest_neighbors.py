"""Candidate Retrieval Audit using Independent Nearest-Neighbor Sources.

Methodology per User Directive:
1. Evaluates on held-out folds:
   - Seen Questions (5-fold stratified multi-trace out-of-row split, 1419 traces)
   - Unseen Questions (5-fold GroupKFold on question_id, 1500 traces)
   - Composite metric: 0.957 * Seen + 0.043 * Unseen
2. Independent Nearest-Neighbor Sources:
   - Visual neighbors of handwriting crops (L2-normalized ConvNeXt, Swin, TrOCR, and 1,920-d Unified Vision)
   - Text neighbors of question / work context (L2-normalized dense text embeddings, 384-d)
   - Multimodal neighbors (balanced visual + text cosine similarity)
3. Strict Fold-Isolation:
   - Candidates generated strictly from fold-training rows T_k.
   - Validation trace v in V_k is strictly excluded from its neighbor pool.
4. Quantitative Metrics:
   - Overall Recall@1, Recall@2, Recall@3, Recall@5, Recall@10
   - Recovery Rate on Champion Union Top-3 Misses (where true label is missing from both S550 and S554 Top-3)
   - Breakdown across question support categories (Clean, Strong Support, Weak Support, Novel Label)
   - MAP@3 impact if expanded candidates are scored.
"""
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, cast
from typing_extensions import TypedDict
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

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

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)

def compute_recalls(y_true: List[str], candidate_lists: List[List[str]], k_list: List[int] = [1, 2, 3, 5, 10]) -> Dict[str, float]:
    recalls = {}
    n = len(y_true)
    for k in k_list:
        hits = sum(1 for i in range(n) if y_true[i] in candidate_lists[i][:k])
        recalls[f"Recall@{k}"] = round(hits / max(n, 1), 4)
    return recalls

class DataLoader:
    def __init__(self, data_dir: Path, art_dir: Path):
        self.tax = TaxonomyManager(data_dir / "taxonomy_v1.csv")
        self.classes = self.tax.all_codes(include_none=True)
        self.none_idx = self.classes.index("NONE")
        self.n_classes = len(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.train_df = cast(pd.DataFrame, pd.read_csv(data_dir / "train.csv"))
        self.y_raw = self.train_df["root_code"].fillna("NONE").astype(str).tolist()
        self.n_samples = len(self.train_df)

        # Baseline probability components
        self.oof_m15 = np.load(art_dir / "oof_m15_balanced_fusion.npy")
        self.oof_dual = np.load(art_dir / "oof_arcface_dualhead_448_soft.npy")
        self.oof_comp = np.load(art_dir / "oof_arcface_composite_448.npy")
        self.oof_v1 = np.load(art_dir / "oof_arcface_pseudo_3500.npy")
        self.oof_v2 = np.load(art_dir / "oof_arcface_pseudo_3500_v2.npy")
        self.honest_text = np.load(art_dir / "honest_oof_text.npy")
        self.oof_sem_text = np.load(art_dir / "oof_semantic_text.npy")
        self.oof_convnext = np.load(art_dir / "oof_convnext_vision.npy")
        self.oof_swin = np.load(art_dir / "oof_swin_vision.npy")
        self.oof_finetuned = np.load(art_dir / "oof_finetuned_vision.npy")
        self.oof_trocr = np.load(art_dir / "oof_trocr_vision.npy")
        self.oof_clean_arcface = np.load(art_dir / "oof_clean_arcface_2328.npy")
        self.oof_bin_v1 = np.load(art_dir / "oof_compact_binary_prob.npy")
        self.oof_bin_v2 = np.load(art_dir / "oof_compact_binary_prob_v2.npy")

        # Independent Nearest-Neighbor Feature Embeddings
        # 1. Text dense embeddings (384-d)
        self.text_emb = l2_normalize(np.load(art_dir / "train_text_embeddings_384.npy"))

        # 2. Visual dense embeddings of handwriting crops
        conv_raw = np.load(art_dir / "train_convnext_features.npy")
        swin_raw = np.load(art_dir / "train_swin_features.npy")
        trocr_raw = np.load(art_dir / "train_trocr_features.npy")

        self.conv_emb = l2_normalize(conv_raw)
        self.swin_emb = l2_normalize(swin_raw)
        self.trocr_emb = l2_normalize(trocr_raw)

        # Unified multimodal visual stack (1920-d)
        vis_concat = np.hstack([self.conv_emb, self.swin_emb, self.trocr_emb])
        self.vis_unified_emb = l2_normalize(vis_concat)

    def build_fold_clean_multimodal(self, trn_idx: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
        trn_df_fold = self.train_df.iloc[trn_idx]
        val_df_fold = self.train_df.iloc[val_idx]
        prior_eng = AdaptiveQuestionPriorEngine(classes=self.classes).fit(trn_df_fold)
        val_prior = prior_eng.predict_proba(val_df_fold).astype(np.float64)

        p_clean_val = (
            0.6448 * val_prior +
            0.0839 * sharp(self.honest_text[val_idx], 0.6) +
            0.0401 * sharp(self.oof_sem_text[val_idx], 0.6) +
            0.0458 * self.oof_convnext[val_idx] +
            0.0454 * self.oof_swin[val_idx] +
            0.0483 * self.oof_finetuned[val_idx] +
            0.0455 * self.oof_trocr[val_idx] +
            0.0462 * sharp(self.oof_clean_arcface[val_idx], 0.8)
        )
        return p_clean_val

    def generate_fold_probs(self, trn_idx: np.ndarray, val_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Counter]]:
        trn_df_fold = self.train_df.iloc[trn_idx]
        val_df_fold = self.train_df.iloc[val_idx]
        n_val = len(val_idx)

        # Frequency calibration mult
        p_base_v1_trn = 0.95 * self.oof_m15[trn_idx] + 0.02 * self.oof_dual[trn_idx] + 0.03 * self.oof_comp[trn_idx]
        top3_trn = np.argsort(-p_base_v1_trn.astype(np.float64), axis=1)[:, :3]
        pred_counts = Counter(self.classes[idx] for row in top3_trn for idx in row)
        true_counts = Counter(trn_df_fold["root_code"].fillna("NONE").values)
        mult = np.ones(self.n_classes, dtype=np.float64)
        for i, c in enumerate(self.classes):
            if c != "NONE":
                mult[i] = (max(true_counts.get(c, 1), 1) / max(pred_counts.get(c, 1), 1)) ** 0.05
        mult[self.none_idx] = 0.9840

        p_base_v1_val = (0.95 * self.oof_m15[val_idx] + 0.02 * self.oof_dual[val_idx] + 0.03 * self.oof_comp[val_idx]) * mult
        p_base_v1_val[:, self.none_idx] *= 0.9995

        p_base_v2_val = (0.88 * self.oof_m15[val_idx] + 0.12 * self.oof_v2[val_idx]) * mult
        p_base_v2_val[:, self.none_idx] *= 0.9995

        # Transductive question boost from trn_idx
        q_label_dist: Dict[str, Counter] = {
            q: Counter(grp["root_code"].fillna("NONE").values)
            for q, grp in trn_df_fold.groupby("question_id")
        }

        p_boosted_v1 = np.copy(p_base_v1_val)
        p_boosted_v2 = np.copy(p_base_v2_val)
        for i in range(n_val):
            qid = val_df_fold.iloc[i]["question_id"]
            if qid in q_label_dist:
                allowed_misconceptions = {c for c, cnt in q_label_dist[qid].items() if c != "NONE"}
                for ci, c in enumerate(self.classes):
                    if c in allowed_misconceptions:
                        p_boosted_v1[i, ci] *= 1.20
                        p_boosted_v2[i, ci] *= 1.20

        p_arc_v1 = (0.95 * self.oof_m15[val_idx] + 0.05 * self.oof_v1[val_idx]) * mult
        p_arc_v2 = (0.88 * self.oof_m15[val_idx] + 0.12 * self.oof_v2[val_idx]) * mult

        p_champ_v1 = 0.70 * p_base_v1_val + 0.25 * p_boosted_v1 + 0.05 * p_arc_v1
        p_champ_v2 = 0.65 * p_base_v2_val + 0.25 * p_boosted_v2 + 0.10 * p_arc_v2

        p_clean = self.build_fold_clean_multimodal(trn_idx, val_idx)
        p_s550 = 0.96 * p_champ_v1 + 0.04 * p_clean
        p_s554 = 0.96 * p_champ_v2 + 0.04 * p_clean

        return p_s550, p_s554, q_label_dist

    def rank_and_gate_full(self, p_matrix: np.ndarray, val_idx: np.ndarray, q_label_dist: Dict[str, Counter], beta: float = 0.82) -> List[List[str]]:
        """Returns full ranking of all 76 classes with champion precision gating applied to Top 3."""
        cand = p_matrix.copy()
        n = len(cand)
        row_idx = np.arange(n)
        r1 = np.argmax(cand, axis=1)

        cand[row_idx, r1] = -1e9
        cand[r1 != self.none_idx, self.none_idx] *= beta
        rem_ranks = np.argsort(-cand, axis=1)  # (n, 75)

        full_ranks = np.column_stack([r1[:, None], rem_ranks])  # (n, 76)
        preds = [[self.classes[idx] for idx in full_ranks[i]] for i in range(n)]

        # Apply champion precision gating
        p_bin = self.oof_bin_v1[val_idx]
        val_df = self.train_df.iloc[val_idx].reset_index(drop=True)

        for i in range(n):
            if preds[i][0] == "NONE" and p_bin[i] >= 0.78:
                row_p = p_matrix[i].copy()
                p_none = row_p[self.none_idx]
                row_p[self.none_idx] = -1.0
                top_m_idx = int(np.argmax(row_p))
                top_m_code = self.classes[top_m_idx]
                gap = p_none - row_p[top_m_idx]

                if 0 < gap < 0.04:
                    qid = val_df.iloc[i]["question_id"]
                    allowed = {c for c, cnt in q_label_dist.get(qid, Counter()).items() if c != "NONE"}
                    if top_m_code in allowed:
                        row_p[top_m_idx] = -1.0
                        sec_m = self.classes[int(np.argmax(row_p))]
                        # Reorder top candidates: [top_m_code, "NONE", sec_m, rest...]
                        old_list = preds[i]
                        new_head = [top_m_code, "NONE", sec_m]
                        new_tail = [c for c in old_list if c not in new_head]
                        preds[i] = new_head + new_tail
        return preds

def query_nearest_neighbors(
    trn_emb: np.ndarray,
    val_emb: np.ndarray,
    trn_labels: List[str],
    all_classes: List[str],
    top_k_neighbors: int = 25,
    tau: float = 0.10
) -> Tuple[List[List[str]], np.ndarray]:
    """Queries training set neighbors for each validation row.

    Returns:
    - distinct_neighbor_labels: list of ordered unique classes from neighbors
    - nn_prob_matrix: softmax-weighted probability distribution across all 76 classes
    """
    n_val = len(val_emb)
    n_classes = len(all_classes)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    # Cosine similarities: shape (n_val, n_trn)
    sim_matrix = np.dot(val_emb, trn_emb.T)

    distinct_labels_per_row: List[List[str]] = []
    nn_prob_matrix = np.zeros((n_val, n_classes), dtype=np.float32)

    for i in range(n_val):
        row_sim = sim_matrix[i]
        top_idx = np.argsort(-row_sim)[:top_k_neighbors]
        top_sims = row_sim[top_idx]

        # Weights via temperature-scaled softmax
        exp_sims = np.exp(np.clip(top_sims / tau, -50.0, 50.0))
        weights = exp_sims / np.maximum(exp_sims.sum(), 1e-12)

        # Accumulate class probabilities
        seen_labels: List[str] = []
        for j, trn_pos in enumerate(top_idx):
            lbl = trn_labels[trn_pos]
            c_idx = class_to_idx[lbl]
            nn_prob_matrix[i, c_idx] += weights[j]
            if lbl not in seen_labels:
                seen_labels.append(lbl)

        distinct_labels_per_row.append(seen_labels)

    return distinct_labels_per_row, nn_prob_matrix

def main():
    print("=" * 95)
    print("CANDIDATE-RETRIEVAL AUDIT USING INDEPENDENT NEAREST-NEIGHBOR SOURCES")
    print("Zero Leakage | Fold-Training Isolation | Separate Seen & Unseen Breakdown")
    print("=" * 95)

    start_time = time.time()
    data = DataLoader(DATA_DIR, ART_DIR)
    train_df = data.train_df
    y_true_all = data.y_raw
    n_samples = data.n_samples

    # Build splits
    # 1. Transductive Seen Split (1,419 multi-trace rows)
    q_counts = Counter(train_df["question_id"])
    multi_q_mask = np.array([q_counts[q] >= 2 for q in train_df["question_id"]])
    sub_df = train_df.iloc[multi_q_mask].copy().reset_index(drop=True)
    sub_indices_global = np.where(multi_q_mask)[0]
    n_seen = len(sub_df)

    kf_seen = KFold(n_splits=5, shuffle=True, random_state=42)
    seen_splits = []
    for trn_loc, val_loc in kf_seen.split(sub_df):
        seen_splits.append((sub_indices_global[trn_loc], sub_indices_global[val_loc]))

    # 2. Inductive Unseen Split (1,500 traces across 462 questions)
    gkf = GroupKFold(n_splits=5)
    unseen_splits = list(gkf.split(train_df, groups=train_df["question_id"]))

    # Sources to evaluate
    retrieval_sources = [
        "Champion S550 Baseline",
        "Champion S554 Baseline",
        "Champion S550+S554 Union",
        "Visual NN (ConvNeXt 768-d)",
        "Visual NN (Swin 768-d)",
        "Visual NN (Unified 1920-d)",
        "Text NN (Context 384-d)",
        "Multimodal NN (Vis + Text)",
        "S550 Top3 + Visual NN Expansion",
        "S550 Top3 + Text NN Expansion",
        "S550 Top3 + Multimodal NN Expansion",
        "Blended Softmax (S550 + Multimodal NN)"
    ]

    # Containers for out-of-fold candidates
    seen_cands: Dict[str, List[List[str]]] = {name: [[] for _ in range(n_seen)] for name in retrieval_sources}
    unseen_cands: Dict[str, List[List[str]]] = {name: [[] for _ in range(n_samples)] for name in retrieval_sources}

    # Tracking misses from both champions: (index, gold, question_id, support_cat)
    seen_champion_misses: List[Dict[str, Any]] = []
    unseen_champion_misses: List[Dict[str, Any]] = []

    # =========================================================================
    # STEP 1: EVALUATE TRANSDUCTIVE SEEN QUESTIONS (1,419 traces)
    # =========================================================================
    print("\n" + "=" * 95)
    print("EVALUATING TRANSDUCTIVE SEEN QUESTIONS (95.7% Competition Proxy)")
    print("=" * 95)

    for fold_idx, (trn_idx, val_idx) in enumerate(seen_splits, start=1):
        n_val = len(val_idx)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)
        y_val = [y_true_all[i] for i in val_idx]
        y_trn = [y_true_all[i] for i in trn_idx]

        # 1. Champion probabilities and full rankings
        p_s550, p_s554, q_dist = data.generate_fold_probs(trn_idx, val_idx)
        ranks_550 = data.rank_and_gate_full(p_s550, val_idx, q_dist)
        ranks_554 = data.rank_and_gate_full(p_s554, val_idx, q_dist)

        # 2. Nearest Neighbors from independent feature spaces (strictly from trn_idx)
        # a) ConvNeXt
        nn_conv_labels, nn_conv_probs = query_nearest_neighbors(
            data.conv_emb[trn_idx], data.conv_emb[val_idx], y_trn, data.classes
        )
        # b) Swin
        nn_swin_labels, nn_swin_probs = query_nearest_neighbors(
            data.swin_emb[trn_idx], data.swin_emb[val_idx], y_trn, data.classes
        )
        # c) Unified Vision
        nn_vis_labels, nn_vis_probs = query_nearest_neighbors(
            data.vis_unified_emb[trn_idx], data.vis_unified_emb[val_idx], y_trn, data.classes
        )
        # d) Text Dense
        nn_text_labels, nn_text_probs = query_nearest_neighbors(
            data.text_emb[trn_idx], data.text_emb[val_idx], y_trn, data.classes
        )
        # e) Multimodal (0.5 visual + 0.5 text)
        mm_sim = 0.5 * np.dot(data.vis_unified_emb[val_idx], data.vis_unified_emb[trn_idx].T) + \
                 0.5 * np.dot(data.text_emb[val_idx], data.text_emb[trn_idx].T)
        nn_mm_labels: List[List[str]] = []
        nn_mm_probs = np.zeros((n_val, data.n_classes), dtype=np.float32)
        for i in range(n_val):
            top_j = np.argsort(-mm_sim[i])[:25]
            top_s = mm_sim[i][top_j]
            exp_s = np.exp(np.clip(top_s / 0.10, -50.0, 50.0))
            w = exp_s / np.maximum(exp_s.sum(), 1e-12)
            u_lbls: List[str] = []
            for k_pos, trn_pos in enumerate(top_j):
                lbl = y_trn[trn_pos]
                nn_mm_probs[i, data.class_to_idx[lbl]] += w[k_pos]
                if lbl not in u_lbls:
                    u_lbls.append(lbl)
            nn_mm_labels.append(u_lbls)

        # 3. Build candidate lists for each source
        val_local_pos = [np.where(sub_indices_global == idx)[0][0] for idx in val_idx]

        for i in range(n_val):
            gold = y_val[i]
            r550 = ranks_550[i]
            r554 = ranks_554[i]

            # Union Top-3 of champions
            union_top3 = []
            for c in r550[:3] + r554[:3]:
                if c not in union_top3:
                    union_top3.append(c)
            # Pad union with remaining 550 ranks
            union_full = union_top3 + [c for c in r550 if c not in union_top3]

            # Standalone NN rankings from probabilities
            nn_vis_ranked = [data.classes[c_idx] for c_idx in np.argsort(-nn_vis_probs[i])]
            nn_text_ranked = [data.classes[c_idx] for c_idx in np.argsort(-nn_text_probs[i])]
            nn_mm_ranked = [data.classes[c_idx] for c_idx in np.argsort(-nn_mm_probs[i])]

            # S550 Top3 + NN Expansions
            def expand_with_nn(base_top3: List[str], nn_pool: List[str], fallback: List[str]) -> List[str]:
                exp = list(base_top3[:3])
                for c in nn_pool:
                    if c not in exp:
                        exp.append(c)
                for c in fallback:
                    if c not in exp:
                        exp.append(c)
                return exp

            exp_vis = expand_with_nn(r550, nn_vis_labels[i], r550)
            exp_text = expand_with_nn(r550, nn_text_labels[i], r550)
            exp_mm = expand_with_nn(r550, nn_mm_labels[i], r550)

            # Blended Softmax (0.85 S550 + 0.15 MM NN)
            p_hybrid = 0.85 * p_s550[i] + 0.15 * nn_mm_probs[i]
            blended_ranked = [data.classes[c_idx] for c_idx in np.argsort(-p_hybrid)]

            # Check if this trace is missed by both champions
            is_champ_miss = (gold not in r550[:3]) and (gold not in r554[:3])
            if is_champ_miss:
                qid = val_df.iloc[i]["question_id"]
                q_counts_tr = q_dist.get(qid, Counter())
                supp = q_counts_tr.get(gold, 0)
                if gold == "NONE":
                    cat = "Clean (NONE)"
                elif supp >= 2:
                    cat = "Strong Support (>=2)"
                elif supp == 1:
                    cat = "Weak Support (=1)"
                else:
                    cat = "Novel Label (=0)"

                seen_champion_misses.append({
                    "val_idx": val_idx[i],
                    "gold": gold,
                    "qid": qid,
                    "cat": cat,
                    "s550_top3": r550[:3],
                    "s554_top3": r554[:3],
                    "exp_vis_5": exp_vis[:5],
                    "exp_vis_10": exp_vis[:10],
                    "exp_text_5": exp_text[:5],
                    "exp_text_10": exp_text[:10],
                    "exp_mm_5": exp_mm[:5],
                    "exp_mm_10": exp_mm[:10],
                    "recovered_vis_5": gold in exp_vis[:5],
                    "recovered_vis_10": gold in exp_vis[:10],
                    "recovered_text_5": gold in exp_text[:5],
                    "recovered_text_10": gold in exp_text[:10],
                    "recovered_mm_5": gold in exp_mm[:5],
                    "recovered_mm_10": gold in exp_mm[:10]
                })

            # Store predictions
            loc = val_local_pos[i]
            seen_cands[retrieval_sources[0]][loc] = r550
            seen_cands[retrieval_sources[1]][loc] = r554
            seen_cands[retrieval_sources[2]][loc] = union_full
            seen_cands[retrieval_sources[3]][loc] = [data.classes[c] for c in np.argsort(-nn_conv_probs[i])]
            seen_cands[retrieval_sources[4]][loc] = [data.classes[c] for c in np.argsort(-nn_swin_probs[i])]
            seen_cands[retrieval_sources[5]][loc] = nn_vis_ranked
            seen_cands[retrieval_sources[6]][loc] = nn_text_ranked
            seen_cands[retrieval_sources[7]][loc] = nn_mm_ranked
            seen_cands[retrieval_sources[8]][loc] = exp_vis
            seen_cands[retrieval_sources[9]][loc] = exp_text
            seen_cands[retrieval_sources[10]][loc] = exp_mm
            seen_cands[retrieval_sources[11]][loc] = blended_ranked

    # =========================================================================
    # STEP 2: EVALUATE INDUCTIVE UNSEEN QUESTIONS (1,500 traces)
    # =========================================================================
    print("\n" + "=" * 95)
    print("EVALUATING INDUCTIVE UNSEEN QUESTIONS (100% Question-Held-Out)")
    print("=" * 95)

    for fold_idx, (trn_idx, val_idx) in enumerate(unseen_splits, start=1):
        n_val = len(val_idx)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)
        y_val = [y_true_all[i] for i in val_idx]
        y_trn = [y_true_all[i] for i in trn_idx]

        p_s550, p_s554, q_dist = data.generate_fold_probs(trn_idx, val_idx)
        ranks_550 = data.rank_and_gate_full(p_s550, val_idx, q_dist)
        ranks_554 = data.rank_and_gate_full(p_s554, val_idx, q_dist)

        # Independent NN embeddings
        nn_conv_labels, nn_conv_probs = query_nearest_neighbors(
            data.conv_emb[trn_idx], data.conv_emb[val_idx], y_trn, data.classes
        )
        nn_swin_labels, nn_swin_probs = query_nearest_neighbors(
            data.swin_emb[trn_idx], data.swin_emb[val_idx], y_trn, data.classes
        )
        nn_vis_labels, nn_vis_probs = query_nearest_neighbors(
            data.vis_unified_emb[trn_idx], data.vis_unified_emb[val_idx], y_trn, data.classes
        )
        nn_text_labels, nn_text_probs = query_nearest_neighbors(
            data.text_emb[trn_idx], data.text_emb[val_idx], y_trn, data.classes
        )

        mm_sim = 0.5 * np.dot(data.vis_unified_emb[val_idx], data.vis_unified_emb[trn_idx].T) + \
                 0.5 * np.dot(data.text_emb[val_idx], data.text_emb[trn_idx].T)
        nn_mm_labels = []
        nn_mm_probs = np.zeros((n_val, data.n_classes), dtype=np.float32)
        for i in range(n_val):
            top_j = np.argsort(-mm_sim[i])[:25]
            top_s = mm_sim[i][top_j]
            exp_s = np.exp(np.clip(top_s / 0.10, -50.0, 50.0))
            w = exp_s / np.maximum(exp_s.sum(), 1e-12)
            u_lbls = []
            for k_pos, trn_pos in enumerate(top_j):
                lbl = y_trn[trn_pos]
                nn_mm_probs[i, data.class_to_idx[lbl]] += w[k_pos]
                if lbl not in u_lbls:
                    u_lbls.append(lbl)
            nn_mm_labels.append(u_lbls)

        for i in range(n_val):
            gold = y_val[i]
            r550 = ranks_550[i]
            r554 = ranks_554[i]

            union_top3 = []
            for c in r550[:3] + r554[:3]:
                if c not in union_top3:
                    union_top3.append(c)
            union_full = union_top3 + [c for c in r550 if c not in union_top3]

            nn_vis_ranked = [data.classes[c_idx] for c_idx in np.argsort(-nn_vis_probs[i])]
            nn_text_ranked = [data.classes[c_idx] for c_idx in np.argsort(-nn_text_probs[i])]
            nn_mm_ranked = [data.classes[c_idx] for c_idx in np.argsort(-nn_mm_probs[i])]

            exp_vis = expand_with_nn(r550, nn_vis_labels[i], r550)
            exp_text = expand_with_nn(r550, nn_text_labels[i], r550)
            exp_mm = expand_with_nn(r550, nn_mm_labels[i], r550)

            p_hybrid = 0.85 * p_s550[i] + 0.15 * nn_mm_probs[i]
            blended_ranked = [data.classes[c_idx] for c_idx in np.argsort(-p_hybrid)]

            is_champ_miss = (gold not in r550[:3]) and (gold not in r554[:3])
            if is_champ_miss:
                unseen_champion_misses.append({
                    "val_idx": val_idx[i],
                    "gold": gold,
                    "cat": "Unseen Question",
                    "recovered_vis_5": gold in exp_vis[:5],
                    "recovered_vis_10": gold in exp_vis[:10],
                    "recovered_text_5": gold in exp_text[:5],
                    "recovered_text_10": gold in exp_text[:10],
                    "recovered_mm_5": gold in exp_mm[:5],
                    "recovered_mm_10": gold in exp_mm[:10]
                })

            global_pos = val_idx[i]
            unseen_cands[retrieval_sources[0]][global_pos] = r550
            unseen_cands[retrieval_sources[1]][global_pos] = r554
            unseen_cands[retrieval_sources[2]][global_pos] = union_full
            unseen_cands[retrieval_sources[3]][global_pos] = [data.classes[c] for c in np.argsort(-nn_conv_probs[i])]
            unseen_cands[retrieval_sources[4]][global_pos] = [data.classes[c] for c in np.argsort(-nn_swin_probs[i])]
            unseen_cands[retrieval_sources[5]][global_pos] = nn_vis_ranked
            unseen_cands[retrieval_sources[6]][global_pos] = nn_text_ranked
            unseen_cands[retrieval_sources[7]][global_pos] = nn_mm_ranked
            unseen_cands[retrieval_sources[8]][global_pos] = exp_vis
            seen_pos = global_pos
            unseen_cands[retrieval_sources[9]][global_pos] = exp_text
            unseen_cands[retrieval_sources[10]][global_pos] = exp_mm
            unseen_cands[retrieval_sources[11]][global_pos] = blended_ranked

    # =========================================================================
    # COMPUTE SUMMARY RETRIEVAL METRICS
    # =========================================================================
    y_seen_true = [y_true_all[i] for i in sub_indices_global]
    audit_report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seen_metrics": {},
        "unseen_metrics": {},
        "composite_metrics": {},
        "champion_miss_recovery": {}
    }

    print("\n" + "=" * 115)
    print(f"{'Retrieval Source / Pool':40s} | {'Seen R@3':8s} | {'Seen R@5':8s} | {'Seen R@10':9s} | {'Unseen R@3':10s} | {'Unseen R@5':10s} | {'Unseen R@10':11s} | {'Comp R@5':8s}")
    print("-" * 115)

    base_seen_r5 = 0.0
    for name in retrieval_sources:
        s_rec = compute_recalls(y_seen_true, seen_cands[name])
        u_rec = compute_recalls(y_true_all, unseen_cands[name])

        comp_r3 = 0.957 * s_rec["Recall@3"] + 0.043 * u_rec["Recall@3"]
        comp_r5 = 0.957 * s_rec["Recall@5"] + 0.043 * u_rec["Recall@5"]
        comp_r10 = 0.957 * s_rec["Recall@10"] + 0.043 * u_rec["Recall@10"]

        # MAP@3
        s_map3 = calculate_map3(y_seen_true, [c[:3] for c in seen_cands[name]])
        u_map3 = calculate_map3(y_true_all, [c[:3] for c in unseen_cands[name]])
        comp_map3 = 0.957 * s_map3 + 0.043 * u_map3

        audit_report["seen_metrics"][name] = {**s_rec, "MAP@3": round(s_map3, 5)}
        audit_report["unseen_metrics"][name] = {**u_rec, "MAP@3": round(u_map3, 5)}
        audit_report["composite_metrics"][name] = {
            "Recall@3": round(comp_r3, 4),
            "Recall@5": round(comp_r5, 4),
            "Recall@10": round(comp_r10, 4),
            "MAP@3": round(comp_map3, 5)
        }

        print(f"{name:40s} | {s_rec['Recall@3']:.1%}   | {s_rec['Recall@5']:.1%}   | {s_rec['Recall@10']:.1%}    | {u_rec['Recall@3']:.1%}     | {u_rec['Recall@5']:.1%}     | {u_rec['Recall@10']:.1%}      | {comp_r5:.1%}")

    # =========================================================================
    # CHAMPION MISS RECOVERY AUDIT
    # =========================================================================
    n_seen_misses = len(seen_champion_misses)
    n_unseen_misses = len(unseen_champion_misses)

    print("\n" + "=" * 95)
    print("CHAMPION UNION TOP-3 MISS RECOVERY AUDIT")
    print("Traces where True Label is NOT in S550 Top-3 AND NOT in S554 Top-3")
    print("=" * 95)
    print(f"Total Seen Misses   : {n_seen_misses} / {n_seen} ({n_seen_misses/n_seen:.1%})")
    print(f"Total Unseen Misses : {n_unseen_misses} / {n_samples} ({n_unseen_misses/n_samples:.1%})")

    # Seen Recovery Stats
    def summarize_recovery(miss_list: List[Dict[str, Any]], title: str):
        total = len(miss_list)
        print(f"\n--- {title} (Total Misses = {total}) ---")
        for source_key, label in [("vis", "Visual NN"), ("text", "Text NN"), ("mm", "Multimodal NN")]:
            r5 = sum(1 for m in miss_list if m[f"recovered_{source_key}_5"])
            r10 = sum(1 for m in miss_list if m[f"recovered_{source_key}_10"])
            pct5 = (r5 / max(total, 1)) * 100.0
            pct10 = (r10 / max(total, 1)) * 100.0
            print(f"  {label:15s} -> Recovered @ 5: {r5:3d} ({pct5:5.1f}%) | Recovered @ 10: {r10:3d} ({pct10:5.1f}%)")

    summarize_recovery(seen_champion_misses, "Seen Questions Miss Recovery")
    summarize_recovery(unseen_champion_misses, "Unseen Questions Miss Recovery")

    # Recovery Breakdown by Question Support Category (Seen)
    cat_counts = Counter(m["cat"] for m in seen_champion_misses)
    print("\n--- Seen Miss Recovery Breakdown by Support Category ---")
    for cat, cnt in sorted(cat_counts.items()):
        sub_m = [m for m in seen_champion_misses if m["cat"] == cat]
        rec_mm5 = sum(1 for m in sub_m if m["recovered_mm_5"])
        rec_mm10 = sum(1 for m in sub_m if m["recovered_mm_10"])
        print(f"  Category: {cat:30s} | Total Misses: {cnt:3d} | MM Recovered @ 5: {rec_mm5:3d} ({rec_mm5/cnt:.1%}) | MM Recovered @ 10: {rec_mm10:3d} ({rec_mm10/cnt:.1%})")

    # Save artifact
    out_path = ART_DIR / "candidate_retrieval_nearest_neighbor_audit_report.json"
    audit_report["runtime_seconds"] = round(time.time() - start_time, 2)
    audit_report["champion_miss_recovery"] = {
        "seen_total_misses": n_seen_misses,
        "unseen_total_misses": n_unseen_misses,
        "seen_recovery_rates": {
            "visual_nn": {
                "r5": round(sum(1 for m in seen_champion_misses if m["recovered_vis_5"]) / max(n_seen_misses, 1), 4),
                "r10": round(sum(1 for m in seen_champion_misses if m["recovered_vis_10"]) / max(n_seen_misses, 1), 4)
            },
            "text_nn": {
                "r5": round(sum(1 for m in seen_champion_misses if m["recovered_text_5"]) / max(n_seen_misses, 1), 4),
                "r10": round(sum(1 for m in seen_champion_misses if m["recovered_text_10"]) / max(n_seen_misses, 1), 4)
            },
            "multimodal_nn": {
                "r5": round(sum(1 for m in seen_champion_misses if m["recovered_mm_5"]) / max(n_seen_misses, 1), 4),
                "r10": round(sum(1 for m in seen_champion_misses if m["recovered_mm_10"]) / max(n_seen_misses, 1), 4)
            }
        },
        "unseen_recovery_rates": {
            "visual_nn": {
                "r5": round(sum(1 for m in unseen_champion_misses if m["recovered_vis_5"]) / max(n_unseen_misses, 1), 4),
                "r10": round(sum(1 for m in unseen_champion_misses if m["recovered_vis_10"]) / max(n_unseen_misses, 1), 4)
            },
            "text_nn": {
                "r5": round(sum(1 for m in unseen_champion_misses if m["recovered_text_5"]) / max(n_unseen_misses, 1), 4),
                "r10": round(sum(1 for m in unseen_champion_misses if m["recovered_text_10"]) / max(n_unseen_misses, 1), 4)
            },
            "multimodal_nn": {
                "r5": round(sum(1 for m in unseen_champion_misses if m["recovered_mm_5"]) / max(n_unseen_misses, 1), 4),
                "r10": round(sum(1 for m in unseen_champion_misses if m["recovered_mm_10"]) / max(n_unseen_misses, 1), 4)
            }
        }
    }

    with open(out_path, "w") as f:
        json.dump(audit_report, f, indent=2)
    print(f"\n[Saved Audit Report] {out_path.name}")

if __name__ == "__main__":
    main()

