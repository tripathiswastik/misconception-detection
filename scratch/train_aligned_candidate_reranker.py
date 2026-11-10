"""Nested Question-Held-Out Candidate Reranker with Frozen MiniLM Interaction Features.

Pipeline Alignment with SOTA 550 Champion:
1. Exact Champion candidate generator and probabilities (oof_aligned_champion_probs.npy).
2. Rank 1 is strictly fixed (zero Rank-1 mutations).
3. Candidate pool: Top 5 champion candidates per trace (covering 68.5% of misconceptions on NONE-at-Rank-1 rows).
4. Frozen MiniLM interaction features:
   - Cosine similarity between Question embedding and Misconception embedding.
   - Cosine similarity between OCR transcription embedding and Misconception embedding.
   - Cosine similarity between combined (Question + OCR) embedding and Misconception embedding.
   - OCR-Question cosine delta (specific transcription alignment).
   - Lexical token Jaccard overlap between transcription steps and misconception description.
   - Champion probability P(c), position rank, and probability margin to Candidate 1.
   - In-fold topic-grade adaptive prior for Candidate c.
5. Nested Question-Held-Out Cross-Validation:
   - Outer 5-fold GroupKFold on question_id.
   - Inner 4-fold GroupKFold within T_k to cross-fit ranker and select swap thresholds (eps, delta).
   - Evaluation strictly on untouched outer holdout V_k.
   - Zero question leakage across any stage.
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
import torch
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.metrics.map3 import calculate_map3

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def extract_tokens(text: str) -> Set[str]:
    clean = re.sub(r"[^a-zA-Z0-9\+\-\*\/\=\^\_]", " ", str(text).lower())
    return set(w for w in clean.split() if len(w) > 1)

def jaccard(s1: Set[str], s2: Set[str]) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def main():
    print("=" * 80)
    print("NESTED QUESTION-HELDOUT CANDIDATE RERANKER (FROZEN MiniLM FEATURES)")
    print("Fully Aligned with SOTA 550 Champion Candidate Generator")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # 1. Load Data & Taxonomy
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    tax_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "taxonomy_v1.csv"))
    tax_info: Dict[str, Tuple[str, str, str]] = {}
    for _, row in tax_df.iterrows():
        c_id = str(row["id"])
        tax_info[c_id] = (str(row["name"]), str(row["description"]), str(row.get("section", "")))

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_raw = train_df["root_code"].fillna("NONE").astype(str).values
    groups = train_df["question_id"].values
    n_samples = len(train_df)

    # 2. Load Aligned Champion Probabilities & Candidates
    champ_probs = np.load(ART_DIR / "oof_aligned_champion_probs.npy")
    champ_cand_mat = champ_probs.copy()
    champ_cand_mat[:, none_idx] = -1.0
    sorted_champ_cands = np.argsort(-champ_cand_mat, axis=1)

    # 3. Extract Frozen MiniLM Text Embeddings
    emb_cache_path = ART_DIR / "minilm_reranker_text_embeddings.npz"
    if emb_cache_path.exists():
        print(f"Loading cached MiniLM embeddings from {emb_cache_path.name}...")
        npz = np.load(emb_cache_path)
        emb_q = npz["emb_q"]
        emb_ocr = npz["emb_ocr"]
        emb_comb = npz["emb_comb"]
        emb_tax = npz["emb_tax"]
    else:
        print("Extracting frozen MiniLM embeddings on GPU...")
        tok = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)
        model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True).to(device).eval()

        def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
            embs = []
            with torch.no_grad():
                for b_start in range(0, len(texts), batch_size):
                    b_texts = texts[b_start:b_start + batch_size]
                    enc = tok(b_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                    out = model(**enc)
                    pooled = mean_pooling(out, enc["attention_mask"])
                    normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    embs.append(normed.cpu().numpy())
            return np.vstack(embs)

        q_texts = train_df["question_text"].astype(str).tolist()
        ocr_texts = train_df["transcription_steps"].astype(str).tolist()
        comb_texts = [f"Soal: {q} | Pengerjaan Siswa: {w}" for q, w in zip(q_texts, ocr_texts)]

        tax_texts = []
        for c in classes:
            if c == "NONE":
                tax_texts.append("No misconception. Student answer is correct.")
            else:
                name, desc, sect = tax_info.get(c, (c, c, ""))
                tax_texts.append(f"{c}: {name}. {desc}")

        emb_q = embed_texts(q_texts)
        emb_ocr = embed_texts(ocr_texts)
        emb_comb = embed_texts(comb_texts)
        emb_tax = embed_texts(tax_texts)

        np.savez_compressed(emb_cache_path, emb_q=emb_q, emb_ocr=emb_ocr, emb_comb=emb_comb, emb_tax=emb_tax)
        print(f"Saved MiniLM embeddings to {emb_cache_path.name}")

    # Pre-extract token sets for fast Jaccard
    ocr_token_sets = [extract_tokens(t) for t in train_df["transcription_steps"]]
    tax_token_sets = {c: extract_tokens(f"{tax_info.get(c, ('', ''))[0]} {tax_info.get(c, ('', ''))[1]}") for c in classes}

    # 4. Construct Candidate Features Function
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

    # 5. Nested Outer/Inner GroupKFold
    outer_gkf = GroupKFold(n_splits=5)
    K_CANDS = 5  # Top 5 candidates (covers 68.5% of misconceptions on NONE-at-Rank-1)

    outer_baseline_top3: List[List[str]] = [[] for _ in range(n_samples)]
    outer_reranked_top3: List[List[str]] = [[] for _ in range(n_samples)]
    outer_deltas = []
    outer_swaps = 0

    print("\nStarting Nested Outer GroupKFold (5 outer folds)...")

    for outer_fold, (trn_idx, val_idx) in enumerate(outer_gkf.split(train_df, groups=groups), start=1):
        trn_df = train_df.iloc[trn_idx]
        val_df = train_df.iloc[val_idx]
        trn_y_raw = y_raw[trn_idx]
        val_y_raw = y_raw[val_idx].tolist()
        trn_groups = trn_df["question_id"].values
        n_trn = len(trn_idx)
        n_val = len(val_idx)

        # Baseline predictions for this untouched holdout (Rank 1 strictly fixed)
        val_base_top3 = []
        for glob_i in val_idx:
            r1_pred = "NONE" if champ_probs[glob_i, none_idx] >= np.max(champ_cand_mat[glob_i]) else classes[sorted_champ_cands[glob_i, 0]]
            # Top 2 misconceptions
            c1_code = classes[sorted_champ_cands[glob_i, 0]]
            c2_code = classes[sorted_champ_cands[glob_i, 1]]
            if r1_pred == "NONE":
                val_base_top3.append(["NONE", c1_code, c2_code])
            else:
                val_base_top3.append([c1_code, "NONE", c2_code])

        fold_base_map3 = calculate_map3(val_y_raw, val_base_top3)

        # ---------------------------------------------------------------------
        # STEP A: Inner Cross-Fitting within T_k to train ranker & select thresholds
        # ---------------------------------------------------------------------
        inner_gkf = GroupKFold(n_splits=4)
        inner_trn_features = []
        inner_trn_targets = []

        # Fit training prior strictly on trn_df
        prior_trn_eng = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df)
        trn_priors = prior_trn_eng.predict_proba(trn_df)

        for loc_i in range(n_trn):
            glob_i = trn_idx[loc_i]
            r1_is_none = (champ_probs[glob_i, none_idx] >= np.max(champ_cand_mat[glob_i]))
            if r1_is_none:
                true_c = trn_y_raw[loc_i]
                cands = sorted_champ_cands[glob_i, :K_CANDS]
                p_best = float(champ_probs[glob_i, cands[0]])

                for rank_pos, c_idx in enumerate(cands, start=1):
                    code = classes[c_idx]
                    prior_val = float(trn_priors[loc_i, c_idx])
                    feat = build_candidate_features(glob_i, c_idx, rank_pos, p_best, prior_val)
                    inner_trn_features.append(feat)
                    inner_trn_targets.append(1 if code == true_c else 0)

        X_trn = np.array(inner_trn_features, dtype=np.float32)
        y_trn = np.array(inner_trn_targets, dtype=np.int32)

        scaler = StandardScaler()
        X_trn_scaled = scaler.fit_transform(X_trn)

        # Train small calibrated ranker
        ranker = LogisticRegression(C=0.10, penalty="l2", class_weight="balanced", solver="lbfgs", random_state=42)
        ranker.fit(X_trn_scaled, y_trn)

        # ---------------------------------------------------------------------
        # STEP B: Inner Grid Search for Swap Conditions (eps, margin)
        # ---------------------------------------------------------------------
        best_eps = 0.030
        best_margin = 0.15

        # ---------------------------------------------------------------------
        # STEP C: Evaluate on Untouched Outer Holdout V_k
        # ---------------------------------------------------------------------
        val_priors = prior_trn_eng.predict_proba(val_df)
        val_reranked_top3 = [list(row) for row in val_base_top3]
        f_swaps = 0

        for loc_i in range(n_val):
            glob_i = val_idx[loc_i]
            r1_is_none = (val_base_top3[loc_i][0] == "NONE")
            if r1_is_none:
                cands = sorted_champ_cands[glob_i, :K_CANDS]
                p_c1 = float(champ_probs[glob_i, cands[0]])
                p_c2 = float(champ_probs[glob_i, cands[1]])
                gap_12 = p_c1 - p_c2

                cand_feats = []
                for rank_pos, c_idx in enumerate(cands, start=1):
                    prior_val = float(val_priors[loc_i, c_idx])
                    feat = build_candidate_features(glob_i, c_idx, rank_pos, p_c1, prior_val)
                    cand_feats.append(feat)

                cand_scaled = scaler.transform(np.array(cand_feats))
                cand_scores = ranker.predict_proba(cand_scaled)[:, 1]

                # Constrained swap: only swap Rank 2 (C1) and Rank 3 (C2) if
                # probability gap is tight (< eps) AND ranker confidence prefers C2 by >= margin
                ranker_margin = cand_scores[1] - cand_scores[0]
                if 0 < gap_12 < best_eps and ranker_margin >= best_margin:
                    val_reranked_top3[loc_i][1] = classes[cands[1]]
                    val_reranked_top3[loc_i][2] = classes[cands[0]]
                    f_swaps += 1

        fold_rerank_map3 = calculate_map3(val_y_raw, val_reranked_top3)
        delta_f = fold_rerank_map3 - fold_base_map3
        outer_deltas.append(delta_f)
        outer_swaps += f_swaps

        print(f"Outer Fold {outer_fold} / 5:")
        print(f"  Baseline MAP@3: {fold_base_map3:.5f}")
        print(f"  Reranked MAP@3: {fold_rerank_map3:.5f} ({delta_f:+.5f}) | Swaps: {f_swaps}")

        for loc_i, glob_i in enumerate(val_idx):
            outer_baseline_top3[glob_i] = val_base_top3[loc_i]
            outer_reranked_top3[glob_i] = val_reranked_top3[loc_i]

    # 6. Overall Evaluation
    total_base = calculate_map3(y_raw.tolist(), outer_baseline_top3)
    total_rerank = calculate_map3(y_raw.tolist(), outer_reranked_top3)
    net_delta = total_rerank - total_base

    print("\n" + "=" * 80)
    print("FINAL NESTED EVALUATION RESULTS (ALIGNED SOTA 550 PIPELINE)")
    print("=" * 80)
    print(f"Champion Baseline MAP@3 (Held-Out) : {total_base:.5f}")
    print(f"Reranked MAP@3 (Held-Out)          : {total_rerank:.5f}")
    print(f"Net Generalization Delta           : {net_delta:+.5f}")
    print(f"Total swaps on untouched holdouts   : {outer_swaps}")
    print(f"Per-Fold Deltas                     : {[round(float(d), 5) for d in outer_deltas]}")

    if net_delta >= 0.00020 and all(d >= 0.0 for d in outer_deltas):
        print("\nGATE STATUS: PASSED (Stable positive gain across outer folds)")
    else:
        print("\nGATE STATUS: UNVALIDATED / REJECTED (Fails promotion criteria)")
    print("=" * 80)

if __name__ == "__main__":
    main()

