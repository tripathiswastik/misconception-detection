"""Audit Semantic Question Similarity and Unlabelled Question Traces.

Analyzes:
1. For the 303 test questions NOT in train.csv:
   - Question text cosine similarities to train questions
   - Distribution of maximum similarities
2. For the 2,835 unlabelled traces on the 375 test questions:
   - Question representation and trace counts
"""
from pathlib import Path
import sys
from typing import cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

def main():
    print("=" * 80, flush=True)
    print("AUDITING SEMANTIC QUESTION SIMILARITY & UNLABELLED QUESTIONS", flush=True)
    print("=" * 80, flush=True)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    unl_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "unlabelled.csv"))

    train_qids = train_df["question_id"].unique()
    test_qids = test_df["question_id"].unique()

    print(f"Total Unique Questions in Train: {len(train_qids)}")
    print(f"Total Unique Questions in Test : {len(test_qids)}")

    overlap_qids = set(train_qids) & set(test_qids)
    unseen_test_qids = set(test_qids) - set(train_qids)

    print(f"Test Questions in Train        : {len(overlap_qids)} / {len(test_qids)} ({len(overlap_qids)/len(test_qids):.1%})")
    print(f"Test Questions UNSEEN in Train : {len(unseen_test_qids)} / {len(test_qids)} ({len(unseen_test_qids)/len(test_qids):.1%})")

    # Load 384-d text embeddings
    tr_emb = np.load(ART_DIR / "train_text_embeddings_384.npy")
    te_emb = np.load(ART_DIR / "test_text_embeddings_384.npy")

    # Extract unique question embeddings
    # Question text embeddings are identical for all traces of the same question
    tr_q_map = {}
    for i, qid in enumerate(train_df["question_id"]):
        if qid not in tr_q_map:
            tr_q_map[qid] = tr_emb[i] / np.linalg.norm(tr_emb[i])

    te_q_map = {}
    for i, qid in enumerate(test_df["question_id"]):
        if qid not in te_q_map:
            te_q_map[qid] = te_emb[i] / np.linalg.norm(te_emb[i])

    tr_q_matrix = np.array([tr_q_map[qid] for qid in train_qids])
    print(f"\nUnique Question Embedding Matrix: Train {tr_q_matrix.shape}")

    # For each unseen test question, find cosine similarity to train questions
    sims = []
    top_sim_per_q = []
    for qid in unseen_test_qids:
        q_vec = te_q_map[qid]
        cos_sims = tr_q_matrix @ q_vec
        best_idx = np.argmax(cos_sims)
        best_sim = cos_sims[best_idx]
        best_train_qid = train_qids[best_idx]
        top_sim_per_q.append((qid, best_train_qid, best_sim))
        sims.append(best_sim)

    sims = np.array(sims)
    print(f"\nCosine Similarity of 303 Unseen Test Questions to Nearest Train Question:")
    print(f"  Mean Top-1 Similarity: {sims.mean():.4f}")
    print(f"  Median Top-1 Sim    : {np.median(sims):.4f}")
    print(f"  Min Top-1 Sim       : {sims.min():.4f}")
    print(f"  Max Top-1 Sim       : {sims.max():.4f}")
    print(f"  Sim >= 0.90         : {(sims >= 0.90).sum()} / {len(sims)} ({(sims >= 0.90).mean():.1%})")
    print(f"  Sim >= 0.85         : {(sims >= 0.85).sum()} / {len(sims)} ({(sims >= 0.85).mean():.1%})")
    print(f"  Sim >= 0.80         : {(sims >= 0.80).sum()} / {len(sims)} ({(sims >= 0.80).mean():.1%})")

    # Sample top matching questions
    top_sim_per_q.sort(key=lambda x: x[2], reverse=True)
    print("\nTop 5 Closest Test-Train Question Pairs:")
    for q_te, q_tr, sim in top_sim_per_q[:5]:
        te_text = test_df[test_df["question_id"] == q_te]["question_text"].iloc[0][:60]
        tr_text = train_df[train_df["question_id"] == q_tr]["question_text"].iloc[0][:60]
        tr_misc = list(set(train_df[train_df["question_id"] == q_tr]["root_code"].dropna()))
        print(f"  Sim={sim:.4f} | Test: '{te_text}...'")
        print(f"               Train: '{tr_text}...' -> Misconceptions: {tr_misc}")

    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()

