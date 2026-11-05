"""Extract and persist Multilingual Transformer Semantic Embeddings & Honest OOF Predictions.

Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 on cuda:0 to:
1. Batch encode unique question stems (topic, grade, question, correct answer) into 384-d vectors.
2. Fit in-fold regularized linear classifier on 5 folds to evaluate honest OOF MAP@3.
3. Apply in-fold domain constraints (DomainConstraintEngine).
4. Persist:
   - artifacts/oof_semantic_text.npy: (N_train, 76)
   - artifacts/test_semantic_text.npy: (N_test, 76)
   - artifacts/semantic_text_metadata.json
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, cast
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, PATHS
from src.data.dataset import MathMisconceptionData
from src.data.splits import FoldSplitter
from src.data.taxonomy import TaxonomyManager
from src.diagnoser.domain_mask import DomainConstraintEngine
from src.diagnoser.transformer_embed import MultilingualMathEmbedder
from src.metrics.map3 import calculate_map3


def format_question_context(row: pd.Series) -> str:
    topic = str(row.get("topic_name", "")).strip()
    grade = str(row.get("grade", "")).strip()
    q_text = str(row.get("question_text", "")).strip()
    ans = str(row.get("correct_answer", "")).strip()
    return f"Topik matematika: {topic}. Kelas {grade}. Soal: {q_text}. Kunci jawaban: {ans}."


def run_semantic_text_pipeline() -> Dict[str, Any]:
    print("\n" + "=" * 65)
    print("[SemanticText] Starting Multilingual Transformer Text Pipeline")
    print("=" * 65 + "\n")

    tax = TaxonomyManager(PATHS.taxonomy_csv)
    all_classes = tax.all_codes(include_none=True)
    n_classes = len(all_classes)

    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    test_df = data_mgr.load_test()
    n_train = len(train_df)
    n_test = len(test_df)
    y_true = train_df["root_code"].astype(str).values

    # 1. Initialize GPU embedder
    embedder = MultilingualMathEmbedder(taxonomy=tax, use_gpu=True)
    print(f"[SemanticText] Embedder running on device: {embedder.device}")

    # 2. Extract embeddings for train and test contexts
    print(f"[SemanticText] Formatting context texts for {n_train} train and {n_test} test rows...")
    train_texts = [format_question_context(train_df.iloc[i]) for i in range(n_train)]
    test_texts = [format_question_context(test_df.iloc[i]) for i in range(n_test)]

    print(f"[SemanticText] Encoding train texts into 384-d vectors...")
    train_embeds = embedder.encode_texts(train_texts, batch_size=64)
    print(f"[SemanticText] Encoding test texts into 384-d vectors...")
    test_embeds = embedder.encode_texts(test_texts, batch_size=64)

    # 3. 5-Fold Cross-Validation for Honest OOF
    splitter = FoldSplitter(n_splits=CONFIG.n_splits, random_state=CONFIG.random_state)
    folds = splitter.load_splits(PATHS.splits_cache)

    oof_probs = np.zeros((n_train, n_classes), dtype=np.float32)
    test_probs = np.zeros((n_test, n_classes), dtype=np.float32)
    fold_scores = []

    print(f"[SemanticText] Evaluating 5-fold honest OOF with in-fold domain masking...")
    for fold_idx, (tr_idx, val_idx) in enumerate(folds, start=1):
        tr_X, val_X = train_embeds[tr_idx], train_embeds[val_idx]
        tr_y, val_y = y_true[tr_idx], y_true[val_idx]

        fold_tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        # Train regularized linear model on dense semantic embeddings
        clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=42)
        clf.fit(tr_X, tr_y)

        # Raw predicted probabilities
        val_preds_raw = clf.predict_proba(val_X)
        test_preds_raw = clf.predict_proba(test_embeds)
        model_classes = list(clf.classes_)

        # Align to master 76 classes
        val_aligned = np.zeros((len(val_idx), n_classes), dtype=np.float32)
        test_aligned = np.zeros((n_test, n_classes), dtype=np.float32)
        for m_idx, c_name in enumerate(model_classes):
            if c_name in all_classes:
                master_idx = all_classes.index(c_name)
                val_aligned[:, master_idx] = val_preds_raw[:, m_idx]
                test_aligned[:, master_idx] = test_preds_raw[:, m_idx]

        # In-fold domain masking (strictly fit on fold_tr_df)
        in_fold_masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(fold_tr_df)
        val_topics = [str(t) for t in fold_val_df["topic_name"].values]
        val_grades = [int(float(g)) for g in fold_val_df["grade"].values]
        val_calibrated = in_fold_masker.apply_mask_batch(
            val_aligned, all_classes, val_topics, val_grades, damping=CONFIG.domain_mask_damping
        )

        oof_probs[val_idx] = val_calibrated

        # Accumulate fold test prediction
        test_topics = [str(t) for t in test_df["topic_name"].values]
        test_grades = [int(float(g)) for g in test_df["grade"].values]
        test_calibrated = in_fold_masker.apply_mask_batch(
            test_aligned, all_classes, test_topics, test_grades, damping=CONFIG.domain_mask_damping
        )
        test_probs += test_calibrated / len(folds)

        # Compute fold MAP@3
        top_idx = np.argsort(-val_calibrated, axis=1, kind="mergesort")[:, :3]
        fold_preds = [[all_classes[k] for k in row] for row in top_idx]
        fold_map = calculate_map3(list(val_y), fold_preds)
        fold_scores.append(fold_map)
        print(f"  Fold {fold_idx} Honest OOF MAP@3: {fold_map:.4f}")

    # Overall OOF score
    overall_top = np.argsort(-oof_probs, axis=1, kind="mergesort")[:, :3]
    overall_preds = [[all_classes[k] for k in row] for row in overall_top]
    overall_map3 = calculate_map3(list(y_true), overall_preds)
    mean_map3 = float(np.mean(fold_scores))
    std_map3 = float(np.std(fold_scores))
    print(f"\n[SemanticText] Overall Honest OOF MAP@3: {overall_map3:.4f} (Mean: {mean_map3:.4f} +/- {std_map3:.4f})")

    # Persist artifacts
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifacts_dir / "oof_semantic_text.npy", oof_probs)
    np.save(artifacts_dir / "test_semantic_text.npy", test_probs)
    np.save(artifacts_dir / "train_text_embeddings_384.npy", train_embeds)
    np.save(artifacts_dir / "test_text_embeddings_384.npy", test_embeds)

    meta = {
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "embedding_dim": 384,
        "n_train": n_train,
        "n_test": n_test,
        "honest_oof_map3": overall_map3,
        "fold_mean_map3": mean_map3,
        "fold_std_map3": std_map3
    }
    with open(artifacts_dir / "semantic_text_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[SemanticText] Saved artifacts to {artifacts_dir}/oof_semantic_text.npy and test_semantic_text.npy")
    return meta


if __name__ == "__main__":
    run_semantic_text_pipeline()
