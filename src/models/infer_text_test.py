"""Generate 5-Fold Ensemble Test Probabilities for M5 Text Model.

Outputs:
- artifacts/test_text.npy: (3498, 76) aligned text test prediction probability matrix.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, PATHS
from src.data.dataset import MathMisconceptionData
from src.data.splits import FoldSplitter
from src.data.taxonomy import TaxonomyManager
from src.diagnoser.baseline_text import TextMisconceptionDiagnoser
from src.diagnoser.domain_mask import DomainConstraintEngine


def run_text_test_inference():
    print("[TextTest] Loading dataset, taxonomy, and cache...", flush=True)
    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    test_df = data_mgr.load_test()
    tax = TaxonomyManager(PATHS.taxonomy_csv)
    all_classes = tax.all_codes(include_none=True)

    # Attach transcriptions cache to test
    if PATHS.transcriptions_cache.exists():
        with open(PATHS.transcriptions_cache, "r", encoding="utf-8") as f:
            transcriptions = json.load(f)
        test_df["transcription_steps"] = test_df["trace_id"].map(transcriptions).fillna("")
        print(f"[TextTest] Loaded {len(transcriptions)} test transcriptions from cache.", flush=True)
    else:
        test_df["transcription_steps"] = ""

    # Load canonical splits
    splitter = FoldSplitter(n_splits=CONFIG.n_splits, random_state=CONFIG.random_state)
    folds = splitter.load_splits(PATHS.splits_cache)

    domain_masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df)

    n_test = len(test_df)
    test_topics = [str(t) for t in test_df["topic_name"].values]
    test_grades = [int(g) for g in test_df["grade"].values]

    ensemble_test_probs = np.zeros((n_test, len(all_classes)), dtype=np.float32)

    print(f"[TextTest] Predicting {n_test} test traces across {len(folds)} folds...", flush=True)

    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        fold_train = train_df.iloc[train_idx].reset_index(drop=True)

        model = TextMisconceptionDiagnoser(taxonomy_manager=tax, alpha=CONFIG.default_alpha)
        model.fit(fold_train)

        raw_test_probs = model.predict_proba(test_df)
        model_classes = list(model.classes_)

        # Align to master 76 classes
        aligned_probs = np.zeros((n_test, len(all_classes)), dtype=np.float32)
        for m_idx, class_name in enumerate(model_classes):
            if class_name in all_classes:
                master_idx = all_classes.index(class_name)
                aligned_probs[:, master_idx] = raw_test_probs[:, m_idx]

        # Row normalize
        row_sums = aligned_probs.sum(axis=1, keepdims=True)
        aligned_probs = np.divide(aligned_probs, row_sums, where=row_sums > 0)

        # Apply domain masking
        calibrated_probs = domain_masker.apply_mask_batch(
            aligned_probs, all_classes, test_topics, test_grades, damping=CONFIG.domain_mask_damping
        )

        ensemble_test_probs += calibrated_probs / len(folds)
        print(f"  Fold {fold_idx}/{len(folds)} text inference complete.", flush=True)

    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifacts_dir / "test_text.npy", ensemble_test_probs)
    print(f"[TextTest] Saved {artifacts_dir / 'test_text.npy'} with shape {ensemble_test_probs.shape}")

    return ensemble_test_probs


if __name__ == "__main__":
    run_text_test_inference()
