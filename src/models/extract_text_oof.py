"""Extract and persist baseline M5 text model Out-Of-Fold (OOF) probabilities.

Saves:
- data/splits.json: Canonical 5-fold split assignments.
- artifacts/oof_text.npy: (1500, 76) aligned text prediction probability matrix.
- artifacts/taxonomy_classes.json: Aligned 76-class ordering.
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
from src.metrics.map3 import calculate_map3


def run_text_oof_extraction():
    print("[TextOOF] Initializing dataset and authoritative taxonomy...")
    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    tax = TaxonomyManager(PATHS.taxonomy_csv)
    all_classes = tax.all_codes(include_none=True)
    n_samples = len(train_df)

    # 1. Generate & persist reproducible 5-fold splits
    splitter = FoldSplitter(n_splits=CONFIG.n_splits, random_state=CONFIG.random_state)
    folds = splitter.create_folds(train_df, target_col="root_code")
    splitter.save_splits(folds, PATHS.splits_cache)
    print(f"[TextOOF] Saved reproducible 5-fold splits to: {PATHS.splits_cache}")

    # 2. Setup domain constraint engine
    domain_masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df)

    oof_probs = np.zeros((n_samples, len(all_classes)), dtype=np.float32)
    y_true = train_df["root_code"].astype(str).values

    print(f"[TextOOF] Training M5 (alpha={CONFIG.default_alpha}) across {len(folds)} folds...")
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val = train_df.iloc[val_idx].reset_index(drop=True)

        model = TextMisconceptionDiagnoser(taxonomy_manager=tax, alpha=CONFIG.default_alpha)
        model.fit(fold_train)

        raw_val_probs = model.predict_proba(fold_val)
        model_classes = list(model.classes_)

        # Align to master 76 classes
        aligned_probs = np.zeros((len(val_idx), len(all_classes)), dtype=np.float32)
        for m_idx, class_name in enumerate(model_classes):
            if class_name in all_classes:
                master_idx = all_classes.index(class_name)
                aligned_probs[:, master_idx] = raw_val_probs[:, m_idx]

        # Row normalize
        row_sums = aligned_probs.sum(axis=1, keepdims=True)
        aligned_probs = np.divide(aligned_probs, row_sums, where=row_sums > 0)

        # Apply domain masking
        topics = [str(t) for t in fold_val["topic_name"].values]
        grades = [int(g) for g in fold_val["grade"].values]
        calibrated_probs = domain_masker.apply_mask_batch(
            aligned_probs, all_classes, topics, grades, damping=CONFIG.domain_mask_damping
        )

        oof_probs[val_idx] = calibrated_probs

        # Calculate fold MAP@3
        top_indices = np.argsort(-calibrated_probs, axis=1, kind="mergesort")[:, :3]
        fold_preds = [[all_classes[idx] for idx in row] for row in top_indices]
        fold_y_true = [str(y_true[idx]) for idx in val_idx]
        fold_map = calculate_map3(fold_y_true, fold_preds)
        fold_scores.append(fold_map)
        print(f"  Fold {fold_idx}/{CONFIG.n_splits} MAP@3: {fold_map:.4f}")

    # Overall OOF MAP@3
    overall_top = np.argsort(-oof_probs, axis=1, kind="mergesort")[:, :3]
    overall_preds = [[all_classes[idx] for idx in row] for row in overall_top]
    overall_map3 = calculate_map3(list(y_true), overall_preds)
    print(f"[TextOOF] Overall M5 OOF MAP@3: {overall_map3:.4f} (Mean: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f})")

    # Persist artifacts
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifacts_dir / "oof_text.npy", oof_probs)
    with open(artifacts_dir / "taxonomy_classes.json", "w", encoding="utf-8") as f:
        json.dump(all_classes, f, indent=2)

    print(f"[TextOOF] Saved {artifacts_dir / 'oof_text.npy'} with shape {oof_probs.shape}")
    return oof_probs, all_classes, overall_map3


if __name__ == "__main__":
    run_text_oof_extraction()
