"""Distill 0.55049 SOTA Champion Soft Targets for 3,500 Unlabelled Traces.

Replaces the legacy primitive teacher (0.67 Prior + 0.33 Text) with the
empirically validated 0.55049 SOTA Champion Ensemble Teacher:
1. Adaptive Question Prior (empirical question distribution + semantic fallback).
2. Frequency Recalibration Multipliers (mult).
3. Transductive 1.20x Question Boosting (strictly on train-observed misconceptions).
4. Probability Simplex normalization (sum to 1.0, non-negative).

Outputs:
- artifacts/pseudo_3500_champion_soft_targets_76.npy (3500, 76)
"""
import json
from pathlib import Path
import sys
from typing import Dict, List, Set, cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.diagnoser.baseline_text import TextMisconceptionDiagnoser

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"


def distill_champion_soft_targets():
    print("=" * 80)
    print("DISTILLING 0.55049 SOTA CHAMPION TARGETS ON 3,500 UNLABELLED TRACES")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    meta_df = cast(pd.DataFrame, pd.read_csv(ART_DIR / "pseudo_3500_metadata.csv"))
    n_pseudo = len(meta_df)
    print(f"Loaded {n_pseudo} traces across {meta_df['question_id'].nunique()} questions.")

    # 1. Fit Adaptive Question Prior
    print("[1/4] Fitting Adaptive Question Prior...")
    prior_engine = AdaptiveQuestionPriorEngine(
        classes=classes, m_q_base=10.0, m_tg=10.0, adaptive_nq=True, adaptive_power=0.70,
        use_semantic_backoff=False
    ).fit(train_df)
    pseudo_prior = prior_engine.predict_proba(meta_df)

    # 2. Fit Text Diagnoser
    print("[2/4] Fitting Domain Text Diagnoser...")
    text_model = TextMisconceptionDiagnoser(taxonomy_manager=tax, alpha=5e-5)
    text_model.fit(train_df)
    raw_text_probs = text_model.predict_proba(meta_df)

    aligned_text_probs = np.zeros((n_pseudo, n_classes), dtype=np.float32)
    for m_idx, c in enumerate(text_model.classes_):
        if c in classes:
            aligned_text_probs[:, classes.index(c)] = raw_text_probs[:, m_idx]

    # 3. Base Champion Calibration (mult)
    print("[3/4] Applying Champion Frequency Multiplier Calibration...")
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    from collections import Counter
    top3_train = np.argsort(-oof_m15.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_train for idx in row)
    true_counts = Counter(train_df["root_code"].fillna("NONE").values)

    mult = np.ones(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840

    # Base weighted probability (70% Prior + 30% Text)
    p_base = (0.70 * pseudo_prior + 0.30 * aligned_text_probs) * mult
    p_base[:, none_idx] *= 0.9995

    # 4. Transductive Question Boost (1.20x on train-observed misconceptions)
    print("[4/4] Applying Transductive 1.20x Question Conditioning...")
    q_allowed = {q: set(grp["root_code"].dropna().unique()) for q, grp in train_df.groupby("question_id")}
    p_boosted = np.copy(p_base)
    for i in range(n_pseudo):
        qid = meta_df.iloc[i]["question_id"]
        if qid in q_allowed:
            for ci, code in enumerate(classes):
                if code != "NONE" and code in q_allowed[qid]:
                    p_boosted[i, ci] *= 1.20

    # 5. Composite SOTA Teacher (75% base + 25% boosted)
    p_champion = 0.75 * p_base + 0.25 * p_boosted

    # Project to probability simplex
    p_champion = np.clip(p_champion, 1e-7, 1.0)
    p_champion = p_champion / p_champion.sum(axis=1, keepdims=True)
    p_champion = p_champion.astype(np.float32)

    # Sanity checks
    assert p_champion.shape == (3500, 76), f"Wrong shape: {p_champion.shape}"
    assert np.all(p_champion >= 0), "Negative probabilities found!"
    assert np.allclose(p_champion.sum(axis=1), 1.0, atol=1e-5), "Rows do not sum to 1.0!"

    out_path = ART_DIR / "pseudo_3500_champion_soft_targets_76.npy"
    np.save(out_path, p_champion)
    print(f"\n[SUCCESS] Saved: {out_path} ({p_champion.shape})")

    # Compare with old primitive targets
    old_targets = np.load(ART_DIR / "pseudo_3500_soft_targets_76.npy")
    diff = np.abs(p_champion - old_targets).mean()
    print(f"Mean Absolute Difference vs Legacy Primitive Teacher: {diff:.5f}")

    old_r1 = [classes[idx] for idx in np.argmax(old_targets, axis=1)]
    new_r1 = [classes[idx] for idx in np.argmax(p_champion, axis=1)]
    r1_diffs = sum(1 for a, b in zip(old_r1, new_r1) if a != b)
    print(f"Rank-1 Target Differences: {r1_diffs} / 3500 ({r1_diffs/3500:.1%})")
    print(f"Old Teacher NONE count: {sum(1 for x in old_r1 if x == 'NONE')} / 3500")
    print(f"Champion Teacher NONE count: {sum(1 for x in new_r1 if x == 'NONE')} / 3500")
    print("=" * 80)


if __name__ == "__main__":
    distill_champion_soft_targets()
