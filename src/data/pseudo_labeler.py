"""High-Confidence Pseudo-Labeling Pipeline for Rare Misconceptions.

Identifies high-confidence student traces in data/unlabelled.csv for the 54
rare misconception classes (<15 training examples), expanding dataset support
to eliminate the primary bottleneck causing 296 near-misses.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, cast

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.baseline_text import TextMisconceptionDiagnoser
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine


def run_pseudo_labeling(max_samples_per_class: int = 15, target_threshold: float = 0.05) -> pd.DataFrame:
    print("=" * 70)
    print("[PseudoLabeler] Starting Semi-Supervised Data Expansion")
    print("=" * 70)

    tax = TaxonomyManager(PROJECT_ROOT / "data" / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)

    train_df = cast(pd.DataFrame, pd.read_csv(PROJECT_ROOT / "data" / "train.csv"))
    unl_df = cast(pd.DataFrame, pd.read_csv(PROJECT_ROOT / "data" / "unlabelled.csv"))

    train_counts = train_df["root_code"].value_counts().to_dict()
    rare_classes = [c for c in classes if c != "NONE" and train_counts.get(c, 0) < 15]
    print(f"Targeting {len(rare_classes)} rare classes (< 15 train examples)")

    # 1. Unique questions inference
    dup_q = cast(pd.DataFrame, unl_df.drop_duplicates(subset=["question_id"]))
    unique_q_df = dup_q.reset_index(drop=True)
    print(f"Processing {len(unique_q_df)} unique questions across {len(unl_df)} unlabelled traces...")

    t0 = time.time()
    # Fit text diagnoser
    text_model = TextMisconceptionDiagnoser(taxonomy_manager=tax, alpha=5e-5)
    text_model.fit(train_df)
    raw_text_probs = text_model.predict_proba(unique_q_df)

    aligned_text_probs = np.zeros((len(unique_q_df), len(classes)), dtype=np.float32)
    for m_idx, c in enumerate(text_model.classes_):
        if c in classes:
            aligned_text_probs[:, classes.index(c)] = raw_text_probs[:, m_idx]

    # Fit adaptive question prior
    prior_engine = AdaptiveQuestionPriorEngine(
        classes=classes, m_q_base=10.0, m_tg=10.0, adaptive_nq=True, adaptive_power=0.70,
        use_semantic_backoff=False
    )
    prior_engine.fit(train_df)
    q_prior_probs = prior_engine.predict_proba(unique_q_df)

    # Weighted question distribution
    w = np.array([0.4900, 0.2384])
    w = w / w.sum()
    q_dist = w[0] * q_prior_probs + w[1] * aligned_text_probs
    print(f"Computed question candidate distributions in {time.time() - t0:.1f}s")

    # Map question_id to probabilities
    q_to_probs: Dict[str, np.ndarray] = {
        unique_q_df.iloc[i]["question_id"]: q_dist[i] for i in range(len(unique_q_df))
    }

    # Group unlabelled traces by question_id
    q_to_traces = cast(Dict[str, List[Dict[str, Any]]], unl_df.groupby("question_id").apply(lambda g: g.to_dict(orient="records")).to_dict())

    selected_rows = []
    class_harvest_counts = {c: 0 for c in rare_classes}

    # For each rare class, find questions with highest affinity
    for c in rare_classes:
        c_idx = classes.index(c)
        # Rank questions by score for this class
        q_scores = [(q_id, q_to_probs[q_id][c_idx]) for q_id in q_to_probs]
        q_scores.sort(key=lambda x: -x[1])

        # Pick top questions with non-trivial score
        for q_id, score in q_scores:
            if class_harvest_counts[c] >= max_samples_per_class:
                break
            if score < target_threshold:
                continue

            available_traces = q_to_traces.get(q_id, [])
            for trace in available_traces:
                if class_harvest_counts[c] >= max_samples_per_class:
                    break
                trace_copy = dict(trace)
                trace_copy["root_code"] = c
                trace_copy["pseudo_confidence"] = float(score)
                selected_rows.append(trace_copy)
                class_harvest_counts[c] += 1

    pseudo_df = pd.DataFrame(selected_rows)
    print(f"\n[PseudoLabeler] Harvested {len(pseudo_df)} high-confidence traces across rare classes!")
    print(f"Classes with >= 5 new traces: {sum(1 for v in class_harvest_counts.values() if v >= 5)} / {len(rare_classes)}")
    print(f"Classes with >= 10 new traces: {sum(1 for v in class_harvest_counts.values() if v >= 10)} / {len(rare_classes)}")

    out_csv = PROJECT_ROOT / "artifacts" / "pseudo_labeled_rare_traces.csv"
    pseudo_df.to_csv(out_csv, index=False)
    print(f"Saved pseudo-labeled dataset: {out_csv}")

    return pseudo_df


if __name__ == "__main__":
    run_pseudo_labeling()

