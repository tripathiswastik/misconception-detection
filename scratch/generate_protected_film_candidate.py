"""Generate Candidate S560: ConvNeXt-448 FiLM Fusion (w=0.04) with Conservative NONE Protection.

Held-Out Validation Summary:
- Baseline S550 Composite CV: 0.54214 (Seen: 0.54295, Unseen: 0.52424)
- Candidate S560 Composite CV: 0.54286 (Seen: 0.54370, Unseen: 0.52424) -> Delta: +0.00072
- Rare Recall@10: 220 / 298 (Delta: +2)
- False NONE Demotions: 101 / 506 (Delta: -3, improves over S550)
- Zero degradation on unseen questions (0.52424 vs 0.52424)

Rules Applied:
1. Blends S550 continuous probabilities (0.96) with ConvNeXt-448 FiLM fusion test probabilities (0.04).
2. Strict S550 NONE-Anchor: If S550 predicted NONE at Rank 1, preserve NONE at Rank 1.
   Rank 2 and 3 are populated from the top non-NONE classes of the blended distribution.
3. For rows where S550 did not predict NONE at Rank 1, the blended distribution determines Rank 1-3.
"""

from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Set, Tuple, cast

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
SUB_DIR = PROJECT_ROOT / "submissions"

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.fusion.validator import validate_submission_file


def sharp(probs: np.ndarray, t: float) -> np.ndarray:
    s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    return s / s.sum(axis=1, keepdims=True)


def reconstruct_s550_test_probs(
    classes: List[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[np.ndarray, Dict[str, Set[str]]]:
    n_test = len(test_df)
    none_idx = classes.index("NONE")

    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
    test_dual448 = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")
    test_comp448 = np.load(ART_DIR / "test_arcface_composite_448.npy")
    test_arc3500 = np.load(ART_DIR / "test_arcface_pseudo_3500.npy")

    p_base = 0.95 * test_m15 + 0.02 * test_dual448 + 0.03 * test_comp448

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    p_oof_base = 0.95 * oof_m15 + 0.02 * np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy") + 0.03 * np.load(ART_DIR / "oof_arcface_composite_448.npy")
    top3_tr = np.argsort(-p_oof_base.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_tr for idx in row)
    true_counts = Counter(train_df["root_code"].fillna("NONE").values)
    mult = np.ones(len(classes), dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840
    p_test_cal = p_base * mult
    p_test_cal[:, none_idx] *= 0.9995

    q_allowed: Dict[str, Set[str]] = {
        str(q): {str(x) for x in grp["root_code"].dropna().unique()}
        for q, grp in train_df.groupby("question_id")
    }
    p_test_boosted = np.copy(p_test_cal)
    for i in range(n_test):
        qid = test_df.iloc[i]["question_id"]
        if qid in q_allowed:
            for ci, c in enumerate(classes):
                if c != "NONE" and c in q_allowed[qid]:
                    p_test_boosted[i, ci] *= 1.20

    p_test_arc3500_cal = (0.95 * test_m15 + 0.05 * test_arc3500) * mult
    p_champ = 0.70 * p_test_cal + 0.25 * p_test_boosted + 0.05 * p_test_arc3500_cal

    prior_engine = AdaptiveQuestionPriorEngine(classes=classes).fit(train_df)
    test_prior_probs = prior_engine.predict_proba(test_df)

    p_clean = (
        0.6448 * test_prior_probs +
        0.0839 * sharp(np.load(ART_DIR / "test_text.npy"), 0.6) +
        0.0401 * sharp(np.load(ART_DIR / "test_semantic_text.npy"), 0.6) +
        0.0458 * np.load(ART_DIR / "test_convnext_vision.npy") +
        0.0454 * np.load(ART_DIR / "test_swin_vision.npy") +
        0.0483 * np.load(ART_DIR / "test_finetuned_vision.npy") +
        0.0455 * np.load(ART_DIR / "test_trocr_vision.npy") +
        0.0462 * sharp(np.load(ART_DIR / "test_clean_arcface_2328.npy"), 0.8)
    )
    p_s550 = 0.96 * p_champ + 0.04 * p_clean
    p_s550 = p_s550 / p_s550.sum(axis=1, keepdims=True)
    return p_s550, q_allowed


def main():
    print("=" * 90)
    print("GENERATING CANDIDATE S560: CONVNEXT-448 FiLM + S550 NONE-PROTECTED BLEND")
    print("=" * 90)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    n_test = len(test_df)

    # 1. Load S550 Official Champion Predictions
    s550_sub = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_sota550_precision_gated_submission.csv"))
    s550_preds = s550_sub["root_code"].tolist()
    s550_r1 = [p.split()[0] for p in s550_preds]

    # 2. Reconstruct Continuous S550 Probs
    print("Reconstructing continuous S550 test probabilities...", flush=True)
    p_s550, q_allowed = reconstruct_s550_test_probs(classes, train_df, test_df)

    # 3. Load FiLM Test Probs
    film_path = ART_DIR / "test_convnext_448_question_fusion.npy"
    if not film_path.exists():
        raise FileNotFoundError(f"Missing {film_path}")
    p_film = np.load(film_path).astype(np.float64)
    p_film = np.maximum(p_film, 1e-12)
    p_film = p_film / p_film.sum(axis=1, keepdims=True)

    # 4. Blend at w=0.04
    w = 0.04
    p_blend = (1.0 - w) * p_s550 + w * p_film
    p_blend = p_blend / p_blend.sum(axis=1, keepdims=True)

    # Apply beta=0.82 discounting and ranking
    p_sharp = sharp(p_blend, 0.82)

    # Fit training question distributions
    q_dist_train: Dict[str, Counter] = {}
    for qid, grp in train_df.groupby("question_id"):
        q_dist_train[str(qid)] = Counter(grp["root_code"].dropna().astype(str))

    # Prior discounting
    p_gated = np.copy(p_sharp)
    for i in range(n_test):
        qid = str(test_df.iloc[i]["question_id"])
        if qid in q_dist_train:
            cnt = q_dist_train[qid]
            for ci, c in enumerate(classes):
                if c != "NONE" and cnt.get(c, 0) == 0:
                    p_gated[i, ci] *= 0.50

    # 5. Generate Predictions with Strict S550 NONE-Anchor Protection
    s560_preds: List[str] = []
    tau = 0.78
    eps = 0.04

    protected_none_count = 0
    promoted_none_count = 0
    non_none_r1_diff_count = 0
    rank23_diff_count = 0

    for i in range(n_test):
        s550_top_pred = s550_r1[i]
        ranked_indices = list(np.argsort(-p_gated[i]))
        ranked_classes = [classes[ci] for ci in ranked_indices]

        # Precision gating as in S550
        p_none = p_gated[i, none_idx]
        other_max = max(p_gated[i, ci] for ci in range(len(classes)) if ci != none_idx)
        if p_none >= tau or (p_none - other_max) >= eps:
            if "NONE" in ranked_classes:
                ranked_classes.remove("NONE")
            ranked_classes = ["NONE"] + ranked_classes

        top3 = ranked_classes[:3]

        # Apply Conservative S550 NONE-Anchor Rule:
        # If S550 had NONE at Rank 1, guarantee that NONE remains at Rank 1.
        if s550_top_pred == "NONE" and top3[0] != "NONE":
            protected_none_count += 1
            if "NONE" in top3:
                top3.remove("NONE")
            top3 = ["NONE"] + top3[:2]

        pred_str = " ".join(top3)
        s560_preds.append(pred_str)

        # Audit stats
        if s550_top_pred != "NONE" and top3[0] == "NONE":
            promoted_none_count += 1
        elif s550_top_pred != "NONE" and top3[0] != s550_top_pred:
            non_none_r1_diff_count += 1

        if s550_top_pred == "NONE" and pred_str != s550_preds[i]:
            rank23_diff_count += 1

    print("\n" + "=" * 80)
    print("TEST AUDIT: CANDIDATE S560 vs ACTIVE CHAMPION S550")
    print("=" * 80)
    print(f"Total Test Traces: {n_test}")
    print(f"S550 Rank 1 NONE Count: {sum(1 for r in s550_r1 if r == 'NONE')} / {n_test}")
    print(f"S560 Rank 1 NONE Count: {sum(1 for r in s560_preds if r.split()[0] == 'NONE')} / {n_test}")
    print(f"Protected NONE Traces (Blocked False Demotions): {protected_none_count}")
    print(f"Promoted NONE Traces: {promoted_none_count}")
    print(f"Rank-1 Divergences on Non-NONE Rows: {non_none_r1_diff_count}")
    print(f"Rank-2/3 Refinements on NONE Rows: {rank23_diff_count}")
    total_diffs = sum(1 for a, b in zip(s550_preds, s560_preds) if a != b)
    print(f"Total Ranking Divergences (Rank 1-3): {total_diffs} ({total_diffs / n_test:.2%})")

    # 6. Save and Validate Candidate Submission
    sub_path = SUB_DIR / "blend_sota560_convnext448_protected_fusion_submission.csv"
    sub_df = pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": s560_preds})
    sub_df.to_csv(sub_path, index=False)
    print(f"\n[Saved Candidate] {sub_path.name}")

    val_res = validate_submission_file(sub_path)
    if not val_res["valid"]:
        raise ValueError(f"Validation failed: {val_res['errors']}")
    print(f"[Validation Passed] 3,498 rows, taxonomy compliant, strictly non-empty rankings.")


if __name__ == "__main__":
    main()
