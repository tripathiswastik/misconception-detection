"""Check test qualifying rows for precision-targeted flips."""
import sys
from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine

art = PROJECT_ROOT / "artifacts"
sub_dir = PROJECT_ROOT / "submissions"
data_dir = PROJECT_ROOT / "data"

tax = TaxonomyManager(data_dir / "taxonomy_v1.csv")
classes = tax.all_codes(include_none=True)
none_idx = classes.index("NONE")

train_df = cast(pd.DataFrame, pd.read_csv(data_dir / "train.csv"))
test_df = cast(pd.DataFrame, pd.read_csv(data_dir / "test.csv"))
q_allowed = {q: set(grp["root_code"].dropna().unique()) for q, grp in train_df.groupby("question_id")}

s549 = cast(pd.DataFrame, pd.read_csv(sub_dir / "blend_breakthrough_sota551_optimal_submission.csv"))
preds549 = s549["root_code"].tolist()
r1_549 = [p.split()[0] for p in preds549]

p_bin = np.load(art / "test_compact_binary_prob.npy")

test_m15 = np.load(art / "test_m15_balanced_fusion.npy")
test_dual448 = np.load(art / "test_arcface_dualhead_448_soft.npy")
test_comp448 = np.load(art / "test_arcface_composite_448.npy")
test_arc3500 = np.load(art / "test_arcface_pseudo_3500.npy")
p_base = 0.95 * test_m15 + 0.02 * test_dual448 + 0.03 * test_comp448

oof_m15 = np.load(art / "oof_m15_balanced_fusion.npy")
p_oof_base = 0.95 * oof_m15 + 0.02 * np.load(art / "oof_arcface_dualhead_448_soft.npy") + 0.03 * np.load(art / "oof_arcface_composite_448.npy")
from collections import Counter
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

p_test_boosted = np.copy(p_test_cal)
for i in range(len(test_df)):
    qid = test_df.iloc[i]["question_id"]
    if qid in q_allowed:
        for ci, c in enumerate(classes):
            if c != "NONE" and c in q_allowed[qid]:
                p_test_boosted[i, ci] *= 1.20

p_test_arc3500_cal = (0.95 * test_m15 + 0.05 * test_arc3500) * mult
p_champ = 0.70 * p_test_cal + 0.25 * p_test_boosted + 0.05 * p_test_arc3500_cal

prior_engine = AdaptiveQuestionPriorEngine(classes=classes).fit(train_df)
test_prior_probs = prior_engine.predict_proba(test_df)
def sharp(probs, t):
    s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    return s / s.sum(axis=1, keepdims=True)
p_clean = (
    0.6448 * test_prior_probs +
    0.0839 * sharp(np.load(art / "test_text.npy"), 0.6) +
    0.0401 * sharp(np.load(art / "test_semantic_text.npy"), 0.6) +
    0.0458 * np.load(art / "test_convnext_vision.npy") +
    0.0454 * np.load(art / "test_swin_vision.npy") +
    0.0483 * np.load(art / "test_finetuned_vision.npy") +
    0.0455 * np.load(art / "test_trocr_vision.npy") +
    0.0462 * sharp(np.load(art / "test_clean_arcface_2328.npy"), 0.8)
)
p_final = 0.96 * p_champ + 0.04 * p_clean

print("Auditing qualified test flips across gating thresholds:")
for tau, eps in [(0.80, 0.03), (0.78, 0.03), (0.78, 0.04), (0.78, 0.05), (0.76, 0.03), (0.76, 0.04), (0.76, 0.05)]:
    qual_rows = []
    for i in range(len(test_df)):
        if r1_549[i] == "NONE" and p_bin[i] >= tau:
            row_p = p_final[i].copy()
            p_none_val = row_p[none_idx]
            row_p[none_idx] = -1.0
            top_misc_idx = int(np.argmax(row_p))
            top_misc_code = classes[top_misc_idx]
            p_misc_val = row_p[top_misc_idx]
            gap = p_none_val - p_misc_val
            if 0 < gap < eps:
                qid = test_df.iloc[i]["question_id"]
                if top_misc_code in q_allowed.get(qid, set()):
                    qual_rows.append(i)
    print(f"tau={tau:.2f}, eps={eps:.2f} -> Qualified Test Flips: {len(qual_rows)} / {len(test_df)} ({len(qual_rows)/len(test_df):.2%})")

