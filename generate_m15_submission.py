"""Generate M15 Balanced Semantic-Prior Ensemble (OOF 0.53822)."""
import json
import sys
from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS
from src.data.taxonomy import TaxonomyManager
from src.fusion.validator import validate_submission_file
from src.metrics.map3 import calculate_map3

tax = TaxonomyManager(PATHS.taxonomy_csv)
classes = tax.all_codes(include_none=True)

train_df = cast(pd.DataFrame, pd.read_csv(PATHS.data_dir / "train.csv"))
test_df = cast(pd.DataFrame, pd.read_csv(PATHS.data_dir / "test.csv"))
y_true = train_df["root_code"].astype(str).tolist()

art = PROJECT_ROOT / "artifacts"
prior_oof = np.load(art / "best_adaptive_prior_oof.npy")
vis_cn_oof = np.load(art / "oof_convnext_vision.npy")
vis_mn_oof = np.load(art / "oof_finetuned_vision.npy")
swin_oof = np.load(art / "oof_swin_vision.npy")
text_sgd_oof = np.load(art / "oof_text.npy")
text_sem_oof = np.load(art / "oof_semantic_text.npy")

m12_oof = np.load(art / "oof_adaptive_fusion.npy")

# Optimal weights (OOF MAP@3 = 0.53822)
w = np.array([0.4900, 0.0191, 0.0497, 0.0077, 0.2384, 0.1951])
# Normalize to sum exactly to 1.0
w = w / w.sum()

m15_oof = (w[0] * prior_oof + w[1] * vis_cn_oof + w[2] * vis_mn_oof +
           w[3] * swin_oof + w[4] * text_sgd_oof + w[5] * text_sem_oof)

def eval_map3(probs):
    top3_idx = np.argsort(-probs, axis=1, kind="mergesort")[:, :3]
    preds = [[classes[idx] for idx in row] for row in top3_idx]
    return calculate_map3(y_true, preds)

def eval_per_sample(probs):
    top3_idx = np.argsort(-probs, axis=1, kind="mergesort")[:, :3]
    scores = np.zeros(len(y_true), dtype=np.float64)
    for i, target in enumerate(y_true):
        for rank in range(3):
            if classes[top3_idx[i, rank]] == target:
                scores[i] = 1.0 / (rank + 1)
                break
    return scores

m15_score = eval_map3(m15_oof)
m12_score = eval_map3(m12_oof)

m15_sample = eval_per_sample(m15_oof)
m12_sample = eval_per_sample(m12_oof)

# 10,000 Paired Bootstrap against M12 (0.53383 LB)
rng = np.random.default_rng(42)
n_boot = 10000
n_samples = len(y_true)
boot_deltas = np.zeros(n_boot)
deltas = m15_sample - m12_sample

for b in range(n_boot):
    idx = rng.integers(0, n_samples, size=n_samples)
    boot_deltas[b] = np.mean(deltas[idx])

mean_d = float(np.mean(boot_deltas))
ci_l = float(np.percentile(boot_deltas, 2.5))
ci_u = float(np.percentile(boot_deltas, 97.5))
p_val = float(np.mean(boot_deltas <= 0))

print("=" * 65)
print(f"M15 BALANCED SEMANTIC-PRIOR ENSEMBLE RESULTS")
print("=" * 65)
print(f"M15 Honest OOF MAP@3: {m15_score:.5f}")
print(f"M12 Baseline OOF:     {m12_score:.5f}")
print(f"Delta vs M12:         {mean_d:+.5f}")
print(f"95% Bootstrap CI:     [{ci_l:+.5f}, {ci_u:+.5f}]")
print(f"p-value:              {p_val:.4f} (Significant: {ci_l > 0 and p_val < 0.05})")

# Generate Test Predictions
print("\n[M15] Generating test set predictions...")
# 1. Test Prior
from src.data.splits import FoldSplitter
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.diagnoser.domain_mask import DomainConstraintEngine

splitter = FoldSplitter(n_splits=5, random_state=42)
folds = splitter.load_splits(PATHS.splits_cache)

train_embeds = np.load(art / "train_text_embeddings_384.npy")
test_embeds = np.load(art / "test_text_embeddings_384.npy")

test_prior = np.zeros((len(test_df), len(classes)), dtype=np.float32)
for tr_idx, _ in folds:
    fold_tr = train_df.iloc[tr_idx].reset_index(drop=True)
    engine = AdaptiveQuestionPriorEngine(
        classes=classes, m_q_base=10.0, m_tg=10.0, adaptive_nq=True, adaptive_power=0.70,
        use_semantic_backoff=True, semantic_embeddings=train_embeds, semantic_top_k=5,
        semantic_threshold=0.70, semantic_weight=0.50
    )
    engine.fit(fold_tr, fold_train_indices=tr_idx)
    test_prior += engine.predict_proba(test_df, eval_embeddings=test_embeds) / len(folds)

# 2. Test Vision
test_cn = np.load(art / "test_convnext_vision.npy")
test_mn_raw = np.load(art / "test_finetuned_vision.npy")
test_swin = np.load(art / "test_swin_vision.npy")

domain_masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df)
test_topics = [str(t) for t in test_df["topic_name"]]
test_grades = [int(float(g)) for g in test_df["grade"]]
test_mn = domain_masker.apply_mask_batch(test_mn_raw, classes, test_topics, test_grades, damping=0.85)

# 3. Test Text
test_sgd = np.load(art / "test_text.npy")
test_sem = np.load(art / "test_semantic_text.npy")

# 4. Fused Test Probabilities
test_fused = (w[0] * test_prior + w[1] * test_cn + w[2] * test_mn +
              w[3] * test_swin + w[4] * test_sgd + w[5] * test_sem)

top3_idx = np.argsort(-test_fused, axis=1, kind="mergesort")[:, :3]
pred_strings = [" ".join([classes[idx] for idx in row]) for row in top3_idx]

sub_df = pd.DataFrame({
    "trace_id": test_df["trace_id"].astype(str),
    "root_code": pred_strings
})

out_path = PATHS.submissions_dir / "multimodal_m15_balanced_fusion_submission.csv"
sub_df.to_csv(out_path, index=False)
print(f"[M15] Saved submission to: {out_path}")

# Validate
val = validate_submission_file(out_path, PATHS.taxonomy_csv)
print(f"[M15] Validation status: {val}")

# Agreement vs M14 (0.53683 LB)
m14_sub = cast(pd.DataFrame, pd.read_csv(PATHS.submissions_dir / "multimodal_hexamodal_rebalanced_submission.csv"))
diff_m14 = int((sub_df["root_code"] != m14_sub["root_code"]).sum())
pct_m14 = diff_m14 / len(sub_df) * 100

print(f"\n[Comparison vs M14 (LB 0.53683)]")
print(f"  Rows identical to M14: {len(sub_df) - diff_m14} ({100 - pct_m14:.1f}%)")
print(f"  Rows selectively updated: {diff_m14} ({pct_m14:.1f}%)")

# Save test probabilities artifact
np.save(art / "test_m15_balanced_fusion.npy", test_fused)
np.save(art / "oof_m15_balanced_fusion.npy", m15_oof)
print(f"[M15] Artifacts saved.")

