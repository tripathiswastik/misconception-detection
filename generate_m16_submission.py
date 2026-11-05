"""Generate M16 Calibrated Power-Sharpened Semantic-Prior Fusion (OOF 0.54444)."""
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
from src.data.splits import FoldSplitter
from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.diagnoser.domain_mask import DomainConstraintEngine
from src.fusion.validator import validate_submission_file
from src.metrics.map3 import calculate_map3

def generate_m16_submission():
    print("=" * 70)
    print("GENERATING MODEL M16: CALIBRATED POWER-SHARPENED FUSION")
    print("=" * 70)

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
    m15_oof = np.load(art / "oof_m15_balanced_fusion.npy")

    # M16 Optimal Weights
    # [Prior: 0.4900, ConvNeXt: 0.01224, MobileNet: 0.044064, Swin-T: 0.004896, SGD: 0.260304, Semantic: 0.188496]
    w = np.array([0.4900, 0.01224, 0.044064, 0.004896, 0.260304, 0.188496], dtype=np.float64)
    w = w / w.sum()

    # Temperature Sharpening: T_sgd = 0.7, T_sem = 1.1
    t_sgd = 0.70
    t_sem = 1.10

    sgd_oof_sharp = text_sgd_oof ** (1.0 / t_sgd)
    sgd_oof_sharp = np.divide(sgd_oof_sharp, sgd_oof_sharp.sum(axis=1, keepdims=True))

    sem_oof_sharp = text_sem_oof ** (1.0 / t_sem)
    sem_oof_sharp = np.divide(sem_oof_sharp, sem_oof_sharp.sum(axis=1, keepdims=True))

    m16_oof = (w[0] * prior_oof + w[1] * vis_cn_oof + w[2] * vis_mn_oof +
               w[3] * swin_oof + w[4] * sgd_oof_sharp + w[5] * sem_oof_sharp)

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

    m16_score = eval_map3(m16_oof)
    m15_score = eval_map3(m15_oof)
    m12_score = eval_map3(m12_oof)

    m16_sample = eval_per_sample(m16_oof)
    m15_sample = eval_per_sample(m15_oof)
    m12_sample = eval_per_sample(m12_oof)

    # 10,000 Paired Bootstrap tests
    rng = np.random.default_rng(42)
    n_boot = 10000
    n_samples = len(y_true)

    # vs M12
    boot_deltas_12 = np.zeros(n_boot)
    deltas_12 = m16_sample - m12_sample
    for b in range(n_boot):
        idx = rng.integers(0, n_samples, size=n_samples)
        boot_deltas_12[b] = np.mean(deltas_12[idx])

    # vs M15
    boot_deltas_15 = np.zeros(n_boot)
    deltas_15 = m16_sample - m15_sample
    for b in range(n_boot):
        idx = rng.integers(0, n_samples, size=n_samples)
        boot_deltas_15[b] = np.mean(deltas_15[idx])

    print(f"M16 Honest OOF MAP@3: {m16_score:.5f}")
    print(f"M15 Baseline OOF:     {m15_score:.5f} (Delta: {m16_score - m15_score:+.5f}, p={np.mean(boot_deltas_15 <= 0):.4f})")
    print(f"M12 Baseline OOF:     {m12_score:.5f} (Delta: {m16_score - m12_score:+.5f}, p={np.mean(boot_deltas_12 <= 0):.4f})")
    print(f"95% CI vs M12:        [{np.percentile(boot_deltas_12, 2.5):+.5f}, {np.percentile(boot_deltas_12, 97.5):+.5f}]")

    # ─────────────────────────────────────────────────────────────────
    # Generate Test Predictions
    # ─────────────────────────────────────────────────────────────────
    print("\n[M16] Generating test set predictions...")
    splitter = FoldSplitter(n_splits=5, random_state=42)
    folds = splitter.load_splits(PATHS.splits_cache)

    train_embeds = np.load(art / "train_text_embeddings_384.npy")
    test_embeds = np.load(art / "test_text_embeddings_384.npy")

    # 1. Test Prior (5-fold averaged)
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

    # 2. Test Vision (ConvNeXt, MobileNet with domain mask, Swin-T)
    test_cn = np.load(art / "test_convnext_vision.npy")
    test_mn_raw = np.load(art / "test_finetuned_vision.npy")
    test_swin = np.load(art / "test_swin_vision.npy")

    domain_masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df)
    test_topics = [str(t) for t in test_df["topic_name"]]
    test_grades = [int(float(g)) for g in test_df["grade"]]
    test_mn = domain_masker.apply_mask_batch(test_mn_raw, classes, test_topics, test_grades, damping=0.85)

    # 3. Test Text with Temperature Sharpening
    test_sgd = np.load(art / "test_text.npy")
    test_sem = np.load(art / "test_semantic_text.npy")

    test_sgd_sharp = test_sgd ** (1.0 / t_sgd)
    test_sgd_sharp = np.divide(test_sgd_sharp, test_sgd_sharp.sum(axis=1, keepdims=True))

    test_sem_sharp = test_sem ** (1.0 / t_sem)
    test_sem_sharp = np.divide(test_sem_sharp, test_sem_sharp.sum(axis=1, keepdims=True))

    # 4. Final Fused Test Predictions
    test_fused = (w[0] * test_prior + w[1] * test_cn + w[2] * test_mn +
                  w[3] * test_swin + w[4] * test_sgd_sharp + w[5] * test_sem_sharp)

    top3_idx = np.argsort(-test_fused, axis=1, kind="mergesort")[:, :3]
    pred_strings = [" ".join([classes[idx] for idx in row]) for row in top3_idx]

    sub_df = pd.DataFrame({
        "trace_id": test_df["trace_id"].astype(str),
        "root_code": pred_strings
    })

    out_path = PATHS.submissions_dir / "multimodal_m16_power_calibrated_submission.csv"
    sub_df.to_csv(out_path, index=False)
    print(f"[M16] Saved submission to: {out_path}")

    # Validate
    val = validate_submission_file(out_path, PATHS.taxonomy_csv)
    print(f"[M16] Validation status: {val}")

    # Agreement vs M15 (0.54149 LB)
    m15_sub = cast(pd.DataFrame, pd.read_csv(PATHS.submissions_dir / "multimodal_m15_balanced_fusion_submission.csv"))
    diff_m15 = int((sub_df["root_code"] != m15_sub["root_code"]).sum())
    pct_m15 = diff_m15 / len(sub_df) * 100

    print(f"\n[Comparison vs M15 (LB 0.54149)]")
    print(f"  Rows identical to M15: {len(sub_df) - diff_m15} ({100 - pct_m15:.1f}%)")
    print(f"  Rows selectively updated: {diff_m15} ({pct_m15:.1f}%)")

    # Save artifacts
    np.save(art / "test_m16_power_calibrated.npy", test_fused)
    np.save(art / "oof_m16_power_calibrated.npy", m16_oof)
    print(f"[M16] Artifacts saved.")

    return out_path


if __name__ == "__main__":
    generate_m16_submission()

