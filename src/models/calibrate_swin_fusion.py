"""Temperature-Calibrated Swin-T Integration & Tri-Vision Multimodal Fusion.

Path 1 Implementation:
1. Evaluates Swin-T linear probe across regularizers C in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0].
2. Applies Temperature Scaling T in [1.0, 1.2, 1.5, 1.8, 2.0, 2.5] on unnormalized logits:
   P_T(c) = softmax(z(c) / T)
3. Applies in-fold domain constraint masking (damping=0.85).
4. Tests Tri-Vision combinations (ConvNeXt + Swin-T + MobileNetV3).
5. Tests full multimodal fusion:
   w_prior * P_prior + w_vis * P_trivision + w_text * P_text
6. Performs 10,000-sample paired bootstrap audit vs Champion baseline (0.52044).
7. Conditionally generates submissions/multimodal_swin_calibrated_submission.csv ONLY if
   statistically significant (p < 0.05, 95% CI > 0).
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, PATHS
from src.data.dataset import MathMisconceptionData
from src.data.splits import FoldSplitter
from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.diagnoser.domain_mask import DomainConstraintEngine
from src.fusion.validator import validate_submission_file
from src.metrics.map3 import calculate_map3


def eval_map3(probs: np.ndarray, y_true: List[str], classes: List[str]) -> float:
    top3_idx = np.argsort(-probs, axis=1, kind="mergesort")[:, :3]
    preds = [[classes[idx] for idx in row] for row in top3_idx]
    return calculate_map3(y_true, preds)


def eval_per_sample(probs: np.ndarray, y_true: List[str], classes: List[str]) -> np.ndarray:
    top3_idx = np.argsort(-probs, axis=1, kind="mergesort")[:, :3]
    scores = np.zeros(len(y_true), dtype=np.float64)
    for i, target in enumerate(y_true):
        preds = [classes[idx] for idx in top3_idx[i]]
        for rank, p in enumerate(preds):
            if p == target:
                scores[i] = 1.0 / (rank + 1)
                break
    return scores


def calc_mean_entropy(probs: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    entropy = -np.sum(p * np.log(p), axis=1)
    return float(np.mean(entropy))


def run_calibrated_swin_fusion():
    print("\n" + "=" * 75)
    print("PATH 1: TEMPERATURE-CALIBRATED SWIN-T INTEGRATION & MULTIMODAL AUDIT")
    print("=" * 75 + "\n")

    tax = TaxonomyManager(PATHS.taxonomy_csv)
    classes = tax.all_codes(include_none=True)
    n_classes = len(classes)

    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    test_df = data_mgr.load_test()
    y_true = train_df["root_code"].astype(str).tolist()
    n_train = len(train_df)
    n_test = len(test_df)

    splitter = FoldSplitter(n_splits=CONFIG.n_splits, random_state=CONFIG.random_state)
    folds = splitter.load_splits(PATHS.splits_cache)

    artifacts_dir = PROJECT_ROOT / "artifacts"

    # 1. Load Champion Baseline OOF
    print("[1/6] Loading Champion Baseline & Multimodal Components...")
    oof_baseline = np.load(artifacts_dir / "oof_adaptive_fusion.npy")
    baseline_map3 = eval_map3(oof_baseline, y_true, classes)
    baseline_sample_scores = eval_per_sample(oof_baseline, y_true, classes)
    print(f"  Champion Baseline OOF MAP@3: {baseline_map3:.5f} (Public LB 0.53383 [CHAMPION])")

    # 2. Load / Compute Combined Adaptive Prior
    if (artifacts_dir / "best_adaptive_prior_oof.npy").exists():
        oof_adapt_prior = np.load(artifacts_dir / "best_adaptive_prior_oof.npy")
        print(f"  Loaded Adaptive Prior OOF (MAP@3: {eval_map3(oof_adapt_prior, y_true, classes):.5f})")
    else:
        print("  Generating Adaptive Prior OOF...")
        train_embeds = np.load(artifacts_dir / "train_text_embeddings_384.npy")
        oof_adapt_prior = np.zeros((n_train, n_classes), dtype=np.float32)
        for fold_idx, (tr_idx, val_idx) in enumerate(folds, start=1):
            fold_tr = train_df.iloc[tr_idx].reset_index(drop=True)
            fold_val = train_df.iloc[val_idx].reset_index(drop=True)
            engine = AdaptiveQuestionPriorEngine(
                classes=classes,
                m_q_base=10.0,
                m_tg=10.0,
                adaptive_nq=True,
                adaptive_power=0.70,
                use_semantic_backoff=True,
                semantic_embeddings=train_embeds,
                semantic_top_k=5,
                semantic_threshold=0.70,
                semantic_weight=0.50
            )
            engine.fit(fold_tr, fold_train_indices=tr_idx)
            val_emb = train_embeds[val_idx]
            oof_adapt_prior[val_idx] = engine.predict_proba(fold_val, eval_indices=val_idx, eval_embeddings=val_emb)

    # 3. Load Vision & Text baselines
    oof_cn = np.load(artifacts_dir / "oof_convnext_vision.npy")
    oof_mn = np.load(artifacts_dir / "oof_finetuned_vision.npy")
    oof_dual_vis = 0.5 * oof_cn + 0.5 * oof_mn

    oof_sgd_text = np.load(artifacts_dir / "honest_oof_text.npy")
    oof_sem_text = np.load(artifacts_dir / "oof_semantic_text.npy")
    oof_enh_text = 0.4 * oof_sgd_text + 0.6 * oof_sem_text

    print(f"  ConvNeXt Entropy: {calc_mean_entropy(oof_cn):.3f} | Dual-Vision Entropy: {calc_mean_entropy(oof_dual_vis):.3f}")

    # 4. Load Swin-T features and run Linear Probe with logit extraction
    print("\n[2/6] Extracting Swin-T Fold Logits across C & Temperature Grid...")
    X_train_raw = np.load(artifacts_dir / "train_swin_features.npy")
    X_test_raw = np.load(artifacts_dir / "test_swin_features.npy")
    X_train = np.asarray(normalize(X_train_raw, norm="l2"), dtype=np.float32)
    X_test = np.asarray(normalize(X_test_raw, norm="l2"), dtype=np.float32)

    C_grid = [0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
    T_grid = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]

    # Pre-fit models to collect raw logits for each fold or load from cache
    cache_path = artifacts_dir / "swin_logits_cache.npz"
    probe_cache: Dict[float, Dict[str, Any]] = {}
    
    if cache_path.exists():
        print("  Found cached Swin-T probe logits! Loading from disk...")
        cached_data = np.load(cache_path)
        for c in C_grid:
            probe_cache[c] = {
                "oof_logits": cached_data[f"oof_c_{c}"],
                "test_logits": cached_data[f"test_c_{c}"]
            }
    else:
        save_dict = {}
        for c in C_grid:
            oof_logits = np.zeros((n_train, n_classes), dtype=np.float32)
            test_logits = np.zeros((n_test, n_classes), dtype=np.float32)

            for tr_idx, va_idx in folds:
                clf = LogisticRegression(C=c, max_iter=250, solver="lbfgs", random_state=42)
                clf.fit(X_train[tr_idx], [y_true[i] for i in tr_idx])
                val_df = clf.decision_function(X_train[va_idx])
                te_df = clf.decision_function(X_test)

                for m_i, c_name in enumerate(clf.classes_):
                    if c_name in classes:
                        c_idx = classes.index(c_name)
                        oof_logits[va_idx, c_idx] = val_df[:, m_i]
                        test_logits[:, c_idx] += te_df[:, m_i] / len(folds)

            probe_cache[c] = {"oof_logits": oof_logits, "test_logits": test_logits}
            save_dict[f"oof_c_{c}"] = oof_logits
            save_dict[f"test_c_{c}"] = test_logits
            print(f"    Fitted Swin-T probe C={c:.2f}")

        np.savez(cache_path, **save_dict)
        print(f"  Cached Swin-T probe logits to {cache_path}")

    # Pre-compute in-fold domain affinity mask matrix for instantaneous evaluation
    print("  Precomputing in-fold domain affinity mask...")
    in_fold_affinity_mask = np.ones((n_train, n_classes), dtype=np.float32)
    damping = 0.85
    for tr_idx, va_idx in folds:
        masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df.iloc[tr_idx])
        t_va = [str(t) for t in train_df.iloc[va_idx]["topic_name"]]
        g_va = [int(float(g)) for g in train_df.iloc[va_idx]["grade"]]
        for local_i, idx in enumerate(va_idx):
            topic = t_va[local_i]
            grade = g_va[local_i]
            for c_idx in range(n_classes):
                in_fold_affinity_mask[idx, c_idx] = masker.compute_affinity(classes[c_idx], topic, grade) ** damping

    # 5. Grid Search over (C, T, Tri-Vision weights)
    print("\n[3/6] Evaluating Calibrated Swin-T & Tri-Vision Multimodal Blend...")
    grid_results = []

    tri_weight_options = [
        (0.40, 0.40, 0.20, "40CN/40SW/20MN"),
        (0.45, 0.35, 0.20, "45CN/35SW/20MN"),
        (0.50, 0.30, 0.20, "50CN/30SW/20MN"),
        (0.35, 0.45, 0.20, "35CN/45SW/20MN"),
        (0.34, 0.33, 0.33, "34CN/33SW/33MN"),
    ]

    for c in C_grid:
        raw_oof_logits = probe_cache[c]["oof_logits"]

        for T in T_grid:
            # Calibrate logits with temperature
            scaled_logits = raw_oof_logits / T
            prob_swin = softmax(scaled_logits, axis=1)

            # Instantaneous vectorized in-fold domain masking
            masked = prob_swin * in_fold_affinity_mask
            sums = masked.sum(axis=1, keepdims=True)
            oof_swin_cal = np.divide(masked, sums, where=sums > 1e-12)

            swin_vis_map3 = eval_map3(oof_swin_cal, y_true, classes)
            swin_entropy = calc_mean_entropy(oof_swin_cal)

            for w_cn, w_sw, w_mn, tri_lbl in tri_weight_options:
                oof_trivis = w_cn * oof_cn + w_sw * oof_swin_cal + w_mn * oof_mn
                trivis_map3 = eval_map3(oof_trivis, y_true, classes)

                # Multimodal blend (champion weights: 70% prior, 14% vis, 16% text)
                fused_oof = 0.70 * oof_adapt_prior + 0.14 * oof_trivis + 0.16 * oof_enh_text
                fused_map3 = eval_map3(fused_oof, y_true, classes)
                delta_base = fused_map3 - baseline_map3

                grid_results.append({
                    "C": c,
                    "T": T,
                    "tri_label": tri_lbl,
                    "weights": (w_cn, w_sw, w_mn),
                    "swin_vis_map3": swin_vis_map3,
                    "swin_entropy": swin_entropy,
                    "trivis_map3": trivis_map3,
                    "fused_map3": fused_map3,
                    "delta_vs_baseline": delta_base,
                    "oof_swin_cal": oof_swin_cal,
                    "oof_trivis": oof_trivis,
                    "fused_oof": fused_oof
                })

    # Sort results by fused multimodal MAP@3
    grid_results.sort(key=lambda x: x["fused_map3"], reverse=True)

    print("\n[Top 10 Calibrated Swin Configurations in Multimodal Blend]")
    print(f"{'Rank':<5} {'C':<6} {'T':<5} {'TriWeights':<16} {'SwinVis':<8} {'Entropy':<8} {'TriVis':<8} {'Fused MAP@3':<12} {'Delta vs Champion':<16}")
    print("-" * 90)
    for r, res in enumerate(grid_results[:10], start=1):
        print(f"{r:<5} {res['C']:<6.2f} {res['T']:<5.2f} {res['tri_label']:<16} "
              f"{res['swin_vis_map3']:<8.4f} {res['swin_entropy']:<8.3f} {res['trivis_map3']:<8.4f} "
              f"{res['fused_map3']:<12.5f} {res['delta_vs_baseline']:+12.5f}")

    best = grid_results[0]
    print("\n" + "=" * 75)
    print("BEST CONFIGURATION IDENTIFIED:")
    print(f"  C:                     {best['C']}")
    print(f"  Temperature T:         {best['T']}")
    print(f"  Tri-Vision Topology:   {best['tri_label']}")
    print(f"  Standalone Swin Vis:   {best['swin_vis_map3']:.5f}")
    print(f"  Swin Entropy:          {best['swin_entropy']:.3f} (cf ConvNeXt {calc_mean_entropy(oof_cn):.3f})")
    print(f"  Tri-Vision OOF MAP@3:  {best['trivis_map3']:.5f} (cf Dual-Vision {eval_map3(oof_dual_vis, y_true, classes):.5f})")
    print(f"  Multimodal OOF MAP@3:  {best['fused_map3']:.5f}")
    print(f"  Delta over Champion:   {best['delta_vs_baseline']:+.5f}")
    print("=" * 75)

    # 6. Fine weight search around best configuration
    print("\n[4/6] Fine Multimodal Weight Exploration around Best Calibrated Tri-Vision...")
    best_trivis = best["oof_trivis"]
    fine_weights = []
    for wp in np.arange(0.66, 0.75, 0.01):
        for wv in np.arange(0.10, 0.20, 0.01):
            wt = 1.0 - (wp + wv)
            if 0.08 <= wt <= 0.22:
                blend = wp * oof_adapt_prior + wv * best_trivis + wt * oof_enh_text
                score = eval_map3(blend, y_true, classes)
                fine_weights.append((score, round(wp, 3), round(wv, 3), round(wt, 3), blend))

    fine_weights.sort(key=lambda x: x[0], reverse=True)
    top_score, opt_wp, opt_wv, opt_wt, opt_blend = fine_weights[0]
    print(f"  Fine Tuned Multimodal: Prior={opt_wp:.3f}, Vis={opt_wv:.3f}, Text={opt_wt:.3f} -> MAP@3: {top_score:.5f} (Delta: {top_score - baseline_map3:+.5f})")

    champ_oof = opt_blend
    champ_score = top_score
    delta_over_base = champ_score - baseline_map3

    # 7. 10,000 Paired Bootstrap Audit
    print("\n[5/6] Running 10,000-Sample Paired Bootstrap Audit vs Champion Baseline...")
    cand_sample_scores = eval_per_sample(champ_oof, y_true, classes)
    deltas = cand_sample_scores - baseline_sample_scores
    n_samples = len(deltas)

    rng = np.random.default_rng(CONFIG.random_state)
    boot_deltas = np.zeros(10000, dtype=np.float64)
    for b in range(10000):
        idx = rng.integers(0, n_samples, size=n_samples)
        boot_deltas[b] = np.mean(deltas[idx])

    mean_delta = float(np.mean(boot_deltas))
    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))
    se_delta = float(np.std(boot_deltas))
    p_value = float(np.mean(boot_deltas <= 0))
    is_significant = (ci_lower > 0) and (p_value < 0.05)

    print(f"  Sample count:                 {n_samples}")
    print(f"  Baseline OOF MAP@3:           {baseline_map3:.5f}")
    print(f"  Candidate OOF MAP@3:          {champ_score:.5f}")
    print(f"  Mean Bootstrap Delta:        {mean_delta:+.5f}")
    print(f"  Standard Error:               {se_delta:.5f}")
    print(f"  95% Confidence Interval:      [{ci_lower:+.5f}, {ci_upper:+.5f}]")
    print(f"  One-Tailed P-Value (<= 0):    {p_value:.5f}")
    print(f"  Statistically Significant:    {is_significant}")

    # Manifest serialization
    manifest = {
        "baseline_map3": baseline_map3,
        "candidate_map3": champ_score,
        "delta": mean_delta,
        "ci_95": [ci_lower, ci_upper],
        "p_value": p_value,
        "is_significant": is_significant,
        "swin_calibration": {
            "C": best["C"],
            "T": best["T"],
            "entropy": best["swin_entropy"],
            "tri_topology": best["tri_label"],
            "tri_weights": list(best["weights"])
        },
        "multimodal_weights": {
            "prior": opt_wp,
            "vision": opt_wv,
            "text": opt_wt
        }
    }

    manifest_path = artifacts_dir / "calibrated_swin_fusion_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[Manifest] Saved audit report to: {manifest_path}")

    # 8. Gated Submission Generation
    print("\n[6/6] Verifying SDLC Deployment Gate...")
    if is_significant:
        print("  >>> GATE PASSED: Statistically significant improvement confirmed! Generating submission...")

        # Compute test predictions for best calibrated Swin
        best_c = best["C"]
        best_t = best["T"]
        w_cn, w_sw, w_mn = best["weights"]

        test_swin_raw_logits = probe_cache[best_c]["test_logits"]
        test_swin_scaled = test_swin_raw_logits / best_t
        test_swin_prob = softmax(test_swin_scaled, axis=1)

        masker_global = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df)
        t_te = [str(t) for t in test_df["topic_name"]]
        g_te = [int(float(g)) for g in test_df["grade"]]
        test_swin_cal = masker_global.apply_mask_batch(test_swin_prob, classes, t_te, g_te, damping=0.85)

        test_cn = np.load(artifacts_dir / "test_convnext_vision.npy")
        test_mn = np.load(artifacts_dir / "test_finetuned_vision.npy")
        test_mn_masked = masker_global.apply_mask_batch(test_mn, classes, t_te, g_te, damping=0.85)

        test_trivis = w_cn * test_cn + w_sw * test_swin_cal + w_mn * test_mn_masked

        # Test Adaptive Prior
        train_embeds = np.load(artifacts_dir / "train_text_embeddings_384.npy")
        test_embeds = np.load(artifacts_dir / "test_text_embeddings_384.npy")
        test_adapt_prior = np.zeros((n_test, n_classes), dtype=np.float32)
        for tr_idx, val_idx in folds:
            fold_tr = train_df.iloc[tr_idx].reset_index(drop=True)
            engine = AdaptiveQuestionPriorEngine(
                classes=classes,
                m_q_base=10.0,
                m_tg=10.0,
                adaptive_nq=True,
                adaptive_power=0.70,
                use_semantic_backoff=True,
                semantic_embeddings=train_embeds,
                semantic_top_k=5,
                semantic_threshold=0.70,
                semantic_weight=0.50
            )
            engine.fit(fold_tr, fold_train_indices=tr_idx)
            test_adapt_prior += engine.predict_proba(test_df, eval_embeddings=test_embeds) / len(folds)

        test_sgd_text = np.load(artifacts_dir / "test_text.npy")
        test_sem_text = np.load(artifacts_dir / "test_semantic_text.npy")
        test_enh_text = 0.4 * test_sgd_text + 0.6 * test_sem_text

        test_fused = opt_wp * test_adapt_prior + opt_wv * test_trivis + opt_wt * test_enh_text

        top_idx = np.argsort(-test_fused, axis=1, kind="mergesort")[:, :3]
        pred_strings = [" ".join([classes[idx] for idx in row]) for row in top_idx]

        sub_df = pd.DataFrame({
            "trace_id": test_df["trace_id"],
            "root_code": pred_strings
        })
        out_csv = PATHS.submissions_dir / "multimodal_swin_calibrated_submission.csv"
        sub_df.to_csv(out_csv, index=False)
        print(f"  Candidate submission written to: {out_csv}")

        val_res = validate_submission_file(out_csv, PATHS.taxonomy_csv)
        print(f"  Validation status: {val_res}")

        # Agreement against Champion
        champ_df = cast(pd.DataFrame, pd.read_csv(PATHS.submissions_dir / "multimodal_adaptive_fusion_submission.csv"))
        diff_count = int((sub_df["root_code"] != champ_df["root_code"]).sum())
        diff_pct = (diff_count / len(sub_df)) * 100
        print(f"  Agreement vs Champion SOTA (0.53383):")
        print(f"    Identical rows: {len(sub_df) - diff_count} ({(100 - diff_pct):.2f}%)")
        print(f"    Diversified rows: {diff_count} ({diff_pct:.2f}%)")
    else:
        print("  >>> GATE DECISION: No statistically significant improvement over Champion (0.53383).")
        print("  >>> In accordance with AGENTS.md SDLC safety rules, champion remains UNTOUCHED.")

    return manifest


if __name__ == "__main__":
    run_calibrated_swin_fusion()
