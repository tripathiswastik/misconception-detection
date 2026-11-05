"""Question-Aware Fold-Safe NONE & Misconception Calibration Engine.

Systematically resolves the 660 non-NONE samples mistakenly ranked with NONE at #1:
1. In-Fold Question Difficulty Ratio:
   R(NONE | q) = [ P_smooth(NONE | q) / P_global(NONE) ] ^ gamma
   Modulates P(NONE) based on question-level empirical error propensity.
2. In-Fold Inverse-Frequency Soft Scaling:
   S(c) = P(c) * (P_global(c) ^ -beta) for c != NONE
3. Strictly Leak-Free:
   All statistics fitted exclusively on fold_train rows (data/splits.json).
4. 10,000 Paired Bootstrap Significance Verification.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, PATHS
from src.data.dataset import MathMisconceptionData
from src.data.splits import FoldSplitter
from src.data.taxonomy import LABEL_NONE, TaxonomyManager
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


def compute_diagnostics(probs: np.ndarray, y_true: List[str], classes: List[str]) -> Dict[str, Any]:
    none_idx = classes.index(LABEL_NONE)
    top1_idx = np.argmax(probs, axis=1)

    actual_none_mask = np.array([y == LABEL_NONE for y in y_true], dtype=bool)
    pred_none_mask = (top1_idx == none_idx)

    actual_none_count = int(actual_none_mask.sum())
    pred_none_count = int(pred_none_mask.sum())
    correct_none_count = int((actual_none_mask & pred_none_mask).sum())
    false_none_count = int((~actual_none_mask & pred_none_mask).sum())

    return {
        "actual_none": actual_none_count,
        "pred_none_top1": pred_none_count,
        "correct_none_top1": correct_none_count,
        "false_none_top1": false_none_count,
        "none_precision": correct_none_count / max(1, pred_none_count),
        "none_recall": correct_none_count / max(1, actual_none_count)
    }


class QuestionAwareCalibrator:
    """Strictly in-fold Question-Aware Probability Calibrator."""

    def __init__(
        self,
        classes: List[str],
        gamma: float = 0.30,
        m_q: float = 5.0,
        m_tg: float = 5.0,
        beta: float = 0.0
    ):
        self.classes = list(classes)
        self.none_idx = self.classes.index(LABEL_NONE)
        self.n_classes = len(self.classes)
        self.gamma = gamma
        self.m_q = m_q
        self.m_tg = m_tg
        self.beta = beta

        # Fitted in-fold stats
        self.p_global_none: float = 0.337
        self.p_global_class: np.ndarray = np.ones(self.n_classes, dtype=np.float32) / self.n_classes
        self.tg_none_rates: Dict[Tuple[str, int], float] = {}
        self.q_none_rates: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, fold_train_df: pd.DataFrame, target_col: str = "root_code") -> "QuestionAwareCalibrator":
        n_samples = len(fold_train_df)
        y_train = fold_train_df[target_col].astype(str).values

        # 1. Global rates
        none_count = float(np.sum(y_train == LABEL_NONE))
        self.p_global_none = (none_count + 1.0) / (n_samples + 2.0)

        class_counts = np.zeros(self.n_classes, dtype=np.float32)
        for y in y_train:
            idx = self.classes.index(y) if y in self.classes else -1
            if idx >= 0:
                class_counts[idx] += 1.0
        self.p_global_class = (class_counts + 1.0) / (n_samples + self.n_classes)

        # 2. Topic + Grade NONE rates
        self.tg_none_rates.clear()
        tg_grouped = fold_train_df.groupby(["topic_name", "grade"])
        for tg_key_raw, group in tg_grouped:
            tg_tuple = cast(Tuple[Any, Any], tg_key_raw)
            topic, grade = tg_tuple[0], tg_tuple[1]
            try:
                g_int = int(float(grade))
            except (ValueError, TypeError):
                g_int = -1
            key = (str(topic), g_int)
            n_tg = len(group)
            none_tg = float(np.sum(group[target_col].astype(str) == LABEL_NONE))
            smoothed_tg = (none_tg + self.m_tg * self.p_global_none) / (n_tg + self.m_tg)
            self.tg_none_rates[key] = smoothed_tg

        # 3. Question-level NONE rates
        self.q_none_rates.clear()
        q_grouped = fold_train_df.groupby("question_id")
        for q_id, group in q_grouped:
            q_id_str = str(q_id)
            topic = str(group["topic_name"].iloc[0])
            try:
                g_int = int(float(group["grade"].iloc[0]))
            except (ValueError, TypeError):
                g_int = -1
            tg_prior = self.tg_none_rates.get((topic, g_int), self.p_global_none)

            n_q = len(group)
            none_q = float(np.sum(group[target_col].astype(str) == LABEL_NONE))
            smoothed_q = (none_q + self.m_q * tg_prior) / (n_q + self.m_q)
            self.q_none_rates[q_id_str] = smoothed_q

        self.is_fitted = True
        return self

    def calibrate(self, df: pd.DataFrame, raw_probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("QuestionAwareCalibrator must be fitted before calibrate.")

        n_samples = len(df)
        calibrated = np.copy(raw_probs)

        # Optional C3: Soft inverse-frequency scaling on non-NONE classes
        if self.beta > 0.0:
            scale_factors = (self.p_global_class ** (-self.beta))
            scale_factors[self.none_idx] = 1.0
            calibrated = calibrated * scale_factors

        # C1: Question-specific difficulty ratio
        for i, (q_id, topic, grade) in enumerate(zip(df["question_id"], df["topic_name"], df["grade"])):
            q_id_str = str(q_id)
            try:
                g_int = int(float(grade))
            except (ValueError, TypeError):
                g_int = -1
            key = (str(topic), g_int)

            if q_id_str in self.q_none_rates:
                q_rate = self.q_none_rates[q_id_str]
            elif key in self.tg_none_rates:
                q_rate = self.tg_none_rates[key]
            else:
                q_rate = self.p_global_none

            # Ratio of question-level NONE rate to global NONE rate
            ratio = float(q_rate / max(1e-4, self.p_global_none))
            mod_factor = float(ratio ** self.gamma)

            calibrated[i, self.none_idx] *= mod_factor

        # Row normalize
        r_sums = calibrated.sum(axis=1, keepdims=True)
        return np.divide(calibrated, r_sums, where=r_sums > 0)


def run_calibration_study():
    print("\n" + "=" * 75)
    print("QUESTION-AWARE FOLD-SAFE NONE & MISCONCEPTION CALIBRATION STUDY")
    print("=" * 75 + "\n")

    tax = TaxonomyManager(PATHS.taxonomy_csv)
    classes = tax.all_codes(include_none=True)

    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    test_df = data_mgr.load_test()
    n_train = len(train_df)
    n_test = len(test_df)
    y_true = train_df["root_code"].astype(str).tolist()

    splitter = FoldSplitter(n_splits=CONFIG.n_splits, random_state=CONFIG.random_state)
    folds = splitter.load_splits(PATHS.splits_cache)

    artifacts_dir = PROJECT_ROOT / "artifacts"
    raw_oof = np.load(artifacts_dir / "oof_adaptive_fusion.npy")
    test_raw = pd.read_csv(PATHS.submissions_dir / "multimodal_adaptive_fusion_submission.csv")

    base_map3 = eval_map3(raw_oof, y_true, classes)
    base_diag = compute_diagnostics(raw_oof, y_true, classes)
    base_scores = eval_per_sample(raw_oof, y_true, classes)

    print(f"[Baseline Champion Model] OOF MAP@3: {base_map3:.5f}")
    print(f"  Diagnostics: Actual NONE={base_diag['actual_none']}, Top1 NONE={base_diag['pred_none_top1']}, Correct NONE={base_diag['correct_none_top1']}, False NONE={base_diag['false_none_top1']}")

    # Parameter Grid Search
    gamma_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70]
    mq_values = [2.0, 5.0, 10.0]
    beta_values = [0.0, 0.05, 0.10]

    sweep_results = []
    print(f"\n[Grid Search] Sweeping {len(gamma_values) * len(mq_values) * len(beta_values)} parameter combinations across 5 folds...")

    for mq in mq_values:
        for beta in beta_values:
            for gamma in gamma_values:
                calibrated_oof = np.zeros_like(raw_oof)

                for tr_idx, val_idx in folds:
                    fold_tr = train_df.iloc[tr_idx].reset_index(drop=True)
                    fold_val = train_df.iloc[val_idx].reset_index(drop=True)

                    calibrator = QuestionAwareCalibrator(
                        classes=classes, gamma=gamma, m_q=mq, m_tg=5.0, beta=beta
                    )
                    calibrator.fit(fold_tr)

                    val_raw = raw_oof[val_idx]
                    val_cal = calibrator.calibrate(fold_val, val_raw)
                    calibrated_oof[val_idx] = val_cal

                score = eval_map3(calibrated_oof, y_true, classes)
                diag = compute_diagnostics(calibrated_oof, y_true, classes)
                delta = score - base_map3

                sweep_results.append({
                    "gamma": gamma,
                    "m_q": mq,
                    "beta": beta,
                    "map3": score,
                    "delta": delta,
                    "false_none": diag["false_none_top1"],
                    "correct_none": diag["correct_none_top1"],
                    "pred_none_top1": diag["pred_none_top1"]
                })

    sweep_results.sort(key=lambda x: x["map3"], reverse=True)

    print("\nTop 10 Calibration Configurations:")
    print(f"{'Rank':<5} | {'MAP@3':<9} | {'Delta':<9} | {'gamma':<6} | {'m_q':<5} | {'beta':<5} | {'False NONE':<11} | {'Correct NONE'}")
    print("-" * 80)
    for rank, r in enumerate(sweep_results[:10], start=1):
        print(f"{rank:<5} | {r['map3']:<9.5f} | {r['delta']:<+9.5f} | {r['gamma']:<6.2f} | {r['m_q']:<5.1f} | {r['beta']:<5.2f} | {r['false_none']:<11} | {r['correct_none']}")

    champ = sweep_results[0]
    print("\n" + "=" * 75)
    print(f"CHAMPION CALIBRATION CONFIGURATION:")
    print(f"  OOF MAP@3:       {champ['map3']:.5f} (Delta vs Baseline: {champ['delta']:+.5f})")
    print(f"  Parameters:      gamma={champ['gamma']}, m_q={champ['m_q']}, beta={champ['beta']}")
    print(f"  False NONE Top1: {champ['false_none']} (Reduced by {base_diag['false_none_top1'] - champ['false_none']})")
    print(f"  Correct NONE:    {champ['correct_none']} (Preserved {champ['correct_none'] / base_diag['correct_none_top1'] * 100:.1f}%)")
    print("=" * 75)

    # Reconstruct optimal calibrated OOF for bootstrap testing
    best_calibrated_oof = np.zeros_like(raw_oof)
    for tr_idx, val_idx in folds:
        fold_tr = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_val = train_df.iloc[val_idx].reset_index(drop=True)
        calibrator = QuestionAwareCalibrator(
            classes=classes, gamma=champ["gamma"], m_q=champ["m_q"], m_tg=5.0, beta=champ["beta"]
        )
        calibrator.fit(fold_tr)
        best_calibrated_oof[val_idx] = calibrator.calibrate(fold_val, raw_oof[val_idx])

    # 10,000 Paired Bootstrap Significance Test
    print("\n[QuestionCalibrator] Running 10,000 paired bootstrap resamples on per-sample deltas...")
    cal_scores = eval_per_sample(best_calibrated_oof, y_true, classes)
    deltas = cal_scores - base_scores
    n_samples = len(deltas)

    rng = np.random.default_rng(42)
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

    print(f"  Mean Delta:             {mean_delta:+.5f}")
    print(f"  Standard Error:          {se_delta:.5f}")
    print(f"  95% Confidence Interval: [{ci_lower:+.5f}, {ci_upper:+.5f}]")
    print(f"  P-value (one-tailed <= 0): {p_value:.4f}")
    print(f"  Statistically Significant (p < 0.05): {is_significant}")

    # Serialize manifest
    manifest = {
        "baseline_oof_map3": base_map3,
        "calibrated_oof_map3": champ["map3"],
        "delta": mean_delta,
        "ci_95": [ci_lower, ci_upper],
        "p_value": p_value,
        "is_significant": is_significant,
        "champion_params": {
            "gamma": champ["gamma"],
            "m_q": champ["m_q"],
            "beta": champ["beta"]
        },
        "diagnostics_before": base_diag,
        "diagnostics_after": {
            "false_none": champ["false_none"],
            "correct_none": champ["correct_none"],
            "pred_none_top1": champ["pred_none_top1"]
        }
    }

    manifest_path = artifacts_dir / "question_aware_calibration_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[QuestionCalibrator] Saved calibration manifest to: {manifest_path}")

    # Save calibrated OOF
    np.save(artifacts_dir / "oof_question_calibrated.npy", best_calibrated_oof)
    print(f"[QuestionCalibrator] Saved calibrated OOF to: {artifacts_dir / 'oof_question_calibrated.npy'}")

    # Generate test submission ONLY if statistically significant (p < 0.05, CI > 0)
    if is_significant:
        print("\n[QuestionCalibrator] Applying question-aware calibration to test predictions...")
        # Fit calibrator on 100% of train_df to maximize sample efficiency for test
        full_calibrator = QuestionAwareCalibrator(
            classes=classes, gamma=champ["gamma"], m_q=champ["m_q"], m_tg=5.0, beta=champ["beta"]
        )
        full_calibrator.fit(train_df)

        # Reconstruct uncalibrated test probabilities from champion submission components
        oof_cn = np.load(artifacts_dir / "test_convnext_vision.npy")
        oof_mn = np.load(artifacts_dir / "test_finetuned_vision.npy")
        test_sgd = np.load(artifacts_dir / "test_text.npy")
        test_sem = np.load(artifacts_dir / "test_semantic_text.npy")

        domain_masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df)
        test_topics = [str(t) for t in test_df["topic_name"]]
        test_grades = [int(float(g)) for g in test_df["grade"]]
        test_mn_masked = domain_masker.apply_mask_batch(oof_mn, classes, test_topics, test_grades, damping=0.85)

        test_dual_vis = 0.5 * oof_cn + 0.5 * test_mn_masked
        test_enh_text = 0.4 * test_sgd + 0.6 * test_sem

        # Test adaptive prior
        from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
        train_embeds = np.load(artifacts_dir / "train_text_embeddings_384.npy")
        test_embeds = np.load(artifacts_dir / "test_text_embeddings_384.npy")

        test_prior = np.zeros((n_test, len(classes)), dtype=np.float32)
        for tr_idx, _ in folds:
            fold_tr = train_df.iloc[tr_idx].reset_index(drop=True)
            engine = AdaptiveQuestionPriorEngine(
                classes=classes, m_q_base=10.0, m_tg=10.0, adaptive_nq=True, adaptive_power=0.70,
                use_semantic_backoff=True, semantic_embeddings=train_embeds, semantic_top_k=5,
                semantic_threshold=0.70, semantic_weight=0.50
            )
            engine.fit(fold_tr, fold_train_indices=tr_idx)
            test_prior += engine.predict_proba(test_df, eval_embeddings=test_embeds) / len(folds)

        test_raw_probs = 0.70 * test_prior + 0.14 * test_dual_vis + 0.16 * test_enh_text
        test_calibrated_probs = full_calibrator.calibrate(test_df, test_raw_probs)

        top_idx = np.argsort(-test_calibrated_probs, axis=1, kind="mergesort")[:, :3]
        pred_strings = [" ".join([classes[idx] for idx in row]) for row in top_idx]

        sub_df = pd.DataFrame({
            "trace_id": test_df["trace_id"],
            "root_code": pred_strings
        })
        out_csv = PATHS.submissions_dir / "multimodal_calibrated_fusion_submission.csv"
        sub_df.to_csv(out_csv, index=False)
        print(f"[QuestionCalibrator] Saved calibrated candidate to: {out_csv}")

        val_res = validate_submission_file(out_csv, PATHS.taxonomy_csv)
        print(f"[QuestionCalibrator] Validation status: {val_res}")

        # SOTA agreement
        sota_df = cast(pd.DataFrame, pd.read_csv(PATHS.submissions_dir / "multimodal_adaptive_fusion_submission.csv"))
        diff_count = int((sub_df["root_code"] != sota_df["root_code"]).sum())
        diff_pct = (diff_count / len(sub_df)) * 100
        print(f"\n[Agreement against 0.53383 SOTA Champion]")
        print(f"  Total test rows:        {len(sub_df)}")
        print(f"  Identical to 0.53383:   {len(sub_df) - diff_count} ({(100 - diff_pct):.2f}%)")
        print(f"  Selectively diversified: {diff_count} ({diff_pct:.2f}%)")
    else:
        print(f"\n[QuestionCalibrator] Paired bootstrap did not achieve statistical significance (p={p_value:.4f} >= 0.05, 95% CI [{ci_lower:+.5f}, {ci_upper:+.5f}] crosses zero).")
        print("[QuestionCalibrator] In strict adherence to our SDLC governance, test submission replacement is REJECTED. Champion 0.53383 remains preserved.")

    return manifest


if __name__ == "__main__":
    run_calibration_study()
