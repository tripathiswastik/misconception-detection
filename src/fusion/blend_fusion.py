"""Systematic Comparison of Probability Blending vs. Rank Blending for M6 and M7.

Combines:
- M6: Frozen MobileNetV3 Vision + M5 Math-Enhanced Text (LB: 0.48833, OOF: 0.5409)
- M7: Partially Fine-Tuned MobileNetV3 Vision + M5 Math-Enhanced Text (LB: 0.48749, OOF: 0.5397)

Evaluates:
1. Probability Blending: P_blend = lambda * P_M6 + (1 - lambda) * P_M7
2. Rank Blending: R_blend = alpha * R_M6 + (1 - alpha) * R_M7 (Borda Count on Top-3)

Outputs:
- artifacts/blend_weights.json: Optimal blending parameters and comparative metrics.
- submissions/multimodal_blended_submission.csv: New blended candidate submission.
(Preserves submissions/multimodal_fused_submission.csv intact as fallback.)
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, PATHS
from src.data.dataset import MathMisconceptionData
from src.data.taxonomy import TaxonomyManager
from src.diagnoser.domain_mask import DomainConstraintEngine
from src.fusion.validator import validate_submission_file
from src.metrics.map3 import calculate_map3
from src.models.fusion import MultimodalLateFusion


def run_blend_evaluation() -> Dict[str, Any]:
    artifacts_dir = PROJECT_ROOT / "artifacts"
    
    # 1. Verify existence of all required matrices
    required_files = {
        "oof_text": artifacts_dir / "oof_text.npy",
        "oof_vision_m6": artifacts_dir / "oof_vision.npy",
        "oof_vision_m7": artifacts_dir / "oof_finetuned_vision.npy",
        "test_text": artifacts_dir / "test_text.npy",
        "test_vision_m6": artifacts_dir / "test_vision.npy",
        "test_vision_m7": artifacts_dir / "test_finetuned_vision.npy",
    }
    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact '{name}' at: {path}")

    print("\n=======================================================")
    print("[Blending] Initializing Probability vs. Rank Blending Evaluation")
    print("=======================================================\n")

    # 2. Load probability distributions
    oof_text = np.load(required_files["oof_text"])
    oof_vis_m6 = np.load(required_files["oof_vision_m6"])
    oof_vis_m7 = np.load(required_files["oof_vision_m7"])
    test_text = np.load(required_files["test_text"])
    test_vis_m6 = np.load(required_files["test_vision_m6"])
    test_vis_m7 = np.load(required_files["test_vision_m7"])

    tax = TaxonomyManager(PATHS.taxonomy_csv)
    classes = tax.all_codes(include_none=True)

    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    test_df = data_mgr.load_test()
    y_true = list(train_df["root_code"].astype(str).values)

    # 3. Apply Domain Constraint Masking to Vision branches
    domain_masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df)
    train_topics = [str(t) for t in train_df["topic_name"].values]
    train_grades = [int(g) for g in train_df["grade"].values]
    test_topics = [str(t) for t in test_df["topic_name"].values]
    test_grades = [int(g) for g in test_df["grade"].values]

    oof_m6_masked = domain_masker.apply_mask_batch(
        oof_vis_m6, classes, train_topics, train_grades, damping=CONFIG.domain_mask_damping
    )
    oof_m7_masked = domain_masker.apply_mask_batch(
        oof_vis_m7, classes, train_topics, train_grades, damping=CONFIG.domain_mask_damping
    )
    test_m6_masked = domain_masker.apply_mask_batch(
        test_vis_m6, classes, test_topics, test_grades, damping=CONFIG.domain_mask_damping
    )
    test_m7_masked = domain_masker.apply_mask_batch(
        test_vis_m7, classes, test_topics, test_grades, damping=CONFIG.domain_mask_damping
    )

    # 4. Construct M6 and M7 Fused Probabilities (w_vision = 0.25)
    p_m6_oof = MultimodalLateFusion.apply_fusion(oof_text, oof_m6_masked, vision_weight=0.25)
    p_m7_oof = MultimodalLateFusion.apply_fusion(oof_text, oof_m7_masked, vision_weight=0.25)
    p_m6_test = MultimodalLateFusion.apply_fusion(test_text, test_m6_masked, vision_weight=0.25)
    p_m7_test = MultimodalLateFusion.apply_fusion(test_text, test_m7_masked, vision_weight=0.25)

    # Baseline MAP@3 calculations
    top_m6_oof = np.argsort(-p_m6_oof, axis=1, kind="mergesort")[:, :3]
    top_m7_oof = np.argsort(-p_m7_oof, axis=1, kind="mergesort")[:, :3]
    score_m6_oof = calculate_map3(y_true, [[classes[i] for i in r] for r in top_m6_oof])
    score_m7_oof = calculate_map3(y_true, [[classes[i] for i in r] for r in top_m7_oof])

    print(f"M6 Fused OOF MAP@3: {score_m6_oof:.4f} (Public LB: 0.48833)")
    print(f"M7 Fused OOF MAP@3: {score_m7_oof:.4f} (Public LB: 0.48749)")

    # 5. Probability Blending Sweep: P_blend = lambda * P_M6 + (1 - lambda) * P_M7
    print("\n--- 1. Probability Blending Sweep ---")
    prob_sweep_results = []
    best_prob_score = -1.0
    best_prob_lambda = 0.5

    for lam in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        p_blend = lam * p_m6_oof + (1.0 - lam) * p_m7_oof
        top_indices = np.argsort(-p_blend, axis=1, kind="mergesort")[:, :3]
        score = calculate_map3(y_true, [[classes[i] for i in r] for r in top_indices])
        prob_sweep_results.append({"lambda_m6": lam, "lambda_m7": round(1.0 - lam, 2), "oof_map3": round(score, 5)})
        print(f"  lambda={lam:.1f} (M6={lam*100:3.0f}%, M7={(1-lam)*100:3.0f}%): OOF MAP@3 = {score:.4f}")
        if score > best_prob_score:
            best_prob_score = score
            best_prob_lambda = lam

    print(f"  --> Optimal Probability Blend: lambda={best_prob_lambda:.1f} | OOF MAP@3 = {best_prob_score:.4f}")

    # 6. Rank Blending (Borda Count) Sweep
    print("\n--- 2. Rank Blending (Borda Count on Top-3) Sweep ---")
    rank_sweep_results = []
    best_rank_score = -1.0
    best_rank_alpha = 0.5

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        rank_scores = np.zeros_like(p_m6_oof)
        for i in range(len(y_true)):
            # M6: 1st place -> 3 pts, 2nd -> 2 pts, 3rd -> 1 pt
            for r_pos, c_idx in enumerate(top_m6_oof[i]):
                rank_scores[i, c_idx] += alpha * (3.0 - r_pos)
            # M7: 1st place -> 3 pts, 2nd -> 2 pts, 3rd -> 1 pt
            for r_pos, c_idx in enumerate(top_m7_oof[i]):
                rank_scores[i, c_idx] += (1.0 - alpha) * (3.0 - r_pos)

        top_indices = np.argsort(-rank_scores, axis=1, kind="mergesort")[:, :3]
        score = calculate_map3(y_true, [[classes[i] for i in r] for r in top_indices])
        rank_sweep_results.append({"alpha_m6": alpha, "alpha_m7": round(1.0 - alpha, 2), "oof_map3": round(score, 5)})
        print(f"  alpha={alpha:.1f} (M6={alpha*100:3.0f}%, M7={(1-alpha)*100:3.0f}%): Rank OOF MAP@3 = {score:.4f}")
        if score > best_rank_score:
            best_rank_score = score
            best_rank_alpha = alpha

    print(f"  --> Optimal Rank Blend: alpha={best_rank_alpha:.1f} | OOF MAP@3 = {best_rank_score:.4f}")

    # 7. Comparison Summary
    print("\n=======================================================")
    print("[Blending] Comparative Summary:")
    print(f"  M6 Fused OOF:          {score_m6_oof:.4f} (Public LB: 0.48833)")
    print(f"  M7 Fused OOF:          {score_m7_oof:.4f} (Public LB: 0.48749)")
    print(f"  Best Rank Blend:       {best_rank_score:.4f} (alpha={best_rank_alpha:.1f})")
    print(f"  Best Probability Blend:{best_prob_score:.4f} (lambda={best_prob_lambda:.1f})")
    print(f"  Net OOF Gain:          +{best_prob_score - score_m6_oof:.4f} vs M6")
    print("=======================================================\n")

    # 8. Generate Candidate Blended Submission (using optimal probability blend)
    test_blend = best_prob_lambda * p_m6_test + (1.0 - best_prob_lambda) * p_m7_test
    top_test_indices = np.argsort(-test_blend, axis=1, kind="mergesort")[:, :3]
    pred_strings = [" ".join([classes[idx] for idx in row]) for row in top_test_indices]

    submission_df = pd.DataFrame({
        "trace_id": test_df["trace_id"].astype(str),
        "root_code": pred_strings
    })

    PATHS.submissions_dir.mkdir(parents=True, exist_ok=True)
    out_path = PATHS.submissions_dir / "multimodal_blended_submission.csv"
    submission_df.to_csv(out_path, index=False)
    print(f"[Blending] Generated blended submission: {out_path} ({len(submission_df)} rows)")

    validation = validate_submission_file(out_path, PATHS.taxonomy_csv)
    print(f"[Blending] Submission validation: {validation}")
    if not validation["valid"]:
        raise ValueError(f"Submission validation failed: {validation['errors']}")

    # 9. Persist blend configuration
    blend_metadata = {
        "m6_oof_map3": score_m6_oof,
        "m7_oof_map3": score_m7_oof,
        "best_prob_lambda_m6": best_prob_lambda,
        "best_prob_lambda_m7": 1.0 - best_prob_lambda,
        "best_prob_oof_map3": best_prob_score,
        "best_rank_alpha_m6": best_rank_alpha,
        "best_rank_alpha_m7": 1.0 - best_rank_alpha,
        "best_rank_oof_map3": best_rank_score,
        "probability_sweep": prob_sweep_results,
        "rank_sweep": rank_sweep_results
    }
    with open(artifacts_dir / "blend_weights.json", "w", encoding="utf-8") as f:
        json.dump(blend_metadata, f, indent=2)

    return blend_metadata


if __name__ == "__main__":
    run_blend_evaluation()
