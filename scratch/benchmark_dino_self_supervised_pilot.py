"""Self-Supervised Vision Pilot: Benchmark DINO (ViT-S/16) against Existing Visual Stack.

Methodological Protocol per User Directive:
1. Representation Comparison:
   - Baseline: Existing Visual Stack (ConvNeXt-448 768-d + Swin-T 768-d + TrOCR DeiT 384-d + Strokes 24-d = 1944-d).
   - Candidate A: Frozen DINO ViT-S/16 Standalone (384-d).
   - Candidate B: Augmented Visual Stack (Existing 1944-d + DINO 384-d = 2328-d).
2. Identical Classifier & Calibration:
   - Same regularized linear metric classifier across all conditions.
   - Same in-fold adaptive question prior engine (0.55 * P_vis + 0.45 * P_prior).
3. Separate Reporting for Seen & Unseen Questions:
   - Inductive Evaluation: 5-Fold GroupKFold on question_id (100% unseen questions, zero question leakage).
   - Transductive Evaluation: 5-Fold Stratified Split on multi-trace questions (95.7% test-mix proxy).
   - Composite Metric: 0.957 * Seen MAP@3 + 0.043 * Unseen MAP@3.
4. Granular Retrieval Audit:
   - Computes Recall@1, Recall@2, Recall@3, and Recall@5 to measure whether DINO brings new candidates into top-3.
"""
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.metrics.map3 import calculate_map3

def compute_recall_at_k(y_true: List[str], top_preds: List[List[str]], k_list: List[int] = [1, 2, 3, 5]) -> Dict[str, float]:
    recalls = {}
    n = len(y_true)
    for k in k_list:
        hits = sum(1 for i in range(n) if y_true[i] in top_preds[i][:k])
        recalls[f"Recall@{k}"] = round(hits / max(n, 1), 4)
    return recalls

def evaluate_feature_matrix_gkf(
    X_features: np.ndarray,
    train_df: pd.DataFrame,
    y_raw: np.ndarray,
    y_indices: np.ndarray,
    groups: np.ndarray,
    classes: List[str],
    n_splits: int = 5
) -> Tuple[float, List[float], List[List[str]], Dict[str, float]]:
    """Evaluates features under strict 5-fold GroupKFold (Unseen Questions)."""
    n_samples = len(train_df)
    n_classes = len(classes)
    gkf = GroupKFold(n_splits=n_splits)

    oof_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    fold_scores: List[float] = []

    for fold_num, (trn_idx, val_idx) in enumerate(gkf.split(train_df, y_indices, groups=groups), start=1):
        trn_df = train_df.iloc[trn_idx]
        val_df = train_df.iloc[val_idx]
        val_y = y_raw[val_idx].tolist()

        clf = RidgeClassifier(alpha=10.0, random_state=42)
        clf.fit(X_features[trn_idx], y_indices[trn_idx])

        # Calibrated probabilities via softmax over decision function
        df_val = clf.decision_function(X_features[val_idx])
        exp_df = np.exp(df_val - np.max(df_val, axis=1, keepdims=True))
        p_clf_val = exp_df / exp_df.sum(axis=1, keepdims=True)

        p_clf_full = np.zeros((len(val_idx), n_classes), dtype=np.float64)
        for col_i, cls_i in enumerate(clf.classes_):
            p_clf_full[:, cls_i] = p_clf_val[:, col_i]

        prior_eng = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df)
        p_prior_val = prior_eng.predict_proba(val_df).astype(np.float64)

        p_blend = 0.55 * p_clf_full + 0.45 * p_prior_val
        oof_probs[val_idx] = p_blend

        top3_val = [[classes[idx] for idx in row[:3]] for row in np.argsort(-p_blend, axis=1)]
        s_f = calculate_map3(val_y, top3_val)
        fold_scores.append(s_f)

    all_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-oof_probs, axis=1)]
    overall_map3 = calculate_map3(y_raw.tolist(), all_top3)
    recall_stats = compute_recall_at_k(y_raw.tolist(), all_top3)
    return overall_map3, fold_scores, all_top3, recall_stats

def evaluate_feature_matrix_stratified(
    X_features: np.ndarray,
    train_df: pd.DataFrame,
    y_raw: np.ndarray,
    y_indices: np.ndarray,
    classes: List[str],
    n_splits: int = 5
) -> Tuple[float, List[float], List[List[str]], Dict[str, float]]:
    """Evaluates features under 5-fold stratified split on multi-trace questions (Seen Questions)."""
    # Filter to questions with >= 2 traces (multi-trace questions)
    q_counts = Counter(train_df["question_id"])
    multi_q_mask = np.array([q_counts[q] >= 2 for q in train_df["question_id"]])

    sub_df = train_df.iloc[multi_q_mask].copy().reset_index(drop=True)
    sub_X = X_features[multi_q_mask]
    sub_y_raw = y_raw[multi_q_mask]
    sub_y_indices = y_indices[multi_q_mask]
    n_sub = len(sub_df)
    n_classes = len(classes)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probs = np.zeros((n_sub, n_classes), dtype=np.float64)
    fold_scores: List[float] = []

    for trn_idx, val_idx in kf.split(sub_df):
        trn_df = sub_df.iloc[trn_idx]
        val_df = sub_df.iloc[val_idx]
        val_y = sub_y_raw[val_idx].tolist()

        clf = RidgeClassifier(alpha=10.0, random_state=42)
        clf.fit(sub_X[trn_idx], sub_y_indices[trn_idx])

        df_val = clf.decision_function(sub_X[val_idx])
        exp_df = np.exp(df_val - np.max(df_val, axis=1, keepdims=True))
        p_clf_val = exp_df / exp_df.sum(axis=1, keepdims=True)

        p_clf_full = np.zeros((len(val_idx), n_classes), dtype=np.float64)
        for col_i, cls_i in enumerate(clf.classes_):
            p_clf_full[:, cls_i] = p_clf_val[:, col_i]

        prior_eng = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df)
        p_prior_val = prior_eng.predict_proba(val_df).astype(np.float64)

        p_blend = 0.55 * p_clf_full + 0.45 * p_prior_val
        oof_probs[val_idx] = p_blend

        top3_val = [[classes[idx] for idx in row[:3]] for row in np.argsort(-p_blend, axis=1)]
        s_f = calculate_map3(val_y, top3_val)
        fold_scores.append(s_f)

    all_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-oof_probs, axis=1)]
    overall_map3 = calculate_map3(sub_y_raw.tolist(), all_top3)
    recall_stats = compute_recall_at_k(sub_y_raw.tolist(), all_top3)
    return overall_map3, fold_scores, all_top3, recall_stats

def main():
    print("=" * 80)
    print("SELF-SUPERVISED VISION PILOT: DINO (ViT-S/16) BENCHMARK")
    print("Comparing DINO Frozen Embeddings against Existing Visual Stack")
    print("=" * 80)

    start_time = time.time()

    # 1. Load Taxonomy & Data
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_raw = train_df["root_code"].fillna("NONE").astype(str).values
    y_indices = np.array([class_to_idx[c] for c in y_raw], dtype=np.int64)
    groups = train_df["question_id"].values
    n_samples = len(train_df)

    # 2. Load Visual Features
    print("\nLoading visual representations...")
    cn_feat = np.load(ART_DIR / "train_convnext_features.npy")
    sw_feat = np.load(ART_DIR / "train_swin_features.npy")
    tr_feat = np.load(ART_DIR / "train_trocr_features.npy")
    st_feat = np.load(ART_DIR / "train_perceptual_strokes.npy")
    dino_feat = np.load(ART_DIR / "train_dino_features_384.npy")

    # Normalized representations
    cn_norm = cast(np.ndarray, normalize(cn_feat))
    sw_norm = cast(np.ndarray, normalize(sw_feat))
    tr_norm = cast(np.ndarray, normalize(tr_feat))
    st_norm = cast(np.ndarray, normalize(st_feat))
    dino_norm = cast(np.ndarray, normalize(dino_feat))

    # Feature Configurations to Compare
    configs: Dict[str, np.ndarray] = {
        "1. Baseline Visual Stack (1944-d)": np.hstack([cn_norm, sw_norm, tr_norm, st_norm]),
        "2. DINO ViT-S/16 Standalone (384-d)": dino_norm,
        "3. Baseline Stack + DINO (2328-d)": np.hstack([cn_norm, sw_norm, tr_norm, st_norm, dino_norm]),
        "4. DINO + ConvNeXt + Swin (1920-d)": np.hstack([dino_norm, cn_norm, sw_norm])
    }

    results: Dict[str, Any] = {}

    print("\n" + "=" * 80)
    print("STEP 1: INDUCTIVE EVALUATION (100% UNSEEN QUESTIONS - GroupKFold)")
    print("=" * 80)

    for name, feat_mat in configs.items():
        score, f_scores, top3, recalls = evaluate_feature_matrix_gkf(
            feat_mat, train_df, y_raw, y_indices, groups, classes, n_splits=5
        )
        results[name] = {
            "unseen_map3": round(score, 5),
            "unseen_folds": [round(s, 5) for s in f_scores],
            "unseen_recalls": recalls
        }
        print(f"\n{name}:")
        print(f"  Unseen Questions MAP@3: {score:.5f} | Folds: {[round(s, 5) for s in f_scores]}")
        print(f"  Retrieval: {recalls}")

    print("\n" + "=" * 80)
    print("STEP 2: TRANSDUCTIVE EVALUATION (SEEN QUESTIONS - 95.7% Test-Mix Proxy)")
    print("=" * 80)

    for name, feat_mat in configs.items():
        score, f_scores, top3, recalls = evaluate_feature_matrix_stratified(
            feat_mat, train_df, y_raw, y_indices, classes, n_splits=5
        )
        results[name]["seen_map3"] = round(score, 5)
        results[name]["seen_folds"] = [round(s, 5) for s in f_scores]
        results[name]["seen_recalls"] = recalls

        # Composite score: 0.957 * Seen + 0.043 * Unseen
        unseen_sc = results[name]["unseen_map3"]
        composite = 0.957 * score + 0.043 * unseen_sc
        results[name]["composite_map3"] = round(composite, 5)

        print(f"\n{name}:")
        print(f"  Seen Questions MAP@3   : {score:.5f} | Folds: {[round(s, 5) for s in f_scores]}")
        print(f"  Retrieval: {recalls}")
        print(f"  Competition Composite  : {composite:.5f}")

    # Comparative Deltas against Baseline
    base_name = "1. Baseline Visual Stack (1944-d)"
    base_unseen = results[base_name]["unseen_map3"]
    base_seen = results[base_name]["seen_map3"]
    base_comp = results[base_name]["composite_map3"]
    base_r3_unseen = results[base_name]["unseen_recalls"]["Recall@3"]
    base_r3_seen = results[base_name]["seen_recalls"]["Recall@3"]

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON vs BASELINE VISUAL STACK")
    print("=" * 80)
    print(f"{'Configuration':38s} | {'Unseen MAP@3':12s} | {'Seen MAP@3':10s} | {'Composite':10s} | {'Unseen R@3':10s} | {'Seen R@3':8s}")
    print("-" * 105)

    for name, data in results.items():
        u_d = data["unseen_map3"] - base_unseen
        s_d = data["seen_map3"] - base_seen
        c_d = data["composite_map3"] - base_comp
        r3_u_d = data["unseen_recalls"]["Recall@3"] - base_r3_unseen
        r3_s_d = data["seen_recalls"]["Recall@3"] - base_r3_seen

        print(f"{name:38s} | {data['unseen_map3']:.5f} ({u_d:+.5f}) | {data['seen_map3']:.5f} ({s_d:+.5f}) | {data['composite_map3']:.5f} ({c_d:+.5f}) | {data['unseen_recalls']['Recall@3']:.1%} ({r3_u_d:+.1%}) | {data['seen_recalls']['Recall@3']:.1%} ({r3_s_d:+.1%})")

    # Decision Gate
    aug_name = "3. Baseline Stack + DINO (2328-d)"
    aug_comp_delta = results[aug_name]["composite_map3"] - base_comp
    aug_unseen_delta = results[aug_name]["unseen_map3"] - base_unseen
    dino_promoted = (aug_comp_delta >= 0.00050) and (aug_unseen_delta >= 0.0)

    print("\n" + "=" * 80)
    print("DECISION GATE: " + ("PROCEED WITH DINO" if dino_promoted else "STOP / DINO DOES NOT IMPROVE VISUAL STACK"))
    print("=" * 80)

    # Save artifact
    out_artifact = {
        "pilot_name": "dino_self_supervised_vision_pilot",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methodology": {
            "encoder": "facebook/dino-vits16 (self-supervised ViT-S/16, frozen [CLS] token)",
            "inductive_scheme": "5-Fold GroupKFold on question_id (100% unseen questions)",
            "transductive_scheme": "5-Fold Stratified Split on multi-trace questions (95.7% test-mix proxy)",
            "composite_formula": "0.957 * Seen MAP@3 + 0.043 * Unseen MAP@3"
        },
        "results": results,
        "deltas_vs_baseline": {
            "dino_standalone": {
                "composite_delta": round(results["2. DINO ViT-S/16 Standalone (384-d)"]["composite_map3"] - base_comp, 5),
                "unseen_delta": round(results["2. DINO ViT-S/16 Standalone (384-d)"]["unseen_map3"] - base_unseen, 5)
            },
            "baseline_plus_dino": {
                "composite_delta": round(aug_comp_delta, 5),
                "unseen_delta": round(aug_unseen_delta, 5)
            }
        },
        "decision": {
            "promoted": dino_promoted,
            "recommendation": "Incorporate DINO into ensemble" if dino_promoted else "Stop DINO self-supervised exploration; does not outperform ConvNeXt + Swin stack"
        },
        "runtime_seconds": round(time.time() - start_time, 2)
    }

    report_path = ART_DIR / "dino_self_supervised_pilot_report.json"
    with open(report_path, "w") as f:
        json.dump(out_artifact, f, indent=2)
    print(f"\n[Saved Report Artifact] {report_path.name}")

if __name__ == "__main__":
    main()

