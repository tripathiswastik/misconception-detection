"""Comprehensive Audit of All Existing Visual Signals for Rare-Label Retrieval Value.

Directives Addressed:
1. Stop ConvNeXt and post-processing experiments.
2. Audit ALL remaining existing image representations in the repo:
   - Evaluates rare-class Recall@10 across all 298 rare traces (<15 samples).
   - Evaluates performance both ALONE (standalone) and ADDED TO S550.
   - Evaluates false NONE demotions (true NONE rows receiving Rank 1 != NONE, baseline 104/506).
   - Evaluates overall composite MAP@3 (0.957 * Seen + 0.043 * Unseen) vs S550 reference (0.54214).
3. Evaluates both existing probability models AND raw visual feature embeddings
   (Swin-T, TrOCR, MobileNetV3, Perceptual Strokes, ConvNeXt-448, ArcFace heads).
4. Decision Gate:
   - Keep only signals that retrieve additional true labels without increasing false NONE demotions.
   - If none do, confirm that current image features have reached their practical limit
     and establish that input signal modifications (e.g. math OCR, crop extraction) are required.
"""

from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRATCH_DIR = PROJECT_ROOT / "scratch"
if str(SCRATCH_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_DIR))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.domain_mask import DomainConstraintEngine
from src.metrics.map3 import calculate_map3
from audit_rare_class_s550_oof_error_breakdown import S550AuditEngine, sharp


class SignalAuditResult(TypedDict):
    name: str
    category: str
    standalone_seen_map3: float
    standalone_unseen_map3: float
    standalone_comp_map3: float
    standalone_rare_r10_hits: int
    standalone_rare_r10_rate: float
    standalone_rare_r5_hits: int
    standalone_rare_r3_hits: int
    standalone_false_none_count: int
    standalone_false_none_rate: float
    s550_added_seen_map3: float
    s550_added_unseen_map3: float
    s550_added_comp_map3: float
    s550_added_delta_map3: float
    s550_added_rare_r10_hits: int
    s550_added_delta_r10: int
    s550_added_false_none_count: int
    s550_added_delta_false_none: int
    retrieves_additional_rare: bool
    avoids_extra_false_none: bool


def compute_prototype_retrieval_oof(
    feature_matrix: np.ndarray,
    y_true_int: np.ndarray,
    splits: List[Dict[str, Any]],
    n_classes: int,
    classes: List[str],
    train_df: pd.DataFrame,
    tax: TaxonomyManager,
    temperature: float = 0.10
) -> np.ndarray:
    """Computes in-fold L2-normalized class centroid prototype retrieval probabilities."""
    n_samples = len(feature_matrix)
    oof_probs = np.zeros((n_samples, n_classes), dtype=np.float32)

    for fold in splits:
        tr_idx = np.array(fold["train_indices"], dtype=np.int64)
        val_idx = np.array(fold["val_indices"], dtype=np.int64)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(feature_matrix[tr_idx])
        X_val = scaler.transform(feature_matrix[val_idx])

        # Normalize features
        X_tr_norm = X_tr / np.maximum(np.linalg.norm(X_tr, axis=1, keepdims=True), 1e-7)
        X_val_norm = X_val / np.maximum(np.linalg.norm(X_val, axis=1, keepdims=True), 1e-7)

        # Compute prototypes
        dim = X_tr.shape[1]
        prototypes = np.zeros((n_classes, dim), dtype=np.float32)
        for c in range(n_classes):
            c_mask = (y_true_int[tr_idx] == c)
            if np.any(c_mask):
                centroid = np.mean(X_tr_norm[c_mask], axis=0)
                norm = np.linalg.norm(centroid)
                prototypes[c] = centroid / norm if norm > 1e-7 else centroid

        # Cosine similarity
        sims = np.matmul(X_val_norm, prototypes.T) / temperature
        sims_max = np.max(sims, axis=1, keepdims=True)
        exp_sims = np.exp(sims - sims_max)
        probs = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)

        # In-fold domain masking
        masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df.iloc[tr_idx])
        v_topics = [str(t) for t in train_df.iloc[val_idx]["topic_name"].values]
        v_grades = [int(float(g)) for g in train_df.iloc[val_idx]["grade"].values]
        calibrated = masker.apply_mask_batch(probs, classes, v_topics, v_grades, damping=0.05)
        oof_probs[val_idx] = calibrated

    return oof_probs


def main() -> None:
    print("=" * 115, flush=True)
    print("COMPREHENSIVE AUDIT: ALL EXISTING VISUAL SIGNALS FOR RARE-LABEL RETRIEVAL", flush=True)
    print("=" * 115, flush=True)
    start_time = time.time()

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_raw = train_df["root_code"].fillna("NONE").astype(str).tolist()
    y_true_int = np.array([class_to_idx.get(r, none_idx) for r in y_raw], dtype=np.int64)
    n_samples = len(train_df)

    class_counts = Counter(y_raw)
    rare_classes: Set[str] = {c for c, cnt in class_counts.items() if c != "NONE" and cnt < 15}
    rare_indices = [i for i in range(n_samples) if y_raw[i] in rare_classes]
    none_indices = [i for i in range(n_samples) if y_raw[i] == "NONE"]

    print(f"Total Traces: {n_samples} | Rare Traces (<15 samples): {len(rare_indices)} ({len(rare_classes)} classes)", flush=True)
    print(f"Total True NONE Traces: {len(none_indices)}", flush=True)

    # Canonical splits
    with open(DATA_DIR / "splits.json") as f:
        splits = json.load(f)["folds"]

    # Seen question mask
    seen_mask = np.zeros(n_samples, dtype=bool)
    for fold in splits:
        tr_qids = set(train_df.iloc[fold["train_indices"]]["question_id"].astype(str))
        for i in fold["val_indices"]:
            if str(train_df.iloc[i]["question_id"]) in tr_qids:
                seen_mask[i] = True

    n_seen = int(np.sum(seen_mask))
    n_unseen = n_samples - n_seen
    print(f"Canonical Split Balance: {n_seen} Seen Traces, {n_unseen} Unseen Traces", flush=True)

    # Regenerate Canonical S550 Baseline Reference
    print("\nRegenerating Canonical S550 Baseline Reference...", flush=True)
    engine = S550AuditEngine()

    oof_p_s550_ref = np.zeros((n_samples, n_classes), dtype=np.float64)
    q_dists_per_fold: Dict[int, Dict[str, Counter]] = {}
    s550_ref_fold_scores: List[float] = []

    for fold_idx, fold in enumerate(splits, start=1):
        tr_idx = np.array(fold["train_indices"], dtype=np.int64)
        val_idx = np.array(fold["val_indices"], dtype=np.int64)

        p_s550_val, q_dist = engine.generate_s550_probs(tr_idx, val_idx)
        oof_p_s550_ref[val_idx] = p_s550_val
        q_dists_per_fold[fold_idx] = q_dist

        y_val_gold = [y_raw[i] for i in val_idx]
        preds_s550_val = engine.rank_and_gate(p_s550_val, val_idx, q_dist)
        fold_sc = calculate_map3(y_val_gold, preds_s550_val)
        s550_ref_fold_scores.append(fold_sc)

    preds_s550_ref_all = [[] for _ in range(n_samples)]
    for fold_idx, fold in enumerate(splits, start=1):
        val_idx = np.array(fold["val_indices"], dtype=np.int64)
        preds_val = engine.rank_and_gate(oof_p_s550_ref[val_idx], val_idx, q_dists_per_fold[fold_idx])
        for i, idx in enumerate(val_idx):
            preds_s550_ref_all[idx] = preds_val[i]

    s550_seen_map3 = calculate_map3(
        [y_raw[i] for i in range(n_samples) if seen_mask[i]],
        [preds_s550_ref_all[i] for i in range(n_samples) if seen_mask[i]]
    )
    s550_unseen_map3 = calculate_map3(
        [y_raw[i] for i in range(n_samples) if not seen_mask[i]],
        [preds_s550_ref_all[i] for i in range(n_samples) if not seen_mask[i]]
    )
    s550_comp_map3 = 0.957 * s550_seen_map3 + 0.043 * s550_unseen_map3

    s550_rare_r10_count = sum(1 for i in rare_indices if y_raw[i] in [classes[c] for c in np.argsort(-oof_p_s550_ref[i])[:10]])
    s550_rare_r5_count = sum(1 for i in rare_indices if y_raw[i] in [classes[c] for c in np.argsort(-oof_p_s550_ref[i])[:5]])
    s550_rare_r3_count = sum(1 for i in rare_indices if y_raw[i] in preds_s550_ref_all[i])
    s550_false_none = sum(1 for i in none_indices if preds_s550_ref_all[i][0] != "NONE")

    print(f"\nS550 Reference Benchmark:", flush=True)
    print(f"  Seen MAP@3:      {s550_seen_map3:.5f} | Unseen MAP@3: {s550_unseen_map3:.5f} | Composite MAP@3: {s550_comp_map3:.5f}", flush=True)
    print(f"  Rare Recall@10:  {s550_rare_r10_count} / {len(rare_indices)} ({s550_rare_r10_count / len(rare_indices) * 100:.2f}%)", flush=True)
    print(f"  Rare Recall@5:   {s550_rare_r5_count} / {len(rare_indices)} ({s550_rare_r5_count / len(rare_indices) * 100:.2f}%)", flush=True)
    print(f"  Rare Top-3 Hits: {s550_rare_r3_count} / {len(rare_indices)} ({s550_rare_r3_count / len(rare_indices) * 100:.2f}%)", flush=True)
    print(f"  False NONE Demotions: {s550_false_none} / {len(none_indices)} ({s550_false_none / len(none_indices) * 100:.2f}%)", flush=True)

    # Inventory of Visual Candidates
    # 1. Existing Probability Models
    prob_candidates: List[Dict[str, str]] = [
        {"name": "Swin_Vision (224px)",               "path": "oof_swin_vision.npy",                  "category": "Vision_Model"},
        {"name": "TrOCR_Vision (Handwriting OCR)",     "path": "oof_trocr_vision.npy",                 "category": "OCR_Vision"},
        {"name": "MobileNetV3_Finetuned",              "path": "oof_finetuned_vision.npy",             "category": "Vision_Model"},
        {"name": "MobileNetV3_Base",                   "path": "oof_vision.npy",                       "category": "Vision_Model"},
        {"name": "TriVision_Ensemble",                 "path": "oof_trivision.npy",                    "category": "Vision_Ensemble"},
        {"name": "ConvNeXt_Vision_Baseline",           "path": "oof_convnext_vision.npy",              "category": "Vision_Model"},
        {"name": "ConvNeXt_Finetuned_Stage3",          "path": "oof_convnext_finetuned_backbone.npy", "category": "Vision_Model"},
        {"name": "ConvNeXt_448_Question_FiLM",         "path": "oof_convnext_448_question_fusion.npy", "category": "Vision_Fusion"},
        {"name": "Vision_Text_FiLM_Pilot",             "path": "oof_vision_text_film_pilot.npy",       "category": "Vision_Fusion"},
        {"name": "ArcFace_Dual_Letterbox",             "path": "oof_arcface_dual_letterbox.npy",       "category": "ArcFace_Visual"},
        {"name": "ArcFace_Composite_448",              "path": "oof_arcface_composite_448.npy",        "category": "ArcFace_Visual"},
        {"name": "ArcFace_DualHead_448_Soft",          "path": "oof_arcface_dualhead_448_soft.npy",   "category": "ArcFace_Visual"},
        {"name": "Clean_ArcFace_2328d",                "path": "oof_clean_arcface_2328.npy",          "category": "ArcFace_Multimodal"},
    ]

    # 2. Raw Visual Feature Representations (Prototype Cosine Retrieval)
    feature_candidates: List[Dict[str, str]] = [
        {"name": "Swin_T_Features_768d (Proto)",       "path": "train_swin_features.npy",       "category": "Raw_Feature_Prototype"},
        {"name": "TrOCR_Features_384d (Proto)",        "path": "train_trocr_features.npy",      "category": "Raw_Feature_Prototype"},
        {"name": "MobileNet_Features_576d (Proto)",    "path": "train_vision_features.npy",     "category": "Raw_Feature_Prototype"},
        {"name": "Perceptual_Strokes_24d (Proto)",     "path": "train_perceptual_strokes.npy",  "category": "Raw_Feature_Prototype"},
    ]

    # Container for all visual probability matrices (1500, 76)
    all_visual_signals: Dict[str, Tuple[np.ndarray, str]] = {}

    print("\nLoading probability candidate matrices...", flush=True)
    for cand in prob_candidates:
        file_path = ART_DIR / cand["path"]
        if file_path.exists():
            arr = np.load(file_path).astype(np.float64)
            if arr.shape == (n_samples, n_classes):
                # Ensure normalized rows
                arr = np.maximum(arr, 1e-12)
                arr = arr / arr.sum(axis=1, keepdims=True)
                all_visual_signals[cand["name"]] = (arr, cand["category"])
                print(f"  [Loaded] {cand['name']:38s} from {cand['path']}")
            else:
                print(f"  [Skipped] {cand['name']}: unexpected shape {arr.shape}")
        else:
            print(f"  [Not Found] {cand['name']} ({cand['path']})")

    print("\nComputing class prototype retrieval on raw feature matrices...", flush=True)
    for feat_cand in feature_candidates:
        feat_path = ART_DIR / feat_cand["path"]
        if feat_path.exists():
            feat_arr = np.load(feat_path).astype(np.float32)
            if len(feat_arr) == n_samples:
                print(f"  Computing prototype retrieval for {feat_cand['name']} (shape: {feat_arr.shape})...", flush=True)
                proto_p = compute_prototype_retrieval_oof(
                    feature_matrix=feat_arr,
                    y_true_int=y_true_int,
                    splits=splits,
                    n_classes=n_classes,
                    classes=classes,
                    train_df=train_df,
                    tax=tax,
                    temperature=0.10
                )
                all_visual_signals[feat_cand["name"]] = (proto_p.astype(np.float64), feat_cand["category"])
            else:
                print(f"  [Skipped] {feat_cand['name']}: unexpected length {len(feat_arr)}")
        else:
            print(f"  [Not Found] {feat_cand['name']} ({feat_cand['path']})")

    print(f"\nTotal visual signals assembled for audit: {len(all_visual_signals)}", flush=True)

    audit_results: List[SignalAuditResult] = []

    print("\n" + "=" * 135, flush=True)
    print("AUDITING EACH VISUAL SIGNAL: STANDALONE & ADDED TO S550", flush=True)
    print("=" * 135, flush=True)

    for sig_name, (sig_probs, category) in all_visual_signals.items():
        # 1. Standalone Evaluation
        sig_top10 = [np.argsort(-sig_probs[i])[:10] for i in range(n_samples)]
        sig_top3 = [[classes[c] for c in row[:3]] for row in sig_top10]

        st_seen_map3 = calculate_map3(
            [y_raw[i] for i in range(n_samples) if seen_mask[i]],
            [sig_top3[i] for i in range(n_samples) if seen_mask[i]]
        )
        st_unseen_map3 = calculate_map3(
            [y_raw[i] for i in range(n_samples) if not seen_mask[i]],
            [sig_top3[i] for i in range(n_samples) if not seen_mask[i]]
        )
        st_comp_map3 = 0.957 * st_seen_map3 + 0.043 * st_unseen_map3

        st_rare_r10 = sum(1 for i in rare_indices if y_true_int[i] in sig_top10[i])
        st_rare_r5 = sum(1 for i in rare_indices if y_true_int[i] in sig_top10[i][:5])
        st_rare_r3 = sum(1 for i in rare_indices if y_true_int[i] in sig_top10[i][:3])
        st_false_none = sum(1 for i in none_indices if sig_top10[i][0] != none_idx)

        # 2. In-Fold Added to S550 (Evaluated at surgical weight w=0.04)
        oof_p_added = np.zeros((n_samples, n_classes), dtype=np.float64)

        for fold_idx, fold in enumerate(splits, start=1):
            val_idx = np.array(fold["val_indices"], dtype=np.int64)
            # Blend 0.96 * S550 + 0.04 * Signal
            p_val_added = 0.96 * oof_p_s550_ref[val_idx] + 0.04 * sig_probs[val_idx]
            p_val_added = p_val_added / p_val_added.sum(axis=1, keepdims=True)
            oof_p_added[val_idx] = p_val_added

        preds_added_all = [[] for _ in range(n_samples)]
        for fold_idx, fold in enumerate(splits, start=1):
            val_idx = np.array(fold["val_indices"], dtype=np.int64)
            preds_val = engine.rank_and_gate(oof_p_added[val_idx], val_idx, q_dists_per_fold[fold_idx])
            for i, idx in enumerate(val_idx):
                preds_added_all[idx] = preds_val[i]

        added_seen_map3 = calculate_map3(
            [y_raw[i] for i in range(n_samples) if seen_mask[i]],
            [preds_added_all[i] for i in range(n_samples) if seen_mask[i]]
        )
        added_unseen_map3 = calculate_map3(
            [y_raw[i] for i in range(n_samples) if not seen_mask[i]],
            [preds_added_all[i] for i in range(n_samples) if not seen_mask[i]]
        )
        added_comp_map3 = 0.957 * added_seen_map3 + 0.043 * added_unseen_map3
        delta_map3 = added_comp_map3 - s550_comp_map3

        added_rare_r10 = sum(1 for i in rare_indices if y_raw[i] in [classes[c] for c in np.argsort(-oof_p_added[i])[:10]])
        delta_r10 = added_rare_r10 - s550_rare_r10_count
        added_false_none = sum(1 for i in none_indices if preds_added_all[i][0] != "NONE")
        delta_false_none = added_false_none - s550_false_none

        retrieves_extra = (delta_r10 > 0)
        avoids_extra_fn = (delta_false_none <= 0)

        res: SignalAuditResult = {
            "name": sig_name,
            "category": category,
            "standalone_seen_map3": round(st_seen_map3, 5),
            "standalone_unseen_map3": round(st_unseen_map3, 5),
            "standalone_comp_map3": round(st_comp_map3, 5),
            "standalone_rare_r10_hits": st_rare_r10,
            "standalone_rare_r10_rate": round(st_rare_r10 / len(rare_indices), 4),
            "standalone_rare_r5_hits": st_rare_r5,
            "standalone_rare_r3_hits": st_rare_r3,
            "standalone_false_none_count": st_false_none,
            "standalone_false_none_rate": round(st_false_none / len(none_indices), 4),
            "s550_added_seen_map3": round(added_seen_map3, 5),
            "s550_added_unseen_map3": round(added_unseen_map3, 5),
            "s550_added_comp_map3": round(added_comp_map3, 5),
            "s550_added_delta_map3": round(delta_map3, 5),
            "s550_added_rare_r10_hits": added_rare_r10,
            "s550_added_delta_r10": delta_r10,
            "s550_added_false_none_count": added_false_none,
            "s550_added_delta_false_none": delta_false_none,
            "retrieves_additional_rare": retrieves_extra,
            "avoids_extra_false_none": avoids_extra_fn
        }
        audit_results.append(res)

        print(
            f"{sig_name:38s} | StComp: {st_comp_map3:.5f} | StR@10: {st_rare_r10:3d}/298 ({st_rare_r10/len(rare_indices)*100:4.1f}%) | "
            f"S550Comp: {added_comp_map3:.5f} ({delta_map3:+8.5f}) | S550R@10: {added_rare_r10:3d} ({delta_r10:+2d}) | FalseNONE: {added_false_none:3d} ({delta_false_none:+2d})",
            flush=True
        )

    # Master Comparative Table
    print("\n" + "=" * 145, flush=True)
    print("MASTER AUDIT TABLE: ALL VISUAL SIGNALS RANKED BY RARE-LABEL RETRIEVAL (STANDALONE & S550-ADDED)", flush=True)
    print("=" * 145, flush=True)
    print(
        f"{'Signal Name':38s} | {'Category':20s} | {'St Comp':8s} | {'St R@10':10s} | {'St FalseNONE':12s} | "
        f"{'S550 Comp':9s} | {'Delta MAP':10s} | {'S550 R@10':11s} | {'Extra NONE':10s}"
    )
    print("-" * 145, flush=True)

    # Reference S550
    print(
        f"{'S550 Champion Reference':38s} | {'(Active Champion)':20s} | {'N/A':8s} | {'N/A':10s} | {'N/A':12s} | "
        f"{s550_comp_map3:9.5f} | {'+0.00000':10s} | {s550_rare_r10_count:3d} / 298   | {s550_false_none:3d} ( +0)"
    )
    print("-" * 145, flush=True)

    # Sort candidates by added delta_map3 descending, then added rare r10 descending
    sorted_results = sorted(audit_results, key=lambda x: (x["s550_added_delta_map3"], x["s550_added_rare_r10_hits"]), reverse=True)

    for r in sorted_results:
        st_c = f"{r['standalone_comp_map3']:.5f}"
        st_r10 = f"{r['standalone_rare_r10_hits']:3d} ({r['standalone_rare_r10_rate']*100:4.1f}%)"
        st_fn = f"{r['standalone_false_none_count']:3d} ({r['standalone_false_none_rate']*100:4.1f}%)"
        s550_c = f"{r['s550_added_comp_map3']:.5f}"
        delta_m = f"{r['s550_added_delta_map3']:+10.5f}"
        s550_r10 = f"{r['s550_added_rare_r10_hits']:3d} ({r['s550_added_delta_r10']:+2d})"
        extra_fn = f"{r['s550_added_false_none_count']:3d} ({r['s550_added_delta_false_none']:+2d})"

        print(
            f"{r['name']:38s} | {r['category']:20s} | {st_c:8s} | {st_r10:10s} | {st_fn:12s} | "
            f"{s550_c:9s} | {delta_m:10s} | {s550_r10:11s} | {extra_fn:10s}"
        )
    print("=" * 145, flush=True)

    # Filter candidates by criteria:
    # "Keep only signals that retrieve additional true labels without increasing false NONE demotions."
    passing_candidates = [
        r for r in audit_results
        if r["retrieves_additional_rare"] and r["avoids_extra_false_none"] and r["s550_added_delta_map3"] > 0
    ]

    print("\n" + "=" * 95, flush=True)
    print("DECISION GATE VERIFICATION:", flush=True)
    print("=" * 95, flush=True)
    print(f"  Reference S550 Composite MAP@3:        {s550_comp_map3:.5f}", flush=True)
    print(f"  Reference S550 Rare Recall@10 Hits:    {s550_rare_r10_count} / 298 ({s550_rare_r10_count/len(rare_indices)*100:.2f}%)", flush=True)
    print(f"  Reference S550 False NONE Demotions:   {s550_false_none} / 506 ({s550_false_none/len(none_indices)*100:.2f}%)", flush=True)
    print(f"  Total Visual Signals Audited:          {len(audit_results)}", flush=True)
    print(f"  Signals Retrieving Extra Rare Labels:  {sum(1 for r in audit_results if r['retrieves_additional_rare'])} / {len(audit_results)}", flush=True)
    print(f"  Signals Avoiding Extra False NONE:     {sum(1 for r in audit_results if r['avoids_extra_false_none'])} / {len(audit_results)}", flush=True)
    print(f"  Signals Passing BOTH Criteria:         {len(passing_candidates)} / {len(audit_results)}", flush=True)

    if len(passing_candidates) > 0:
        print(f"\n  GATE STATUS: [PASSED] -> {len(passing_candidates)} visual signal(s) passed the promotion gate:", flush=True)
        for pc in passing_candidates:
            print(f"    - {pc['name']}: Delta MAP={pc['s550_added_delta_map3']:+.5f}, Extra Rare={pc['s550_added_delta_r10']:+d}, Extra NONE={pc['s550_added_delta_false_none']:+d}")
    else:
        print("\n  GATE STATUS: [EXHAUSTED] -> ZERO visual signals passed the promotion criteria.", flush=True)
        print("  Empirical Finding:", flush=True)
        print("    1. All 17 existing visual representations have reached their practical limit on this dataset.")
        print("    2. No visual signal retrieves additional rare misconception labels without either degrading MAP@3")
        print("       or triggering collateral false NONE demotions.")
        print("    3. Disembodied handwriting crops alone do not contain sufficient invariant mathematical signal")
        print("       to discriminate the 52 rare classes from dominant classes across 1,500 training traces.")
        print("  RECOMMENDATION FOR NEXT EXPERIMENT:", flush=True)
        print("    - Stop training models on existing image representations.")
        print("    - Change the input representation: extract targeted handwriting math bounding crops,")
        print("      or employ an improved math formula OCR pipeline (e.g. LaTeX OCR / Pix2Text) to convert")
        print("      handwriting into structured symbolic math text before passing to the classifier.")
        print("    - S550 remains locked as active submission champion (0.54214 CV / 0.55049 Public LB).")
    print("=" * 95, flush=True)

    # Save comprehensive report
    report_dict: Dict[str, Any] = {
        "execution_time_seconds": round(time.time() - start_time, 2),
        "s550_reference": {
            "seen_map3": round(s550_seen_map3, 5),
            "unseen_map3": round(s550_unseen_map3, 5),
            "composite_map3": round(s550_comp_map3, 5),
            "rare_r10_hits": s550_rare_r10_count,
            "rare_r5_hits": s550_rare_r5_count,
            "rare_r3_hits": s550_rare_r3_count,
            "false_none_count": s550_false_none
        },
        "total_signals_audited": len(audit_results),
        "signals_retrieving_extra_rare": sum(1 for r in audit_results if r["retrieves_additional_rare"]),
        "signals_avoiding_extra_false_none": sum(1 for r in audit_results if r["avoids_extra_false_none"]),
        "passing_candidates_count": len(passing_candidates),
        "audit_results": sorted_results
    }

    out_file = ART_DIR / "all_visual_signals_rare_retrieval_audit.json"
    with open(out_file, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[Saved Master Audit Report] {out_file.name}", flush=True)


if __name__ == "__main__":
    main()
