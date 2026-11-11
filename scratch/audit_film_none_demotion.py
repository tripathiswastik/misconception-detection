"""Audit ConvNeXt-448 Question FiLM False NONE Demotion and Test Conservative NONE Protection.

Objectives:
1. Re-run S550 reference and S550 + FiLM fusion (at w=0.04 and w=0.05).
2. Isolate the EXACT one trace among 506 true NONE rows where S550 was correct (Rank 1 NONE),
   but S550 + FiLM demoted NONE away from Rank 1.
3. Inspect that trace's details: fold, question_id, topic, grade, S550 probs, FiLM probs, blended probs.
4. Isolate the exact traces where FiLM improved rare-class Recall@10 (from 218 to 220).
5. Evaluate whether a conservative, fold-safe NONE protection rule can prevent this 1 false demotion
   while strictly retaining:
   - Composite MAP@3 > 0.54214
   - Rare Recall@10 >= 220
   - False NONE <= 104 (zero excess false demotions)
6. If the complete held-out gate passes, report candidate; if not, definitively close the existing-visual-signal path.
"""

from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, cast
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRATCH_DIR = PROJECT_ROOT / "scratch"
if str(SCRATCH_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_DIR))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.metrics.map3 import calculate_map3
from audit_rare_class_s550_oof_error_breakdown import S550AuditEngine, sharp


def run_audit() -> None:
    print("=" * 100, flush=True)
    print("AUDIT: FiLM CANDIDATE FALSE NONE DEMOTION & CONSERVATIVE PROTECTION TEST", flush=True)
    print("=" * 100, flush=True)

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

    # Canonical splits
    with open(DATA_DIR / "splits.json") as f:
        splits = json.load(f)["folds"]

    seen_mask = np.zeros(n_samples, dtype=bool)
    for fold in splits:
        tr_qids = set(train_df.iloc[fold["train_indices"]]["question_id"].astype(str))
        for i in fold["val_indices"]:
            if str(train_df.iloc[i]["question_id"]) in tr_qids:
                seen_mask[i] = True

    # 1. Generate Canonical S550 Baseline
    print("\n[Step 1] Generating S550 Baseline Reference...", flush=True)
    engine = S550AuditEngine()
    oof_p_s550 = np.zeros((n_samples, n_classes), dtype=np.float64)
    q_dists_per_fold: Dict[int, Dict[str, Counter]] = {}

    for fold_idx, fold in enumerate(splits, start=1):
        tr_idx = np.array(fold["train_indices"], dtype=np.int64)
        val_idx = np.array(fold["val_indices"], dtype=np.int64)
        p_val, q_dist = engine.generate_s550_probs(tr_idx, val_idx)
        oof_p_s550[val_idx] = p_val
        q_dists_per_fold[fold_idx] = q_dist

    preds_s550_all = [[] for _ in range(n_samples)]
    for fold_idx, fold in enumerate(splits, start=1):
        val_idx = np.array(fold["val_indices"], dtype=np.int64)
        preds_val = engine.rank_and_gate(oof_p_s550[val_idx], val_idx, q_dists_per_fold[fold_idx])
        for i, idx in enumerate(val_idx):
            preds_s550_all[idx] = preds_val[i]

    s550_seen_map3 = calculate_map3(
        [y_raw[i] for i in range(n_samples) if seen_mask[i]],
        [preds_s550_all[i] for i in range(n_samples) if seen_mask[i]]
    )
    s550_unseen_map3 = calculate_map3(
        [y_raw[i] for i in range(n_samples) if not seen_mask[i]],
        [preds_s550_all[i] for i in range(n_samples) if not seen_mask[i]]
    )
    s550_comp_map3 = 0.957 * s550_seen_map3 + 0.043 * s550_unseen_map3
    s550_rare_r10_count = sum(1 for i in rare_indices if y_raw[i] in [classes[c] for c in np.argsort(-oof_p_s550[i])[:10]])
    s550_rare_r3_count = sum(1 for i in rare_indices if y_raw[i] in preds_s550_all[i])
    s550_false_none = sum(1 for i in none_indices if preds_s550_all[i][0] != "NONE")

    print(f"  S550 Seen MAP@3: {s550_seen_map3:.5f} | Unseen MAP@3: {s550_unseen_map3:.5f} | Comp: {s550_comp_map3:.5f}")
    print(f"  S550 Rare R@10: {s550_rare_r10_count} / {len(rare_indices)} | Rare R@3: {s550_rare_r3_count} | False NONE Demotions: {s550_false_none} / {len(none_indices)}")

    # 2. Load FiLM Matrix
    film_path = ART_DIR / "oof_convnext_448_question_fusion.npy"
    if not film_path.exists():
        raise FileNotFoundError(f"Missing {film_path}")
    p_film = np.load(film_path).astype(np.float64)
    p_film = np.maximum(p_film, 1e-12)
    p_film = p_film / p_film.sum(axis=1, keepdims=True)

    # 3. Blend at w=0.04 and w=0.05 to analyze behavior
    for w in [0.04, 0.05]:
        print(f"\n" + "-" * 80)
        print(f"[Step 2] Analyzing Blend with FiLM at w={w:.2f}")
        print("-" * 80)
        p_blend = (1.0 - w) * oof_p_s550 + w * p_film
        p_blend = p_blend / p_blend.sum(axis=1, keepdims=True)

        preds_blend_all = [[] for _ in range(n_samples)]
        for fold_idx, fold in enumerate(splits, start=1):
            val_idx = np.array(fold["val_indices"], dtype=np.int64)
            preds_val = engine.rank_and_gate(p_blend[val_idx], val_idx, q_dists_per_fold[fold_idx])
            for i, idx in enumerate(val_idx):
                preds_blend_all[idx] = preds_val[i]

        bl_seen = calculate_map3(
            [y_raw[i] for i in range(n_samples) if seen_mask[i]],
            [preds_blend_all[i] for i in range(n_samples) if seen_mask[i]]
        )
        bl_unseen = calculate_map3(
            [y_raw[i] for i in range(n_samples) if not seen_mask[i]],
            [preds_blend_all[i] for i in range(n_samples) if not seen_mask[i]]
        )
        bl_comp = 0.957 * bl_seen + 0.043 * bl_unseen
        bl_rare_r10 = sum(1 for i in rare_indices if y_raw[i] in [classes[c] for c in np.argsort(-p_blend[i])[:10]])
        bl_false_none = sum(1 for i in none_indices if preds_blend_all[i][0] != "NONE")

        print(f"  w={w:.2f} Seen MAP@3: {bl_seen:.5f} | Unseen MAP@3: {bl_unseen:.5f} | Comp: {bl_comp:.5f} (Delta: {bl_comp - s550_comp_map3:+.5f})")
        print(f"  w={w:.2f} Rare R@10: {bl_rare_r10} (Delta: {bl_rare_r10 - s550_rare_r10_count:+d}) | False NONE: {bl_false_none} (Delta: {bl_false_none - s550_false_none:+d})")

        # Find the EXACT false demotion trace(s)
        new_false_none_idx = [i for i in none_indices if preds_s550_all[i][0] == "NONE" and preds_blend_all[i][0] != "NONE"]
        recovered_none_idx = [i for i in none_indices if preds_s550_all[i][0] != "NONE" and preds_blend_all[i][0] == "NONE"]

        print(f"  New False NONE Demotions count: {len(new_false_none_idx)} | Recovered NONE count: {len(recovered_none_idx)}")
        for idx in new_false_none_idx:
            row = train_df.iloc[idx]
            fold_found = [f_i for f_i, f in enumerate(splits, 1) if idx in f["val_indices"]][0]
            print(f"\n  >>> ISOLATED NEW FALSE NONE DEMOTION (Row {idx}):")
            print(f"      Fold: {fold_found} | Seen Question: {seen_mask[idx]} | Question ID: {row['question_id']}")
            print(f"      Topic: {row.get('topic_name', 'N/A')} | Grade: {row.get('grade', 'N/A')}")
            print(f"      True Label: {y_raw[idx]}")
            print(f"      S550 Top 3: {preds_s550_all[idx]} (Prob NONE: {oof_p_s550[idx, none_idx]:.5f})")
            print(f"      FiLM Top 3: {[classes[c] for c in np.argsort(-p_film[idx])[:3]]} (Prob NONE: {p_film[idx, none_idx]:.5f})")
            print(f"      Blend Top 3: {preds_blend_all[idx]} (Prob NONE: {p_blend[idx, none_idx]:.5f})")
            # Probabilities of the conflicting class
            promoted_class = preds_blend_all[idx][0]
            prom_idx = class_to_idx[promoted_class]
            print(f"      Promoted Class: {promoted_class}")
            print(f"        S550 prob {promoted_class}: {oof_p_s550[idx, prom_idx]:.5f} vs NONE: {oof_p_s550[idx, none_idx]:.5f}")
            print(f"        FiLM prob {promoted_class}: {p_film[idx, prom_idx]:.5f} vs NONE: {p_film[idx, none_idx]:.5f}")
            print(f"        Blend prob {promoted_class}: {p_blend[idx, prom_idx]:.5f} vs NONE: {p_blend[idx, none_idx]:.5f}")

        # Find the rare label gains
        rare_gained = [i for i in rare_indices if (y_raw[i] not in [classes[c] for c in np.argsort(-oof_p_s550[i])[:10]]) and (y_raw[i] in [classes[c] for c in np.argsort(-p_blend[i])[:10]])]
        rare_lost = [i for i in rare_indices if (y_raw[i] in [classes[c] for c in np.argsort(-oof_p_s550[i])[:10]]) and (y_raw[i] not in [classes[c] for c in np.argsort(-p_blend[i])[:10]])]
        print(f"\n  Rare Recall@10 Transitions:")
        print(f"    Gained Traces ({len(rare_gained)}): {rare_gained}")
        for idx in rare_gained:
            f_i = [f_i for f_i, f in enumerate(splits, 1) if idx in f["val_indices"]][0]
            r_c = y_raw[idx]
            r_ci = class_to_idx[r_c]
            s550_rank = list(np.argsort(-oof_p_s550[idx])).index(r_ci) + 1
            blend_rank = list(np.argsort(-p_blend[idx])).index(r_ci) + 1
            print(f"      Row {idx} (Fold {f_i}, Q={train_df.iloc[idx]['question_id']}): Class {r_c} rank {s550_rank} -> {blend_rank}")
        print(f"    Lost Traces ({len(rare_lost)}): {rare_lost}")

    # Now let's explore fold-safe conservative NONE protection rules on w=0.04
    print("\n" + "=" * 100)
    print("[Step 3] Systematic Fold-Safe Testing of Conservative NONE Protection Rules (w=0.04)")
    print("=" * 100)

    # Let's inspect the baseline S550 rank_and_gate logic to understand how NONE gating is applied!
    # In S550, rank_and_gate does:
    # 1. p_sharp = sharp(p, 0.82)
    # 2. q_dist prior discounting
    # 3. precision-gated NONE check: if p(NONE) >= tau (0.78) or margin > eps...
    # Let's test a fold-safe conservative guard.

    w = 0.04
    p_blend = (1.0 - w) * oof_p_s550 + w * p_film
    p_blend = p_blend / p_blend.sum(axis=1, keepdims=True)

    # Candidate Protection Strategies:
    # Rule 1: S550-Anchor Protection.
    # If in S550, NONE was Rank 1 AND S550's P_s550(NONE) - P_s550(top_other) >= delta_thresh,
    # does FiLM's slight displacement get suppressed?
    # We test delta_thresh from 0.00 to 0.10.
    print("\nStrategy A: S550 Rank 1 Anchor (Preserve NONE at Rank 1 if S550 predicted NONE at Rank 1 and margin >= delta)")
    for delta_thresh in [0.00, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]:
        preds_prot = [[] for _ in range(n_samples)]
        for fold_idx, fold in enumerate(splits, start=1):
            val_idx = np.array(fold["val_indices"], dtype=np.int64)
            preds_val = engine.rank_and_gate(p_blend[val_idx], val_idx, q_dists_per_fold[fold_idx])
            # Apply protection rule on val_idx
            for local_i, idx in enumerate(val_idx):
                pred_list = list(preds_val[local_i])
                s550_r1 = preds_s550_all[idx][0]
                if s550_r1 == "NONE" and pred_list[0] != "NONE":
                    # Check margin in S550
                    p_s = oof_p_s550[idx]
                    other_max = max(p_s[c] for c in range(n_classes) if c != none_idx)
                    margin = p_s[none_idx] - other_max
                    if margin >= delta_thresh:
                        # Re-anchor NONE to Rank 1
                        pred_list.remove("NONE") if "NONE" in pred_list else None
                        pred_list = ["NONE"] + pred_list[:2]
                preds_prot[idx] = pred_list

        sc_seen = calculate_map3(
            [y_raw[i] for i in range(n_samples) if seen_mask[i]],
            [preds_prot[i] for i in range(n_samples) if seen_mask[i]]
        )
        sc_unseen = calculate_map3(
            [y_raw[i] for i in range(n_samples) if not seen_mask[i]],
            [preds_prot[i] for i in range(n_samples) if not seen_mask[i]]
        )
        sc_comp = 0.957 * sc_seen + 0.043 * sc_unseen
        fn_count = sum(1 for i in none_indices if preds_prot[i][0] != "NONE")
        r10_count = sum(1 for i in rare_indices if y_raw[i] in [classes[c] for c in np.argsort(-p_blend[i])[:10]])
        r3_count = sum(1 for i in rare_indices if y_raw[i] in preds_prot[i])

        print(f"  delta={delta_thresh:.3f} | Comp MAP@3: {sc_comp:.5f} (Delta: {sc_comp - s550_comp_map3:+.5f}) | "
              f"Rare R@10: {r10_count} | Rare R@3: {r3_count} | False NONE: {fn_count} (Delta: {fn_count - s550_false_none:+d})")

    # Strategy B: Cross-Validation Fold-Isolated Tuning (True 5-fold CV)
    # Calibrate delta_thresh per fold on training indices ONLY, then evaluate on val_idx
    print("\nStrategy B: Strict 5-Fold Fold-Isolated Rule Selection")
    # For each fold, we test candidate deltas on train_indices to minimize (false_none_increase) while maximizing MAP@3
    # Then evaluate strictly on val_indices.
    fold_val_preds = [[] for _ in range(n_samples)]
    for fold_idx, fold in enumerate(splits, start=1):
        tr_idx = np.array(fold["train_indices"], dtype=np.int64)
        val_idx = np.array(fold["val_indices"], dtype=np.int64)

        # On training indices: test deltas to find the smallest delta that prevents false NONE demotions without hurting score
        best_delta = 0.0
        best_val = -1e9
        for d in [0.00, 0.005, 0.01, 0.02, 0.03, 0.05]:
            p_tr_blend = p_blend[tr_idx]
            p_tr_s550 = oof_p_s550[tr_idx]
            preds_tr = engine.rank_and_gate(p_tr_blend, tr_idx, q_dists_per_fold[fold_idx])
            # Apply candidate rule on train
            tr_fn = 0
            for l_i, g_i in enumerate(tr_idx):
                p_list = list(preds_tr[l_i])
                s550_r1 = preds_s550_all[g_i][0]
                if s550_r1 == "NONE" and p_list[0] != "NONE":
                    p_s = p_tr_s550[l_i]
                    other_max = max(p_s[c] for c in range(n_classes) if c != none_idx)
                    margin = p_s[none_idx] - other_max
                    if margin >= d:
                        p_list = ["NONE"] + [x for x in p_list if x != "NONE"][:2]
                if y_raw[g_i] == "NONE" and p_list[0] != "NONE":
                    tr_fn += 1
            # We want minimum false NONE demotions on train
            score_crit = -tr_fn
            if score_crit > best_val:
                best_val = score_crit
                best_delta = d

        print(f"  Fold {fold_idx}: Selected best_delta={best_delta:.3f} on train split")
        # Apply to validation split
        preds_v = engine.rank_and_gate(p_blend[val_idx], val_idx, q_dists_per_fold[fold_idx])
        for l_i, g_i in enumerate(val_idx):
            p_list = list(preds_v[l_i])
            s550_r1 = preds_s550_all[g_i][0]
            if s550_r1 == "NONE" and p_list[0] != "NONE":
                p_s = oof_p_s550[g_i]
                other_max = max(p_s[c] for c in range(n_classes) if c != none_idx)
                margin = p_s[none_idx] - other_max
                if margin >= best_delta:
                    p_list = ["NONE"] + [x for x in p_list if x != "NONE"][:2]
            fold_val_preds[g_i] = p_list

    cv_seen = calculate_map3(
        [y_raw[i] for i in range(n_samples) if seen_mask[i]],
        [fold_val_preds[i] for i in range(n_samples) if seen_mask[i]]
    )
    cv_unseen = calculate_map3(
        [y_raw[i] for i in range(n_samples) if not seen_mask[i]],
        [fold_val_preds[i] for i in range(n_samples) if not seen_mask[i]]
    )
    cv_comp = 0.957 * cv_seen + 0.043 * cv_unseen
    cv_fn = sum(1 for i in none_indices if fold_val_preds[i][0] != "NONE")
    cv_rare_r10 = sum(1 for i in rare_indices if y_raw[i] in [classes[c] for c in np.argsort(-p_blend[i])[:10]])
    cv_rare_r3 = sum(1 for i in rare_indices if y_raw[i] in fold_val_preds[i])

    print(f"\nFold-Isolated Protected Blend Results:")
    print(f"  Seen MAP@3:      {cv_seen:.5f} (S550: {s550_seen_map3:.5f}, Delta: {cv_seen - s550_seen_map3:+.5f})")
    print(f"  Unseen MAP@3:    {cv_unseen:.5f} (S550: {s550_unseen_map3:.5f}, Delta: {cv_unseen - s550_unseen_map3:+.5f})")
    print(f"  Composite MAP@3: {cv_comp:.5f} (S550: {s550_comp_map3:.5f}, Delta: {cv_comp - s550_comp_map3:+.5f})")
    print(f"  Rare Recall@10:  {cv_rare_r10} / {len(rare_indices)} (Delta: {cv_rare_r10 - s550_rare_r10_count:+d})")
    print(f"  Rare Top-3 Hits: {cv_rare_r3} / {len(rare_indices)} (S550: {s550_rare_r3_count})")
    print(f"  False NONE Demotions: {cv_fn} / {len(none_indices)} (S550: {s550_false_none}, Delta: {cv_fn - s550_false_none:+d})")

    # Check Gate
    passes_gate = (cv_comp > s550_comp_map3) and (cv_rare_r10 >= s550_rare_r10_count + 2) and (cv_fn <= s550_false_none)
    print(f"\nComplete Held-Out Promotion Gate Passed? {'YES - CANDIDATE PASSES' if passes_gate else 'NO - GATE FAILED'}")

    # Save detailed JSON report
    report = {
        "s550_reference": {
            "composite_map3": s550_comp_map3,
            "seen_map3": s550_seen_map3,
            "unseen_map3": s550_unseen_map3,
            "rare_r10": s550_rare_r10_count,
            "rare_r3": s550_rare_r3_count,
            "false_none": s550_false_none,
        },
        "isolated_false_none_trace": {
            "row_idx": new_false_none_idx[0] if new_false_none_idx else None,
            "details": {
                "question_id": str(train_df.iloc[new_false_none_idx[0]]["question_id"]) if new_false_none_idx else "",
                "seen_status": bool(seen_mask[new_false_none_idx[0]]) if new_false_none_idx else False,
                "s550_p_none": float(oof_p_s550[new_false_none_idx[0], none_idx]) if new_false_none_idx else 0.0,
                "film_p_none": float(p_film[new_false_none_idx[0], none_idx]) if new_false_none_idx else 0.0,
                "promoted_class": str(preds_blend_all[new_false_none_idx[0]][0]) if new_false_none_idx else "",
            } if new_false_none_idx else {},
        },
        "rare_gained_traces": rare_gained,
        "fold_isolated_protected_results": {
            "composite_map3": cv_comp,
            "seen_map3": cv_seen,
            "unseen_map3": cv_unseen,
            "delta_map3": cv_comp - s550_comp_map3,
            "rare_r10": cv_rare_r10,
            "delta_rare_r10": cv_rare_r10 - s550_rare_r10_count,
            "false_none": cv_fn,
            "delta_false_none": cv_fn - s550_false_none,
            "passes_gate": passes_gate,
        }
    }
    with open(ART_DIR / "film_none_protection_audit_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {ART_DIR / 'film_none_protection_audit_report.json'}")


if __name__ == "__main__":
    run_audit()
