"""Systematic Prior Smoothing Ablation Study (Experiments P1 - P4).

Tests:
- P1: Global shrinkage sweep (m_q in {4, 6, 8, 10, 12, 14, 16})
- P2: Adaptive sample-size shrinkage m_q(n_q)
- P3: Semantic question-similarity backoff for unseen questions
- P4: Hierarchical topic-grade smoothing sweep (m_tg in {1, 3, 5, 8, 10})

Strictly in-fold: Zero validation leakage.
Outputs:
- artifacts/prior_smoothing_ablation_report.json
- artifacts/best_adaptive_prior_oof.npy
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, PATHS
from src.data.dataset import MathMisconceptionData
from src.data.splits import FoldSplitter
from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
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


def run_prior_ablation():
    print("\n" + "=" * 70)
    print("PHASE 1: QUESTION-PRIOR SMOOTHING ABLATION (P1 - P4)")
    print("=" * 70 + "\n")

    tax = TaxonomyManager(PATHS.taxonomy_csv)
    classes = tax.all_codes(include_none=True)
    n_classes = len(classes)

    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    n_train = len(train_df)
    y_true = train_df["root_code"].astype(str).tolist()

    splitter = FoldSplitter(n_splits=CONFIG.n_splits, random_state=CONFIG.random_state)
    folds = splitter.load_splits(PATHS.splits_cache)

    artifacts_dir = PROJECT_ROOT / "artifacts"
    embeds_path = artifacts_dir / "train_text_embeddings_384.npy"
    semantic_embeds = np.load(embeds_path) if embeds_path.exists() else None
    if semantic_embeds is not None:
        print(f"[Ablation] Loaded 384-d semantic embeddings: {semantic_embeds.shape}")

    # Identify Seen vs Unseen questions in 5-fold OOF
    oof_seen_mask = np.zeros(n_train, dtype=bool)
    for tr_idx, val_idx in folds:
        train_qids = set(train_df.iloc[tr_idx]["question_id"].values)
        for idx in val_idx:
            if train_df.iloc[idx]["question_id"] in train_qids:
                oof_seen_mask[idx] = True

    seen_y = [y_true[i] for i in range(n_train) if oof_seen_mask[i]]
    unseen_y = [y_true[i] for i in range(n_train) if not oof_seen_mask[i]]

    def evaluate_engine_config(
        m_q_base: float = 10.0,
        m_tg: float = 5.0,
        adaptive_nq: bool = False,
        adaptive_power: float = 0.5,
        use_semantic_backoff: bool = False,
        semantic_top_k: int = 3,
        semantic_threshold: float = 0.70,
        semantic_weight: float = 0.50
    ) -> Tuple[float, float, float, np.ndarray]:
        oof_probs = np.zeros((n_train, n_classes), dtype=np.float32)

        for tr_idx, val_idx in folds:
            fold_tr = train_df.iloc[tr_idx].reset_index(drop=True)
            fold_val = train_df.iloc[val_idx].reset_index(drop=True)

            engine = AdaptiveQuestionPriorEngine(
                classes=classes,
                m_q_base=m_q_base,
                m_tg=m_tg,
                adaptive_nq=adaptive_nq,
                adaptive_power=adaptive_power,
                use_semantic_backoff=use_semantic_backoff,
                semantic_embeddings=semantic_embeds,
                semantic_top_k=semantic_top_k,
                semantic_threshold=semantic_threshold,
                semantic_weight=semantic_weight
            )
            engine.fit(fold_tr, fold_train_indices=tr_idx)

            val_embeds = semantic_embeds[val_idx] if semantic_embeds is not None else None
            p_val = engine.predict_proba(fold_val, eval_indices=val_idx, eval_embeddings=val_embeds)
            oof_probs[val_idx] = p_val

        overall_map3 = eval_map3(oof_probs, y_true, classes)
        seen_map3 = eval_map3(oof_probs[oof_seen_mask], seen_y, classes)
        unseen_map3 = eval_map3(oof_probs[~oof_seen_mask], unseen_y, classes)
        return overall_map3, seen_map3, unseen_map3, oof_probs

    # 1. Baseline Reference (M10: m_q=10.0, m_tg=5.0)
    base_map3, base_seen, base_unseen, base_oof = evaluate_engine_config(m_q_base=10.0, m_tg=5.0)
    base_scores = eval_per_sample(base_oof, y_true, classes)
    print(f"[Reference M10 Prior] Overall: {base_map3:.4f} | Seen: {base_seen:.4f} | Unseen: {base_unseen:.4f}")

    results = []

    # 2. Experiment P1: Shrinkage Sweep
    print("\n--- Running P1: Global Shrinkage Sweep (m_q) ---")
    mq_values = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    for mq in mq_values:
        ov, sn, un, oof_mat = evaluate_engine_config(m_q_base=mq, m_tg=5.0)
        delta = ov - base_map3
        print(f"  P1 (m_q={mq:4.1f}): Overall={ov:.4f} (Seen={sn:.4f}, Unseen={un:.4f}) [Delta={delta:+.4f}]")
        results.append({
            "experiment": "P1_shrinkage_sweep",
            "name": f"P1_mq_{mq}",
            "params": {"m_q": mq, "m_tg": 5.0},
            "overall_map3": ov,
            "seen_map3": sn,
            "unseen_map3": un,
            "delta_vs_baseline": delta
        })

    # 3. Experiment P2: Adaptive Sample-Size Shrinkage m_q(n_q)
    print("\n--- Running P2: Adaptive Sample-Size Shrinkage m_q(n_q) ---")
    p2_configs = [
        (8.0, 0.3), (10.0, 0.3), (12.0, 0.3),
        (8.0, 0.5), (10.0, 0.5), (12.0, 0.5),
        (10.0, 0.7), (14.0, 0.5)
    ]
    for m0, power in p2_configs:
        ov, sn, un, oof_mat = evaluate_engine_config(m_q_base=m0, m_tg=5.0, adaptive_nq=True, adaptive_power=power)
        delta = ov - base_map3
        print(f"  P2 (m0={m0:4.1f}, pow={power:.2f}): Overall={ov:.4f} (Seen={sn:.4f}, Unseen={un:.4f}) [Delta={delta:+.4f}]")
        results.append({
            "experiment": "P2_adaptive_nq",
            "name": f"P2_m0_{m0}_pow_{power}",
            "params": {"m_q_base": m0, "power": power, "m_tg": 5.0},
            "overall_map3": ov,
            "seen_map3": sn,
            "unseen_map3": un,
            "delta_vs_baseline": delta
        })

    # 4. Experiment P3: Semantic Question-Text Similarity Backoff
    print("\n--- Running P3: Semantic Question Backoff for Unseen Questions ---")
    p3_configs = [
        (3, 0.65, 0.30),
        (3, 0.70, 0.50),
        (3, 0.75, 0.50),
        (5, 0.70, 0.50),
        (3, 0.70, 0.70)
    ]
    for k, thresh, w in p3_configs:
        ov, sn, un, oof_mat = evaluate_engine_config(
            m_q_base=10.0, m_tg=5.0,
            use_semantic_backoff=True, semantic_top_k=k, semantic_threshold=thresh, semantic_weight=w
        )
        delta = ov - base_map3
        print(f"  P3 (k={k}, thresh={thresh:.2f}, w={w:.2f}): Overall={ov:.4f} (Seen={sn:.4f}, Unseen={un:.4f}) [Delta={delta:+.4f}]")
        results.append({
            "experiment": "P3_semantic_backoff",
            "name": f"P3_k_{k}_th_{thresh}_w_{w}",
            "params": {"top_k": k, "threshold": thresh, "weight": w, "m_q": 10.0, "m_tg": 5.0},
            "overall_map3": ov,
            "seen_map3": sn,
            "unseen_map3": un,
            "delta_vs_baseline": delta
        })

    # 5. Experiment P4: Hierarchical Topic-Grade Smoothing Sweep
    print("\n--- Running P4: Topic-Grade Smoothing Sweep (m_tg) ---")
    mtg_values = [1.0, 3.0, 5.0, 8.0, 10.0, 15.0]
    for mtg in mtg_values:
        ov, sn, un, oof_mat = evaluate_engine_config(m_q_base=10.0, m_tg=mtg)
        delta = ov - base_map3
        print(f"  P4 (m_tg={mtg:4.1f}): Overall={ov:.4f} (Seen={sn:.4f}, Unseen={un:.4f}) [Delta={delta:+.4f}]")
        results.append({
            "experiment": "P4_topic_grade_sweep",
            "name": f"P4_mtg_{mtg}",
            "params": {"m_q": 10.0, "m_tg": mtg},
            "overall_map3": ov,
            "seen_map3": sn,
            "unseen_map3": un,
            "delta_vs_baseline": delta
        })

    # Find the single best configuration across P1 - P4
    best_exp = max(results, key=lambda x: x["overall_map3"])
    print("\n" + "=" * 70)
    print(f"BEST PRIOR CONFIGURATION: {best_exp['name']}")
    print(f"  Overall OOF MAP@3: {best_exp['overall_map3']:.4f} (Delta: {best_exp['delta_vs_baseline']:+.4f})")
    print(f"  Seen Qs MAP@3:    {best_exp['seen_map3']:.4f}")
    print(f"  Unseen Qs MAP@3:  {best_exp['unseen_map3']:.4f}")
    print(f"  Parameters:       {best_exp['params']}")
    print("=" * 70)

    # Re-generate and save the best prior OOF matrix
    p = best_exp["params"]
    _, _, _, best_oof_matrix = evaluate_engine_config(
        m_q_base=p.get("m_q", p.get("m_q_base", 10.0)),
        m_tg=p.get("m_tg", 5.0),
        adaptive_nq=best_exp["experiment"] == "P2_adaptive_nq",
        adaptive_power=p.get("power", 0.5),
        use_semantic_backoff=best_exp["experiment"] == "P3_semantic_backoff",
        semantic_top_k=p.get("top_k", 3),
        semantic_threshold=p.get("threshold", 0.70),
        semantic_weight=p.get("weight", 0.50)
    )

    np.save(artifacts_dir / "best_adaptive_prior_oof.npy", best_oof_matrix)
    print(f"[Ablation] Saved optimal prior OOF to: {artifacts_dir / 'best_adaptive_prior_oof.npy'}")

    # Serialize complete ablation report
    report = {
        "baseline_m10_prior": {
            "overall_map3": base_map3,
            "seen_map3": base_seen,
            "unseen_map3": base_unseen,
            "m_q": 10.0,
            "m_tg": 5.0
        },
        "best_configuration": best_exp,
        "all_experiments": results
    }
    with open(artifacts_dir / "prior_smoothing_ablation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Ablation] Report written to: {artifacts_dir / 'prior_smoothing_ablation_report.json'}")

    return report


if __name__ == "__main__":
    run_prior_ablation()
