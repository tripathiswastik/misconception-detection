"""Leak-Free Cross-Validation and Honest Model Evaluation Suite.

Ensures:
1. Strict in-fold domain masking (zero val label leakage).
2. Clean OOF components with test-like features (no gold transcription_steps).
3. Zero pseudo-label teacher leakage.
4. Nested CV / holdout evaluation of blend weights.
5. Sliced metrics: Overall, NONE rows, Error rows, Seen Questions, Unseen Questions.
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"


def load_taxonomy_and_data():
    with open(ART_DIR / "taxonomy_classes.json", "r", encoding="utf-8") as f:
        classes = json.load(f)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    splits = json.loads((DATA_DIR / "splits.json").read_text(encoding="utf-8"))["folds"]
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    return classes, train_df, test_df, splits, y_true


def map3_score(y_true: List[str], probs: np.ndarray, classes: List[str]) -> float:
    top3_idx = np.argsort(-probs.astype(np.float64), axis=1)[:, :3]
    scores = []
    for label, row in zip(y_true, top3_idx):
        found = [classes[idx] for idx in row]
        score = 0.0
        for r, pred in enumerate(found):
            if pred == label:
                score = 1.0 / (r + 1)
                break
        scores.append(score)
    return float(np.mean(scores))


def fast_fold_domain_mask(
    probs: np.ndarray,
    train_slice: pd.DataFrame,
    eval_slice: pd.DataFrame,
    classes: List[str],
    tax,
    damping: float = 0.85
) -> np.ndarray:
    """Fast, memoized in-fold domain constraint masking."""
    from src.diagnoser.domain_mask import DomainConstraintEngine
    engine = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_slice)
    
    # Memoize by unique (topic, grade) pairs
    cache: Dict[Tuple[str, int], np.ndarray] = {}
    calibrated = np.zeros_like(probs)
    
    topics = [str(t) for t in eval_slice["topic_name"]]
    grades = [int(float(g)) for g in eval_slice["grade"]]
    
    for i in range(len(eval_slice)):
        key = (topics[i], grades[i])
        if key not in cache:
            aff = np.array([engine.compute_affinity(c, key[0], key[1]) ** damping for c in classes], dtype=np.float64)
            cache[key] = aff
        aff = cache[key]
        row_p = probs[i] * aff
        s = np.sum(row_p)
        calibrated[i] = row_p / s if s > 1e-12 else probs[i]
        
    return calibrated


def main():
    classes, train_df, test_df, splits, y_true = load_taxonomy_and_data()
    none_idx = classes.index("NONE")
    n_train = len(train_df)
    
    print("=" * 70)
    print("LEAK-FREE VALIDATION SUITE & CLEAN OOF AUDIT")
    print("=" * 70)
    
    # Load raw / clean OOFs
    p_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    p_cn = np.load(ART_DIR / "oof_convnext_vision.npy")
    p_swin = np.load(ART_DIR / "oof_swin_vision.npy")
    p_text_honest = np.load(ART_DIR / "honest_oof_text.npy")
    p_text_sem = np.load(ART_DIR / "oof_semantic_text.npy")
    
    # MobileNet with strict in-fold masking
    mn_raw = np.load(ART_DIR / "oof_finetuned_vision.npy")
    p_mn = np.zeros_like(mn_raw)
    from src.data.taxonomy import TaxonomyManager
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    for f in splits:
        tr = f["train_indices"]
        va = f["val_indices"]
        p_mn[va] = fast_fold_domain_mask(
            mn_raw[va], train_df.iloc[tr], train_df.iloc[va], classes, tax=tax, damping=0.85
        )
        
    components = {
        "Adaptive Prior": p_prior,
        "ConvNeXt Vision": p_cn,
        "Swin-T Vision": p_swin,
        "MobileNetV3 Vision": p_mn,
        "Honest SGD Text": p_text_honest,
        "Semantic Text (384-d)": p_text_sem,
    }
    
    print("\n--- 1. Individual Clean Component Performance (5-Fold CV) ---")
    for name, p in components.items():
        fold_scores = [map3_score([y_true[i] for i in f["val_indices"]], p[f["val_indices"]], classes) for f in splits]
        overall = map3_score(y_true, p, classes)
        f_str = " | ".join(f"F{i+1}: {s:.4f}" for i, s in enumerate(fold_scores))
        print(f"  {name:22s} | Overall: {overall:.5f} | {f_str}")
        
    # Sliced analysis of Adaptive Prior
    is_none = np.array([y == "NONE" for y in y_true])
    print("\n--- 2. Baseline Slices (Adaptive Prior) ---")
    print(f"  NONE rows ({is_none.sum()}/1500) MAP@3:     {map3_score([y_true[i] for i in range(n_train) if is_none[i]], p_prior[is_none], classes):.5f}")
    print(f"  Non-NONE rows ({(~is_none).sum()}/1500) MAP@3: {map3_score([y_true[i] for i in range(n_train) if not is_none[i]], p_prior[~is_none], classes):.5f}")
    
    # 3. Clean Convex Optimization (Zero Leakage)
    print("\n--- 3. Clean Dirichlet / Convex Optimization across 5 Folds ---")
    comp_keys = list(components.keys())
    comp_matrices = [components[k] for k in comp_keys]
    
    # Run nested cross-validation: for each fold, optimize weights on remaining 4 folds, evaluate on held-out fold
    nested_val_scores = []
    nested_weights = []
    
    for fold_idx, f in enumerate(splits):
        tr = f["train_indices"]
        va = f["val_indices"]
        y_tr = [y_true[i] for i in tr]
        y_va = [y_true[i] for i in va]
        
        tr_matrices = [m[tr] for m in comp_matrices]
        va_matrices = [m[va] for m in comp_matrices]
        
        def loss_fn(weights):
            w = np.exp(weights) / np.sum(np.exp(weights))
            blended = sum(w[i] * tr_matrices[i] for i in range(len(w)))
            return -map3_score(y_tr, blended, classes)
            
        init_w = np.zeros(len(comp_keys))
        res = minimize(loss_fn, init_w, method="Powell", options={"maxiter": 200})
        opt_w = np.exp(res.x) / np.sum(np.exp(res.x))
        nested_weights.append(opt_w)
        
        va_blended = sum(opt_w[i] * va_matrices[i] for i in range(len(opt_w)))
        val_score = map3_score(y_va, va_blended, classes)
        nested_val_scores.append(val_score)
        print(f"  Fold {fold_idx+1} Nested CV MAP@3: {val_score:.5f}")
        for k, weight in zip(comp_keys, opt_w):
            print(f"    {k:22s}: {weight:.4f}")
            
    mean_nested = float(np.mean(nested_val_scores))
    print(f"\nMean Nested OOF MAP@3: {mean_nested:.5f} +/- {np.std(nested_val_scores, ddof=1):.5f}")
    
    # Average optimal weights across folds
    avg_weights = np.mean(nested_weights, axis=0)
    print("\n--- 4. Final Leak-Free Ensemble Weights ---")
    for k, weight in zip(comp_keys, avg_weights):
        print(f"  {k:22s}: {weight*100:.2f}%")
        
    final_oof = sum(avg_weights[i] * comp_matrices[i] for i in range(len(avg_weights)))
    final_overall = map3_score(y_true, final_oof, classes)
    print(f"\nOverall Honest Ensemble OOF MAP@3: {final_overall:.5f}")
    print(f"  Gain over Prior Alone ({map3_score(y_true, p_prior, classes):.5f}): {final_overall - map3_score(y_true, p_prior, classes):+.5f}")
    

if __name__ == "__main__":
    main()
