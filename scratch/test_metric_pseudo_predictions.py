"""Test Metric-Based (Cosine / Nearest Centroid) Predictions on Pseudo Features.

Removes the prior probability penalty so rare classes with distinct visual features
can be identified without being crushed by the majority-class intercept.
"""
from collections import Counter
from pathlib import Path
import sys
from typing import cast
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager

def main():
    print("=" * 80)
    print("METRIC-BASED (COSINE CENTROID) PREDICTIONS ON 1,465 PSEUDO TRACES")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_raw = train_df["root_code"].fillna("NONE").astype(str).values
    y_indices = np.array([class_to_idx[c] for c in y_raw], dtype=np.int64)

    # 1. Build 2328-d Train Features
    cn_tr = np.load(ART_DIR / "train_convnext_features.npy")
    sw_tr = np.load(ART_DIR / "train_swin_features.npy")
    tr_tr = np.load(ART_DIR / "train_trocr_features.npy")
    tx_tr = np.load(ART_DIR / "train_text_embeddings_384.npy")
    st_tr = np.load(ART_DIR / "train_perceptual_strokes.npy")

    X_train_2328 = np.hstack([
        normalize(cn_tr), normalize(sw_tr), normalize(tr_tr), normalize(tx_tr), normalize(st_tr)
    ])
    X_train_norm = normalize(X_train_2328, norm="l2")

    # 2. Compute Class Centroids in 2328-d Metric Space
    centroids = np.zeros((len(classes), X_train_norm.shape[1]), dtype=np.float32)
    counts = Counter(y_indices)
    for c_idx in range(len(classes)):
        mask = (y_indices == c_idx)
        if mask.sum() > 0:
            c_mean = X_train_norm[mask].mean(axis=0)
            centroids[c_idx] = c_mean / (np.linalg.norm(c_mean) + 1e-12)

    # 3. Load 2328-d Pseudo Features
    pseudo_df = cast(pd.DataFrame, pd.read_csv(ART_DIR / "pseudo_1465_metadata.csv"))
    pseudo_raw = np.load(ART_DIR / "pseudo_1465_features_2328.npy")
    X_pseudo_2328 = np.hstack([
        normalize(pseudo_raw[:, :768]),
        normalize(pseudo_raw[:, 768:1536]),
        normalize(pseudo_raw[:, 1536:1920]),
        normalize(pseudo_raw[:, 1920:2304]),
        normalize(pseudo_raw[:, 2304:2328])
    ])
    X_pseudo_norm = normalize(X_pseudo_2328, norm="l2")

    # 4. Compute Cosine Similarity Matrix (1465, 76)
    sims = np.dot(X_pseudo_norm, centroids.T)

    top1_sim_idx = np.argmax(sims, axis=1)
    top1_sims = np.max(sims, axis=1)
    top1_codes = [classes[i] for i in top1_sim_idx]

    sorted_sims = np.sort(sims, axis=1)[:, ::-1]
    margins = sorted_sims[:, 0] - sorted_sims[:, 1]
    print(f"\nCosine Margins: min={margins.min():.5f}, median={np.median(margins):.5f}, p90={np.percentile(margins, 90):.5f}, max={margins.max():.5f}")
    print(f"Top 10 margins: {[round(float(x), 5) for x in np.sort(margins)[-10:]]}")

    # Rare classes
    rare_10 = set(c for c, cnt in Counter(y_raw).items() if cnt <= 10)

    print("\nTop 15 predicted classes on pseudo traces by metric cosine similarity:")
    for c, cnt in Counter(top1_codes).most_common(15):
        is_rare = c in rare_10
        print(f"  {c:10s}: {cnt:4d} traces (Rare? {str(is_rare):5s})")

    # Filter candidates where top-1 is a rare class with positive cosine margin
    rare_matches = []
    for threshold in [0.005, 0.010, 0.020, 0.030]:
        count = sum(1 for i in range(len(pseudo_df)) if top1_codes[i] in rare_10 and margins[i] >= threshold)
        print(f"Rare-Class Traces with Margin >= {threshold:.3f}: {count}")

    for i in range(len(pseudo_df)):
        c = top1_codes[i]
        sim = top1_sims[i]
        margin = margins[i]
        if c in rare_10 and margin >= 0.015:
            rare_matches.append({
                "trace_id": pseudo_df.iloc[i]["trace_id"],
                "question_id": pseudo_df.iloc[i]["question_id"],
                "topic_name": pseudo_df.iloc[i]["topic_name"],
                "predicted_code": c,
                "cosine_sim": sim,
                "margin": margin,
                "idx": i
            })

    rare_df = pd.DataFrame(rare_matches)
    print(f"\nTotal Selected Rare-Class Traces (Margin >= 0.015): {len(rare_df)}")
    if len(rare_df) > 0:
        print("\nBreakdown by rare class:")
        print(rare_df["predicted_code"].value_counts())
        print("\nSample top candidates:")
        print(rare_df.sort_values("margin", ascending=False).head(10))

    print("=" * 80)

if __name__ == "__main__":
    main()
