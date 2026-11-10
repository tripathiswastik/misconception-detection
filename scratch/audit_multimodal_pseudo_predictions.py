"""Audit Genuine Multimodal Predictions on Pseudo Features (2328-d).

Instead of relying on static question-prior copies, train a multimodal linear probe
on the 1,500 real labeled traces and evaluate what it predicts on the 1,465 unlabelled student traces.
"""
from collections import Counter
from pathlib import Path
import sys
from typing import cast
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager

def main():
    print("=" * 80)
    print("AUDITING GENUINE MULTIMODAL FEATURE PREDICTIONS ON 1,465 PSEUDO TRACES")
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
    print(f"Train 2328-d features: {X_train_2328.shape}")

    # 2. Load 2328-d Pseudo Features
    pseudo_df = cast(pd.DataFrame, pd.read_csv(ART_DIR / "pseudo_1465_metadata.csv"))
    pseudo_raw = np.load(ART_DIR / "pseudo_1465_features_2328.npy")
    # Features in pseudo_1465 are [cn(768), sw(768), tr(384), tx(384), st(24)]
    X_pseudo_2328 = np.hstack([
        normalize(pseudo_raw[:, :768]),
        normalize(pseudo_raw[:, 768:1536]),
        normalize(pseudo_raw[:, 1536:1920]),
        normalize(pseudo_raw[:, 1920:2304]),
        normalize(pseudo_raw[:, 2304:2328])
    ])
    print(f"Pseudo 2328-d features: {X_pseudo_2328.shape}")

    # 3. Train Multimodal Classifier on 1,500 labeled traces
    clf = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=42)
    clf.fit(X_train_2328, y_indices)

    # 4. Predict on Pseudo Features
    probs = clf.predict_proba(X_pseudo_2328)
    # Map back to full 76 classes
    full_probs = np.zeros((len(pseudo_df), len(classes)), dtype=np.float64)
    for col_idx, cls_idx in enumerate(clf.classes_):
        full_probs[:, cls_idx] = probs[:, col_idx]

    top1_idx = np.argmax(full_probs, axis=1)
    top1_probs = np.max(full_probs, axis=1)
    top1_codes = [classes[i] for i in top1_idx]

    sorted_p = np.sort(full_probs, axis=1)[:, ::-1]
    margins = sorted_p[:, 0] - sorted_p[:, 1]

    # Subgroup of rare classes (<= 10 in train)
    counts = Counter(y_raw)
    rare_10 = set(c for c, cnt in counts.items() if cnt <= 10)

    print(f"\nTop 10 predicted classes on pseudo traces by genuine multimodal classifier:")
    for c, cnt in Counter(top1_codes).most_common(10):
        is_rare = c in rare_10
        print(f"  {c:10s}: {cnt:4d} traces (Rare? {str(is_rare):5s})")

    # Filter high-confidence genuine rare-class predictions
    rare_candidates = []
    for i in range(len(pseudo_df)):
        c = top1_codes[i]
        p = top1_probs[i]
        m = margins[i]
        if c in rare_10 and p >= 0.40 and m >= 0.15:
            rare_candidates.append({
                "trace_id": pseudo_df.iloc[i]["trace_id"],
                "question_id": pseudo_df.iloc[i]["question_id"],
                "topic_name": pseudo_df.iloc[i]["topic_name"],
                "predicted_code": c,
                "confidence": p,
                "margin": m,
                "feature_row_idx": i
            })

    rare_cand_df = pd.DataFrame(rare_candidates)
    print(f"\nGenuine Rare-Class Predictions with Conf >= 0.40 and Margin >= 0.15: {len(rare_cand_df)}")
    if len(rare_cand_df) > 0:
        print(rare_cand_df["predicted_code"].value_counts())
        print("\nSample candidates:")
        print(rare_cand_df.head(10))

    print("=" * 80)

if __name__ == "__main__":
    main()

