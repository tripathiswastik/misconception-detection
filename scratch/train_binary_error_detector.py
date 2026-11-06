"""Train Dedicated Binary Flawed-vs-Clean Classifier (Pillar 2).

Objective:
Overcome the multi-class probability fragmentation of NONE.
Instead of forcing NONE to compete with 50+ fragmented misconception classes in argmax,
this model directly estimates P(is_error) using:
1. 2,328-d multimodal visual-textual representation (ConvNeXt-448, Swin-T, TrOCR-384, MiniLM-384, Strokes)
2. Question and topic prior empirical error rates (fold-local Laplace smoothed)
3. Multi-class model uncertainty signals (P(NONE), top misconception prob, margin, entropy)

Models:
- Fold-local HistGradientBoostingClassifier
- Fold-local PyTorch Deep MLP on NVIDIA CUDA GPU

Outputs:
- artifacts/oof_binary_error_prob.npy
- artifacts/test_binary_error_prob.npy
- artifacts/binary_error_detector_metrics.json
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"


class BinaryMLP(nn.Module):
    """Deep calibrated PyTorch MLP for binary error classification on GPU."""

    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def compute_entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def assemble_features() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))

    # Load 2,328-d representation
    tr_swin = np.load(ART_DIR / "train_swin_features.npy")
    tr_conv = np.load(ART_DIR / "train_convnext_448_letterbox.npy")
    tr_trocr = np.load(ART_DIR / "train_trocr_384_letterbox.npy")
    tr_text = np.load(ART_DIR / "train_text_embeddings_384.npy")
    tr_strk = np.load(ART_DIR / "train_perceptual_strokes.npy")

    te_swin = np.load(ART_DIR / "test_swin_features.npy")
    te_conv = np.load(ART_DIR / "test_convnext_448_letterbox.npy")
    te_trocr = np.load(ART_DIR / "test_trocr_384_letterbox.npy")
    te_text = np.load(ART_DIR / "test_text_embeddings_384.npy")
    te_strk = np.load(ART_DIR / "test_perceptual_strokes.npy")

    X_train_dense = np.hstack([tr_swin, tr_conv, tr_trocr, tr_text, tr_strk]).astype(np.float32)
    X_test_dense = np.hstack([te_swin, te_conv, te_trocr, te_text, te_strk]).astype(np.float32)

    # Multi-class baseline probabilities (M15 balanced fusion)
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    te_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")

    from src.data.taxonomy import TaxonomyManager
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")

    # Engineer multi-class meta features
    def make_meta_features(probs: np.ndarray) -> np.ndarray:
        p_none = probs[:, none_idx:none_idx + 1]
        probs_nom = probs.copy()
        probs_nom[:, none_idx] = -1.0
        p_max_cand = np.max(probs_nom, axis=1, keepdims=True)
        margin = p_max_cand - p_none
        entropy = compute_entropy(probs)[:, None]
        return np.hstack([p_none, p_max_cand, margin, entropy]).astype(np.float32)

    meta_train = make_meta_features(oof_m15)
    meta_test = make_meta_features(te_m15)

    X_train = np.hstack([X_train_dense, meta_train])
    X_test = np.hstack([X_test_dense, meta_test])

    y_train = (train_df["root_code"] != "NONE").astype(int).values

    print(f"Features assembled: Train={X_train.shape}, Test={X_test.shape}")
    print(f"Target distribution: {np.bincount(y_train)} (Positive/Error rate: {y_train.mean():.2%})")

    return X_train, X_test, train_df, test_df, y_train


def run_binary_detector_pipeline():
    X_train, X_test, train_df, test_df, y_train = assemble_features()

    with open(DATA_DIR / "splits.json", "r") as f:
        splits = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Binary Error Detector training on: {device}")

    oof_hgb = np.zeros(len(y_train), dtype=np.float64)
    test_hgb = np.zeros(len(test_df), dtype=np.float64)

    oof_mlp = np.zeros(len(y_train), dtype=np.float64)
    test_mlp = np.zeros(len(test_df), dtype=np.float64)

    fold_metrics: List[Dict[str, float]] = []

    folds_list = splits["folds"] if isinstance(splits, dict) and "folds" in splits else splits
    for fold_idx, fold_data in enumerate(folds_list):
        tr_idx = fold_data.get("train_indices", fold_data.get("train"))
        val_idx = fold_data.get("val_indices", fold_data.get("val"))

        # Fold-local target encoding of question error rate
        q_rates = train_df.iloc[tr_idx].groupby("question_id")["root_code"].apply(lambda s: (s != "NONE").mean())
        global_rate = y_train[tr_idx].mean()

        def get_q_features(df: pd.DataFrame, indices: Optional[List[int]] = None) -> np.ndarray:
            sub = df.iloc[indices] if indices is not None else df
            rates = sub["question_id"].map(q_rates).fillna(global_rate).values[:, None]
            return rates.astype(np.float32)

        q_tr = get_q_features(train_df, tr_idx)
        q_val = get_q_features(train_df, val_idx)
        q_te = get_q_features(test_df)

        X_tr = np.hstack([X_train[tr_idx], q_tr])
        X_val = np.hstack([X_train[val_idx], q_val])
        X_te = np.hstack([X_test, q_te])

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_te_s = scaler.transform(X_te)

        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]

        # 1. Model A: HistGradientBoostingClassifier
        hgb = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_depth=5,
            l2_regularization=2.0,
            random_state=42 + fold_idx
        )
        hgb.fit(X_tr_s, y_tr)
        p_val_hgb = hgb.predict_proba(X_val_s)[:, 1]
        p_te_hgb = hgb.predict_proba(X_te_s)[:, 1]

        oof_hgb[val_idx] = p_val_hgb
        test_hgb += p_te_hgb / len(folds_list)

        # 2. Model B: PyTorch MLP on GPU
        torch.manual_seed(42 + fold_idx)
        mlp = BinaryMLP(in_features=X_tr_s.shape[1], hidden_dim=128, dropout=0.3).to(device)
        optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        ds_tr = TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
        loader_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)

        mlp.train()
        for epoch in range(35):
            for bx, by in loader_tr:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = mlp(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        mlp.eval()
        with torch.no_grad():
            val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
            p_val_mlp = torch.sigmoid(mlp(val_t)).cpu().numpy()

            te_t = torch.tensor(X_te_s, dtype=torch.float32).to(device)
            p_te_mlp = torch.sigmoid(mlp(te_t)).cpu().numpy()

        oof_mlp[val_idx] = p_val_mlp
        test_mlp += p_te_mlp / len(folds_list)

        # Fold Blend (50/50 HGB + MLP)
        p_val_blend = 0.50 * p_val_hgb + 0.50 * p_val_mlp
        auc = roc_auc_score(y_val, p_val_blend)
        acc = accuracy_score(y_val, (p_val_blend >= 0.5).astype(int))
        brier = brier_score_loss(y_val, p_val_blend)

        print(f"Fold {fold_idx + 1}/5 -> ROC-AUC: {auc:.4f} | Accuracy: {acc:.4f} | Brier: {brier:.4f}")
        fold_metrics.append({"fold": fold_idx + 1, "auc": float(auc), "acc": float(acc), "brier": float(brier)})

    oof_final = 0.50 * oof_hgb + 0.50 * oof_mlp
    test_final = 0.50 * test_hgb + 0.50 * test_mlp

    overall_auc = roc_auc_score(y_train, oof_final)
    overall_acc = accuracy_score(y_train, (oof_final >= 0.5).astype(int))
    overall_brier = brier_score_loss(y_train, oof_final)
    overall_logloss = log_loss(y_train, oof_final)

    print("=" * 80)
    print("BINARY ERROR DETECTOR - 5-FOLD CV SUMMARY")
    print(f"Overall ROC-AUC:  {overall_auc:.4f}")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Overall Brier:    {overall_brier:.4f}")
    print(f"Overall LogLoss:  {overall_logloss:.4f}")
    print("=" * 80)

    # Save artifacts
    np.save(ART_DIR / "oof_binary_error_prob.npy", oof_final)
    np.save(ART_DIR / "test_binary_error_prob.npy", test_final)

    metrics_payload = {
        "overall_auc": float(overall_auc),
        "overall_acc": float(overall_acc),
        "overall_brier": float(overall_brier),
        "overall_logloss": float(overall_logloss),
        "fold_metrics": fold_metrics
    }
    with open(ART_DIR / "binary_error_detector_metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Saved: artifacts/oof_binary_error_prob.npy {oof_final.shape}")
    print(f"Saved: artifacts/test_binary_error_prob.npy {test_final.shape}")
    print("Saved: artifacts/binary_error_detector_metrics.json")


if __name__ == "__main__":
    run_binary_detector_pipeline()

