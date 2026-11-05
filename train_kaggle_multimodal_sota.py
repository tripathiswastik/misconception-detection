"""End-to-End Multimodal Deep ArcFace Training Pipeline for Kaggle GPU & Local.

Pillar 4 SOTA:
1. 100% Self-Contained: Uses extracted 1,944-d multimodal features (Swin-T 768 + ConvNeXt 768 + MiniLM 384 + Strokes 24).
   Requires ZERO external image datasets in /kaggle/input/.
2. Deep ArcFace Head: Enhances fine-grained angular margin separation between subtle misconception classes.
3. 5-Fold Stratified CV: Uses exact benchmark splits from data/splits.json.
4. Linear Ensembling: Blends with Active Champion M15 (Public LB 0.54149).
5. Output Deliverables:
   - oof_kaggle_sota.npy (1500, 76)
   - test_kaggle_sota.npy (3498, 76)
   - blend_m15_kaggle_submission.csv (3498 rows, columns: ['trace_id', 'root_code'])
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Path Detection (Auto-detects Kaggle vs Local Environment)
# ---------------------------------------------------------------------------
def resolve_environment_paths() -> Tuple[Path, Path, Path]:
    # 1. Check /kaggle/working/data (extracted zip in Kaggle working dir)
    if Path("/kaggle/working/data/train.csv").exists():
        print("[Environment] Found extracted project files in /kaggle/working")
        return Path("/kaggle/working/data"), Path("/kaggle/working/artifacts"), Path("/kaggle/working")

    # 2. Check current working directory
    cwd = Path.cwd()
    if (cwd / "data" / "train.csv").exists():
        print(f"[Environment] Found project in current working directory: {cwd}")
        out_p = Path("/kaggle/working") if Path("/kaggle").exists() else cwd / "submissions"
        return cwd / "data", cwd / "artifacts", out_p

    # 3. Check relative to script
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "data" / "train.csv").exists():
        print(f"[Environment] Found project at script dir: {script_dir}")
        out_p = Path("/kaggle/working") if Path("/kaggle").exists() else script_dir / "submissions"
        return script_dir / "data", script_dir / "artifacts", out_p

    # 4. Fallback search
    for search_root in [Path("/kaggle/working"), Path("/kaggle/input"), Path(".")]:
        for root, _, files in os.walk(search_root):
            if "train.csv" in files:
                dp = Path(root)
                ap = dp.parent / "artifacts" if (dp.parent / "artifacts").exists() else dp
                op = Path("/kaggle/working") if Path("/kaggle").exists() else Path("submissions")
                print(f"[Environment] Found train.csv at: {dp}")
                return dp, ap, op

    return Path("data"), Path("artifacts"), Path("submissions")


DATA_DIR, ART_DIR, OUTPUT_DIR = resolve_environment_paths()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Data Dir:   {DATA_DIR}")
print(f"Artifacts:  {ART_DIR}")
print(f"Output Dir: {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Metric Calculation
# ---------------------------------------------------------------------------
def compute_map3(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, np.ndarray]:
    top3 = np.argsort(-y_probs.astype(np.float64), axis=1)[:, :3]
    scores = []
    for true_label, preds in zip(y_true, top3):
        if true_label == preds[0]:
            scores.append(1.0)
        elif true_label == preds[1]:
            scores.append(0.5)
        elif true_label == preds[2]:
            scores.append(1.0 / 3.0)
        else:
            scores.append(0.0)
    arr = np.array(scores, dtype=np.float64)
    return float(np.mean(arr)), arr


# ---------------------------------------------------------------------------
# Deep ArcFace Angular Margin Head & Network
# ---------------------------------------------------------------------------
class ArcMarginProduct(nn.Module):
    """ArcFace margin cosine head for angular misconception separation."""

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.20):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = float(np.cos(m))
        self.sin_m = float(np.sin(m))
        self.th = float(np.cos(np.pi - m))
        self.mm = float(np.sin(np.pi - m) * m)

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if label is None:
            return cosine * self.s

        sine = torch.sqrt(1.0 - torch.clamp(cosine**2, 0.0, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class MultimodalDeepArcFace(nn.Module):
    """Multimodal Deep Network over fused Vision-Language-Perceptual representations."""

    def __init__(self, in_features: int = 1944, hidden1: int = 768, hidden2: int = 384, n_classes: int = 76):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head = ArcMarginProduct(hidden2, n_classes, s=30.0, m=0.20)

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.encoder(x)
        return self.head(feat, label)


# ---------------------------------------------------------------------------
# Kaggle Pipeline Execution
# ---------------------------------------------------------------------------
def run_kaggle_pipeline():
    print("=" * 75)
    print("[Kaggle SOTA] Multimodal Deep ArcFace Training Pipeline")
    print("=" * 75)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Labels & Splits
    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))

    with open(ART_DIR / "taxonomy_classes.json") as f:
        tax_classes: List[str] = json.load(f)
    class_to_idx = {c: i for i, c in enumerate(tax_classes)}
    n_classes = len(tax_classes)
    none_idx = class_to_idx["NONE"]

    y_train = np.array([class_to_idx[r] for r in train_df["root_code"]], dtype=np.int64)
    is_none = (y_train == none_idx)

    with open(DATA_DIR / "splits.json") as f:
        splits_data = json.load(f)
    folds = splits_data["folds"]

    # 2. Load Self-Contained Multimodal Features
    print("\nLoading pre-extracted multimodal feature representations...")
    swin_tr = np.load(ART_DIR / "train_swin_features.npy")
    conv_tr = np.load(ART_DIR / "train_convnext_features.npy")
    text_tr = np.load(ART_DIR / "train_text_embeddings_384.npy")
    strk_tr = np.load(ART_DIR / "train_perceptual_strokes.npy")

    swin_te = np.load(ART_DIR / "test_swin_features.npy")
    conv_te = np.load(ART_DIR / "test_convnext_features.npy")
    text_te = np.load(ART_DIR / "test_text_embeddings_384.npy")
    strk_te = np.load(ART_DIR / "test_perceptual_strokes.npy")

    X_train_raw = np.hstack([swin_tr, conv_tr, text_tr, strk_tr]).astype(np.float32)
    X_test_raw = np.hstack([swin_te, conv_te, text_te, strk_te]).astype(np.float32)

    in_dim = X_train_raw.shape[1]
    print(f"Multimodal Feature Vector Dim: {in_dim} (Swin 768 + ConvNeXt 768 + MiniLM 384 + Strokes 24)")
    print(f"Train Shape: {X_train_raw.shape}, Test Shape: {X_test_raw.shape}")

    # Load M15 Champion Baseline for Ensembling
    m15_oof = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
    m15_map3, m15_scores = compute_map3(y_train, m15_oof)
    print(f"\nM15 Champion Baseline: MAP@3 = {m15_map3:.5f} (NONE: {np.mean(m15_scores[is_none]):.4f}, Non-NONE: {np.mean(m15_scores[~is_none]):.4f})")

    # 3. Stratified 5-Fold Cross-Validation
    oof_probs = np.zeros((len(train_df), n_classes), dtype=np.float32)
    test_fold_preds = np.zeros((len(folds), len(test_df), n_classes), dtype=np.float32)

    epochs = 25
    batch_size = 64
    lr = 1e-3

    print(f"\nTraining Deep ArcFace across {len(folds)} folds ({epochs} epochs/fold, batch_size={batch_size}, lr={lr})...")
    start_time = time.time()

    for fold_num, fold in enumerate(folds, 1):
        tr_idx = np.array(fold["train_indices"])
        val_idx = np.array(fold["val_indices"])

        # Feature normalization per fold
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_raw[tr_idx])
        X_val = scaler.transform(X_train_raw[val_idx])
        X_te = scaler.transform(X_test_raw)

        tr_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_train[tr_idx]))
        val_ds = TensorDataset(torch.from_numpy(X_val).float())
        te_ds = TensorDataset(torch.from_numpy(X_te).float())

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
        te_loader = DataLoader(te_ds, batch_size=128, shuffle=False)

        model = MultimodalDeepArcFace(in_features=in_dim, hidden1=768, hidden2=384, n_classes=n_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        for epoch in range(epochs):
            model.train()
            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx, by)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Validation inference
        model.eval()
        val_preds = []
        with torch.no_grad():
            for (bx,) in val_loader:
                bx = bx.to(device)
                logits = model(bx, label=None)
                probs = F.softmax(logits, dim=-1)
                val_preds.append(probs.cpu().numpy())
        fold_val_probs = np.vstack(val_preds)
        oof_probs[val_idx] = fold_val_probs

        # Test inference
        te_preds = []
        with torch.no_grad():
            for (bx,) in te_loader:
                bx = bx.to(device)
                logits = model(bx, label=None)
                probs = F.softmax(logits, dim=-1)
                te_preds.append(probs.cpu().numpy())
        test_fold_preds[fold_num - 1] = np.vstack(te_preds)

        fold_map3, fold_scores = compute_map3(y_train[val_idx], fold_val_probs)
        val_none = is_none[val_idx]
        print(f"  Fold {fold_num}/{len(folds)} MAP@3: {fold_map3:.5f} | NONE: {np.mean(fold_scores[val_none]):.4f} | Non-NONE: {np.mean(fold_scores[~val_none]):.4f}")

    elapsed = time.time() - start_time
    test_probs = np.mean(test_fold_preds, axis=0)

    # 4. Overall Model Performance
    arc_map3, arc_scores = compute_map3(y_train, oof_probs)
    print("-" * 75)
    print(f"Deep ArcFace Overall OOF MAP@3: {arc_map3:.5f} ({elapsed:.1f}s)")
    print(f"  -> NONE MAP@3:     {np.mean(arc_scores[is_none]):.4f}")
    print(f"  -> Non-NONE MAP@3: {np.mean(arc_scores[~is_none]):.4f}")

    # 5. Linear Calibration & Ensembling with M15 Champion
    print("\n" + "=" * 75)
    print("Linear Ensembling Sweep with Active Champion M15 (Public LB: 0.54149)")
    print("=" * 75)
    weights = [0.00, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    best_w = 0.00
    best_blend_score = m15_map3

    for w in weights:
        blend_oof = (1.0 - w) * m15_oof + w * oof_probs
        score, sc_arr = compute_map3(y_train, blend_oof)
        none_s = float(np.mean(sc_arr[is_none]))
        non_none_s = float(np.mean(sc_arr[~is_none]))
        star = " ★" if score > best_blend_score else ""
        print(f"  w={w:.2f} ArcFace + {1.0 - w:.2f} M15: MAP@3 = {score:.5f} | NONE = {none_s:.4f} | Non-NONE = {non_none_s:.4f}{star}")
        if score > best_blend_score:
            best_blend_score = score
            best_w = w

    print(f"\nOptimal Linear Blend Weight: w* = {best_w:.2f} (OOF MAP@3 = {best_blend_score:.5f}, Delta = {best_blend_score - m15_map3:+.5f})")

    # Compute Final Blended Test Probabilities
    if best_w > 0.0:
        final_test_probs = (1.0 - best_w) * test_m15 + best_w * test_probs
    else:
        # Default conservative blend if strictly equal
        final_test_probs = 0.90 * test_m15 + 0.10 * test_probs

    # 6. Save Artifacts & Submissions
    print("\n" + "=" * 75)
    print("Writing Final Outputs & Competition Submissions...")
    print("=" * 75)

    np.save(OUTPUT_DIR / "oof_kaggle_sota.npy", oof_probs)
    np.save(OUTPUT_DIR / "test_kaggle_sota.npy", test_probs)
    print(f"  Saved: {OUTPUT_DIR / 'oof_kaggle_sota.npy'}")
    print(f"  Saved: {OUTPUT_DIR / 'test_kaggle_sota.npy'}")

    # Build Top-3 predictions for submission
    def build_submission(probs: np.ndarray, filename: str):
        top3_indices = np.argsort(-probs.astype(np.float64), axis=1)[:, :3]
        pred_strings = []
        for row in top3_indices:
            pred_strings.append(" ".join(tax_classes[idx] for idx in row))

        sub_df = pd.DataFrame({
            "trace_id": test_df["trace_id"],
            "root_code": pred_strings,
        })
        out_path = OUTPUT_DIR / filename
        sub_df.to_csv(out_path, index=False)
        print(f"  Generated: {out_path} ({len(sub_df)} rows, columns: {list(sub_df.columns)})")
        print(f"    Sample: {sub_df.iloc[0].to_dict()}")

    build_submission(final_test_probs, "blend_m15_kaggle_submission.csv")
    build_submission(test_probs, "kaggle_arcface_submission.csv")

    print("\n Pipeline Completed Successfully!")


if __name__ == "__main__":
    run_kaggle_pipeline()
