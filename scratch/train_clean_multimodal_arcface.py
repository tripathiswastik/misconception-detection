"""Train Clean, Leak-Free 2,328-d Multimodal ArcFace Model across 5 Canonical Folds.

100% Leak-Free:
- Trained strictly on 1,500 Gold Train Traces (zero full-dataset pseudo-label leakage).
- Evaluated on canonical 5-fold cross-validation (splits.json).
- In-fold StandardScaler and in-fold Domain Constraint Masking.
- Direct MAP@3 Soft-Ranking Margin Loss + Class-Weighted Focal Loss.
- Exports oof_clean_arcface_2328.npy and test_clean_arcface_2328.npy.
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.domain_mask import DomainConstraintEngine
from src.evaluation.cross_validator import ModelCrossValidator

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"


class GoldDataset(Dataset):
    def __init__(self, features: np.ndarray, hard_labels: np.ndarray, binary_targets: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.hard_labels = torch.from_numpy(hard_labels).long()
        self.binary_targets = torch.from_numpy(binary_targets).float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[index], self.hard_labels[index], self.binary_targets[index]


class ArcFaceClassifier(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale: float = 30.0, margin: float = 0.30):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1))
        if labels is None:
            return cosine * self.scale
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.scale


class CleanDualHeadArcFaceNet(nn.Module):
    def __init__(self, in_dim: int = 2328, num_classes: int = 76, embed_dim: int = 384):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(768, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.arcface_head = ArcFaceClassifier(embed_dim, num_classes, scale=30.0, margin=0.30)
        self.binary_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.backbone(x)
        tax_logits = self.arcface_head(emb, labels)
        bin_logits = self.binary_head(emb).squeeze(-1)
        return tax_logits, bin_logits


class CompositeRankingLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, margin: float = 1.0, top_k: int = 5, lambda_rank: float = 0.20):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.margin = margin
        self.top_k = top_k
        self.lambda_rank = lambda_rank

    def forward(self, tax_logits: torch.Tensor, bin_logits: torch.Tensor, hard_labels: torch.Tensor, bin_targets: torch.Tensor) -> torch.Tensor:
        batch_sz = tax_logits.shape[0]
        # 1. Focal Loss
        ce = F.cross_entropy(tax_logits, hard_labels, reduction="none")
        pt = torch.exp(-ce)
        at = self.alpha[hard_labels]
        focal_loss = (at * ((1.0 - pt) ** self.gamma) * ce).mean()

        # 2. Soft-Ranking Proxy Margin Loss
        target_scores = tax_logits[torch.arange(batch_sz, device=tax_logits.device), hard_labels].unsqueeze(1)
        mask = torch.ones_like(tax_logits, dtype=torch.bool)
        mask[torch.arange(batch_sz, device=tax_logits.device), hard_labels] = False
        neg_scores = tax_logits[mask].view(batch_sz, -1)
        top_negatives, _ = torch.topk(neg_scores, min(self.top_k, neg_scores.shape[1]), dim=-1)
        rank_loss = F.relu(self.margin - (target_scores - top_negatives)).mean()

        # 3. Binary Head Loss
        bin_loss = F.binary_cross_entropy_with_logits(bin_logits, bin_targets)

        return focal_loss + self.lambda_rank * rank_loss + 0.30 * bin_loss


def train_clean_arcface():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    none_idx = class_to_idx["NONE"]
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    y_true_str = train_df["root_code"].fillna("NONE").astype(str).tolist()
    y_true_int = np.array([class_to_idx[r] for r in y_true_str], dtype=np.int64)
    y_bin = np.array([1.0 if y == "NONE" else 0.0 for y in y_true_str], dtype=np.float32)

    with open(DATA_DIR / "splits.json") as f:
        splits = json.load(f)["folds"]

    # Load 2,328-d features
    swin_tr = np.load(ART_DIR / "train_swin_features.npy")
    conv448_tr = np.load(ART_DIR / "train_convnext_448_letterbox.npy")
    text_tr = np.load(ART_DIR / "train_text_embeddings_384.npy")
    trocr_tr = np.load(ART_DIR / "train_trocr_features.npy")
    strk_tr = np.load(ART_DIR / "train_perceptual_strokes.npy")

    swin_te = np.load(ART_DIR / "test_swin_features.npy")
    conv448_te = np.load(ART_DIR / "test_convnext_448_letterbox.npy")
    text_te = np.load(ART_DIR / "test_text_embeddings_384.npy")
    trocr_te = np.load(ART_DIR / "test_trocr_features.npy")
    strk_te = np.load(ART_DIR / "test_perceptual_strokes.npy")

    X_train_orig = np.hstack([swin_tr, conv448_tr, text_tr, trocr_tr, strk_tr]).astype(np.float32)
    X_test_orig = np.hstack([swin_te, conv448_te, text_te, trocr_te, strk_te]).astype(np.float32)

    print(f"Features: Train={X_train_orig.shape}, Test={X_test_orig.shape}", flush=True)

    oof_probs = np.zeros((len(train_df), n_classes), dtype=np.float32)
    test_probs = np.zeros((len(test_df), n_classes), dtype=np.float32)

    epochs = 25
    batch_size = 64
    lr = 1e-3

    for fold_idx, fold in enumerate(splits, start=1):
        tr_idx = np.array(fold["train_indices"], dtype=np.int64)
        val_idx = np.array(fold["val_indices"], dtype=np.int64)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_orig[tr_idx])
        X_val = scaler.transform(X_train_orig[val_idx])
        X_te = scaler.transform(X_test_orig)

        # In-fold alpha weights
        counts = np.zeros(n_classes, dtype=np.int64)
        for v in y_true_int[tr_idx]:
            counts[v] += 1
        raw_a = len(tr_idx) / (n_classes * np.maximum(counts, 1.0))
        alpha = np.clip(raw_a, 0.20, 5.0).astype(np.float32)
        alpha_t = torch.from_numpy(alpha / alpha.mean()).float().to(device)

        train_ds = GoldDataset(X_tr, y_true_int[tr_idx], y_bin[tr_idx])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        model = CleanDualHeadArcFaceNet(in_dim=2328, num_classes=n_classes, embed_dim=384).to(device)
        criterion = CompositeRankingLoss(alpha=alpha_t, margin=1.0, top_k=5, lambda_rank=0.20)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        model.train()
        for ep in range(epochs):
            for batch_x, batch_y, batch_bin in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_bin = batch_bin.to(device)

                optimizer.zero_grad()
                tax_logits, bin_logits = model(batch_x, batch_y)
                loss = criterion(tax_logits, bin_logits, batch_y, batch_bin)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            scheduler.step()

        # Inference on Val and Test
        model.eval()
        with torch.no_grad():
            v_x = torch.from_numpy(X_val).float().to(device)
            v_logits, _ = model(v_x, labels=None)
            val_p = F.softmax(v_logits, dim=-1).cpu().numpy()

            te_x = torch.from_numpy(X_te).float().to(device)
            te_logits, _ = model(te_x, labels=None)
            test_p = F.softmax(te_logits, dim=-1).cpu().numpy()

        # Fast in-fold domain masking
        masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df.iloc[tr_idx])
        v_topics = [str(t) for t in train_df.iloc[val_idx]["topic_name"].values]
        v_grades = [int(float(g)) for g in train_df.iloc[val_idx]["grade"].values]
        val_calibrated = masker.apply_mask_batch(val_p, classes, v_topics, v_grades, damping=0.05)
        oof_probs[val_idx] = val_calibrated

        te_topics = [str(t) for t in test_df["topic_name"].values]
        te_grades = [int(float(g)) for g in test_df["grade"].values]
        te_calibrated = masker.apply_mask_batch(test_p, classes, te_topics, te_grades, damping=0.05)
        test_probs += te_calibrated / len(splits)

        print(f"  Fold {fold_idx}/5 Completed", flush=True)

    # Save clean artifacts
    np.save(ART_DIR / "oof_clean_arcface_2328.npy", oof_probs)
    np.save(ART_DIR / "test_clean_arcface_2328.npy", test_probs)
    print(f"\nSaved artifacts/oof_clean_arcface_2328.npy and artifacts/test_clean_arcface_2328.npy")

    # Evaluate with ModelCrossValidator
    cv = ModelCrossValidator(taxonomy=tax, use_canonical_splits=True)
    res = cv.evaluate_probabilities(oof_probs, train_df, model_name="Clean_Multimodal_ArcFace_2328", classes=classes)
    print(res.diagnostic_report())


if __name__ == "__main__":
    train_clean_arcface()

