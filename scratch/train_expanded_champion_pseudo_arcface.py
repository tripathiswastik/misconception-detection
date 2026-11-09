"""Train Expanded Multimodal ArcFace Model with 0.55049 Champion Soft Targets.

Upgrades:
- Uses pseudo_3500_champion_soft_targets_76.npy distilled from the 0.55049 Champion Ensemble.
- Evaluates with 100% leak-free honest OOF pipeline.
- Outputs:
  * artifacts/oof_arcface_pseudo_3500_v2.npy (1500, 76)
  * artifacts/test_arcface_pseudo_3500_v2.npy (3498, 76)
"""
import json
import math
from pathlib import Path
import sys
import time
from typing import List, Optional, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"


class MixedTargetDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        is_pseudo: np.ndarray,
        hard_labels: np.ndarray,
        soft_targets: np.ndarray,
        binary_targets: np.ndarray,
    ):
        self.features = torch.from_numpy(features).float()
        self.is_pseudo = torch.from_numpy(is_pseudo).bool()
        self.hard_labels = torch.from_numpy(hard_labels).long()
        self.soft_targets = torch.from_numpy(soft_targets).float()
        self.binary_targets = torch.from_numpy(binary_targets).float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[index],
            self.is_pseudo[index],
            self.hard_labels[index],
            self.soft_targets[index],
            self.binary_targets[index],
        )


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.20):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(
        self,
        input: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        is_pseudo: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine * self.s
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        if is_pseudo is not None:
            # Apply margin strictly on gold labels, keep smooth cosine on pseudo
            margin_mask = (~is_pseudo).unsqueeze(1).float()
            output = (one_hot * phi) * margin_mask + (one_hot * cosine) * (1.0 - margin_mask) + ((1.0 - one_hot) * cosine)
        else:
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class DualHeadArcFaceNet(nn.Module):
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
        self.arc_head = ArcMarginProduct(hidden2, n_classes, s=30.0, m=0.20)
        self.binary_head = nn.Linear(hidden2, 1)

    def forward(
        self,
        x: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        is_pseudo: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        tax_logits = self.arc_head(feat, label, is_pseudo)
        bin_logit = self.binary_head(feat).squeeze(-1)
        return tax_logits, bin_logit


class CoLearnCompositeLoss(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: float = 2.0,
        margin: float = 1.0,
        top_k: int = 5,
        lambda_rank: float = 0.20,
        w_pseudo: float = 0.40,
        w_binary: float = 0.40,
    ):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.margin = margin
        self.top_k = top_k
        self.lambda_rank = lambda_rank
        self.w_pseudo = w_pseudo
        self.w_binary = w_binary

    def forward(
        self,
        tax_logits: torch.Tensor,
        bin_logits: torch.Tensor,
        is_pseudo: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_targets: torch.Tensor,
        binary_targets: torch.Tensor,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=tax_logits.device)

        # 1. Gold Samples Loss: Class-Weighted Focal + Margin Ranking Proxy
        gold_mask = ~is_pseudo
        if gold_mask.any():
            g_logits = tax_logits[gold_mask]
            g_targets = hard_labels[gold_mask]

            ce_loss = F.cross_entropy(g_logits, g_targets, reduction="none")
            pt = torch.exp(-ce_loss)
            focal_weight = self.alpha[g_targets] * torch.pow(1.0 - pt, self.gamma)
            focal_loss = (focal_weight * ce_loss).mean()

            # Multi-Class Margin Ranking Proxy Loss
            s_target = g_logits.gather(1, g_targets.unsqueeze(1))
            mask_distractors = torch.ones_like(g_logits, dtype=torch.bool)
            mask_distractors.scatter_(1, g_targets.unsqueeze(1), False)
            distractor_logits = g_logits.masked_fill(~mask_distractors, -float("inf"))

            k = min(self.top_k, distractor_logits.shape[1] - 1)
            top_k_neg, _ = torch.topk(distractor_logits, k=k, dim=1)
            rank_loss = F.relu(self.margin - (s_target - top_k_neg)).mean()

            loss = loss + focal_loss + self.lambda_rank * rank_loss

        # 2. Pseudo Samples Loss: Soft-Target Cross Entropy
        pseudo_mask = is_pseudo
        if pseudo_mask.any():
            p_logits = tax_logits[pseudo_mask]
            p_targets = soft_targets[pseudo_mask]
            log_probs = F.log_softmax(p_logits, dim=-1)
            pseudo_ce = -(p_targets * log_probs).sum(dim=-1).mean()
            loss = loss + self.w_pseudo * pseudo_ce

        # 3. Auxiliary Binary Correctness Task
        bce_loss = F.binary_cross_entropy_with_logits(bin_logits, binary_targets)
        loss = loss + self.w_binary * bce_loss

        return loss


def calculate_map3(y_true: List[str], y_pred: List[List[str]]) -> float:
    scores = []
    for true_label, preds in zip(y_true, y_pred):
        if true_label == preds[0]:
            scores.append(1.0)
        elif true_label == preds[1]:
            scores.append(0.5)
        elif true_label == preds[2]:
            scores.append(1.0 / 3.0)
        else:
            scores.append(0.0)
    return float(np.mean(scores))


def main():
    print("=" * 80)
    print("TRAINING EXPANDED ARCFACE V2 WITH 0.55049 CHAMPION SOFT TARGETS")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device}", flush=True)

    with open(ART_DIR / "taxonomy_classes.json", "r", encoding="utf-8") as f:
        tax_classes: List[str] = json.load(f)
    n_classes = len(tax_classes)
    none_idx = tax_classes.index("NONE")

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    y_train_orig = train_df["root_code"].fillna("NONE").values
    y_train_int = np.array([tax_classes.index(y) for y in y_train_orig], dtype=np.int64)
    y_train_bin = np.array([1.0 if y == "NONE" else 0.0 for y in y_train_orig], dtype=np.float32)

    with open(DATA_DIR / "splits.json") as f:
        splits_data = json.load(f)
    folds = splits_data["folds"]

    class_counts = np.zeros(n_classes, dtype=np.int64)
    for val in y_train_int:
        class_counts[val] += 1
    raw_alpha = len(y_train_int) / (n_classes * np.maximum(class_counts, 1.0))
    alpha_clamped = np.clip(raw_alpha, 0.20, 5.0).astype(np.float32)
    alpha_norm = (alpha_clamped / alpha_clamped.mean()).astype(np.float32)
    alpha_tensor = torch.from_numpy(alpha_norm).float()

    print("Loading 1,944-d multimodal feature matrices...", flush=True)
    swin_tr = np.load(ART_DIR / "train_swin_features.npy")
    conv448_tr = np.load(ART_DIR / "train_convnext_448_letterbox.npy")
    text_tr = np.load(ART_DIR / "train_text_embeddings_384.npy")
    strk_tr = np.load(ART_DIR / "train_perceptual_strokes.npy")

    swin_te = np.load(ART_DIR / "test_swin_features.npy")
    conv448_te = np.load(ART_DIR / "test_convnext_448_letterbox.npy")
    text_te = np.load(ART_DIR / "test_text_embeddings_384.npy")
    strk_te = np.load(ART_DIR / "test_perceptual_strokes.npy")

    X_train_orig = np.hstack([swin_tr, conv448_tr, text_tr, strk_tr]).astype(np.float32)
    X_test = np.hstack([swin_te, conv448_te, text_te, strk_te]).astype(np.float32)
    print(f"Train Gold: {X_train_orig.shape}, Test: {X_test.shape}", flush=True)

    X_pseudo = np.load(ART_DIR / "pseudo_3500_features_1944.npy").astype(np.float32)
    # LOAD THE UPGRADED 0.55049 CHAMPION SOFT TARGETS
    soft_targets_pseudo = np.load(ART_DIR / "pseudo_3500_champion_soft_targets_76.npy").astype(np.float32)
    bin_targets_pseudo = soft_targets_pseudo[:, none_idx].astype(np.float32)
    print(f"Upgraded Champion Pseudo Targets: {soft_targets_pseudo.shape}", flush=True)

    n_orig = len(X_train_orig)
    n_pseudo = len(X_pseudo)
    print(f"Total Effective Training Set: {n_orig + n_pseudo} samples (3.33x Gold)\n", flush=True)

    soft_targets_gold = np.zeros((n_orig, n_classes), dtype=np.float32)
    for i, idx in enumerate(y_train_int):
        soft_targets_gold[i, idx] = 1.0

    epochs = 25
    batch_size = 64
    lr = 1e-3

    oof_probs = np.zeros((n_orig, n_classes), dtype=np.float32)
    test_fold_preds = np.zeros((len(folds), len(X_test), n_classes), dtype=np.float32)

    loss_fn = CoLearnCompositeLoss(
        alpha=alpha_tensor,
        gamma=2.0,
        margin=1.0,
        top_k=5,
        lambda_rank=0.20,
        w_pseudo=0.40,
        w_binary=0.40,
    ).to(device)

    t0 = time.time()
    for fold_num, fold in enumerate(folds, 1):
        tr_idx = np.array(fold["train_indices"])
        val_idx = np.array(fold["val_indices"])

        X_tr = np.vstack([X_train_orig[tr_idx], X_pseudo])
        is_pseudo_tr = np.concatenate([np.zeros(len(tr_idx), dtype=bool), np.ones(n_pseudo, dtype=bool)])
        hard_labels_tr = np.concatenate([y_train_int[tr_idx], np.zeros(n_pseudo, dtype=np.int64)])
        soft_targets_tr = np.vstack([soft_targets_gold[tr_idx], soft_targets_pseudo])
        bin_targets_tr = np.concatenate([y_train_bin[tr_idx], bin_targets_pseudo])

        X_val = X_train_orig[val_idx]

        scaler = StandardScaler()
        X_tr_norm = scaler.fit_transform(X_tr)
        X_val_norm = scaler.transform(X_val)
        X_te_norm = scaler.transform(X_test)

        train_ds = MixedTargetDataset(X_tr_norm, is_pseudo_tr, hard_labels_tr, soft_targets_tr, bin_targets_tr)
        val_ds = TensorDataset(torch.from_numpy(X_val_norm).float())
        te_ds = TensorDataset(torch.from_numpy(X_te_norm).float())

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
        te_loader = DataLoader(te_ds, batch_size=128, shuffle=False)

        model = DualHeadArcFaceNet(in_features=1944, hidden1=768, hidden2=384, n_classes=n_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        for epoch in range(epochs):
            model.train()
            for bx, bis_ps, by_hard, by_soft, by_bin in train_loader:
                bx = bx.to(device)
                bis_ps = bis_ps.to(device)
                by_hard = by_hard.to(device)
                by_soft = by_soft.to(device)
                by_bin = by_bin.to(device)

                optimizer.zero_grad()
                tax_logits, bin_logits = model(bx, by_hard, bis_ps)
                loss = loss_fn(tax_logits, bin_logits, bis_ps, by_hard, by_soft, by_bin)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Validation inference: Pure 76-Class Softmax
        model.eval()
        val_logits_list = []
        with torch.no_grad():
            for (b_vx,) in val_loader:
                b_vx = b_vx.to(device)
                v_logits, _ = model(b_vx)
                val_logits_list.append(v_logits.cpu())
        val_logits = torch.cat(val_logits_list, dim=0)
        val_probs = F.softmax(val_logits, dim=-1).numpy()
        oof_probs[val_idx] = val_probs

        val_preds = [[tax_classes[idx] for idx in row] for row in np.argsort(-val_probs, axis=1)[:, :3]]
        val_targets = [tax_classes[idx] for idx in y_train_int[val_idx]]
        fold_map3 = calculate_map3(val_targets, val_preds)
        print(f"Fold {fold_num} Pure Softmax MAP@3: {fold_map3:.5f}")

        # Test inference
        te_logits_list = []
        with torch.no_grad():
            for (b_tx,) in te_loader:
                b_tx = b_tx.to(device)
                t_logits, _ = model(b_tx)
                te_logits_list.append(t_logits.cpu())
        te_logits = torch.cat(te_logits_list, dim=0)
        te_probs = F.softmax(te_logits, dim=-1).numpy()
        test_fold_preds[fold_num - 1] = te_probs

    elapsed = time.time() - t0
    test_probs = np.mean(test_fold_preds, axis=0)

    # Global OOF MAP@3 evaluation
    oof_preds = [[tax_classes[idx] for idx in row] for row in np.argsort(-oof_probs, axis=1)[:, :3]]
    oof_targets = [tax_classes[idx] for idx in y_train_int]
    raw_oof_map3 = calculate_map3(oof_targets, oof_preds)
    print("\n" + "=" * 60)
    print(f"Training Complete in {elapsed:.1f}s")
    print(f"Raw ArcFace V2 OOF MAP@3: {raw_oof_map3:.5f}")
    print("=" * 60)

    # Compare with V1 (legacy primitive targets)
    old_oof = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
    old_preds = [[tax_classes[idx] for idx in row] for row in np.argsort(-old_oof, axis=1)[:, :3]]
    old_map3 = calculate_map3(oof_targets, old_preds)
    print(f"Legacy ArcFace V1 OOF MAP@3: {old_map3:.5f}")
    print(f"Net Gain from Champion Targets: {raw_oof_map3 - old_map3:+.5f}")

    # Save artifacts
    np.save(ART_DIR / "oof_arcface_pseudo_3500_v2.npy", oof_probs)
    np.save(ART_DIR / "test_arcface_pseudo_3500_v2.npy", test_probs)
    print("\nSaved artifacts:")
    print("  - artifacts/oof_arcface_pseudo_3500_v2.npy")
    print("  - artifacts/test_arcface_pseudo_3500_v2.npy")
    print("=" * 80)


if __name__ == "__main__":
    main()

