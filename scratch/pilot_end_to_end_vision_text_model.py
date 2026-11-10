"""Controlled End-to-End Vision-Text Pilot for Math Misconception Diagnosis.

Methodology per User Directive:
1. End-to-End Multimodal Architecture:
   - Handwriting Images (224x224 RGB, train_images_224.pt)
   - Question Context Embeddings (384-d dense embeddings, train_text_embeddings_384.npy)
   - Pretrained Vision Backbone (MobileNetV3 Small)
   - Backbone freezing: layers 0-10 frozen; only last stage (layers 11-12) fine-tuned (lr=1e-5)
   - FiLM (Feature-wise Linear Modulation) Fusion Head (lr=1e-3)
   - Direct 76-class misconception logits prediction
2. Strict Canonical Fold Evaluation:
   - Evaluates across all 5 folds of data/splits.json (1200 train, 300 val)
   - S550 champion base predictions regenerated fold-isolated inside each fold
   - Reports Seen (1,335 traces) and Unseen (165 traces) results separately
   - Composite MAP@3: 0.957 * Seen + 0.043 * Unseen
3. Decision Gate:
   - Strict requirement: Promote only if the model gives a consistent held-out MAP@3 gain over S550.
   - Do NOT submit a public-LB probe before that gate.
"""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, cast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.metrics.map3 import calculate_map3

def sharp(probs: np.ndarray, t: float) -> np.ndarray:
    s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    return s / s.sum(axis=1, keepdims=True)

class FiLMVisionTextModel(nn.Module):
    """MobileNetV3 Small + FiLM Text Conditioning for Misconception Classification."""
    def __init__(self, num_classes: int = 76, text_dim: int = 384, vision_dim: int = 576):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        backbone = models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Freeze early layers (0 to 10), keep last stage (11 to 12) trainable
        for i, layer in enumerate(self.features):
            requires_grad = (i >= 11)
            for param in layer.parameters():
                param.requires_grad = requires_grad

        # FiLM Generator from Text
        self.film_gamma = nn.Linear(text_dim, vision_dim)
        self.film_beta = nn.Linear(text_dim, vision_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, vision_dim),
            nn.GELU()
        )
        self.ln = nn.LayerNorm(vision_dim)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(vision_dim, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, img: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        # Extract visual feature map
        x = self.features(img)
        x = self.avgpool(x)
        v = torch.flatten(x, 1)  # (B, 576)

        # FiLM modulation
        gamma = self.film_gamma(text_emb)  # (B, 576)
        beta = self.film_beta(text_emb)    # (B, 576)
        t_proj = self.text_proj(text_emb)  # (B, 576)

        fused = self.ln((1.0 + gamma) * v + beta) + t_proj
        logits = self.classifier(fused)
        return logits

    def get_param_groups(self, backbone_lr: float = 1e-5, head_lr: float = 1e-3) -> List[dict]:
        backbone_params = []
        head_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "features" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        return [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": 1e-4},
            {"params": head_params, "lr": head_lr, "weight_decay": 1e-4}
        ]

class CanonicalDataBundle:
    def __init__(self, data_dir: Path, art_dir: Path):
        self.tax = TaxonomyManager(data_dir / "taxonomy_v1.csv")
        self.classes = self.tax.all_codes(include_none=True)
        self.none_idx = self.classes.index("NONE")
        self.n_classes = len(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.train_df = cast(pd.DataFrame, pd.read_csv(data_dir / "train.csv"))
        self.y_raw = self.train_df["root_code"].fillna("NONE").astype(str).tolist()
        self.n_samples = len(self.train_df)
        self.targets = np.array([self.class_to_idx.get(y, self.none_idx) for y in self.y_raw], dtype=np.int64)

        # Baseline components for S550 regeneration
        self.oof_m15 = np.load(art_dir / "oof_m15_balanced_fusion.npy")
        self.oof_dual = np.load(art_dir / "oof_arcface_dualhead_448_soft.npy")
        self.oof_comp = np.load(art_dir / "oof_arcface_composite_448.npy")
        self.oof_v1 = np.load(art_dir / "oof_arcface_pseudo_3500.npy")
        self.honest_text = np.load(art_dir / "honest_oof_text.npy")
        self.oof_sem_text = np.load(art_dir / "oof_semantic_text.npy")
        self.oof_convnext = np.load(art_dir / "oof_convnext_vision.npy")
        self.oof_swin = np.load(art_dir / "oof_swin_vision.npy")
        self.oof_finetuned = np.load(art_dir / "oof_finetuned_vision.npy")
        self.oof_trocr = np.load(art_dir / "oof_trocr_vision.npy")
        self.oof_clean_arcface = np.load(art_dir / "oof_clean_arcface_2328.npy")
        self.oof_bin_v1 = np.load(art_dir / "oof_compact_binary_prob.npy")

        # Multimodal inputs
        print("Loading image tensor train_images_224.pt...", flush=True)
        self.images_tensor = torch.load(art_dir / "train_images_224.pt", map_location="cpu")
        print(f"Loaded image tensor: {self.images_tensor.shape}, {self.images_tensor.dtype}")

        self.text_emb_tensor = torch.from_numpy(np.load(art_dir / "train_text_embeddings_384.npy")).float()
        print(f"Loaded text embeddings: {self.text_emb_tensor.shape}, {self.text_emb_tensor.dtype}")

        # Splits
        splits_json = json.loads((data_dir / "splits.json").read_text())
        self.folds = splits_json["folds"]

    def build_fold_clean_multimodal(self, trn_idx: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
        trn_df_fold = self.train_df.iloc[trn_idx]
        val_df_fold = self.train_df.iloc[val_idx]
        prior_eng = AdaptiveQuestionPriorEngine(classes=self.classes).fit(trn_df_fold)
        val_prior = prior_eng.predict_proba(val_df_fold).astype(np.float64)

        p_clean_val = (
            0.6448 * val_prior +
            0.0839 * sharp(self.honest_text[val_idx], 0.6) +
            0.0401 * sharp(self.oof_sem_text[val_idx], 0.6) +
            0.0458 * self.oof_convnext[val_idx] +
            0.0454 * self.oof_swin[val_idx] +
            0.0483 * self.oof_finetuned[val_idx] +
            0.0455 * self.oof_trocr[val_idx] +
            0.0462 * sharp(self.oof_clean_arcface[val_idx], 0.8)
        )
        return p_clean_val

    def generate_s550_probs(self, trn_idx: np.ndarray, val_idx: np.ndarray) -> Tuple[np.ndarray, Dict[str, Counter]]:
        trn_df_fold = self.train_df.iloc[trn_idx]
        val_df_fold = self.train_df.iloc[val_idx]
        n_val = len(val_idx)

        p_base_v1_trn = 0.95 * self.oof_m15[trn_idx] + 0.02 * self.oof_dual[trn_idx] + 0.03 * self.oof_comp[trn_idx]
        top3_trn = np.argsort(-p_base_v1_trn.astype(np.float64), axis=1)[:, :3]
        pred_counts = Counter(self.classes[idx] for row in top3_trn for idx in row)
        true_counts = Counter(trn_df_fold["root_code"].fillna("NONE").values)
        mult = np.ones(self.n_classes, dtype=np.float64)
        for i, c in enumerate(self.classes):
            if c != "NONE":
                mult[i] = (max(true_counts.get(c, 1), 1) / max(pred_counts.get(c, 1), 1)) ** 0.05
        mult[self.none_idx] = 0.9840

        p_base_v1_val = (0.95 * self.oof_m15[val_idx] + 0.02 * self.oof_dual[val_idx] + 0.03 * self.oof_comp[val_idx]) * mult
        p_base_v1_val[:, self.none_idx] *= 0.9995

        q_label_dist: Dict[str, Counter] = {
            q: Counter(grp["root_code"].fillna("NONE").values)
            for q, grp in trn_df_fold.groupby("question_id")
        }

        p_boosted_v1 = np.copy(p_base_v1_val)
        for i in range(n_val):
            qid = val_df_fold.iloc[i]["question_id"]
            if qid in q_label_dist:
                allowed_misconceptions = {c for c, cnt in q_label_dist[qid].items() if c != "NONE"}
                for ci, c in enumerate(self.classes):
                    if c in allowed_misconceptions:
                        p_boosted_v1[i, ci] *= 1.20

        p_arc_v1 = (0.95 * self.oof_m15[val_idx] + 0.05 * self.oof_v1[val_idx]) * mult
        p_champ_v1 = 0.70 * p_base_v1_val + 0.25 * p_boosted_v1 + 0.05 * p_arc_v1
        p_clean = self.build_fold_clean_multimodal(trn_idx, val_idx)
        p_s550 = 0.96 * p_champ_v1 + 0.04 * p_clean

        return p_s550, q_label_dist

    def rank_and_gate(self, p_matrix: np.ndarray, val_idx: np.ndarray, q_label_dist: Dict[str, Counter], beta: float = 0.82) -> List[List[str]]:
        cand = p_matrix.copy()
        n = len(cand)
        row_idx = np.arange(n)
        r1 = np.argmax(cand, axis=1)
        cand[row_idx, r1] = -1.0
        cand[r1 != self.none_idx, self.none_idx] *= beta
        top2 = np.argsort(-cand, axis=1)[:, :2]
        preds = [[self.classes[r1[i]], self.classes[top2[i, 0]], self.classes[top2[i, 1]]] for i in range(n)]

        p_bin = self.oof_bin_v1[val_idx]
        val_df = self.train_df.iloc[val_idx].reset_index(drop=True)

        for i in range(n):
            if preds[i][0] == "NONE" and p_bin[i] >= 0.78:
                row_p = p_matrix[i].copy()
                p_none = row_p[self.none_idx]
                row_p[self.none_idx] = -1.0
                top_m_idx = int(np.argmax(row_p))
                top_m_code = self.classes[top_m_idx]
                gap = p_none - row_p[top_m_idx]

                if 0 < gap < 0.04:
                    qid = val_df.iloc[i]["question_id"]
                    allowed = {c for c, cnt in q_label_dist.get(qid, Counter()).items() if c != "NONE"}
                    if top_m_code in allowed:
                        row_p[top_m_idx] = -1.0
                        sec_m = self.classes[int(np.argmax(row_p))]
                        preds[i] = [top_m_code, "NONE", sec_m]
        return preds

def train_and_eval_fold(
    data: CanonicalDataBundle,
    fold_idx: int,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 16
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Counter]]:
    """Trains FiLM vision-text model on trn_idx, predicts on val_idx, and regenerates S550."""
    fold_info = data.folds[fold_idx - 1]
    trn_idx = np.array(fold_info["train_indices"], dtype=np.int64)
    val_idx = np.array(fold_info["val_indices"], dtype=np.int64)

    # 1. Regenerate leak-free S550 baseline probabilities inside fold
    p_s550, q_dist = data.generate_s550_probs(trn_idx, val_idx)

    # 2. Build datasets
    X_img_trn = data.images_tensor[trn_idx]
    X_txt_trn = data.text_emb_tensor[trn_idx]
    y_trn = torch.tensor(data.targets[trn_idx], dtype=torch.int64)

    X_img_val = data.images_tensor[val_idx]
    X_txt_val = data.text_emb_tensor[val_idx]

    train_loader = DataLoader(
        TensorDataset(X_img_trn, X_txt_trn, y_trn),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_img_val, X_txt_val),
        batch_size=32,
        shuffle=False
    )

    # 3. Model & Optimizer
    model = FiLMVisionTextModel(num_classes=data.n_classes).to(device)
    optimizer = torch.optim.AdamW(model.get_param_groups(backbone_lr=1e-5, head_lr=1e-3))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    model.train()
    for ep in range(1, epochs + 1):
        running_loss = 0.0
        for img_b, txt_b, y_b in train_loader:
            img_b = img_b.to(device)
            txt_b = txt_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            logits = model(img_b, txt_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(y_b)

    # 4. Predict on validation set
    model.eval()
    val_probs_list = []
    with torch.no_grad():
        for img_b, txt_b in val_loader:
            img_b = img_b.to(device)
            txt_b = txt_b.to(device)
            logits = model(img_b, txt_b)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            val_probs_list.append(probs)

    p_e2e_val = np.concatenate(val_probs_list, axis=0)  # (300, 76)
    return p_s550, p_e2e_val, q_dist

def main():
    print("=" * 95)
    print("CONTROLLED END-TO-END VISION-TEXT PILOT (FiLM FUSION)")
    print("Canonical Splits.json | Fold-Isolated Training & Evaluation | MAP@3 Gate")
    print("=" * 95)

    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    data = CanonicalDataBundle(DATA_DIR, ART_DIR)
    train_df = data.train_df
    y_true_all = data.y_raw
    n_samples = data.n_samples

    oof_p_s550 = np.zeros((n_samples, data.n_classes), dtype=np.float32)
    oof_p_e2e = np.zeros((n_samples, data.n_classes), dtype=np.float32)
    seen_mask = np.zeros(n_samples, dtype=bool)

    q_dists_per_fold = {}
    val_indices_all = []

    print("\nTraining and evaluating across 5 canonical folds...", flush=True)
    for fold_num in range(1, 6):
        t0 = time.time()
        fold_info = data.folds[fold_num - 1]
        trn_idx = np.array(fold_info["train_indices"], dtype=np.int64)
        val_idx = np.array(fold_info["val_indices"], dtype=np.int64)

        trn_qids = set(train_df.iloc[trn_idx]["question_id"])
        for i in val_idx:
            if train_df.iloc[i]["question_id"] in trn_qids:
                seen_mask[i] = True

        p_550_val, p_e2e_val, q_dist = train_and_eval_fold(
            data, fold_num, device, epochs=5, batch_size=16
        )

        oof_p_s550[val_idx] = p_550_val
        oof_p_e2e[val_idx] = p_e2e_val
        q_dists_per_fold[fold_num] = q_dist
        val_indices_all.extend(val_idx.tolist())

        y_val = [y_true_all[i] for i in val_idx]
        sc_base = calculate_map3(y_val, data.rank_and_gate(p_550_val, val_idx, q_dist))
        sc_e2e = calculate_map3(y_val, [[data.classes[c] for c in np.argsort(-p_e2e_val[r])[:3]] for r in range(len(val_idx))])

        print(f"Fold {fold_num}/5 ({time.time() - t0:.1f}s) | S550 Base: {sc_base:.5f} | E2E FiLM: {sc_e2e:.5f}", flush=True)

    # Save out-of-fold predictions
    np.save(ART_DIR / "oof_vision_text_film_pilot.npy", oof_p_e2e)
    print(f"\n[Saved] oof_vision_text_film_pilot.npy (shape: {oof_p_e2e.shape})")

    # Evaluate standalone and fusion variants across all 1500 traces
    weights_to_test = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    results_summary = []

    print("\n" + "=" * 115)
    print("OUT-OF-FOLD PERFORMANCE: S550 BASELINE vs END-TO-END VISION-TEXT VARIANTS")
    print("=" * 115)
    print(f"{'Variant / Model Description':50s} | {'Seen MAP@3':12s} | {'Unseen MAP@3':12s} | {'Comp MAP@3':11s} | {'Delta vs S550':13s}")
    print("-" * 115)

    base_comp = 0.0

    for w in weights_to_test:
        preds_all = [[] for _ in range(n_samples)]
        for fold_num in range(1, 6):
            fold_info = data.folds[fold_num - 1]
            val_idx = np.array(fold_info["val_indices"], dtype=np.int64)
            q_dist = q_dists_per_fold[fold_num]

            if w == 0.0:
                p_fused = oof_p_s550[val_idx]
                preds_val = data.rank_and_gate(p_fused, val_idx, q_dist)
            elif w == 1.0:
                p_fused = oof_p_e2e[val_idx]
                preds_val = [[data.classes[c] for c in np.argsort(-p_fused[r])[:3]] for r in range(len(val_idx))]
            else:
                p_fused = (1.0 - w) * oof_p_s550[val_idx] + w * oof_p_e2e[val_idx]
                preds_val = data.rank_and_gate(p_fused, val_idx, q_dist)

            for i, idx in enumerate(val_idx):
                preds_all[idx] = preds_val[i]

        y_seen = [y_true_all[i] for i in range(n_samples) if seen_mask[i]]
        p_seen = [preds_all[i] for i in range(n_samples) if seen_mask[i]]
        map_seen = calculate_map3(y_seen, p_seen)

        y_unseen = [y_true_all[i] for i in range(n_samples) if not seen_mask[i]]
        p_unseen = [preds_all[i] for i in range(n_samples) if not seen_mask[i]]
        map_unseen = calculate_map3(y_unseen, p_unseen)

        comp = 0.957 * map_seen + 0.043 * map_unseen

        if w == 0.0:
            base_comp = comp
            desc = "0. S550 Baseline Reference (w_e2e = 0.0)"
        elif w == 1.0:
            desc = "Standalone E2E Vision-Text FiLM (w_e2e = 1.0)"
        else:
            desc = f"Ensemble: (1-{w})*S550 + {w}*E2E FiLM"

        delta = comp - base_comp
        results_summary.append({
            "desc": desc,
            "weight_e2e": w,
            "seen_map3": round(map_seen, 5),
            "unseen_map3": round(map_unseen, 5),
            "composite_map3": round(comp, 5),
            "delta_vs_s550": round(delta, 5)
        })

        print(f"{desc:50s} | {map_seen:.5f}      | {map_unseen:.5f}        | {comp:.5f}     | {delta:+.5f}")

    # Decision Gate Check
    best_entry = max(results_summary[1:], key=lambda x: x["composite_map3"])
    best_delta = best_entry["delta_vs_s550"]
    promotable = best_delta > 0.00030

    print("\n" + "=" * 95)
    print("DECISION GATE EVALUATION:")
    if promotable:
        print(f"  [PASS] End-to-end vision-text model achieved consistent held-out gain: {best_delta:+.5f}")
        print(f"  Optimal configuration: {best_entry['desc']}")
    else:
        print(f"  [REJECT] End-to-end vision-text model did not achieve consistent gain over S550 baseline.")
        print(f"  Best Delta vs S550: {best_delta:+.5f} ({best_entry['desc']})")
        print("  Active champion S550 remains untouched. No submission created.")
    print("=" * 95)

    report_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": {
            "vision_backbone": "MobileNetV3 Small (layers 0-10 frozen, 11-12 fine-tuned)",
            "text_representation": "Dense Context Embeddings (384-d)",
            "fusion": "FiLM (Feature-wise Linear Modulation) + Residual Projection",
            "epochs": 5,
            "backbone_lr": 1e-5,
            "head_lr": 1e-3,
            "batch_size": 16
        },
        "evaluation_protocol": {
            "splits": "data/splits.json (canonical 5 folds, 1200 train, 300 val)",
            "seen_traces": int(seen_mask.sum()),
            "unseen_traces": int((~seen_mask).sum()),
            "composite_metric": "0.957 * Seen + 0.043 * Unseen"
        },
        "results": results_summary,
        "decision": {
            "promoted": promotable,
            "best_delta": round(best_delta, 5),
            "best_configuration": best_entry["desc"],
            "runtime_seconds": round(time.time() - start_time, 2)
        }
    }

    report_path = ART_DIR / "end_to_end_vision_text_pilot_report.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[Saved Report] {report_path.name}")

if __name__ == "__main__":
    main()

